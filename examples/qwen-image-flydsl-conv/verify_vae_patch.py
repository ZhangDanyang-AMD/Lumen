#!/usr/bin/env python3
"""Check the vae_conv patch against the unpatched VAE, before spending 8 GPUs on it.

Three questions, in the order that matters:

1. Does it change the answer? The T=1 causal rewrite is an algebraic identity, so
   in exact arithmetic the patched encoder returns the reference. In BF16 it does
   not: a different convolution algorithm sums the reduction in a different
   order, and 40-odd layers compound that. So "bitwise identical" is the wrong
   bar and would fail for a correct patch.

   The bar used instead is an FP32 encode of the same input. A patch that only
   reorders arithmetic lands about as close to FP32 as the stock BF16 model does.
   A patch that mishandles causal padding lands much further away -- that is the
   failure mode worth catching, and it is not subtle when measured this way.
   Per-layer isolation below separates the rewrite from the kernel.
2. Is FlyDSL actually running? Lumen demotes to torch on its own when a problem
   is out of range, and a silent demotion would look like a successful
   integration that changed nothing. ``dispatch._backend_cache`` says which
   backend won.
3. Is the whole encode faster, not just the convolutions in isolation?

    PYTHONPATH=$LUMEN_PYTHONPATH python3 $EXAMPLE_DIR/verify_vae_patch.py [--dtype bf16|fp32]
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lumen_vae_conv import patch_vae_convs, snr_db  # noqa: E402

MODELS = os.environ.get("QWEN_IMAGE_DIR", "/work/models/Qwen-Image")


def head(t):
    print(f"\n=== {t} ===")


def timed_encode(vae, x, iters=20, warmup=5):
    for _ in range(warmup):
        with torch.no_grad():
            vae.encode(x)
    torch.cuda.synchronize()
    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    s.record()
    for _ in range(iters):
        with torch.no_grad():
            vae.encode(x)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # milliseconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument(
        "--dtype",
        default="bf16",
        choices=("bf16", "fp32"),
        help="fp32 reproduces what the trainer actually loads "
        "(modeling_qwen_image_condition.py hardcodes torch_dtype=torch.float32), "
        "which the FlyDSL kernel cannot take",
    )
    args = ap.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    from diffusers import AutoencoderKLQwenImage

    head("load")
    vae = AutoencoderKLQwenImage.from_pretrained(MODELS, subfolder="vae", torch_dtype=dtype)
    vae = vae.to("cuda").eval()
    vae.requires_grad_(False)
    print(f"{MODELS}  dtype={vae.dtype}")

    torch.manual_seed(1234)
    S = args.size
    x = torch.randn(1, 3, 1, S, S, device="cuda", dtype=dtype).clamp(-1, 1)
    print(f"input {tuple(x.shape)}")

    head("unpatched (reference)")
    with torch.no_grad():
        ref = vae.encode(x).latent_dist.parameters.clone()
    t_ref = timed_encode(vae, x)
    print(f"latents {tuple(ref.shape)} {ref.dtype}")
    print(f"encode  {t_ref:.3f} ms")

    head("apply patch")
    stats = patch_vae_convs(vae, verbose=print)
    for name, k, s in stats["patched"]:
        print(f"  patched {name:44} k={k} stride={s}")
    for name, why in stats["skipped"]:
        print(f"  skipped {name:44} {why}")

    head("patched")
    with torch.no_grad():
        test = vae.encode(x).latent_dist.parameters.clone()
    t_new = timed_encode(vae, x)
    print(f"latents {tuple(test.shape)} {test.dtype}")
    print(f"encode  {t_new:.3f} ms   ({t_ref / t_new:.2f}x, {t_ref - t_new:+.3f} ms)")

    head(f"numerics: both {args.dtype.upper()} encoders against an FP32 encode of the same input")
    vae32 = AutoencoderKLQwenImage.from_pretrained(MODELS, subfolder="vae", torch_dtype=torch.float32)
    vae32 = vae32.to("cuda").eval()
    vae32.requires_grad_(False)
    with torch.no_grad():
        ref32 = vae32.encode(x.float()).latent_dist.parameters.clone()
    del vae32
    torch.cuda.empty_cache()

    snr_stock = snr_db(ref32, ref)
    snr_patched = snr_db(ref32, test)
    print(f"  stock BF16   vs FP32 : {snr_stock:6.1f} dB")
    print(f"  patched BF16 vs FP32 : {snr_patched:6.1f} dB   ({snr_patched - snr_stock:+.1f} dB)")
    print(f"  patched      vs stock: {snr_db(ref, test):6.1f} dB")
    diff = (ref.float() - test.float()).abs()
    print(f"  max abs diff (patched vs stock): {diff.max().item():.3e}")
    print(f"  reference magnitude            : {ref.float().abs().max().item():.3e}")

    head("per-layer isolation: is it the rewrite or the kernel?")
    import lumen.ops.conv as conv_ops
    import torch.nn.functional as Fn

    probe = dict(vae.named_modules())["encoder.down_blocks.0.conv1"]
    w, b = probe.weight, probe.bias
    xl = torch.randn(1, w.shape[1], 1, S, S, device="cuda", dtype=dtype)
    pad = list(probe._padding)
    with torch.no_grad():
        exact32 = Fn.conv3d(Fn.pad(xl.float(), pad), w.float(), b.float(), padding=0)
        as_model = Fn.conv3d(Fn.pad(xl, pad), w, b, padding=0)
        rewritten_torch = Fn.conv2d(xl[:, :, 0], w[:, :, -1].contiguous(), b, padding=(pad[2], pad[0])).unsqueeze(2)
        rewritten_lumen = conv_ops.conv2d(xl[:, :, 0], w[:, :, -1].contiguous(), b, padding=(pad[2], pad[0])).unsqueeze(
            2
        )
    print("  encoder.down_blocks.0.conv1, one layer, against its own FP32 result")
    print(f"    conv3d as the model calls it : {snr_db(exact32, as_model):6.1f} dB")
    print(f"    rewritten to conv2d (torch)  : {snr_db(exact32, rewritten_torch):6.1f} dB")
    print(f"    rewritten to conv2d (lumen)  : {snr_db(exact32, rewritten_lumen):6.1f} dB")
    print(f"    torch conv3d vs torch conv2d : {snr_db(as_model, rewritten_torch):6.1f} dB  (the rewrite alone)")
    print(f"    torch conv2d vs lumen conv2d : {snr_db(rewritten_torch, rewritten_lumen):6.1f} dB  (the kernel alone)")

    head("which backend actually ran")
    from lumen.ops.dispatch import FLYDSL_FALLBACK_ORDER, _backend_cache

    order = [b.name for b in FLYDSL_FALLBACK_ORDER]
    print(f"  chain           : {order}")
    print(f"  _backend_cache  : { {k: v for k, v in _backend_cache.items()} }")
    idx = _backend_cache.get("conv2d")
    if idx is None:
        chosen = "not locked yet (fewer than 3 warmup successes)"
    else:
        chosen = order[idx] if idx < len(order) else f"index {idx}"
    print(f"  conv2d locked to: {chosen}")

    head("verdict")
    if dtype is torch.float32:
        print("  FP32 mode: this is what the trainer runs today.")
        print(f"  conv2d backend: {chosen}")
        print("  The FlyDSL kernel is BF16-only, so Lumen demotes to torch before")
        print("  the dispatcher is even reached -- which is why the cache is empty")
        print("  rather than showing TORCH. The patch still rewrites conv3d to")
        print("  conv2d, so it is not a no-op, but no FlyDSL code runs.")
        sys.exit(0)

    problems = []
    # The patched encoder must stay in the same accuracy class as the stock BF16
    # one. A 3 dB allowance is roughly "no worse than a factor of two in noise
    # power"; mishandled causal padding costs far more than that.
    if snr_patched < snr_stock - 3.0:
        problems.append(
            f"patched BF16 is {snr_stock - snr_patched:.1f} dB further from FP32 than stock BF16 "
            f"({snr_patched:.1f} vs {snr_stock:.1f}) -- more than reordered arithmetic explains"
        )
    if idx != 0:
        problems.append(f"conv2d did not lock to FLYDSL (got {chosen}) -- the patch may be running on torch")
    if not stats["patched"]:
        problems.append("no module was patched")
    if problems:
        for p in problems:
            print(f"  FAIL {p}")
        sys.exit(1)
    print(f"  {len(stats['patched'])} convolutions on FlyDSL")
    print(f"  accuracy: {snr_patched:.1f} dB vs FP32, against {snr_stock:.1f} dB for the stock BF16 model")
    print(f"  speed   : whole encode {t_ref:.3f} -> {t_new:.3f} ms ({t_ref / t_new:.2f}x)")
    print("\nPASS")


if __name__ == "__main__":
    main()
