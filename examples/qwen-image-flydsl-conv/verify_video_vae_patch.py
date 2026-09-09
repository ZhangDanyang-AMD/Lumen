#!/usr/bin/env python3
"""Check the video (T>1) path of the vae_conv patch on Wan's VAE.

Same three questions as ``verify_vae_patch.py``, but the first one has a
different shape here. On images the patch rewrites a 3-D convolution as a 2-D
one, so there is an algebraic identity to get wrong. On video there is no
identity: the module's causal padding is left exactly as it is and only the
kernel underneath changes. That makes the failure mode narrower -- either the
kernel gets the same arguments torch got, or it does not -- but it also means
"bitwise identical" is still the wrong bar, because a different convolution
algorithm sums its reduction in a different order and 40-odd layers compound
that.

The bar is an FP32 encode of the same clip. A patch that only reorders
arithmetic lands about as close to FP32 as the stock BF16 model does. One that
loses the asymmetric time padding lands far away, and it is not subtle: dropping
the leading zero frames shifts every output frame in time.

Both patch modes are measured, because their coverage differs by an order of
magnitude on a clip (2-10% of convolution time versus 88-96%, see
``trace_video_vae_convs.py``).

    PYTHONPATH=$LUMEN_PYTHONPATH python3 $EXAMPLE_DIR/verify_video_vae_patch.py \
        --model /work/models-extra/Wan2.1-T2V-1.3B --frames 17

Wan's causal encoder consumes a clip as 1 + 4k frames, so --frames wants that
form; 17 is one first frame plus four chunks, which is enough for every distinct
convolution shape to appear. 81 is what configs/dit/wan2.1_I2V_1.3B_lora.yaml
actually trains on.
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lumen_vae_conv import patch_vae_convs, snr_db  # noqa: E402

DEFAULT_MODEL = os.environ.get("WAN_VAE", "/work/models-extra/Wan2.1-T2V-1.3B")


def head(t):
    print(f"\n=== {t} ===")


def timed_encode(vae, x, iters, warmup=2):
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
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--klass", default="AutoencoderKLWan")
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument(
        "--dtype",
        default="bf16",
        choices=("bf16", "fp32"),
        help="fp32 reproduces what the trainer loads "
        "(modeling_wan_condition.py asks for torch_dtype=torch.float32), which the "
        "FlyDSL kernel cannot take",
    )
    args = ap.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    import diffusers

    klass = getattr(diffusers, args.klass)

    def load(dt=dtype):
        vae = klass.from_pretrained(args.model, subfolder="vae", torch_dtype=dt)
        vae = vae.to("cuda").eval()
        vae.requires_grad_(False)
        return vae

    head("load")
    t0 = time.time()
    vae = load()
    print(f"{args.klass} from {args.model} in {time.time() - t0:.1f}s, dtype={vae.dtype}")

    torch.manual_seed(1234)
    x = torch.randn(1, 3, args.frames, args.height, args.width, device="cuda", dtype=dtype).clamp(-1, 1)
    print(f"input {tuple(x.shape)}")

    head("unpatched (reference)")
    with torch.no_grad():
        ref = vae.encode(x).latent_dist.parameters.clone()
    t_ref = timed_encode(vae, x, args.iters)
    print(f"latents {tuple(ref.shape)} {ref.dtype}")
    print(f"encode  {t_ref:.1f} ms")
    del vae
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ modes
    results = {}
    for mode, video in (("vae_conv", False), ("vae_conv_video", True)):
        head(f"{mode} (video={video})")
        vae = load()
        stats = patch_vae_convs(vae, video=video, verbose=print)
        with torch.no_grad():
            test = vae.encode(x).latent_dist.parameters.clone()
        t_new = timed_encode(vae, x, args.iters)
        print(f"encode  {t_new:.1f} ms   ({t_ref / t_new:.2f}x, {t_new - t_ref:+.1f} ms)")
        results[mode] = {"latents": test, "ms": t_new, "stats": stats}
        if video:
            for name, k, how in stats["patched"]:
                print(f"  patched {name:44} k={k} -> {how}")
            for name, why in stats["skipped"]:
                print(f"  skipped {name:44} {why}")
        del vae
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- numerics
    head(f"numerics: every {args.dtype.upper()} encoder against an FP32 encode of the same clip")
    vae32 = load(torch.float32)
    with torch.no_grad():
        ref32 = vae32.encode(x.float()).latent_dist.parameters.clone()
    del vae32
    torch.cuda.empty_cache()

    snr_stock = snr_db(ref32, ref)
    print(f"  stock {args.dtype:4}                vs FP32 : {snr_stock:6.1f} dB")
    for mode in results:
        s = snr_db(ref32, results[mode]["latents"])
        results[mode]["snr"] = s
        print(f"  {mode:26} vs FP32 : {s:6.1f} dB   ({s - snr_stock:+.1f} dB vs stock)")
    for mode in results:
        d = (ref.float() - results[mode]["latents"].float()).abs()
        print(f"  {mode:26} vs stock: {snr_db(ref, results[mode]['latents']):6.1f} dB, max abs diff {d.max():.3e}")
    print(f"  reference magnitude                      : {ref.float().abs().max().item():.3e}")

    # ------------------------------------------------------ per-layer isolation
    head("per-layer isolation: the kernel, on a real T>1 shape")
    import lumen.ops.conv as conv_ops

    vae = load()
    probe = dict(vae.named_modules())["encoder.down_blocks.0.conv1"]
    w, b, pad = probe.weight, probe.bias, list(probe._padding)
    # Four real frames, the extent an interior chunk of the clip carries.
    xl = torch.randn(1, w.shape[1], 4, args.height, args.width, device="cuda", dtype=dtype)
    with torch.no_grad():
        padded32 = F.pad(xl.float(), pad)
        exact32 = F.conv3d(padded32, w.float(), b.float(), padding=0)
        padded = F.pad(xl, pad)
        as_model = F.conv3d(padded, w, b, padding=0)
        via_lumen = conv_ops.conv3d(padded, w, b, padding=0)
        # What losing the asymmetric time padding would look like, for scale.
        symmetric = F.conv3d(xl, w, b, padding=1)
    print(f"  encoder.down_blocks.0.conv1, input {tuple(xl.shape)}, causal pad {tuple(pad)}")
    print(f"    torch conv3d, as the model calls it  : {snr_db(exact32, as_model):6.1f} dB vs its own FP32")
    print(f"    lumen conv3d, same arguments         : {snr_db(exact32, via_lumen):6.1f} dB vs its own FP32")
    print(f"    torch conv3d vs lumen conv3d         : {snr_db(as_model, via_lumen):6.1f} dB  (the kernel alone)")
    print(f"    symmetric padding=1 instead of causal: {snr_db(as_model, symmetric):6.1f} dB  (the bug to catch)")
    del vae, padded32, exact32
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ backend
    head("which backend actually ran")
    from lumen.ops.dispatch import FLYDSL_FALLBACK_ORDER, _backend_cache

    order = [b.name for b in FLYDSL_FALLBACK_ORDER]
    print(f"  chain          : {order}")
    print(f"  _backend_cache : { {k: v for k, v in _backend_cache.items()} }")
    locked = {}
    for op in ("conv2d", "conv3d"):
        idx = _backend_cache.get(op)
        locked[op] = None if idx is None else (order[idx] if idx < len(order) else f"index {idx}")
        print(f"  {op} locked to: {locked[op] or 'not locked (fewer than 3 warmup successes)'}")

    # ------------------------------------------------------------------ verdict
    head("verdict")
    if dtype is torch.float32:
        print("  FP32 mode: this is what the trainer runs today.")
        print("  The FlyDSL kernel is BF16-only, so Lumen demotes before the")
        print("  dispatcher is reached and the cache stays empty rather than")
        print("  showing TORCH -- no FlyDSL code runs at all.")
        print("  What is left is the conv3d->conv2d rewrite, which is arithmetic")
        print("  rather than a kernel and so survives in FP32. On a clip it")
        print("  reaches only the uncached T=1 calls, which is why both modes")
        print("  measure the same here: the T>1 half of the patch does nothing")
        print("  until the VAE runs in BF16.")
        for mode in results:
            r = results[mode]
            print(f"  {mode:14} {r['ms']:8.1f} ms vs {t_ref:.1f} ms unpatched  ({t_ref / r['ms']:.2f}x)")
        sys.exit(0)

    problems = []
    for mode in results:
        # Same accuracy class as the stock BF16 model. 3 dB is roughly "no worse
        # than twice the noise power"; mishandled causal padding costs far more.
        if results[mode]["snr"] < snr_stock - 3.0:
            problems.append(
                f"{mode} is {snr_stock - results[mode]['snr']:.1f} dB further from FP32 than stock BF16 "
                f"({results[mode]['snr']:.1f} vs {snr_stock:.1f}) -- more than reordered arithmetic explains"
            )
    if locked["conv3d"] != "FLYDSL":
        problems.append(f"conv3d did not lock to FLYDSL (got {locked['conv3d']}) -- the T>1 path ran on torch")
    if not results["vae_conv_video"]["stats"]["patched"]:
        problems.append("vae_conv_video patched nothing")
    if problems:
        for p in problems:
            print(f"  FAIL {p}")
        sys.exit(1)

    n = len(results["vae_conv_video"]["stats"]["patched"])
    print(f"  {n} convolutions rebound, conv3d on FLYDSL")
    print(f"  accuracy : {results['vae_conv_video']['snr']:.1f} dB vs FP32, stock BF16 is {snr_stock:.1f} dB")
    print(f"  whole encode, {args.frames} frames at {args.height}x{args.width}, mean of {args.iters}:")
    print(f"    unpatched      {t_ref:8.1f} ms")
    for mode in results:
        print(f"    {mode:14} {results[mode]['ms']:8.1f} ms   ({t_ref / results[mode]['ms']:.2f}x)")
    print("\n  One encode on one GPU. What fraction of a training step this is")
    print("  is a separate measurement -- do not quote it as a training speedup.")
    print("\nPASS")


if __name__ == "__main__":
    main()
