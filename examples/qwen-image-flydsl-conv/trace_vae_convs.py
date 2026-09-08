#!/usr/bin/env python3
"""Trace the real Qwen-Image VAE encoder convolutions at the training resolution.

``bench_vae_encode.py`` had to estimate this inventory by scaling an upstream
1024x1024 trace, because at the time no single container had both diffusers and a
working flydsl. A sidecar flydsl 0.3.2 on PYTHONPATH removes that split, so the
numbers here come from forward hooks on the actual AutoencoderKLQwenImage the
trainer loads, at the actual shape the trainer feeds it.

The trainer's condition model builds its input as
``to_tensor(image).unsqueeze(0).unsqueeze(2)`` -- ``(1, 3, 1, H, W)``, one image
at a time, under ``torch.no_grad()``. So every convolution below is T=1,
batch-1, forward-only.

    PYTHONPATH=$LUMEN_PYTHONPATH python3 $EXAMPLE_DIR/trace_vae_convs.py [--size 256]

Prints, per distinct convolution configuration: how many times one encode calls
it, and torch bf16 vs Lumen NCHW vs Lumen NHWC timings, then the frequency
weighted total for a whole encode.
"""

import argparse
import collections
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS = os.environ.get("QWEN_IMAGE_DIR", "/work/models/Qwen-Image")


def head(t):
    print(f"\n=== {t} ===")


def timed(fn, iters=40, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters  # microseconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1024, help="square image extent the trainer resizes to")
    # Steady-state at the default resolution, not the 10-step average: the
    # average is dominated by step 1's kernel autotune and would understate the
    # convolutions' share of a step. Pass --step-ms when changing --size; 256
    # measured ~3950 ms on the same node.
    ap.add_argument(
        "--step-ms", type=float, default=3400.0, help="measured steady-state step time, for the share column"
    )
    args = ap.parse_args()

    from diffusers import AutoencoderKLQwenImage

    head("load")
    t0 = time.time()
    vae = AutoencoderKLQwenImage.from_pretrained(MODELS, subfolder="vae", torch_dtype=torch.bfloat16)
    vae = vae.to("cuda").eval()
    vae.requires_grad_(False)
    print(f"loaded in {time.time() - t0:.1f}s  dtype={vae.dtype}  device={vae.device}")

    convs = [(n, m) for n, m in vae.named_modules() if isinstance(m, (nn.Conv2d, nn.Conv3d))]
    enc = [(n, m) for n, m in convs if n.startswith("encoder")]
    print(f"{len(convs)} conv modules in the VAE, {len(enc)} in the encoder")

    calls = collections.OrderedDict()

    def make_hook(name, m):
        def hook(mod, inputs):
            x = inputs[0]
            cache_x = inputs[1] if len(inputs) > 1 else None
            key = (
                type(mod).__name__,
                tuple(x.shape),
                mod.weight.shape[0],
                tuple(mod.kernel_size),
                tuple(mod.stride),
                tuple(getattr(mod, "_padding", ())) or tuple(mod.padding),
                cache_x is not None,
            )
            entry = calls.setdefault(key, {"n": 0, "names": []})
            entry["n"] += 1
            if len(entry["names"]) < 2:
                entry["names"].append(name)

        return hook

    handles = [m.register_forward_pre_hook(make_hook(n, m)) for n, m in convs]

    S = args.size
    x = torch.randn(1, 3, 1, S, S, device="cuda", dtype=torch.bfloat16).clamp(-1, 1)

    head(f"one encode at {S}x{S}, input {tuple(x.shape)}")
    with torch.no_grad():
        out = vae.encode(x).latent_dist.parameters
    print(f"latent parameters {tuple(out.shape)}  {out.dtype}")
    for h in handles:
        h.remove()

    total_calls = sum(v["n"] for v in calls.values())
    print(f"{len(calls)} distinct configurations, {total_calls} convolution calls per encode")
    print(f"any call carrying cache_x: {any(k[6] for k in calls)}")

    head("per-configuration timing (torch bf16 vs Lumen)")
    import lumen.ops.conv as conv_ops

    hdr = f"{'layer':32}{'in shape':>21}{'k':>8}{'s':>6}{'n':>4}{'torch':>10}{'NCHW':>10}{'NHWC':>10}{'best':>7}"
    print(hdr)
    print("-" * len(hdr))

    tot_t = tot_n = tot_h = 0.0
    for key, meta in calls.items():
        cls, shape, cout, ksize, stride, pad, has_cache = key
        n = meta["n"]
        name = meta["names"][0]
        cin = shape[1]
        kshow = "x".join(str(v) for v in ksize)
        sshow = "x".join(str(v) for v in stride)

        x_in = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(cout, cin, *ksize, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(cout, device="cuda", dtype=torch.bfloat16)

        if cls == "QwenImageCausalConv3d":
            # The module pre-pads with F.pad(_padding) and then convolves with
            # padding 0, so time the padded extent rather than the incoming one.
            x_pad = F.pad(x_in, list(pad))
            if x_pad.shape[2] != ksize[0]:
                print(f"{name[:32]:32}{str(tuple(shape)):>21}   T>1 after padding, skipped")
                continue
            # Causal padding puts kT-1 zero frames in front, so at T=1 only
            # weight[:, :, -1] ever multiplies real pixels.
            x4 = x_pad[:, :, -1]
            w2 = w[:, :, -1].contiguous()
            st = tuple(stride[1:])
            x4h = x4.permute(0, 2, 3, 1).contiguous()
            t_t = timed(lambda: F.conv3d(x_pad, w, b, stride=stride, padding=0))
            t_n = timed(lambda: conv_ops.conv2d(x4, w2, b, stride=st, padding=0))
            t_h = timed(
                lambda: conv_ops.conv2d(x4h, w2, b, stride=st, padding=0, input_layout="NHWC", output_layout="NHWC")
            )
        else:
            p = tuple(pad)
            xh = x_in.permute(0, 2, 3, 1).contiguous()
            t_t = timed(lambda: F.conv2d(x_in, w, b, stride=stride, padding=p))
            t_n = timed(lambda: conv_ops.conv2d(x_in, w, b, stride=stride, padding=p))
            t_h = timed(
                lambda: conv_ops.conv2d(xh, w, b, stride=stride, padding=p, input_layout="NHWC", output_layout="NHWC")
            )

        tot_t += t_t * n
        tot_n += t_n * n
        tot_h += t_h * n
        print(
            f"{name[:32]:32}{str(tuple(shape)):>21}{kshow:>8}{sshow:>6}{n:>4}"
            f"{t_t:>8.1f}us{t_n:>8.1f}us{t_h:>8.1f}us{t_t / min(t_n, t_h):>6.2f}x"
        )

    head(f"whole encode at {S}x{S}, convolutions only")
    print(f"  torch bf16   {tot_t / 1000:>9.3f} ms")
    print(f"  lumen NCHW   {tot_n / 1000:>9.3f} ms   {tot_t / tot_n:.2f}x")
    print(f"  lumen NHWC   {tot_h / 1000:>9.3f} ms   {tot_t / tot_h:.2f}x")
    best = min(tot_n, tot_h)
    saved = (tot_t - best) / 1000
    print(f"  best saving  {saved:>9.3f} ms per encode")

    head("against a training step")
    print(f"  baseline step        {args.step_ms:>10.1f} ms")
    print(f"  conv share of step   {tot_t / 1000 / args.step_ms * 100:>10.4f} %")
    print(f"  best saving          {saved:>10.4f} ms = {saved / args.step_ms * 100:.4f} % of a step")


if __name__ == "__main__":
    main()
