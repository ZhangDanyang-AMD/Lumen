#!/usr/bin/env python3
"""Trace a video VAE's convolutions, to size the T>1 case the image patch skips.

``lumen_vae_conv.py`` only rewrites convolutions whose time extent is 1, because
that is where the causal 3-D convolution collapses to an exact 2-D one. VeOmni's
video DiTs (Wan, LTX-2, MiniMax-H3) run their VAEs over real clips, so none of
their convolutions take that path.

That is a statement about the patch, not about the kernel: ``lumen.ops.conv3d``
accepts 5-D filters. Whether extending the patch to video is worth doing depends
on how the FlyDSL kernel compares to torch on those shapes, which is what this
measures. For each distinct convolution one encode performs, it reports the time
extent, torch's time, and Lumen's -- and separates the totals into the part the
current patch can already claim and the part it cannot.

    PYTHONPATH=$LUMEN_PYTHONPATH python3 trace_video_vae_convs.py \
        --model /work/models-extra/Wan2.1-T2V-1.3B --frames 17 --height 480 --width 832
"""

import argparse
import collections
import time

import torch
import torch.nn as nn


def head(t):
    print(f"\n=== {t} ===")


def timed(fn, iters=10, warmup=3):
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
    ap.add_argument("--model", default="/work/models-extra/Wan2.1-T2V-1.3B")
    ap.add_argument("--klass", default="AutoencoderKLWan", help="diffusers autoencoder class")
    ap.add_argument("--frames", type=int, default=17, help="clip length; Wan's causal VAE wants 4k+1")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    args = ap.parse_args()

    import diffusers

    klass = getattr(diffusers, args.klass)

    head("load")
    t0 = time.time()
    vae = klass.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.bfloat16)
    vae = vae.to("cuda").eval()
    vae.requires_grad_(False)
    print(f"{args.klass} from {args.model} in {time.time() - t0:.1f}s")

    convs = [(n, m) for n, m in vae.named_modules() if isinstance(m, (nn.Conv2d, nn.Conv3d))]
    print(f"{len(convs)} conv modules, {sum(1 for _, m in convs if isinstance(m, nn.Conv3d))} of them 3-D")

    calls = collections.OrderedDict()

    def make_hook(name):
        def hook(mod, inputs):
            x = inputs[0]
            key = (
                type(mod).__name__,
                tuple(x.shape),
                mod.weight.shape[0],
                tuple(mod.kernel_size),
                tuple(mod.stride),
                tuple(getattr(mod, "_padding", ())) or tuple(mod.padding),
            )
            entry = calls.setdefault(key, {"n": 0, "name": name})
            entry["n"] += 1

        return hook

    handles = [m.register_forward_pre_hook(make_hook(n)) for n, m in convs]

    x = torch.randn(1, 3, args.frames, args.height, args.width, device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    head(f"one encode, input {tuple(x.shape)}")
    with torch.no_grad():
        out = vae.encode(x).latent_dist
    params = out.parameters if hasattr(out, "parameters") else out
    print(f"latent {tuple(params.shape)}  {params.dtype}")
    for h in handles:
        h.remove()
    print(f"{len(calls)} distinct configurations, {sum(v['n'] for v in calls.values())} calls per encode")

    head("per configuration: torch vs Lumen")
    import lumen.ops.conv as conv_ops

    hdr = f"{'layer':34}{'in shape':>24}{'T':>4}{'k':>8}{'n':>4}{'torch':>11}{'lumen':>11}{'ratio':>8}"
    print(hdr)
    print("-" * len(hdr))

    tot_t1_torch = tot_t1_lumen = 0.0  # time extent 1 -- what the current patch covers
    tot_tn_torch = tot_tn_lumen = 0.0  # time extent > 1 -- what it does not
    failures = []

    for key, meta in calls.items():
        cls, shape, cout, ksize, stride, pad = key
        n, name = meta["n"], meta["name"]
        is3d = len(ksize) == 3
        cin = shape[1]
        t_extent = shape[2] if (is3d and len(shape) == 5) else 1

        x_in = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(cout, cin, *ksize, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(cout, device="cuda", dtype=torch.bfloat16)

        # Reproduce what the module hands the convolution: a causal module has
        # already moved its padding into _padding and convolves with 0.
        import torch.nn.functional as Fn

        if len(pad) == 6:
            x_in = Fn.pad(x_in, list(pad))
            p = 0
        else:
            p = tuple(pad)

        conv = Fn.conv3d if is3d else Fn.conv2d
        op = conv_ops.conv3d if is3d else conv_ops.conv2d
        try:
            t_torch = timed(lambda: conv(x_in, w, b, stride=stride, padding=p))
            t_lumen = timed(lambda: op(x_in, w, b, stride=stride, padding=p))
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {str(exc)[:70]}")
            continue

        if t_extent > 1:
            tot_tn_torch += t_torch * n
            tot_tn_lumen += t_lumen * n
        else:
            tot_t1_torch += t_torch * n
            tot_t1_lumen += t_lumen * n

        kshow = "x".join(str(v) for v in ksize)
        print(
            f"{name[:34]:34}{str(tuple(shape)):>24}{t_extent:>4}{kshow:>8}{n:>4}"
            f"{t_torch:>9.1f}us{t_lumen:>9.1f}us{t_torch / t_lumen:>7.2f}x"
        )

    head("totals for one encode")
    tot_torch = tot_t1_torch + tot_tn_torch
    tot_lumen = tot_t1_lumen + tot_tn_lumen
    print(f"  all convolutions      torch {tot_torch / 1000:8.2f} ms   lumen {tot_lumen / 1000:8.2f} ms")
    print(
        f"  time extent == 1      torch {tot_t1_torch / 1000:8.2f} ms   lumen {tot_t1_lumen / 1000:8.2f} ms"
        f"   ({tot_t1_torch / tot_torch * 100:.1f}% of conv time)"
    )
    print(
        f"  time extent >  1      torch {tot_tn_torch / 1000:8.2f} ms   lumen {tot_tn_lumen / 1000:8.2f} ms"
        f"   ({tot_tn_torch / tot_torch * 100:.1f}% of conv time)"
    )

    head("what this means for the patch")
    print(f"  Covered by lumen_vae_conv.py today : {tot_t1_torch / tot_torch * 100:5.1f}% of convolution time")
    if tot_tn_torch > 0:
        speedup = tot_tn_torch / tot_tn_lumen
        print(f"  On the uncovered T>1 shapes, FlyDSL is {speedup:.2f}x torch")
        if speedup > 1.15:
            print("  -> extending the patch to T>1 looks worth doing")
        else:
            print("  -> extending the patch to T>1 would buy little at these shapes")
    if failures:
        print("\n  configurations Lumen could not take:")
        for f in failures:
            print(f"    {f}")


if __name__ == "__main__":
    main()
