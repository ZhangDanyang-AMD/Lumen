#!/usr/bin/env python3
"""Trace a video VAE's convolutions and attribute them to a patch mode.

``lumen_vae_conv.py`` has two rewrites, and which of them a given call is
eligible for is not something you can read off the module list. A causal 3-D
convolution collapses to an exact 2-D one only when its time extent is 1 *and*
no feature cache supplied its leading frames, and a video VAE encodes a clip in
chunks that carry a cache, so plenty of T == 1 calls are still not reducible.

For every distinct convolution one encode performs this reports the shape as the
kernel actually sees it, whether a cache was present, torch's time and Lumen's.
It then splits the total four ways -- reducible to 2-D, needing the 3-D kernel,
pointwise, plain 2-D resampler -- so the two patch modes can be sized separately
rather than assumed.

    PYTHONPATH=$LUMEN_PYTHONPATH python3 trace_video_vae_convs.py \
        --model /work/models-extra/Wan2.1-T2V-1.3B --frames 17 --height 480 --width 832
"""

import argparse
import collections
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumen_vae_conv import _degenerate_to_conv2d, _is_causal_conv3d


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


# The four ways a convolution can relate to the patch. Order is report order.
BUCKETS = (
    ("reducible", "causal, T=1, uncached -> exact conv2d   (vae_conv covers)"),
    ("kernel3d", "causal, T>1 or cached  -> conv3d kernel  (vae_conv_video adds)"),
    ("plain2d", "plain nn.Conv2d resampler                (vae_conv_video adds)"),
    ("pointwise", "spatial kernel 1                         (left on torch)"),
)


def classify(mod, x, cached):
    """Which bucket does one call fall into?"""
    ks = tuple(mod.kernel_size)
    if max(ks[-2:]) < 2:
        return "pointwise"
    if not _is_causal_conv3d(mod):
        return "plain2d"
    if not cached and x.dim() == 5 and x.shape[2] == 1 and _degenerate_to_conv2d(mod) is not None:
        return "reducible"
    return "kernel3d"


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
            # A causal module takes the cache as its second positional argument.
            cached = len(inputs) > 1 and inputs[1] is not None
            key = (
                tuple(x.shape),
                cached,
                mod.weight.shape[0],
                tuple(mod.kernel_size),
                tuple(mod.stride),
                tuple(getattr(mod, "_padding", ())) or tuple(mod.padding),
                classify(mod, x, cached),
            )
            entry = calls.setdefault((name, key), {"n": 0})
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

    head("per configuration: torch vs Lumen, on the shape the kernel sees")
    import lumen.ops.conv as conv_ops

    hdr = (
        f"{'layer':30}{'kernel input':>22}{'k':>8}{'cache':>6}{'n':>4}"
        f"{'torch':>11}{'lumen':>11}{'ratio':>8}  bucket"
    )
    print(hdr)
    print("-" * (len(hdr) + 4))

    totals = {b: [0.0, 0.0, 0] for b, _ in BUCKETS}  # torch us, lumen us, calls
    reducible_2d = [0.0, 0.0]  # torch conv2d, lumen conv2d -- what the rewrite buys
    failures = []

    for (name, key), meta in calls.items():
        shape, cached, cout, ksize, stride, pad, bucket = key
        n = meta["n"]
        is3d = len(ksize) == 3
        cin = shape[1]

        x_in = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(cout, cin, *ksize, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(cout, device="cuda", dtype=torch.bfloat16)

        # Reproduce what the module hands the convolution. A causal module has
        # moved its padding into _padding and convolves with 0; when a cache is
        # present it concatenates instead of padding, which lands on the same
        # shape, so one F.pad models both.
        if len(pad) == 6:
            x_in = F.pad(x_in, list(pad))
            p = 0
        else:
            p = tuple(pad)

        conv = F.conv3d if is3d else F.conv2d
        op = conv_ops.conv3d if is3d else conv_ops.conv2d
        try:
            t_torch = timed(lambda: conv(x_in, w, b, stride=stride, padding=p))
            t_lumen = timed(lambda: op(x_in, w, b, stride=stride, padding=p))
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {str(exc)[:70]}")
            continue

        totals[bucket][0] += t_torch * n
        totals[bucket][1] += t_lumen * n
        totals[bucket][2] += n

        if bucket == "reducible":
            # The 2-D rewrite is the alternative for this bucket, so measure it
            # rather than crediting the bucket with the 3-D numbers.
            x2 = x_in[:, :, -1]
            w2 = w[:, :, -1].contiguous()
            sp = (pad[2], pad[0]) if len(pad) == 6 else p
            reducible_2d[0] += timed(lambda: F.conv2d(x2, w2, b, stride=stride[1:], padding=sp)) * n
            reducible_2d[1] += timed(lambda: conv_ops.conv2d(x2, w2, b, stride=stride[1:], padding=sp)) * n

        kshow = "x".join(str(v) for v in ksize)
        print(
            f"{name[:30]:30}{str(tuple(x_in.shape)):>22}{kshow:>8}{'yes' if cached else '-':>6}{n:>4}"
            f"{t_torch:>9.1f}us{t_lumen:>9.1f}us{t_torch / t_lumen:>7.2f}x  {bucket}"
        )

    head("totals for one encode, by bucket")
    tot_torch = sum(v[0] for v in totals.values())
    print(f"{'bucket':12}{'calls':>7}{'torch':>11}{'lumen':>11}{'ratio':>8}{'share':>8}   what it is")
    for b, desc in BUCKETS:
        t, l, n = totals[b]
        if n == 0:
            continue
        print(
            f"{b:12}{n:>7}{t / 1000:>9.2f}ms{l / 1000:>9.2f}ms{t / l:>7.2f}x"
            f"{t / tot_torch * 100:>7.1f}%   {desc}"
        )
    print(f"{'ALL':12}{sum(v[2] for v in totals.values()):>7}{tot_torch / 1000:>9.2f}ms")

    head("what each patch mode claims")
    red_t, red_l = totals["reducible"][0], totals["reducible"][1]
    if red_t:
        print(
            f"  vae_conv        (T=1 rewrite only): {red_t / 1000:6.2f}ms of {tot_torch / 1000:.2f}ms"
            f" = {red_t / tot_torch * 100:4.1f}% of convolution time"
        )
        print(
            f"                  those calls as conv2d: torch {reducible_2d[0] / 1000:.2f}ms,"
            f" lumen {reducible_2d[1] / 1000:.2f}ms"
            f"  (rewrite {red_t / reducible_2d[0]:.2f}x, +kernel {red_t / reducible_2d[1]:.2f}x)"
        )
        covered = tot_torch - red_t + reducible_2d[1]
    else:
        print("  vae_conv        (T=1 rewrite only):   0.00ms -- no call is reducible")
        covered = tot_torch
    add_t = totals["kernel3d"][0] + totals["plain2d"][0]
    add_l = totals["kernel3d"][1] + totals["plain2d"][1]
    if add_t:
        print(
            f"  vae_conv_video  (adds the kernel)  : {add_t / 1000:6.2f}ms of {tot_torch / 1000:.2f}ms"
            f" = {add_t / tot_torch * 100:4.1f}%, at {add_t / add_l:.2f}x"
        )
    best = covered - add_t + add_l
    print(f"\n  one encode's convolutions, torch          : {tot_torch / 1000:6.2f}ms")
    print(f"  with vae_conv       (image patch as it is): {covered / 1000:6.2f}ms  ({tot_torch / covered:.2f}x)")
    print(f"  with vae_conv_video (both rewrites)       : {best / 1000:6.2f}ms  ({tot_torch / best:.2f}x)")
    print("\n  Microbenchmarks of isolated convolutions. The share of a training step")
    print("  they represent is a separate measurement -- see the example README.")
    if failures:
        print("\n  configurations Lumen could not take:")
        for f in failures:
            print(f"    {f}")


if __name__ == "__main__":
    main()
