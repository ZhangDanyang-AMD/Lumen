#!/usr/bin/env python3
"""Single-GPU timing of the NAVSIM GRPO 1-frame encode: torch BF16 against this example's FlyDSL conv.

One process, one card, one seeded (1, 3, 1, 384, 672) BF16 frame. Each variant is timed
over --iters synchronized encodes after --warmup, in the order --order gives (run a
second round on another card in the reverse order and average the medians):

  torch      stock BF16, cudnn.deterministic=False (MIOpen's default algorithms)
  torchdet   stock BF16, cudnn.deterministic=True (what SimWAM's NAVSIM eval sets)
  flydsl     the VAE patched by flydsl_conv.patch_vae

Then, as kernel-level evidence, torch.profiler records --prof-iters encodes of the stock
(cudnn.deterministic=False) and the FlyDSL model and sums the GPU time of every kernel,
split into convolution kernels and the rest, per encode. Timing a single convolution
call with a synchronize around it measures mostly launch and host time (a 1x1 conv
takes as long as a 3x3 one that way), so the profiler is what shows the kernels.

    python3 bench_encode.py --order torch,torchdet,flydsl --out round1.json
"""

import argparse
import json
import os
import re
import statistics as st
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flydsl_conv as FC  # noqa: E402
import wan22_vae as W  # noqa: E402


def timed(fn, iters, warmup):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    q = sorted(ts)
    return {
        "median_ms": st.median(ts),
        "p10_ms": q[len(q) // 10],
        "p90_ms": q[(9 * len(q)) // 10],
        "cv_pct": 100 * st.pstdev(ts) / st.mean(ts),
    }


_CONV_KERNEL = re.compile(r"conv|igemm|miopen|winograd|implicit|xdlops|naive_|im2col|gemm", re.I)


def profile(model, x, iters):
    """GPU kernel time per encode from torch.profiler, split into convolution kernels and the rest."""
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as tprofile

    with torch.no_grad():
        for _ in range(5):
            W.encode(model, x)
        torch.cuda.synchronize()
        with tprofile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                W.encode(model, x)
            torch.cuda.synchronize()
    rows = []
    for e in prof.key_averages():
        t = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
        if t > 0 and e.count:
            rows.append((e.key, t / 1e3 / iters, e.count / iters))
    conv = [r for r in rows if _CONV_KERNEL.search(r[0])]
    other = [r for r in rows if not _CONV_KERNEL.search(r[0])]
    top = sorted(rows, key=lambda r: -r[1])[:6]
    return {
        "gpu_ms": sum(r[1] for r in rows),
        "conv_ms": sum(r[1] for r in conv),
        "other_ms": sum(r[1] for r in other),
        "conv_launches": sum(r[2] for r in conv),
        "top": [(k[:60], round(t, 3), round(n, 1)) for k, t, n in top],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--order", default="torch,torchdet,flydsl")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--prof-iters", type=int, default=20)
    ap.add_argument("--out")
    args = ap.parse_args()
    if FC.load() is None:
        raise SystemExit("aiter FlyDSL conv unavailable: install aiter main >= 78a14ae5d with flydsl 0.5.0")
    torch.backends.cudnn.benchmark = False

    stock = W.load(dtype=torch.bfloat16, device="cuda")
    patched = W.load(dtype=torch.bfloat16, device="cuda")
    FC.patch_vae(patched)
    g = torch.Generator(device="cuda").manual_seed(0)
    x = (torch.rand(W.NAVSIM_FRAME, device="cuda", generator=g) * 2 - 1).bfloat16()

    l0 = FC.lookup_stats()["calls"]
    with torch.no_grad(), FC.Trace() as trace:
        W.encode(patched, x)
    l1 = FC.lookup_stats()["calls"]
    lookups = {k: l1[k] - l0[k] for k in l1}
    FC.warn_if_untuned()

    res = {
        "device": torch.cuda.get_device_name(0),
        "order": args.order,
        "iters": args.iters,
        "warmup": args.warmup,
        "lookups_per_encode": lookups,
        "encode": {},
        "convs": [],
    }
    for v in args.order.split(","):
        torch.backends.cudnn.deterministic = v == "torchdet"
        model = patched if v == "flydsl" else stock
        e = res["encode"][v] = timed(lambda: W.encode(model, x), args.iters, args.warmup)
        print(f"ENCODE {v:9s} median {e['median_ms']:.3f} ms (cv {e['cv_pct']:.2f}%)", flush=True)
    torch.backends.cudnn.deterministic = False

    res["convs"] = [
        {"call": [list(v) if isinstance(v, tuple) else v for v in c], "calls_per_encode": n}
        for c, n in trace.calls.items()
    ]
    res["kernels"] = {v: profile(m, x, args.prof_iters) for v, m in (("torch", stock), ("flydsl", patched))}
    for v, k in res["kernels"].items():
        print(
            f"KERNELS {v:6s} per encode: GPU {k['gpu_ms']:.3f} ms = conv {k['conv_ms']:.3f} ms "
            f"({k['conv_launches']:.0f} launches) + other {k['other_ms']:.3f} ms; top {k['top'][:3]}",
            flush=True,
        )
    print("LOOKUPS", lookups, flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
        print("WROTE", args.out)


if __name__ == "__main__":
    main()
