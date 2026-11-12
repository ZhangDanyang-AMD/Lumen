#!/usr/bin/env python3
"""Single-GPU check of LumenConfig(aiter_conv=True) on SimWAM's Wan2.2 VAE at NAVSIM's shapes.

Encodes (1, 3, 9, 384, 672) and decodes (1, 48, 3, 24, 42) through the stock BF16
WanVideoVAE38 and through the patched one, and reports for each:

  * convolution calls on aiter vs torch (a fallback is otherwise silent)
  * patched against stock BF16 at rtol = atol = 1e-5: max|d|, how many elements
    are outside, PASS/FAIL
  * SNR of stock and patched against an FP32 encode/decode, and ms per call

then runs every distinct NAVSIM convolution on {-1, 0, 1} data, where the exact
answer is known (wan22_vae.ternary_conv_check), at the same tolerance.

    SIMWAM_DIR=... WAN22_VAE=... python3 verify_vae.py

Needs aiter with Triton conv3d (aiter >= c8325e00c, or AITER_SRC; see README).
"""

import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wan22_vae as W  # noqa: E402

RTOL = ATOL = 1e-5


def snr_db(ref, test):
    noise = (ref.float() - test.float()).pow(2).sum()
    return float("inf") if noise == 0 else (10 * torch.log10(ref.float().pow(2).sum() / noise)).item()


def timed(fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def compare(got, want):
    d = (got.float() - want.float()).abs()
    outside = (d > ATOL + RTOL * want.float().abs()).sum().item()
    return d.max().item(), outside, want.numel()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--skip-ternary", action="store_true", help="skip the per-convolution exact check")
    args = ap.parse_args()

    import lumen.ops.conv as lc
    from lumen.config import LumenConfig

    if lc._conv3d() is None:
        raise SystemExit("aiter Triton conv3d unavailable: install aiter >= c8325e00c or set AITER_SRC")

    base = W.load(device="cpu")
    n_c3 = sum(type(m).__name__ == "CausalConv3d" for m in base.modules())
    n_c2 = sum(type(m) is torch.nn.Conv2d for m in base.modules())
    print(f"WanVideoVAE38: {sum(p.numel() for p in base.parameters()) / 1e9:.4f}B params, {n_c3} CausalConv3d, "
          f"{n_c2} Conv2d, z_dim {base.z_dim}, spatial {base.upsampling_factor}x, "
          f"temporal {base.temporal_downsample_factor}x")  # fmt: skip

    torch.manual_seed(0)
    x32 = torch.rand(W.NAVSIM_VIDEO, device="cuda") * 2 - 1
    vae32 = copy.deepcopy(base).cuda()
    z32 = W.encode(vae32, x32)
    v32 = W.decode(vae32, z32)
    del vae32
    per_frame, tokens = W.dit_tokens(z32.shape)
    print(f"input {tuple(x32.shape)} -> latent {tuple(z32.shape)} -> decode {tuple(v32.shape)}; "
          f"DiT patch {list(W.DIT_PATCH)}: {per_frame} tokens per latent frame, {tokens} total")  # fmt: skip
    if tuple(z32.shape) != W.NAVSIM_LATENT or tuple(v32.shape) != W.NAVSIM_VIDEO or tokens != 756:
        raise SystemExit("shapes do not match SimWAM NAVSIM")

    stock = copy.deepcopy(base).cuda().bfloat16()
    patched = copy.deepcopy(base).cuda().bfloat16()
    LumenConfig(aiter_conv=True).enable(patched)
    bound = sum(getattr(m._conv_forward, "_lumen_conv", False) for m in W.convs(patched))
    print(f"LumenConfig(aiter_conv=True): {bound} of {n_c3 + n_c2} convolutions bound")
    x = x32.bfloat16()
    z = W.encode(stock, x)  # decode input for both, so decode is compared on the same latent

    verdict = True
    for what, fn, inp, ref32 in (("encode", W.encode, x, z32), ("decode", W.decode, z, v32)):
        want = fn(stock, inp)
        with W.ConvCalls(patched, lc) as calls:
            got = fn(patched, inp)
        ms_stock = timed(lambda: fn(stock, inp), args.iters)
        ms_patched = timed(lambda: fn(patched, inp), args.iters)
        d, outside, n = compare(got, want)
        ok = outside == 0 and calls.torch == 0
        verdict &= ok
        print(f"{what}: conv calls {calls.as_dict()}   patched vs stock BF16 max|d| {d:.3e}, "
              f"{outside}/{n} outside rtol=atol=1e-5 -> {'PASS' if ok else 'FAIL'}")  # fmt: skip
        print(f"        SNR vs FP32: stock {snr_db(ref32, want):5.1f} dB, patched {snr_db(ref32, got):5.1f} dB;"
              f"   ms: stock {ms_stock:7.1f}, patched {ms_patched:7.1f}")  # fmt: skip

    if not args.skip_ternary:
        gen = torch.Generator().manual_seed(0)
        sigs = W.conv_signatures(base)
        exact = fallback = 0
        failed, winograd = [], []
        for sig in sigs:
            route = W.aiter_route(sig)
            aiter_calls, d, got, ref = W.ternary_conv_check(sig, lc, gen)
            if aiter_calls != 1:
                fallback += 1
                failed.append((sig, route, "torch fallback"))
                continue
            if W.is_winograd(route):
                winograd.append((sig, route, d))
            elif compare(got, ref)[1] == 0:
                exact += 1
            else:
                failed.append((sig, route, f"max|d| {d:.3e}"))
        print(f"ternary per-convolution check, {len(sigs)} NAVSIM shapes: {exact} exact at rtol=atol=1e-5, "
              f"{len(winograd)} Winograd (not exact by construction, not asserted), "
              f"{len(failed)} failed ({fallback} fell back to torch)")  # fmt: skip
        for sig, route, d in winograd:
            print(f"    Winograd {route}: {sig} max|d| {d:.3e}")
        for sig, route, why in failed:
            print(f"    FAILED {route}: {sig} {why}")
        verdict &= not failed
    print("RESULT:", "PASS" if verdict else "FAIL")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
