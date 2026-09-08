#!/usr/bin/env python3
"""Verify Lumen's own conv3d: reached without AITER, correct, and faster.

The point of this path is that it is Lumen-only -- the kernel lives in
lumen/kernels/conv/ and talks to the installed flydsl compiler directly, so no
AITER release has to carry it. This checks that claim (aiter must never appear in
sys.modules), then correctness against fp32 torch, then throughput.

    PYTHONPATH=$LUMEN_PYTHONPATH python3 $EXAMPLE_DIR/verify_lumen_conv.py
"""

import sys

import torch
import torch.nn.functional as F


def head(t):
    print(f"\n=== {t} ===")


head("wiring")
# lumen/__init__.py imports lumen.quantize, which imports aiter. That is a
# pre-existing dependency of the package, unrelated to conv. What matters for
# this op is narrower: does the *conv path* pull in aiter? Snapshot before and
# after to answer that rather than the broader question.
import lumen  # noqa: E402

_aiter_after_lumen = {m for m in sys.modules if m == "aiter" or m.startswith("aiter.")}

import lumen.ops.conv as conv_ops  # noqa: E402
from lumen.kernels.conv.conv3d_implicit import conv3d_implicit  # noqa: E402
from lumen.ops.dispatch import (  # noqa: E402
    FLYDSL_FALLBACK_ORDER,
    Backend,
    _probe_flydsl_conv3d,
)

_aiter_after_conv = {m for m in sys.modules if m == "aiter" or m.startswith("aiter.")}

print(f"lumen    {lumen.__version__}  {lumen.__file__}")
print(f"ops.conv {conv_ops.__file__}")
print(f"kernel   {conv3d_implicit.__module__}")
print(f"exports  {conv_ops.__all__}")
print(f"Backend  {[b.name for b in Backend]}")
print(f"chain    {[b.name for b in FLYDSL_FALLBACK_ORDER]}")
print(f"probe    _probe_flydsl_conv3d() = {_probe_flydsl_conv3d()}")

try:
    import aiter

    print(f"aiter    {getattr(aiter, '__file__', '?')}  (pulled in by lumen.quantize)")
except Exception as exc:
    print(f"aiter    not importable ({type(exc).__name__})")

added = _aiter_after_conv - _aiter_after_lumen
print(f"aiter modules added by importing the conv path: {sorted(added) if added else 'none'}")

# Reaching the op through the lazy re-export too, which is what a caller writing
# `lumen.ops.conv3d(...)` gets.
import lumen.ops as ops  # noqa: E402

print(f"lumen.ops.conv3d resolves: {ops.conv3d is conv_ops.conv3d}")

arch = torch.cuda.get_device_properties(0).gcnArchName
print(f"device   {arch}")

MIN_SNR = 25


def snr(ref, test):
    signal = torch.norm(ref.float()).pow(2)
    noise = torch.norm(ref.float() - test.float()).pow(2)
    return 10.0 * torch.log10(signal / (noise + 1e-12)).item()


def ref_conv(x, w, b=None, **kw):
    """fp32 reference, for correctness only.

    Deliberately not the timing baseline: fp32 convolution is far slower than
    bf16, so comparing against it would inflate the speedup several-fold.
    """
    conv = {2: F.conv2d, 3: F.conv3d}[w.dim() - 2]
    out = conv(x.float(), w.float(), bias=b.float() if b is not None else None, **kw)
    return out.to(x.dtype)


def torch_conv_bf16(x, w, b=None, **kw):
    """The timing baseline: what the model runs today, same dtype as the kernel."""
    conv = {2: F.conv2d, 3: F.conv3d}[w.dim() - 2]
    return conv(x, w, bias=b, **kw)


def timed(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters


def rand(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)


# Qwen-Image VAE at 1024x1024, plus the hottest 1328 layer. (cin, cout, hw, stride,
# padding, prepad, calls per VAE forward)
SHAPES = [
    ("enc_conv_in", 3, 96, 1024, 1, 1, False, 1),
    ("enc_e0_res__dec_d3_res", 96, 96, 1024, 1, 1, False, 10),
    ("enc_e1_res2__dec_d2_res", 192, 192, 512, 1, 1, False, 9),
    ("enc_e2_res2__dec_d1_res", 384, 384, 256, 1, 1, False, 8),
    ("enc_e3_mid__dec_mid_d0", 384, 384, 128, 1, 1, False, 18),
    ("dec_conv_out", 96, 3, 1024, 1, 1, False, 1),
    ("dec_bottleneck_1328", 384, 384, 166, 1, 1, False, 18),
    ("dec_d3_res_hot_1328", 96, 96, 1328, 1, 1, False, 10),
    ("enc_e0_downsample", 96, 96, 1024, 2, 0, True, 1),
    ("enc_e2_downsample_spatial", 384, 384, 256, 2, 0, True, 1),
    ("dec_d2_upsample", 192, 96, 1024, 1, 1, False, 1),
]

head("correctness + throughput on Qwen-Image VAE shapes")
hdr = f"{'layer':28}{'freq':>5}{'torch bf16':>13}{'NCHW':>11}{'NHWC':>11}{'NCHW':>7}{'NHWC':>7}{'SNR dB':>9}"
print(hdr)
print("-" * len(hdr))

tot_t = tot_n = tot_h = 0.0
worst_snr = 1e9
failures = []
for sid, cin, cout, hw, stride, pad, prepad, freq in SHAPES:
    torch.manual_seed(8800 + cin + cout + hw)
    x = rand(1, cin, hw, hw)
    if prepad:
        x = F.pad(x, (0, 1, 0, 1))
    w, b = rand(cout, cin, 3, 3), rand(cout)

    ref = ref_conv(x, w, b, stride=stride, padding=pad)
    out = conv_ops.conv2d(x, w, b, stride=stride, padding=pad)
    if out.shape != ref.shape:
        failures.append(f"{sid}: shape {tuple(out.shape)} != {tuple(ref.shape)}")
        continue
    s_nchw = snr(ref, out)

    xh = x.permute(0, 2, 3, 1).contiguous()
    out_h = conv_ops.conv2d(xh, w, b, stride=stride, padding=pad, input_layout="NHWC", output_layout="NHWC")
    s_nhwc = snr(ref, out_h.permute(0, 3, 1, 2))
    worst_snr = min(worst_snr, s_nchw, s_nhwc)
    if min(s_nchw, s_nhwc) < MIN_SNR:
        failures.append(f"{sid}: SNR {min(s_nchw, s_nhwc):.1f} dB below {MIN_SNR}")

    t_t = timed(lambda: torch_conv_bf16(x, w, b, stride=stride, padding=pad))
    t_n = timed(lambda: conv_ops.conv2d(x, w, b, stride=stride, padding=pad))
    t_h = timed(
        lambda: conv_ops.conv2d(xh, w, b, stride=stride, padding=pad, input_layout="NHWC", output_layout="NHWC")
    )
    tot_t += t_t * freq
    tot_n += t_n * freq
    tot_h += t_h * freq
    print(
        f"{sid:28}{freq:>5}{t_t:>9.1f}us{t_n:>9.1f}us{t_h:>9.1f}us"
        f"{t_t / t_n:>6.2f}x{t_t / t_h:>6.2f}x{min(s_nchw, s_nhwc):>9.1f}"
    )

head("frequency-weighted, one VAE forward")
print(f"  torch bf16 (baseline) {tot_t / 1000:>8.2f} ms   1.00x")
print(f"  lumen conv2d NCHW     {tot_n / 1000:>8.2f} ms  {tot_t / tot_n:>5.2f}x")
print(f"  lumen conv2d NHWC     {tot_h / 1000:>8.2f} ms  {tot_t / tot_h:>5.2f}x")
print(f"  channels-last saves   {(tot_n - tot_h) / tot_n * 100:>7.1f}% over NCHW")

head("conv3d (5D filter)")
x3, w3, b3 = rand(1, 32, 4, 16, 16), rand(64, 32, 3, 3, 3), rand(64)
out3 = conv_ops.conv3d(x3, w3, b3, stride=1, padding=1)
ref3 = ref_conv(x3, w3, b3, stride=1, padding=1)
s3 = snr(ref3, out3)
print(f"  shape {tuple(out3.shape)}  SNR {s3:.1f} dB")
if s3 < MIN_SNR:
    failures.append(f"conv3d: SNR {s3:.1f} dB")

head("autograd routes to torch (the kernel has no backward)")
xg = rand(1, 32, 16, 16).requires_grad_(True)
wg = rand(64, 32, 3, 3).requires_grad_(True)
og = conv_ops.conv2d(xg, wg, stride=1, padding=1)
print(f"  requires_grad={og.requires_grad}  grad_fn={type(og.grad_fn).__name__ if og.grad_fn else None}")
og.float().pow(2).mean().backward()
ok_grad = xg.grad is not None and wg.grad is not None and torch.isfinite(xg.grad).all()
print(f"  grads present and finite: {bool(ok_grad)}")
if not (og.requires_grad and ok_grad):
    failures.append("autograd: no usable gradient")

head("unsupported dtype demotes to torch instead of failing")
x32 = torch.randn(1, 32, 16, 16, device="cuda", dtype=torch.float32)
w32 = torch.randn(64, 32, 3, 3, device="cuda", dtype=torch.float32)
try:
    o32 = conv_ops.conv2d(x32, w32, stride=1, padding=1)
    ok32 = o32.dtype is torch.float32 and torch.allclose(o32, F.conv2d(x32, w32, stride=1, padding=1))
    print(f"  fp32 -> torch fallback: dtype {o32.dtype}, matches F.conv2d: {ok32}")
    if not ok32:
        failures.append("fp32 fallback produced a wrong result")
except Exception as exc:
    print(f"  fp32 raised {type(exc).__name__}: {exc}")
    failures.append("fp32 did not demote to torch")

head("argument errors raise ValueError, not AssertionError")
cases = [
    ("mixed dtype", lambda: conv_ops.conv2d(x32, rand(32, 32, 3, 3))),
    ("negative padding", lambda: conv_ops.conv2d(rand(1, 32, 16, 16), rand(32, 32, 3, 3), padding=-1)),
    ("bad layout", lambda: conv_ops.conv2d(rand(1, 32, 16, 16), rand(32, 32, 3, 3), input_layout="NCDHW")),
    ("wrong filter rank", lambda: conv_ops.conv2d(rand(1, 32, 4, 16, 16), rand(32, 32, 3, 3, 3))),
    ("bias wrong size", lambda: conv_ops.conv2d(rand(1, 32, 16, 16), rand(64, 32, 3, 3), rand(32))),
    ("groups not divisible", lambda: conv_ops.conv2d(rand(1, 30, 16, 16), rand(30, 10, 3, 3), groups=4)),
]
for label, fn in cases:
    try:
        fn()
        print(f"  {label:22} NO RAISE (unexpected)")
        failures.append(f"argument error {label} did not raise")
    except ValueError as exc:
        print(f"  {label:22} ValueError: {str(exc)[:50]}")
    except AssertionError:
        print(f"  {label:22} AssertionError (would abort a fallback chain)")
        failures.append(f"argument error {label} raised AssertionError")

head("verdict")
# Anything aiter touched during the timed section would also show up here.
_aiter_at_end = {m for m in sys.modules if m == "aiter" or m.startswith("aiter.")}
conv_added = _aiter_at_end - _aiter_after_lumen
if conv_added:
    failures.append(f"conv path pulled in aiter modules: {sorted(conv_added)}")

if failures:
    for f in failures:
        print(f"  FAIL {f}")
    sys.exit(1)
print(f"  correctness: worst SNR {worst_snr:.1f} dB (threshold {MIN_SNR})")
print(f"  throughput : {tot_t / tot_n:.2f}x NCHW, {tot_t / tot_h:.2f}x channels-last")
print("  aiter      : conv path added no aiter module (lumen.quantize imports it")
print("               at package level, which is pre-existing and unrelated)")
print("\nPASS")
