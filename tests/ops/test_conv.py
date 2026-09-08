###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Tests for lumen.ops.conv: FlyDSL implicit-GEMM convolution.

Covers:
  - Forward correctness vs F.conv2d / F.conv3d (BF16)
  - Qwen-Image VAE shapes, the workload this op exists for
  - Channels-last input/output layouts
  - stride, padding, dilation, groups, bias, padding_mode
  - Autograd routing: the kernel has no backward, so a tensor that requires one
    must come back from torch with a working grad_fn
  - Precondition errors surface as ValueError, not AssertionError, so a
    dispatcher can fall back instead of aborting

Reference: PyTorch F.conv2d / F.conv3d in fp32.
"""

import pytest
import torch
import torch.nn.functional as F
from conftest import compute_snr

import lumen.ops.conv as conv_ops

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# BF16 in, BF16 out, reducing over C*R*S: the deviation from an fp32 reference is
# rounding, not error. GEMM-class threshold from the test guide.
_MIN_SNR = 25


def _ref(x, weight, bias=None, **kw):
    """fp32 reference; conv rank taken from the filter."""
    conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[weight.dim() - 2]
    out = conv(x.float(), weight.float(), bias=bias.float() if bias is not None else None, **kw)
    return out.to(x.dtype)


def _rand(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)


# (cin, cout, hw) from forward-hook traces of AutoencoderKLQwenImage at 1024x1024,
# plus the hottest 1328x1328 layer. All 3x3, stride 1, padding 1.
QWENIMAGE_3X3 = [
    (3, 96, 256),
    (96, 96, 256),
    (96, 192, 128),
    (192, 384, 64),
    (384, 384, 32),
    (384, 32, 32),
    (16, 384, 32),
]
# Resample downsample2d: ZeroPad2d((0,1,0,1)) then stride-2 padding-0 conv, so
# the kernel sees an odd extent.
QWENIMAGE_DOWN = [(96, 96, 128), (192, 192, 64), (384, 384, 32)]


@_CUDA
@pytest.mark.parametrize("cin,cout,hw", QWENIMAGE_3X3, ids=[f"c{a}_{b}_hw{c}" for a, b, c in QWENIMAGE_3X3])
def test_conv2d_qwenimage_3x3(cin, cout, hw):
    x, w = _rand(1, cin, hw, hw), _rand(cout, cin, 3, 3)
    b = _rand(cout)
    out = conv_ops.conv2d(x, w, b, stride=1, padding=1)
    ref = _ref(x, w, b, stride=1, padding=1)
    assert out.shape == ref.shape == (1, cout, hw, hw)
    assert compute_snr(ref, out) > _MIN_SNR


_DOWN_PARAMS = [(c, hw) for c, _, hw in QWENIMAGE_DOWN]


@_CUDA
@pytest.mark.parametrize("c,hw", _DOWN_PARAMS, ids=[f"c{c}_hw{hw}" for c, hw in _DOWN_PARAMS])
def test_conv2d_qwenimage_downsample(c, hw):
    """Odd extent with stride 2: the Resample downsample path."""
    x = F.pad(_rand(1, c, hw, hw), (0, 1, 0, 1))
    w, b = _rand(c, c, 3, 3), _rand(c)
    out = conv_ops.conv2d(x, w, b, stride=2, padding=0)
    ref = _ref(x, w, b, stride=2, padding=0)
    assert out.shape == ref.shape
    assert compute_snr(ref, out) > _MIN_SNR


@_CUDA
def test_conv3d_fwd():
    x, w = _rand(1, 32, 4, 16, 16), _rand(64, 32, 3, 3, 3)
    b = _rand(64)
    out = conv_ops.conv3d(x, w, b, stride=1, padding=1)
    ref = _ref(x, w, b, stride=1, padding=1)
    assert out.shape == ref.shape == (1, 64, 4, 16, 16)
    assert compute_snr(ref, out) > _MIN_SNR


@_CUDA
def test_conv2d_channels_last():
    """NHWC on both sides must match NCHW, and skips the kernel's transpose."""
    x, w, b = _rand(1, 64, 64, 64), _rand(128, 64, 3, 3), _rand(128)
    ref = _ref(x, w, b, stride=1, padding=1)

    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    out = conv_ops.conv2d(x_nhwc, w, b, stride=1, padding=1, input_layout="NHWC", output_layout="NHWC")
    assert out.shape == (1, 64, 64, 128)
    assert compute_snr(ref, out.permute(0, 3, 1, 2)) > _MIN_SNR

    # Mixed: channels-last in, channels-first out.
    out_cf = conv_ops.conv2d(x_nhwc, w, b, stride=1, padding=1, input_layout="NHWC")
    assert out_cf.shape == ref.shape
    assert compute_snr(ref, out_cf) > _MIN_SNR


@_CUDA
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dilation", [1, 2])
def test_conv2d_stride_padding_dilation(stride, padding, dilation):
    x, w = _rand(1, 32, 32, 32), _rand(32, 32, 3, 3)
    out = conv_ops.conv2d(x, w, stride=stride, padding=padding, dilation=dilation)
    ref = _ref(x, w, stride=stride, padding=padding, dilation=dilation)
    assert out.shape == ref.shape
    assert compute_snr(ref, out) > _MIN_SNR


@_CUDA
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_conv2d_groups(groups):
    c = 64
    x, w = _rand(1, c, 32, 32), _rand(c, c // groups, 3, 3)
    out = conv_ops.conv2d(x, w, stride=1, padding=1, groups=groups)
    ref = _ref(x, w, stride=1, padding=1, groups=groups)
    assert out.shape == ref.shape
    assert compute_snr(ref, out) > _MIN_SNR


@_CUDA
@pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
def test_conv2d_padding_mode(mode):
    x, w = _rand(1, 32, 32, 32), _rand(32, 32, 3, 3)
    out = conv_ops.conv2d(x, w, stride=1, padding=1, padding_mode=mode)
    if mode == "zeros":
        ref = _ref(x, w, stride=1, padding=1)
    else:
        ref = _ref(F.pad(x.float(), (1, 1, 1, 1), mode=mode).to(x.dtype), w, stride=1, padding=0)
    assert out.shape == ref.shape
    assert compute_snr(ref, out) > _MIN_SNR


@_CUDA
def test_conv2d_no_bias():
    x, w = _rand(1, 32, 32, 32), _rand(64, 32, 3, 3)
    out = conv_ops.conv2d(x, w, None, stride=1, padding=1)
    assert compute_snr(_ref(x, w, None, stride=1, padding=1), out) > _MIN_SNR


@_CUDA
def test_conv2d_padding_same():
    x, w = _rand(1, 32, 30, 30), _rand(32, 32, 3, 3)
    out = conv_ops.conv2d(x, w, padding="same")
    assert out.shape == x.shape
    assert compute_snr(_ref(x, w, padding=1), out) > _MIN_SNR


@_CUDA
def test_conv2d_autograd_routes_to_torch():
    """A tensor needing grad must come back differentiable.

    The kernel has no backward, so the op sends these to torch. Without that the
    output would have no grad_fn and the graph would break silently.
    """
    x = _rand(1, 32, 16, 16).requires_grad_(True)
    w = _rand(64, 32, 3, 3).requires_grad_(True)
    out = conv_ops.conv2d(x, w, stride=1, padding=1)

    assert out.requires_grad and out.grad_fn is not None
    out.float().pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert w.grad is not None and torch.isfinite(w.grad).all()

    # Values still have to match the no-grad path.
    with torch.no_grad():
        eager = conv_ops.conv2d(x.detach(), w.detach(), stride=1, padding=1)
    assert compute_snr(eager, out.detach()) > 40


@_CUDA
def test_conv2d_inference_mode_has_no_grad_fn():
    """Without requires_grad the FlyDSL path is used and the output is a leaf."""
    x, w = _rand(1, 32, 16, 16), _rand(64, 32, 3, 3)
    out = conv_ops.conv2d(x, w, stride=1, padding=1)
    assert not out.requires_grad


@_CUDA
def test_conv2d_fp32_demotes_to_torch():
    """A dtype the kernel cannot do must fall back, not fail.

    ``_flydsl_precheck`` raises ValueError for non-BF16, which is in
    ``try_backends``' catchable set, so the chain demotes to torch and the caller
    still gets a correct result.
    """
    x = torch.randn(1, 32, 16, 16, device="cuda", dtype=torch.float32)
    w = torch.randn(64, 32, 3, 3, device="cuda", dtype=torch.float32)
    out = conv_ops.conv2d(x, w, stride=1, padding=1)
    ref = F.conv2d(x, w, stride=1, padding=1)
    assert out.dtype is torch.float32
    torch.testing.assert_close(out, ref)


@_CUDA
class TestPreconditions:
    """Argument errors must raise ValueError, not AssertionError.

    AssertionError is not in the dispatcher's catchable set, so an assert leaking
    out of the kernel would abort a fallback chain instead of demoting. These are
    caller mistakes, so they are raised rather than demoted -- unlike an
    unsupported dtype, which demotes (see test_conv2d_fp32_demotes_to_torch).
    """

    def test_mixed_dtype(self):
        x = torch.randn(1, 32, 16, 16, device="cuda", dtype=torch.float32)
        with pytest.raises(ValueError, match="same dtype"):
            conv_ops.conv2d(x, _rand(32, 32, 3, 3))

    def test_rank_mismatch(self):
        with pytest.raises(ValueError, match="expected a 2D filter"):
            conv_ops.conv2d(_rand(1, 32, 4, 16, 16), _rand(32, 32, 3, 3, 3))

    def test_bad_layout(self):
        with pytest.raises(ValueError, match="input_layout"):
            conv_ops.conv2d(_rand(1, 32, 16, 16), _rand(32, 32, 3, 3), input_layout="NCDHW")

    def test_negative_padding(self):
        with pytest.raises(ValueError, match="negative padding"):
            conv_ops.conv2d(_rand(1, 32, 16, 16), _rand(32, 32, 3, 3), padding=-1)

    def test_zero_stride(self):
        with pytest.raises(ValueError, match="stride"):
            conv_ops.conv2d(_rand(1, 32, 16, 16), _rand(32, 32, 3, 3), stride=0)

    def test_bad_padding_mode(self):
        with pytest.raises(ValueError, match="padding_mode"):
            conv_ops.conv2d(_rand(1, 32, 16, 16), _rand(32, 32, 3, 3), padding_mode="wrap")

    def test_groups_not_divisible(self):
        with pytest.raises(ValueError, match="divisible by groups"):
            conv_ops.conv2d(_rand(1, 30, 16, 16), _rand(30, 10, 3, 3), groups=4)

    def test_bias_wrong_size(self):
        with pytest.raises(ValueError, match="bias must be 1-D"):
            conv_ops.conv2d(_rand(1, 32, 16, 16), _rand(64, 32, 3, 3), _rand(32))


@_CUDA
class TestLumenConv3d:
    def test_construction(self):
        mod = conv_ops.LumenConv3d(32, 64, 3, padding=1, device="cuda")
        assert mod.weight.shape == (64, 32, 3, 3, 3)
        assert mod.bias is not None and mod.bias.shape == (64,)

    def test_forward_shape(self):
        mod = conv_ops.LumenConv3d(32, 64, 3, padding=1, device="cuda")
        with torch.no_grad():
            mod.weight.normal_(0, 0.02)
            mod.bias.zero_()
            out = mod(_rand(1, 32, 4, 16, 16))
        assert out.shape == (1, 64, 4, 16, 16)

    def test_no_bias(self):
        mod = conv_ops.LumenConv3d(32, 64, 3, padding=1, bias=False, device="cuda")
        assert mod.bias is None

    def test_matches_functional(self):
        mod = conv_ops.LumenConv3d(32, 64, 3, padding=1, device="cuda")
        with torch.no_grad():
            mod.weight.normal_(0, 0.02)
            mod.bias.normal_(0, 0.02)
            x = _rand(1, 32, 4, 16, 16)
            out = mod(x)
            ref = _ref(x, mod.weight, mod.bias, stride=1, padding=1)
        assert compute_snr(ref, out) > _MIN_SNR
