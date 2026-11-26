###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""TE MXFP8BlockScaling reference numerics, and Lumen's fast MXFP8 path checked against them."""

import pytest
import torch

from lumen.ops.quantize import mxfp8_te_reference as te

E4M3_MAX = te.E4M3_MAX


@pytest.mark.parametrize(
    "amax,biased",
    [
        (448.0, 127),  # exactly representable: scale 1
        (460.0, 128),  # just above 448: TE rounds the exponent up (torchao keeps 127 and clips)
        (230.0, 127),  # 230/448 < 1: scale 1, 230 stays exact
        (224.0, 126),  # 224/448 = 0.5 = 2^-1 exactly
        (500.0, 128),
        (1e-30, 19),  # tiny: ceil(log2(1e-30 / 448)) = -108
        (0.0, 0),
        (3e38, 247),  # fp32 max / 448 stays below 2^128, so the 254 clamp is never reached from fp32
    ],
)
def test_e8m0_scale_rounds_up(amax, biased):
    assert te.e8m0_scale_exponent(torch.tensor([amax])).item() == biased


def test_no_block_saturates_and_scale_is_tight():
    torch.manual_seed(0)
    x = torch.randn(64, 256) * torch.logspace(-3, 3, 64).unsqueeze(1)
    q, biased = te.quantize(x, axis=-1)
    blocks = x.reshape(64, -1, 32)
    scaled_amax = blocks.abs().amax(-1) * torch.exp2((127 - biased).float())
    assert (scaled_amax <= E4M3_MAX).all()  # round-up exponent: amax never clips
    assert (scaled_amax > E4M3_MAX / 2).all()  # and never wastes a binade
    assert q.float().abs().max() <= E4M3_MAX


def test_axes_and_partial_blocks():
    torch.manual_seed(0)
    x = torch.randn(48, 80) * torch.logspace(-2, 2, 48).unsqueeze(1)  # per-row scale differs from per-column
    row, col = te.fake_quantize(x, -1), te.fake_quantize(x, 0)
    assert row.shape == col.shape == x.shape
    assert not torch.equal(row, col)  # blocks along K vs along M differ
    torch.testing.assert_close(te.fake_quantize(x.t(), 0), row.t())
    assert ((row - x).abs() / x.abs().clamp_min(1e-3)).max() < 0.07  # e4m3: 3 mantissa bits


def test_linear_operand_axes_and_main_grad():
    torch.manual_seed(0)
    x = torch.randn(64, 96, requires_grad=True)
    w = torch.randn(32, 96, requires_grad=True)
    dy = torch.randn(64, 32)
    y = te.te_mxfp8_linear(x, w)
    y.backward(dy)
    fq = te.fake_quantize
    torch.testing.assert_close(y, fq(x, -1) @ fq(w, -1).t())  # fprop: X, W rowwise (along K)
    torch.testing.assert_close(x.grad, fq(dy, -1) @ fq(w, 0))  # dgrad: dY rowwise x W columnwise
    torch.testing.assert_close(w.grad, fq(dy, 0).t() @ fq(x, 0))  # wgrad: dY, X columnwise (along M)

    w2 = w.detach().clone().requires_grad_(True)
    te.attach_main_grad([w2])
    for _ in range(2):
        te.te_mxfp8_linear(x.detach(), w2).backward(dy)
    assert w2.grad is None
    torch.testing.assert_close(w2.main_grad, 2 * w.grad.float())


def _gfx950():
    return torch.cuda.is_available() and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


@pytest.mark.skipif(not _gfx950(), reason="Lumen's MXFP8 kernels need gfx950")
def test_lumen_mxfp8_vs_te():
    """Lumen's MXFP8 linear against the TE MXFP8BlockScaling reference.

    Bit-exact: every quantized operand (data and E8M0 bytes), i.e. X/W rowwise for fprop, dY rowwise
    and W columnwise for dgrad, dY/X columnwise for wgrad, all quantized from the bf16 tensors.
    Close (fp32 summation order only): fprop, dgrad (bf16 out) and wgrad (fp32).
    """
    from lumen.ops.quantize.linear import _mxfp8_colwise, quantize_input, quantized_linear

    torch.manual_seed(0)
    fp8 = torch.float8_e4m3fn
    x = (torch.randn(256, 512, device="cuda") * 3).bfloat16()
    w = (torch.randn(384, 512, device="cuda") * 0.05).bfloat16()
    dy = torch.randn(256, 384, device="cuda").bfloat16()

    def same(name, lumen_q, lumen_s, ref_t, axis):
        q, b = te.quantize(ref_t, axis)
        if axis == 0:  # Lumen stores columnwise copies transposed
            q = q.t()
        ls = lumen_s.view(torch.uint8).to(torch.int32)
        assert torch.equal(ls.reshape(b.shape), b), f"{name}: E8M0 bytes differ"
        assert torch.equal(lumen_q.view(torch.uint8), q.contiguous().view(torch.uint8)), f"{name}: E4M3 data differ"

    for name, t in [("X rowwise", x), ("W rowwise", w), ("dY rowwise", dy)]:
        d = quantize_input(t, "mxfp8", fp8, 32)
        same(name, d.data, d.scale, t, -1)
    for name, t in [("X columnwise", x), ("W columnwise", w), ("dY columnwise", dy)]:
        qd, qs = _mxfp8_colwise(t, fp8)
        same(name, qd, qs, t, 0)

    out = {}
    for name, fn in [("te", te.te_mxfp8_linear),
                     ("lumen", lambda a, b: quantized_linear(a, b, scaling_type="mxfp8", block_size=32))]:  # fmt: skip
        xs, ws = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
        y = fn(xs, ws)
        y.backward(dy)
        out[name] = (y.float(), xs.grad.float(), ws.grad.float())
    for i, n in enumerate(["fprop", "dgrad", "wgrad"]):
        r, o = out["te"][i], out["lumen"][i]
        e = ((o - r).norm() / r.norm()).item()
        print(f"{n}: ||lumen - te|| / ||te|| = {e:.3e}")
        assert e < 1e-2, n  # same quantized operands; bf16 output rounding / summation order only
