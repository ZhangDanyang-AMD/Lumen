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
    """Reports how far Lumen's fast MXFP8 path (aiter torchao-style scales) is from TE's numerics.

    The KDA Megatron reference uses ``te_mxfp8_linear``; this test documents the gap and fails only
    if the scale rule disagrees in a way other than the known round-down (one binade smaller).
    """
    from lumen.ops.quantize.linear import quantized_linear
    from lumen.ops.quantize.ops import convert_to_mxfp8

    torch.manual_seed(0)
    x = (torch.randn(256, 512, device="cuda") * 3).bfloat16()
    _, lumen_scale = convert_to_mxfp8(x, block_size=32, axis=-1, float8_dtype_pt=torch.float8_e4m3fn)
    te_scale = te.e8m0_scale_exponent(x.float().reshape(256, -1, 32).abs().amax(-1))
    diff = te_scale - lumen_scale.view(256, -1).to(torch.int32)
    print(f"scale bytes differing: {(diff != 0).float().mean().item():.2%}, values {diff.unique().tolist()}")
    assert set(diff.unique().tolist()) <= {0, 1}

    w = (torch.randn(384, 512, device="cuda") * 0.05).bfloat16()
    dy = torch.randn(256, 384, device="cuda").bfloat16()
    out = {}
    for name, fn in [("te", te.te_mxfp8_linear),
                     ("lumen", lambda a, b: quantized_linear(a, b, scaling_type="mxfp8", block_size=32))]:  # fmt: skip
        xs, ws = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
        y = fn(xs, ws)
        y.backward(dy)
        out[name] = (y.float(), xs.grad.float(), ws.grad.float())
    for i, n in enumerate(["fprop", "dgrad", "wgrad"]):
        r, o = out["te"][i], out["lumen"][i]
        print(f"{n}: ||lumen - te|| / ||te|| = {((o - r).norm() / r.norm()).item():.3e}")
