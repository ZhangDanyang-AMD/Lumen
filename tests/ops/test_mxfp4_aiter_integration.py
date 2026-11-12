"""Numerical and layout checks of the public upstream MXFP4 adapters."""

import pytest
import torch

from lumen.ops.quantize import mxfp4 as adapters
from lumen.ops.quantize import ops


def _gfx950():
    return (
        torch.cuda.is_available()
        and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
    )


pytestmark = pytest.mark.skipif(not _gfx950(), reason="gfx950 required")


@pytest.fixture(autouse=True)
def forbid_legacy(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail(f"unexpected legacy fallback: {args}")

    monkeypatch.setattr(adapters, "_legacy", fail)


@pytest.mark.parametrize("use_sr", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_quant_matches_public_api_and_reconstructs(dtype, use_sr):
    from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant

    x = torch.randn(256, 256, device="cuda", dtype=dtype)
    packed, scales = ops.convert_to_mxfp4(
        x, use_sr=use_sr, philox_seed=123 if use_sr else None, philox_offset=0
    )
    ref, ref_scale = dynamic_mxfp4_quant(
        x,
        use_sr=use_sr,
        philox_seed=123 if use_sr else None,
        philox_offset=0,
        backend="triton",
    )
    torch.testing.assert_close(packed, ref.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(scales, ref_scale.view(torch.uint8), rtol=0, atol=0)
    decoded = ops.convert_from_mxfp4(packed, scales, output_dtype=torch.float32)
    snr = 10 * torch.log10(
        x.float().square().sum() / (x.float() - decoded).square().sum()
    )
    assert snr > 10


def test_blockscale_and_transpose_match_public_api():
    from aiter.ops.triton.quant.fast_transpose import transpose_packed_fp4
    from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant_blockscale

    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    packed, scales = ops.convert_to_mxfp4_2d(x)
    ref, ref_scale = dynamic_mxfp4_quant_blockscale(x)
    torch.testing.assert_close(packed, ref, rtol=0, atol=0)
    torch.testing.assert_close(scales, ref_scale, rtol=0, atol=0)
    transposed = ops.transpose_packed_fp4(packed)
    torch.testing.assert_close(transposed, transpose_packed_fp4(packed), rtol=0, atol=0)
    torch.testing.assert_close(
        ops.transpose_packed_fp4(transposed), packed, rtol=0, atol=0
    )


@pytest.mark.parametrize("use_sr", [False, True])
@pytest.mark.parametrize(
    "swizzle,shuffle", [(False, False), (True, False), (True, True)]
)
def test_h16_fusions_match_upstream_in_each_layout(use_sr, swizzle, shuffle):
    from aiter.ops.triton.quant.dequant_hadamard_quant_mxfp4 import (
        dequant_hadamard_quant_mxfp4,
    )
    from aiter.ops.triton.quant.fused_quant_transpose_hadamard_mxfp4 import (
        fused_quant_transpose_hadamard_mxfp4,
    )

    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    sign = torch.ones(16, device="cuda", dtype=torch.float32)
    sign[::2] = -1
    kwargs = dict(
        use_sr_row=use_sr,
        use_sr_transposed=use_sr,
        philox_seed=456 if use_sr else None,
        philox_offset=0,
        swizzle_scale=swizzle,
        shuffle_col=shuffle,
    )
    got = ops.dual_layout_quant_mxfp4(x, sign, **kwargs)
    ref = fused_quant_transpose_hadamard_mxfp4(x, sign, **kwargs)
    for actual, expected in zip(got, ref):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    kwargs = dict(
        use_sr=use_sr,
        philox_seed=456 if use_sr else None,
        philox_offset=0,
        swizzle_scale=swizzle,
        shuffle_data=shuffle,
        in_scale_swizzled=swizzle,
    )
    got_requant = ops.dequant_hadamard_quant_mxfp4(got[0], got[1], sign, **kwargs)
    ref_requant = dequant_hadamard_quant_mxfp4(got[0], got[1], sign, **kwargs)
    for actual, expected in zip(got_requant, ref_requant):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_swiglu_backward_writes_supplied_buffer():
    from lumen.ops.fused_swiglu import fused_swiglu, fused_swiglu_backward

    x = torch.randn(128, 512, device="cuda", dtype=torch.float32, requires_grad=True)
    gate, up = x.chunk(2, dim=-1)
    reference = torch.nn.functional.silu(gate) * up
    dy = torch.randn_like(reference)
    (grad,) = torch.autograd.grad(reference, x, dy)
    out = torch.empty_like(x)
    got = fused_swiglu_backward(dy, x.detach(), out=out)
    assert got.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        fused_swiglu(x.detach()), reference, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(got, grad, rtol=1e-5, atol=1e-6)


def test_mxfp8_gemm_32_value_scales_match_dequantized_matmul():
    from lumen.ops.quantize.linear import _gemm_mxfp8_triton

    a = torch.randn(128, 4096, device="cuda").to(torch.float8_e4m3fn)
    w = torch.randn(4096, 4096, device="cuda").to(torch.float8_e4m3fn)
    # Raw E8M0 exponent 127 represents a unit scale.
    sa = torch.full((128, 128), 127, dtype=torch.uint8, device="cuda")
    sw = torch.full((4096, 128), 127, dtype=torch.uint8, device="cuda")
    got = _gemm_mxfp8_triton(a, w, sa, sw)
    reference = (a.float() @ w.float().t()).to(torch.bfloat16)
    snr = 10 * torch.log10(
        reference.float().square().sum()
        / (reference.float() - got.float()).square().sum()
    )
    assert snr > 25
