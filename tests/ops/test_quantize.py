# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for Lumen quantization ops, comparing against torchao reference implementation."""

import pytest
import torch
import triton
from torchao.kernel.blockwise_quantization import fp8_blockwise_act_quant
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import (
    MXTensor,
)
from torchao.prototype.mx_formats.mx_tensor import to_dtype as torchao_to_dtype
from torchao.prototype.mx_formats.mx_tensor import to_mx as torchao_to_mx
from torchao.quantization.quant_primitives import (
    _dequantize_affine_float8,
    _quantize_affine_float8,
)

from lumen.ops.quantize import (
    convert_from_mxfp8,
    convert_to_mxfp8,
    dequant_fp8_tensorwise_impl,
    quant_fp8_blockwise_impl,
    quant_fp8_tensorwise_impl,
)


def compute_snr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Signal-to-noise ratio in dB. x is reference."""
    x, y = x.float(), y.float()
    signal_power = torch.norm(x).pow(2)
    noise_power = torch.norm(x - y).pow(2)
    return (10 * torch.log10(signal_power / (noise_power + 1e-12))).detach().item()


# ---------------------------------------------------------------------------
# Tensorwise FP8
# ---------------------------------------------------------------------------

SHAPES = [(64, 128), (128, 256), (256, 512)]
SHAPE_IDS = [f"{m}x{n}" for m, n in SHAPES]


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("dtype_in", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_quant_fp8_tensorwise_vs_torchao(shape, dtype_in, fp8_dtype):
    """Compare Lumen tensorwise quant against torchao _quantize_affine_float8."""
    torch.manual_seed(42)
    x = torch.randn(*shape, device="cuda", dtype=dtype_in)
    fp8_max = torch.finfo(fp8_dtype).max
    amax = x.abs().max().float().clamp(min=1e-6)
    scale_torchao = amax / fp8_max
    scale_lumen = scale_torchao

    x_fp8_lumen = quant_fp8_tensorwise_impl(x, scale_lumen, fp8_dtype)
    x_fp8_torchao = _quantize_affine_float8(x, scale_torchao, fp8_dtype)

    torch.testing.assert_close(
        x_fp8_lumen.float(),
        x_fp8_torchao.float(),
        atol=1e-2,
        rtol=1e-2,
        msg="FP8 quant outputs should match",
    )

    x_deq_lumen = _dequantize_affine_float8(x_fp8_lumen, scale_torchao, torch.float32)
    x_deq_torchao = _dequantize_affine_float8(x_fp8_torchao, scale_torchao, torch.float32)
    snr = compute_snr(x.float(), x_deq_lumen)
    assert snr >= 8.0, f"SNR {snr:.1f} dB too low"
    torch.testing.assert_close(x_deq_lumen, x_deq_torchao, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("dtype_in", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_dequant_fp8_tensorwise_vs_torchao(shape, dtype_in, fp8_dtype):
    """Quantize with torchao, dequant with both Lumen and torchao, compare."""
    torch.manual_seed(42)
    x = torch.randn(*shape, device="cuda", dtype=dtype_in)
    fp8_max = torch.finfo(fp8_dtype).max
    amax = x.abs().max().float().clamp(min=1e-6)
    scale = amax / fp8_max

    x_fp8 = _quantize_affine_float8(x, scale, fp8_dtype)
    x_deq_lumen = dequant_fp8_tensorwise_impl(x_fp8, scale, dtype_in)
    x_deq_torchao = _dequantize_affine_float8(x_fp8, scale, dtype_in)

    torch.testing.assert_close(
        x_deq_lumen.float(),
        x_deq_torchao.float(),
        atol=1e-2,
        rtol=1e-2,
        msg="Dequant outputs should match",
    )
    snr = compute_snr(x.float(), x_deq_lumen.float())
    assert snr >= 8.0, f"SNR {snr:.1f} dB too low"


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_quant_fp8_tensorwise_zeros(shape, fp8_dtype):
    """Both implementations should map zeros to zero."""
    x = torch.zeros(*shape, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    x_fp8_lumen = quant_fp8_tensorwise_impl(x, scale, fp8_dtype)
    x_fp8_torchao = _quantize_affine_float8(x, scale, fp8_dtype)

    torch.testing.assert_close(x_fp8_lumen.float(), x_fp8_torchao.float())
    assert (x_fp8_lumen == 0).all()
    assert (x_fp8_torchao == 0).all()


# ---------------------------------------------------------------------------
# Blockwise FP8
# ---------------------------------------------------------------------------

BLOCK_SIZE = 128


def _blockwise_quant_ref(x, block_size, fp8_dtype, axis=1):
    """Pure PyTorch blockwise FP8 quantization reference (axis=1 only)."""
    M, N = x.shape
    fp8_max = torch.finfo(fp8_dtype).max
    x_f32 = x.float()
    x_blocked = x_f32.reshape(M, N // block_size, block_size)
    amax = x_blocked.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scales = amax / fp8_max  # (M, N//block_size, 1)
    x_scaled = (x_blocked / scales).clamp(-fp8_max, fp8_max)
    x_fp8 = x_scaled.reshape(M, N).to(fp8_dtype)
    scales = scales.squeeze(-1)  # (M, N//block_size)
    return x_fp8, scales


@pytest.mark.parametrize("shape", [(128, 256), (256, 512)], ids=["128x256", "256x512"])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_quant_fp8_blockwise_vs_torchao(shape, fp8_dtype):
    """Compare Lumen blockwise (axis=1) against torchao or PyTorch reference."""
    M, N = shape
    if N % BLOCK_SIZE != 0:
        pytest.skip(f"N={N} not divisible by block_size={BLOCK_SIZE}")

    torch.manual_seed(42)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    x_fp8_lumen, scales_lumen = quant_fp8_blockwise_impl(x, fp8_dtype, axis=1, block_size=BLOCK_SIZE)
    try:
        x_fp8_ref, scales_ref = fp8_blockwise_act_quant(x, BLOCK_SIZE, fp8_dtype)
    except (AssertionError, RuntimeError):
        x_fp8_ref, scales_ref = _blockwise_quant_ref(x, BLOCK_SIZE, fp8_dtype)

    x_deq_lumen = _dequantize_affine_float8(x_fp8_lumen, scales_lumen, torch.float32)
    x_deq_ref = _dequantize_affine_float8(x_fp8_ref, scales_ref, torch.float32)

    snr_lumen = compute_snr(x.float(), x_deq_lumen)
    snr_ref = compute_snr(x.float(), x_deq_ref)
    # e5m2 has lower precision (2 mantissa bits) → lower SNR expected
    snr_floor = 4.0 if fp8_dtype == torch.float8_e5m2 else 8.0
    assert snr_lumen >= snr_floor, f"Lumen SNR {snr_lumen:.1f} dB too low"
    assert snr_ref >= snr_floor, f"Reference SNR {snr_ref:.1f} dB too low"
    tol = 0.5 if fp8_dtype == torch.float8_e5m2 else 1e-1
    torch.testing.assert_close(x_deq_lumen, x_deq_ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("shape", [(256, 128), (512, 256)], ids=["256x128", "512x256"])
def test_quant_fp8_blockwise_axis0(shape):
    """Self-roundtrip for axis=0 (no torchao equivalent)."""
    M, N = shape
    if M % BLOCK_SIZE != 0:
        pytest.skip(f"M={M} not divisible by block_size={BLOCK_SIZE}")

    torch.manual_seed(42)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    x_fp8, scales = quant_fp8_blockwise_impl(x, torch.float8_e4m3fn, axis=0, block_size=BLOCK_SIZE)
    assert x_fp8.dtype == torch.float8_e4m3fn
    assert scales.shape == (triton.cdiv(M, BLOCK_SIZE), N)

    x_deq = _dequantize_affine_float8(x_fp8, scales, torch.float32)
    snr = compute_snr(x.float(), x_deq)
    assert snr >= 8.0, f"SNR {snr:.1f} dB too low"


# ---------------------------------------------------------------------------
# MXFP8
# ---------------------------------------------------------------------------

MX_BLOCK_SIZES = [32, 64]
MX_SHAPES = [(64, 128), (128, 256)]


@pytest.mark.parametrize("shape", MX_SHAPES, ids=[f"{m}x{n}" for m, n in MX_SHAPES])
@pytest.mark.parametrize("block_size", MX_BLOCK_SIZES)
def test_mxfp8_vs_torchao(shape, block_size):
    """Compare Lumen MXFP8 quant outputs against torchao, then cross-dequant."""
    M, N = shape
    if N % block_size != 0:
        pytest.skip(f"N={N} not divisible by block_size={block_size}")

    torch.manual_seed(42)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    data_lp_lumen, scales_lumen = convert_to_mxfp8(
        x.float(),
        block_size=block_size,
        axis=-1,
        float8_dtype_pt=torch.float8_e4m3fn,
    )

    # torchao quantization for reference (EVEN mode matches Lumen's round-even scaling)
    scale_ref, data_lp_ref = torchao_to_mx(
        x.float().cpu().contiguous(),
        torch.float8_e4m3fn,
        block_size,
        scaling_mode=ScaleCalculationMode.EVEN,
    )

    # Compare quantized FP8 data
    lumen_flat = data_lp_lumen.cpu().flatten().view(torch.float8_e4m3fn).view(torch.uint8)
    ref_flat = data_lp_ref.flatten().view(torch.uint8)
    min_len = min(lumen_flat.numel(), ref_flat.numel())
    fp8_match = (lumen_flat[:min_len] == ref_flat[:min_len]).float().mean().item()
    assert fp8_match >= 0.95, f"FP8 data match rate {fp8_match:.2%} < 95%"

    # Compare scales (torchao returns float8_e8m0fnu, Lumen returns uint8; bitwise reinterpret)
    s_lumen = scales_lumen.cpu().flatten()
    s_ref = scale_ref.flatten().view(torch.uint8)
    min_slen = min(s_lumen.numel(), s_ref.numel())
    scale_match = (s_lumen[:min_slen] == s_ref[:min_slen]).float().mean().item()
    assert scale_match >= 0.95, f"Scale match rate {scale_match:.2%} < 95%"

    # Cross-dequant: Lumen quant → torchao dequant
    _e8m0 = scale_ref.dtype  # float8_e8m0fnu
    x_deq_lumen_cpu = torchao_to_dtype(
        data_lp_lumen.cpu(),
        scales_lumen.cpu().view(_e8m0),
        torch.float8_e4m3fn,
        block_size,
        torch.float32,
    )

    snr = compute_snr(
        x.float().cpu(),
        x_deq_lumen_cpu,
    )
    assert snr >= 6.0, f"SNR {snr:.1f} dB too low"
    assert not torch.isnan(x_deq_lumen_cpu).any()
    assert not torch.isinf(x_deq_lumen_cpu).any()


@pytest.mark.parametrize("shape", MX_SHAPES, ids=[f"{m}x{n}" for m, n in MX_SHAPES])
@pytest.mark.parametrize("block_size", MX_BLOCK_SIZES)
def test_mxfp8_scale_and_data_agreement_with_torchao(shape, block_size):
    """Verify Lumen and torchao produce matching scales (>95%) and FP8 data (>90%)."""
    M, N = shape
    if N % block_size != 0:
        pytest.skip(f"N={N} not divisible by block_size={block_size}")

    torch.manual_seed(42)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    data_lp_lumen, scales_lumen = convert_to_mxfp8(
        x.float(),
        block_size=block_size,
        axis=-1,
        float8_dtype_pt=torch.float8_e4m3fn,
    )
    scale_ref, data_lp_ref = torchao_to_mx(
        x.float().cpu().contiguous(),
        torch.float8_e4m3fn,
        block_size,
        scaling_mode=ScaleCalculationMode.EVEN,
    )

    # Scales agreement (torchao returns float8_e8m0fnu; bitwise reinterpret to uint8)
    s_lumen = scales_lumen.cpu().flatten()
    s_ref = scale_ref.flatten().view(torch.uint8)
    min_slen = min(s_lumen.numel(), s_ref.numel())
    scale_match = (s_lumen[:min_slen] == s_ref[:min_slen]).float().mean().item()
    assert scale_match >= 0.95, f"Scale match rate {scale_match:.2%} < 95%"

    # FP8 data agreement
    d_lumen = data_lp_lumen.cpu().flatten().view(torch.float8_e4m3fn).view(torch.uint8)
    d_ref = data_lp_ref.flatten().view(torch.uint8)
    min_dlen = min(d_lumen.numel(), d_ref.numel())
    data_match = (d_lumen[:min_dlen] == d_ref[:min_dlen]).float().mean().item()
    assert data_match >= 0.90, f"FP8 data match rate {data_match:.2%} < 90%"


@pytest.mark.parametrize("shape", MX_SHAPES, ids=[f"{m}x{n}" for m, n in MX_SHAPES])
@pytest.mark.parametrize("block_size", MX_BLOCK_SIZES)
def test_mxfp8_vs_torchao_mxtensor(shape, block_size):
    """Compare Lumen quantized tensors AND roundtrip vs torchao MXTensor API."""
    M, N = shape
    if N % block_size != 0:
        pytest.skip(f"N={N} not divisible by block_size={block_size}")

    torch.manual_seed(42)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    data_lp_lumen, scales_lumen = convert_to_mxfp8(
        x.float(),
        block_size=block_size,
        axis=-1,
        float8_dtype_pt=torch.float8_e4m3fn,
    )

    mx_ref = MXTensor.to_mx(
        x.float().cpu().contiguous(),
        torch.float8_e4m3fn,
        block_size,
        scaling_mode=ScaleCalculationMode.EVEN,
    )

    # Compare quantized FP8 data directly
    data_lp_ref = mx_ref.qdata.flatten()
    data_lp_lumen_flat = data_lp_lumen.cpu().flatten().view(torch.float8_e4m3fn)
    min_len = min(data_lp_lumen_flat.numel(), data_lp_ref.numel())
    fp8_match = (
        (data_lp_lumen_flat[:min_len].view(torch.uint8) == data_lp_ref[:min_len].view(torch.uint8))
        .float()
        .mean()
        .item()
    )
    assert fp8_match >= 0.95, f"FP8 data match rate {fp8_match:.2%} < 95%"

    # Compare E8M0 scales directly (bitwise reinterpret float8_e8m0fnu → uint8)
    scales_ref = mx_ref.scale.flatten().view(torch.uint8)
    scales_lumen_flat = scales_lumen.cpu().flatten()
    min_slen = min(scales_lumen_flat.numel(), scales_ref.numel())
    scale_match = (scales_lumen_flat[:min_slen] == scales_ref[:min_slen]).float().mean().item()
    assert scale_match >= 0.95, f"Scale match rate {scale_match:.2%} < 95%"

    # Compare dequantized results
    x_deq_lumen = convert_from_mxfp8(
        data_lp_lumen,
        scales_lumen,
        output_dtype=torch.float32,
        block_size=block_size,
        axis=-1,
    )
    x_deq_torchao = mx_ref.dequantize()

    snr = compute_snr(x.float().cpu(), x_deq_lumen.cpu())
    assert snr >= 6.0, f"SNR {snr:.1f} dB too low"
    torch.testing.assert_close(
        x_deq_lumen.cpu(),
        x_deq_torchao.cpu(),
        atol=1e-1,
        rtol=1e-1,
    )


def test_mxfp8_zeros():
    """Both implementations should handle zeros."""
    M, N = 64, 128
    block_size = 64
    x = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    data_lp, scales = convert_to_mxfp8(x.float(), block_size=block_size, axis=-1)
    x_deq = convert_from_mxfp8(data_lp, scales, block_size=block_size, axis=-1)
    torch.testing.assert_close(x_deq, x.float())

    scale_ref, data_ref = torchao_to_mx(
        x.float().cpu(),
        torch.float8_e4m3fn,
        block_size,
        scaling_mode=ScaleCalculationMode.EVEN,
    )
    x_deq_ref = torchao_to_dtype(
        data_ref,
        scale_ref,
        torch.float8_e4m3fn,
        block_size,
        torch.float32,
    )
    torch.testing.assert_close(x_deq_ref, x.float().cpu())


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_mxfp8_dtype_variants(fp8_dtype, shape=(64, 128), block_size=64):
    """Test MXFP8 with different FP8 element dtypes, compared against torchao."""
    M, N = shape
    if N % block_size != 0:
        pytest.skip(f"N={N} not divisible by block_size={block_size}")

    torch.manual_seed(42)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    # Lumen
    data_lp, scales = convert_to_mxfp8(
        x.float(),
        block_size=block_size,
        axis=-1,
        float8_dtype_pt=fp8_dtype,
    )

    # torchao (EVEN mode matches Lumen's round-even scaling)
    scale_ref, data_lp_ref = torchao_to_mx(
        x.float().cpu().contiguous(),
        fp8_dtype,
        block_size,
        scaling_mode=ScaleCalculationMode.EVEN,
    )

    # Compare FP8 data
    d_lumen = data_lp.cpu().flatten().view(fp8_dtype).view(torch.uint8)
    d_ref = data_lp_ref.flatten().view(torch.uint8)
    min_dlen = min(d_lumen.numel(), d_ref.numel())
    data_match = (d_lumen[:min_dlen] == d_ref[:min_dlen]).float().mean().item()
    assert data_match >= 0.95, f"FP8 data match rate {data_match:.2%} < 95%"

    # Compare scales (torchao returns float8_e8m0fnu; bitwise reinterpret to uint8)
    s_lumen = scales.cpu().flatten()
    s_ref = scale_ref.flatten().view(torch.uint8)
    min_slen = min(s_lumen.numel(), s_ref.numel())
    scale_match = (s_lumen[:min_slen] == s_ref[:min_slen]).float().mean().item()
    assert scale_match >= 0.95, f"Scale match rate {scale_match:.2%} < 95%"

    # Compare dequantized results
    x_deq = convert_from_mxfp8(data_lp, scales, block_size=block_size, axis=-1)
    x_deq_ref = torchao_to_dtype(
        data_lp_ref,
        scale_ref,
        fp8_dtype,
        block_size,
        torch.float32,
    )

    assert not torch.isnan(x_deq).any()
    assert not torch.isinf(x_deq).any()
    snr = compute_snr(x.float().cpu(), x_deq.cpu())
    assert snr >= 6.0, f"Lumen roundtrip SNR {snr:.1f} dB too low"
    torch.testing.assert_close(
        x_deq.cpu(),
        x_deq_ref.cpu(),
        atol=1e-1,
        rtol=1e-1,
    )
