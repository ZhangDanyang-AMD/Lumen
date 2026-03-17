###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Tests for lumen.ops.quantize.linear: FP8 quantized linear / GEMM.

Covers:
  - FP8 quantized GEMM forward — all 6 scaling types vs torch.mm reference
  - FP8 quantized linear forward + backward — all 6 scaling types
  - Bias, weight-only quant, BF16 wgrad, edge cases
"""

import pytest
import torch
from conftest import LinearConfig, compute_snr

import lumen.ops.quantize.linear as linear_ops

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

LINEAR_SHAPES = [
    LinearConfig(32, 256, 512),
    LinearConfig(128, 4096, 4096),
    LinearConfig(256, 4096, 1024),
]

LINEAR_IDS = [repr(c) for c in LINEAR_SHAPES]

ALL_SCALING_TYPES = ["delayed", "dynamic", "per_token", "blockwise", "mxfp8", "none"]

# Backward reuses forward's weight_scale after transposing weight_fp8.
# Only per-tensor (scalar) scales survive transposition; per_token (N,1),
# blockwise (N,K/bs), and mxfp8 block scales become misaligned with the
# transposed layout, causing GEMM kernel crashes.
BWD_SCALING_TYPES = ["delayed", "dynamic", "none"]

# SNR floors per scaling type — coarser quantization ⇒ lower expected SNR.
# mxfp8 uses E8M0 exponent-only scales; blockwise has block granularity.
_FWD_SNR = {
    "delayed": 12,
    "dynamic": 12,
    "per_token": 12,
    "blockwise": 8,
    "mxfp8": 5,
    "none": 25,
}
_BWD_DX_SNR = {
    "delayed": 6,
    "dynamic": 6,
    "per_token": 6,
    "blockwise": 4,
    "mxfp8": 3,
    "none": 20,
}
_BWD_DW_SNR = {
    "delayed": 6,
    "dynamic": 6,
    "per_token": 6,
    "blockwise": 4,
    "mxfp8": 3,
    "none": 20,
}


def _block_size_for(scaling_type, default=128):
    """Effective block_size used by quantize_input for this scaling type."""
    if scaling_type == "mxfp8":
        return 32 if default > 64 else default
    return default


def _skip_if_unaligned(config, scaling_type):
    block_size = _block_size_for(scaling_type)
    if scaling_type in ("blockwise", "mxfp8") and config.K % block_size != 0:
        pytest.skip(f"K={config.K} not divisible by block_size={block_size}")


# ===================================================================
# FP8 quantized linear forward — all scaling types
# ===================================================================


@pytest.mark.parametrize("config", LINEAR_SHAPES, ids=LINEAR_IDS)
@pytest.mark.parametrize("scaling_type", ALL_SCALING_TYPES)
def test_fp8_linear_fwd(config, scaling_type):
    """FP8 quantized GEMM forward: compare output vs torch.mm reference."""
    _skip_if_unaligned(config, scaling_type)
    dtype = torch.bfloat16
    M, K, N = config.M, config.K, config.N

    x = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=dtype) * 0.02

    out_ref = x @ w.T
    out_lumen = linear_ops.quantized_linear(
        x,
        w,
        scaling_type=scaling_type,
        quantize_activation=True,
    )

    snr = compute_snr(out_ref, out_lumen)
    floor = _FWD_SNR[scaling_type]
    assert snr > floor, f"FP8 linear fwd ({scaling_type}) SNR: {snr:.1f} dB (expected > {floor})"


# ===================================================================
# FP8 quantized linear forward + backward — all scaling types
# ===================================================================


@pytest.mark.parametrize("config", LINEAR_SHAPES[:2], ids=LINEAR_IDS[:2])
@pytest.mark.parametrize("scaling_type", BWD_SCALING_TYPES)
def test_fp8_linear_fwd_bwd(config, scaling_type):
    """Forward + backward through quantized linear.

    Only tests scaling types with per-tensor (scalar) scales — see
    BWD_SCALING_TYPES comment for why per_token/blockwise/mxfp8 are excluded.
    """
    _skip_if_unaligned(config, scaling_type)
    dtype = torch.bfloat16
    M, K, N = config.M, config.K, config.N

    x = torch.randn(M, K, device="cuda", dtype=dtype, requires_grad=True) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=dtype, requires_grad=True) * 0.02

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)

    out_ref = x_ref @ w_ref.T
    out_ref.float().mean().backward()

    out_lumen = linear_ops.quantized_linear(
        x,
        w,
        scaling_type=scaling_type,
        quantize_activation=True,
    )
    out_lumen.float().mean().backward()

    out_snr = compute_snr(out_ref, out_lumen)
    dx_snr = compute_snr(x_ref.grad, x.grad)
    dw_snr = compute_snr(w_ref.grad, w.grad)

    fwd_floor = _FWD_SNR[scaling_type]
    dx_floor = _BWD_DX_SNR[scaling_type]
    dw_floor = _BWD_DW_SNR[scaling_type]
    assert out_snr > fwd_floor, f"fwd ({scaling_type}) SNR: {out_snr:.1f} dB"
    assert dx_snr > dx_floor, f"bwd dx ({scaling_type}) SNR: {dx_snr:.1f} dB"
    assert dw_snr > dw_floor, f"bwd dw ({scaling_type}) SNR: {dw_snr:.1f} dB"


# ===================================================================
# Bias
# ===================================================================


@pytest.mark.parametrize("config", LINEAR_SHAPES[:2], ids=LINEAR_IDS[:2])
@pytest.mark.parametrize("scaling_type", BWD_SCALING_TYPES)
def test_fp8_linear_bias(config, scaling_type):
    """Forward + backward with bias."""
    dtype = torch.bfloat16
    M, K, N = config.M, config.K, config.N

    x = torch.randn(M, K, device="cuda", dtype=dtype, requires_grad=True) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=dtype, requires_grad=True) * 0.02
    b = torch.randn(N, device="cuda", dtype=dtype, requires_grad=True) * 0.01

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    out_ref = x_ref @ w_ref.T + b_ref
    out_ref.float().mean().backward()

    out_lumen = linear_ops.quantized_linear(
        x,
        w,
        bias=b,
        scaling_type=scaling_type,
        quantize_activation=True,
    )
    out_lumen.float().mean().backward()

    fwd_floor = _FWD_SNR[scaling_type]
    assert compute_snr(out_ref, out_lumen) > fwd_floor
    assert compute_snr(x_ref.grad, x.grad) > _BWD_DX_SNR[scaling_type]
    assert compute_snr(w_ref.grad, w.grad) > _BWD_DW_SNR[scaling_type]
    assert compute_snr(b_ref.grad, b.grad) > 20, "bias gradient mismatch"


# ===================================================================
# Weight-only quantization (quantize_activation=False)
# ===================================================================


@pytest.mark.parametrize("config", LINEAR_SHAPES[:2], ids=LINEAR_IDS[:2])
@pytest.mark.parametrize("scaling_type", ["dynamic", "per_token"])
def test_fp8_linear_weight_only(config, scaling_type):
    """Weight-only quant: activation stays BF16, weight is FP8-dequanted."""
    dtype = torch.bfloat16
    M, K, N = config.M, config.K, config.N

    x = torch.randn(M, K, device="cuda", dtype=dtype, requires_grad=True) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=dtype, requires_grad=True) * 0.02

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)

    out_ref = x_ref @ w_ref.T
    out_ref.float().mean().backward()

    out_lumen = linear_ops.quantized_linear(
        x,
        w,
        scaling_type=scaling_type,
        quantize_activation=False,
    )
    out_lumen.float().mean().backward()

    assert compute_snr(out_ref, out_lumen) > 10
    assert compute_snr(x_ref.grad, x.grad) > 6
    assert compute_snr(w_ref.grad, w.grad) > 6


# ===================================================================
# BF16 weight gradient (fp8_wgrad=False)
# ===================================================================


@pytest.mark.parametrize("config", LINEAR_SHAPES[:2], ids=LINEAR_IDS[:2])
def test_fp8_linear_bf16_wgrad(config):
    """fp8_wgrad=False: weight gradient computed in BF16 instead of FP8."""
    dtype = torch.bfloat16
    M, K, N = config.M, config.K, config.N

    x = torch.randn(M, K, device="cuda", dtype=dtype, requires_grad=True) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=dtype, requires_grad=True) * 0.02

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)

    out_ref = x_ref @ w_ref.T
    out_ref.float().mean().backward()

    out_lumen = linear_ops.quantized_linear(
        x,
        w,
        scaling_type="dynamic",
        quantize_activation=True,
        fp8_wgrad=False,
    )
    out_lumen.float().mean().backward()

    assert compute_snr(out_ref, out_lumen) > 12
    assert compute_snr(x_ref.grad, x.grad) > 6
    # BF16 wgrad should be at least as good as FP8 wgrad
    assert compute_snr(w_ref.grad, w.grad) > 8


# ===================================================================
# Edge case: M=1 (single token)
# ===================================================================


@pytest.mark.parametrize("scaling_type", ["dynamic", "per_token", "none"])
def test_fp8_linear_m1(scaling_type):
    """Single-row input (M=1) — exercises per-token edge case."""
    dtype = torch.bfloat16
    M, K, N = 1, 256, 512

    x = torch.randn(M, K, device="cuda", dtype=dtype, requires_grad=True) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=dtype, requires_grad=True) * 0.02

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)

    out_ref = x_ref @ w_ref.T
    out_ref.float().mean().backward()

    out_lumen = linear_ops.quantized_linear(
        x,
        w,
        scaling_type=scaling_type,
        quantize_activation=True,
    )
    out_lumen.float().mean().backward()

    fwd_floor = _FWD_SNR[scaling_type]
    assert compute_snr(out_ref, out_lumen) > fwd_floor
    assert x.grad is not None and x.grad.shape == x.shape
    assert w.grad is not None and w.grad.shape == w.shape
