###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Tests for lumen.ops.quantize.linear: FP8 quantized linear / GEMM.

Covers:
  - FP8 quantized GEMM forward — compare dequantized output vs torch.mm reference
  - FP8 quantized linear forward + backward
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


# ===================================================================
# FP8 quantized linear forward
# ===================================================================


@pytest.mark.parametrize("config", LINEAR_SHAPES, ids=LINEAR_IDS)
def test_fp8_linear_fwd(config):
    """FP8 quantized GEMM forward: compare dequantized output vs torch.mm reference."""
    dtype = torch.bfloat16
    M, K, N = config.M, config.K, config.N

    x = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=dtype) * 0.02

    # Reference: torch.mm (Y = X @ W^T)
    out_ref = x @ w.T

    # Lumen FP8 quantized linear (dynamic scaling)
    out_lumen = linear_ops.quantized_linear(x, w, scaling_type="dynamic", quantize_activation=True)

    snr = compute_snr(out_ref, out_lumen)
    assert snr > 15, f"FP8 linear fwd SNR: {snr:.1f} dB (expected > 15)"


# ===================================================================
# FP8 quantized linear forward + backward
# ===================================================================


@pytest.mark.parametrize("config", LINEAR_SHAPES, ids=LINEAR_IDS)
def test_fp8_linear_fwd_bwd(config):
    """Forward + backward through quantized linear."""
    dtype = torch.bfloat16
    M, K, N = config.M, config.K, config.N

    x = torch.randn(M, K, device="cuda", dtype=dtype, requires_grad=True) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=dtype, requires_grad=True) * 0.02

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)

    # Reference: BF16 linear
    out_ref = x_ref @ w_ref.T
    loss_ref = out_ref.float().mean()
    loss_ref.backward()

    # Lumen FP8 quantized linear
    out_lumen = linear_ops.quantized_linear(x, w, scaling_type="dynamic", quantize_activation=True)
    loss_lumen = out_lumen.float().mean()
    loss_lumen.backward()

    out_snr = compute_snr(out_ref, out_lumen)
    dx_snr = compute_snr(x_ref.grad, x.grad)
    dw_snr = compute_snr(w_ref.grad, w.grad)

    assert out_snr > 12, f"FP8 linear fwd SNR: {out_snr:.1f} dB"
    assert dx_snr > 8, f"FP8 linear bwd dx SNR: {dx_snr:.1f} dB"
    assert dw_snr > 8, f"FP8 linear bwd dw SNR: {dw_snr:.1f} dB"
