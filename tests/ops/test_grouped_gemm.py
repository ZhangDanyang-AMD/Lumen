###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Tests for lumen.ops.gemm.grouped_gemm: grouped GEMM for MoE.

Covers:
  - BF16 grouped GEMM forward — compare against grouped_gemm_ref
  - Different expert counts and token distributions
"""

import pytest
import torch
from conftest import compute_snr, grouped_gemm_ref

import lumen.ops.gemm.grouped_gemm as gemm_ops


def _make_group_sizes(num_experts, tokens, device):
    """Create group_sizes that sum to tokens, distributed across experts."""
    base = tokens // num_experts
    remainder = tokens % num_experts
    sizes = [base + (1 if i < remainder else 0) for i in range(num_experts)]
    return torch.tensor(sizes, dtype=torch.int32, device=device)


# ---------------------------------------------------------------------------
# Parametrize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_experts", [4, 8])
@pytest.mark.parametrize("tokens", [64, 256])
@pytest.mark.parametrize("hidden", [256, 512])
def test_grouped_gemm_fwd(num_experts, tokens, hidden):
    """Compare grouped GEMM against grouped_gemm_ref from conftest."""
    dtype = torch.bfloat16
    N = hidden  # output dim per expert
    K = hidden  # input dim

    device = "cuda"
    group_sizes = _make_group_sizes(num_experts, tokens, device)
    total_tokens = int(group_sizes.sum().item())

    lhs = torch.randn(total_tokens, K, device=device, dtype=dtype) * 0.1
    # grouped_gemm_ref expects rhs (G, N, K); AITER gmm expects rhs (G, K, N)
    rhs_ref = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.02

    out_ref = grouped_gemm_ref(lhs, rhs_ref, group_sizes)

    # Lumen grouped_gemm uses AITER gmm: rhs (G, K, N)
    rhs_lumen = rhs_ref.transpose(1, 2)  # (G, N, K) -> (G, K, N)
    out_lumen = gemm_ops.grouped_gemm(lhs, rhs_lumen, group_sizes, scaling_type="none")

    snr = compute_snr(out_ref, out_lumen)
    assert snr > 25, f"Grouped GEMM fwd SNR: {snr:.1f} dB (expected > 25)"


@pytest.mark.parametrize("num_experts", [4, 8])
@pytest.mark.parametrize("tokens", [64, 256])
@pytest.mark.parametrize("hidden", [256, 512])
def test_grouped_gemm_shapes(num_experts, tokens, hidden):
    """Test grouped GEMM with different expert counts and token distributions."""
    dtype = torch.bfloat16
    N = hidden
    K = hidden

    device = "cuda"
    group_sizes = _make_group_sizes(num_experts, tokens, device)
    total_tokens = int(group_sizes.sum().item())

    lhs = torch.randn(total_tokens, K, device=device, dtype=dtype) * 0.1
    rhs_ref = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.02
    rhs_lumen = rhs_ref.transpose(1, 2)

    out_ref = grouped_gemm_ref(lhs, rhs_ref, group_sizes)
    out_lumen = gemm_ops.grouped_gemm(lhs, rhs_lumen, group_sizes, scaling_type="none")

    assert out_lumen.shape == out_ref.shape
    assert out_lumen.shape == (total_tokens, N)

    snr = compute_snr(out_ref, out_lumen)
    assert snr > 25, f"Grouped GEMM shapes SNR: {snr:.1f} dB"
