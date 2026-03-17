###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Tests for lumen.ops.attention: multi-head attention (BSHD layout).

Covers:
  - Forward vs PyTorch reference (BF16, compute_snr)
  - Forward + backward: dQ/dK/dV gradients vs reference
  - Causal attention forward
  - GQA (num_head_q > num_head_kv) forward

Reference: attention_ref from conftest (pure PyTorch BSHD).
"""

import pytest
import torch
from conftest import AttnConfig, attention_ref, compute_snr

import lumen.ops.attention as attn_ops

# ---------------------------------------------------------------------------
# Configurations (BSHD layout)
# ---------------------------------------------------------------------------

ATTN_CONFIGS = [
    AttnConfig(128, 128, 8, 8, 64, 64),  # standard
    AttnConfig(256, 256, 16, 4, 64, 64),  # GQA
    AttnConfig(512, 512, 8, 8, 128, 128),  # larger head dim
]

ATTN_IDS = [repr(c) for c in ATTN_CONFIGS]


def _make_tensors(config: AttnConfig, batch_size: int, dtype, device, requires_grad=False):
    """Create q, k, v in BSHD layout."""
    q = (
        torch.randn(
            batch_size,
            config.seqlen_q,
            config.num_head_q,
            config.head_dim_qk,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        * 0.02
    )
    k = (
        torch.randn(
            batch_size,
            config.seqlen_kv,
            config.num_head_kv,
            config.head_dim_qk,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        * 0.02
    )
    v = (
        torch.randn(
            batch_size,
            config.seqlen_kv,
            config.num_head_kv,
            config.head_dim_v,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        * 0.02
    )
    return q, k, v


# ===================================================================
# Forward vs reference
# ===================================================================


@pytest.mark.parametrize("config", ATTN_CONFIGS, ids=ATTN_IDS)
def test_attention_fwd(config):
    """BF16 forward vs PyTorch reference (attention_ref), using compute_snr."""
    dtype = torch.bfloat16
    device = "cuda"
    batch_size = 2
    sm_scale = config.head_dim_qk ** (-0.5)

    q, k, v = _make_tensors(config, batch_size, dtype, device)

    out_ref = attention_ref(q, k, v, sm_scale, causal=False)

    out = attn_ops.attention(
        q,
        k,
        v,
        softmax_scale=sm_scale,
        causal=False,
        backend_type="aiter_triton",
    )

    snr = compute_snr(out_ref, out)
    assert snr > 20, f"Attention forward SNR: {snr:.1f} dB"


# ===================================================================
# Forward + backward: dQ/dK/dV vs reference
# ===================================================================


@pytest.mark.parametrize("config", ATTN_CONFIGS, ids=ATTN_IDS)
def test_attention_fwd_bwd(config):
    """BF16 forward+backward, compare dQ/dK/dV gradients vs reference."""
    dtype = torch.bfloat16
    device = "cuda"
    batch_size = 2
    sm_scale = config.head_dim_qk ** (-0.5)

    q, k, v = _make_tensors(config, batch_size, dtype, device, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    # Reference forward + backward
    out_ref = attention_ref(q_ref, k_ref, v_ref, sm_scale, causal=False)
    loss_ref = out_ref.float().mean()
    loss_ref.backward()

    # Lumen forward + backward
    out = attn_ops.attention(
        q,
        k,
        v,
        softmax_scale=sm_scale,
        causal=False,
        backend_type="aiter_triton",
    )
    loss = out.float().mean()
    loss.backward()

    dq_snr = compute_snr(q_ref.grad, q.grad)
    dk_snr = compute_snr(k_ref.grad, k.grad)
    dv_snr = compute_snr(v_ref.grad, v.grad)

    min_snr = 15
    assert dq_snr > min_snr, f"dQ gradient SNR: {dq_snr:.1f} dB"
    assert dk_snr > min_snr, f"dK gradient SNR: {dk_snr:.1f} dB"
    assert dv_snr > min_snr, f"dV gradient SNR: {dv_snr:.1f} dB"


# ===================================================================
# Causal attention forward
# ===================================================================


@pytest.mark.parametrize("config", ATTN_CONFIGS, ids=ATTN_IDS)
def test_attention_causal(config):
    """Causal attention forward."""
    dtype = torch.bfloat16
    device = "cuda"
    batch_size = 2
    sm_scale = config.head_dim_qk ** (-0.5)

    q, k, v = _make_tensors(config, batch_size, dtype, device)

    out_ref = attention_ref(q, k, v, sm_scale, causal=True)

    out = attn_ops.attention(
        q,
        k,
        v,
        softmax_scale=sm_scale,
        causal=True,
        backend_type="aiter_triton",
    )

    snr = compute_snr(out_ref, out)
    assert snr > 20, f"Causal attention forward SNR: {snr:.1f} dB"


# ===================================================================
# GQA (num_head_q > num_head_kv) forward
# ===================================================================


@pytest.mark.parametrize(
    "config",
    [
        AttnConfig(256, 256, 16, 4, 64, 64),  # GQA: 16 query heads, 4 KV heads
        AttnConfig(128, 128, 8, 2, 64, 64),  # GQA: 8 query heads, 2 KV heads
    ],
    ids=["sq256_sk256_hq16_hkv4_dqk64_dv64", "sq128_sk128_hq8_hkv2_dqk64_dv64"],
)
def test_attention_gqa(config):
    """GQA (num_head_q > num_head_kv) forward."""
    dtype = torch.bfloat16
    device = "cuda"
    batch_size = 2
    sm_scale = config.head_dim_qk ** (-0.5)

    assert config.num_head_q > config.num_head_kv, "Config must be GQA (num_head_q > num_head_kv)"

    q, k, v = _make_tensors(config, batch_size, dtype, device)

    out_ref = attention_ref(q, k, v, sm_scale, causal=False)

    out = attn_ops.attention(
        q,
        k,
        v,
        softmax_scale=sm_scale,
        causal=False,
        backend_type="aiter_triton",
    )

    snr = compute_snr(out_ref, out)
    assert snr > 20, f"GQA attention forward SNR: {snr:.1f} dB"
