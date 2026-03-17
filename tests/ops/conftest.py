###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from dataclasses import dataclass

import pytest
import torch

DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def attention_ref(q, k, v, sm_scale, causal=False):
    """Pure PyTorch multi-head attention reference (BSHD layout)."""
    B, SQ, HQ, D = q.shape
    _, SK, HK, _ = k.shape
    DV = v.shape[-1]
    gqa_ratio = HQ // HK

    q_f32 = q.float()
    k_f32 = k.float()
    v_f32 = v.float()

    if gqa_ratio > 1:
        k_f32 = k_f32.unsqueeze(3).expand(B, SK, HK, gqa_ratio, D).reshape(B, SK, HQ, D)
        v_f32 = v_f32.unsqueeze(3).expand(B, SK, HK, gqa_ratio, DV).reshape(B, SK, HQ, DV)

    q_t = q_f32.transpose(1, 2)
    k_t = k_f32.transpose(1, 2)
    v_t = v_f32.transpose(1, 2)

    attn = torch.matmul(q_t, k_t.transpose(-2, -1)) * sm_scale

    if causal:
        row_idx = torch.arange(SQ, device=q.device).unsqueeze(1)
        col_idx = torch.arange(SK, device=q.device).unsqueeze(0)
        col_offset = SQ - SK
        mask = row_idx >= (col_offset + col_idx)
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v_t)
    out = out.transpose(1, 2)
    return out.to(q.dtype)


def rmsnorm_ref(x, weight, eps=1e-6):
    """Pure PyTorch RMSNorm reference."""
    x_f32 = x.float()
    rms = torch.sqrt(torch.mean(x_f32**2, dim=-1, keepdim=True) + eps)
    return (x_f32 / rms * weight.float()).to(x.dtype)


def layernorm_ref(x, weight, bias=None, eps=1e-5):
    """PyTorch LayerNorm reference."""
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def grouped_gemm_ref(lhs, rhs, group_sizes, bias=None):
    """Sequential per-expert GEMM reference for MoE."""
    outputs = []
    offset = 0
    for g, size in enumerate(group_sizes.tolist()):
        size = int(size)
        if size == 0:
            continue
        x_g = lhs[offset : offset + size]
        w_g = rhs[g]
        out_g = x_g @ w_g.T
        if bias is not None:
            out_g = out_g + bias[g]
        outputs.append(out_g)
        offset += size
    return torch.cat(outputs, dim=0) if outputs else torch.empty(0, rhs.shape[1], device=lhs.device, dtype=lhs.dtype)


def cross_entropy_ref(logits, target, label_smoothing=0.0, ignore_idx=-100):
    """PyTorch cross-entropy reference."""
    V = logits.shape[-1]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, V),
        target.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
        ignore_index=ignore_idx,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_snr(x, y):
    """Compute Signal-to-Noise Ratio in dB."""
    signal = torch.norm(x.float()).pow(2)
    if signal < 1e-12:
        return float("inf") if torch.allclose(x.float(), y.float(), atol=1e-7) else 0.0
    noise = torch.norm(x.float() - y.float()).pow(2)
    return 10.0 * torch.log10(signal / (noise + 1e-12)).item()


def check_close(a, b, atol=1e-2, rtol=1e-2, tol_err_ratio=0.05, msg=""):
    """Check that most elements are close, allowing a small fraction of outliers."""
    is_close = torch.isclose(a.float(), b.float(), rtol=rtol, atol=atol)
    err_ratio = 1.0 - is_close.float().mean().item()
    assert err_ratio <= tol_err_ratio, (
        f"{msg}: {err_ratio:.2%} elements exceed tolerance "
        f"(atol={atol}, rtol={rtol}, max_allowed={tol_err_ratio:.0%})"
    )


# ---------------------------------------------------------------------------
# Data-class configs
# ---------------------------------------------------------------------------


@dataclass
class AttnConfig:
    seqlen_q: int
    seqlen_kv: int
    num_head_q: int
    num_head_kv: int
    head_dim_qk: int
    head_dim_v: int

    def __repr__(self):
        return (
            f"sq{self.seqlen_q}_sk{self.seqlen_kv}_"
            f"hq{self.num_head_q}_hkv{self.num_head_kv}_"
            f"dqk{self.head_dim_qk}_dv{self.head_dim_v}"
        )


@dataclass
class NormConfig:
    M: int
    N: int

    def __repr__(self):
        return f"M{self.M}_N{self.N}"


@dataclass
class LinearConfig:
    M: int
    K: int
    N: int

    def __repr__(self):
        return f"M{self.M}_K{self.K}_N{self.N}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def seed_rng():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)


@pytest.fixture
def device():
    return DEVICE
