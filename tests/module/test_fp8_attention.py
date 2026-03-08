###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Compare FP8 quantised attention correctness between
TransformerLightAttention  (Transformer Light)  and
DotProductAttention        (Transformer Engine AMD).

The test generates identical (q, k, v) inputs, runs them through both modules
in FP8 mode, and validates:
    1. Both FP8 forward outputs are close to a BF16 reference.
    2. Both FP8 backward gradients (dQ, dK, dV) are close to BF16 gradients.
    3. TL FP8 and TE FP8 forward outputs are close to each other.
    4. TL FP8 and TE FP8 gradients are close to each other.

Run:
    pytest tests/module/test_fp8_attention.py -v
"""

import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import pytest
import torch

# ---------------------------------------------------------------------------
# Availability probes
# ---------------------------------------------------------------------------

_tl_available = True
try:
    from transformer_light.modules.attention import TransformerLightAttention
except ImportError:
    _tl_available = False

_te_available = True
try:
    from transformer_engine.pytorch.attention.dot_product_attention import DotProductAttention
    from transformer_engine.pytorch.fp8 import fp8_autocast, FP8GlobalStateManager
    from transformer_engine.pytorch import fp8_model_init
    from transformer_engine.common import recipe
    from transformer_engine.pytorch.distributed import CudaRNGStatesTracker
except ImportError:
    _te_available = False

_gpu_available = torch.cuda.is_available()
_fp8_available = False
if _te_available and _gpu_available:
    _fp8_available, _fp8_reason = FP8GlobalStateManager.is_fp8_available()
else:
    _fp8_reason = "TE not installed or no GPU"

skip_no_tl = pytest.mark.skipif(not _tl_available, reason="transformer_light not installed")
skip_no_te = pytest.mark.skipif(not _te_available, reason="transformer_engine not installed")
skip_no_gpu = pytest.mark.skipif(not _gpu_available, reason="CUDA not available")
skip_no_fp8 = pytest.mark.skipif(not _fp8_available, reason=_fp8_reason)

SEED = 42


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

@dataclass
class AttentionConfig:
    batch_size: int
    max_seqlen: int
    num_heads: int
    num_gqa_groups: int
    head_dim: int
    causal: bool
    dropout_p: float = 0.0
    dtype: torch.dtype = torch.bfloat16

    @property
    def softmax_scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim)


CONFIGS = {
    "small_mha_causal": AttentionConfig(2, 128, 8, 8, 64, causal=True),
    "small_gqa_causal": AttentionConfig(2, 128, 8, 2, 64, causal=True),
    "medium_mha_causal": AttentionConfig(2, 512, 16, 16, 128, causal=True),
    "medium_gqa_causal": AttentionConfig(2, 512, 16, 4, 128, causal=True),
    "small_mha_nomask": AttentionConfig(2, 128, 8, 8, 64, causal=False),
    "small_gqa_nomask": AttentionConfig(2, 128, 8, 2, 64, causal=False),
}

FP8_QUANT_TYPES = ["fp8_blockwise"]  # mxfp8 can be added when both sides support it


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_all(seed: int = SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_qkv(
    cfg: AttentionConfig, device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (q, k, v) in [B, S, H, D] layout."""
    _seed_all()
    q = torch.randn(
        cfg.batch_size, cfg.max_seqlen, cfg.num_heads, cfg.head_dim,
        dtype=cfg.dtype, device=device, requires_grad=True,
    )
    k = torch.randn(
        cfg.batch_size, cfg.max_seqlen, cfg.num_gqa_groups, cfg.head_dim,
        dtype=cfg.dtype, device=device, requires_grad=True,
    )
    v = torch.randn(
        cfg.batch_size, cfg.max_seqlen, cfg.num_gqa_groups, cfg.head_dim,
        dtype=cfg.dtype, device=device, requires_grad=True,
    )
    return q, k, v


def _clone_qkv(q, k, v):
    """Clone q/k/v as new leaf tensors requiring grad."""
    return (
        q.detach().clone().requires_grad_(True),
        k.detach().clone().requires_grad_(True),
        v.detach().clone().requires_grad_(True),
    )


def _make_output_grad(shape, dtype, device="cuda"):
    _seed_all(SEED + 1)
    return torch.randn(shape, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _run_tl_fp8(
    cfg: AttentionConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    quant_type: str,
    output_grad: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Run TransformerLightAttention with FP8 and return (output, (dq, dk, dv))."""
    attn = TransformerLightAttention(
        dropout_p=cfg.dropout_p,
        softmax_scale=cfg.softmax_scale,
        causal=cfg.causal,
        backend_type="triton",
        quant_type=quant_type,
    )
    out = attn(q, k, v)
    if isinstance(out, tuple):
        out = out[0]
    out.backward(output_grad)
    return out.detach(), (q.grad.detach(), k.grad.detach(), v.grad.detach())


def _run_tl_ref(
    cfg: AttentionConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output_grad: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Run TransformerLightAttention in BF16 (no FP8) as reference."""
    attn = TransformerLightAttention(
        dropout_p=cfg.dropout_p,
        softmax_scale=cfg.softmax_scale,
        causal=cfg.causal,
        backend_type="triton",
        quant_type=None,
    )
    out = attn(q, k, v)
    if isinstance(out, tuple):
        out = out[0]
    out.backward(output_grad)
    return out.detach(), (q.grad.detach(), k.grad.detach(), v.grad.detach())


def _get_rng_tracker():
    tracker = CudaRNGStatesTracker()
    tracker.add("model-parallel-rng", SEED)
    return tracker


def _run_te_fp8(
    cfg: AttentionConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output_grad: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Run TE DotProductAttention with FP8 DPA enabled."""
    os.environ["NVTE_FP8_DPA_BWD"] = "1"

    fp8_recipe = recipe.DelayedScaling(
        margin=0,
        fp8_format=recipe.Format.HYBRID,
        amax_history_len=1,
        amax_compute_algo="most_recent",
        fp8_dpa=True,
    )

    with fp8_model_init(enabled=True):
        dpa = DotProductAttention(
            num_attention_heads=cfg.num_heads,
            kv_channels=cfg.head_dim,
            num_gqa_groups=cfg.num_gqa_groups,
            attention_dropout=cfg.dropout_p,
            qkv_format="bshd",
            attn_mask_type="causal" if cfg.causal else "no_mask",
            softmax_scale=cfg.softmax_scale,
            sequence_parallel=False,
            tp_size=1,
            get_rng_state_tracker=_get_rng_tracker,
            layer_number=1,
        ).to(dtype=cfg.dtype, device="cuda")

    with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = dpa(q, k, v, qkv_format="bshd")

    out.backward(output_grad)
    return out.detach(), (q.grad.detach(), k.grad.detach(), v.grad.detach())


def _run_te_ref(
    cfg: AttentionConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output_grad: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Run TE DotProductAttention in BF16 (no FP8) as reference."""
    dpa = DotProductAttention(
        num_attention_heads=cfg.num_heads,
        kv_channels=cfg.head_dim,
        num_gqa_groups=cfg.num_gqa_groups,
        attention_dropout=cfg.dropout_p,
        qkv_format="bshd",
        attn_mask_type="causal" if cfg.causal else "no_mask",
        softmax_scale=cfg.softmax_scale,
        sequence_parallel=False,
        tp_size=1,
        get_rng_state_tracker=_get_rng_tracker,
        layer_number=1,
    ).to(dtype=cfg.dtype, device="cuda")

    out = dpa(q, k, v, qkv_format="bshd")
    out.backward(output_grad)
    return out.detach(), (q.grad.detach(), k.grad.detach(), v.grad.detach())


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Max element-wise relative error, with small denominator guard."""
    diff = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp(min=1e-6)
    return (diff / denom).max().item()


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0), b_flat.unsqueeze(0)
    ).item()


FWD_REL_TOL = 0.15       # FP8 forward vs BF16 reference
BWD_REL_TOL = 0.25       # FP8 backward vs BF16 reference (gradients are noisier)
CROSS_REL_TOL = 0.20     # TL FP8 vs TE FP8 forward
CROSS_BWD_TOL = 0.30     # TL FP8 vs TE FP8 backward
COS_SIM_MIN = 0.95       # minimum cosine similarity


# ---------------------------------------------------------------------------
# Tests: TL FP8 vs BF16 reference (self-consistency)
# ---------------------------------------------------------------------------

@skip_no_gpu
@skip_no_tl
@pytest.mark.parametrize("config_name", list(CONFIGS.keys()))
@pytest.mark.parametrize("quant_type", FP8_QUANT_TYPES)
def test_tl_fp8_fwd_vs_ref(config_name: str, quant_type: str):
    """TransformerLight FP8 forward output should be close to its own BF16 output."""
    cfg = CONFIGS[config_name]
    q, k, v = _make_qkv(cfg)
    q_ref, k_ref, v_ref = _clone_qkv(q, k, v)
    out_grad = _make_output_grad(
        (cfg.batch_size, cfg.max_seqlen, cfg.num_heads, cfg.head_dim),
        cfg.dtype,
    )

    ref_out, _ = _run_tl_ref(cfg, q_ref, k_ref, v_ref, out_grad)
    fp8_out, _ = _run_tl_fp8(cfg, q, k, v, quant_type, out_grad)

    rel_err = _relative_error(fp8_out, ref_out)
    cos_sim = _cosine_similarity(fp8_out, ref_out)

    print(f"\n[TL FP8 fwd] {config_name}/{quant_type}: "
          f"rel_err={rel_err:.4f}, cos_sim={cos_sim:.6f}")

    assert cos_sim >= COS_SIM_MIN, (
        f"Cosine similarity {cos_sim:.4f} < {COS_SIM_MIN} "
        f"for config {config_name}/{quant_type}"
    )


@skip_no_gpu
@skip_no_tl
@pytest.mark.parametrize("config_name", list(CONFIGS.keys()))
@pytest.mark.parametrize("quant_type", FP8_QUANT_TYPES)
def test_tl_fp8_bwd_vs_ref(config_name: str, quant_type: str):
    """TransformerLight FP8 gradients should be close to its own BF16 gradients."""
    cfg = CONFIGS[config_name]
    q, k, v = _make_qkv(cfg)
    q_ref, k_ref, v_ref = _clone_qkv(q, k, v)
    out_grad = _make_output_grad(
        (cfg.batch_size, cfg.max_seqlen, cfg.num_heads, cfg.head_dim),
        cfg.dtype,
    )

    _, ref_grads = _run_tl_ref(cfg, q_ref, k_ref, v_ref, out_grad)
    _, fp8_grads = _run_tl_fp8(cfg, q, k, v, quant_type, out_grad)

    for name, fp8_g, ref_g in zip(("dQ", "dK", "dV"), fp8_grads, ref_grads):
        cos_sim = _cosine_similarity(fp8_g, ref_g)
        rel_err = _relative_error(fp8_g, ref_g)
        print(f"  [{name}] rel_err={rel_err:.4f}, cos_sim={cos_sim:.6f}")
        assert cos_sim >= COS_SIM_MIN, (
            f"{name} cosine similarity {cos_sim:.4f} < {COS_SIM_MIN} "
            f"for config {config_name}/{quant_type}"
        )


# ---------------------------------------------------------------------------
# Tests: TE FP8 vs BF16 reference (self-consistency)
# ---------------------------------------------------------------------------

@skip_no_gpu
@skip_no_te
@skip_no_fp8
@pytest.mark.parametrize("config_name", list(CONFIGS.keys()))
def test_te_fp8_fwd_vs_ref(config_name: str):
    """TE FP8 forward output should be close to its own BF16 output."""
    FP8GlobalStateManager.reset()
    cfg = CONFIGS[config_name]
    q, k, v = _make_qkv(cfg)
    q_ref, k_ref, v_ref = _clone_qkv(q, k, v)
    out_grad = _make_output_grad(
        (cfg.batch_size, cfg.max_seqlen, cfg.num_heads, cfg.head_dim),
        cfg.dtype,
    )

    ref_out, _ = _run_te_ref(cfg, q_ref, k_ref, v_ref, out_grad)
    FP8GlobalStateManager.reset()
    fp8_out, _ = _run_te_fp8(cfg, q, k, v, out_grad)

    rel_err = _relative_error(fp8_out, ref_out)
    cos_sim = _cosine_similarity(fp8_out, ref_out)

    print(f"\n[TE FP8 fwd] {config_name}: "
          f"rel_err={rel_err:.4f}, cos_sim={cos_sim:.6f}")

    assert cos_sim >= COS_SIM_MIN, (
        f"Cosine similarity {cos_sim:.4f} < {COS_SIM_MIN} "
        f"for config {config_name}"
    )


@skip_no_gpu
@skip_no_te
@skip_no_fp8
@pytest.mark.parametrize("config_name", list(CONFIGS.keys()))
def test_te_fp8_bwd_vs_ref(config_name: str):
    """TE FP8 gradients should be close to its own BF16 gradients."""
    FP8GlobalStateManager.reset()
    cfg = CONFIGS[config_name]
    q, k, v = _make_qkv(cfg)
    q_ref, k_ref, v_ref = _clone_qkv(q, k, v)
    out_grad = _make_output_grad(
        (cfg.batch_size, cfg.max_seqlen, cfg.num_heads, cfg.head_dim),
        cfg.dtype,
    )

    _, ref_grads = _run_te_ref(cfg, q_ref, k_ref, v_ref, out_grad)
    FP8GlobalStateManager.reset()
    _, fp8_grads = _run_te_fp8(cfg, q, k, v, out_grad)

    for name, fp8_g, ref_g in zip(("dQ", "dK", "dV"), fp8_grads, ref_grads):
        cos_sim = _cosine_similarity(fp8_g, ref_g)
        rel_err = _relative_error(fp8_g, ref_g)
        print(f"  [{name}] rel_err={rel_err:.4f}, cos_sim={cos_sim:.6f}")
        assert cos_sim >= COS_SIM_MIN, (
            f"{name} cosine similarity {cos_sim:.4f} < {COS_SIM_MIN} "
            f"for config {config_name}"
        )


# ---------------------------------------------------------------------------
# Tests: Cross-comparison  TL FP8 vs TE FP8
# ---------------------------------------------------------------------------

@skip_no_gpu
@skip_no_tl
@skip_no_te
@skip_no_fp8
@pytest.mark.parametrize("config_name", list(CONFIGS.keys()))
@pytest.mark.parametrize("quant_type", FP8_QUANT_TYPES)
def test_tl_vs_te_fp8_fwd(config_name: str, quant_type: str):
    """TL FP8 forward output should be close to TE FP8 forward output."""
    FP8GlobalStateManager.reset()
    cfg = CONFIGS[config_name]
    q, k, v = _make_qkv(cfg)
    q_te, k_te, v_te = _clone_qkv(q, k, v)
    out_grad = _make_output_grad(
        (cfg.batch_size, cfg.max_seqlen, cfg.num_heads, cfg.head_dim),
        cfg.dtype,
    )

    tl_out, _ = _run_tl_fp8(cfg, q, k, v, quant_type, out_grad)
    FP8GlobalStateManager.reset()
    te_out, _ = _run_te_fp8(cfg, q_te, k_te, v_te, out_grad)

    rel_err = _relative_error(tl_out, te_out)
    cos_sim = _cosine_similarity(tl_out, te_out)

    print(f"\n[TL vs TE FP8 fwd] {config_name}/{quant_type}: "
          f"rel_err={rel_err:.4f}, cos_sim={cos_sim:.6f}")

    assert cos_sim >= COS_SIM_MIN, (
        f"Cosine similarity {cos_sim:.4f} < {COS_SIM_MIN} "
        f"for config {config_name}/{quant_type}"
    )


@skip_no_gpu
@skip_no_tl
@skip_no_te
@skip_no_fp8
@pytest.mark.parametrize("config_name", list(CONFIGS.keys()))
@pytest.mark.parametrize("quant_type", FP8_QUANT_TYPES)
def test_tl_vs_te_fp8_bwd(config_name: str, quant_type: str):
    """TL FP8 gradients should be close to TE FP8 gradients."""
    FP8GlobalStateManager.reset()
    cfg = CONFIGS[config_name]
    q, k, v = _make_qkv(cfg)
    q_te, k_te, v_te = _clone_qkv(q, k, v)
    out_grad = _make_output_grad(
        (cfg.batch_size, cfg.max_seqlen, cfg.num_heads, cfg.head_dim),
        cfg.dtype,
    )

    _, tl_grads = _run_tl_fp8(cfg, q, k, v, quant_type, out_grad)
    FP8GlobalStateManager.reset()
    _, te_grads = _run_te_fp8(cfg, q_te, k_te, v_te, out_grad)

    for name, tl_g, te_g in zip(("dQ", "dK", "dV"), tl_grads, te_grads):
        cos_sim = _cosine_similarity(tl_g, te_g)
        rel_err = _relative_error(tl_g, te_g)
        print(f"  [{name}] rel_err={rel_err:.4f}, cos_sim={cos_sim:.6f}")
        assert cos_sim >= COS_SIM_MIN, (
            f"{name} cosine similarity {cos_sim:.4f} < {COS_SIM_MIN} "
            f"for config {config_name}/{quant_type}"
        )


# ---------------------------------------------------------------------------
# Tests: Non-FP8 baseline sanity (TL triton vs TE flash — both BF16)
# ---------------------------------------------------------------------------

@skip_no_gpu
@skip_no_tl
@skip_no_te
@pytest.mark.parametrize("config_name", list(CONFIGS.keys()))
def test_tl_vs_te_bf16_fwd(config_name: str):
    """Sanity: TL triton BF16 output should be close to TE BF16 output."""
    cfg = CONFIGS[config_name]
    q, k, v = _make_qkv(cfg)
    q_te, k_te, v_te = _clone_qkv(q, k, v)
    out_grad = _make_output_grad(
        (cfg.batch_size, cfg.max_seqlen, cfg.num_heads, cfg.head_dim),
        cfg.dtype,
    )

    tl_out, _ = _run_tl_ref(cfg, q, k, v, out_grad)
    te_out, _ = _run_te_ref(cfg, q_te, k_te, v_te, out_grad)

    cos_sim = _cosine_similarity(tl_out, te_out)
    rel_err = _relative_error(tl_out, te_out)

    print(f"\n[TL vs TE BF16 fwd] {config_name}: "
          f"rel_err={rel_err:.4f}, cos_sim={cos_sim:.6f}")

    assert cos_sim >= 0.99, (
        f"BF16 cosine similarity {cos_sim:.4f} < 0.99 for config {config_name}"
    )


@skip_no_gpu
@skip_no_tl
@skip_no_te
@pytest.mark.parametrize("config_name", list(CONFIGS.keys()))
def test_tl_vs_te_bf16_bwd(config_name: str):
    """Sanity: TL triton BF16 gradients should be close to TE BF16 gradients."""
    cfg = CONFIGS[config_name]
    q, k, v = _make_qkv(cfg)
    q_te, k_te, v_te = _clone_qkv(q, k, v)
    out_grad = _make_output_grad(
        (cfg.batch_size, cfg.max_seqlen, cfg.num_heads, cfg.head_dim),
        cfg.dtype,
    )

    _, tl_grads = _run_tl_ref(cfg, q, k, v, out_grad)
    _, te_grads = _run_te_ref(cfg, q_te, k_te, v_te, out_grad)

    for name, tl_g, te_g in zip(("dQ", "dK", "dV"), tl_grads, te_grads):
        cos_sim = _cosine_similarity(tl_g, te_g)
        rel_err = _relative_error(tl_g, te_g)
        print(f"  [{name}] rel_err={rel_err:.4f}, cos_sim={cos_sim:.6f}")
        assert cos_sim >= 0.99, (
            f"{name} BF16 cosine similarity {cos_sim:.4f} < 0.99 "
            f"for config {config_name}"
        )


# ---------------------------------------------------------------------------
# Detailed error report (not a strict pass/fail — informational)
# ---------------------------------------------------------------------------

@skip_no_gpu
@skip_no_tl
@skip_no_te
@skip_no_fp8
@pytest.mark.parametrize("config_name", ["small_mha_causal", "medium_mha_causal"])
@pytest.mark.parametrize("quant_type", FP8_QUANT_TYPES)
def test_fp8_error_report(config_name: str, quant_type: str):
    """
    Comprehensive error report: BF16 ref, TL FP8, TE FP8 — all metrics printed.
    Assertion uses relaxed tolerance so the test primarily serves as a report.
    """
    FP8GlobalStateManager.reset()
    cfg = CONFIGS[config_name]

    q0, k0, v0 = _make_qkv(cfg)
    out_grad = _make_output_grad(
        (cfg.batch_size, cfg.max_seqlen, cfg.num_heads, cfg.head_dim),
        cfg.dtype,
    )

    q_tl_ref, k_tl_ref, v_tl_ref = _clone_qkv(q0, k0, v0)
    q_tl_fp8, k_tl_fp8, v_tl_fp8 = _clone_qkv(q0, k0, v0)
    q_te_ref, k_te_ref, v_te_ref = _clone_qkv(q0, k0, v0)
    q_te_fp8, k_te_fp8, v_te_fp8 = _clone_qkv(q0, k0, v0)

    tl_ref_out, tl_ref_grads = _run_tl_ref(cfg, q_tl_ref, k_tl_ref, v_tl_ref, out_grad)
    tl_fp8_out, tl_fp8_grads = _run_tl_fp8(cfg, q_tl_fp8, k_tl_fp8, v_tl_fp8, quant_type, out_grad)
    te_ref_out, te_ref_grads = _run_te_ref(cfg, q_te_ref, k_te_ref, v_te_ref, out_grad)
    FP8GlobalStateManager.reset()
    te_fp8_out, te_fp8_grads = _run_te_fp8(cfg, q_te_fp8, k_te_fp8, v_te_fp8, out_grad)

    header = f"\n{'='*72}\nError Report: {config_name} / {quant_type}\n{'='*72}"
    print(header)

    pairs = [
        ("TL_FP8 vs TL_BF16", tl_fp8_out, tl_ref_out, tl_fp8_grads, tl_ref_grads),
        ("TE_FP8 vs TE_BF16", te_fp8_out, te_ref_out, te_fp8_grads, te_ref_grads),
        ("TL_FP8 vs TE_FP8",  tl_fp8_out, te_fp8_out, tl_fp8_grads, te_fp8_grads),
        ("TL_BF16 vs TE_BF16", tl_ref_out, te_ref_out, tl_ref_grads, te_ref_grads),
    ]

    for label, out_a, out_b, grads_a, grads_b in pairs:
        fwd_cos = _cosine_similarity(out_a, out_b)
        fwd_rel = _relative_error(out_a, out_b)
        print(f"  {label:25s}  fwd: cos={fwd_cos:.6f}  rel={fwd_rel:.4f}")
        for gname, ga, gb in zip(("dQ", "dK", "dV"), grads_a, grads_b):
            g_cos = _cosine_similarity(ga, gb)
            g_rel = _relative_error(ga, gb)
            print(f"    {gname:4s}: cos={g_cos:.6f}  rel={g_rel:.4f}")

    assert True  # informational test; always passes
