"""Triton-fused forward-KL loss for Eagle3 speculative distillation.

Two custom Triton kernels (forward + backward) fuse
softmax + log_softmax + KL divergence into a 2-pass online-softmax
computation that **never materializes the [N, V] probability tensors**.

Compared to the torch.compile version this replaces:
  - Saves 2×N×V×4 bytes HBM (no teacher-probs / draft-log-probs intermediate)
  - Reduces HBM traffic from ~10V to ~6V reads per row
  - Casts bf16→f32 inside the kernel (no separate .float() materialization)

The lm_head GEMMs remain as F.linear (hipBLAS) — hand-written Triton
cannot beat vendor BLAS for large GEMM.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

__all__ = ["fused_forward_kl_loss"]


# ── Triton kernels ──────────────────────────────────────────────────────────


@triton.jit
def _kl_fwd_kernel(
    D_ptr,
    T_ptr,
    LOSS_ptr,
    N,
    V,
    stride_d,
    stride_t,
    BLOCK_V: tl.constexpr,
):
    """Forward KL per row: loss = -Σ p·log q, p=softmax(T), q=softmax(D).

    Pass 1 — online softmax: compute running (max, sum_exp) for both
    teacher and draft logits in a single sweep.
    Pass 2 — accumulate KL: re-read logits, compute
    -Σ (exp(t-t_m)/t_s) · (d - d_m - log(d_s)).
    """
    row = tl.program_id(0)
    if row >= N:
        return

    d_base = D_ptr + row * stride_d
    t_base = T_ptr + row * stride_t

    # ── pass 1: online softmax denominators ──
    d_m: tl.float32 = -float("inf")
    d_s: tl.float32 = 0.0
    t_m: tl.float32 = -float("inf")
    t_s: tl.float32 = 0.0

    for off in range(0, V, BLOCK_V):
        cols = off + tl.arange(0, BLOCK_V)
        mask = cols < V
        dv = tl.load(d_base + cols, mask=mask, other=-float("inf")).to(tl.float32)
        tv = tl.load(t_base + cols, mask=mask, other=-float("inf")).to(tl.float32)

        bm = tl.max(dv, axis=0)
        nm = tl.maximum(d_m, bm)
        d_s = d_s * tl.exp(d_m - nm) + tl.sum(tl.exp(dv - nm), axis=0)
        d_m = nm

        bm = tl.max(tv, axis=0)
        nm = tl.maximum(t_m, bm)
        t_s = t_s * tl.exp(t_m - nm) + tl.sum(tl.exp(tv - nm), axis=0)
        t_m = nm

    log_d_s = tl.log(d_s)

    # ── pass 2: KL accumulation ──
    acc: tl.float32 = 0.0
    for off in range(0, V, BLOCK_V):
        cols = off + tl.arange(0, BLOCK_V)
        mask = cols < V
        dv = tl.load(d_base + cols, mask=mask, other=-float("inf")).to(tl.float32)
        tv = tl.load(t_base + cols, mask=mask, other=-float("inf")).to(tl.float32)
        p_unnorm = tl.exp(tv - t_m)
        log_q = dv - d_m - log_d_s
        # mask avoids 0 * (-inf) = NaN for padding positions
        acc += tl.sum(tl.where(mask, p_unnorm * log_q, 0.0), axis=0)

    tl.store(LOSS_ptr + row, -acc / t_s)


@triton.jit
def _kl_bwd_kernel(
    D_ptr,
    T_ptr,
    GRAD_ptr,
    GRAD_OUT_ptr,
    inv_N,
    N,
    V,
    stride_d,
    stride_t,
    stride_g,
    BLOCK_V: tl.constexpr,
):
    """Backward: grad_draft[i,j] = (grad_out/N) · (q_ij - p_ij)."""
    row = tl.program_id(0)
    if row >= N:
        return

    d_base = D_ptr + row * stride_d
    t_base = T_ptr + row * stride_t
    g_base = GRAD_ptr + row * stride_g

    gs = tl.load(GRAD_OUT_ptr).to(tl.float32) * inv_N

    # ── pass 1: online softmax denominators ──
    d_m: tl.float32 = -float("inf")
    d_s: tl.float32 = 0.0
    t_m: tl.float32 = -float("inf")
    t_s: tl.float32 = 0.0

    for off in range(0, V, BLOCK_V):
        cols = off + tl.arange(0, BLOCK_V)
        mask = cols < V
        dv = tl.load(d_base + cols, mask=mask, other=-float("inf")).to(tl.float32)
        tv = tl.load(t_base + cols, mask=mask, other=-float("inf")).to(tl.float32)

        bm = tl.max(dv, axis=0)
        nm = tl.maximum(d_m, bm)
        d_s = d_s * tl.exp(d_m - nm) + tl.sum(tl.exp(dv - nm), axis=0)
        d_m = nm

        bm = tl.max(tv, axis=0)
        nm = tl.maximum(t_m, bm)
        t_s = t_s * tl.exp(t_m - nm) + tl.sum(tl.exp(tv - nm), axis=0)
        t_m = nm

    # ── pass 2: write grad = gs · (q - p) ──
    for off in range(0, V, BLOCK_V):
        cols = off + tl.arange(0, BLOCK_V)
        mask = cols < V
        dv = tl.load(d_base + cols, mask=mask, other=-float("inf")).to(tl.float32)
        tv = tl.load(t_base + cols, mask=mask, other=-float("inf")).to(tl.float32)
        q = tl.exp(dv - d_m) / d_s
        p = tl.exp(tv - t_m) / t_s
        tl.store(g_base + cols, gs * (q - p), mask=mask)


# ── Autograd wrapper ───────────────────────────────────────────────────────

_BLOCK_V = 4096
_NUM_WARPS = 4


class _TritonFusedKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, draft_logits, teacher_logits):
        N, V = draft_logits.shape
        loss = torch.empty(N, dtype=torch.float32, device=draft_logits.device)
        _kl_fwd_kernel[(N,)](
            draft_logits,
            teacher_logits,
            loss,
            N,
            V,
            draft_logits.stride(0),
            teacher_logits.stride(0),
            BLOCK_V=_BLOCK_V,
            num_warps=_NUM_WARPS,
        )
        ctx.save_for_backward(draft_logits, teacher_logits)
        return loss.mean()

    @staticmethod
    def backward(ctx, grad_output):
        draft_l, teacher_l = ctx.saved_tensors
        N, V = draft_l.shape
        grad = torch.empty(N, V, dtype=torch.float32, device=draft_l.device)
        _kl_bwd_kernel[(N,)](
            draft_l,
            teacher_l,
            grad,
            grad_output,
            1.0 / N,
            N,
            V,
            draft_l.stride(0),
            teacher_l.stride(0),
            grad.stride(0),
            BLOCK_V=_BLOCK_V,
            num_warps=_NUM_WARPS,
        )
        return grad.to(draft_l.dtype), None


# ── High-level API ─────────────────────────────────────────────────────────


_CHUNK = 2048


def _forward_impl(hs, ths, norm_weight, draft_lm_head_weight, target_lm_head_weight, norm_eps):
    try:
        from lumen.ops.normalization.rmsnorm import rmsnorm

        norm_hs = rmsnorm(hs, norm_weight, norm_eps)
    except ImportError:
        hs_f32 = hs.float()
        variance = hs_f32.pow(2).mean(-1, keepdim=True)
        norm_hs = (hs_f32 * torch.rsqrt(variance + norm_eps)).to(hs.dtype) * norm_weight

    N = norm_hs.shape[0]
    loss_parts = []
    acc_parts = []
    for cs in range(0, N, _CHUNK):
        ce = min(cs + _CHUNK, N)
        draft_logits = F.linear(norm_hs[cs:ce], draft_lm_head_weight)
        with torch.no_grad():
            teacher_logits = F.linear(ths[cs:ce], target_lm_head_weight)
        chunk_loss = _TritonFusedKL.apply(draft_logits, teacher_logits)
        loss_parts.append(chunk_loss * (ce - cs))
        with torch.no_grad():
            acc_parts.append((draft_logits.argmax(-1) == teacher_logits.argmax(-1)).float().sum())
        del draft_logits, teacher_logits

    loss = sum(loss_parts) / N
    with torch.no_grad():
        acc = sum(acc_parts) / N
    return loss, acc


def fused_forward_kl_loss(
    draft_hidden_states,
    target_hidden_states,
    loss_mask,
    norm_weight,
    draft_lm_head_weight,
    target_lm_head_weight,
    norm_eps,
    gradient_checkpointing=False,
):
    """Triton-fused forward-KL loss from hidden states.

    Flow: index_select → RMSNorm → lm_head GEMMs → Triton KL kernel.
    The KL kernel uses 2-pass online-softmax to avoid materializing
    the [N, V] teacher-probs and draft-log-probs tensors.

    Args:
        draft_hidden_states: (B, T, H) draft pre-norm hidden states
        target_hidden_states: (B, T, D) teacher last hidden states
        loss_mask: (B, T) float mask (1 = valid, 0 = padding)
        norm_weight: (H,) draft model's output RMSNorm weight
        draft_lm_head_weight: (V, H) draft lm_head weight
        target_lm_head_weight: (V, D) teacher lm_head weight
        norm_eps: float — RMSNorm epsilon
        gradient_checkpointing: if True, wrap in activation checkpoint
    """
    valid_idx = loss_mask.flatten().nonzero().squeeze(-1)
    if valid_idx.numel() == 0:
        zero = draft_hidden_states.sum() * 0.0
        return zero, zero.detach()

    torch._dynamo.maybe_mark_dynamic(valid_idx, 0)

    hs_flat = draft_hidden_states.reshape(-1, draft_hidden_states.shape[-1])
    ths_flat = target_hidden_states.reshape(-1, target_hidden_states.shape[-1])
    hs = hs_flat.index_select(0, valid_idx)
    ths = ths_flat.index_select(0, valid_idx)

    args = (hs, ths, norm_weight, draft_lm_head_weight, target_lm_head_weight, norm_eps)

    if gradient_checkpointing:
        from torch.utils.checkpoint import checkpoint

        return checkpoint(_forward_impl, *args, use_reentrant=False)
    return _forward_impl(*args)
