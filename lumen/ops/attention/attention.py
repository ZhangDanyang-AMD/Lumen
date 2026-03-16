###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch

from lumen.core.grad_quant import quantize_grad_tensor


def _is_aiter_available() -> bool:
    try:
        import aiter  # noqa: F401

        return True
    except ImportError:
        return False


if _is_aiter_available():
    from aiter.ops.mha import flash_attn_func

from lumen.kernels.attention.attention_impl import attention_backward as triton_fp8_backward
from lumen.kernels.attention.attention_impl import attention_forward as triton_fp8_forward
from lumen.kernels.attention.attention_impl import attention_mxfp8_backward as triton_mxfp8_backward
from lumen.kernels.attention.attention_impl import attention_mxfp8_forward as triton_mxfp8_forward

__all__ = ["attention", "attention_fp8_quant"]

_FP8_BACKENDS = ("aiter_triton_fp8", "aiter_csrc_fp8", "aiter_asm_fp8")

_QUANT_TYPE_ALIAS = {
    "fp8_blockwise": "blockwise",
}


# ---------------------------------------------------------------------------
# torch.compile support
# ---------------------------------------------------------------------------
def _mark_allow_in_graph(*classes):
    try:
        from torch._dynamo import allow_in_graph

        for cls in classes:
            allow_in_graph(cls)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Autograd Functions
# ---------------------------------------------------------------------------


class AttentionTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        return_lse,
        return_softmax,
        is_grad_enabled,
        use_fp8,
        grad_quant_type=None,
        prefer_asm=False,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        (output, softmax_lse, exp_scores, q, k, v, q_scale, k_scale, v_scale, p_scale, rng_state) = triton_fp8_forward(
            q,
            k,
            v,
            use_fp8,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            bias,
            alibi_slopes,
            return_softmax,
            prefer_asm=prefer_asm,
        )

        if is_grad:
            ctx.save_for_backward(
                q,
                k,
                v,
                output,
                softmax_lse,
                alibi_slopes,
                bias,
                q_scale,
                k_scale,
                v_scale,
                rng_state,
            )
            ctx.sm_scale = softmax_scale
            ctx.p_scale = p_scale
            ctx.causal = causal
            ctx.use_fp8 = use_fp8
            ctx.dropout_p = dropout_p
            ctx.window_size = window_size
            ctx.cu_seqlens_q = 0
            ctx.cu_seqlens_k = 0
            ctx.max_seqlens_q = q.shape[1]
            ctx.max_seqlens_k = k.shape[1]
            ctx.grad_quant_type = grad_quant_type

        result = [output]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(exp_scores)
        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        (q, k, v, o, softmax_lse, alibi_slopes, bias, q_scale, k_scale, v_scale, rng_state) = ctx.saved_tensors

        dq, dk, dv = triton_fp8_backward(
            do,
            q,
            k,
            v,
            o,
            q_scale,
            k_scale,
            v_scale,
            ctx.p_scale,
            softmax_lse,
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_k,
            ctx.max_seqlens_q,
            ctx.max_seqlens_k,
            ctx.sm_scale,
            ctx.causal,
            alibi_slopes,
            ctx.use_fp8,
            rng_state=rng_state,
            dropout_p=ctx.dropout_p,
            bias=bias,
            window_size=ctx.window_size,
        )
        gqt = ctx.grad_quant_type
        dq = quantize_grad_tensor(dq, gqt)
        dk = quantize_grad_tensor(dk, gqt)
        dv = quantize_grad_tensor(dv, gqt)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None


class AttentionTritonMXFP8Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        return_lse,
        return_softmax,
        is_grad_enabled,
        use_mxfp8,
        block_m_fwd=64,
        block_n_fwd=64,
        block_m_dq_bwd=64,
        block_n_dq_bwd=64,
        block_m_dkv_bwd=64,
        block_n_dkv_bwd=64,
        quant_block_size=32,
        grad_quant_type=None,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        (output, softmax_lse, exp_scores, q, k, v, q_scale, k_scale, v_scale, p_scale, rng_state) = (
            triton_mxfp8_forward(
                q,
                k,
                v,
                use_mxfp8,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                bias,
                alibi_slopes,
                return_softmax,
                block_m_fwd,
                block_n_fwd,
                quant_block_size,
            )
        )

        if is_grad:
            ctx.save_for_backward(
                q,
                k,
                v,
                output,
                softmax_lse,
                alibi_slopes,
                bias,
                q_scale,
                k_scale,
                v_scale,
                rng_state,
            )
            ctx.use_mxfp8 = use_mxfp8
            ctx.p_scale = p_scale
            ctx.sm_scale = softmax_scale
            ctx.causal = causal
            ctx.dropout_p = dropout_p
            ctx.window_size = window_size
            ctx.layout = "bshd"
            ctx.block_m_dq_bwd = block_m_dq_bwd
            ctx.block_n_dq_bwd = block_n_dq_bwd
            ctx.block_m_dkv_bwd = block_m_dkv_bwd
            ctx.block_n_dkv_bwd = block_n_dkv_bwd
            ctx.quant_block_size = quant_block_size
            ctx.cu_seqlens_q = torch.tensor(0, device="cuda")
            ctx.cu_seqlens_k = torch.tensor(0, device="cuda")
            ctx.max_seqlens_q = q.shape[1]
            ctx.max_seqlens_k = k.shape[1]
            ctx.grad_quant_type = grad_quant_type

        result = [output]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(exp_scores)
        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        (q, k, v, o, softmax_lse, alibi_slopes, bias, q_scale, k_scale, v_scale, rng_state) = ctx.saved_tensors

        dq, dk, dv = triton_mxfp8_backward(
            do,
            q,
            k,
            v,
            o,
            softmax_lse,
            q_scale,
            k_scale,
            v_scale,
            ctx.sm_scale,
            ctx.p_scale,
            alibi_slopes,
            ctx.causal,
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_k,
            ctx.max_seqlens_q,
            ctx.max_seqlens_k,
            ctx.use_mxfp8,
            ctx.block_m_dq_bwd,
            ctx.block_n_dq_bwd,
            ctx.block_m_dkv_bwd,
            ctx.block_n_dkv_bwd,
            ctx.quant_block_size,
            rng_state=rng_state,
            dropout_p=ctx.dropout_p,
            bias=bias,
            window_size=ctx.window_size,
        )
        gqt = ctx.grad_quant_type
        dq = quantize_grad_tensor(dq, gqt)
        dk = quantize_grad_tensor(dk, gqt)
        dv = quantize_grad_tensor(dv, gqt)
        return (
            dq,
            dk,
            dv,
        ) + (None,) * 18


_mark_allow_in_graph(AttentionTritonFunction, AttentionTritonMXFP8Function)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def attention(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    bias=None,
    alibi_slopes=None,
    deterministic=True,
    return_lse=False,
    return_attn_probs=False,
    backend_type: str = "auto",
    cp_param_bundle=None,
    grad_quant_type: Optional[str] = None,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    # Resolve "auto": prefer CK csrc when available, fall back to Triton.
    if backend_type == "auto":
        backend_type = "aiter_csrc" if _is_aiter_available() else "aiter_triton"

    # Context-parallelism path
    if cp_param_bundle is not None:
        assert "cp_group" in cp_param_bundle
        assert "cp_comm_type" in cp_param_bundle
        from lumen.ops.attention.attention_with_cp_a2a import (
            AttentionAiterFunctionCPA2A,
            AttentionTritonFunctionCPA2A,
        )

        cp_group = cp_param_bundle["cp_group"]
        cp_comm_type = cp_param_bundle["cp_comm_type"]
        if backend_type == "aiter_csrc" and cp_comm_type == "a2a":
            return AttentionAiterFunctionCPA2A.apply(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                bias,
                alibi_slopes,
                deterministic,
                return_lse,
                return_attn_probs,
                torch.is_grad_enabled(),
                cp_group,
            )
        elif backend_type == "aiter_triton" and cp_comm_type == "a2a":
            return AttentionTritonFunctionCPA2A.apply(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                bias,
                alibi_slopes,
                return_lse,
                return_attn_probs,
                torch.is_grad_enabled(),
                False,
                cp_group,
                grad_quant_type,
            )
        else:
            raise NotImplementedError(f"not supported backend_type {backend_type} cp_comm_type {cp_comm_type} yet")

    # Single-GPU path: prefer CK csrc, fall back to Triton.
    if backend_type == "aiter_csrc":
        if not _is_aiter_available():
            raise RuntimeError(
                "AITER is not installed. The aiter_csrc attention backend requires "
                "'aiter' — install it or use backend_type='aiter_triton'."
            )
        _return_softmax = return_attn_probs and dropout_p > 0
        try:
            return flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                bias=bias,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_lse=return_lse,
                return_attn_probs=_return_softmax,
            )
        except RuntimeError as _aiter_err:
            import warnings

            warnings.warn(
                f"AITER flash-attention kernel unavailable for this configuration "
                f"(q={tuple(q.shape)}, k={tuple(k.shape)}, causal={causal}): "
                f"{_aiter_err}. Falling back to aiter_triton backend.",
                RuntimeWarning,
                stacklevel=2,
            )
            return AttentionTritonFunction.apply(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                bias,
                alibi_slopes,
                return_lse,
                return_attn_probs,
                torch.is_grad_enabled(),
                False,
                grad_quant_type,
            )
    elif backend_type == "aiter_triton":
        return AttentionTritonFunction.apply(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            bias,
            alibi_slopes,
            return_lse,
            return_attn_probs,
            torch.is_grad_enabled(),
            False,
            grad_quant_type,
        )
    else:
        raise NotImplementedError(f"backend_type {backend_type} not supported")


def attention_fp8_quant(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    backend_type: str = "aiter_triton",
    cp_param_bundle=None,
    quant_type: str = "blockwise",
    block_m_fwd: int = 64,
    block_n_fwd: int = 64,
    block_m_dq_bwd: int = 64,
    block_n_dq_bwd: int = 64,
    block_m_dkv_bwd: int = 64,
    block_n_dkv_bwd: int = 64,
    quant_block_size: int = 32,
    grad_quant_type: Optional[str] = None,
):
    assert backend_type in (
        "aiter_triton",
        "aiter_triton_fp8",
        "aiter_csrc_fp8",
        "aiter_asm_fp8",
    ), (
        f"attention_fp8_quant only supports aiter_triton/aiter_triton_fp8/"
        f"aiter_csrc_fp8/aiter_asm_fp8 backends, got {backend_type}"
    )

    # Normalise legacy quant_type names
    quant_type = _QUANT_TYPE_ALIAS.get(quant_type, quant_type)

    # "none" means no FP8 quantisation — delegate to bf16 attention
    if quant_type == "none":
        _effective_backend = "aiter_csrc" if _is_aiter_available() else "aiter_triton"
        return attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            bias=bias,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_lse=return_lse,
            return_attn_probs=return_attn_probs,
            backend_type=_effective_backend,
            cp_param_bundle=cp_param_bundle,
            grad_quant_type=grad_quant_type,
        )

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    prefer_asm = backend_type == "aiter_asm_fp8"

    # Context-parallelism path
    if cp_param_bundle is not None:
        assert "cp_group" in cp_param_bundle
        assert "cp_comm_type" in cp_param_bundle
        from lumen.ops.attention.attention_with_cp_a2a import (
            AttentionTritonFunctionCPA2A,
            AttentionTritonMXFP8FunctionCPA2A,
        )

        cp_group = cp_param_bundle["cp_group"]
        cp_comm_type = cp_param_bundle["cp_comm_type"]
        if cp_comm_type != "a2a":
            raise NotImplementedError(f"not supported backend_type {backend_type} cp_comm_type {cp_comm_type} yet")
        if quant_type == "mxfp8":
            return AttentionTritonMXFP8FunctionCPA2A.apply(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                bias,
                alibi_slopes,
                return_lse,
                return_attn_probs,
                torch.is_grad_enabled(),
                True,
                cp_group,
                block_m_fwd,
                block_n_fwd,
                block_m_dq_bwd,
                block_n_dq_bwd,
                block_m_dkv_bwd,
                block_n_dkv_bwd,
                quant_block_size,
                grad_quant_type,
            )
        elif quant_type in ("blockwise", "dynamic", "delayed", "per_token"):
            return AttentionTritonFunctionCPA2A.apply(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                bias,
                alibi_slopes,
                return_lse,
                return_attn_probs,
                torch.is_grad_enabled(),
                True,
                cp_group,
                grad_quant_type,
            )
        else:
            raise NotImplementedError(f"not supported quant_type {quant_type}")

    # Single-GPU path
    if quant_type == "mxfp8":
        return AttentionTritonMXFP8Function.apply(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            bias,
            alibi_slopes,
            return_lse,
            return_attn_probs,
            torch.is_grad_enabled(),
            True,
            block_m_fwd,
            block_n_fwd,
            block_m_dq_bwd,
            block_n_dq_bwd,
            block_m_dkv_bwd,
            block_n_dkv_bwd,
            quant_block_size,
            grad_quant_type,
        )
    elif quant_type in ("blockwise", "dynamic", "delayed", "per_token"):
        return AttentionTritonFunction.apply(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            bias,
            alibi_slopes,
            return_lse,
            return_attn_probs,
            torch.is_grad_enabled(),
            True,
            grad_quant_type,
            prefer_asm,
        )
    else:
        raise NotImplementedError(f"not supported quant_type {quant_type} backend_type {backend_type} yet")
