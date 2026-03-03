###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Literal, Optional

import torch

from transformer_light.quantize import is_aiter_available

if is_aiter_available():
    from aiter.ops.mha import flash_attn_func

from transformer_light.kernels.attention.attention_triton_impl import (
    attention_mxfp8_forward_triton_impl,
    attention_triton_backward_impl,
    attention_triton_forward_impl,
    attention_triton_mxfp8_backward_triton_impl,
    is_cdna4,
)
from transformer_light.ops.attention.attention_cp_dispatcher import (
    dispatch_attention_cp_functions,
)
from transformer_light.ops.attention.attention_utils import (
    block_scaling_node,
    block_scaling_node_mxfp8,
    get_f8_fwd_dtype,
    quant_p_scale_mxfp8,
    quant_v_get_p_scale,
)

__all__ = ["attention", "attention_fp8_quant"]


def _attention_aiter_impl(
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
    return_softmax,
):
    """Thin wrapper around ``flash_attn_func`` from aiter.

    This is a plain differentiable function (not a torch.autograd.Function) so
    autograd flows through ``flash_attn_func`` directly.
    """
    _return_softmax = return_softmax and dropout_p > 0
    attn_result = flash_attn_func(
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

    # flash_attn_func returns (out,) or (out, lse) or (out, lse, S_dmask)
    if not return_lse and not _return_softmax:
        # scalar result
        return attn_result
    if isinstance(attn_result, tuple):
        return attn_result
    return attn_result


# Backwards-compatible alias: callers that used ``AttentionCKFunction.apply(...)``
# can now use ``AttentionAiterFunction.apply(...)`` with the same positional args.
class AttentionAiterFunction:
    @staticmethod
    def apply(
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
        return_softmax,
        is_grad_enabled,
        is_v3_atomic_fp32: Optional[bool] = True,
        how_v3_bf16_cvt: Optional[int] = 1,
    ):
        return _attention_aiter_impl(
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
            return_softmax,
        )


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
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        q, q_scale = block_scaling_node(q, use_fp8)
        k, k_scale = block_scaling_node(k, use_fp8)
        v, v_scale, p_scale = quant_v_get_p_scale(v, use_fp8)

        output, softmax_lse, exp_scores = attention_triton_forward_impl(
            q,
            k,
            v,
            p_scale,
            q_scale,
            k_scale,
            v_scale,
            dropout_p,
            softmax_scale,
            causal,
            window_size[0],
            window_size[1],
            bias,
            alibi_slopes,
            return_softmax,
            use_fp8,
        )

        if is_grad:
            # q, k, v should be fp8 when set use_fp8 to True
            ctx.save_for_backward(q, k, v, output, softmax_lse, alibi_slopes, bias, q_scale, k_scale, v_scale)

            ctx.sm_scale = softmax_scale
            ctx.p_scale = p_scale
            ctx.causal = causal
            ctx.use_fp8 = use_fp8
            ctx.cu_seqlens_q = 0
            ctx.cu_seqlens_k = 0
            ctx.max_seqlens_q = q.shape[1]
            ctx.max_seqlens_k = k.shape[1]

        result = [output]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(exp_scores)
        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        (q, k, v, o, softmax_lse, alibi_slopes, bias, q_scale, k_scale, v_scale) = ctx.saved_tensors
        assert bias is None, "Currently bias is not supported by fa backward function."
        assert do.dtype is torch.bfloat16, f"do should be bfloat16 but get {do.dtype}"

        dq, dk, dv = attention_triton_backward_impl(
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
            None,
            None,
            None,
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_k,
            ctx.max_seqlens_q,
            ctx.max_seqlens_k,
            ctx.sm_scale,
            ctx.causal,
            -1,
            -1,
            alibi_slopes,
            ctx.use_fp8,
        )

        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


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
        block_m_fwd: int = 64,  # block of query seq len in fwd
        block_n_fwd: int = 64,  # block of key/value seq len in fwd
        block_m_dq_bwd: int = 64,  # block of dq seq len in bwd
        block_n_dq_bwd: int = 64,  # block of dq seq len in bwd
        block_m_dkv_bwd: int = 64,  # block of dkv seq len in bwd
        block_n_dkv_bwd: int = 64,  # block of dkv seq len in bwd
        quant_block_size: int = 32,
    ):
        assert is_cdna4(), "mxfp8 is only supported by gfx950 and newer version"
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        if use_mxfp8:
            q, q_scale = block_scaling_node_mxfp8(
                q,
                quant_block_size,
                "bshd",
                is_2d_block=True,
                float8_dtype_pt=get_f8_fwd_dtype(),
                cu_seqlens=0,
                max_seqlens=q.shape[1],
            )
            k, k_scale = block_scaling_node_mxfp8(
                k,
                quant_block_size,
                "bshd",
                is_2d_block=True,
                float8_dtype_pt=get_f8_fwd_dtype(),
                cu_seqlens=0,
                max_seqlens=k.shape[1],
            )
            v, v_scale = block_scaling_node_mxfp8(
                v,
                quant_block_size,
                "bshd",
                is_2d_block=True,
                float8_dtype_pt=get_f8_fwd_dtype(),
                cu_seqlens=0,
                max_seqlens=k.shape[1],
            )
            p_scale = quant_p_scale_mxfp8()
        else:
            q_scale = torch.scalar_tensor(1.0, device=q.device)
            k_scale = torch.scalar_tensor(1.0, device=q.device)
            v_scale = torch.scalar_tensor(1.0, device=q.device)
            p_scale = 127

        output, softmax_lse, exp_scores = attention_mxfp8_forward_triton_impl(
            q=q,
            k=k,
            v=v,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            p_scale=p_scale,
            sm_scale=softmax_scale,
            alibi_slopes=alibi_slopes,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            bias=bias,
            dropout_p=dropout_p,
            return_softmax=return_softmax,
            use_mxfp8=use_mxfp8,
            block_m=block_m_fwd,
            block_n=block_n_fwd,
            quant_block_size=quant_block_size,
        )

        if is_grad:
            # q, k, v should be fp8 when set use_fp8 to True
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
            )
            ctx.use_mxfp8 = use_mxfp8
            ctx.p_scale = p_scale
            ctx.sm_scale = softmax_scale
            ctx.causal = causal
            ctx.dropout_p = dropout_p
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

        result = [output]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(exp_scores)
        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        (q, k, v, o, softmax_lse, alibi_slopes, bias, q_scale, k_scale, v_scale) = ctx.saved_tensors
        assert bias is None, "Currently bias is not supported by fa backward function."
        assert do.dtype is torch.bfloat16, f"do should be bfloat16 but get {do.dtype}"

        dq, dk, dv = attention_triton_mxfp8_backward_triton_impl(
            do=do,
            q=q,
            k=k,
            v=v,
            o=o,
            softmax_lse=softmax_lse,
            dq=None,
            dk=None,
            dv=None,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            sm_scale=ctx.sm_scale,
            p_scale=ctx.p_scale,
            alibi_slopes=alibi_slopes,
            causal=ctx.causal,
            window_size_left=-1,
            window_size_right=-1,
            cu_seqlens_q=ctx.cu_seqlens_q,
            cu_seqlens_k=ctx.cu_seqlens_k,
            max_seqlen_q=ctx.max_seqlens_q,
            max_seqlen_k=ctx.max_seqlens_k,
            use_mxfp8=ctx.use_mxfp8,
            block_m_dq_bwd=ctx.block_m_dq_bwd,
            block_n_dq_bwd=ctx.block_n_dq_bwd,
            block_m_dkv_bwd=ctx.block_m_dkv_bwd,
            block_n_dkv_bwd=ctx.block_n_dkv_bwd,
            quant_block_size=ctx.quant_block_size,
        )
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def attention(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    bias=None,
    alibi_slopes=None,
    deterministic=True,
    return_lse=False,
    return_attn_probs=False,
    backend_type: str = "aiter",  # 'aiter', 'triton'
    cp_param_bundle=None,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if cp_param_bundle is not None:  # CP
        assert "cp_group" in cp_param_bundle
        assert "cp_comm_type" in cp_param_bundle
        return dispatch_attention_cp_functions(
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
            backend_type,
            None,
            cp_param_bundle["cp_group"],
            cp_param_bundle["cp_comm_type"],
        )

    if backend_type == "aiter":
        if not is_aiter_available():
            raise RuntimeError(
                "AITER is not installed. The aiter attention backend requires "
                "'aiter' — install it or use backend_type='triton'."
            )
        return AttentionAiterFunction.apply(
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
        )
    elif backend_type == "triton":
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
    window_size=(-1, -1),  # -1 means infinite context window
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    backend_type: str = "triton",  # for now 'triton' only
    cp_param_bundle=None,
    # following parameters will be used in mxfp8
    quant_type: Literal["fp8_blockwise", "mxfp8"] = "fp8_blockwise",  # "fp8", "mxfp8"
    block_m_fwd: int = 64,  # block of query seq len in fwd
    block_n_fwd: int = 64,  # block of key/value seq len in fwd
    block_m_dq_bwd: int = 64,  # block of dq seq len in bwd
    block_n_dq_bwd: int = 64,  # block of dq seq len in bwd
    block_m_dkv_bwd: int = 64,  # block of dkv seq len in bwd
    block_n_dkv_bwd: int = 64,  # block of dkv seq len in bwd
    quant_block_size: int = 32,
):
    assert backend_type == "triton", "attention_fp8 only support triton backend"
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if cp_param_bundle is not None:  # CP

        assert "cp_group" in cp_param_bundle
        assert "cp_comm_type" in cp_param_bundle
        return dispatch_attention_cp_functions(
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
            backend_type,
            quant_type,
            cp_param_bundle["cp_group"],
            cp_param_bundle["cp_comm_type"],
            block_m_fwd=block_m_fwd,
            block_n_fwd=block_n_fwd,
            block_m_dq_bwd=block_m_dq_bwd,
            block_n_dq_bwd=block_n_dq_bwd,
            block_m_dkv_bwd=block_m_dkv_bwd,
            block_n_dkv_bwd=block_n_dkv_bwd,
            quant_block_size=quant_block_size,
        )

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
        )
    elif quant_type == "fp8_blockwise":
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
        )
    else:
        raise NotImplementedError(f"not supported quant_type {quant_type} backend_type {backend_type} yet")
