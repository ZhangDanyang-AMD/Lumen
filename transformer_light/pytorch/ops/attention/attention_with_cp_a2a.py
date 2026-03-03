###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import lru_cache
from typing import Optional, Tuple

import torch

from transformer_light.quantize import is_aiter_available

if is_aiter_available():
    from aiter.ops.mha import flash_attn_func

from transformer_light.kernels.attention.attention_triton_impl import (
    attention_mxfp8_forward_triton_impl,
    attention_triton_backward_impl,
    attention_triton_forward_impl,
    attention_triton_mxfp8_backward_triton_impl,
    get_f8_fwd_dtype,
    is_cdna4,
)
from transformer_light.ops.attention.attention_utils import (
    block_scaling_node,
    block_scaling_node_mxfp8,
    quant_p_scale_mxfp8,
    quant_v_get_p_scale,
)


@lru_cache
def get_attention_cp_a2a_helper(b, s, h_q, h_kv, d_qk, d_v, seq_dim, n):
    attn_helper = AttentionCPA2AHelper(b, s, h_q, h_kv, d_qk, d_v, seq_dim, n)
    return attn_helper


class AttentionCPA2AHelper:
    """AttentionCPA2AHelper: a helper to transpose tensor for CP A2A"""

    def __init__(self, b, s, h_q, h_kv, d_qk, d_v, seq_dim, n):
        assert seq_dim == 1, "only_support bshd yet"
        self.seq_dim = seq_dim

        self.qkv_shape_traits = ((n, b, s, h_q, d_qk), (n, b, s, h_kv, d_qk), (n, b, s, h_kv, d_v))

        self.o_shape_traits = (n, b, s, h_q, d_v)

        self.combine_splits = (
            b * s * h_q * d_qk // n // n,
            b * s * h_kv * d_qk // n // n,
            b * s * h_kv * d_v // n // n,
        )

    def combine_qkv_before_a2a(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Combine and reshape qkv before all2all

        Args:
            q (torch.Tensor): query tensor (b, s // n, h_q, d_qk)
            k (torch.Tensor): key tensor (b, s // n, h_kv, d_qk)
            v (torch.Tensor): value tensor (b, s // n, h_kv, d_v)

        Returns:
            qkv (torch.Tensor): qkv combined tensor (n, -1)
        """
        # [b, s // n, h, d] -> [b, s // n, n, h // n, d] -> [n, b, s // n, h // n, d] -> [n, -1]
        q, k, v = (
            x.view(b, s // n, n, h // n, d).movedim(-3, 0).contiguous().view(n, -1)
            for x, (n, b, s, h, d) in zip((q, k, v), self.qkv_shape_traits)
        )

        qkv = torch.cat((q, k, v), dim=1).contiguous()
        return qkv

    def splits_qkv_after_a2a(self, qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split and reshape qkv before all2all

        Args:
            qkv (torch.Tensor): qkv tensor of local heads (n, -1)

        Returns:
            q_local_heads, k_local_heads, v_local_heads (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): (b, s, h // n, d)
        """
        q, k, v = torch.split(qkv, self.combine_splits, dim=1)
        # [n, b, s // n, h // n, d] -> [b, n, s // n, h // n, d] -> [b, s, h // n, d]
        q, k, v = (
            x.view(n, b, s // n, h // n, d).movedim(0, 1).contiguous().view(b, s, h // n, d)
            for x, (n, b, s, h, d) in zip((q, k, v), self.qkv_shape_traits)
        )
        return q, k, v

    def reshape_o_before_a2a(self, o: torch.Tensor) -> torch.Tensor:
        """Reshape output before all2all

        Args:
            o (torch.Tensor): output of local heads (b, s, h // n, d)

        Returns:
            o_reshaped (torch.Tensor): (n, b, s // n, h // n, d)
        """

        # [b, s, h // n, d] -> [b, n, s // n, h // n, d] -> [n, b, s // n, h // n, d]
        n, b, s, h, d = self.o_shape_traits
        o = o.view(b, n, s // n, h // n, d).movedim(1, 0).contiguous()
        return o

    def reshape_o_after_a2a(self, o: torch.Tensor) -> torch.Tensor:
        """Reshape output after all2all

        Args:
            o (torch.Tensor): output of local seq (n, b, s // n, h // n, d)

        Returns:
            o_reshaped (torch.Tensor): (b, s // n, h, d)
        """
        n, b, s, h, d = self.o_shape_traits
        # [n, b, s // n, h // n, d] -> [b, s // n, n, h // n, d] -> [b, s // n, h, d]
        o = o.movedim(0, -3).contiguous().view(b, s // n, h, d)

        return o

    def reshape_do_before_a2a(self, d_o: torch.Tensor) -> torch.Tensor:
        """Reshape output grad before all2all

        Args:
            d_o (torch.Tensor): output grad of local seq (b, s // n, h, d)

        Returns:
            d_o_reshaped torch.Tensor: (n, b, s // n, h // n, d)
        """
        # [b, s // n, h, d] -> [b, s // n, n , h // n, d] -> [n, b, s // n, h // n, d]
        n, b, s, h, d = self.o_shape_traits
        d_o = d_o.view(b, s // n, n, h // n, d).movedim(-3, 0).contiguous()
        return d_o

    def reshape_do_after_a2a(self, d_o: torch.Tensor) -> torch.Tensor:
        """Reshape output grad after all2all

        Args:
            d_o (torch.Tensor): output grad of local head (n, b, s // n, h // n, d)

        Returns:
            d_o_reshaped torch.Tensor: (b, s, h // n, d)
        """
        # [n, b, s // n, h // n, d] -> [b, n, s // n, h // n, d] -> [b, s, h // n, d]
        n, b, s, h, d = self.o_shape_traits
        d_o = d_o.movedim(0, 1).contiguous().view(b, s, h // n, d)
        return d_o

    def combine_dqkv_before_a2a(self, dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
        """Combine qkv tensor of local heads before a2a

        Args:
            dq (torch.Tensor): dq local heads (b, s, h // n, d)
            dk (torch.Tensor): dk local heads (b, s, h // n, d)
            dv (torch.Tensor): dv local heads (b, s, h // n, d)

        Returns:
            d_qkv torch.Tensor: dqkv of local heads (n, -1)
        """

        # [b, s, h // n, d] -> [b, n, s // n, h // n, d] -> [n, b, s // n, h // n, d] -> [n, -1]
        dq, dk, dv = (
            x.view(b, n, s // n, h // n, d).movedim(1, 0).contiguous().view(n, -1)
            for x, (n, b, s, h, d) in zip((dq, dk, dv), self.qkv_shape_traits)
        )
        dqkv = torch.cat((dq, dk, dv), dim=1).contiguous()

        return dqkv

    def split_dqkv_after_a2a(self, dqkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combine qkv tensor of local seq after a2a

        Args:
            dqkv (torch.Tensor): dqkv of local seq (n, -1)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: dq, dk, dv of local seq (b, s // n, h, d)
        """
        # [n, b, s // n, h // n, d] -> [b, s // n, n, h // n, d] -> [b, s // n, h, d]
        dq, dk, dv = torch.split(dqkv, self.combine_splits, dim=1)
        dq, dk, dv = (
            x.view(n, b, s // n, h // n, d).movedim(0, -3).contiguous().view(b, s // n, h, d)
            for x, (n, b, s, h, d) in zip((dq, dk, dv), self.qkv_shape_traits)
        )
        return dq, dk, dv


class AttentionTritonFunctionCPA2A(torch.autograd.Function):
    """
    QKV split by attention heads and a2a
    Refer the paper `DeepSpeed Ulysses <https://arxiv.org/abs/2309.14509>` for detail.
    """

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
        is_grad,
        use_fp8,
        cp_group,
    ):
        assert bias is None

        n = cp_group.size()
        b, s, h_q, d_qk = q.shape
        _, _, h_kv, d_v = v.shape
        s = s * n
        assert h_q % n == 0
        assert h_kv % n == 0
        # bshd only
        seq_dim = 1
        attn_helper = get_attention_cp_a2a_helper(b, s, h_q, h_kv, d_qk, d_v, seq_dim, n)

        qkv = attn_helper.combine_qkv_before_a2a(q, k, v)
        qkv_out = torch.empty_like(qkv)
        torch.distributed.all_to_all_single(qkv_out, qkv, group=cp_group, async_op=False)
        q_local_heads, k_local_heads, v_local_heads = attn_helper.splits_qkv_after_a2a(qkv_out)

        q_local_heads, q_scale = block_scaling_node(q_local_heads, use_fp8=use_fp8)
        k_local_heads, k_scale = block_scaling_node(k_local_heads, use_fp8=use_fp8)
        v_local_heads, v_scale, p_scale = quant_v_get_p_scale(v_local_heads, use_fp8)

        output_local_heads, softmax_lse, exp_scores = attention_triton_forward_impl(
            q_local_heads,
            k_local_heads,
            v_local_heads,
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

        # save_ctx for backward
        if is_grad:
            # q, k, v should be fp8 when set use_fp8 to True
            ctx.save_for_backward(
                q_local_heads,
                k_local_heads,
                v_local_heads,
                output_local_heads,
                softmax_lse,
                alibi_slopes,
                bias,
                q_scale,
                k_scale,
                v_scale,
            )
            ctx.sm_scale = softmax_scale
            ctx.p_scale = p_scale
            ctx.causal = causal
            ctx.use_fp8 = use_fp8
            ctx.cu_seqlens_q = torch.tensor(0, device="cuda")
            ctx.cu_seqlens_k = torch.tensor(0, device="cuda")
            ctx.max_seqlens_q = q_local_heads.shape[1]
            ctx.max_seqlens_k = k_local_heads.shape[1]
            ctx.attn_helper = attn_helper
            ctx.seq_dim = seq_dim
            ctx.cp_group = cp_group

        output_local_heads = attn_helper.reshape_o_before_a2a(output_local_heads)
        output_local_tokens = torch.empty_like(output_local_heads)
        torch.distributed.all_to_all_single(
            output_local_tokens, output_local_heads, group=cp_group, async_op=False
        )
        output_local_tokens = attn_helper.reshape_o_after_a2a(output_local_tokens)

        result = [output_local_tokens]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(exp_scores)
        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q_local_heads,
            k_local_heads,
            v_local_heads,
            output_local_heads,
            softmax_lse,
            alibi_slopes,
            bias,
            q_scale,
            k_scale,
            v_scale,
        ) = ctx.saved_tensors
        assert bias is None, "Currently bias is not supported by fa backward function."
        assert dout.dtype is torch.bfloat16, f"dout should be bfloat16 but get {dout.dtype}"
        attn_helper = ctx.attn_helper

        dout = attn_helper.reshape_do_before_a2a(dout)
        dout_local_heads = torch.empty_like(dout)
        torch.distributed.all_to_all_single(dout_local_heads, dout, group=ctx.cp_group)
        dout_local_heads = attn_helper.reshape_do_after_a2a(dout_local_heads)

        dq_local_heads, dk_local_heads, dv_local_heads = attention_triton_backward_impl(
            dout_local_heads,
            q_local_heads,
            k_local_heads,
            v_local_heads,
            output_local_heads,
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

        dqkv = attn_helper.combine_dqkv_before_a2a(dq_local_heads, dk_local_heads, dv_local_heads)
        dqkv_out = torch.empty_like(dqkv)
        torch.distributed.all_to_all_single(dqkv_out, dqkv, group=ctx.cp_group)
        dq_local_tokens, dk_local_tokens, dv_local_tokens = attn_helper.split_dqkv_after_a2a(dqkv_out)

        return (
            dq_local_tokens,
            dk_local_tokens,
            dv_local_tokens,
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


class AttentionTritonMXFP8FunctionCPA2A(torch.autograd.Function):
    """
    QKV split by attention heads and a2a
    Refer the paper `DeepSpeed Ulysses <https://arxiv.org/abs/2309.14509>` for detail.
    """

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
        is_grad,
        use_mxfp8,
        cp_group,
        block_m_fwd: int = 64,  # block of query seq len in fwd
        block_n_fwd: int = 64,  # block of key/value seq len in fwd
        block_m_dq_bwd: int = 64,  # block of dq seq len in bwd
        block_n_dq_bwd: int = 64,  # block of dq seq len in bwd
        block_m_dkv_bwd: int = 64,  # block of dkv seq len in bwd
        block_n_dkv_bwd: int = 64,  # block of dkv seq len in bwd
        quant_block_size: int = 32,
    ):
        assert is_cdna4(), "mxfp8 is only supported by gfx950 and newer version"
        assert bias is None

        n = cp_group.size()
        b, s, h_q, d_qk = q.shape
        _, _, h_kv, d_v = v.shape
        s = s * n
        assert h_q % n == 0
        assert h_kv % n == 0
        # bshd only
        seq_dim = 1
        attn_helper = get_attention_cp_a2a_helper(b, s, h_q, h_kv, d_qk, d_v, seq_dim, n)

        qkv = attn_helper.combine_qkv_before_a2a(q, k, v)
        qkv_out = torch.empty_like(qkv)
        torch.distributed.all_to_all_single(qkv_out, qkv, group=cp_group, async_op=False)
        q_local_heads, k_local_heads, v_local_heads = attn_helper.splits_qkv_after_a2a(qkv_out)

        if use_mxfp8:
            q_local_heads, q_scale = block_scaling_node_mxfp8(
                q_local_heads,
                quant_block_size,
                "bshd",
                is_2d_block=True,
                float8_dtype_pt=get_f8_fwd_dtype(),
                cu_seqlens=0,
                max_seqlens=q.shape[1],
            )
            k_local_heads, k_scale = block_scaling_node_mxfp8(
                k_local_heads,
                quant_block_size,
                "bshd",
                is_2d_block=True,
                float8_dtype_pt=get_f8_fwd_dtype(),
                cu_seqlens=0,
                max_seqlens=k.shape[1],
            )
            v_local_heads, v_scale = block_scaling_node_mxfp8(
                v_local_heads,
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

        output_local_heads, softmax_lse, exp_scores = attention_mxfp8_forward_triton_impl(
            q=q_local_heads,
            k=k_local_heads,
            v=v_local_heads,
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

        # save_ctx for backward
        if is_grad:
            # q, k, v should be mxfp8 when set use_fp8 to True
            ctx.save_for_backward(
                q_local_heads,
                k_local_heads,
                v_local_heads,
                output_local_heads,
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
            ctx.max_seqlens_q = q_local_heads.shape[1]
            ctx.max_seqlens_k = k_local_heads.shape[1]

            ctx.attn_helper = attn_helper
            ctx.seq_dim = seq_dim
            ctx.cp_group = cp_group

        output_local_heads = attn_helper.reshape_o_before_a2a(output_local_heads)
        output_local_tokens = torch.empty_like(output_local_heads)
        torch.distributed.all_to_all_single(
            output_local_tokens, output_local_heads, group=cp_group, async_op=False
        )
        output_local_tokens = attn_helper.reshape_o_after_a2a(output_local_tokens)

        result = [output_local_tokens]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(exp_scores)
        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q_local_heads,
            k_local_heads,
            v_local_heads,
            output_local_heads,
            softmax_lse,
            alibi_slopes,
            bias,
            q_scale,
            k_scale,
            v_scale,
        ) = ctx.saved_tensors
        assert is_cdna4(), "mxfp8 is only supported by gfx950 and newer version"
        assert bias is None, "Currently bias is not supported by fa backward function."
        assert dout.dtype is torch.bfloat16, f"dout should be bfloat16 but get {dout.dtype}"
        attn_helper = ctx.attn_helper

        dout = attn_helper.reshape_do_before_a2a(dout)
        dout_local_heads = torch.empty_like(dout)
        torch.distributed.all_to_all_single(dout_local_heads, dout, group=ctx.cp_group)
        dout_local_heads = attn_helper.reshape_do_after_a2a(dout_local_heads)

        dq_local_heads, dk_local_heads, dv_local_heads = attention_triton_mxfp8_backward_triton_impl(
            do=dout_local_heads,
            q=q_local_heads,
            k=k_local_heads,
            v=v_local_heads,
            o=output_local_heads,
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

        dqkv = attn_helper.combine_dqkv_before_a2a(dq_local_heads, dk_local_heads, dv_local_heads)
        dqkv_out = torch.empty_like(dqkv)
        torch.distributed.all_to_all_single(dqkv_out, dqkv, group=ctx.cp_group)
        dq_local_tokens, dk_local_tokens, dv_local_tokens = attn_helper.split_dqkv_after_a2a(dqkv_out)

        return (
            dq_local_tokens,
            dk_local_tokens,
            dv_local_tokens,
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
            None,
        )


class _A2ASingle(torch.autograd.Function):
    """Differentiable all-to-all-single communication wrapper.

    The backward of an all-to-all with equal splits is another all-to-all,
    so this makes the a2a communication transparent to autograd.
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        out = torch.empty_like(x)
        torch.distributed.all_to_all_single(out, x, group=group, async_op=False)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.empty_like(grad_output)
        torch.distributed.all_to_all_single(grad_input, grad_output, group=ctx.group, async_op=False)
        return grad_input, None


class AttentionAiterFunctionCPA2A:
    """
    QKV split by attention heads and a2a, backed by flash_attn_func from aiter.
    Refer the paper `DeepSpeed Ulysses <https://arxiv.org/abs/2309.14509>` for detail.

    Unlike the CK variant this is **not** a torch.autograd.Function.  Instead it
    composes differentiable building blocks (``_A2ASingle`` for communication and
    ``flash_attn_func`` for the attention kernel) so that autograd handles the
    backward pass automatically.
    """

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
        cp_group,
    ):
        assert bias is None

        n = cp_group.size()
        b, s, h_q, d_qk = q.shape
        _, _, h_kv, d_v = v.shape
        s = s * n
        assert h_q % n == 0
        assert h_kv % n == 0
        # bshd only
        seq_dim = 1
        attn_helper = get_attention_cp_a2a_helper(b, s, h_q, h_kv, d_qk, d_v, seq_dim, n)

        # Combine Q, K, V and all-to-all: local tokens -> local heads
        qkv = attn_helper.combine_qkv_before_a2a(q, k, v)
        qkv_out = _A2ASingle.apply(qkv, cp_group)
        q_local_heads, k_local_heads, v_local_heads = attn_helper.splits_qkv_after_a2a(qkv_out)

        # Call flash_attn_func from aiter (always request lse for backward support)
        _return_softmax = return_softmax and dropout_p > 0
        attn_result = flash_attn_func(
            q_local_heads,
            k_local_heads,
            v_local_heads,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            bias=bias,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_lse=True,
            return_attn_probs=_return_softmax,
        )

        # Parse flash_attn_func result
        if _return_softmax:
            output_local_heads, softmax_lse, S_dmask = attn_result
        else:
            output_local_heads, softmax_lse = attn_result
            S_dmask = None

        # All-to-all output: local heads -> local tokens
        output_local_heads = attn_helper.reshape_o_before_a2a(output_local_heads)
        output_local_tokens = _A2ASingle.apply(output_local_heads, cp_group)
        output_local_tokens = attn_helper.reshape_o_after_a2a(output_local_tokens)

        # Build result
        result = [output_local_tokens]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return result[0] if len(result) == 1 else tuple(result)
