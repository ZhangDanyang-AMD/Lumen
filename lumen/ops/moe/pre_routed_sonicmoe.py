###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Qwen3 pre-routed BF16 expert MLP without patching AITER.

Tokens are already expert-major and every router score is one. The forward
fuses the up-projection with SwiGLU, and the backward skips the identity
gather and the ``dout * 1`` rounding trip.
"""

from __future__ import annotations

import os

import torch

from lumen.ops.moe.sonicmoe_config import install_sonicmoe_gfx950_config


def _cu_seqlens(expert_frequency: torch.Tensor) -> torch.Tensor:
    counts = expert_frequency.to(dtype=torch.int32)
    zeros = torch.zeros(1, dtype=torch.int32, device=counts.device)
    return torch.cat((zeros, counts.cumsum(dim=0, dtype=torch.int32)))


def _use_fused_swiglu(x: torch.Tensor) -> bool:
    return (
        os.environ.get("LUMEN_EXPERT_FUSED_SWIGLU", "0") == "1"
        and os.environ.get("SONIC_MOE_GROUPED_GEMM_BACKEND", "triton") == "triton"
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
    )


class _PreRoutedSwiGLUExperts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, expert_frequency):
        install_sonicmoe_gfx950_config()
        from aiter.ops.triton.moe.sonicmoe import _router_forward, grouped_gemm

        if expert_frequency.dim() != 1:
            raise ValueError("expert_frequency must be one-dimensional")
        tokens, hidden = x.shape
        experts = expert_frequency.shape[0]
        if w1.shape[0] != experts or w1.shape[1] != hidden:
            raise ValueError(
                f"w1 must be [{experts}, {hidden}, 2I], got {tuple(w1.shape)}"
            )
        intermediate_full = w1.shape[2]
        if intermediate_full % 2:
            raise ValueError("SwiGLU w1 must have an even last dimension")
        intermediate = intermediate_full // 2
        if tuple(w2.shape) != (experts, intermediate, hidden):
            raise ValueError(
                f"w2 must be [{experts}, {intermediate}, {hidden}], got {tuple(w2.shape)}"
            )

        cu_seqlens = _cu_seqlens(expert_frequency)
        hidden_up = torch.empty(
            tokens, intermediate_full, dtype=x.dtype, device=x.device
        )
        if _use_fused_swiglu(x):
            from lumen.ops.moe.sonicmoe_swiglu import grouped_gemm_swiglu

            activated = torch.empty(tokens, intermediate, dtype=x.dtype, device=x.device)
            grouped_gemm_swiglu(x, w1, cu_seqlens, hidden_up, activated)
        else:
            from aiter.ops.triton.moe.sonicmoe import activation_fwd

            grouped_gemm(x, w1, cu_seqlens, out=hidden_up)
            activated = activation_fwd(hidden_up, intermediate, "swiglu", False)

        projected = torch.empty(tokens, hidden, dtype=x.dtype, device=x.device)
        grouped_gemm(activated, w2, cu_seqlens, out=projected)
        output = torch.empty_like(projected)
        identity = torch.arange(tokens, dtype=torch.int32, device=x.device)
        scores = torch.ones(tokens, dtype=torch.float32, device=x.device)
        _router_forward(
            y=projected,
            o=output,
            topk_scores=scores,
            s_reverse_scatter_idx=identity,
            num_activated_expert_per_token_offset=None,
            varlen_K_max=1,
            H=hidden,
            is_varlen_K=False,
        )

        ctx.intermediate = intermediate
        ctx.fused = _use_fused_swiglu(x)
        ctx.save_for_backward(x, w1, w2, hidden_up, cu_seqlens, scores)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        from aiter.ops.triton.moe.sonicmoe import (
            _up_projection_backward_act,
            activation_bwd,
            activation_fwd,
            grouped_gemm,
        )

        x, w1, w2, hidden_up, cu_seqlens, scores = ctx.saved_tensors
        tokens = x.shape[0]
        intermediate = ctx.intermediate
        grad_hidden = torch.empty_like(hidden_up)
        activated = torch.empty(
            tokens, intermediate, dtype=hidden_up.dtype, device=hidden_up.device
        )
        grad_activated = torch.empty_like(activated)
        if ctx.fused and grad_output.is_contiguous():
            from lumen.ops.moe.sonicmoe_swiglu import grouped_gemm_swiglu_bwd

            grouped_gemm_swiglu_bwd(
                grad_output,
                w2,
                cu_seqlens,
                hidden_up,
                scores,
                grad_activated,
                activated,
                grad_hidden,
            )
        else:
            grouped_gemm(
                grad_output,
                w2,
                cu_seqlens,
                out=grad_activated,
                B_is_transposed=True,
            )
            activated_value = activation_fwd(hidden_up, intermediate, "swiglu", False)
            activated.copy_(activated_value)
            grad_hidden.copy_(
                activation_bwd(hidden_up, grad_activated, intermediate, "swiglu", False)
            )

        grad_w2 = torch.empty_like(w2)
        grouped_gemm(
            activated,
            grad_output if grad_output.is_contiguous() else grad_output.contiguous(),
            cu_seqlens,
            out=grad_w2,
            A_is_transposed=True,
        )
        grad_input = torch.empty_like(x)
        _up_projection_backward_act(
            w1=w1,
            dx_expanded=grad_input,
            dh=grad_hidden,
            db1=None,
            expert_frequency_offset=cu_seqlens,
            is_glu_activation=True,
            concat_layout=False,
            grouped_weight_layout=True,
        )
        grad_w1 = torch.empty_like(w1)
        grouped_gemm(
            x,
            grad_hidden,
            cu_seqlens,
            out=grad_w1,
            A_is_transposed=True,
        )
        return grad_input, grad_w1, grad_w2, None


def pre_routed_swiglu_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    expert_frequency: torch.Tensor,
) -> torch.Tensor:
    """Run one pre-routed BF16 SwiGLU expert MLP."""
    return _PreRoutedSwiGLUExperts.apply(hidden_states, w1, w2, expert_frequency)
