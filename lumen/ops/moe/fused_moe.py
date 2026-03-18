###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""End-to-end fused MoE using AITER Triton kernels.

Combines token sorting (moe_align_block_size_triton) and fused GEMM
(fused_moe) into a single high-level API.  Supports BF16 and FP8
quantized expert weights.

AITER kernel mapping:
  token alignment -> moe_align_block_size_triton  (Triton, 4-stage sort+pad)
  fused GEMM      -> fused_moe                    (Triton, sort+GEMM+weight)
"""

import functools
from typing import Dict, List, Optional

import torch
import triton.language as tl

from lumen.ops.dispatch import (
    _probe_aiter_triton_fused_moe,
    _probe_aiter_triton_moe_align,
)

_DEFAULT_MOE_CONFIG: Dict[str, int] = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
}


# ── Lazy AITER getters ──────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def _get_aiter_moe_align_block_size():
    from aiter.ops.triton.moe.moe_align_block_size import moe_align_block_size_triton

    return moe_align_block_size_triton


@functools.lru_cache(maxsize=1)
def _get_aiter_fused_moe():
    from aiter.ops.triton.moe.moe_op import fused_moe

    return fused_moe


# ── Helpers ─────────────────────────────────────────────────────────────────


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align_tokens(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
) -> tuple:
    """Sort and align tokens by expert assignment with block-size padding.

    Returns:
        (sorted_token_ids, expert_ids, num_tokens_post_pad)
    """
    moe_align = _get_aiter_moe_align_block_size()
    numel = topk_ids.numel()
    max_num_tokens_padded = numel + num_experts * (block_size - 1)
    max_num_m_blocks = _ceil_div(max_num_tokens_padded, block_size)

    device = topk_ids.device
    sorted_token_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device)
    expert_ids = torch.empty(max_num_m_blocks, dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=device)

    moe_align(
        topk_ids.to(torch.int32),
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
    )
    torch.cuda.synchronize()
    return sorted_token_ids, expert_ids, num_tokens_post_pad


# ── Public API ──────────────────────────────────────────────────────────────


def fused_moe_triton(
    hidden_states: torch.Tensor,
    expert_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    k: int,
    block_size: int = 128,
    mul_routed_weight: bool = True,
    expert_weights_scale: Optional[torch.Tensor] = None,
    activation_scale: Optional[torch.Tensor] = None,
    use_fp8: bool = False,
    block_shape: Optional[List[int]] = None,
    config: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    """End-to-end fused MoE using AITER Triton kernels.

    Fuses token sorting + GEMM + weight multiplication in a single pass.
    Requires AITER Triton MoE kernels.

    The kernel computes ``hidden_states @ expert_weights[e].T`` per expert,
    so ``expert_weights`` must be stored in **[E, N, K]** layout (AITER
    convention) where N = intermediate_dim and K = hidden_dim.

    Args:
        hidden_states: Input activations [num_tokens, hidden_dim].
        expert_weights: Expert weight matrices
            [num_experts, intermediate_dim, hidden_dim].
        topk_ids: Top-k expert IDs per token [num_tokens, k], int32/int64.
        topk_weights: Routing weights [num_tokens, k], float32.
        num_experts: Total number of experts.
        k: Number of experts per token.
        block_size: Block size for alignment padding (default 128).
        mul_routed_weight: Multiply output by routing weights.
        expert_weights_scale: Scale for expert weights in FP8 mode.
        activation_scale: Scale for activations in FP8 mode.
        use_fp8: Use FP8 quantization.
        block_shape: Block shape [block_n, block_k] for grouped quantization.
        config: Kernel tuning parameters (BLOCK_SIZE_M/N/K, GROUP_SIZE_M).

    Returns:
        Output tensor [num_tokens, k, intermediate_dim].
    """
    assert _probe_aiter_triton_moe_align(), "AITER Triton moe_align_block_size is required but not available."
    assert _probe_aiter_triton_fused_moe(), "AITER Triton fused_moe is required but not available."
    assert hidden_states.is_cuda, "fused_moe_triton requires CUDA tensors"

    fused_moe_fn = _get_aiter_fused_moe()
    moe_config = config if config is not None else _DEFAULT_MOE_CONFIG

    sorted_token_ids, expert_ids, num_tokens_post_pad = _align_tokens(
        topk_ids,
        num_experts,
        block_size,
    )

    num_tokens = hidden_states.shape[0]
    intermediate_dim = expert_weights.shape[1]
    C = torch.empty(
        (num_tokens, k, intermediate_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    fused_moe_fn(
        hidden_states,
        expert_weights,
        C,
        activation_scale,
        expert_weights_scale,
        None,  # B_zp
        topk_weights,
        topk_ids.to(torch.int32),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        mul_routed_weight,
        k,
        tl.float16 if hidden_states.dtype == torch.float16 else tl.bfloat16,
        use_fp8,
        False,  # use_int8_w8a16
        False,  # use_int4_w4a16
        block_shape=block_shape,
        config=moe_config,
    )
    torch.cuda.synchronize()
    return C
