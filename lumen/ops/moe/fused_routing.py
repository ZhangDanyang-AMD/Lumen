###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Fused MoE token routing operations — AITER backend only.

Provides fused top-k with softmax, token-to-expert permute, and reverse
unpermute.  Dispatches to AITER ASM/HIP kernels.  No PyTorch fallback;
AITER must be available.

AITER kernel mapping:
  fused_topk      -> aiter.topk_softmax      (ASM, fused softmax + top-k + renorm)
  fused_permute   -> aiter.moe_sorting_fwd   (HIP, fused sort + pad + offset)
  fused_unpermute -> aiter.moe_sum           (ASM, fused scatter-back + weighted sum)
"""

import functools
from typing import Tuple

import torch

from lumen.ops.dispatch import (
    _probe_aiter_moe_sorting,
    _probe_aiter_moe_sum,
    _probe_aiter_moe_topk_softmax,
)

BLOCK_SIZE_M = 32


# ── Lazy AITER getters ──────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def _get_aiter_topk_softmax():
    from aiter.ops.moe_op import topk_softmax

    return topk_softmax


@functools.lru_cache(maxsize=1)
def _get_aiter_moe_sorting_fwd():
    from aiter.ops.moe_sorting import moe_sorting_fwd

    return moe_sorting_fwd


@functools.lru_cache(maxsize=1)
def _get_aiter_moe_sum():
    from aiter.ops.moe_op import moe_sum

    return moe_sum


# ── fused_topk ──────────────────────────────────────────────────────────────


def fused_topk(
    logits: torch.Tensor,
    k: int,
    softmax_first: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused top-k with softmax for MoE gating (AITER ASM).

    Args:
        logits: Gate logits [num_tokens, num_experts].
        k: Number of top experts per token.
        softmax_first: If True (default), apply softmax before top-k and
            renormalize selected weights to sum to 1.  If False, top-k
            selects on raw logit values; weights are NOT renormalized.

    Returns:
        Tuple of (weights, indices) where:
        - weights: [num_tokens, k] expert weights (renormalized when
          softmax_first=True, raw otherwise)
        - indices: [num_tokens, k] selected expert indices
    """
    assert _probe_aiter_moe_topk_softmax() and logits.is_cuda, (
        "AITER topk_softmax ASM kernel is required but not available. "
        "Ensure AITER is installed with MoE support and input is on CUDA."
    )

    topk_softmax = _get_aiter_topk_softmax()
    num_tokens = logits.shape[0]
    topk_weights = torch.empty(num_tokens, k, dtype=torch.float32, device=logits.device)
    topk_indices = torch.empty(num_tokens, k, dtype=torch.int32, device=logits.device)
    token_expert_indices = torch.empty(num_tokens, k, dtype=torch.int32, device=logits.device)
    # AITER's 5th param is `need_renorm`: when True, softmax is applied
    # before top-k and weights are renormalized — same semantics as our
    # `softmax_first` flag.
    topk_softmax(topk_weights, topk_indices, token_expert_indices, logits.float(), softmax_first)
    torch.cuda.synchronize()
    return topk_weights, topk_indices.to(torch.int64)


# ── fused_permute ───────────────────────────────────────────────────────────


def fused_permute(
    tokens: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    num_experts: int,
    block_size: int = BLOCK_SIZE_M,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused token-to-expert permute (AITER HIP).

    Routes tokens to their assigned experts, sorted by expert id for
    coalesced memory access in the subsequent grouped GEMM.

    Returns the 5-tuple from ``moe_sorting_fwd``:
      (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf)
    which is the native format consumed by AITER's fused MoE GEMM kernels.

    Args:
        tokens: Input tokens [num_tokens, hidden_size].
        indices: Expert assignments [num_tokens, k], int64.
        weights: Expert weights [num_tokens, k], float.
        num_experts: Total number of experts.
        block_size: Padding block size (default 32).

    Returns:
        (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf)
    """
    assert _probe_aiter_moe_sorting() and tokens.is_cuda, (
        "AITER moe_sorting_fwd HIP kernel is required but not available. "
        "Ensure AITER is installed with MoE support and input is on CUDA."
    )

    moe_sorting_fwd = _get_aiter_moe_sorting_fwd()
    num_tokens, topk = indices.shape
    hidden_size = tokens.shape[-1]
    max_num_tokens_padded = int(indices.numel() + num_experts * block_size - topk)
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)

    device = tokens.device
    sorted_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(max_num_tokens_padded, dtype=torch.float32, device=device)
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    moe_buf = torch.empty((num_tokens, hidden_size), dtype=tokens.dtype, device=device)

    moe_sorting_fwd(
        indices.to(torch.int32),
        weights.float(),
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        num_experts,
        block_size,
    )
    torch.cuda.synchronize()
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


# ── fused_unpermute ─────────────────────────────────────────────────────────


def fused_unpermute(
    expert_output: torch.Tensor,
    sort_order: torch.Tensor,
    num_tokens: int,
    k: int,
) -> torch.Tensor:
    """Reverse permute: scatter expert outputs back to original token order (AITER ASM).

    Args:
        expert_output: [num_tokens * k, hidden_size] expert-sorted output.
        sort_order: [num_tokens * k] sorting indices from fused_permute.
        num_tokens: Original number of tokens.
        k: Number of experts per token.

    Returns:
        Reconstructed output [num_tokens, hidden_size].
    """
    assert _probe_aiter_moe_sum() and expert_output.is_cuda, (
        "AITER moe_sum ASM kernel is required but not available. "
        "Ensure AITER is installed with MoE support and input is on CUDA."
    )

    moe_sum = _get_aiter_moe_sum()
    hidden_size = expert_output.shape[-1]

    # The unsort+reshape is cheap PyTorch index ops; moe_sum does the
    # heavy weighted reduction in AITER ASM.  A fused Triton unsort
    # would save one global-memory pass but is not justified given the
    # small cost relative to the reduction itself.
    unsort_order = torch.argsort(sort_order)
    unsorted = expert_output[unsort_order]
    input_3d = unsorted.reshape(num_tokens, k, hidden_size).contiguous()

    output = torch.empty(num_tokens, hidden_size, dtype=expert_output.dtype, device=expert_output.device)
    moe_sum(input_3d, output)
    torch.cuda.synchronize()
    return output
