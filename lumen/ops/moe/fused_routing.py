###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Fused MoE token routing operations.

Provides fused top-k with softmax, token-to-expert permute, and reverse
unpermute. Dispatches to AITER Triton kernels when available.
"""

import functools
import logging
from typing import Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _probe_aiter_moe_sorting():
    """Check if AITER MoE sorting/permute kernels are available."""
    try:
        from aiter.ops.triton.moe import moe_sorting as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


def fused_topk(
    logits: torch.Tensor,
    k: int,
    softmax_first: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused top-k with softmax for MoE gating.

    Args:
        logits: Gate logits [num_tokens, num_experts].
        k: Number of top experts per token.
        softmax_first: If True, apply softmax before top-k.

    Returns:
        Tuple of (weights, indices) where:
        - weights: [num_tokens, k] softmax-normalized weights
        - indices: [num_tokens, k] selected expert indices
    """
    if softmax_first:
        probs = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, k, dim=-1)
        # Renormalize weights to sum to 1
        weights = weights / weights.sum(dim=-1, keepdim=True)
    else:
        weights, indices = torch.topk(logits, k, dim=-1)
        # Apply softmax to selected logits
        weights = F.softmax(weights, dim=-1)

    return weights, indices


def fused_permute(
    tokens: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused token-to-expert permute.

    Routes tokens to their assigned experts.

    Args:
        tokens: Input tokens [num_tokens, hidden_size].
        indices: Expert assignments [num_tokens, k].
        weights: Expert weights [num_tokens, k].
        num_experts: Total number of experts.

    Returns:
        Tuple of (permuted_tokens, sorted_indices, expert_offsets) where:
        - permuted_tokens: [num_tokens * k, hidden_size] sorted by expert
        - sorted_indices: [num_tokens * k] sort order for unpermute
        - expert_offsets: [num_experts + 1] cumulative token counts per expert
    """
    num_tokens, k = indices.shape

    # Flatten indices and weights
    flat_indices = indices.reshape(-1)  # [num_tokens * k]
    flat_token_ids = torch.arange(num_tokens, device=tokens.device).unsqueeze(1).expand(-1, k).reshape(-1)
    flat_weights = weights.reshape(-1)  # [num_tokens * k]

    # Sort by expert index for coalesced access
    sort_order = torch.argsort(flat_indices, stable=True)
    sorted_expert_ids = flat_indices[sort_order]
    sorted_token_ids = flat_token_ids[sort_order]
    sorted_weights = flat_weights[sort_order]

    # Gather tokens in expert-sorted order
    permuted_tokens = tokens[sorted_token_ids] * sorted_weights.unsqueeze(-1)

    # Compute expert offsets
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=tokens.device)
    ones = torch.ones_like(sorted_expert_ids)
    expert_offsets.scatter_add_(0, sorted_expert_ids + 1, ones)
    expert_offsets = expert_offsets.cumsum(0)

    return permuted_tokens, sort_order, expert_offsets


def fused_unpermute(
    expert_output: torch.Tensor,
    sort_order: torch.Tensor,
    num_tokens: int,
    k: int,
) -> torch.Tensor:
    """Reverse permute: scatter expert outputs back to original token order.

    Args:
        expert_output: [num_tokens * k, hidden_size] expert-sorted output.
        sort_order: [num_tokens * k] sorting indices from fused_permute.
        num_tokens: Original number of tokens.
        k: Number of experts per token.

    Returns:
        Reconstructed output [num_tokens, hidden_size].
    """
    hidden_size = expert_output.shape[-1]

    # Unsort back to original order
    unsort_order = torch.argsort(sort_order)
    unsorted = expert_output[unsort_order]  # [num_tokens * k, hidden_size]

    # Sum contributions from k experts per token
    result = unsorted.reshape(num_tokens, k, hidden_size).sum(dim=1)

    return result
