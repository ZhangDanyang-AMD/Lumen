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

AITER packed ID format
~~~~~~~~~~~~~~~~~~~~~~
``moe_sorting_fwd`` internally encodes ``sorted_token_ids`` as::

    packed_id = (topk_slot << 24) | token_id

with padding sentinel ``(k << 24) | num_tokens``.  The padded total
(``num_valid_ids[0]``) is ``num_experts * block_size``, not ``num_tokens * k``.

``fused_permute`` post-processes this into **token-major flat indices**::

    flat_idx = token_id * k + slot_id       range [0, num_tokens * k)

so that ``sorted_ids // k == token_id`` and ``sorted_ids % k == slot_id``,
and the output length is exactly ``num_tokens * k`` (compatible with
``fused_unpermute``).

**Caveat — duplicate-expert dedup**: when both topk slots for a token
select the same expert, AITER keeps only one assignment.  The missing
slot is filled with a zero-weight identity entry so the pipeline still
works, but the weighted sum for that token will be approximate.
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


# ── AITER packed-ID decode ──────────────────────────────────────────────────


def decode_aiter_sorted_ids(
    raw_sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    num_tokens: int,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode AITER ``moe_sorting_fwd`` packed IDs.

    AITER encodes ``(topk_slot << 24) | token_id`` with padding
    sentinel ``(k << 24) | num_tokens``.

    Args:
        raw_sorted_ids: Raw sorted_ids from ``moe_sorting_fwd``.
        num_valid_ids: ``num_valid_ids`` tensor (index 0 = padded total).
        num_tokens: Original token count.
        k: Top-k value.

    Returns:
        token_ids: ``[N]`` int32 token indices ``(0 .. num_tokens-1)``.
        slot_ids:  ``[N]`` int32 topk slot indices ``(0 .. k-1)``.
        valid_mask: ``[total_padded]`` bool — ``True`` for non-padding.
    """
    total_padded = num_valid_ids[0].item()
    raw = raw_sorted_ids[:total_padded]
    sentinel = (k << 24) | num_tokens
    valid_mask = raw != sentinel
    valid = raw[valid_mask]
    token_ids = (valid & 0xFFFFFF).to(torch.int32)
    slot_ids = (valid >> 24).to(torch.int32)
    return token_ids, slot_ids, valid_mask


def _build_flat_sort_order(
    token_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    valid_weights: torch.Tensor,
    num_tokens: int,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a ``[num_tokens * k]`` flat sort order from decoded AITER IDs.

    Token-major encoding: ``flat_idx = token_id * k + slot_id`` so that
    ``flat_idx // k == token_id`` and ``flat_idx % k == slot_id``.

    Missing entries (from AITER duplicate-expert dedup) are filled with
    identity indices and zero weights.
    """
    device = token_ids.device
    n_total = num_tokens * k

    flat_indices = token_ids.long() * k + slot_ids.long()

    present = torch.zeros(n_total, dtype=torch.bool, device=device)
    present[flat_indices] = True
    missing = torch.where(~present)[0]

    full_order = torch.empty(n_total, dtype=torch.int32, device=device)
    full_weights = torch.zeros(n_total, dtype=torch.float32, device=device)

    n_valid = flat_indices.shape[0]
    full_order[:n_valid] = flat_indices.to(torch.int32)
    full_weights[:n_valid] = valid_weights

    if missing.numel() > 0:
        full_order[n_valid:] = missing.to(torch.int32)

    return full_order, full_weights


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

    The raw AITER packed output is decoded into **token-major flat
    indices** of length ``num_tokens * k``::

        sorted_ids[i] = token_id * k + slot_id

    Use ``sorted_ids // k`` to recover token indices and
    ``sorted_ids % k`` to recover topk-slot indices.

    Args:
        tokens: Input tokens ``[num_tokens, hidden_size]``.
        indices: Expert assignments ``[num_tokens, k]``, int64.
        weights: Expert weights ``[num_tokens, k]``, float.
        num_experts: Total number of experts.
        block_size: Padding block size (default 32).

    Returns:
        ``(sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
        moe_buf)`` where ``sorted_ids`` and ``sorted_weights`` have
        length ``num_tokens * k`` in token-major flat encoding.
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
    raw_sorted_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device)
    raw_sorted_weights = torch.empty(max_num_tokens_padded, dtype=torch.float32, device=device)
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    moe_buf = torch.empty((num_tokens, hidden_size), dtype=tokens.dtype, device=device)

    moe_sorting_fwd(
        indices.to(torch.int32),
        weights.float(),
        raw_sorted_ids,
        raw_sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        num_experts,
        block_size,
    )
    torch.cuda.synchronize()

    token_ids, slot_ids, valid_mask = decode_aiter_sorted_ids(
        raw_sorted_ids,
        num_valid_ids,
        num_tokens,
        topk,
    )
    total_padded = num_valid_ids[0].item()
    valid_weights = raw_sorted_weights[:total_padded][valid_mask]

    sorted_ids, sorted_weights = _build_flat_sort_order(
        token_ids,
        slot_ids,
        valid_weights,
        num_tokens,
        topk,
    )

    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


# ── fused_unpermute ─────────────────────────────────────────────────────────


def fused_unpermute(
    expert_output: torch.Tensor,
    sort_order: torch.Tensor,
    num_tokens: int,
    k: int,
) -> torch.Tensor:
    """Reverse permute: scatter expert outputs back to original token order (AITER ASM).

    Accepts ``sort_order`` in the token-major flat encoding produced by
    :func:`fused_permute` (``flat_idx = token_id * k + slot_id``).
    Both ``expert_output`` and ``sort_order`` must have exactly
    ``num_tokens * k`` entries.

    Args:
        expert_output: ``[num_tokens * k, hidden_size]`` expert-sorted
            output, with rows ordered to match ``sort_order``.
        sort_order: ``[num_tokens * k]`` flat indices from
            :func:`fused_permute`.
        num_tokens: Original number of tokens.
        k: Number of experts per token.

    Returns:
        Reconstructed output ``[num_tokens, hidden_size]``.
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
