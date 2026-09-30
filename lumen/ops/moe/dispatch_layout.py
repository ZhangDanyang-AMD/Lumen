###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Tensor layout helpers for expert-major MoE dispatch."""

import functools
from collections.abc import Sequence
from typing import Literal

import torch


def transpose_variable_chunks(
    tensor: torch.Tensor,
    counts: torch.Tensor | Sequence[Sequence[int]],
    *,
    source_layout: Literal["sender_major", "expert_major"],
    fused: bool = False,
) -> torch.Tensor:
    """Transpose variable token chunks between sender-major and expert-major.

    ``counts[sender][expert]`` describes each chunk length. Sender-major input
    is laid out as ``[s0e0, s0e1, ..., s1e0, ...]``; expert-major input uses
    ``[e0s0, e0s1, ..., e1s0, ...]``. The operation is differentiable with
    respect to ``tensor``.
    """
    count_tensor = counts if isinstance(counts, torch.Tensor) else None
    if count_tensor is not None:
        if count_tensor.dim() != 2:
            raise ValueError("counts must be a sender-by-expert matrix")
        num_senders, num_experts = count_tensor.shape
        count_rows = None
    else:
        count_rows = counts
        num_senders = len(count_rows)
        num_experts = len(count_rows[0]) if num_senders else 0
    if num_senders == 0:
        return tensor
    if count_rows is not None and any(
        len(row) != num_experts for row in count_rows
    ):
        raise ValueError("counts must be a rectangular sender-by-expert matrix")
    if num_senders == 1 or num_experts == 1:
        return tensor

    if source_layout == "sender_major":
        input_pairs = [
            (sender, expert)
            for sender in range(num_senders)
            for expert in range(num_experts)
        ]
        output_pairs = [
            (sender, expert)
            for expert in range(num_experts)
            for sender in range(num_senders)
        ]
    elif source_layout == "expert_major":
        input_pairs = [
            (sender, expert)
            for expert in range(num_experts)
            for sender in range(num_senders)
        ]
        output_pairs = [
            (sender, expert)
            for sender in range(num_senders)
            for expert in range(num_experts)
        ]
    else:
        raise ValueError(f"Unsupported source_layout: {source_layout}")

    if fused and tensor.is_cuda:
        try:
            from transformer_engine.pytorch.permutation import (
                moe_sort_chunks_by_index,
            )
        except ImportError:
            moe_sort_chunks_by_index = None
        if moe_sort_chunks_by_index is not None:
            if count_tensor is None:
                count_tensor = torch.tensor(
                    count_rows,
                    dtype=torch.int32,
                    device=tensor.device,
                )
            else:
                count_tensor = count_tensor.to(device=tensor.device, dtype=torch.int32)
            if source_layout == "sender_major":
                split_sizes_tensor = count_tensor.reshape(-1)
                sorted_indices = (
                    torch.arange(
                        num_senders * num_experts,
                        dtype=torch.int32,
                        device=tensor.device,
                    )
                    .view(num_senders, num_experts)
                    .T.reshape(-1)
                )
            else:
                split_sizes_tensor = count_tensor.T.contiguous().reshape(-1)
                sorted_indices = (
                    torch.arange(
                        num_senders * num_experts,
                        dtype=torch.int32,
                        device=tensor.device,
                    )
                    .view(num_experts, num_senders)
                    .T.reshape(-1)
                )
            return moe_sort_chunks_by_index(
                tensor,
                split_sizes_tensor,
                sorted_indices,
            )

    if count_rows is None:
        count_rows = count_tensor.tolist()
    split_sizes = [
        int(count_rows[sender][expert]) for sender, expert in input_pairs
    ]
    if sum(split_sizes) != tensor.shape[0]:
        raise ValueError(
            f"Chunk counts sum to {sum(split_sizes)}, expected {tensor.shape[0]}"
        )
    chunks = tensor.split(split_sizes, dim=0)
    chunk_by_pair = dict(zip(input_pairs, chunks))
    return torch.cat([chunk_by_pair[pair] for pair in output_pairs], dim=0)


_BLOCK = 256


@functools.cache
def _token_reduce_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _fwd(
        src,
        rows,
        weights,
        out,
        stride_s,
        stride_o,
        hidden,
        topk: tl.constexpr,
        block: tl.constexpr,
    ):
        token = tl.program_id(0)
        tile = tl.program_id(1)
        offs = tile * block + tl.arange(0, block)
        mask = offs < hidden
        acc = tl.zeros((block,), dtype=tl.float32)
        for slot in tl.static_range(topk):
            row = tl.load(rows + token * topk + slot)
            val = tl.load(src + row * stride_s + offs, mask=mask, other=0.0).to(tl.float32)
            weight = tl.load(weights + token * topk + slot).to(tl.float32)
            acc += val * weight
        tl.store(out + token * stride_o + offs, acc.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _bwd_src(
        grad_out,
        rows,
        weights,
        grad_src,
        stride_g,
        stride_s,
        hidden,
        topk: tl.constexpr,
        block: tl.constexpr,
    ):
        index = tl.program_id(0)
        tile = tl.program_id(1)
        token = index // topk
        row = tl.load(rows + index)
        weight = tl.load(weights + index).to(tl.float32)
        offs = tile * block + tl.arange(0, block)
        mask = offs < hidden
        grad = tl.load(grad_out + token * stride_g + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(grad_src + row * stride_s + offs, (grad * weight).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _bwd_weight(
        grad_out,
        src,
        rows,
        grad_weight,
        stride_g,
        stride_s,
        hidden,
        topk: tl.constexpr,
        block: tl.constexpr,
    ):
        index = tl.program_id(0)
        token = index // topk
        row = tl.load(rows + index)
        acc = tl.zeros((block,), dtype=tl.float32)
        for hidden_offset in range(0, hidden, block):
            offs = hidden_offset + tl.arange(0, block)
            mask = offs < hidden
            grad = tl.load(
                grad_out + token * stride_g + offs, mask=mask, other=0.0
            ).to(tl.float32)
            value = tl.load(src + row * stride_s + offs, mask=mask, other=0.0).to(tl.float32)
            acc += grad * value
        tl.store(grad_weight + index, tl.sum(acc))

    return _fwd, _bwd_src, _bwd_weight


class _WeightedTokenReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, returned: torch.Tensor, order: torch.Tensor, weights: torch.Tensor):
        import triton

        tokens, topk = weights.shape
        hidden = returned.shape[-1]
        slots = returned.shape[0]
        rows = torch.empty(slots, dtype=torch.int32, device=returned.device)
        rows.scatter_(
            0,
            order.to(dtype=torch.int64),
            torch.arange(slots, device=returned.device, dtype=torch.int32),
        )
        out = torch.empty(tokens, hidden, dtype=returned.dtype, device=returned.device)
        fwd, _, _ = _token_reduce_kernels()
        fwd[(tokens, triton.cdiv(hidden, _BLOCK))](
            returned,
            rows,
            weights,
            out,
            returned.stride(0),
            out.stride(0),
            hidden,
            topk,
            _BLOCK,
        )
        ctx.save_for_backward(returned, rows, weights)
        ctx.topk = topk
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        import triton

        returned, rows, weights = ctx.saved_tensors
        if grad_out.stride(-1) != 1:
            grad_out = grad_out.contiguous()
        topk = ctx.topk
        hidden = returned.shape[-1]
        slots = returned.shape[0]
        _, bwd_src, bwd_weight = _token_reduce_kernels()
        grad_returned = grad_weights = None
        if ctx.needs_input_grad[0]:
            grad_returned = torch.empty_like(returned)
            bwd_src[(slots, triton.cdiv(hidden, _BLOCK))](
                grad_out,
                rows,
                weights,
                grad_returned,
                grad_out.stride(0),
                grad_returned.stride(0),
                hidden,
                topk,
                _BLOCK,
            )
        if ctx.needs_input_grad[2]:
            grad_weight_flat = torch.empty(slots, dtype=torch.float32, device=returned.device)
            bwd_weight[(slots,)](
                grad_out,
                returned,
                rows,
                grad_weight_flat,
                grad_out.stride(0),
                returned.stride(0),
                hidden,
                topk,
                _BLOCK,
            )
            grad_weights = grad_weight_flat.view_as(weights).to(dtype=weights.dtype)
        return grad_returned, None, grad_weights


def weighted_token_reduce(
    returned: torch.Tensor,
    order: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Sum top-k expert outputs back onto their tokens.

    ``returned`` is expert-sorted and ``order`` maps each sorted row to its
    original ``[token, slot]`` index. ``weights`` keeps that original token
    order. The reduction gathers each token's rows instead of atomically
    scattering them.
    """
    if returned.dtype != torch.bfloat16:
        raise TypeError("weighted_token_reduce expects bfloat16 expert outputs")
    if not returned.is_cuda:
        unsorted = torch.empty_like(returned)
        unsorted[order] = returned
        return (unsorted.view(*weights.shape, returned.shape[-1]) * weights.unsqueeze(-1)).sum(dim=1)
    return _WeightedTokenReduce.apply(returned.contiguous(), order, weights.contiguous())
