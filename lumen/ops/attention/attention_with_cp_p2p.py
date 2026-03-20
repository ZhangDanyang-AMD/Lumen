###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Ring-based P2P context parallelism for attention.

Implements ring attention where KV chunks are circulated between CP ranks
using point-to-point (isend/irecv) communication. Each rank computes a
partial attention with the received KV chunk and accumulates results using
online softmax (log-sum-exp correction).

This complements the existing A2A-based CP in attention_with_cp_a2a.py.
Ring CP is preferred when:
- The number of CP ranks is large (reduces per-rank memory)
- Bandwidth is limited (only sends to one neighbor at a time)
"""

import logging
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _ring_send_recv_kv(
    send_k: torch.Tensor,
    send_v: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_rank: int,
    cp_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Send KV to next rank and receive from previous rank in the ring.

    A stream sync before the P2P ensures that any in-flight compute kernel
    (attention, contiguous copy) has finished writing the send buffer before
    RCCL reads it on the NCCL stream.
    """
    next_rank = (cp_rank + 1) % cp_size
    prev_rank = (cp_rank - 1) % cp_size

    send_k_c = send_k.contiguous()
    send_v_c = send_v.contiguous()
    recv_k = torch.empty_like(send_k_c)
    recv_v = torch.empty_like(send_v_c)

    torch.cuda.current_stream().synchronize()

    send_k_op = dist.P2POp(dist.isend, send_k_c, group=cp_group, group_peer=next_rank)
    send_v_op = dist.P2POp(dist.isend, send_v_c, group=cp_group, group_peer=next_rank)
    recv_k_op = dist.P2POp(dist.irecv, recv_k, group=cp_group, group_peer=prev_rank)
    recv_v_op = dist.P2POp(dist.irecv, recv_v, group=cp_group, group_peer=prev_rank)

    reqs = dist.batch_isend_irecv([send_k_op, send_v_op, recv_k_op, recv_v_op])
    for req in reqs:
        req.wait()

    return recv_k, recv_v


def _online_softmax_update(
    out_accum: torch.Tensor,
    lse_accum: torch.Tensor,
    out_new: torch.Tensor,
    lse_new: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Online softmax accumulation (log-sum-exp correction).

    Merges two partial attention outputs using the numerically stable
    online softmax algorithm.
    """
    lse_max = torch.maximum(lse_accum, lse_new)
    exp_old = torch.exp(lse_accum - lse_max)
    exp_new = torch.exp(lse_new - lse_max)

    denom = exp_old + exp_new
    out = (out_accum * exp_old.unsqueeze(-1) + out_new * exp_new.unsqueeze(-1)) / denom.unsqueeze(-1)
    lse = lse_max + torch.log(denom)

    return out, lse


class AttentionCPP2PFunction(torch.autograd.Function):
    """Ring-based context parallelism for attention via P2P communication.

    Forward: iterate over CP ranks, async send/recv KV chunks, compute
    partial attention, accumulate with online softmax.

    Backward: reverse ring, send/recv dKV chunks.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cp_group: dist.ProcessGroup,
        cp_size: int,
        cp_rank: int,
        attn_fn: Callable,
        causal: bool = True,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        B, Sq, H, D = q.shape
        device = q.device
        dtype = q.dtype

        out_accum = torch.zeros(B, Sq, H, D, device=device, dtype=dtype)
        lse_accum = torch.full((B, Sq, H), float("-inf"), device=device, dtype=torch.float32)

        current_k = k
        current_v = v

        for step in range(cp_size):
            is_local = step == 0
            source_rank = (cp_rank - step) % cp_size
            is_future = causal and source_rank > cp_rank

            if not is_future:
                use_causal = causal and is_local
                out_chunk, lse_chunk = attn_fn(
                    q,
                    current_k,
                    current_v,
                    causal=use_causal,
                    softmax_scale=softmax_scale,
                )

                if step == 0:
                    out_accum = out_chunk
                    lse_accum = lse_chunk
                else:
                    out_accum, lse_accum = _online_softmax_update(
                        out_accum,
                        lse_accum,
                        out_chunk,
                        lse_chunk,
                    )

            if step < cp_size - 1:
                current_k, current_v = _ring_send_recv_kv(
                    current_k,
                    current_v,
                    cp_group,
                    cp_rank,
                    cp_size,
                )

        ctx.save_for_backward(q, k, v, lse_accum)
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        ctx.attn_fn = attn_fn
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale

        return out_accum

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, lse = ctx.saved_tensors
        cp_group = ctx.cp_group
        cp_size = ctx.cp_size
        cp_rank = ctx.cp_rank

        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)

        current_k = k
        current_v = v

        for step in range(cp_size):
            is_local = step == 0
            source_rank = (cp_rank - step) % cp_size
            is_future = ctx.causal and source_rank > cp_rank

            if not is_future:
                use_causal = ctx.causal and is_local

                q_req_grad = q.detach().requires_grad_(True)
                k_req_grad = current_k.detach().requires_grad_(True)
                v_req_grad = current_v.detach().requires_grad_(True)

                with torch.enable_grad():
                    out_chunk, _ = ctx.attn_fn(
                        q_req_grad,
                        k_req_grad,
                        v_req_grad,
                        causal=use_causal,
                        softmax_scale=ctx.softmax_scale,
                    )
                    out_chunk.backward(grad_output)

                grad_q.add_(q_req_grad.grad)

                if is_local:
                    grad_k.add_(k_req_grad.grad)
                    grad_v.add_(v_req_grad.grad)

            if step < cp_size - 1:
                current_k, current_v = _ring_send_recv_kv(
                    current_k,
                    current_v,
                    cp_group,
                    cp_rank,
                    cp_size,
                )

        return grad_q, grad_k, grad_v, None, None, None, None, None, None


def attention_cp_p2p(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_size: int,
    cp_rank: int,
    attn_fn: Callable,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Ring-based P2P context parallelism for attention.

    Each CP rank holds a local chunk of Q, K, V along the sequence dimension.
    KV chunks are circulated in a ring and partial attention is accumulated
    using online softmax.
    """
    return AttentionCPP2PFunction.apply(
        q,
        k,
        v,
        cp_group,
        cp_size,
        cp_rank,
        attn_fn,
        causal,
        softmax_scale,
    )
