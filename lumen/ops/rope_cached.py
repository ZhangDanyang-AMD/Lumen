###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Autograd wrapper for the cached rotary embedding."""

from __future__ import annotations


def make_cached_rope_fn(rotary):
    """Return an autograd function that applies *rotary* to Q and K.

    The forward rebuilds the cache on the activation's device. A CPU-built
    cache first differs at position 64 for Qwen3-8B. The backward applies the
    conjugate rotation (``-sin``).
    """
    import torch
    from aiter.rotary_embedding import _apply_rotary_emb

    class CachedRoPEFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, positions):
            nonlocal rotary
            if rotary.cos_cache.device != q.device:
                with torch.device(q.device):
                    rotary = type(rotary)(
                        rotary.head_size,
                        rotary.rotary_dim,
                        rotary.max_position_embeddings,
                        rotary.base,
                        rotary.is_neox_style,
                        q.dtype,
                    )
            elif rotary.cos_cache.dtype != q.dtype:
                rotary.cos_cache = rotary.cos_cache.to(device=q.device, dtype=q.dtype)
                rotary.sin_cache = rotary.sin_cache.to(device=q.device, dtype=q.dtype)
            ctx.rotary = rotary
            ctx.save_for_backward(positions)
            q_out = q.detach().clone()
            k_out = k.detach().clone()
            return rotary(positions, q_out, k_out)

        @staticmethod
        def backward(ctx, grad_q, grad_k):
            positions, = ctx.saved_tensors
            index = positions.reshape(-1)
            # Cache is (positions, 1, 1, rotary_dim / 2). Squeeze the
            # two size-1 axes, matching RotaryEmbedding.forward_native.
            cos = ctx.rotary.cos_cache.index_select(0, index).squeeze(-2).squeeze(-2)
            sin = ctx.rotary.sin_cache.index_select(0, index).squeeze(-2).squeeze(-2)
            grad_q = _apply_rotary_emb(grad_q, cos, -sin, True)
            grad_k = _apply_rotary_emb(grad_k, cos, -sin, True)
            return grad_q, grad_k, None

    return CachedRoPEFn
