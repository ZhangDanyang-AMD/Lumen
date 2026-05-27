###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Monkey-patch ``torch.nn.functional.scaled_dot_product_attention`` with
AITER-backed Lumen attention (CK csrc → Triton fallback).

Usage::

    from lumen.ops.attention.hf_patch import patch_sdpa
    patch_sdpa()   # one-shot, idempotent

After this call every ``F.scaled_dot_product_attention(q, k, v, …)`` in the
process is routed through :func:`lumen.ops.attention.attention`, which
dispatches to AITER CK kernels (with Triton fallback).

Layout handling:
    PyTorch ``F.scaled_dot_product_attention`` uses **(B, H, T, D)**.
    AITER / Lumen ``attention()`` uses **(B, T, H, D)**.
    The wrapper transposes automatically.

When ``attn_mask`` is a 2-D ``[B, T]`` padding mask (1 = valid, 0 = pad),
the wrapper converts it to ``cu_seqlens`` and dispatches through aiter's
Triton ``flash_attn_varlen_func`` with ``causal=True``.  This avoids the
O(T²) memory of materializing a 4-D mask and uses O(N) flash attention
instead.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_original_sdpa = None

# Triton varlen flash attention — preferred on MI350 (CK backward SIGABRTs)
_flash_attn_varlen = None
try:
    from aiter.ops.triton.attention.mha import flash_attn_varlen_func as _flash_attn_varlen
except ImportError:
    pass


def _varlen_flash_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: float | None,
) -> torch.Tensor:
    """Route through Triton flash_attn_varlen_func for 2-D padding masks.

    Inputs: query/key/value in F.sdpa layout **(B, H, T, D)**.
    ``attn_mask`` is **(B, T)** with 1 = valid, 0 = pad.
    """
    B, H_q, T, D = query.shape
    H_kv = key.shape[1]

    # Transpose to (B, T, H, D) for varlen API
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    # GQA expansion
    num_kv_groups = H_q // H_kv
    if num_kv_groups > 1:
        k = k.unsqueeze(3).expand(-1, -1, -1, num_kv_groups, -1).reshape(B, T, H_q, D)
        v = v.unsqueeze(3).expand(-1, -1, -1, num_kv_groups, -1).reshape(B, T, H_q, D)

    seqlens = attn_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = F.pad(seqlens.cumsum(0, dtype=torch.int32), (1, 0))
    max_seqlen = int(seqlens.max().item())

    valid_mask = attn_mask.bool()
    q_packed = q[valid_mask]
    k_packed = k[valid_mask]
    v_packed = v[valid_mask]

    fa_out = _flash_attn_varlen(
        q_packed, k_packed, v_packed,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
        softmax_scale=scale,
        return_lse=True,
    )
    attn_out_packed = fa_out[0] if isinstance(fa_out, tuple) else fa_out

    out = torch.zeros(B, T, H_q, D, dtype=attn_out_packed.dtype, device=attn_out_packed.device)
    out[valid_mask] = attn_out_packed

    # Back to (B, H, T, D)
    return out.transpose(1, 2)


def patch_sdpa() -> None:
    """Replace ``F.scaled_dot_product_attention`` with Lumen/AITER attention.

    Idempotent — calling multiple times is safe.
    """
    global _original_sdpa
    if _original_sdpa is not None:
        return

    from lumen.ops.attention import attention as _lumen_attention

    _original_sdpa = F.scaled_dot_product_attention

    def _lumen_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # 2-D padding mask → Triton flash_attn_varlen_func (O(N) memory)
        if (
            _flash_attn_varlen is not None
            and attn_mask is not None
            and attn_mask.dim() == 2
        ):
            return _varlen_flash_sdpa(query, key, value, attn_mask, scale)

        # F.sdpa: (B, H, T, D) → AITER: (B, T, H, D)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        out = _lumen_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=scale,
            causal=is_causal,
            bias=attn_mask,
        )

        # Back to (B, H, T, D)
        return out.transpose(1, 2)

    F.scaled_dot_product_attention = _lumen_sdpa
    logger.info("Patched F.scaled_dot_product_attention with Lumen/AITER attention (varlen+Triton for 2D masks)")


def patch_hf_sdpa() -> None:
    """Alias kept for ``fsdp_backend.py`` compatibility."""
    patch_sdpa()
