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
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_original_sdpa = None


def patch_sdpa() -> None:
    """Replace ``F.scaled_dot_product_attention`` with Lumen/AITER attention.

    Idempotent — calling multiple times is safe.
    """
    global _original_sdpa
    if _original_sdpa is not None:
        return

    from lumen.ops.attention import attention as _lumen_attention

    _original_sdpa = F.scaled_dot_product_attention

    def _keep_mask_to_bias(query: torch.Tensor, attn_mask: torch.Tensor | None):
        # SDPA bool mask: True = attend. Lumen bias is additive.
        if attn_mask is None or attn_mask.dtype != torch.bool:
            return attn_mask
        bias = torch.zeros(attn_mask.shape, dtype=query.dtype, device=attn_mask.device)
        return bias.masked_fill(~attn_mask, torch.finfo(query.dtype).min)

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
        if kwargs:
            return _original_sdpa(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale, **kwargs,
            )
        # F.sdpa: (B, H, T, D) → AITER: (B, T, H, D)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        try:
            out = _lumen_attention(
                q, k, v,
                dropout_p=dropout_p,
                softmax_scale=scale,
                causal=is_causal,
                bias=_keep_mask_to_bias(query, attn_mask),
            )
        except Exception as exc:
            logger.warning("Lumen attention fell back to torch SDPA: %s", exc)
            return _original_sdpa(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale,
            )
        return out.transpose(1, 2)

    F.scaled_dot_product_attention = _lumen_sdpa
    logger.info("Patched F.scaled_dot_product_attention with Lumen/AITER attention")


def patch_hf_sdpa() -> None:
    """Alias kept for ``fsdp_backend.py`` compatibility."""
    patch_sdpa()
