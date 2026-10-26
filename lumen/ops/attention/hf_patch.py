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
        # F.sdpa: (B, H, T, D) → AITER: (B, T, H, D)
        # HF supplies BHSD views. Materialize BSHD so AITER's gfx1250 FMHA
        # assembly capability gate is selected (and its head-dim-contiguous
        # contract is explicit).
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()

        # HF Llama may materialize its ordinary decoder causal mask as a 4-D
        # additive SDPA mask even when no user attention mask was supplied.
        # The gfx1250 ASM kernel represents that mask through its causal flag
        # and does not accept it as an arbitrary bias tensor.
        bias = attn_mask
        causal = is_causal
        if attn_mask is not None and attn_mask.dim() == 4 and q.shape[1] == k.shape[1]:
            bias = None
            causal = True

        out = _lumen_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=scale,
            causal=causal,
            bias=bias,
        )

        # Back to (B, H, T, D)
        return out.transpose(1, 2)

    F.scaled_dot_product_attention = _lumen_sdpa
    logger.info("Patched F.scaled_dot_product_attention with Lumen/AITER attention")


def patch_hf_sdpa() -> None:
    """Alias kept for ``fsdp_backend.py`` compatibility."""
    patch_sdpa()
