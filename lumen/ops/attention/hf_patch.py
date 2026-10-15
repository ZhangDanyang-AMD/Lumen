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
import os

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_original_sdpa = None

_BACKEND_ALIASES = {
    "": "auto",
    "auto": "auto",
    "csrc": "aiter_csrc",
    "aiter_csrc": "aiter_csrc",
    "opus": "aiter_opus",
    "aiter_opus": "aiter_opus",
    "triton": "aiter_triton",
    "aiter_triton": "aiter_triton",
}


def _resolve_backend() -> str:
    raw = os.environ.get("LUMEN_ATTN_BACKEND", "").strip().lower()
    backend = _BACKEND_ALIASES.get(raw)
    if backend is None:
        raise ValueError(
            f"LUMEN_ATTN_BACKEND={raw!r} is not one of "
            f"{sorted(set(_BACKEND_ALIASES) - {''})}"
        )
    return backend


def patch_sdpa() -> None:
    """Replace ``F.scaled_dot_product_attention`` with Lumen/AITER attention.

    Idempotent — calling multiple times is safe.
    """
    global _original_sdpa
    if _original_sdpa is not None:
        return

    from lumen.ops.attention import attention as _lumen_attention

    _original_sdpa = F.scaled_dot_product_attention
    backend = _resolve_backend()

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
            backend_type=backend,
        )

        # Back to (B, H, T, D)
        return out.transpose(1, 2)

    F.scaled_dot_product_attention = _lumen_sdpa
    # Logging is usually configured after this patch runs, so print instead.
    print(
        f"Patched F.scaled_dot_product_attention with Lumen/AITER attention "
        f"(backend={backend})",
        flush=True,
    )


def patch_hf_sdpa() -> None:
    """Alias kept for ``fsdp_backend.py`` compatibility."""
    patch_sdpa()
