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

    def _keep_mask_to_bias(query: torch.Tensor, key: torch.Tensor, attn_mask: torch.Tensor | None):
        # SDPA bool mask: True = attend. Lumen bias is additive.
        # SDPA broadcasts the mask to (B, H, Sq, Sk). A size-1 axis produced
        # by unsqueeze keeps a non-zero stride, and the attention kernel
        # indexes heads with that stride, which reads past the allocation.
        # expand() gives those axes stride 0.
        if attn_mask is None:
            return None
        b, h, sq, _ = query.shape
        sk = key.shape[-2]
        target = (b, h, sq, sk)
        if attn_mask.dtype == torch.bool:
            bias = torch.zeros(attn_mask.shape, dtype=query.dtype, device=attn_mask.device)
            bias = bias.masked_fill(~attn_mask, torch.finfo(query.dtype).min)
        else:
            bias = attn_mask
        if tuple(bias.shape) == target:
            return bias
        view = bias
        while view.dim() < 4:
            view = view.unsqueeze(0)
        if view.dim() != 4 or any(size not in (1, want) for size, want in zip(view.shape, target)):
            raise ValueError(
                f"attn_mask shape {tuple(attn_mask.shape)} is not broadcastable to {target}"
            )
        return view.expand(target)

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
        # The Lumen/AITER backward does not apply an additive bias: the Triton
        # backward kernels have no bias argument and recompute probabilities
        # without the mask, and CK's bias backward is wrong for short query
        # lengths. With a mask that blocks keys the gradients come out wrong
        # (cosine 0.4 to 0.9 against fp32, norms 1.2x to 4x too large) and can
        # overflow in bf16. Forward is fine, so keep Lumen for forward-only
        # calls and for unmasked attention, and let torch handle masked calls
        # that need a gradient.
        if attn_mask is not None and torch.is_grad_enabled() and (
            query.requires_grad or key.requires_grad or value.requires_grad
        ):
            return _original_sdpa(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale,
            )
        # F.sdpa: (B, H, T, D) → AITER: (B, T, H, D). The kernels
        # require a contiguous last dimension and assert it.
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        bias = _keep_mask_to_bias(query, key, attn_mask)
        try:
            # The Triton attn_fwd kernel adds the head stride to the bias
            # pointer and ignores the batch stride, so a per-batch mask has
            # to be launched one batch at a time.
            if bias is not None and q.shape[0] > 1 and bias.stride(0) != 0:
                out = torch.cat(
                    [
                        _lumen_attention(
                            q[i : i + 1], k[i : i + 1], v[i : i + 1],
                            dropout_p=dropout_p,
                            softmax_scale=scale,
                            causal=is_causal,
                            bias=bias[i : i + 1],
                        )
                        for i in range(q.shape[0])
                    ],
                    dim=0,
                )
            else:
                out = _lumen_attention(
                    q, k, v,
                    dropout_p=dropout_p,
                    softmax_scale=scale,
                    causal=is_causal,
                    bias=bias,
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
