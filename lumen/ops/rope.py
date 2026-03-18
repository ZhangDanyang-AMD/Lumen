###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Fused Rotary Positional Embedding (RoPE) with AITER backend dispatch.

Dispatches to AITER's fused QKV split + RoPE kernel when available,
with a pure-PyTorch fallback for compatibility.
"""

import functools
import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _probe_aiter_fused_rope():
    """Check if AITER fused QKV split + RoPE kernel is available."""
    try:
        from aiter.ops.triton.rope import fused_qkv_split_qk_rope as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


def _apply_rotary_pos_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    """Pure PyTorch RoPE application.

    Args:
        x: Input tensor [..., seq_len, head_dim].
        cos: Cosine frequencies [seq_len, head_dim] or [1, seq_len, 1, head_dim].
        sin: Sine frequencies (same shape as cos).
        interleaved: If True, use interleaved (GPT-J) layout.
    """
    if interleaved:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        cos = cos[..., 0::2] if cos.shape[-1] == x.shape[-1] else cos
        sin = sin[..., 0::2] if sin.shape[-1] == x.shape[-1] else sin
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        return torch.stack((out1, out2), dim=-1).flatten(-2)
    else:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        # Broadcast cos/sin to match x shape
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        out1 = x1 * cos[..., :half] - x2 * sin[..., :half]
        out2 = x2 * cos[..., :half] + x1 * sin[..., :half]
        return torch.cat((out1, out2), dim=-1)


def fused_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply fused RoPE to query and key tensors.

    Dispatches to AITER fused_qkv_split_qk_rope when available.

    Args:
        q: Query tensor.
        k: Key tensor.
        cos: Cosine frequencies.
        sin: Sine frequencies.
        interleaved: Use interleaved (GPT-J) layout.

    Returns:
        Tuple of (q_rotated, k_rotated).
    """
    if _probe_aiter_fused_rope():
        try:
            from aiter.ops.triton.rope import fused_qkv_split_qk_rope

            q_rot, k_rot = fused_qkv_split_qk_rope(q, k, cos, sin)
            return q_rot, k_rot
        except (RuntimeError, TypeError) as e:
            logger.warning("fused_rope: AITER kernel failed (%s), using PyTorch fallback", e)

    q_rot = _apply_rotary_pos_emb_torch(q, cos, sin, interleaved)
    k_rot = _apply_rotary_pos_emb_torch(k, cos, sin, interleaved)
    return q_rot, k_rot
