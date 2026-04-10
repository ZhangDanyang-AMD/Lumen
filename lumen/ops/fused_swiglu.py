###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused SwiGLU activation dispatch.

Dispatches to AITER's Triton fused SwiGLU kernels which compute
``silu(y1) * y2`` (forward) and the full backward in single kernel
launches each, eliminating the ~6-8 separate elementwise launches that
Megatron's ``@jit_fuser`` swiglu/swiglu_back produce.

Enable globally via ``LUMEN_FUSED_SWIGLU=1`` (installed by megatron_patches).
"""

import logging

import torch

logger = logging.getLogger(__name__)


def _probe_aiter_swiglu() -> bool:
    """Return True if AITER fused SwiGLU kernels are importable."""
    from lumen.ops.dispatch import _probe_aiter_swiglu as _probe

    return _probe()


def fused_swiglu(y: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU forward via AITER Triton kernel."""
    from aiter.ops.triton.activation import swiglu_fwd

    return swiglu_fwd(y)


def fused_swiglu_backward(grad_output: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU backward via AITER Triton kernel."""
    from aiter.ops.triton.activation import swiglu_bwd

    return swiglu_bwd(grad_output, y)
