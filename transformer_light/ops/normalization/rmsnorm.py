###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""RMSNorm with AITER backend (preferred) and ``torch.nn.RMSNorm`` fallback.

When the AITER package is available, delegates to ``aiter.ops.triton.
normalization.rmsnorm.rms_norm`` which provides a highly-optimised Triton
kernel with full autograd support.  Falls back to PyTorch's native
``torch.nn.functional.rms_norm`` (available since PyTorch 2.4) when AITER
is not installed.

Provides:
    - :func:`rmsnorm` — functional API  (autograd-aware)
    - :class:`TransformerLightRMSNorm` — ``nn.Module`` API
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from transformer_light.core.grad_quant import quantize_grad_tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend selection: prefer AITER, fall back to torch.nn.RMSNorm
# ---------------------------------------------------------------------------

_USE_AITER = False
_aiter_rms_norm = None

try:
    from aiter.ops.triton.normalization.rmsnorm import rms_norm as _aiter_rms_norm
    _USE_AITER = True
    logger.info("RMSNorm: using AITER backend")
except ImportError:
    logger.info("RMSNorm: AITER not available, using torch.nn.functional fallback")


def _torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fallback using PyTorch native ``torch.nn.functional.rms_norm``."""
    return torch.nn.functional.rms_norm(x, weight.shape, weight, eps)


# ---------------------------------------------------------------------------
# Grad-quantized wrapper (autograd Function)
# ---------------------------------------------------------------------------


class _RMSNormGradQuant(torch.autograd.Function):
    """Thin wrapper that applies gradient quantization after the inner
    RMSNorm backward.  When ``grad_quant_type`` is ``None`` we never even
    instantiate this — the raw ``rmsnorm`` path is used instead.
    """

    @staticmethod
    def forward(ctx, x, weight, eps, grad_quant_type):
        ctx.grad_quant_type = grad_quant_type
        ctx.eps = eps
        ctx.save_for_backward(x, weight)
        if _USE_AITER:
            orig_shape = x.shape
            x_2d = x.reshape(-1, x.shape[-1])
            y = _aiter_rms_norm(x_2d, weight, eps)
            return y.reshape(orig_shape)
        return _torch_rmsnorm(x, weight, eps)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        gqt = ctx.grad_quant_type

        with torch.enable_grad():
            x_detached = x.detach().requires_grad_(True)
            w_detached = weight.detach().requires_grad_(True)
            if _USE_AITER:
                orig_shape = x_detached.shape
                x_2d = x_detached.reshape(-1, x_detached.shape[-1])
                y = _aiter_rms_norm(x_2d, w_detached, ctx.eps)
                y = y.reshape(orig_shape)
            else:
                y = _torch_rmsnorm(x_detached, w_detached, ctx.eps)
            torch.autograd.backward(y, grad_output)

        dx = x_detached.grad
        dw = w_detached.grad

        dx = quantize_grad_tensor(dx, gqt)
        dw = quantize_grad_tensor(dw, gqt)
        return dx, dw, None, None


# ---------------------------------------------------------------------------
# Public functional API
# ---------------------------------------------------------------------------

def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    grad_quant_type: Optional[str] = None,
) -> torch.Tensor:
    """Apply RMSNorm.

    Uses AITER when available (optimised Triton kernel with autograd),
    otherwise falls back to ``torch.nn.functional.rms_norm``.

    Args:
        x: Input tensor ``(*, hidden_size)``.
        weight: Learnable scale ``(hidden_size,)``.
        eps: Epsilon for numerical stability.
        grad_quant_type: Gradient quantization format — ``"fp8"``,
            ``"mxfp8"``, ``"fp4"``, or ``None`` (disabled).

    Returns:
        Normalised tensor with same shape as *x*.
    """
    if grad_quant_type is not None:
        return _RMSNormGradQuant.apply(x, weight, eps, grad_quant_type)

    if _USE_AITER:
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        y = _aiter_rms_norm(x_2d, weight, eps)
        return y.reshape(orig_shape)
    return _torch_rmsnorm(x, weight, eps)


# ---------------------------------------------------------------------------
# nn.Module API
# ---------------------------------------------------------------------------

class TransformerLightRMSNorm(nn.Module):
    """RMSNorm backed by AITER (preferred) or PyTorch native.

    Drop-in replacement for ``torch.nn.RMSNorm``, TE ``RMSNorm``, or
    Megatron-Core ``RMSNorm``.

    Args:
        hidden_size: Last dimension of the input.
        eps: Epsilon for numerical stability.
        grad_quant_type: Gradient quantization format — ``"fp8"``,
            ``"mxfp8"``, ``"fp4"``, or ``None`` (disabled).

    Example::

        norm = TransformerLightRMSNorm(4096)
        y = norm(x)  # x: (batch, seq, 4096)
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        grad_quant_type: Optional[str] = None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.grad_quant_type = grad_quant_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps, self.grad_quant_type)

    def extra_repr(self) -> str:
        backend = "aiter" if _USE_AITER else "torch"
        gq = f", grad_quant={self.grad_quant_type}" if self.grad_quant_type else ""
        return f"{self.weight.shape[0]}, eps={self.eps}, backend={backend}{gq}"
