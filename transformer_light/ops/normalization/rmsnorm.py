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

import torch
import torch.nn as nn

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
# Public functional API
# ---------------------------------------------------------------------------

def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMSNorm.

    Uses AITER when available (optimised Triton kernel with autograd),
    otherwise falls back to ``torch.nn.functional.rms_norm``.

    Args:
        x: Input tensor ``(*, hidden_size)``.
        weight: Learnable scale ``(hidden_size,)``.
        eps: Epsilon for numerical stability.

    Returns:
        Normalised tensor with same shape as *x*.
    """
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

    Example::

        norm = TransformerLightRMSNorm(4096)
        y = norm(x)  # x: (batch, seq, 4096)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        backend = "aiter" if _USE_AITER else "torch"
        return f"{self.weight.shape[0]}, eps={self.eps}, backend={backend}"
