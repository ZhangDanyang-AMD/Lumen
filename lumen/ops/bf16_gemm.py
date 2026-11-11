###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""BF16 GEMM patch used by ``LumenConfig.enable``.

``patch_gemm`` rebinds ``nn.Linear.forward``. ``linear`` is the same kernel
for a functional ``Y = X @ W.T`` (LoRA A/B). AITER has no autograd for
``gemm_a16w16``, so the backward is a torch matmul. Anything the kernel
rejects stays on ``F.linear``.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_FP16 = (torch.float16, torch.bfloat16)
_logged = False


def _use_aiter(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return x.is_cuda and x.dtype in _FP16 and weight.dtype in _FP16


class _GemmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        from lumen.ops.quantize.linear import dispatch_gemm

        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        with torch.no_grad():
            return dispatch_gemm(x, weight, scaling_type="none", bias=bias)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).to(dtype=weight.dtype)
        x_flat = x.reshape(-1, x.shape[-1]).to(dtype=weight.dtype)
        grad_x = (grad_flat @ weight).reshape(x.shape).to(dtype=x.dtype)
        grad_w = grad_flat.transpose(0, 1) @ x_flat
        grad_b = grad_flat.sum(0) if ctx.has_bias else None
        return grad_x, grad_w, grad_b


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``Y = X @ W.T`` (+ bias). AITER tuned bf16 GEMM, else ``F.linear``."""
    global _logged
    if not _use_aiter(x, weight):
        return F.linear(x, weight, bias)
    try:
        return _GemmFn.apply(x, weight, bias)
    except Exception as exc:
        if not _logged:
            logger.warning("aiter gemm fell back to torch: %s", exc)
            _logged = True
        return F.linear(x, weight, bias)


def patch_gemm(model: nn.Module) -> int:
    """Route every ``nn.Linear`` in ``model`` through :func:`linear`. Returns the count."""
    count = 0
    for module in model.modules():
        if not isinstance(module, nn.Linear) or getattr(module.forward, "_lumen_gemm", False):
            continue

        def _forward(inp, mod=module):
            return linear(inp, mod.weight, mod.bias)

        _forward._lumen_gemm = True
        module.forward = _forward
        count += 1
    return count
