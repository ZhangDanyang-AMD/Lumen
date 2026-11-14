###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Autograd wrappers around aiter's RMSNorm kernels.

``rmsnorm2d_fwd`` is the plain norm. ``rmsnorm2d_fwd_with_add`` fuses the
residual add. The backward recomputes ``rsigma`` and calls aiter's
``_rmsnorm_backward``. The classes are built on first use so importing this
module does not import aiter.
"""

from __future__ import annotations

_cached = None


def rmsnorm_fns():
    """Return ``(RMSNormFn, AddRMSNormFn)``."""
    global _cached
    if _cached is not None:
        return _cached

    import torch
    from aiter import rmsnorm2d_fwd, rmsnorm2d_fwd_with_add

    class RMSNormFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, eps):
            ctx.save_for_backward(x, weight)
            ctx.eps = eps
            x_2d = x.reshape(-1, x.shape[-1])
            y = rmsnorm2d_fwd(x_2d, weight, eps)
            return y.reshape(x.shape)

        @staticmethod
        def backward(ctx, grad_output):
            x, weight = ctx.saved_tensors
            from aiter.ops.triton.normalization.rmsnorm import _rmsnorm_backward

            x_2d = x.reshape(-1, x.shape[-1])
            go_2d = grad_output.reshape(-1, grad_output.shape[-1])
            # _rmsnorm_backward multiplies by rsigma. eps is about 1e-6;
            # rsigma is rsqrt(mean(x^2) + eps), the same value the
            # residual path recomputes before calling this kernel.
            variance = x_2d.float().pow(2).mean(dim=-1)
            rsigma = torch.rsqrt(variance + ctx.eps)
            dx, dw = _rmsnorm_backward(go_2d, x_2d, weight, rsigma)
            return dx.reshape(x.shape), dw, None

    class AddRMSNormFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, residual, weight, eps):
            x_2d = x.reshape(-1, x.shape[-1])
            residual_2d = residual.reshape(-1, residual.shape[-1])
            out = torch.empty_like(x_2d)
            residual_out = torch.empty_like(x_2d)
            rmsnorm2d_fwd_with_add(
                out,
                x_2d,
                residual_2d,
                residual_out,
                weight,
                float(eps),
                use_model_sensitive_rmsnorm=0,
            )
            ctx.save_for_backward(residual_out, weight)
            ctx.eps = float(eps)
            ctx.shape = x.shape
            return out.view(x.shape), residual_out.view(x.shape)

        @staticmethod
        def backward(ctx, grad_out, grad_residual):
            residual_out, weight = ctx.saved_tensors
            from aiter.ops.triton.normalization.rmsnorm import _rmsnorm_backward

            grad_2d = grad_out.reshape(-1, grad_out.shape[-1])
            variance = residual_out.float().pow(2).mean(dim=-1)
            rsigma = torch.rsqrt(variance + ctx.eps)
            dx, dw = _rmsnorm_backward(grad_2d, residual_out, weight, rsigma)
            if grad_residual is not None:
                dx = dx + grad_residual.reshape(-1, grad_residual.shape[-1])
            dx = dx.view(ctx.shape)
            return dx, dx, dw, None

    _cached = (RMSNormFn, AddRMSNormFn)
    return _cached
