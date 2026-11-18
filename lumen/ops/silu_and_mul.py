###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Autograd wrapper around ``aiter.silu_and_mul``."""

from __future__ import annotations

_cached = None


def silu_and_mul_fn():
    """Return the SiLU-and-mul autograd function, built on first use."""
    global _cached
    if _cached is not None:
        return _cached

    import torch
    from aiter import silu_and_mul

    class SiluAndMulFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            out = torch.empty(
                [*x.shape[:-1], x.shape[-1] // 2],
                device=x.device, dtype=x.dtype,
            )
            silu_and_mul(out, x)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            d = x.shape[-1] // 2
            gate, up = x[..., :d], x[..., d:]
            sig = torch.sigmoid(gate)
            d_gate = grad_output * up * (sig + gate * sig * (1.0 - sig))
            d_up = grad_output * (gate * sig)
            return torch.cat([d_gate, d_up], dim=-1)

    _cached = SiluAndMulFn
    return _cached
