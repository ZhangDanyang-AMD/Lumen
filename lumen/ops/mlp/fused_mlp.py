###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Fused MLP operations with AITER backend dispatch.

Provides fused gated and ungated MLP forward operations that dispatch
to AITER Triton kernels when available, with PyTorch sequential fallback.
"""

import functools
import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _probe_aiter_fused_gated():
    try:
        from aiter.ops.triton.gemm.feed_forward import ff_a16w16_fused_gated as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_fused_ungated():
    try:
        from aiter.ops.triton.gemm.feed_forward import ff_a16w16_fused_ungated as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


# Map Lumen activation names to AITER kernel names
_ACTIVATION_TO_AITER = {
    "swiglu": "silu",
    "geglu": "gelu",
    "reglu": "relu",
    "gelu": "gelu",
    "relu": "relu",
    "silu": "silu",
}

_ACTIVATION_FNS = {
    "swiglu": lambda x: F.silu(x),
    "geglu": lambda x: F.gelu(x),
    "reglu": lambda x: F.relu(x),
    "gelu": lambda x: F.gelu(x),
    "relu": lambda x: F.relu(x),
    "silu": lambda x: F.silu(x),
}


def fused_gated_mlp(
    x: torch.Tensor,
    w_up: torch.Tensor,
    w_gate: torch.Tensor,
    w_down: torch.Tensor,
    activation: str = "swiglu",
    bias_up: Optional[torch.Tensor] = None,
    bias_gate: Optional[torch.Tensor] = None,
    bias_down: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused gated MLP: down(gate_act(gate(x)) * up(x)).

    Dispatches to AITER ff_a16w16_fused_gated when available.
    """
    use_bias = bias_up is not None or bias_gate is not None or bias_down is not None
    if _probe_aiter_fused_gated() and not use_bias:
        try:
            from aiter.ops.triton.gemm.feed_forward import ff_a16w16_fused_gated

            aiter_act = _ACTIVATION_TO_AITER.get(activation, "silu")
            orig_shape = x.shape
            x_2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
            # AITER: w_up (N, K) = (2*hidden, input), w_down (N//2, K) = (hidden, input)
            w_combined = torch.cat([w_gate, w_up], dim=0)  # (2*hidden, input)
            w_down_aiter = w_down.T  # (hidden, input) for AITER
            out = ff_a16w16_fused_gated(x_2d, w_combined, w_down_aiter, dtype=x.dtype, activation=aiter_act)
            return out.reshape(orig_shape[:-1] + (out.shape[-1],))
        except (RuntimeError, TypeError):
            pass

    # PyTorch fallback
    act_fn = _ACTIVATION_FNS.get(activation, F.silu)
    gate_out = F.linear(x, w_gate, bias_gate)
    up_out = F.linear(x, w_up, bias_up)
    hidden = act_fn(gate_out) * up_out
    return F.linear(hidden, w_down, bias_down)


def fused_mlp(
    x: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    activation: str = "gelu",
    bias_up: Optional[torch.Tensor] = None,
    bias_down: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused ungated MLP: down(act(up(x))).

    Dispatches to AITER ff_a16w16_fused_ungated when available.
    """
    use_bias = bias_up is not None or bias_down is not None
    if _probe_aiter_fused_ungated() and not use_bias:
        try:
            from aiter.ops.triton.gemm.feed_forward import ff_a16w16_fused_ungated

            aiter_act = _ACTIVATION_TO_AITER.get(activation, "gelu")
            orig_shape = x.shape
            x_2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
            # AITER: w_up (N, K), w_down (N, K); our w_down is (K, N)
            w_down_aiter = w_down.T
            out = ff_a16w16_fused_ungated(x_2d, w_up, w_down_aiter, dtype=x.dtype, activation=aiter_act)
            return out.reshape(orig_shape[:-1] + (out.shape[-1],))
        except (RuntimeError, TypeError):
            pass

    # PyTorch fallback
    act_fn = _ACTIVATION_FNS.get(activation, F.gelu)
    hidden = act_fn(F.linear(x, w_up, bias_up))
    return F.linear(hidden, w_down, bias_down)


# ---------------------------------------------------------------------------
# FP8 activation storage for reduced memory in backward
# ---------------------------------------------------------------------------


class _FusedGatedMLPFP8Store(torch.autograd.Function):
    """Fused gated MLP with FP8 activation storage for backward."""

    @staticmethod
    def forward(ctx, x, w_gate, w_up, w_down, activation, bias_gate, bias_up, bias_down, fp8_dtype):
        act_fn = _ACTIVATION_FNS.get(activation, F.silu)
        gate_out = F.linear(x, w_gate, bias_gate)
        up_out = F.linear(x, w_up, bias_up)
        hidden = act_fn(gate_out) * up_out
        output = F.linear(hidden, w_down, bias_down)

        # Store activations in FP8
        amax_x = x.abs().amax().clamp(min=1e-12)
        amax_h = hidden.abs().amax().clamp(min=1e-12)
        fp8_max = torch.finfo(fp8_dtype).max if hasattr(torch.finfo(fp8_dtype), "max") else 448.0

        scale_x = fp8_max / amax_x
        scale_h = fp8_max / amax_h

        x_fp8 = (x.float() * scale_x).to(fp8_dtype)
        hidden_fp8 = (hidden.float() * scale_h).to(fp8_dtype)

        ctx.save_for_backward(
            x_fp8,
            hidden_fp8,
            w_gate,
            w_up,
            w_down,
            scale_x.unsqueeze(0),
            scale_h.unsqueeze(0),
        )
        ctx.activation = activation
        ctx.has_bias = (bias_gate is not None, bias_up is not None, bias_down is not None)
        ctx.fp8_dtype = fp8_dtype

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x_fp8, hidden_fp8, w_gate, w_up, w_down, scale_x, scale_h = ctx.saved_tensors

        # Dequantize
        x = (x_fp8.to(torch.float32) / scale_x).to(grad_output.dtype)
        hidden = (hidden_fp8.to(torch.float32) / scale_h).to(grad_output.dtype)

        # Backward through down projection
        grad_hidden = grad_output @ w_down
        grad_w_down = grad_output.reshape(-1, grad_output.shape[-1]).T @ hidden.reshape(-1, hidden.shape[-1])
        grad_bias_down = grad_output.sum(dim=tuple(range(grad_output.dim() - 1))) if ctx.has_bias[2] else None

        # Backward through activation * up
        act_fn = _ACTIVATION_FNS.get(ctx.activation, F.silu)
        gate_out = F.linear(x, w_gate)
        up_out = F.linear(x, w_up)

        with torch.enable_grad():
            gate_out_g = gate_out.detach().requires_grad_(True)
            act_val = act_fn(gate_out_g)
        act_grad = torch.autograd.grad(act_val.sum(), gate_out_g, retain_graph=False)[0]

        grad_gate = grad_hidden * up_out * act_grad
        grad_up = grad_hidden * act_fn(gate_out)

        grad_x = grad_gate @ w_gate + grad_up @ w_up
        grad_w_gate = grad_gate.reshape(-1, grad_gate.shape[-1]).T @ x.reshape(-1, x.shape[-1])
        grad_w_up = grad_up.reshape(-1, grad_up.shape[-1]).T @ x.reshape(-1, x.shape[-1])

        grad_bias_gate = grad_gate.sum(dim=tuple(range(grad_gate.dim() - 1))) if ctx.has_bias[0] else None
        grad_bias_up = grad_up.sum(dim=tuple(range(grad_up.dim() - 1))) if ctx.has_bias[1] else None

        return (
            grad_x,
            grad_w_gate,
            grad_w_up,
            grad_w_down,
            None,
            grad_bias_gate,
            grad_bias_up,
            grad_bias_down,
            None,
        )


def fused_gated_mlp_fp8_store(
    x: torch.Tensor,
    w_up: torch.Tensor,
    w_gate: torch.Tensor,
    w_down: torch.Tensor,
    activation: str = "swiglu",
    bias_up: Optional[torch.Tensor] = None,
    bias_gate: Optional[torch.Tensor] = None,
    bias_down: Optional[torch.Tensor] = None,
    fp8_dtype=torch.float8_e4m3fn,
) -> torch.Tensor:
    """Fused gated MLP with FP8 activation storage."""
    return _FusedGatedMLPFP8Store.apply(x, w_gate, w_up, w_down, activation, bias_gate, bias_up, bias_down, fp8_dtype)


class _FusedMLPFP8Store(torch.autograd.Function):
    """Fused ungated MLP with FP8 activation storage for backward."""

    @staticmethod
    def forward(ctx, x, w_up, w_down, activation, bias_up, bias_down, fp8_dtype):
        act_fn = _ACTIVATION_FNS.get(activation, F.gelu)
        hidden = act_fn(F.linear(x, w_up, bias_up))
        output = F.linear(hidden, w_down, bias_down)

        # Store activations in FP8
        amax_x = x.abs().amax().clamp(min=1e-12)
        amax_h = hidden.abs().amax().clamp(min=1e-12)
        fp8_max = torch.finfo(fp8_dtype).max if hasattr(torch.finfo(fp8_dtype), "max") else 448.0

        scale_x = fp8_max / amax_x
        scale_h = fp8_max / amax_h

        x_fp8 = (x.float() * scale_x).to(fp8_dtype)
        hidden_fp8 = (hidden.float() * scale_h).to(fp8_dtype)

        ctx.save_for_backward(
            x_fp8,
            hidden_fp8,
            w_up,
            w_down,
            scale_x.unsqueeze(0),
            scale_h.unsqueeze(0),
        )
        ctx.activation = activation
        ctx.has_bias = (bias_up is not None, bias_down is not None)
        ctx.fp8_dtype = fp8_dtype

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x_fp8, hidden_fp8, w_up, w_down, scale_x, scale_h = ctx.saved_tensors

        # Dequantize
        x = (x_fp8.to(torch.float32) / scale_x).to(grad_output.dtype)
        hidden = (hidden_fp8.to(torch.float32) / scale_h).to(grad_output.dtype)

        # Backward through down projection
        grad_hidden = grad_output @ w_down
        grad_w_down = grad_output.reshape(-1, grad_output.shape[-1]).T @ hidden.reshape(-1, hidden.shape[-1])
        grad_bias_down = grad_output.sum(dim=tuple(range(grad_output.dim() - 1))) if ctx.has_bias[1] else None

        # Backward through activation
        act_fn = _ACTIVATION_FNS.get(ctx.activation, F.gelu)
        up_out = F.linear(x, w_up)

        with torch.enable_grad():
            up_out_g = up_out.detach().requires_grad_(True)
            act_val = act_fn(up_out_g)
        act_grad = torch.autograd.grad(act_val.sum(), up_out_g, retain_graph=False)[0]

        grad_pre_act = grad_hidden * act_grad
        grad_x = grad_pre_act @ w_up
        grad_w_up = grad_pre_act.reshape(-1, grad_pre_act.shape[-1]).T @ x.reshape(-1, x.shape[-1])
        grad_bias_up = grad_pre_act.sum(dim=tuple(range(grad_pre_act.dim() - 1))) if ctx.has_bias[0] else None

        return (
            grad_x,
            grad_w_up,
            grad_w_down,
            None,
            grad_bias_up,
            grad_bias_down,
            None,
        )


def fused_mlp_fp8_store(
    x: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    activation: str = "gelu",
    bias_up: Optional[torch.Tensor] = None,
    bias_down: Optional[torch.Tensor] = None,
    fp8_dtype=torch.float8_e4m3fn,
) -> torch.Tensor:
    """Fused ungated MLP with FP8 activation storage."""
    return _FusedMLPFP8Store.apply(x, w_up, w_down, activation, bias_up, bias_down, fp8_dtype)
