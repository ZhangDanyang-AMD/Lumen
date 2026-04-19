###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Fused RMSNorm + FP8 Quantization with pre-quantized activation bypass.

Combines RMSNorm and FP8 per-tensor quantization into a single AITER Triton
kernel call, producing **both** the BF16 normalised output (for downstream
attention / MLP computation) and the FP8 quantized activation (for GEMM).

The FP8 result is injected into the next ``quant_forward()`` call via
thread-local storage, skipping the separate ``quantize_input()`` pass.

The BF16 norm output is wrapped in :class:`_FusedNQGNorm` — a custom
``autograd.Function`` that preserves gradient flow by saving the input and
``rsigma`` for backward, then delegating to AITER's Triton RMSNorm backward
kernel.  This ensures the fused kernel's BF16 output is used directly in
the forward pass (no double-compute, no FP8/BF16 mismatch) while backward
gradients remain correct.

Public API
----------
- :func:`fused_norm_quant_for_linear` — high-level: norm+quant fused,
  sets thread-local for the downstream linear, updates amax bookkeeping.
- :func:`fused_rmsnorm_fp8` — low-level: just the kernel call returning
  ``(fp8_out, bf16_norm_out, scale)``.

Gated by ``LUMEN_FUSED_NORM_QUANT_GEMM=1`` environment variable.
"""

import logging
import os
from typing import Optional, Tuple

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

logger = logging.getLogger(__name__)

_FUSED_NQG_ENABLED = os.environ.get("LUMEN_FUSED_NORM_QUANT_GEMM", "0") == "1"

_fused_fn = None
_fused_fn_probed = False


def _get_fused_kernel():
    """Lazy-load the AITER fused RMSNorm+FP8Quant Triton kernel."""
    global _fused_fn, _fused_fn_probed
    if _fused_fn_probed:
        return _fused_fn
    _fused_fn_probed = True
    try:
        from aiter.ops.triton.quant.fused_fp8_quant import (
            fused_rms_fp8_per_tensor_static_quant,
        )

        _fused_fn = fused_rms_fp8_per_tensor_static_quant
    except ImportError:
        _fused_fn = None
    return _fused_fn


def _get_norm_params(norm_module: torch.nn.Module) -> Tuple[torch.Tensor, float]:
    """Extract weight and eps from a norm module (handles Lumen wrapper types)."""
    from lumen.ops.fused_residual_norm import get_norm_params

    return get_norm_params(norm_module)


def _unwrap_linear(linear_module: torch.nn.Module) -> torch.nn.Module:
    """Unwrap LoRA / other adapters to reach the base linear with quant metadata."""
    return getattr(linear_module, "base_layer", linear_module)


# ---------------------------------------------------------------------------
# Low-level API
# ---------------------------------------------------------------------------


def fused_rmsnorm_fp8(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Fused RMSNorm + per-tensor static FP8 quantization.

    Calls AITER's ``fused_rms_fp8_per_tensor_static_quant`` with
    ``output_unquantized_inp1=True`` and ``output_rsigma=True`` to produce
    all outputs in one pass.

    Args:
        x: Input tensor ``(M, K)`` (must be 2D, contiguous).
        norm_weight: RMSNorm scale ``(K,)``.
        eps: RMSNorm epsilon.
        scale: Per-tensor quantization scale ``(1,)`` — ``amax / fp8_max``.
        fp8_dtype: Target FP8 dtype (e.g. ``torch.float8_e4m3fnuz``).

    Returns:
        ``(fp8_out, bf16_norm_out, scale, rsigma)`` on success, or ``None``
        if the kernel is unavailable.  ``rsigma`` has shape ``(M,)`` in
        float32.
    """
    fn = _get_fused_kernel()
    if fn is None:
        return None
    try:
        fp8_out, bf16_out, _, _, rsigma = fn(
            x,
            norm_weight,
            eps,
            scale,
            dtype_quant=fp8_dtype,
            output_unquantized_inp1=True,
            output_rsigma=True,
        )
        return fp8_out, bf16_out, scale, rsigma
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Autograd wrapper for fused kernel BF16 output
# ---------------------------------------------------------------------------


class _FusedNQGNorm(Function):
    """Attach autograd backward to the fused kernel's BF16 norm output.

    The fused ``fused_rms_fp8_per_tensor_static_quant`` kernel produces a
    numerically correct BF16 RMSNorm output and ``rsigma`` but does not
    participate in PyTorch autograd.  This Function re-establishes the
    gradient chain:

    * **Forward** — returns the pre-computed ``bf16_norm_out`` as-is (no
      extra compute), saves ``(x_2d, weight, rsigma)`` for backward.
      ``rsigma`` is provided by the fused kernel (computed as a free
      byproduct of the norm — zero additional memory reads).
    * **Backward** — delegates to AITER's Triton ``_rmsnorm_backward``
      kernel (the same kernel used by the standard ``rms_norm()`` path),
      producing correct ``dx`` and ``dweight``.

    Using the fused kernel's own BF16 output avoids the precision mismatch
    that arises when a second, independent norm kernel computes a slightly
    different result.
    """

    @staticmethod
    def forward(ctx, x_2d, weight, rsigma, bf16_norm_out):
        ctx.save_for_backward(x_2d, weight, rsigma)
        return bf16_norm_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x_2d, weight, rsigma = ctx.saved_tensors
        try:
            from aiter.ops.triton.normalization.rmsnorm import (
                _rmsnorm_backward,
            )

            dx, dw = _rmsnorm_backward(grad_output, x_2d, weight, rsigma)
        except ImportError:
            dx = grad_output
            dw = None
        return dx, dw, None, None


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def fused_norm_quant_for_linear(
    hidden_states: torch.Tensor,
    norm_module: torch.nn.Module,
    linear_module: torch.nn.Module,
) -> Tuple[Optional[torch.Tensor], bool]:
    """Fuse RMSNorm + FP8 quant and set pre-quantized activation for a linear.

    Performs the following in one shot:

    1. Extracts norm weight/eps from *norm_module*.
    2. Looks up the scaling manager, scaling type, FP8 dtype, and activation
       tensor ID from *linear_module* (unwrapping LoRA if needed).
    3. Obtains the quantization scale from the scaling manager.
    4. Calls the AITER fused kernel to produce both BF16 norm output and
       FP8 quantized activation.
    5. Stores the FP8 result via thread-local for the downstream
       ``quant_forward()`` to consume (skipping ``quantize_input()``).
    6. Updates the amax history for delayed scaling bookkeeping.

    Args:
        hidden_states: Input hidden states ``(*, H)``.
        norm_module: The ``input_layernorm`` or ``pre_mlp_layernorm`` module.
        linear_module: The downstream ``linear_qkv`` or ``linear_fc1`` module
            (may be wrapped by LoRA).

    Returns:
        ``(bf16_norm_out, True)`` on success, or ``(None, False)`` if fused
        path is unavailable or any precondition fails. Callers should fall
        back to the standard norm path on ``False``.
    """
    if not _FUSED_NQG_ENABLED:
        return None, False

    base = _unwrap_linear(linear_module)

    mgr = getattr(base, "_lumen_scaling_manager", None)
    if mgr is None:
        return None, False

    scaling_type = getattr(base, "_lumen_scaling_type", None)
    if scaling_type not in ("delayed", "dynamic"):
        return None, False

    fp8_dtype = getattr(base, "_lumen_fp8_dtype", None)
    if fp8_dtype is None:
        return None, False

    act_tid = getattr(base, "_lumen_act_tensor_id", "activation")

    orig_shape = hidden_states.shape
    x_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()

    norm_w, eps = _get_norm_params(norm_module)

    scale, precomputed_amax = mgr.get_scale(act_tid, x_2d, return_amax=True)
    if scale is None:
        return None, False

    result = fused_rmsnorm_fp8(x_2d, norm_w, eps, scale, fp8_dtype)
    if result is None:
        return None, False

    fp8_out, bf16_fused, scale, rsigma = result

    from lumen.quantize import _set_pre_quantized_activation

    _set_pre_quantized_activation(fp8_out, scale)

    if precomputed_amax is not None:
        mgr.update_amax_value(act_tid, precomputed_amax)
    else:
        mgr.update_amax(act_tid, x_2d)

    bf16_norm_out = _FusedNQGNorm.apply(x_2d, norm_w, rsigma, bf16_fused)

    return bf16_norm_out.reshape(orig_shape), True
