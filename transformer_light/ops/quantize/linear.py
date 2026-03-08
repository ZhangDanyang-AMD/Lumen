###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Quantized linear forward + backward with explicit autograd.

Architecture mirrors ``pytorch/ops/attention/attention.py``:

- ``QuantizedLinearFunction`` — ``torch.autograd.Function`` that quantizes
  input/weight, runs FP8 GEMM, and dequantizes in both forward and backward.
- ``quantized_linear()`` — convenience functional API.

Supports two backends:

- **aiter** — AITER ``per_token_quant_hip`` + ``hipb_mm`` (CK / hipBLASLt).
- **triton** — Transformer Light Triton blockwise quant + ``torch._scaled_mm``.

The nn.Module wrapper lives in
:class:`transformer_light.modules.quantize.TransformerLightLinear`.
"""

from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from transformer_light.ops.quantize.ops import (
    quant_fp8_blockwise_impl,
    convert_to_mxfp8,
    convert_from_mxfp8,
)

__all__ = ["QuantizedLinearFunction", "quantized_linear"]


# ---------------------------------------------------------------------------
# torch.compile support — treat the custom autograd Function as an opaque
# graph node so Dynamo does not trace through the FP8 quantization internals
# (which contain .item() calls and mutable ScalingManager state).
# ---------------------------------------------------------------------------
def _mark_allow_in_graph(cls):
    try:
        from torch._dynamo import allow_in_graph
        allow_in_graph(cls)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# AITER backend helpers
# ---------------------------------------------------------------------------

def _aiter_quant(x: torch.Tensor, dtype: torch.dtype):
    from aiter.ops.quant import per_token_quant_hip
    return per_token_quant_hip(x, quant_dtype=dtype)


def _aiter_mm(a: torch.Tensor, b: torch.Tensor, scale_a, scale_b):
    from aiter.ops.gradlib import hipb_mm
    return hipb_mm(a, b, scaleA=scale_a, scaleB=scale_b)


# ---------------------------------------------------------------------------
# Triton backend helpers
# ---------------------------------------------------------------------------

def _triton_quant(x: torch.Tensor, dtype: torch.dtype, block_size: int = 128):
    """Blockwise FP8 quantization via Triton kernel."""
    orig_shape = x.shape
    flat = x.reshape(-1, orig_shape[-1]).contiguous()
    x_fp8, x_scales = quant_fp8_blockwise_impl(flat, dtype=dtype, axis=1, block_size=block_size)
    return x_fp8.view(orig_shape), x_scales


def _triton_mm(a_fp8: torch.Tensor, b_fp8: torch.Tensor, scale_a, scale_b):
    """Scaled matrix multiply using PyTorch's native scaled_mm.

    Handles mixed scale formats: per-tensor (scalar/1-element) from delayed
    scaling and per-block (2-D) from blockwise quantization.
    """
    if isinstance(scale_a, (int, float)):
        scale_a = torch.tensor(scale_a, dtype=torch.float32, device=a_fp8.device)
    if isinstance(scale_b, (int, float)):
        scale_b = torch.tensor(scale_b, dtype=torch.float32, device=b_fp8.device)
    scale_a = scale_a.to(dtype=torch.float32, device=a_fp8.device)
    scale_b = scale_b.to(dtype=torch.float32, device=b_fp8.device)
    return torch._scaled_mm(a_fp8, b_fp8, scale_a=scale_a, scale_b=scale_b)


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------

class QuantizedLinearFunction(torch.autograd.Function):
    """FP8 quantized linear: quant -> GEMM -> dequant, for both fwd and bwd.

    When ``quantize_activation`` is False, only the weight is quantized and
    the input is kept in original precision (weight-only FP8).  This matches
    the ``FP8_ACTIVATION=False`` mode in TE/MLPerf configs.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        scaling_manager,
        backend: str,
        fp8_dtype: torch.dtype,
        block_size: int,
        tensor_id: str = "weight",
        quantize_activation: bool = True,
    ) -> torch.Tensor:
        if not quantize_activation:
            weight_fp8, weight_scale = scaling_manager.quantize(tensor_id, weight)
            weight_dequant = weight_fp8.to(input.dtype) * weight_scale
            output = F.linear(input, weight_dequant, bias)
            ctx.save_for_backward(input, weight_fp8, weight_scale)
            ctx.scaling_manager = scaling_manager
            ctx.has_bias = bias is not None
            ctx.quantize_activation = False
            ctx.tensor_id = tensor_id
            return output

        if backend == "aiter":
            input_fp8, input_scale = _aiter_quant(input, fp8_dtype)
            weight_fp8, weight_scale = scaling_manager.quantize(tensor_id, weight)
            output = _aiter_mm(input_fp8, weight_fp8, input_scale, weight_scale)
        else:
            input_fp8, input_scale = _triton_quant(input, fp8_dtype, block_size)
            weight_fp8, weight_scale = scaling_manager.quantize(tensor_id, weight)
            output = _triton_mm(
                input_fp8.reshape(-1, input_fp8.shape[-1]),
                weight_fp8.t(),
                input_scale, weight_scale,
            )
            output = output.view(*input.shape[:-1], weight.shape[0])

        if bias is not None:
            output = output + bias

        ctx.save_for_backward(input_fp8, weight_fp8, input_scale, weight_scale)
        ctx.scaling_manager = scaling_manager
        ctx.backend = backend
        ctx.fp8_dtype = fp8_dtype
        ctx.block_size = block_size
        ctx.has_bias = bias is not None
        ctx.tensor_id = tensor_id
        ctx.quantize_activation = True
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not ctx.quantize_activation:
            input_tensor, weight_fp8, weight_scale = ctx.saved_tensors
            weight_dequant = weight_fp8.to(grad_output.dtype) * weight_scale
            grad_input = grad_output @ weight_dequant
            grad_weight = grad_output.reshape(-1, grad_output.shape[-1]).t() @ input_tensor.reshape(-1, input_tensor.shape[-1])
            grad_weight = ctx.scaling_manager.quantize_grad(grad_weight)
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1))) if ctx.has_bias else None
            return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

        input_fp8, weight_fp8, input_scale, weight_scale = ctx.saved_tensors
        backend = ctx.backend
        block_size = ctx.block_size

        # For HYBRID format the backward pass uses E5M2 dtype; the
        # ScalingManager stores the backward dtype separately.
        mgr = ctx.scaling_manager
        bwd_dtype = getattr(mgr, "fp8_dtype_bwd", ctx.fp8_dtype)

        if backend == "aiter":
            grad_fp8, grad_scale = _aiter_quant(grad_output, bwd_dtype)
            grad_input = _aiter_mm(grad_fp8, weight_fp8.t(), grad_scale, weight_scale)
            grad_weight = _aiter_mm(
                grad_fp8.reshape(-1, grad_output.shape[-1]).t(),
                input_fp8,
                grad_scale, input_scale,
            )
        else:
            grad_fp8, grad_scale = _triton_quant(grad_output, bwd_dtype, block_size)
            grad_flat = grad_fp8.reshape(-1, grad_output.shape[-1])
            input_flat = input_fp8.reshape(-1, input_fp8.shape[-1])
            grad_input = _triton_mm(grad_flat, weight_fp8, grad_scale, weight_scale)
            grad_input = grad_input.view_as(grad_output).view(*grad_output.shape[:-1], weight_fp8.shape[-1])
            grad_weight = _triton_mm(grad_flat.t(), input_flat, grad_scale, input_scale)

        grad_weight = ctx.scaling_manager.quantize_grad(grad_weight)

        grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1))) if ctx.has_bias else None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


_mark_allow_in_graph(QuantizedLinearFunction)


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def quantized_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    scaling_manager=None,
    backend: str = "aiter",
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    block_size: int = 128,
    tensor_id: str = "weight",
    quantize_activation: bool = True,
) -> torch.Tensor:
    """Functional quantized linear — mirrors ``attention()`` in the attention module.

    Gradient quantization is controlled by
    ``scaling_manager.config.quantize_grad`` (set via :class:`QuantConfig`).

    Args:
        input: Input tensor ``[*, in_features]``.
        weight: Weight matrix ``[out_features, in_features]``.
        bias: Optional bias ``[out_features]``.
        scaling_manager: A :class:`~transformer_light.quantize.ScalingManager`.
            Also carries the gradient quantization config.
        backend: ``"aiter"`` or ``"triton"``.
        fp8_dtype: Target FP8 dtype (default ``torch.float8_e4m3fn``).
        block_size: Block size for blockwise quantization (triton backend).
        tensor_id: Unique identifier for this layer's weight in the
            :class:`ScalingManager` amax history.  When called via
            ``quant.enable()``, each ``nn.Linear`` gets its own ID
            (e.g. ``"decoder.layers.0.mlp.linear_fc1.weight"``).
        quantize_activation: If ``True`` (default), quantize both input and
            weight.  If ``False``, only quantize the weight (weight-only FP8).

    Returns:
        Output tensor ``[*, out_features]``.
    """
    from transformer_light.quantize import is_aiter_available
    if quantize_activation and backend == "aiter" and not is_aiter_available():
        raise RuntimeError(
            "AITER is not installed. Install it or use backend='triton'."
        )

    if scaling_manager is None:
        from transformer_light.quantize import ScalingManager
        scaling_manager = ScalingManager()

    return QuantizedLinearFunction.apply(
        input, weight, bias, scaling_manager, backend, fp8_dtype, block_size,
        tensor_id, quantize_activation,
    )
