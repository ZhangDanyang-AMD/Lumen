###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Quantized linear forward + backward with explicit autograd and
multi-backend ASM → CK → Triton fallback.

All backends are AITER implementations — no torch.nn.functional fallbacks.

All AITER GEMM kernels follow TN layout convention:
    ``Y = X @ W^T``  where X is (M, K) and W is (N, K).

Supports all 6 scaling modes:
    - ``delayed``    — per-tensor FP8 (delayed scaling from amax history)
    - ``dynamic``    — per-tensor FP8 (current scaling from current amax)
    - ``per_token``  — per-row FP8 dynamic scaling
    - ``blockwise``  — per-block FP8 scaling (e.g. block=128)
    - ``mxfp8``      — microscaling FP8
    - ``none``       — BF16 passthrough (no quantization)

GEMM backends are selected automatically:
    - **ASM**: ``gemm_a8w8_asm``, ``gemm_a8w8_blockscale_bpreshuffle_asm``
    - **CK**: ``gemm_a8w8_ck``, ``gemm_a8w8_blockscale_ck``
    - **Triton**: ``gemm_a8w8`` (per-tensor), ``gemm_a8w8_blockscale``,
      ``gemm_a8w8_per_token_scale``, ``gemm_a16w16`` (BF16),
      ``gemm_mxfp8`` (MXFP8 with E8M0 scales)
    - **hipBLASLt**: ``hipb_mm`` (per-tensor)
"""

import logging as _logging
from typing import Optional

import torch

from lumen.ops.dispatch import (
    Backend,
    _probe_aiter_ck_gemm,
    _probe_aiter_quant,
    _probe_aiter_triton_gemm,
    _probe_aiter_triton_gemm_mxfp8,
    _probe_aiter_triton_quant,
    _probe_aiter_tuned_gemm_bf16,
    try_backends,
)
from lumen.quantize.config import _get_float8_e4m3

_logger = _logging.getLogger(__name__)

__all__ = ["QuantizedLinearFunction", "quantized_linear"]


def _mark_allow_in_graph(cls):
    try:
        from torch._dynamo import allow_in_graph

        allow_in_graph(cls)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Quantization helpers (all via AITER)
# ---------------------------------------------------------------------------


def _quant_per_tensor_hip(x, dtype):
    from aiter.ops.quant import per_tensor_quant_hip

    return per_tensor_quant_hip(x, quant_dtype=dtype)


def _quant_per_tensor_triton(x, dtype):
    from aiter.ops.quant import per_tensor_quant_triton

    return per_tensor_quant_triton(x, quant_dtype=dtype)


def _quant_per_token_hip(x, dtype):
    from aiter.ops.quant import per_token_quant_hip

    return per_token_quant_hip(x, quant_dtype=dtype)


def _quant_per_token_triton(x, dtype):
    from aiter.ops.quant import pertoken_quant

    return pertoken_quant(x, quant_dtype=dtype)


def _quant_blockwise(x, dtype, block_size=128):
    from lumen.ops.quantize.ops import quant_fp8_blockwise_impl

    orig_shape = x.shape
    flat = x.reshape(-1, orig_shape[-1]).contiguous()
    x_fp8, x_scales = quant_fp8_blockwise_impl(flat, dtype=dtype, axis=1, block_size=block_size)
    return x_fp8.view(orig_shape), x_scales


def quantize_input(x_2d, scaling_type, fp8_dtype, block_size=128, manager=None, tensor_id=None, backward=False):
    """Quantize input tensor according to scaling_type (all via AITER).

    Returns ``(x_quant, x_scale)`` where x_scale shape depends on mode:
    - per-tensor: ``(1,)``
    - per-token: ``(M, 1)``
    - blockwise: ``(M, ceil(N/block_size))``
    - mxfp8: ``(scales_shape,)``
    - none: ``(None, None)``
    """
    if scaling_type == "none":
        return x_2d, None

    if scaling_type == "delayed":
        if manager is not None:
            return manager.quantize(tensor_id or "input", x_2d, backward=backward)
        backends = []
        if _probe_aiter_quant():
            backends.append((Backend.CK, lambda: _quant_per_tensor_hip(x_2d, fp8_dtype)))
        if _probe_aiter_triton_quant():
            backends.append((Backend.TRITON, lambda: _quant_per_tensor_triton(x_2d, fp8_dtype)))
        return try_backends(backends, op_name="quant_delayed_per_tensor")

    if scaling_type == "dynamic":
        backends = []
        if _probe_aiter_quant():
            backends.append((Backend.CK, lambda: _quant_per_tensor_hip(x_2d, fp8_dtype)))
        if _probe_aiter_triton_quant():
            backends.append((Backend.TRITON, lambda: _quant_per_tensor_triton(x_2d, fp8_dtype)))
        return try_backends(backends, op_name="quant_per_tensor")

    if scaling_type == "per_token":
        backends = []
        if _probe_aiter_quant():
            backends.append((Backend.CK, lambda: _quant_per_token_hip(x_2d, fp8_dtype)))
        if _probe_aiter_triton_quant():
            backends.append((Backend.TRITON, lambda: _quant_per_token_triton(x_2d, fp8_dtype)))
        return try_backends(backends, op_name="quant_per_token")

    if scaling_type == "blockwise":
        return _quant_blockwise(x_2d, fp8_dtype, block_size)

    if scaling_type == "mxfp8":
        from lumen.ops.quantize.ops import convert_to_mxfp8

        mxfp8_block = 32 if block_size > 64 else block_size
        return convert_to_mxfp8(x_2d, block_size=mxfp8_block, axis=-1, float8_dtype_pt=fp8_dtype)

    raise ValueError(f"Unknown scaling_type={scaling_type!r}")


# ---------------------------------------------------------------------------
# GEMM dispatch with fallback (all via AITER)
#
# Convention: all AITER kernels compute Y = X @ W^T (TN layout).
#   - x: (M, K)    — activation / LHS
#   - w: (N, K)    — weight / RHS (internally transposed by the kernel)
#   - Y: (M, N)    — output
# ---------------------------------------------------------------------------


def _gemm_per_tensor_hipblas(a_fp8, w_fp8, scale_a, scale_w):
    """hipBLASLt per-tensor GEMM via AITER hipb_mm.

    hipb_mm computes mat1 @ mat2 (NN layout).  Lumen's dispatch convention
    passes w as (N, K), so we must transpose to (K, N) before calling hipb_mm.
    """
    from aiter.ops.gradlib import hipb_create_extension, hipb_mm

    import lumen.ops.quantize.linear as _self

    if not getattr(_self, "_hipblas_initialized", False):
        hipb_create_extension()
        _self._hipblas_initialized = True
    sa = (
        scale_a.float().reshape(1, 1)
        if isinstance(scale_a, torch.Tensor)
        else torch.tensor([[scale_a]], dtype=torch.float32, device=a_fp8.device)
    )
    sw = (
        scale_w.float().reshape(1, 1)
        if isinstance(scale_w, torch.Tensor)
        else torch.tensor([[scale_w]], dtype=torch.float32, device=w_fp8.device)
    )
    return hipb_mm(a_fp8, w_fp8.t().contiguous(), -1, out_dtype=torch.bfloat16, scaleA=sa, scaleB=sw)


def _gemm_per_tensor_ck(a_fp8, w_fp8, scale_a, scale_w):
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_CK

    return gemm_a8w8_CK(a_fp8, w_fp8, scale_a, scale_w)


def _gemm_per_tensor_triton(a_fp8, w_fp8, scale_a, scale_w):
    from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8

    return gemm_a8w8(a_fp8, w_fp8, scale_a, scale_w)


def gemm_per_tensor(a_fp8, w_fp8, scale_a, scale_w):
    """Per-tensor FP8 GEMM: Y = X @ W^T. All AITER backends."""
    backends = []
    if _probe_aiter_quant():
        backends.append((Backend.CK, lambda: _gemm_per_tensor_hipblas(a_fp8, w_fp8, scale_a, scale_w)))
    if _probe_aiter_ck_gemm():
        backends.append((Backend.CK, lambda: _gemm_per_tensor_ck(a_fp8, w_fp8, scale_a, scale_w)))
    if _probe_aiter_triton_gemm():
        backends.append((Backend.TRITON, lambda: _gemm_per_tensor_triton(a_fp8, w_fp8, scale_a, scale_w)))
    return try_backends(backends, op_name="gemm_per_tensor")


def _gemm_per_token_triton(a_fp8, w_fp8, scale_a, scale_w):
    from aiter.ops.triton.gemm.basic.gemm_a8w8_per_token_scale import gemm_a8w8_per_token_scale

    return gemm_a8w8_per_token_scale(a_fp8, w_fp8, scale_a, scale_w)


def gemm_per_token(a_fp8, w_fp8, scale_a, scale_w):
    """Per-token FP8 GEMM: Y = X @ W^T via AITER Triton."""
    backends = []
    if _probe_aiter_triton_gemm():
        backends.append((Backend.TRITON, lambda: _gemm_per_token_triton(a_fp8, w_fp8, scale_a, scale_w)))
    return try_backends(backends, op_name="gemm_per_token")


def _gemm_blockscale_ck(a_fp8, w_fp8, scale_a, scale_w):
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale

    return gemm_a8w8_blockscale(a_fp8, w_fp8, scale_a, scale_w)


def _gemm_blockscale_triton(a_fp8, w_fp8, scale_a, scale_w):
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale

    return gemm_a8w8_blockscale(a_fp8, w_fp8, scale_a, scale_w)


def gemm_blockscale(a_fp8, w_fp8, scale_a, scale_w):
    """Blockwise FP8 GEMM: Y = X @ W^T via AITER CK → Triton."""
    backends = []
    if _probe_aiter_ck_gemm():
        backends.append((Backend.CK, lambda: _gemm_blockscale_ck(a_fp8, w_fp8, scale_a, scale_w)))
    if _probe_aiter_triton_gemm():
        backends.append((Backend.TRITON, lambda: _gemm_blockscale_triton(a_fp8, w_fp8, scale_a, scale_w)))
    return try_backends(backends, op_name="gemm_blockscale")


def _gemm_mxfp8_triton(a_fp8, w_fp8, scale_a, scale_w):
    from aiter.ops.triton.gemm.basic.gemm_mxfp8 import gemm_mxfp8

    block_size = a_fp8.shape[1] // scale_a.shape[-1]
    return gemm_mxfp8(a_fp8, w_fp8, scale_a, scale_w, quant_block_size=block_size)


def gemm_mxfp8(a_fp8, w_fp8, scale_a, scale_w):
    """MXFP8 GEMM: Y = X @ W^T with E8M0 block scales via AITER Triton."""
    backends = []
    if _probe_aiter_triton_gemm_mxfp8():
        backends.append((Backend.TRITON, lambda: _gemm_mxfp8_triton(a_fp8, w_fp8, scale_a, scale_w)))
    return try_backends(backends, op_name="gemm_mxfp8")


def _gemm_bf16_tuned(a, w, bias):
    from aiter.tuned_gemm import gemm_a16w16

    return gemm_a16w16(a, w, bias=bias)


def gemm_bf16(a, w, bias=None):
    """BF16 GEMM: Y = X @ W^T via AITER tuned_gemm (hipBLASLt / ASM / Triton)."""
    backends = []
    if _probe_aiter_tuned_gemm_bf16():
        backends.append((Backend.CK, lambda: _gemm_bf16_tuned(a, w, bias)))
    return try_backends(backends, op_name="gemm_bf16")


def dispatch_gemm(a, w, scale_a, scale_w, scaling_type, bias=None):
    """Route GEMM to the appropriate AITER backend based on scaling_type.

    All kernels compute ``Y = A @ W^T`` (TN layout).  ``w`` must have
    shape ``(N, K)`` — same as PyTorch Linear weight convention.

    For backward dgrad, pass ``weight.t().contiguous()`` as ``w`` so that
    ``A @ W^T = grad @ weight``.

    Args:
        a: Activation / LHS tensor ``(M, K)``.
        w: Weight / RHS tensor ``(N, K)``.
        scale_a: Activation scale.
        scale_w: Weight scale.
        scaling_type: One of the 6 supported modes.
        bias: Optional bias.

    Returns:
        Output tensor ``(M, N)``.
    """
    if scaling_type == "none":
        return gemm_bf16(a, w, bias)

    if scaling_type in ("delayed", "dynamic"):
        out = gemm_per_tensor(a, w, scale_a, scale_w)
    elif scaling_type == "per_token":
        out = gemm_per_token(a, w, scale_a, scale_w)
    elif scaling_type == "blockwise":
        out = gemm_blockscale(a, w, scale_a, scale_w)
    elif scaling_type == "mxfp8":
        out = gemm_mxfp8(a, w, scale_a, scale_w)
    else:
        raise ValueError(f"Unknown scaling_type={scaling_type!r}")

    if bias is not None:
        out = out + bias
    return out


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------


class QuantizedLinearFunction(torch.autograd.Function):
    """FP8 quantized linear: quant -> GEMM -> dequant, for both fwd and bwd.

    Supports all 6 scaling modes via ``scaling_type`` parameter.
    Backend selection uses ASM → CK → Triton fallback automatically.
    All backends are AITER implementations.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        scaling_manager,
        scaling_type: str,
        fp8_dtype: torch.dtype,
        block_size: int,
        tensor_id: str = "weight",
        quantize_activation: bool = True,
        fp8_wgrad: bool = True,
    ) -> torch.Tensor:
        # weight is [N, K] — standard PyTorch Linear convention
        if scaling_type == "none":
            output = gemm_bf16(input, weight, bias)
            ctx.save_for_backward(input, weight)
            ctx.has_bias = bias is not None
            ctx.scaling_type = "none"
            return output

        if not quantize_activation:
            weight_fp8, weight_scale = quantize_input(
                weight,
                scaling_type,
                fp8_dtype,
                block_size,
                scaling_manager,
                tensor_id,
            )
            weight_dequant = (weight_fp8.to(input.dtype) * weight_scale).to(input.dtype)
            output = gemm_bf16(input, weight_dequant, bias)
            ctx.save_for_backward(input, weight_fp8, weight_scale)
            ctx.scaling_manager = scaling_manager
            ctx.has_bias = bias is not None
            ctx.quantize_activation = False
            ctx.scaling_type = scaling_type
            ctx.fp8_wgrad = True
            ctx.tensor_id = tensor_id
            return output

        input_2d = input.reshape(-1, input.shape[-1]).contiguous()

        input_fp8, input_scale = quantize_input(
            input_2d,
            scaling_type,
            fp8_dtype,
            block_size,
        )
        weight_fp8, weight_scale = quantize_input(
            weight,
            scaling_type,
            fp8_dtype,
            block_size,
            scaling_manager,
            tensor_id,
        )

        # Forward: Y = input @ weight^T  (TN layout, weight is [N, K])
        output = dispatch_gemm(input_fp8, weight_fp8, input_scale, weight_scale, scaling_type)
        output = output.view(*input.shape[:-1], weight.shape[0])

        if bias is not None:
            output = output + bias

        ctx.save_for_backward(input_fp8, weight_fp8, input_scale, weight_scale)
        ctx.scaling_manager = scaling_manager
        ctx.scaling_type = scaling_type
        ctx.fp8_dtype = fp8_dtype
        ctx.block_size = block_size
        ctx.has_bias = bias is not None
        ctx.tensor_id = tensor_id
        ctx.quantize_activation = True
        ctx.fp8_wgrad = fp8_wgrad
        ctx.input_shape = input.shape
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        scaling_type = ctx.scaling_type

        if scaling_type == "none":
            input_tensor, weight = ctx.saved_tensors
            # dgrad: grad @ weight = dispatch(grad, weight^T) since kernel does A @ W^T
            grad_input = dispatch_gemm(
                grad_output,
                weight.t().contiguous(),
                None,
                None,
                "none",
            )
            # wgrad: grad^T @ input = dispatch(grad^T, input^T) since (grad^T) @ (input^T)^T = grad^T @ input
            grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
            input_flat = input_tensor.reshape(-1, input_tensor.shape[-1])
            grad_weight = dispatch_gemm(
                grad_flat.t().contiguous(),
                input_flat.t().contiguous(),
                None,
                None,
                "none",
            )
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1))) if ctx.has_bias else None
            return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

        if not ctx.quantize_activation:
            input_tensor, weight_fp8, weight_scale = ctx.saved_tensors
            weight_dequant = (weight_fp8.to(grad_output.dtype) * weight_scale).to(grad_output.dtype)
            grad_input = dispatch_gemm(
                grad_output,
                weight_dequant.t().contiguous(),
                None,
                None,
                "none",
            )
            grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
            input_flat = input_tensor.reshape(-1, input_tensor.shape[-1])
            grad_weight = dispatch_gemm(
                grad_flat.t().contiguous(),
                input_flat.t().contiguous(),
                None,
                None,
                "none",
            )
            if ctx.scaling_manager is not None:
                grad_weight = ctx.scaling_manager.quantize_grad(grad_weight)
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1))) if ctx.has_bias else None
            return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

        input_fp8, weight_fp8, input_scale, weight_scale = ctx.saved_tensors
        fp8_dtype = ctx.fp8_dtype
        block_size = ctx.block_size

        mgr = ctx.scaling_manager
        bwd_dtype = getattr(mgr, "fp8_dtype_bwd", fp8_dtype) if mgr else fp8_dtype

        grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

        bwd_scaling = "dynamic" if scaling_type in ("per_token", "blockwise") else scaling_type
        grad_fp8, grad_scale = quantize_input(grad_flat, bwd_scaling, bwd_dtype, block_size)

        # dgrad: grad @ weight  →  dispatch(grad, weight^T) since kernel does A @ W^T
        grad_input = dispatch_gemm(
            grad_fp8,
            weight_fp8.t().contiguous(),
            grad_scale,
            weight_scale,
            bwd_scaling,
        )
        grad_input = grad_input.view(*grad_output.shape[:-1], weight_fp8.shape[-1])

        # wgrad: grad^T @ input  →  dispatch(grad^T, input^T)
        # AITER's per-tensor FP8 GEMM backends (hipBLASLt / CK) crash with
        # uncatchable SIGABRT on transposed wgrad tensors regardless of
        # dimension, so always dequantize to BF16 before computing wgrad.
        # mxfp8 / blockwise wgrad would use different GEMM paths and may
        # support FP8 wgrad in the future.
        _MIN_FP8_K = 64
        wgrad_k = grad_fp8.shape[0]
        _use_fp8_wgrad = ctx.fp8_wgrad and wgrad_k >= _MIN_FP8_K and bwd_scaling not in ("delayed", "dynamic")
        if _use_fp8_wgrad:
            grad_weight = dispatch_gemm(
                grad_fp8.t().contiguous(),
                input_fp8.t().contiguous(),
                grad_scale,
                input_scale,
                bwd_scaling,
            )
        else:
            grad_bf16 = grad_fp8.to(torch.bfloat16) * grad_scale
            input_bf16 = input_fp8.to(torch.bfloat16) * input_scale
            grad_weight = dispatch_gemm(
                grad_bf16.t().contiguous(),
                input_bf16.t().contiguous(),
                None,
                None,
                "none",
            )

        if mgr is not None:
            grad_weight = mgr.quantize_grad(grad_weight)

        grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1))) if ctx.has_bias else None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None


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
    backend: str = "auto",
    scaling_type: str = "delayed",
    fp8_dtype: Optional[torch.dtype] = None,
    block_size: int = 128,
    tensor_id: str = "weight",
    quantize_activation: bool = True,
    fp8_wgrad: bool = True,
) -> torch.Tensor:
    """Functional quantized linear with multi-backend fallback (all AITER).

    Args:
        input: Input tensor ``[*, in_features]``.
        weight: Weight matrix ``[out_features, in_features]``.
        bias: Optional bias ``[out_features]``.
        scaling_manager: A :class:`~lumen.quantize.ScalingManager`.
        backend: Legacy parameter (ignored, auto-fallback is always used).
        scaling_type: One of ``"delayed"``, ``"dynamic"``, ``"per_token"``,
            ``"blockwise"``, ``"mxfp8"``, ``"none"``.
        fp8_dtype: Target FP8 dtype.  ``None`` auto-detects based on GPU
            architecture (``float8_e4m3fnuz`` on gfx942, ``float8_e4m3fn``
            on gfx950+).
        block_size: Block size for blockwise/MXFP8 quantization.
        tensor_id: Unique identifier for this layer's weight.
        quantize_activation: If ``True``, quantize both input and weight.
        fp8_wgrad: If ``True``, compute weight gradient in FP8.

    Returns:
        Output tensor ``[*, out_features]``.
    """
    if fp8_dtype is None:
        fp8_dtype = _get_float8_e4m3()
        _logger.info("quantized_linear: auto-detected fp8_dtype=%s", fp8_dtype)

    if scaling_manager is None:
        from lumen.quantize import ScalingManager

        scaling_manager = ScalingManager(fp8_dtype=fp8_dtype)

    return QuantizedLinearFunction.apply(
        input,
        weight,
        bias,
        scaling_manager,
        scaling_type,
        fp8_dtype,
        block_size,
        tensor_id,
        quantize_activation,
        fp8_wgrad,
    )
