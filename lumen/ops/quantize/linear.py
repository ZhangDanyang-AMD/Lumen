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

Supports all 7 scaling modes:
    - ``delayed``      — per-tensor FP8 (delayed scaling from amax history)
    - ``dynamic``      — per-tensor FP8 (current scaling from current amax)
    - ``per_token``    — per-row FP8 dynamic scaling
    - ``blockwise``    — per-block FP8 scaling (e.g. block=128)
    - ``blockwise2d``  — 2D block FP8 scaling (same kernel, 2D scale management)
    - ``mxfp8``        — microscaling FP8
    - ``none``         — BF16 passthrough (no quantization)

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
    _probe_aiter_hipblas,
    _probe_aiter_quant,
    _probe_aiter_triton_gemm,
    _probe_aiter_triton_gemm_mxfp8,
    _probe_aiter_triton_quant,
    _probe_aiter_tuned_gemm_bf16,
    try_backends,
)
from lumen.quantize.config import _get_float8_e4m3
from lumen.quantize.descriptor import FP8Descriptor

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


def quantize_input(
    x_2d,
    scaling_type,
    fp8_dtype,
    block_size=128,
    manager=None,
    tensor_id=None,
    backward=False,
) -> Optional[FP8Descriptor]:
    """Quantize input tensor according to scaling_type (all via AITER).

    Returns an :class:`~lumen.quantize.descriptor.FP8Descriptor` bundling ``data`` and
    ``scale``, or ``None`` when ``scaling_type == "none"`` (BF16 passthrough).

    Scale tensor shapes by mode:
    - per-tensor: ``(1,)``
    - per-token: ``(M, 1)``
    - blockwise / blockwise2d: ``(M, ceil(N/block_size))``
    - mxfp8: ``(scales_shape,)``
    """
    if scaling_type == "none":
        return None

    if scaling_type == "delayed":
        if manager is not None:
            result = manager.quantize(tensor_id or "input", x_2d, backward=backward)
            if isinstance(result, tuple):
                return FP8Descriptor(data=result[0], scale=result[1], fp8_dtype=fp8_dtype)
            return result
        backends = []
        if _probe_aiter_quant():
            backends.append((Backend.CK, lambda: _quant_per_tensor_hip(x_2d, fp8_dtype)))
        if _probe_aiter_triton_quant():
            backends.append((Backend.TRITON, lambda: _quant_per_tensor_triton(x_2d, fp8_dtype)))
        x_fp8, x_scale = try_backends(backends, op_name="quant_delayed_per_tensor")
        return FP8Descriptor(data=x_fp8, scale=x_scale, fp8_dtype=fp8_dtype)

    if scaling_type == "dynamic":
        backends = []
        if _probe_aiter_quant():
            backends.append((Backend.CK, lambda: _quant_per_tensor_hip(x_2d, fp8_dtype)))
        if _probe_aiter_triton_quant():
            backends.append((Backend.TRITON, lambda: _quant_per_tensor_triton(x_2d, fp8_dtype)))
        x_fp8, x_scale = try_backends(backends, op_name="quant_per_tensor")
        return FP8Descriptor(data=x_fp8, scale=x_scale, fp8_dtype=fp8_dtype)

    if scaling_type == "per_token":
        backends = []
        if _probe_aiter_quant():
            backends.append((Backend.CK, lambda: _quant_per_token_hip(x_2d, fp8_dtype)))
        if _probe_aiter_triton_quant():
            backends.append((Backend.TRITON, lambda: _quant_per_token_triton(x_2d, fp8_dtype)))
        x_fp8, x_scale = try_backends(backends, op_name="quant_per_token")
        return FP8Descriptor(data=x_fp8, scale=x_scale, fp8_dtype=fp8_dtype)

    if scaling_type in ("blockwise", "blockwise2d"):
        x_fp8, x_scale = _quant_blockwise(x_2d, fp8_dtype, block_size)
        return FP8Descriptor(data=x_fp8, scale=x_scale, fp8_dtype=fp8_dtype)

    if scaling_type == "mxfp8":
        from lumen.ops.quantize.ops import convert_to_mxfp8
        from lumen.ops.quantize.padding import pad_to_block

        mxfp8_block = 32 if block_size > 64 else block_size
        x_2d, _orig_m = pad_to_block(x_2d, mxfp8_block, dim=0)
        x_2d, _orig_n = pad_to_block(x_2d, mxfp8_block, dim=-1)
        x_fp8, x_scale = convert_to_mxfp8(x_2d, block_size=mxfp8_block, axis=-1, float8_dtype_pt=fp8_dtype)
        return FP8Descriptor(data=x_fp8, scale=x_scale, fp8_dtype=fp8_dtype)

    raise ValueError(f"Unknown scaling_type={scaling_type!r}")


def _fp8_store_activation(input_2d, fp8_dtype):
    """Quantize activation for memory-efficient storage in save_for_backward."""
    desc = quantize_input(input_2d, "dynamic", fp8_dtype)
    return desc.data, desc.scale


def _fp8_restore_activation(input_fp8, scale, target_dtype):
    """Dequantize stored FP8 activation. Convention: dequant = fp8 * scale."""
    return (input_fp8.to(torch.float32) * scale).to(target_dtype)


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


def _expand_per_tensor_scale(scale, size):
    """Broadcast a per-tensor scale (1,) to a contiguous per-row vector.

    AITER's gemm_a8w8 kernels (CK and Triton) index into scales with
    per-row offsets.  A (1,)-shaped tensor causes out-of-bounds reads.
    """
    return scale.float().reshape(1).expand(size).contiguous()


def _gemm_per_tensor_ck(a_fp8, w_fp8, scale_a, scale_w):
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_CK

    M, N = a_fp8.shape[0], w_fp8.shape[0]
    sa = _expand_per_tensor_scale(scale_a, M).unsqueeze(1)
    sw = _expand_per_tensor_scale(scale_w, N).unsqueeze(1)
    return gemm_a8w8_CK(a_fp8, w_fp8, sa, sw)


def _gemm_per_tensor_triton(a_fp8, w_fp8, scale_a, scale_w):
    from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8

    M, N = a_fp8.shape[0], w_fp8.shape[0]
    sa = _expand_per_tensor_scale(scale_a, M)
    sw = _expand_per_tensor_scale(scale_w, N)
    return gemm_a8w8(a_fp8, w_fp8, sa, sw)


def gemm_per_tensor(a_fp8, w_fp8, scale_a, scale_w):
    """Per-tensor FP8 GEMM: Y = X @ W^T. All AITER backends.

    hipBLASLt is tried last because ``hipb_create_extension`` / ``hipb_mm``
    can SIGSEGV in ``mp.spawn`` child processes (known multi-GPU issue in the
    gradlib C++ globals).  CK and Triton are safe to run first.
    """
    backends = []
    if _probe_aiter_ck_gemm():
        backends.append((Backend.CK, lambda: _gemm_per_tensor_ck(a_fp8, w_fp8, scale_a, scale_w)))
    if _probe_aiter_triton_gemm():
        backends.append((Backend.TRITON, lambda: _gemm_per_tensor_triton(a_fp8, w_fp8, scale_a, scale_w)))
    if _probe_aiter_hipblas():
        backends.append((Backend.HIPBLAS, lambda: _gemm_per_tensor_hipblas(a_fp8, w_fp8, scale_a, scale_w)))
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


def dispatch_gemm(a, w, scale_a=None, scale_w=None, scaling_type="none", bias=None):
    """Route GEMM to the appropriate AITER backend based on scaling_type.

    All kernels compute ``Y = A @ W^T`` (TN layout).  ``w`` must have
    shape ``(N, K)`` — same as PyTorch Linear weight convention.

    For backward dgrad, pass ``weight.t().contiguous()`` as ``w`` so that
    ``A @ W^T = grad @ weight``.

    ``a`` and ``w`` may each be a plain :class:`torch.Tensor` or an
    :class:`~lumen.quantize.descriptor.FP8Descriptor`.  When an argument
    is a descriptor, its ``.data`` and ``.scale`` fields are unpacked
    automatically, overriding the corresponding ``scale_*`` parameter.

    Args:
        a: Activation / LHS tensor ``(M, K)``, or :class:`FP8Descriptor`.
        w: Weight / RHS tensor ``(N, K)``, or :class:`FP8Descriptor`.
        scale_a: Activation scale (ignored for that side if ``a`` is a descriptor).
        scale_w: Weight scale (ignored for that side if ``w`` is a descriptor).
        scaling_type: One of the 6 supported modes.
        bias: Optional bias.

    Returns:
        Output tensor ``(M, N)``.
    """
    if isinstance(a, FP8Descriptor):
        scale_a = a.scale
        a = a.data
    if isinstance(w, FP8Descriptor):
        scale_w = w.scale
        w = w.data

    if scaling_type == "none":
        return gemm_bf16(a, w, bias)

    if scaling_type in ("delayed", "dynamic"):
        out = gemm_per_tensor(a, w, scale_a, scale_w)
    elif scaling_type == "per_token":
        out = gemm_per_token(a, w, scale_a, scale_w)
    elif scaling_type in ("blockwise", "blockwise2d"):
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

    Supports all 7 scaling modes via ``scaling_type`` parameter.
    Backend selection uses ASM → CK → Triton fallback automatically.
    All backends are AITER implementations.

    When ``delay_wgrad=True``, the backward pass computes only dgrad
    (input gradient) and defers the wgrad computation to a later
    ``deferred_wgrad.execute()`` call.  This enables overlapping the
    deferred wgrad GEMM with the next layer's communication.
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
        gradient_accumulation_fusion: bool = False,
        delay_wgrad: bool = False,
        deferred_wgrad=None,
        fp8_activation_store: bool = False,
        activation_tensor_id: Optional[str] = None,
    ) -> torch.Tensor:
        # weight is [N, K] — standard PyTorch Linear convention
        if scaling_type == "none":
            output = gemm_bf16(input, weight, bias)
            if fp8_activation_store:
                input_2d = input.reshape(-1, input.shape[-1]).contiguous()
                input_store, input_scale = _fp8_store_activation(input_2d, fp8_dtype)
                ctx.save_for_backward(input_store, input_scale, weight)
                ctx._input_shape = input.shape
            else:
                ctx.save_for_backward(input, weight)
            ctx.fp8_activation_store = fp8_activation_store
            ctx.has_bias = bias is not None
            ctx.scaling_type = "none"
            ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
            ctx.delay_wgrad = delay_wgrad
            ctx.deferred_wgrad = deferred_wgrad
            ctx.weight_ref = weight
            return output

        if not quantize_activation:
            weight_desc = quantize_input(
                weight,
                scaling_type,
                fp8_dtype,
                block_size,
                scaling_manager,
                tensor_id,
            )
            weight_dequant = (weight_desc.data.to(input.dtype) * weight_desc.scale).to(input.dtype)
            output = gemm_bf16(input, weight_dequant, bias)
            if fp8_activation_store:
                input_2d = input.reshape(-1, input.shape[-1]).contiguous()
                input_store, input_scale = _fp8_store_activation(input_2d, fp8_dtype)
                ctx.save_for_backward(input_store, input_scale, weight_desc.data, weight_desc.scale)
                ctx._input_shape = input.shape
            else:
                ctx.save_for_backward(input, weight_desc.data, weight_desc.scale)
            ctx.fp8_activation_store = fp8_activation_store
            ctx.scaling_manager = scaling_manager
            ctx.has_bias = bias is not None
            ctx.quantize_activation = False
            ctx.scaling_type = scaling_type
            ctx.fp8_wgrad = True
            ctx.tensor_id = tensor_id
            ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
            ctx.delay_wgrad = delay_wgrad
            ctx.deferred_wgrad = deferred_wgrad
            ctx.weight_ref = weight
            return output

        input_2d = input.reshape(-1, input.shape[-1]).contiguous()

        _act_mgr = scaling_manager if activation_tensor_id else None
        _act_tid = activation_tensor_id or "activation"
        input_desc = quantize_input(
            input_2d,
            scaling_type,
            fp8_dtype,
            block_size,
            _act_mgr,
            _act_tid,
        )
        weight_desc = quantize_input(
            weight,
            scaling_type,
            fp8_dtype,
            block_size,
            scaling_manager,
            tensor_id,
        )

        # Forward: Y = input @ weight^T  (TN layout, weight is [N, K])
        output = dispatch_gemm(input_desc, weight_desc, scaling_type=scaling_type)
        output = output.view(*input.shape[:-1], weight.shape[0])

        if bias is not None:
            output = output + bias

        ctx.save_for_backward(
            input_desc.data,
            input_desc.scale,
            weight_desc.data,
            weight_desc.scale,
        )
        ctx.fp8_activation_store = False
        ctx.scaling_manager = scaling_manager
        ctx.scaling_type = scaling_type
        ctx.fp8_dtype = fp8_dtype
        ctx.block_size = block_size
        ctx.has_bias = bias is not None
        ctx.tensor_id = tensor_id
        ctx.quantize_activation = True
        ctx.fp8_wgrad = fp8_wgrad
        ctx.input_shape = input.shape
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.delay_wgrad = delay_wgrad
        ctx.deferred_wgrad = deferred_wgrad
        ctx.weight_ref = weight
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        scaling_type = ctx.scaling_type

        if scaling_type == "none":
            if ctx.fp8_activation_store:
                input_store, input_scale, weight = ctx.saved_tensors
                input_tensor = _fp8_restore_activation(input_store, input_scale, grad_output.dtype)
                input_tensor = input_tensor.view(ctx._input_shape)
            else:
                input_tensor, weight = ctx.saved_tensors
            grad_input = dispatch_gemm(
                grad_output,
                weight.t().contiguous(),
                None,
                None,
                "none",
            )

            if ctx.delay_wgrad and ctx.deferred_wgrad is not None:
                grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
                input_flat = input_tensor.reshape(-1, input_tensor.shape[-1]).contiguous()
                w_ref = ctx.weight_ref
                gaf = ctx.gradient_accumulation_fusion

                def _wgrad_fn():
                    gw = dispatch_gemm(
                        grad_flat.t().contiguous(),
                        input_flat.t().contiguous(),
                        None,
                        None,
                        "none",
                    )
                    if gaf and hasattr(w_ref, "main_grad"):
                        w_ref.main_grad.add_(gw)
                    elif w_ref.grad is not None:
                        w_ref.grad.add_(gw)
                    else:
                        w_ref.grad = gw

                ctx.deferred_wgrad.defer(_wgrad_fn)
                grad_weight = None
            else:
                grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
                input_flat = input_tensor.reshape(-1, input_tensor.shape[-1])
                grad_weight = dispatch_gemm(
                    grad_flat.t().contiguous(),
                    input_flat.t().contiguous(),
                    None,
                    None,
                    "none",
                )
                if ctx.gradient_accumulation_fusion and hasattr(ctx.weight_ref, "main_grad"):
                    ctx.weight_ref.main_grad.add_(grad_weight)
                    grad_weight = None

            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1))) if ctx.has_bias else None
            return (
                grad_input,
                grad_weight,
                grad_bias,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        if not ctx.quantize_activation:
            if ctx.fp8_activation_store:
                input_store, input_scale, weight_fp8, weight_scale = ctx.saved_tensors
                input_tensor = _fp8_restore_activation(input_store, input_scale, grad_output.dtype)
                input_tensor = input_tensor.view(ctx._input_shape)
            else:
                input_tensor, weight_fp8, weight_scale = ctx.saved_tensors
            weight_dequant = (weight_fp8.to(grad_output.dtype) * weight_scale).to(grad_output.dtype)
            grad_input = dispatch_gemm(
                grad_output,
                weight_dequant.t().contiguous(),
                None,
                None,
                "none",
            )

            if ctx.delay_wgrad and ctx.deferred_wgrad is not None:
                grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
                input_flat = input_tensor.reshape(-1, input_tensor.shape[-1]).contiguous()
                mgr = ctx.scaling_manager
                w_ref = ctx.weight_ref
                gaf = ctx.gradient_accumulation_fusion

                def _wgrad_fn():
                    gw = dispatch_gemm(
                        grad_flat.t().contiguous(),
                        input_flat.t().contiguous(),
                        None,
                        None,
                        "none",
                    )
                    if mgr is not None:
                        gw = mgr.quantize_grad(gw)
                    if gaf and hasattr(w_ref, "main_grad"):
                        w_ref.main_grad.add_(gw)
                    elif w_ref.grad is not None:
                        w_ref.grad.add_(gw)
                    else:
                        w_ref.grad = gw

                ctx.deferred_wgrad.defer(_wgrad_fn)
                grad_weight = None
            else:
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
                if ctx.gradient_accumulation_fusion and hasattr(ctx.weight_ref, "main_grad"):
                    ctx.weight_ref.main_grad.add_(grad_weight)
                    grad_weight = None

            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1))) if ctx.has_bias else None
            return (
                grad_input,
                grad_weight,
                grad_bias,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        input_data, input_scale, weight_data, weight_scale = ctx.saved_tensors
        fp8_dtype = ctx.fp8_dtype
        block_size = ctx.block_size
        weight_desc = FP8Descriptor.from_tensors(weight_data, weight_scale, fp8_dtype)

        mgr = ctx.scaling_manager
        bwd_dtype = getattr(mgr, "fp8_dtype_bwd", fp8_dtype) if mgr else fp8_dtype
        if bwd_dtype != fp8_dtype:
            bwd_dtype = fp8_dtype

        grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

        if scaling_type == "blockwise2d" and mgr is not None:
            bwd_scaling = "dynamic"
            grad_fp8, grad_scale = mgr.quantize_bwd_delayed(
                (ctx.tensor_id or "linear") + "_bwd",
                grad_flat,
            )
        else:
            bwd_scaling = "dynamic" if scaling_type in ("per_token", "blockwise", "blockwise2d") else scaling_type
            grad_desc = quantize_input(grad_flat, bwd_scaling, bwd_dtype, block_size)
            grad_fp8, grad_scale = grad_desc.data, grad_desc.scale

        # dgrad: grad @ weight  →  dispatch(grad, weight^T) since kernel does A @ W^T
        # Fall back to BF16 dequant when:
        #  1. Hybrid mode (E4M3 fwd / E5M2 bwd) — AITER FP8 GEMM kernels
        #     don't support mixed-dtype operands.
        #  2. Forward used per_token / blockwise quantization — weight_scale
        #     is multi-element and incompatible with the per-tensor dgrad GEMM.
        # Dequant weight BEFORE transposing so scale dimensions align.
        _mixed_dtype = weight_data.dtype != grad_fp8.dtype
        _needs_dequant = scaling_type in ("per_token", "blockwise", "blockwise2d")
        if _needs_dequant:
            from lumen.ops.quantize.gemm_primitives import _dequant_fp8_weight

            grad_bf16 = (grad_fp8.bfloat16() * grad_scale.bfloat16()).contiguous()
            weight_bf16 = _dequant_fp8_weight(weight_data, weight_scale, block_size).bfloat16()
            grad_input = dispatch_gemm(grad_bf16, weight_bf16.t().contiguous(), None, None, "none")
        elif _mixed_dtype:
            grad_bf16 = (grad_fp8.bfloat16() * grad_scale.bfloat16()).contiguous()
            weight_bf16 = (weight_data.bfloat16() * weight_scale.bfloat16()).contiguous()
            grad_input = dispatch_gemm(grad_bf16, weight_bf16.t().contiguous(), None, None, "none")
            del grad_bf16, weight_bf16
        else:
            grad_input = dispatch_gemm(
                grad_fp8,
                weight_desc.transpose_cached,
                grad_scale,
                weight_desc.scale,
                bwd_scaling,
            )
        grad_input = grad_input.view(*grad_output.shape[:-1], weight_data.shape[-1])

        # wgrad: grad^T @ input  →  dispatch(grad^T, input^T)
        # AITER's per-tensor FP8 GEMM backends (hipBLASLt / CK) crash with
        # uncatchable SIGABRT on transposed wgrad tensors regardless of
        # dimension, so always dequantize to BF16 before computing wgrad.
        # In hybrid mode, the same mixed-dtype constraint applies.
        _MIN_FP8_K = 64
        wgrad_k = grad_fp8.shape[0]
        _use_fp8_wgrad = ctx.fp8_wgrad and wgrad_k >= _MIN_FP8_K and bwd_scaling not in ("delayed", "dynamic")

        if ctx.delay_wgrad and ctx.deferred_wgrad is not None:
            _use_fp8 = _use_fp8_wgrad
            _grad_fp8 = grad_fp8
            _input_data = input_data
            _grad_scale = grad_scale
            _input_scale = input_scale
            _bwd_scaling = bwd_scaling
            _mgr = mgr
            _w_ref = ctx.weight_ref
            _gaf = ctx.gradient_accumulation_fusion

            def _wgrad_fn():
                if _use_fp8:
                    _g = _grad_fp8
                    _gs = _grad_scale
                    if _g.dtype != _input_data.dtype:
                        _recast = quantize_input(
                            (_g.bfloat16() * _gs.bfloat16()).contiguous().reshape(-1, _g.shape[-1]),
                            _bwd_scaling,
                            _input_data.dtype,
                            block_size,
                        )
                        _g, _gs = _recast.data, _recast.scale
                    gw = dispatch_gemm(
                        _g.t().contiguous(),
                        _input_data.t().contiguous(),
                        _gs,
                        _input_scale,
                        _bwd_scaling,
                    )
                else:
                    g_bf16 = (_grad_fp8.bfloat16() * _grad_scale.bfloat16()).contiguous()
                    i_bf16 = (_input_data.bfloat16() * _input_scale.bfloat16()).contiguous()
                    gw = dispatch_gemm(
                        g_bf16.t().contiguous(),
                        i_bf16.t().contiguous(),
                        None,
                        None,
                        "none",
                    )
                if _mgr is not None:
                    gw = _mgr.quantize_grad(gw)
                if _gaf and hasattr(_w_ref, "main_grad"):
                    _w_ref.main_grad.add_(gw)
                elif _w_ref.grad is not None:
                    _w_ref.grad.add_(gw)
                else:
                    _w_ref.grad = gw

            ctx.deferred_wgrad.defer(_wgrad_fn)
            grad_weight = None
        else:
            if _use_fp8_wgrad:
                _g = grad_fp8
                _gs = grad_scale
                if _g.dtype != input_data.dtype:
                    _recast = quantize_input(
                        (_g.bfloat16() * _gs.bfloat16()).contiguous().reshape(-1, _g.shape[-1]),
                        bwd_scaling,
                        input_data.dtype,
                        block_size,
                    )
                    _g, _gs = _recast.data, _recast.scale
                grad_weight = dispatch_gemm(
                    _g.t().contiguous(),
                    input_data.t().contiguous(),
                    _gs,
                    input_scale,
                    bwd_scaling,
                )
            else:
                grad_bf16 = (grad_fp8.bfloat16() * grad_scale.bfloat16()).contiguous()
                input_bf16 = (input_data.bfloat16() * input_scale.bfloat16()).contiguous()
                grad_weight = dispatch_gemm(
                    grad_bf16.t().contiguous(),
                    input_bf16.t().contiguous(),
                    None,
                    None,
                    "none",
                )

            if mgr is not None:
                grad_weight = mgr.quantize_grad(grad_weight)

            if ctx.gradient_accumulation_fusion and hasattr(ctx.weight_ref, "main_grad"):
                ctx.weight_ref.main_grad.add_(grad_weight)
                grad_weight = None

        grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1))) if ctx.has_bias else None

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


_mark_allow_in_graph(QuantizedLinearFunction)


class FP8StoredLinearFunction(torch.autograd.Function):
    """Linear for weights already stored in FP8 (FP8 param storage).

    Unlike :class:`QuantizedLinearFunction`, the weight is never passed as a
    BF16 tensor — only the compact FP8 data + scale enter the autograd graph.
    This prevents PyTorch from pinning a full BF16 copy per layer for the
    entire forward pass, which is critical for fitting 70B models in 192 GB.

    Forward:  quantize input → FP8 GEMM with pre-quantized weight
    Backward: re-dequantize weight from saved FP8 for dgrad; no wgrad
              (frozen base weights).
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight_fp8: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        scaling_manager,
        scaling_type: str,
        fp8_dtype: torch.dtype,
        block_size: int,
        tensor_id: str,
        gradient_accumulation_fusion: bool,
        delay_wgrad: bool,
        deferred_wgrad,
        activation_tensor_id: Optional[str] = None,
    ) -> torch.Tensor:
        input_2d = input.reshape(-1, input.shape[-1]).contiguous()

        _act_mgr = scaling_manager if activation_tensor_id else None
        _act_tid = activation_tensor_id or "activation"
        input_desc = quantize_input(
            input_2d,
            scaling_type,
            fp8_dtype,
            block_size,
            _act_mgr,
            _act_tid,
        )
        input_fp8, input_scale = input_desc.data, input_desc.scale

        N = weight_fp8.shape[0]
        output = dispatch_gemm(
            input_fp8,
            weight_fp8,
            input_scale,
            weight_scale,
            scaling_type,
        )
        output = output.view(*input.shape[:-1], N)

        if bias is not None:
            output = output + bias

        ctx.save_for_backward(input_fp8, weight_fp8, input_scale, weight_scale)
        ctx.scaling_manager = scaling_manager
        ctx.scaling_type = scaling_type
        ctx.fp8_dtype = fp8_dtype
        ctx.block_size = block_size
        ctx.has_bias = bias is not None
        ctx.input_shape = input.shape
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.delay_wgrad = delay_wgrad
        ctx.deferred_wgrad = deferred_wgrad
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_fp8, weight_fp8, input_scale, weight_scale = ctx.saved_tensors
        fp8_dtype = ctx.fp8_dtype
        block_size = ctx.block_size
        scaling_type = ctx.scaling_type

        mgr = ctx.scaling_manager
        bwd_dtype = getattr(mgr, "fp8_dtype_bwd", fp8_dtype) if mgr else fp8_dtype
        if bwd_dtype != fp8_dtype:
            bwd_dtype = fp8_dtype

        grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

        bwd_scaling = "dynamic" if scaling_type in ("per_token", "blockwise", "blockwise2d") else scaling_type
        grad_desc = quantize_input(grad_flat, bwd_scaling, bwd_dtype, block_size)
        grad_fp8, grad_scale = grad_desc.data, grad_desc.scale

        weight_desc = FP8Descriptor.from_tensors(weight_fp8, weight_scale, fp8_dtype)

        _mixed_dtype = weight_fp8.dtype != grad_fp8.dtype
        _needs_dequant = scaling_type in ("per_token", "blockwise", "blockwise2d")
        if _needs_dequant:
            from lumen.ops.quantize.gemm_primitives import _dequant_fp8_weight

            grad_bf16 = (grad_fp8.bfloat16() * grad_scale.bfloat16()).contiguous()
            weight_bf16 = _dequant_fp8_weight(weight_fp8, weight_scale, block_size).bfloat16()
            grad_input = dispatch_gemm(
                grad_bf16,
                weight_bf16.t().contiguous(),
                None,
                None,
                "none",
            )
        elif _mixed_dtype:
            grad_bf16 = (grad_fp8.bfloat16() * grad_scale.bfloat16()).contiguous()
            weight_bf16 = (weight_fp8.bfloat16() * weight_scale.bfloat16()).contiguous()
            grad_input = dispatch_gemm(grad_bf16, weight_bf16.t().contiguous(), None, None, "none")
            del grad_bf16, weight_bf16
        else:
            grad_input = dispatch_gemm(
                grad_fp8,
                weight_desc.transpose_cached,
                grad_scale,
                weight_desc.scale,
                bwd_scaling,
            )
        grad_input = grad_input.view(*grad_output.shape[:-1], weight_fp8.shape[-1])

        grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1))) if ctx.has_bias else None

        return (
            grad_input,
            None,  # weight_fp8 (no grad — frozen)
            None,  # weight_scale
            grad_bias,
            None,  # scaling_manager
            None,  # scaling_type
            None,  # fp8_dtype
            None,  # block_size
            None,  # tensor_id
            None,  # gradient_accumulation_fusion
            None,  # delay_wgrad
            None,  # deferred_wgrad
            None,  # activation_tensor_id
        )


_mark_allow_in_graph(FP8StoredLinearFunction)


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
    gradient_accumulation_fusion: bool = False,
    delay_wgrad: bool = False,
    deferred_wgrad=None,
    fp8_activation_store: bool = False,
    pre_quantized_weight: Optional[tuple] = None,
    activation_tensor_id: Optional[str] = None,
) -> torch.Tensor:
    """Functional quantized linear with multi-backend fallback (all AITER).

    Args:
        input: Input tensor ``[*, in_features]``.
        weight: Weight matrix ``[out_features, in_features]``.
        bias: Optional bias ``[out_features]``.
        scaling_manager: A :class:`~lumen.quantize.ScalingManager`.
        backend: Legacy parameter (ignored, auto-fallback is always used).
        scaling_type: One of ``"delayed"``, ``"dynamic"``, ``"per_token"``,
            ``"blockwise"``, ``"blockwise2d"``, ``"mxfp8"``, ``"none"``.
        fp8_dtype: Target FP8 dtype.  ``None`` auto-detects based on GPU
            architecture (``float8_e4m3fnuz`` on gfx942, ``float8_e4m3fn``
            on gfx950+).
        block_size: Block size for blockwise/MXFP8 quantization.
        tensor_id: Unique identifier for this layer's weight.
        quantize_activation: If ``True``, quantize both input and weight.
        fp8_wgrad: If ``True``, compute weight gradient in FP8.
        delay_wgrad: If ``True``, defer weight gradient computation.
        deferred_wgrad: A :class:`~lumen.modules.parallel_linear._DeferredWgrad`
            instance that collects deferred wgrad closures.
        pre_quantized_weight: Optional ``(fp8_tensor, scale)`` tuple when the
            weight is already stored in FP8.  Bypasses weight quantization
            and avoids materializing a full BF16 weight tensor in the
            autograd graph.

    Returns:
        Output tensor ``[*, out_features]``.
    """
    if fp8_dtype is None:
        fp8_dtype = _get_float8_e4m3()
        _logger.info("quantized_linear: auto-detected fp8_dtype=%s", fp8_dtype)

    if scaling_manager is None:
        from lumen.quantize import ScalingManager

        scaling_manager = ScalingManager(fp8_dtype=fp8_dtype)

    if pre_quantized_weight is not None:
        return FP8StoredLinearFunction.apply(
            input,
            pre_quantized_weight[0],
            pre_quantized_weight[1],
            bias,
            scaling_manager,
            scaling_type,
            fp8_dtype,
            block_size,
            tensor_id,
            gradient_accumulation_fusion,
            delay_wgrad,
            deferred_wgrad,
            activation_tensor_id,
        )

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
        gradient_accumulation_fusion,
        delay_wgrad,
        deferred_wgrad,
        fp8_activation_store,
        activation_tensor_id,
    )
