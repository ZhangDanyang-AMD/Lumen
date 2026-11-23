###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""MXFP4 attention projections for Megatron local-spec layers next to SonicMoE.

``LUMEN_ATTN_LINEAR_RECIPE=a4w4`` runs ``self_attention.linear_qkv`` and
``linear_proj`` with the expert a4w4 recipe: Fprop X(MXFP4) x W(MXFP4 along K),
Dgrad dY(MXFP4) x W(MXFP4 along N, an independent pack), BF16 Wgrad exactly as
Megatron computes it. Every operand uses 1x32 E8M0 "even" scales with
round-to-nearest-even; no stochastic rounding or Hadamard rotation.

Global Lumen linear quantization stays disabled (SonicMoE requires that), so
this only overrides each module's ``_forward_impl``; calls it cannot serve
(bias, tensor-parallel communication, deferred Wgrad, frozen weights) keep the
Megatron path.

The two weight packs are reused across the microbatches of an optimizer step.
Megatron's optimizer writes parameters without bumping ``_version``, so the
cache is keyed on the per-step generation that the SonicMoE pack-cache
lifecycle advances on the same layer's experts, and repacks on every call
whenever that lifecycle is not active.
"""

from __future__ import annotations

import os
import weakref

import torch

_RECIPE_ENV = "LUMEN_ATTN_LINEAR_RECIPE"
_RECIPES = ("bf16", "a4w4")
_LINEAR_NAMES = ("linear_qkv", "linear_proj")


def attention_linear_recipe() -> str:
    recipe = os.environ.get(_RECIPE_ENV, "bf16")
    if recipe not in _RECIPES:
        raise ValueError(f"{_RECIPE_ENV} must be one of {_RECIPES}, got {recipe!r}")
    return recipe


def _aiter_ops():
    from lumen.ops.dispatch import _probe_aiter_mxfp4_linear

    ops = _probe_aiter_mxfp4_linear()
    if ops is None:
        raise RuntimeError(
            f"{_RECIPE_ENV}=a4w4 requires aiter.ops.triton.gemm.basic.mxfp4_linear"
        )
    return ops


def _wgrad_accumulate(weight, grad_output, total_input):
    """Megatron's gradient-accumulation-fusion Wgrad; returns the grad to hand autograd."""
    from megatron.core.tensor_parallel import layers

    kernels = getattr(layers, "fused_weight_gradient_mlp_cuda", None)
    if kernels is None:
        raise RuntimeError(
            "gradient_accumulation_fusion requires fused_weight_gradient_mlp_cuda"
        )
    if weight.main_grad.dtype == torch.float32:
        kernels.wgrad_gemm_accum_fp32(total_input, grad_output, weight.main_grad)
    elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
        kernels.wgrad_gemm_accum_fp16(total_input, grad_output, weight.main_grad)
    else:
        raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
    if not hasattr(weight, "grad_added_to_main_grad"):
        return None
    weight.grad_added_to_main_grad = True
    factory = torch.zeros if getattr(weight, "zero_out_wgrad", False) else torch.empty
    return factory(
        weight.main_grad.shape, dtype=total_input.dtype, device=total_input.device
    )


class _MXFP4LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, packs, gradient_accumulation_fusion):
        gemm, _pack = _aiter_ops()
        (w_k, s_k), dgrad_pack = packs
        x2d = x.reshape(-1, x.shape[-1])
        y = gemm(x2d, w_k, s_k)
        ctx.save_for_backward(x, weight)
        ctx.dgrad_pack = dgrad_pack
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        if gradient_accumulation_fusion and hasattr(weight, "main_grad"):
            ctx.main_grad = weight.main_grad
        return y.view(*x.shape[:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        gemm, _pack = _aiter_ops()
        x, weight = ctx.saved_tensors
        dy = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        x2d = x.reshape(-1, x.shape[-1])
        dx = gemm(dy, *ctx.dgrad_pack).view(x.shape)
        if ctx.gradient_accumulation_fusion:
            weight.main_grad = ctx.main_grad
            grad_weight = _wgrad_accumulate(weight, dy, x2d)
        else:
            grad_weight = dy.t().matmul(x2d)
        return dx, grad_weight, None, None


class _WeightPackCache:
    """Fprop/Dgrad packs of one weight, refreshed once per optimizer step."""

    def __init__(self, module, generation_source=None):
        self._module = weakref.ref(module)
        self._source = (
            weakref.ref(generation_source) if generation_source is not None else None
        )
        self._packs = None
        self._stamp = None

    def _generation(self):
        source = self._source() if self._source is not None else None
        if source is None or not getattr(source, "_pack_cache_enabled", False):
            return None
        return int(getattr(source, "_pack_generation", 0))

    def get(self, weight):
        _gemm, pack = _aiter_ops()
        generation = self._generation()
        if generation is None:
            self._packs = self._stamp = None
            return pack(weight)
        stamp = (generation, weight.data_ptr(), weight._version)
        if self._stamp != stamp:
            self._packs = pack(weight)
            self._stamp = stamp
        return self._packs


def _make_forward_impl(module, original, cache):
    def _forward_impl(input, weight, *args, **kwargs):
        if (
            args
            or kwargs.get("bias") is not None
            or kwargs.get("allreduce_dgrad")
            or kwargs.get("sequence_parallel")
            or kwargs.get("grad_output_buffer") is not None
            or weight is not module.weight
            or not weight.requires_grad
            or weight.dtype != torch.bfloat16
            or input.dtype != torch.bfloat16
        ):
            return original(input, weight, *args, **kwargs)
        return _MXFP4LinearFn.apply(
            input,
            weight,
            cache.get(weight),
            bool(kwargs.get("gradient_accumulation_fusion", False)),
        )

    return _forward_impl


def install_mxfp4_attention_linears(model) -> int:
    """Route each layer's attention projections through MXFP4; returns the count."""
    _aiter_ops()
    count = 0
    for layer in getattr(getattr(model, "decoder", None), "layers", ()):
        attention = getattr(layer, "self_attention", None)
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        source = experts if hasattr(experts, "_pack_generation") else None
        for name in _LINEAR_NAMES:
            linear = getattr(attention, name, None)
            if (
                linear is None
                or getattr(linear, "_lumen_mxfp4_linear", None) is not None
            ):
                continue
            if type(linear).__name__ not in (
                "ColumnParallelLinear",
                "RowParallelLinear",
            ):
                raise TypeError(
                    f"{_RECIPE_ENV}=a4w4 supports Megatron local-spec linears, "
                    f"got {type(linear).__name__} for {name}"
                )
            cache = _WeightPackCache(linear, source)
            linear._lumen_mxfp4_linear = cache
            linear._forward_impl = _make_forward_impl(
                linear, linear._forward_impl, cache
            )
            count += 1
    return count


__all__ = ["attention_linear_recipe", "install_mxfp4_attention_linears"]
