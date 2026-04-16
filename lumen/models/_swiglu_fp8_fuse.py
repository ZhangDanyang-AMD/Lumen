###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused SwiGLU + FP8 quantization bridge.

When enabled via ``LUMEN_FUSED_SWIGLU_QUANT=1``, the SwiGLU activation
stores an FP8-quantized MLP hidden state from AITER's fused kernel. The
next :func:`~lumen.modules.parallel_linear._do_gemm` that quantizes
activations for FP8 picks it up via :func:`pop_swiglu_fp8_cache` and skips
a separate SiLU + mul + quant pass.
"""

from __future__ import annotations

import threading

import torch

_cache = threading.local()


def try_fused_swiglu_fp8(swiglu_input: torch.Tensor, bf16_output: torch.Tensor) -> None:
    """Run fused SiLU*mul + FP8 quant and cache ``(fp8, scale)`` for the next FP8 GEMM."""
    from lumen.ops.dispatch import _probe_aiter_fused_silu_mul_fp8

    if not _probe_aiter_fused_silu_mul_fp8():
        return
    if not swiglu_input.is_cuda:
        return

    from aiter.ops.triton.quant.fused_fp8_quant import fused_silu_mul_fp8_per_tensor_static_quant

    from lumen.ops.quantize.quant_amax_fused import fused_amax_abs
    from lumen.quantize.config import _get_float8_e4m3

    fp8_dtype = _get_float8_e4m3()
    fp8_max = torch.finfo(fp8_dtype).max
    amax = fused_amax_abs(bf16_output.detach()).clamp(min=1e-12)
    scale = (amax / fp8_max).to(dtype=torch.float32, device=swiglu_input.device).reshape(1)

    inp_2d = swiglu_input.reshape(-1, swiglu_input.shape[-1]).contiguous()
    out_fp8 = fused_silu_mul_fp8_per_tensor_static_quant(
        inp_2d,
        scale,
        dtype_quant=fp8_dtype,
        silu_convert_to_inp_type=True,
    )

    _cache.fp8_data = out_fp8
    _cache.scale = scale
    _cache.valid = True


def discard_swiglu_fp8_cache() -> None:
    """Drop any pending fused SwiGLU FP8 cache without consuming it."""
    _cache.valid = False
    _cache.fp8_data = None
    _cache.scale = None


def pop_swiglu_fp8_cache():
    """Pop cached ``(fp8_data, scale)`` if set; otherwise return ``None``."""
    if getattr(_cache, "valid", False):
        result = (_cache.fp8_data, _cache.scale)
        _cache.valid = False
        _cache.fp8_data = None
        _cache.scale = None
        return result
    return None
