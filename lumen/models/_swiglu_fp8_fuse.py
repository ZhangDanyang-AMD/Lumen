###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused SwiGLU + FP8 quantization bridge.

When enabled via ``LUMEN_FUSED_SWIGLU_QUANT=1``, the SwiGLU activation
stores an FP8-quantized MLP hidden state. The next
:func:`~lumen.modules.parallel_linear._do_gemm` that quantizes
activations for FP8 picks it up via :func:`pop_swiglu_fp8_cache` and skips
a separate SiLU + mul + quant pass.

Uses AITER ``dynamic_per_tensor_quant_fp8_i8`` on the BF16 SwiGLU output,
fusing amax computation and quantization in a single pass.
"""

from __future__ import annotations

import threading

import torch

_cache = threading.local()


def try_fused_swiglu_fp8(swiglu_input: torch.Tensor, bf16_output: torch.Tensor) -> None:
    """Run fused SiLU*mul + FP8 quant and cache ``(fp8, scale, amax)`` for the next FP8 GEMM.

    Uses AITER dynamic per-tensor quantization with amax output on the
    already-computed bf16_output, fusing amax computation and quantization
    in one pass.  The amax is passed through to the scaling manager so it
    can skip its own ``fused_amax_abs`` call.
    """
    if not swiglu_input.is_cuda:
        return

    from lumen.quantize.config import _get_float8_e4m3

    fp8_dtype = _get_float8_e4m3()

    out_2d = bf16_output.detach().reshape(-1, bf16_output.shape[-1])
    if not out_2d.is_contiguous():
        out_2d = out_2d.contiguous()

    try:
        from aiter.ops.triton.quant import dynamic_per_tensor_quant_fp8_i8_with_amax

        qx = torch.empty_like(out_2d, dtype=fp8_dtype)
        scale_out = torch.empty(1, dtype=torch.float32, device=out_2d.device)
        amax_out = torch.zeros(1, dtype=torch.float32, device=out_2d.device)
        dynamic_per_tensor_quant_fp8_i8_with_amax(qx, out_2d, scale_out, amax_out)

        _cache.fp8_data = qx
        _cache.scale = scale_out
        _cache.amax = amax_out
        _cache.valid = True
    except (ImportError, RuntimeError):
        return


def discard_swiglu_fp8_cache() -> None:
    """Drop any pending fused SwiGLU FP8 cache without consuming it."""
    _cache.valid = False
    _cache.fp8_data = None
    _cache.scale = None
    _cache.amax = None


def pop_swiglu_fp8_cache():
    """Pop cached ``(fp8_data, scale)`` if set; otherwise return ``None``.

    Also returns ``amax`` as the third element when available.
    """
    if getattr(_cache, "valid", False):
        result = (_cache.fp8_data, _cache.scale, getattr(_cache, "amax", None))
        _cache.valid = False
        _cache.fp8_data = None
        _cache.scale = None
        _cache.amax = None
        return result
    return None
