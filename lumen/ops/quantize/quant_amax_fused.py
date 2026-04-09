###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused static FP8 quantization + amax computation in a single Triton kernel.

For delayed scaling, the normal flow is:
  1. quantize: fp8 = (x * (1/scale)).to(fp8_dtype)   -- uses OLD scale
  2. update_amax: amax = x.abs().amax()               -- records for NEXT step

Steps 1 and 2 both read the full tensor, meaning two full memory passes.
This module fuses them into a single Triton kernel that reads x once,
computes both the FP8 output and the amax simultaneously.

Enabled via ``LUMEN_FUSED_QUANT_AMAX=1`` (checked in scaling_manager.py).
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl


@triton.jit
def _static_quant_amax_kernel(
    qx_ptr,
    x_in_ptr,
    scale_in_ptr,
    amax_out_ptr,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
):
    """Single-pass: quantize x with static scale AND compute global amax.

    Each program (one per row) computes the row-local max(abs(x)) and
    atomically reduces it into ``amax_out_ptr[0]``.  Simultaneously,
    it quantizes x using the provided scale.
    """
    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)

    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg")

    row_amax = tl.max(tl.abs(x))
    tl.atomic_max(amax_out_ptr, row_amax, sem="relaxed")

    scale = tl.load(scale_in_ptr)
    scale_recip = 1.0 / scale
    qx = (x * scale_recip).to(qx_ptr.dtype.element_ty)

    tl.store(qx_ptr + offs, qx, mask=mask)


def static_quant_with_amax(
    x: torch.Tensor,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` with ``scale`` and compute ``amax(abs(x))`` in one pass.

    Args:
        x: Input tensor, must be 2-D ``(M, N)`` and on GPU.
        scale: Per-tensor scale, shape ``(1,)`` or scalar, float32.
        fp8_dtype: Target FP8 dtype (e.g. ``torch.float8_e4m3fnuz``).

    Returns:
        ``(fp8_tensor, amax)`` where ``fp8_tensor`` has the same shape as
        ``x`` and ``amax`` is a scalar float32 tensor on the same device.
    """
    assert x.dim() == 2, f"Expected 2-D input, got {x.dim()}-D"
    x = x.contiguous()
    rows, cols = x.shape

    qx = torch.empty_like(x, dtype=fp8_dtype)
    amax_out = torch.zeros(1, dtype=torch.float32, device=x.device)
    scale_in = scale.float().reshape(1).contiguous()
    if scale_in.device != x.device:
        scale_in = scale_in.to(device=x.device)

    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = (rows,)
    _static_quant_amax_kernel[grid](
        qx,
        x,
        scale_in,
        amax_out,
        cols,
        x.stride(0),
        NUM_COL_POW2=NUM_COL_POW2,
    )

    return qx, amax_out


@functools.lru_cache(maxsize=1)
def _probe_fused_quant_amax() -> bool:
    """Return True if the fused quant+amax kernel is functional."""
    try:
        x = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
        s = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        from lumen.quantize.config import _get_float8_e4m3

        fp8_dt = _get_float8_e4m3()
        qx, am = static_quant_with_amax(x, s, fp8_dt)
        assert qx.shape == x.shape
        assert am.numel() == 1
        return True
    except Exception:
        return False
