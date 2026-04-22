###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused FP8 quantization kernels — single-pass static quant + amax.

For delayed scaling, the normal flow is:
  1. quantize: fp8 = (x * (1/scale)).to(fp8_dtype)   -- uses OLD scale
  2. update_amax: amax = x.abs().amax()               -- records for NEXT step

Steps 1 and 2 both read the full tensor, meaning two full memory passes.
This module fuses them into a single Triton kernel that reads x once,
computes both the FP8 output and the amax simultaneously.

Also provides :func:`fused_amax_abs` — a lightweight single-launch
``amax(abs(x))`` reduction.

Enabled via ``LUMEN_FUSED_QUANT_AMAX=1`` (checked in scaling_manager.py).
"""

from __future__ import annotations

import functools
from typing import Dict

import torch
import triton
import triton.language as tl

_amax_scratch: Dict[torch.device, torch.Tensor] = {}


def _get_amax_scratch(device: torch.device) -> torch.Tensor:
    """Return a reusable 1-element float32 scratch tensor for amax accumulation.

    Avoids allocating a new ``torch.zeros(1)`` on every quant/amax kernel call
    (~2,900 times per training step). The caller MUST zero the tensor before use.
    """
    buf = _amax_scratch.get(device)
    if buf is None:
        buf = torch.zeros(1, dtype=torch.float32, device=device)
        _amax_scratch[device] = buf
    return buf


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
    amax_out = _get_amax_scratch(x.device)
    amax_out.zero_()
    if scale.dtype == torch.float32 and scale.ndim == 1 and scale.numel() == 1 and scale.is_contiguous():
        scale_in = scale
    else:
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

    return qx, amax_out.clone()


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


# ---------------------------------------------------------------------------
# Lightweight amax(abs(x)) kernel — single launch replaces abs() + amax()
# ---------------------------------------------------------------------------


@triton.jit
def _amax_abs_kernel(
    x_ptr,
    amax_out_ptr,
    cols: int,
    x_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
):
    """Compute row-local max(abs(x)) and atomically reduce into a global scalar."""
    pid = tl.program_id(axis=0)
    tl.assume(pid >= 0)
    tl.assume(x_stride_r > 0)

    offs = pid * x_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg")

    row_amax = tl.max(tl.abs(x))
    tl.atomic_max(amax_out_ptr, row_amax, sem="relaxed")


def fused_amax_abs(x: torch.Tensor) -> torch.Tensor:
    """Compute ``amax(abs(x))`` in a single kernel launch.

    Handles arbitrary-dim tensors by flattening to 2-D internally.

    Args:
        x: Input tensor on GPU, any float dtype.

    Returns:
        Scalar float32 tensor on the same device.
    """
    if x.dim() <= 1:
        return x.detach().abs().amax()

    if x.dim() != 2:
        x = x.reshape(-1, x.shape[-1])
    x = x.contiguous()
    rows, cols = x.shape

    amax_out = _get_amax_scratch(x.device)
    amax_out.zero_()
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    _amax_abs_kernel[(rows,)](x, amax_out, cols, x.stride(0), NUM_COL_POW2=NUM_COL_POW2)
    return amax_out.squeeze(0).clone()


# ---------------------------------------------------------------------------
# Fused FP8 dequant: fp8.bfloat16() * scale  in one kernel (no aten::copy_)
# ---------------------------------------------------------------------------


@triton.jit
def _dequant_fp8_bf16_kernel(
    FP8_ptr,
    SCALE_ptr,
    OUT_ptr,
    rows: int,
    cols: int,
    fp8_stride_r: int,
    out_stride_r: int,
    SCALE_IS_SCALAR: tl.constexpr,
    NUM_COL_POW2: tl.constexpr,
):
    """Fused FP8→BF16 dequant: out = fp8.to(f32) * scale.

    Replaces ``fp8_tensor.bfloat16() * scale.bfloat16()`` which generates
    a separate ``aten::copy_`` + ``aten::mul``.
    """
    pid = tl.program_id(axis=0)
    tl.assume(pid >= 0)
    tl.assume(fp8_stride_r > 0)
    tl.assume(out_stride_r > 0)

    offs = tl.arange(0, NUM_COL_POW2)
    mask = offs < cols

    fp8_vals = tl.load(FP8_ptr + pid * fp8_stride_r + offs, mask=mask, other=0.0)
    x_f32 = fp8_vals.to(tl.float32)

    if SCALE_IS_SCALAR:
        s = tl.load(SCALE_ptr).to(tl.float32)
    else:
        s = tl.load(SCALE_ptr).to(tl.float32)

    out = (x_f32 * s).to(tl.bfloat16)
    tl.store(OUT_ptr + pid * out_stride_r + offs, out, mask=mask)


def dequant_fp8_to_bf16(
    fp8_tensor: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Fused FP8 → BF16 dequantization in a single Triton kernel.

    Replaces ``fp8_tensor.bfloat16() * scale.bfloat16()`` (2 ops, 2 kernels)
    with a single kernel that reads FP8, multiplies by scale, and writes BF16.

    Args:
        fp8_tensor: FP8 tensor (M, N).
        scale: Per-tensor scale, shape (1,) or scalar, float32.

    Returns:
        BF16 tensor (M, N).
    """
    if fp8_tensor.dim() == 1:
        fp8_tensor = fp8_tensor.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    if fp8_tensor.dim() != 2:
        orig_shape = fp8_tensor.shape
        fp8_tensor = fp8_tensor.reshape(-1, fp8_tensor.shape[-1])
    else:
        orig_shape = None

    fp8_tensor = fp8_tensor.contiguous()
    rows, cols = fp8_tensor.shape
    out = torch.empty((rows, cols), dtype=torch.bfloat16, device=fp8_tensor.device)

    if scale.dtype == torch.float32 and scale.ndim == 1 and scale.is_contiguous():
        scale_1 = scale
    else:
        scale_1 = scale.float().reshape(-1).contiguous()
    if scale_1.device != fp8_tensor.device:
        scale_1 = scale_1.to(device=fp8_tensor.device)

    NUM_COL_POW2 = triton.next_power_of_2(cols)
    _dequant_fp8_bf16_kernel[(rows,)](
        fp8_tensor,
        scale_1,
        out,
        rows,
        cols,
        fp8_tensor.stride(0),
        out.stride(0),
        SCALE_IS_SCALAR=True,
        NUM_COL_POW2=NUM_COL_POW2,
    )

    if orig_shape is not None:
        out = out.view(orig_shape)
    if squeeze:
        out = out.squeeze(0)
    return out
