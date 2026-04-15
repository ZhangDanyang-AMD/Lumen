###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Fused FP8 cast + transpose kernel.

Combines per-tensor FP8 quantization with matrix transposition into a single
Triton kernel launch, replacing the separate quantize → transpose → contiguous
chain that currently requires multiple kernel launches.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from lumen.ops.quantize.quant_amax_fused import _get_amax_scratch

__all__ = ["cast_transpose_fp8", "cast_transpose_amax_fp8"]

_TORCH_TO_TL_FP8: Dict[torch.dtype, tl.dtype] = {
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e4m3fnuz: tl.float8e4b8,
    torch.float8_e5m2: tl.float8e5,
    torch.float8_e5m2fnuz: tl.float8e5b16,
}


@triton.jit
def _cast_transpose_fp8_kernel(
    X_ptr,
    OUT_ptr,
    OUT_T_ptr,
    scale_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    stride_ot0,
    stride_ot1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    FP8_TY: tl.constexpr,
):
    """Each program handles a BLOCK_M x BLOCK_N tile.

    Reads X[bm:bm+BLOCK_M, bn:bn+BLOCK_N], scales to FP8, writes both:
    - OUT[bm:bm+BLOCK_M, bn:bn+BLOCK_N] (same layout as X)
    - OUT_T[bn:bn+BLOCK_N, bm:bm+BLOCK_M] i.e. OUT_T[n,m] = quant(X[m,n])
    """
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    scale = tl.load(scale_ptr).to(tl.float32)
    inv_scale = 1.0 / scale

    x_scaled = x * inv_scale
    x_clamped = tl.where(
        x_scaled > CLAMP_MAX,
        CLAMP_MAX,
        tl.where(x_scaled < -CLAMP_MAX, -CLAMP_MAX, x_scaled),
    )
    x_fp8 = x_clamped.to(FP8_TY)

    out_ptrs = OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, x_fp8, mask=mask)

    # OUT_T has shape (N, M); element (n, m) gets X[m, n].
    out_t_ptrs = OUT_T_ptr + offs_n[None, :] * stride_ot0 + offs_m[:, None] * stride_ot1
    tl.store(out_t_ptrs, x_fp8, mask=mask)


@triton.jit
def _cast_transpose_amax_fp8_kernel(
    X_ptr,
    OUT_ptr,
    OUT_T_ptr,
    scale_ptr,
    amax_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    stride_ot0,
    stride_ot1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    FP8_TY: tl.constexpr,
):
    """Tile-based fused cast + transpose + amax.

    Same tiling as _cast_transpose_fp8_kernel but also computes per-tile
    max(abs(x)) and atomically reduces it into amax_ptr[0].
    """
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    tile_amax = tl.max(tl.where(mask, tl.abs(x), 0.0))
    tl.atomic_max(amax_ptr, tile_amax, sem="relaxed")

    scale = tl.load(scale_ptr).to(tl.float32)
    inv_scale = 1.0 / scale

    x_scaled = x * inv_scale
    x_clamped = tl.where(
        x_scaled > CLAMP_MAX,
        CLAMP_MAX,
        tl.where(x_scaled < -CLAMP_MAX, -CLAMP_MAX, x_scaled),
    )
    x_fp8 = x_clamped.to(FP8_TY)

    out_ptrs = OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, x_fp8, mask=mask)

    out_t_ptrs = OUT_T_ptr + offs_n[None, :] * stride_ot0 + offs_m[:, None] * stride_ot1
    tl.store(out_t_ptrs, x_fp8, mask=mask)


def cast_transpose_amax_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype,
    *,
    clamp_max: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused cast to FP8 + transpose + amax in a single kernel.

    Combines :func:`cast_transpose_fp8` with amax computation so the
    BF16 input is read exactly once.

    Args:
        x: Input tensor (M, N) in BF16/FP16/FP32 on CUDA.
        scale: Per-tensor scale (1,) float32 — divisor style.
        fp8_dtype: Target FP8 dtype.
        clamp_max: Absolute clamp bound before FP8 cast.

    Returns:
        ``(out, out_t, amax)``: row-major FP8 ``(M, N)``, transposed FP8
        ``(N, M)``, and scalar float32 amax of the *unscaled* input.
    """
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    assert x.is_cuda, "cast_transpose_amax_fp8 requires a CUDA tensor"
    if fp8_dtype not in _TORCH_TO_TL_FP8:
        raise TypeError(f"Unsupported fp8_dtype: {fp8_dtype}")

    M, N = x.shape
    out = torch.empty((M, N), dtype=fp8_dtype, device=x.device)
    out_t = torch.empty((N, M), dtype=fp8_dtype, device=x.device)
    amax_out = _get_amax_scratch(x.device)
    amax_out.zero_()

    if clamp_max is None:
        clamp_max = float(torch.finfo(fp8_dtype).max)

    scale_1 = scale.float().reshape(-1).contiguous()
    if scale_1.device != x.device:
        scale_1 = scale_1.to(device=x.device)

    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    fp8_tl = _TORCH_TO_TL_FP8[fp8_dtype]
    _cast_transpose_amax_fp8_kernel[grid](
        x,
        out,
        out_t,
        scale_1,
        amax_out,
        M,
        N,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        out_t.stride(0),
        out_t.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CLAMP_MAX=clamp_max,
        FP8_TY=fp8_tl,
    )

    return out, out_t, amax_out.clone()


def cast_transpose_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype,
    *,
    clamp_max: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused cast to FP8 + transpose in a single kernel.

    Args:
        x: Input tensor (M, N) in BF16/FP16/FP32 on CUDA.
        scale: Per-tensor scale (1,) float32 — divisor style (same as
            ``tensor / scale`` in :meth:`ScalingManager._quantize_core`).
        fp8_dtype: Target FP8 dtype.
        clamp_max: Absolute clamp bound before FP8 cast. Defaults to
            ``torch.finfo(fp8_dtype).max``.

    Returns:
        ``(out, out_t)``: row-major FP8 ``(M, N)`` and transposed FP8 ``(N, M)``.
    """
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    assert x.is_cuda, "cast_transpose_fp8 requires a CUDA tensor"
    if fp8_dtype not in _TORCH_TO_TL_FP8:
        raise TypeError(f"Unsupported fp8_dtype for cast_transpose_fp8: {fp8_dtype}")

    M, N = x.shape
    out = torch.empty((M, N), dtype=fp8_dtype, device=x.device)
    out_t = torch.empty((N, M), dtype=fp8_dtype, device=x.device)

    if clamp_max is None:
        clamp_max = float(torch.finfo(fp8_dtype).max)

    scale_1 = scale.float().reshape(-1).contiguous()
    if scale_1.device != x.device:
        scale_1 = scale_1.to(device=x.device)

    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    fp8_tl = _TORCH_TO_TL_FP8[fp8_dtype]
    _cast_transpose_fp8_kernel[grid](
        x,
        out,
        out_t,
        scale_1,
        M,
        N,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        out_t.stride(0),
        out_t.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CLAMP_MAX=clamp_max,
        FP8_TY=fp8_tl,
    )

    return out, out_t
