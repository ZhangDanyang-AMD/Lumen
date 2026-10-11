###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Low-level quantization/dequantization ops wrapping Triton and C++ kernels.

These are pure stateless functions — no autograd, no scaling history.  For
autograd-aware quantized linear, see :mod:`~.linear`.  For the nn.Module
wrapper, see :mod:`lumen.modules.quantize`.
"""

import logging
import random
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from aiter.ops.quant import static_per_tensor_quant
from aiter.ops.triton._triton_kernels.quant.quant_fp8_blockwise import (
    compute_scale_and_quant,
    quant_fp8_blockwise_for_act_grad_kernel,
    quant_fp8_blockwise_kernel,
    quant_fp8_blockwise_segment_m_kernel,
)
try:
    from aiter.ops.triton._triton_kernels.quant.quant_fp8_blockwise import (
        quant_fp8_blockwise_for_weight_kernel,
    )
    _HAVE_WEIGHT_BLOCKWISE = True
except ImportError:
    quant_fp8_blockwise_for_weight_kernel = None  # type: ignore[assignment]
    _HAVE_WEIGHT_BLOCKWISE = False
try:
    from aiter.ops.triton._triton_kernels.quant.quant_fp8_blockwise import (
        requant_fp8_row_to_col_kernel,
    )
    _HAVE_REQUANT_ROW_TO_COL = True
except ImportError:
    requant_fp8_row_to_col_kernel = None  # type: ignore[assignment]
    _HAVE_REQUANT_ROW_TO_COL = False
from aiter.ops.triton._triton_kernels.quant.quant_mxfp8 import (
    _convert_from_mxfp8_kernel,
    _convert_to_mxfp8_kernel,
)
from torch.library import triton_op, wrap_triton

logger = logging.getLogger(__name__)


def is_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == "hip" and target.arch == "gfx950"


# ---------------------------------------------------------------------------
# Tensorwise Quantization
# ---------------------------------------------------------------------------


def quant_fp8_tensorwise_impl(x, scale, dtype):
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    static_per_tensor_quant(out, x, scale)
    return out


def dequant_fp8_tensorwise_impl(x, scale_inv, dtype):
    return x.to(dtype) * scale_inv


# ---------------------------------------------------------------------------
# Blockwise Quantization
# ---------------------------------------------------------------------------


@triton_op("lumen::quant_fp8_blockwise_impl", mutates_args=())
def quant_fp8_blockwise_impl(
    x: torch.Tensor,
    dtype: torch.dtype,
    axis: int,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor using blockwise FP8 scaling along *axis*.

    Returns ``(x_fp8, x_scales)`` where ``x_scales`` is per-block in float32.
    """
    assert x.is_contiguous() and x.dim() == 2, "Input must be 2D and contiguous"
    assert axis in (-2, -1, 0, 1), f"axis must be 0 or 1 (or -1, -2), got {axis}"
    axis = axis % 2

    M, N = x.shape
    x_fp8 = torch.empty((M, N), dtype=dtype, device=x.device)
    scales_shape = (triton.cdiv(M, block_size), N) if axis == 0 else (M, triton.cdiv(N, block_size))
    x_scales = torch.empty(scales_shape, dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    wrap_triton(quant_fp8_blockwise_kernel)[grid](
        x,
        x_fp8,
        x_scales,
        M,
        N,
        block_size,
        torch.finfo(dtype).max,
        axis,
    )
    return x_fp8, x_scales


@triton_op("lumen::quant_fp8_blockwise_weight_3d", mutates_args=())
def quant_fp8_blockwise_weight_3d(
    w: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched 128×128 blockwise FP8 quant for expert weights ``[E, K, N]``.

    Returns ``(w_fp8, w_scales)`` with scales ``[E, ceil(K/B), ceil(N/B)]``
    stored as dequant multipliers (``amax / FP8_MAX``).
    """
    assert w.dim() == 3, "expert weights must be [E, K, N]"
    w = w.contiguous()
    num_experts, rows, cols = w.shape
    w_fp8 = torch.empty(num_experts, rows, cols, dtype=dtype, device=w.device)
    scale_k = triton.cdiv(rows, block_size)
    scale_n = triton.cdiv(cols, block_size)
    w_scales = torch.empty(
        num_experts, scale_k, scale_n, dtype=torch.float32, device=w.device
    )
    if num_experts == 0 or rows == 0 or cols == 0:
        return w_fp8, w_scales
    if not _HAVE_WEIGHT_BLOCKWISE:
        raise RuntimeError("quant_fp8_blockwise_for_weight_kernel is not in this aiter build")
    wrap_triton(quant_fp8_blockwise_for_weight_kernel)[(num_experts, scale_k, scale_n)](
        w,
        w_fp8,
        w_scales,
        rows,
        cols,
        block_size,
        torch.finfo(dtype).max,
    )
    return w_fp8, w_scales


@quant_fp8_blockwise_weight_3d.register_fake
def quant_fp8_blockwise_weight_3d_meta(
    w: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert w.dim() == 3, "expert weights must be [E, K, N]"
    num_experts, rows, cols = w.shape
    return (
        torch.empty(num_experts, rows, cols, dtype=dtype, device=w.device),
        torch.empty(
            num_experts,
            triton.cdiv(rows, block_size),
            triton.cdiv(cols, block_size),
            dtype=torch.float32,
            device=w.device,
        ),
    )


@quant_fp8_blockwise_impl.register_fake
def quant_fp8_blockwise_impl_meta(
    x: torch.Tensor,
    dtype: torch.dtype,
    axis: int,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, "Input must be 2D"
    assert axis in (-2, -1, 0, 1), f"axis must be 0 or 1 (or -1, -2), got {axis}"
    axis = axis % 2
    M, N = x.shape
    x_fp8 = torch.empty((M, N), dtype=dtype, device=x.device)
    scales_shape = (triton.cdiv(M, block_size), N) if axis == 0 else (M, triton.cdiv(N, block_size))
    x_scales = torch.empty(scales_shape, dtype=torch.float32, device=x.device)
    return x_fp8, x_scales


@triton_op("lumen::quant_fp8_blockwise_dual_axis_impl", mutates_args=())
def quant_fp8_blockwise_dual_axis_impl(
    x: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused dual-axis blockwise FP8 quant: emit row-wise (1×B) AND col-wise (B×1) in one pass.

    Loads each BLOCK×BLOCK tile of BF16 ``x`` once and writes both layouts —
    saves one BF16 read + amax pass vs calling ``quant_fp8_blockwise_impl``
    twice.  Used in the blockwise2d (Jet-RL §4.2) backward where the same
    grad tensor feeds both DGrad (1×128 along axis=1) and WGrad (128×1 along
    axis=0).

    Returns ``(x_fp8_row, x_scales_row, x_fp8_col, x_scales_col)`` with
    shapes ``(M, N)``, ``(M, N/B)``, ``(M, N)``, ``(M/B, N)``.  Scales are
    stored as dequant multipliers (``amax / FP8_MAX``), matching the
    convention of ``quant_fp8_blockwise_impl``.
    """
    assert x.is_contiguous() and x.dim() == 2, "Input must be 2D and contiguous"
    M, N = x.shape

    x_fp8_row = torch.empty((M, N), dtype=dtype, device=x.device)
    x_fp8_col = torch.empty((M, N), dtype=dtype, device=x.device)
    x_scales_row = torch.empty((M, triton.cdiv(N, block_size)), dtype=torch.float32, device=x.device)
    x_scales_col = torch.empty((triton.cdiv(M, block_size), N), dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    wrap_triton(quant_fp8_blockwise_for_act_grad_kernel)[grid](
        x,
        x_fp8_row,
        x_scales_row,
        x_fp8_col,
        x_scales_col,
        M,
        N,
        block_size,
        torch.finfo(dtype).max,
    )
    return x_fp8_row, x_scales_row, x_fp8_col, x_scales_col


@quant_fp8_blockwise_dual_axis_impl.register_fake
def quant_fp8_blockwise_dual_axis_impl_meta(
    x: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, "Input must be 2D"
    M, N = x.shape
    return (
        torch.empty((M, N), dtype=dtype, device=x.device),
        torch.empty((M, triton.cdiv(N, block_size)), dtype=torch.float32, device=x.device),
        torch.empty((M, N), dtype=dtype, device=x.device),
        torch.empty((triton.cdiv(M, block_size), N), dtype=torch.float32, device=x.device),
    )


if _HAVE_REQUANT_ROW_TO_COL:
    @triton_op("lumen::requant_fp8_row_to_col", mutates_args=())
    def requant_fp8_row_to_col(
        x_fp8: torch.Tensor,
        x_scales: torch.Tensor,
        dtype: torch.dtype,
        block_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Re-quantize FP8 (row-wise 1×block) → FP8 (col-wise block×1) without BF16 roundtrip.

        ``x_fp8`` is (M, K) FP8 with row-wise 1×block_size scales ``x_scales`` (M, K//block_size).
        Returns ``(y_fp8, y_scales)`` where ``y_fp8`` is (M, K) FP8 and ``y_scales`` is
        (M//block_size, K) float32 col-wise dequant scales.  Used in the blockwise/blockwise2d
        WGrad backward to avoid a BF16 intermediate activation copy.
        """
        assert x_fp8.is_contiguous() and x_fp8.dim() == 2, "x_fp8 must be 2D and contiguous"
        M, K = x_fp8.shape
        y_fp8 = torch.empty((M, K), dtype=dtype, device=x_fp8.device)
        y_scales = torch.empty((triton.cdiv(M, block_size), K), dtype=torch.float32, device=x_fp8.device)
        grid = (triton.cdiv(M, block_size), triton.cdiv(K, block_size))
        wrap_triton(requant_fp8_row_to_col_kernel)[grid](
            x_fp8, x_scales, y_fp8, y_scales, M, K, block_size, torch.finfo(dtype).max,
        )
        return y_fp8, y_scales
else:
    def requant_fp8_row_to_col(x_fp8, x_scales, dtype, block_size=128):  # type: ignore[misc]
        raise RuntimeError(
            "requant_fp8_row_to_col_kernel not available in this aiter build; "
            "overlay third_party/aiter/aiter/ops/triton/_triton_kernels/quant/quant_fp8_blockwise.py"
        )


if _HAVE_REQUANT_ROW_TO_COL:
    @requant_fp8_row_to_col.register_fake
    def requant_fp8_row_to_col_meta(
        x_fp8: torch.Tensor,
        x_scales: torch.Tensor,
        dtype: torch.dtype,
        block_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x_fp8.dim() == 2, "x_fp8 must be 2D"
        M, K = x_fp8.shape
        return (
            torch.empty((M, K), dtype=dtype, device=x_fp8.device),
            torch.empty((triton.cdiv(M, block_size), K), dtype=torch.float32, device=x_fp8.device),
        )


def quant_fp8_blockwise_segment_m_impl(
    x: torch.Tensor,
    batch_size: int,
    seg_lens: torch.Tensor,
    seg_indptr: torch.Tensor,
    scales_seg_indptr: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
):
    assert x.is_contiguous() and x.dim() == 2, "Input must be 2D and contiguous"
    M, N = x.shape
    x_fp8 = torch.empty((M, N), dtype=dtype, device=x.device)

    scales_shape = (triton.cdiv(M, block_size) + batch_size, N)
    x_scales = torch.empty(scales_shape, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(M, block_size) + seg_lens.shape[0], triton.cdiv(N, block_size))
    quant_fp8_blockwise_segment_m_kernel[grid](
        x,
        x_fp8,
        x_scales,
        N,
        batch_size,
        seg_indptr,
        scales_seg_indptr,
        block_size,
        torch.finfo(dtype).max,
    )
    return x_fp8, x_scales


@triton.jit
def _quant_fp8_blockwise_segment_m_dual_kernel(
    x_ptr,
    y_ptr,
    x_fp8_ptr,
    y_fp8_ptr,
    x_scales_ptr,
    y_scales_ptr,
    XN,
    YN,
    batch_size,
    seg_indptr,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Quantize two identically segmented matrices in one launch."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    bid = 0
    tile_start = 0
    total_m_block = 0
    for bs in range(batch_size):
        seg_start = tl.load(seg_indptr + bs)
        seg_end = tl.load(seg_indptr + bs + 1)
        tiles = (
            (seg_end - seg_start + BLOCK_SIZE - 1) // BLOCK_SIZE
        ).to(tl.int32)
        if pid_m >= total_m_block:
            bid = bs
            tile_start = total_m_block
        total_m_block += tiles
    if pid_m >= total_m_block:
        return

    seg_start = tl.load(seg_indptr + bid)
    seg_end = tl.load(seg_indptr + bid + 1)
    m_start = seg_start + (pid_m - tile_start) * BLOCK_SIZE
    m_end = min(seg_end, m_start + BLOCK_SIZE)
    if m_end - m_start == 0:
        return

    offs_m = m_start + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if pid_n * BLOCK_SIZE < XN:
        x_mask = (offs_m[:, None] < m_end) & (offs_n[None, :] < XN)
        x_offsets = offs_m[:, None] * XN + offs_n[None, :]
        x = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
        x_fp8, x_scale = compute_scale_and_quant(x, tl.abs(x), 0, FP8_MAX)
        tl.store(
            x_fp8_ptr + x_offsets,
            x_fp8.to(x_fp8_ptr.dtype.element_ty),
            mask=x_mask,
        )
        tl.store(
            x_scales_ptr + pid_m * XN + offs_n,
            tl.reshape(1.0 / x_scale, BLOCK_SIZE),
            mask=offs_n < XN,
        )

    if pid_n * BLOCK_SIZE < YN:
        y_mask = (offs_m[:, None] < m_end) & (offs_n[None, :] < YN)
        y_offsets = offs_m[:, None] * YN + offs_n[None, :]
        y = tl.load(y_ptr + y_offsets, mask=y_mask, other=0.0).to(tl.float32)
        y_fp8, y_scale = compute_scale_and_quant(y, tl.abs(y), 0, FP8_MAX)
        tl.store(
            y_fp8_ptr + y_offsets,
            y_fp8.to(y_fp8_ptr.dtype.element_ty),
            mask=y_mask,
        )
        tl.store(
            y_scales_ptr + pid_m * YN + offs_n,
            tl.reshape(1.0 / y_scale, BLOCK_SIZE),
            mask=offs_n < YN,
        )


def quant_fp8_blockwise_segment_m_dual(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    seg_lens: torch.Tensor,
    seg_indptr: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
):
    """Quantize two matrices with shared expert segment boundaries."""
    assert x.is_contiguous() and y.is_contiguous()
    assert x.dim() == 2 and y.dim() == 2 and x.shape[0] == y.shape[0]
    M, XN = x.shape
    YN = y.shape[1]
    x_fp8 = torch.empty((M, XN), dtype=dtype, device=x.device)
    y_fp8 = torch.empty((M, YN), dtype=dtype, device=y.device)
    scale_rows = triton.cdiv(M, block_size) + batch_size
    x_scales = torch.empty((scale_rows, XN), dtype=torch.float32, device=x.device)
    y_scales = torch.empty((scale_rows, YN), dtype=torch.float32, device=y.device)
    grid = (
        triton.cdiv(M, block_size) + seg_lens.shape[0],
        triton.cdiv(max(XN, YN), block_size),
    )
    _quant_fp8_blockwise_segment_m_dual_kernel[grid](
        x,
        y,
        x_fp8,
        y_fp8,
        x_scales,
        y_scales,
        XN,
        YN,
        batch_size,
        seg_indptr,
        block_size,
        torch.finfo(dtype).max,
    )
    return x_fp8, x_scales, y_fp8, y_scales


# ---------------------------------------------------------------------------
# MXFP8 Conversion
# ---------------------------------------------------------------------------


@triton_op("lumen::convert_to_mxfp8", mutates_args={})
def convert_to_mxfp8(
    data_hp: torch.Tensor,
    block_size: int = 64,
    axis: int = -1,
    is_2d_block: bool = False,
    use_sr: bool = False,
    use_asm: Optional[bool] = None,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
    float8_dtype_pt: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch._check(
        data_hp.shape[axis] % block_size == 0,
        f"tensor shape ({data_hp.shape}) at axis={axis} is not divisible by {block_size}",
    )
    assert not is_2d_block or data_hp.size(-2) % block_size == 0
    assert data_hp.dtype in [torch.float32, torch.bfloat16]
    if use_asm is None:
        use_asm = is_cdna4() and float8_dtype_pt == torch.float8_e4m3fn
    elif use_asm and float8_dtype_pt == torch.float8_e5m2:
        use_asm = False
        logger.warning(f"ASM mode doesn't support {float8_dtype_pt}, falling back to non-ASM implementation")

    data_hp = data_hp.transpose(axis, -1)
    data_shape = data_hp.shape
    data_hp = data_hp.reshape(-1, data_shape[-1])
    data_lp = torch.empty(data_shape, dtype=float8_dtype_pt, device=data_hp.device).reshape(-1, data_shape[-1])

    if is_2d_block:
        scales_shape = (*data_shape[:-2], data_shape[-2] // block_size, data_shape[-1] // block_size)
    else:
        scales_shape = (*data_shape[:-1], data_shape[-1] // block_size)
    scales = torch.ones(scales_shape, dtype=torch.uint8, device=data_hp.device).reshape(-1, scales_shape[-1])
    stride_xm, stride_xn = data_hp.stride()
    stride_ym, stride_yn = data_lp.stride()
    stride_sm, stride_sn = scales.stride()
    M, N = data_hp.shape

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))

    assert M % block_size == 0, "tensor M shape must align to block size"
    assert N % block_size == 0, "tensor N shape must align to block size"

    BLOCK_M = 64 if M >= 64 else M
    BLOCK_N = 64 if N >= 64 else N
    BLOCK_M = block_size if BLOCK_M < block_size else BLOCK_M
    BLOCK_N = block_size if BLOCK_N < block_size else BLOCK_N

    if philox_seed is None:
        philox_seed = random.randint(0, 2**31 - 2)
    if philox_offset is None:
        philox_offset = random.randint(0, 2**31 - 2)
    wrap_triton(_convert_to_mxfp8_kernel)[grid](
        data_hp,
        data_lp,
        scales,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        philox_seed,
        philox_offset,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=block_size,
        IS_2D_BLOCK=is_2d_block,
        USE_SR=use_sr,
        USE_ASM=use_asm,
    )

    return data_lp.reshape(data_shape).transpose(axis, -1), scales.reshape(scales_shape).transpose(axis, -1)


@triton_op("lumen::convert_from_mxfp8", mutates_args={})
def convert_from_mxfp8(
    data_lp: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = 64,
    axis: int = -1,
    is_2d_block: bool = False,
    use_asm: Optional[bool] = None,
) -> torch.Tensor:
    assert output_dtype in [torch.float32, torch.bfloat16]
    if use_asm is None:
        use_asm = is_cdna4() and data_lp.dtype == torch.float8_e4m3fn
    elif use_asm and data_lp.dtype == torch.float8_e5m2:
        use_asm = False
        logger.warning(f"ASM mode doesn't support {data_lp.dtype}, falling back to non-ASM implementation")

    data_lp = data_lp.transpose(axis, -1)
    scales = scales.transpose(axis, -1)
    orig_shape = data_lp.shape
    data_lp = data_lp.reshape(-1, orig_shape[-1])

    scales = scales.reshape(-1, orig_shape[-1] // block_size)
    data_hp = data_lp.new_empty(orig_shape, dtype=output_dtype).reshape(-1, orig_shape[-1])

    stride_xm, stride_xn = data_lp.stride()
    stride_ym, stride_yn = data_hp.stride()
    stride_sm, stride_sn = scales.stride()
    M, N = data_hp.shape

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))

    BLOCK_M = 64 if M >= 64 else M
    BLOCK_N = 64 if N >= 64 else N
    wrap_triton(_convert_from_mxfp8_kernel)[grid](
        data_lp,
        data_hp,
        scales,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=block_size,
        IS_2D_BLOCK=is_2d_block,
        USE_ASM=use_asm,
    )
    return data_hp.reshape(orig_shape).transpose(axis, -1)


@convert_to_mxfp8.register_fake
def _fake_convert_to_mxfp8(
    data_hp: torch.Tensor,
    block_size: int = 64,
    axis: int = -1,
    is_2d_block: bool = False,
    use_sr: bool = False,
    use_asm: Optional[bool] = None,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
    float8_dtype_pt: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    data_hp = data_hp.transpose(axis, -1)
    orig_shape = data_hp.shape

    data_lp = data_hp.new_empty(data_hp.shape, dtype=float8_dtype_pt)
    if is_2d_block:
        scales_shape = (*orig_shape[:-2], orig_shape[-2] // block_size, orig_shape[-1] // block_size)
    else:
        scales_shape = (*orig_shape[:-1], orig_shape[-1] // block_size)

    scales = data_hp.new_empty(scales_shape, dtype=torch.uint8)
    return data_lp, scales.transpose(axis, -1)


@convert_from_mxfp8.register_fake
def _fake_convert_from_mxfp8(
    data_lp: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = 64,
    axis: int = -1,
    is_2d_block: bool = False,
    use_asm: Optional[bool] = None,
) -> torch.Tensor:
    data_hp = data_lp.new_empty(data_lp.shape, dtype=output_dtype)
    return data_hp
