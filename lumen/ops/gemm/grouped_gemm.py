###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Grouped GEMM (MoE) operations with multi-backend fallback.

All backends are AITER implementations — no torch fallbacks.

Supports multiple quantization modes for grouped matrix multiplication
used in Mixture-of-Experts architectures.

Backends:
    - **Triton GMM**: ``gmm`` / ``ptgmm`` / ``nptgmm`` (BF16/FP16 only)
    - **Triton MOE GEMM**: ``moe_gemm_a8w8``, ``moe_gemm_a8w8_blockscale``
    - **Triton MOE per-token**: ``moe_gemm_per_token`` (fused per-token scale)
    - **Triton MOE MXFP8**: ``moe_gemm_mxfp8`` (fused MXFP8 microscaling)
    - **CKTile DeepGEMM**: ``deepgemm`` (BF16/FP16/FP8 grouped flat MM)

Quantization modes:
    - ``none``       — BF16 GMM/ptgmm via AITER Triton
    - ``delayed``    — per-tensor FP8 MOE GEMM via AITER Triton
    - ``dynamic``    — per-tensor FP8 MOE GEMM via AITER Triton
    - ``per_token``  — per-token FP8 fused MOE GEMM via AITER Triton
    - ``blockwise``  — per-block FP8 MOE GEMM via AITER Triton
    - ``mxfp8``      — MXFP8 fused MOE GEMM via AITER Triton
"""

import logging
from typing import Optional

import torch

from lumen.ops.dispatch import (
    Backend,
    _probe_aiter_gmm,
    _probe_aiter_moe_gemm_mxfp8,
    _probe_aiter_moe_gemm_per_token,
    try_backends,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _get_gmm():
    from aiter.ops.triton.gmm import gmm

    return gmm


def _get_ptgmm():
    from aiter.ops.triton.gmm import ptgmm

    return ptgmm


def _get_nptgmm():
    from aiter.ops.triton.gmm import nptgmm

    return nptgmm


def _get_deepgemm():
    from aiter.ops.deepgemm import deepgemm

    return deepgemm


def _get_moe_gemm_a8w8():
    from aiter.ops.triton.moe.moe_op_gemm_a8w8 import moe_gemm_a8w8

    return moe_gemm_a8w8


def _get_moe_gemm_a8w8_blockscale():
    from aiter.ops.triton.moe.moe_op_gemm_a8w8_blockscale import moe_gemm_a8w8_blockscale

    return moe_gemm_a8w8_blockscale


# ---------------------------------------------------------------------------
# BF16 Grouped GEMM (no quantization) — all via AITER
# ---------------------------------------------------------------------------


def _gmm_triton(lhs, rhs, group_sizes, bias=None):
    """BF16 grouped GEMM via AITER Triton gmm."""
    fn = _get_gmm()
    return fn(lhs, rhs, group_sizes, bias=bias)


def grouped_gemm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    scaling_type: str = "none",
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    block_size: int = 128,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Grouped GEMM dispatch with multi-backend fallback (all AITER).

    For BF16 (scaling_type="none"):
        out[g] = lhs[group_start:group_end] @ rhs[g].T + bias[g]

    For FP8 modes:
        Uses AITER MOE GEMM kernels where available.

    Args:
        lhs: Activation tensor ``[total_tokens, K]``.
        rhs: Weight tensor ``[num_experts, N, K]`` (or ``[num_experts, K, N]``
             depending on backend).
        group_sizes: Expert token counts ``[num_experts]``.
        scaling_type: Quantization mode.
        x_scale: Activation scale(s).
        w_scale: Weight scale(s).
        bias: Per-expert bias ``[num_experts, N]`` or ``None``.
        block_size: Block size for blockwise mode.
        fp8_dtype: Target FP8 dtype.

    Returns:
        Output tensor ``[total_tokens, N]``.
    """
    if scaling_type == "none":
        backends = []
        if _probe_aiter_gmm():
            backends.append((Backend.TRITON, lambda: _gmm_triton(lhs, rhs, group_sizes, bias)))
        return try_backends(backends, op_name="grouped_gemm_bf16")

    if scaling_type in ("delayed", "dynamic"):
        backends = []

        def _moe_a8w8():
            fn = _get_moe_gemm_a8w8()
            return fn(lhs, rhs, x_scale, w_scale, group_sizes)

        try:
            _get_moe_gemm_a8w8()
            backends.append((Backend.TRITON, _moe_a8w8))
        except (ImportError, OSError):
            pass

        def _sequential_fallback():
            return _grouped_gemm_fp8_sequential(
                lhs,
                rhs,
                group_sizes,
                x_scale,
                w_scale,
                scaling_type,
                bias,
            )

        backends.append((Backend.TRITON, _sequential_fallback))
        return try_backends(backends, op_name="grouped_gemm_per_tensor")

    if scaling_type == "blockwise":
        backends = []

        def _moe_blockscale():
            fn = _get_moe_gemm_a8w8_blockscale()
            return fn(lhs, rhs, x_scale, w_scale, group_sizes)

        try:
            _get_moe_gemm_a8w8_blockscale()
            backends.append((Backend.TRITON, _moe_blockscale))
        except (ImportError, OSError):
            pass

        def _sequential_fallback():
            return _grouped_gemm_fp8_sequential(
                lhs,
                rhs,
                group_sizes,
                x_scale,
                w_scale,
                scaling_type,
                bias,
            )

        backends.append((Backend.TRITON, _sequential_fallback))
        return try_backends(backends, op_name="grouped_gemm_blockscale")

    if scaling_type == "per_token":
        backends = []

        def _moe_per_token():
            from aiter.ops.triton.moe.moe_gemm_per_token import moe_gemm_per_token

            return moe_gemm_per_token(lhs, rhs, x_scale, w_scale, group_sizes, bias=bias)

        if _probe_aiter_moe_gemm_per_token():
            backends.append((Backend.TRITON, _moe_per_token))

        def _sequential_fallback():
            return _grouped_gemm_fp8_sequential(
                lhs,
                rhs,
                group_sizes,
                x_scale,
                w_scale,
                scaling_type,
                bias,
            )

        backends.append((Backend.TRITON, _sequential_fallback))
        return try_backends(backends, op_name="grouped_gemm_per_token")

    if scaling_type == "mxfp8":
        backends = []

        def _moe_mxfp8():
            from aiter.ops.triton.moe.moe_gemm_mxfp8 import moe_gemm_mxfp8

            return moe_gemm_mxfp8(lhs, rhs, x_scale, w_scale, group_sizes, bias=bias)

        if _probe_aiter_moe_gemm_mxfp8():
            backends.append((Backend.TRITON, _moe_mxfp8))

        def _sequential_fallback():
            return _grouped_gemm_fp8_sequential(
                lhs,
                rhs,
                group_sizes,
                x_scale,
                w_scale,
                scaling_type,
                bias,
            )

        backends.append((Backend.TRITON, _sequential_fallback))
        return try_backends(backends, op_name="grouped_gemm_mxfp8")

    raise ValueError(f"Unknown scaling_type={scaling_type!r}")


def _grouped_gemm_fp8_sequential(lhs, rhs, group_sizes, x_scale, w_scale, scaling_type, bias=None):
    """Sequential per-expert FP8 GEMM via AITER Triton GEMM backends."""
    from lumen.ops.quantize.linear import dispatch_gemm

    outputs = []
    offset = 0
    for g, size in enumerate(group_sizes):
        size = int(size)
        if size == 0:
            continue
        x_g = lhs[offset : offset + size]
        w_g = rhs[g]
        xs = x_scale[g] if x_scale is not None and x_scale.dim() > 0 else x_scale
        ws = w_scale[g] if w_scale is not None and w_scale.dim() > 0 else w_scale
        b_g = bias[g] if bias is not None else None
        out_g = dispatch_gemm(x_g, w_g, xs, ws, scaling_type, b_g)
        outputs.append(out_g)
        offset += size
    if not outputs:
        N = rhs.shape[-1] if rhs.dim() == 3 else rhs.shape[1]
        return torch.empty(0, N, device=lhs.device, dtype=torch.bfloat16)
    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Grouped GEMM backward (wgrad) — all via AITER
# ---------------------------------------------------------------------------


def grouped_gemm_wgrad(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    group_sizes: torch.Tensor,
    scaling_type: str = "none",
) -> torch.Tensor:
    """Grouped GEMM weight gradient: out[g] = grad[:, g_start:g_end].T @ input[g_start:g_end].

    For BF16 uses AITER's ptgmm (Persistent Transposed GMM).
    For FP8 modes uses sequential per-expert wgrad via AITER Triton GEMM.

    Args:
        grad_output: ``[total_tokens, N]``.
        input_tensor: ``[total_tokens, K]``.
        group_sizes: Expert token counts ``[num_experts]``.
        scaling_type: Quantization mode.

    Returns:
        Weight gradient ``[num_experts, N, K]``.
    """
    if scaling_type == "none":
        backends = []
        if _probe_aiter_gmm():

            def _ptgmm():
                fn = _get_ptgmm()
                return fn(grad_output, input_tensor, group_sizes)

            backends.append((Backend.TRITON, _ptgmm))
        return try_backends(backends, op_name="grouped_gemm_wgrad")

    # FP8 wgrad: sequential per-expert via AITER GEMM (in BF16 for numerical stability)
    from lumen.ops.quantize.linear import dispatch_gemm

    num_experts = len(group_sizes)
    N = grad_output.shape[-1]
    K = input_tensor.shape[-1]
    wgrad = torch.zeros(num_experts, N, K, device=grad_output.device, dtype=torch.bfloat16)
    offset = 0
    for g, size in enumerate(group_sizes):
        size = int(size)
        if size == 0:
            continue
        g_bf16 = grad_output[offset : offset + size].to(torch.bfloat16)
        x_bf16 = input_tensor[offset : offset + size].to(torch.bfloat16)
        # wgrad[g] = g^T @ x  →  dispatch(g^T, x^T) since kernel does A @ W^T → (g^T) @ (x^T)^T = g^T @ x
        wgrad[g] = dispatch_gemm(
            g_bf16.t().contiguous(),
            x_bf16.t().contiguous(),
            None,
            None,
            "none",
        )
        offset += size
    return wgrad
