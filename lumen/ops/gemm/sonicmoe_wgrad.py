###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""FP8 grouped wgrad that accepts padded scale rows.

``grouped_wgrad`` stores bf16. ``grouped_wgrad_accumulate`` rounds that bf16
value and adds it into an existing fp32 buffer, matching a bf16 store followed
by ``aten.add_``. Extra scale rows past the compact per-expert prefix are not
read, so the caller does not trim them on the host.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from lumen.ops.moe.sonicmoe_config import install_sonicmoe_gfx950_config


@triton.jit
def _lumen_grouped_gemm_dw_kernel(
    A_ptr,
    B_ptr,
    A_scale_ptr,
    B_scale_ptr,
    C_ptr,
    cu_seqlens_ptr,
    A_idx_ptr,
    stride_ak,
    stride_am,
    stride_bm,
    stride_bn,
    stride_ast,
    stride_ask,
    stride_bst,
    stride_bsn,
    stride_ce,
    stride_ck,
    stride_cn,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    SCALE_BLOCK_SIZE: tl.constexpr,
    BLOCKWISE_FP8: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_GATHER_IDX: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_k_blocks: tl.constexpr = tl.cdiv(K, BLOCK_K)
    num_n_blocks: tl.constexpr = tl.cdiv(N, BLOCK_N)
    blocks_per_expert: tl.constexpr = num_k_blocks * num_n_blocks

    expert_id = pid // blocks_per_expert
    local_pid = pid % blocks_per_expert
    pid_k = local_pid // num_n_blocks
    pid_n = local_pid % num_n_blocks

    expert_start = tl.load(cu_seqlens_ptr + expert_id).to(tl.int32)
    expert_end = tl.load(cu_seqlens_ptr + expert_id + 1).to(tl.int32)
    expert_rows = expert_end - expert_start
    scale_expert_start = 0
    if BLOCKWISE_FP8:
        for expert in range(E):
            if expert < expert_id:
                start = tl.load(cu_seqlens_ptr + expert).to(tl.int32)
                end = tl.load(cu_seqlens_ptr + expert + 1).to(tl.int32)
                scale_expert_start += tl.cdiv(end - start, SCALE_BLOCK_SIZE)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_t = tl.arange(0, BLOCK_T)
    k_mask = offs_k < K
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    a_dtype = A_ptr.dtype.element_ty

    for token_start in range(0, expert_rows, BLOCK_T):
        token_offs = token_start + offs_t
        token_mask = token_offs < expert_rows
        global_token = expert_start + token_offs
        if HAS_GATHER_IDX:
            row_idx = tl.load(A_idx_ptr + global_token, mask=token_mask, other=0).to(tl.int64)
        else:
            row_idx = global_token.to(tl.int64)
        a = tl.load(
            A_ptr
            + offs_k[:, None].to(tl.int64) * stride_am
            + row_idx[None, :] * stride_ak,
            mask=k_mask[:, None] & token_mask[None, :],
            other=0.0,
        ).to(a_dtype)
        b = tl.load(
            B_ptr
            + global_token[:, None].to(tl.int64) * stride_bm
            + offs_n[None, :].to(tl.int64) * stride_bn,
            mask=token_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(a_dtype)
        dot = tl.dot(a, b)
        if BLOCKWISE_FP8:
            scale_token = scale_expert_start + token_start // SCALE_BLOCK_SIZE
            a_scale = tl.load(
                A_scale_ptr + scale_token * stride_ast + offs_k.to(tl.int64) * stride_ask,
                mask=k_mask,
                other=0.0,
            )
            b_scale = tl.load(
                B_scale_ptr + scale_token * stride_bst + offs_n.to(tl.int64) * stride_bsn,
                mask=n_mask,
                other=0.0,
            )
            dot *= a_scale[:, None] * b_scale[None, :]
        acc += dot

    expert_id_i64 = expert_id.to(tl.int64)
    out_ptrs = (
        C_ptr
        + expert_id_i64 * stride_ce
        + offs_k[:, None].to(tl.int64) * stride_ck
        + offs_n[None, :].to(tl.int64) * stride_cn
    )
    out_mask = k_mask[:, None] & n_mask[None, :]
    if ACCUMULATE:
        rounded = acc.to(tl.bfloat16).to(tl.float32)
        previous = tl.load(out_ptrs, mask=out_mask, other=0.0)
        tl.store(out_ptrs, previous + rounded, mask=out_mask)
    else:
        tl.store(out_ptrs, acc.to(C_ptr.dtype.element_ty), mask=out_mask)


def _launch_grouped_wgrad(
    a: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    block_size: int,
    accumulate: bool,
) -> None:
    """Launch grouped FP8 wgrad. Extra scale rows past the compact prefix are unread."""
    install_sonicmoe_gfx950_config()
    from aiter.ops.triton.utils.sonicmoe_config_utils import (
        get_grouped_gemm_dw_config,
        split_launch_config,
    )

    experts = cu_seqlens.shape[0] - 1
    k_dim = a.shape[1]
    n_dim = b.shape[1]
    if tuple(out.shape) != (experts, k_dim, n_dim):
        raise ValueError(
            f"wgrad out shape {tuple(out.shape)} != {(experts, k_dim, n_dim)}"
        )
    min_scale_rows = triton.cdiv(a.shape[0], block_size)
    if a_scale.shape[0] < min_scale_rows or a_scale.shape[1] != k_dim:
        raise ValueError(
            f"A_scale must have shape [>={min_scale_rows}, {k_dim}], got {tuple(a_scale.shape)}"
        )
    if b_scale.shape[0] < min_scale_rows or b_scale.shape[1] != n_dim:
        raise ValueError(
            f"B_scale must have shape [>={min_scale_rows}, {n_dim}], got {tuple(b_scale.shape)}"
        )

    def grid(meta):
        return (
            experts
            * triton.cdiv(k_dim, meta["BLOCK_K"])
            * triton.cdiv(n_dim, meta["BLOCK_N"]),
        )

    use_tuned = os.environ.get("SONIC_MOE_USE_QWEN3_TUNED_GEMM", "0") == "1"
    config = get_grouped_gemm_dw_config(n_dim, k_dim, experts, False, use_tuned)
    constexprs, launch = split_launch_config(config)
    _lumen_grouped_gemm_dw_kernel[grid](
        a,
        b,
        a_scale,
        b_scale,
        out,
        cu_seqlens,
        cu_seqlens,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        b_scale.stride(0),
        b_scale.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        N=n_dim,
        K=k_dim,
        E=experts,
        SCALE_BLOCK_SIZE=block_size,
        BLOCKWISE_FP8=True,
        HAS_GATHER_IDX=False,
        ACCUMULATE=accumulate,
        **constexprs,
        **launch,
    )


def grouped_wgrad(
    a: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Write one grouped FP8 wgrad in bf16, without a host scale-row sync."""
    experts = cu_seqlens.shape[0] - 1
    out = torch.empty(
        experts, a.shape[1], b.shape[1], dtype=torch.bfloat16, device=a.device
    )
    _launch_grouped_wgrad(
        a, b, cu_seqlens, out, a_scale, b_scale, block_size, accumulate=False
    )
    return out


def grouped_wgrad_accumulate(
    a: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Accumulate one grouped FP8 wgrad into ``out``."""
    if out.dtype != torch.float32:
        raise ValueError("grouped wgrad accumulate requires an fp32 out buffer")
    _launch_grouped_wgrad(
        a, b, cu_seqlens, out, a_scale, b_scale, block_size, accumulate=True
    )
    return out
