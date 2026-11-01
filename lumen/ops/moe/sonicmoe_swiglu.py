###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Pre-routed BF16 grouped GEMM + interleaved SwiGLU from AITER 79c1a41.

The current AITER SonicMoE dropped ``grouped_gemm_swiglu`` and its backward.
Qwen3 FSDP BF16 still selects this path with ``LUMEN_EXPERT_FUSED_SWIGLU=1``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _local_tensor(tensor):
    if tensor is not None and hasattr(tensor, "to_local"):
        return tensor.to_local()
    return tensor


@triton.jit
def _grouped_gemm_swiglu_kernel(
    A_ptr,
    B_ptr,
    cu_seqlens_ptr,
    H_ptr,
    activation_ptr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_hm,
    stride_hn,
    stride_act_m,
    stride_act_n,
    I: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Grouped up-projection with an interleaved SwiGLU epilogue."""
    pid = tl.program_id(0)
    cumulative_blocks = 0
    expert_id = 0
    expert_start = 0
    expert_end = 0

    for e in range(E):
        start = tl.load(cu_seqlens_ptr + e).to(tl.int32)
        end = tl.load(cu_seqlens_ptr + e + 1).to(tl.int32)
        blocks = tl.cdiv(end - start, BLOCK_M) * tl.cdiv(I, BLOCK_I)
        if pid >= cumulative_blocks and pid < cumulative_blocks + blocks:
            expert_id = e
            expert_start = start
            expert_end = end
        cumulative_blocks += blocks

    if pid >= cumulative_blocks:
        return

    local_pid = pid
    for e in range(E):
        if e < expert_id:
            start = tl.load(cu_seqlens_ptr + e).to(tl.int32)
            end = tl.load(cu_seqlens_ptr + e + 1).to(tl.int32)
            local_pid -= tl.cdiv(end - start, BLOCK_M) * tl.cdiv(I, BLOCK_I)

    expert_rows = expert_end - expert_start
    num_pid_m = tl.cdiv(expert_rows, BLOCK_M)
    num_pid_i: tl.constexpr = tl.cdiv(I, BLOCK_I)
    num_pid_in_group = GROUP_SIZE_M * num_pid_i
    group_id = local_pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (local_pid % num_pid_in_group) % group_size_m
    pid_i = (local_pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_n = pid_i * (2 * BLOCK_I) + tl.arange(0, 2 * BLOCK_I)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < expert_rows
    i_mask = offs_i < I
    n_mask = offs_n < 2 * I
    global_m = expert_start + offs_m

    acc = tl.zeros((BLOCK_M, 2 * BLOCK_I), dtype=tl.float32)
    expert_id_i64 = expert_id.to(tl.int64)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask = k_offs < K
        a = tl.load(
            A_ptr
            + global_m[:, None].to(tl.int64) * stride_am
            + k_offs[None, :].to(tl.int64) * stride_ak,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            B_ptr
            + expert_id_i64 * stride_be
            + k_offs[:, None].to(tl.int64) * stride_bk
            + offs_n[None, :].to(tl.int64) * stride_bn,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b)

    # Preserve the original BF16 GEMM -> FP32 activation rounding order.
    h = acc.to(H_ptr.dtype.element_ty)
    pairs = tl.reshape(h, (BLOCK_M, BLOCK_I, 2))
    gate_bf16, up_bf16 = tl.split(pairs)
    gate = gate_bf16.to(tl.float32)
    up = up_bf16.to(tl.float32)
    activation = gate * tl.sigmoid(gate) * up

    tl.store(
        H_ptr
        + global_m[:, None].to(tl.int64) * stride_hm
        + offs_n[None, :].to(tl.int64) * stride_hn,
        h,
        mask=m_mask[:, None] & n_mask[None, :],
    )
    tl.store(
        activation_ptr
        + global_m[:, None].to(tl.int64) * stride_act_m
        + offs_i[None, :].to(tl.int64) * stride_act_n,
        activation.to(activation_ptr.dtype.element_ty),
        mask=m_mask[:, None] & i_mask[None, :],
    )


def grouped_gemm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    H: torch.Tensor,
    activation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse Qwen3's pre-routed BF16 up-projection and interleaved SwiGLU."""
    A = _local_tensor(A)
    B = _local_tensor(B)
    cu_seqlens = _local_tensor(cu_seqlens)
    H = _local_tensor(H)
    activation = _local_tensor(activation)

    E, K, N = B.shape
    I = N // 2
    if (
        A.dtype != torch.bfloat16
        or B.dtype != torch.bfloat16
        or N % 2
        or tuple(H.shape) != (A.shape[0], N)
        or tuple(activation.shape) != (A.shape[0], I)
        or cu_seqlens.numel() != E + 1
    ):
        raise ValueError("fused SwiGLU requires Qwen3-style BF16 tensors")
    if not all(t.is_contiguous() for t in (A, B, H, activation)):
        raise ValueError("fused SwiGLU requires contiguous tensors")

    block_m, block_i, block_k = 128, 64, 64
    grid = (
        (triton.cdiv(A.shape[0], block_m) + E - 1)
        * triton.cdiv(I, block_i),
    )
    _grouped_gemm_swiglu_kernel[grid](
        A,
        B,
        cu_seqlens,
        H,
        activation,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        H.stride(0),
        H.stride(1),
        activation.stride(0),
        activation.stride(1),
        I=I,
        K=K,
        E=E,
        BLOCK_M=block_m,
        BLOCK_I=block_i,
        BLOCK_K=block_k,
        GROUP_SIZE_M=8,
        num_warps=4,
        num_stages=2,
    )
    return H, activation


@triton.jit
def _grouped_gemm_swiglu_bwd_kernel(
    A_ptr,
    B_ptr,
    cu_seqlens_ptr,
    H_ptr,
    scores_ptr,
    da_ptr,
    act_ptr,
    dh_ptr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_hm,
    stride_hn,
    stride_da_m,
    stride_da_n,
    stride_act_m,
    stride_act_n,
    stride_dh_m,
    stride_dh_n,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    cumulative_blocks = 0
    expert_id = 0
    expert_start = 0
    expert_end = 0
    for e in range(E):
        start = tl.load(cu_seqlens_ptr + e).to(tl.int32)
        end = tl.load(cu_seqlens_ptr + e + 1).to(tl.int32)
        blocks_this = tl.cdiv(end - start, BLOCK_M) * tl.cdiv(N, BLOCK_N)
        if pid >= cumulative_blocks and pid < cumulative_blocks + blocks_this:
            expert_id = e
            expert_start = start
            expert_end = end
        cumulative_blocks += blocks_this
    if pid >= cumulative_blocks:
        return

    local_pid = pid
    for e in range(E):
        if e < expert_id:
            start = tl.load(cu_seqlens_ptr + e).to(tl.int32)
            end = tl.load(cu_seqlens_ptr + e + 1).to(tl.int32)
            local_pid -= tl.cdiv(end - start, BLOCK_M) * tl.cdiv(N, BLOCK_N)

    num_pid_m = tl.cdiv(expert_end - expert_start, BLOCK_M)
    num_pid_n: tl.constexpr = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = local_pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (local_pid % num_pid_in_group) % group_size_m
    pid_n = (local_pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < (expert_end - expert_start)
    n_mask = offs_n < N
    global_m = expert_start + offs_m
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    expert_id_i64 = expert_id.to(tl.int64)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        a = tl.load(
            A_ptr
            + global_m[:, None].to(tl.int64) * stride_am
            + k_offs[None, :].to(tl.int64) * stride_ak,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            B_ptr
            + expert_id_i64 * stride_be
            + k_offs[:, None].to(tl.int64) * stride_bk
            + offs_n[None, :].to(tl.int64) * stride_bn,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b)

    # Match the unfused path: round the GEMM to BF16, then scale in FP32.
    da = acc.to(tl.bfloat16)
    tl.store(
        da_ptr
        + global_m[:, None].to(tl.int64) * stride_da_m
        + offs_n[None, :].to(tl.int64) * stride_da_n,
        da,
        mask=m_mask[:, None] & n_mask[None, :],
    )
    score = tl.load(scores_ptr + global_m, mask=m_mask, other=0.0)
    da_f = da.to(tl.float32) * score[:, None]
    gate_offs = offs_n * 2
    gate = tl.load(
        H_ptr
        + global_m[:, None].to(tl.int64) * stride_hm
        + gate_offs[None, :].to(tl.int64) * stride_hn,
        mask=m_mask[:, None] & n_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        H_ptr
        + global_m[:, None].to(tl.int64) * stride_hm
        + (gate_offs + 1)[None, :].to(tl.int64) * stride_hn,
        mask=m_mask[:, None] & n_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    sig = tl.sigmoid(gate)
    act_gate = gate * sig
    tl.store(
        act_ptr
        + global_m[:, None].to(tl.int64) * stride_act_m
        + offs_n[None, :].to(tl.int64) * stride_act_n,
        (act_gate * up).to(tl.bfloat16),
        mask=m_mask[:, None] & n_mask[None, :],
    )
    d_up = da_f * act_gate
    d_gate = da_f * up * sig * (1.0 + gate * (1.0 - sig))
    tl.store(
        dh_ptr
        + global_m[:, None].to(tl.int64) * stride_dh_m
        + gate_offs[None, :].to(tl.int64) * stride_dh_n,
        d_gate.to(tl.bfloat16),
        mask=m_mask[:, None] & n_mask[None, :],
    )
    tl.store(
        dh_ptr
        + global_m[:, None].to(tl.int64) * stride_dh_m
        + (gate_offs + 1)[None, :].to(tl.int64) * stride_dh_n,
        d_up.to(tl.bfloat16),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def grouped_gemm_swiglu_bwd(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    hidden: torch.Tensor,
    scores: torch.Tensor,
    grad_activation: torch.Tensor,
    activation: torch.Tensor,
    grad_hidden: torch.Tensor,
) -> None:
    """Fuse the pre-routed down-projection dgrad with interleaved SwiGLU backward.

    ``B`` is ``[E, I, H]``. The GEMM is ``dout @ B^T``. The epilogue rereads the
    saved up-projection once and writes both the recomputed SwiGLU and its
    input gradient.
    """
    A = _local_tensor(A)
    B = _local_tensor(B).transpose(1, 2)
    cu_seqlens = _local_tensor(cu_seqlens)
    hidden = _local_tensor(hidden)
    scores = _local_tensor(scores).contiguous()
    grad_activation = _local_tensor(grad_activation)
    activation = _local_tensor(activation)
    grad_hidden = _local_tensor(grad_hidden)

    E, K, N = B.shape
    if (
        A.dtype != torch.bfloat16
        or hidden.dtype != torch.bfloat16
        or scores.dtype != torch.float32
        or A.shape != (hidden.shape[0], K)
        or hidden.shape[1] != N * 2
        or grad_activation.shape != (A.shape[0], N)
        or activation.shape != (A.shape[0], N)
        or grad_hidden.shape != hidden.shape
        or scores.numel() != A.shape[0]
        or cu_seqlens.numel() != E + 1
    ):
        raise ValueError("fused SwiGLU backward requires Qwen3-style BF16 tensors")

    block_m, block_n, block_k = 128, 64, 64
    grid = ((triton.cdiv(A.shape[0], block_m) + E - 1) * triton.cdiv(N, block_n),)
    _grouped_gemm_swiglu_bwd_kernel[grid](
        A,
        B,
        cu_seqlens,
        hidden,
        scores,
        grad_activation,
        activation,
        grad_hidden,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        hidden.stride(0),
        hidden.stride(1),
        grad_activation.stride(0),
        grad_activation.stride(1),
        activation.stride(0),
        activation.stride(1),
        grad_hidden.stride(0),
        grad_hidden.stride(1),
        N=N,
        K=K,
        E=E,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_SIZE_M=8,
        num_warps=4,
        num_stages=2,
    )
