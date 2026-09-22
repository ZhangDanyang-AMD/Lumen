###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""FlyDSL gfx950 grouped GEMMs for pre-routed SonicMoE.

When ``SONIC_MOE_GEMM_BACKEND=flydsl``, the BF16 pre-routed expert path uses
the native FlyDSL training lifecycle on Qwen3 E16 (``forward_routes_training``
plus retained-state ``sonic_moe_backward_routes``).  Other shapes fall back
to gemm1/gemm2 forward and FlyDSL NN/TN grouped GEMMs for dgrad/wgrad.
Qwen3-local defaults come from
``tests/kernels/bench_sonic_moe_aiter_vs_flydsl.py --sweep-tiles`` on T=8192
with real uneven expert counts.
"""

from __future__ import annotations

import functools
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F


_SORT_UNIT = 64
_DX_BLOCK_M = 16
_DX_BLOCK_N = 64
_DX_BLOCK_K = 64
_DA_BLOCK_M = 64
_DA_BLOCK_N = 64
_DA_BLOCK_K = 64
_QWEN3_LOCAL_E = 16
_QWEN3_H = 2048
_QWEN3_I = 768
_QWEN3_I_FULL = 1536
_QWEN3_NN_DX_TILE = (128, 128, 64)  # dh @ W1.T → dX, N=H=2048
_QWEN3_NN_DA_TILE = (128, 128, 64)  # dy @ W2.T → dA, N=I=768
_QWEN3_TN_DW1_TILE = (128, 128, 64, 0, 2, 2)  # X^T @ dH, [H, 2I]
_QWEN3_TN_DW2_TILE = (128, 128, 64, 0, 2, 2)  # A'^T @ dY, [I, H]
_INSTALLED = False
_PRESHUFFLE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_WEIGHT_KN_CACHE: dict[int, tuple[torch.Tensor, int, torch.Tensor]] = {}
_DUMMY_SCALE: dict[int, torch.Tensor] = {}
_NATIVE_ROUTES_CACHE: dict[
    tuple[int, int],
    tuple[int, int, object, object, torch.Tensor, torch.Tensor],
] = {}
_TILE_OVERRIDE: "FlyDSLTileOverride | None" = None


@dataclass(frozen=True)
class FlyDSLTileOverride:
    """Optional tile override for Qwen3-local FlyDSL kernels.

    Forward tuples are ``(BM, BN, BK)`` or ``(BM, BN, BK, xcd, b_cache_mod)``.
    NN dgrad tuples are ``(BM, BN, BK)``. TN wgrad tuples are
    ``(BM, BN, BK, k_padding, m_waves, n_waves)``.
    """

    gemm1: tuple[int, ...] | None = None
    gemm2: tuple[int, ...] | None = None
    dx: tuple[int, int, int] | None = None
    da: tuple[int, int, int] | None = None
    dw1: tuple[int, ...] | None = None
    dw2: tuple[int, ...] | None = None


@contextmanager
def flydsl_tile_override(**tiles):
    """Temporarily replace Qwen3-local gemm1/gemm2/NN/TN tiles."""
    global _TILE_OVERRIDE
    previous = _TILE_OVERRIDE
    _TILE_OVERRIDE = FlyDSLTileOverride(**tiles)
    try:
        yield _TILE_OVERRIDE
    finally:
        _TILE_OVERRIDE = previous


def _unpack_fwd_tile(tile: tuple[int, ...], default_xcd: int, default_bcm: int):
    block_m, tile_n, tile_k = tile[:3]
    xcd = tile[3] if len(tile) > 3 else default_xcd
    bcm = tile[4] if len(tile) > 4 else default_bcm
    return int(block_m), int(tile_n), int(tile_k), int(xcd), int(bcm)


@functools.lru_cache(maxsize=1)
def _pre_routed_metadata_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        cu_seqlens,
        frequency,
        active_queue,
        sorted_expert_ids,
        num_valid_ids,
        dest_index,
        sorted_token_ids,
        num_tokens,
        capacity,
        NUM_EXPERTS: tl.constexpr,
        SORT_UNIT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < capacity
        selected_expert = tl.full((BLOCK,), NUM_EXPERTS, tl.int32)
        selected_token = tl.full((BLOCK,), num_tokens, tl.int32)
        padded_start = 0
        active_count = 0
        for expert in tl.static_range(NUM_EXPERTS):
            compact_start = tl.load(cu_seqlens + expert)
            compact_end = tl.load(cu_seqlens + expert + 1)
            count = compact_end - compact_start
            padded_count = tl.cdiv(count, SORT_UNIT) * SORT_UNIT
            is_active = count > 0
            lane = tl.arange(0, BLOCK)
            tl.store(
                active_queue + 1 + 2 * active_count + lane,
                expert,
                mask=(pid == 0) & (lane == 0) & is_active,
            )
            tl.store(
                active_queue + 2 + 2 * active_count + lane,
                padded_start,
                mask=(pid == 0) & (lane == 0) & is_active,
            )
            active_count += is_active
            in_expert = (offsets >= padded_start) & (
                offsets < padded_start + padded_count
            )
            is_token = in_expert & (offsets < padded_start + count)
            selected_expert = tl.where(in_expert, expert, selected_expert)
            selected_token = tl.where(
                is_token, compact_start + offsets - padded_start, selected_token
            )
            tl.store(
                frequency + expert + tl.arange(0, BLOCK),
                count,
                mask=(pid == 0) & (tl.arange(0, BLOCK) == 0),
            )
            padded_start += padded_count

        is_token = valid & (selected_token < num_tokens)
        tl.store(sorted_token_ids + offsets, selected_token, mask=valid)
        tl.store(dest_index + selected_token, offsets, mask=is_token)
        is_block_start = valid & ((offsets % SORT_UNIT) == 0)
        tl.store(
            sorted_expert_ids + offsets // SORT_UNIT,
            selected_expert,
            mask=is_block_start,
        )
        tl.store(
            num_valid_ids + tl.arange(0, BLOCK),
            padded_start,
            mask=(pid == 0) & (tl.arange(0, BLOCK) == 0),
        )
        tl.store(
            active_queue + tl.arange(0, BLOCK),
            active_count,
            mask=(pid == 0) & (tl.arange(0, BLOCK) == 0),
        )

    return kernel


@functools.lru_cache(maxsize=1)
def _compact_queue_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(cu_seqlens, frequency, queue, NUM_EXPERTS: tl.constexpr):
        count = 0
        for expert in tl.static_range(NUM_EXPERTS):
            start = tl.load(cu_seqlens + expert)
            end = tl.load(cu_seqlens + expert + 1)
            tokens = end - start
            tl.store(frequency + expert, tokens)
            if tokens > 0:
                tl.store(queue + 1 + 2 * count, expert)
                tl.store(queue + 2 + 2 * count, start)
                count += 1
        tl.store(queue, count)

    return kernel


def padded_row_capacity(
    num_tokens: int, num_experts: int, sort_unit: int = _SORT_UNIT
) -> int:
    """Host-known padded-row bound rounded to ``sort_unit``."""
    if num_tokens < 0 or num_experts <= 0:
        raise ValueError("num_tokens must be >= 0 and num_experts > 0")
    if sort_unit <= 0:
        raise ValueError("sort_unit must be positive")
    raw = num_tokens + num_experts * (sort_unit - 1)
    return ((raw + sort_unit - 1) // sort_unit) * sort_unit


@dataclass
class PreRoutedFlyDSLLayout:
    """Padded expert-major metadata for FlyDSL grouped kernels."""

    frequency: torch.Tensor
    active_queue: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor
    dest_index: torch.Tensor
    sorted_token_ids: torch.Tensor
    padded_capacity: int
    num_tokens: int
    num_experts: int
    sort_unit: int


def build_pre_routed_metadata(
    cu_seqlens: torch.Tensor,
    num_tokens: int | None = None,
    sort_unit: int = _SORT_UNIT,
) -> PreRoutedFlyDSLLayout:
    """Build FlyDSL sorter metadata from unpadded expert-major ``cu_seqlens``.

    Empty experts occupy no padded rows.  ``num_valid_ids[0]`` is the actual
    padded length; activation buffers may be allocated at ``padded_capacity``.
    Pass ``num_tokens`` (the compact activation length) to avoid a D2H sync.
    """
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a rank-1 tensor of length E+1")
    cu_seqlens = cu_seqlens.to(dtype=torch.int32).contiguous()
    device = cu_seqlens.device
    num_experts = int(cu_seqlens.numel() - 1)
    if num_tokens is None:
        num_tokens = int(cu_seqlens[-1].item())
    capacity = padded_row_capacity(num_tokens, num_experts, sort_unit)
    max_blocks = max(capacity // sort_unit, 1)
    if cu_seqlens.is_cuda and num_experts <= 32:
        import triton

        frequency = torch.empty(num_experts, dtype=torch.int32, device=device)
        active_queue = torch.empty(
            1 + 2 * num_experts, dtype=torch.int32, device=device
        )
        sorted_expert_ids = torch.empty(
            max_blocks, dtype=torch.int32, device=device
        )
        num_valid_ids = torch.empty(1, dtype=torch.int32, device=device)
        dest_index = torch.empty(num_tokens, dtype=torch.int32, device=device)
        sorted_token_ids = torch.empty(
            capacity, dtype=torch.int32, device=device
        )
        block = 256
        _pre_routed_metadata_kernel()[(triton.cdiv(capacity, block),)](
            cu_seqlens,
            frequency,
            active_queue,
            sorted_expert_ids,
            num_valid_ids,
            dest_index,
            sorted_token_ids,
            num_tokens,
            capacity,
            NUM_EXPERTS=num_experts,
            SORT_UNIT=sort_unit,
            BLOCK=block,
        )
        return PreRoutedFlyDSLLayout(
            frequency=frequency,
            active_queue=active_queue,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            dest_index=dest_index,
            sorted_token_ids=sorted_token_ids,
            padded_capacity=capacity,
            num_tokens=num_tokens,
            num_experts=num_experts,
            sort_unit=sort_unit,
        )
    frequency = (cu_seqlens[1:] - cu_seqlens[:-1]).contiguous()
    padded_counts = ((frequency + (sort_unit - 1)) // sort_unit) * sort_unit
    padded_cu = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    padded_cu[0] = 0
    padded_cu[1:] = padded_counts.cumsum(0, dtype=torch.int32)
    num_valid_ids = padded_cu[-1:].contiguous()
    block_starts = torch.arange(max_blocks, device=device, dtype=torch.int32) * sort_unit
    sorted_expert_ids = torch.searchsorted(
        padded_cu[1:], block_starts, right=True
    ).to(torch.int32)
    if num_tokens == 0:
        dest_index = torch.empty(0, dtype=torch.int32, device=device)
    else:
        rows = torch.arange(num_tokens, device=device, dtype=torch.int32)
        expert = torch.searchsorted(cu_seqlens[1:], rows, right=True).to(torch.int32)
        dest_index = padded_cu[expert] + (rows - cu_seqlens[expert])
    sorted_token_ids = torch.full(
        (capacity,),
        num_tokens,
        dtype=torch.int32,
        device=device,
    )
    if num_tokens:
        sorted_token_ids[dest_index] = rows
    active = torch.nonzero(frequency, as_tuple=False).flatten().to(torch.int32)
    active_queue = torch.zeros(
        1 + 2 * num_experts, dtype=torch.int32, device=device
    )
    active_queue[0] = int(active.numel())
    if active.numel():
        active_queue[1 : 1 + 2 * active.numel() : 2] = active
        active_queue[2 : 2 + 2 * active.numel() : 2] = padded_cu[:-1].index_select(
            0, active
        )
    return PreRoutedFlyDSLLayout(
        frequency=frequency,
        active_queue=active_queue,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        dest_index=dest_index,
        sorted_token_ids=sorted_token_ids,
        padded_capacity=capacity,
        num_tokens=num_tokens,
        num_experts=num_experts,
        sort_unit=sort_unit,
    )


def _choose_tile(size: int, candidates: tuple[int, ...] = (256, 128, 64)) -> int:
    for tile in candidates:
        if size % tile == 0:
            return tile
    raise ValueError(f"no tile in {candidates} divides {size}")


def _preshuffle_16bit_weight(weight: torch.Tensor) -> torch.Tensor:
    """Convert ``[..., N, K]`` BF16 rows to the 16x16 N-major MFMA layout."""
    n, k = weight.shape[-2:]
    if n % 16 != 0 or k % 32 != 0:
        raise ValueError(f"weight N/K must be divisible by 16/32, got {n}/{k}")
    x = weight.detach().to(dtype=torch.bfloat16).contiguous()
    leading = x.numel() // (n * k)
    return (
        x.view(leading, n // 16, 16, k // 32, 4, 8)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .view_as(x)
    )


def _preshuffled_weights(w1_kn: torch.Tensor, w2_kn: torch.Tensor):
    key = (
        w1_kn.data_ptr(),
        int(w1_kn._version),
        w2_kn.data_ptr(),
        int(w2_kn._version),
    )
    cached = _PRESHUFFLE_CACHE.get(key)
    if cached is not None:
        return cached
    gate_up = _preshuffle_16bit_weight(w1_kn.transpose(1, 2).contiguous())
    down = _preshuffle_16bit_weight(w2_kn.transpose(1, 2).contiguous())
    _PRESHUFFLE_CACHE[key] = (gate_up, down)
    if len(_PRESHUFFLE_CACHE) > 64:
        _PRESHUFFLE_CACHE.pop(next(iter(_PRESHUFFLE_CACHE)))
    return gate_up, down


def _dummy_scale(device: torch.device) -> torch.Tensor:
    index = device.index or 0
    tensor = _DUMMY_SCALE.get(index)
    if tensor is None or tensor.device != device:
        tensor = torch.zeros(1, dtype=torch.uint8, device=device)
        _DUMMY_SCALE[index] = tensor
    return tensor


def _swiglu(h: torch.Tensor) -> torch.Tensor:
    gate, up = h.chunk(2, dim=-1)
    return F.silu(gate) * up


def _swiglu_backward(h: torch.Tensor, dact: torch.Tensor) -> torch.Tensor:
    gate, up = h.chunk(2, dim=-1)
    sigmoid = torch.sigmoid(gate.float())
    silu = gate.float() * sigmoid
    dact_f = dact.float()
    d_up = dact_f * silu
    d_gate = dact_f * up.float() * sigmoid * (1.0 + gate.float() * (1.0 - sigmoid))
    return torch.cat((d_gate, d_up), dim=-1).to(dtype=h.dtype)


@functools.lru_cache(maxsize=32)
def _compile_gemm1(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    block_m: int,
    sorted_block_m: int,
    tile_n: int,
    tile_k: int,
    b_cache_mod: int,
    xcd_swizzle: int,
    skip_epilogue_id_reload: bool,
    a_lds_swizzle: bool,
    device_index: int,
):
    from kernels.moe.moe_2stage_a16wmix.gemm1 import compile_gemm1_a16w4_port

    del device_index
    return compile_gemm1_a16w4_port(
        BM=block_m,
        SORTED_BM=sorted_block_m,
        D_HIDDEN=hidden_size,
        D_INTER=intermediate_size,
        NE=num_experts,
        TOPK=1,
        TILE_N=tile_n,
        TILE_K=tile_k,
        act="silu",
        b_cache_mod=b_cache_mod,
        xcd_swizzle=xcd_swizzle,
        w_dtype="bf16",
        a_dtype="bf16",
        w_layout="standard",
        k_wave=1,
        persist=False,
        round_preact_bf16=True,
        has_bias=False,
        store_route_preactivation=True,
        # Pre-routed K=1 rows are already in route order, so their compact
        # token id is also the original route id.  The current gemm1 contract
        # requires this explicit route-id path when the fast epilogue skips
        # reloading token validity: padding carries the sentinel ``tokens``
        # and must be rejected by ``route_row < nroutes`` before writing the
        # compact preactivation buffer.
        route_preactivation_by_route_id=True,
        skip_epilogue_id_reload=skip_epilogue_id_reload,
        a_lds_swizzle=a_lds_swizzle,
    )


@functools.lru_cache(maxsize=32)
def _compile_gemm2(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    block_m: int,
    sorted_block_m: int,
    tile_n: int,
    tile_k: int,
    b_cache_mod: int,
    xcd_swizzle: int,
    device_index: int,
):
    from kernels.moe.moe_2stage_a16wmix.gemm2 import compile_gemm2_a16w4_port

    del device_index
    stages = 2 if intermediate_size // tile_k > 1 else 1
    return compile_gemm2_a16w4_port(
        BM=block_m,
        SORTED_BM=sorted_block_m,
        NE=num_experts,
        N_OUT=hidden_size,
        D_INTER=intermediate_size,
        TILE_N=tile_n,
        TILE_K=tile_k,
        xcd_swizzle=xcd_swizzle,
        b_cache_mod=b_cache_mod,
        w_dtype="bf16",
        a_dtype="bf16",
        persist=False,
        has_bias=False,
        round_projection_bf16=True,
        output_mode="atomic",
        TOPK=1,
        stages=stages,
    )


def _cu_seqlens_from_counts(counts, num_experts: int, device) -> torch.Tensor:
    counts = torch.as_tensor(counts, dtype=torch.int32)
    if counts.numel() != num_experts:
        raise ValueError(f"Expected {num_experts} expert counts, got {counts.numel()}")
    counts = counts.to(device=device, dtype=torch.int32)
    cu = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    cu[0] = 0
    cu[1:] = counts.cumsum(0, dtype=torch.int32)
    return cu


class _FlyDSLPreRouted(torch.autograd.Function):
    """Pre-routed BF16 Sonic forward (gemm1/gemm2) and grouped backward."""

    @staticmethod
    def forward(ctx, hidden_states, scores, w1, w2, cu_seqlens):
        from kernels.common.tensor_shim import _run_compiled
        from kernels.moe.moe_2stage_a16wmix.gemm1 import gemm1_a16w4_grid
        from kernels.moe.moe_2stage_a16wmix.gemm2 import gemm2_a16w4_grid

        hidden_states = hidden_states.contiguous()
        tokens = int(hidden_states.shape[0])
        hidden_size = int(hidden_states.shape[1])
        intermediate_size = int(w2.shape[1])
        num_experts = int(w1.shape[0])
        device = hidden_states.device
        native_qwen3_profile = (
            hidden_size == 2048
            and intermediate_size == 768
            and num_experts == 16
        )
        if native_qwen3_profile:
            # Port the measured dense Qwen3 winner to the EP-local pre-routed
            # shape. Both stages share route metadata padded to lcm(BM1, BM2).
            gemm1_m, gemm1_n, gemm1_k, gemm1_xcd, gemm1_bcm = (
                128,
                128,
                64,
                8,
                0,
            )
            gemm2_m, gemm2_n, gemm2_k, gemm2_xcd, gemm2_bcm = (
                64,
                256,
                64,
                0,
                0,
            )
        else:
            gemm1_m = gemm2_m = _SORT_UNIT
            gemm1_n = _choose_tile(intermediate_size)
            gemm1_k = _choose_tile(hidden_size)
            gemm2_n = _choose_tile(hidden_size)
            gemm2_k = _choose_tile(intermediate_size)
            gemm1_bcm = gemm2_bcm = 2
            gemm1_xcd, gemm2_xcd = 0, 1
        override = _TILE_OVERRIDE
        if override is not None:
            if override.gemm1 is not None:
                gemm1_m, gemm1_n, gemm1_k, gemm1_xcd, gemm1_bcm = _unpack_fwd_tile(
                    override.gemm1, gemm1_xcd, gemm1_bcm
                )
            if override.gemm2 is not None:
                gemm2_m, gemm2_n, gemm2_k, gemm2_xcd, gemm2_bcm = _unpack_fwd_tile(
                    override.gemm2, gemm2_xcd, gemm2_bcm
                )
        sort_unit = math.lcm(gemm1_m, gemm2_m)
        layout = build_pre_routed_metadata(
            cu_seqlens, num_tokens=tokens, sort_unit=sort_unit
        )
        gate_up, down = _preshuffled_weights(w1, w2)
        dummy = _dummy_scale(device)
        if (gemm2_m * gemm2_k) % 2048:
            raise ValueError(
                f"gemm2 requires BM*TILE_K divisible by 2048, got BM={gemm2_m} TILE_K={gemm2_k}"
            )
        gemm1_max_m_blocks = max(layout.padded_capacity // gemm1_m, 1)
        gemm2_max_m_blocks = max(layout.padded_capacity // gemm2_m, 1)
        intermediate = hidden_states.new_empty(layout.padded_capacity, intermediate_size)
        preactivation = hidden_states.new_empty(tokens, 2 * intermediate_size)
        output = hidden_states.new_zeros(tokens, hidden_size)
        sorted_weights = torch.zeros(
            layout.padded_capacity, dtype=torch.float32, device=device
        )
        if tokens:
            sorted_weights[layout.dest_index] = scores.reshape(-1).float()
        stream = torch.cuda.current_stream(device)
        grid1 = gemm1_a16w4_grid(
            gemm1_m,
            INTER=intermediate_size,
            TILE_N=gemm1_n,
            max_m_blocks=gemm1_max_m_blocks,
        )
        _run_compiled(
            _compile_gemm1(
                hidden_size,
                intermediate_size,
                num_experts,
                gemm1_m,
                sort_unit,
                gemm1_n,
                gemm1_k,
                gemm1_bcm,
                gemm1_xcd,
                native_qwen3_profile,
                native_qwen3_profile,
                device.index or 0,
            ),
            hidden_states.data_ptr(),
            gate_up.data_ptr(),
            dummy.data_ptr(),
            dummy.data_ptr(),
            layout.sorted_expert_ids.data_ptr(),
            layout.num_valid_ids.data_ptr(),
            layout.sorted_token_ids.data_ptr(),
            tokens,
            int(grid1),
            1.0,
            1.0,
            1.0,
            1.0,
            float("inf"),
            intermediate.data_ptr(),
            preactivation.data_ptr(),
            # For pre-routed K=1, route id == compact token id.  Padding rows
            # contain ``tokens`` and are masked by gemm1's nroutes bound.
            layout.sorted_token_ids.data_ptr(),
            tokens,
            stream,
        )
        grid2 = gemm2_a16w4_grid(
            gemm2_m,
            N_OUT=hidden_size,
            TILE_N=gemm2_n,
            max_m_blocks=gemm2_max_m_blocks,
        )
        _run_compiled(
            _compile_gemm2(
                hidden_size,
                intermediate_size,
                num_experts,
                gemm2_m,
                sort_unit,
                gemm2_n,
                gemm2_k,
                gemm2_bcm,
                gemm2_xcd,
                device.index or 0,
            ),
            intermediate.data_ptr(),
            down.data_ptr(),
            dummy.data_ptr(),
            dummy.data_ptr(),
            layout.sorted_expert_ids.data_ptr(),
            layout.num_valid_ids.data_ptr(),
            layout.sorted_token_ids.data_ptr(),
            sorted_weights.data_ptr(),
            tokens,
            gemm2_max_m_blocks,
            int(grid2),
            output.data_ptr(),
            stream,
        )
        ctx.layout = layout
        ctx.save_for_backward(hidden_states, preactivation, w1, w2, scores, cu_seqlens)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, preactivation, w1, w2, scores, cu_seqlens = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        tokens = hidden_states.shape[0]
        dx = torch.zeros_like(hidden_states)
        dw1 = torch.zeros_like(w1)
        dw2 = torch.zeros_like(w2)
        dscores = torch.zeros_like(scores)
        if tokens == 0:
            return dx, dscores, dw1, dw2, None

        layout = ctx.layout
        score_f = scores.reshape(-1).float()
        dout = grad_output
        da_unscaled = torch.empty(
            tokens, w2.shape[1], dtype=hidden_states.dtype, device=hidden_states.device
        )
        _flydsl_grouped_dgrad_padded(dout, w2, da_unscaled, layout)
        a_prime = _swiglu(preactivation)
        dscores = (a_prime.float() * da_unscaled.float()).sum(dim=-1).reshape_as(scores)
        dy = (dout.float() * score_f[:, None]).to(dtype=hidden_states.dtype)
        _flydsl_grouped_wgrad_padded(a_prime, dy, dw2, layout)
        dh = _swiglu_backward(
            preactivation,
            (da_unscaled.float() * score_f[:, None]).to(preactivation.dtype),
        )
        _flydsl_grouped_wgrad_padded(hidden_states, dh, dw1, layout)
        _flydsl_grouped_dgrad_padded(dh, w1, dx, layout)
        return dx, dscores, dw1, dw2, None


def _native_routes_operator(
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    native_weight_layout: bool,
):
    """Return a native routes operator and its logical backward weights."""
    from kernels.moe.sonic import (
        SonicMoE,
        SonicMoEConfig,
        prepare_sonic_bf16_weights,
    )

    key = (w1.data_ptr(), w2.data_ptr())
    versions = (int(w1._version), int(w2._version))
    cached = _NATIVE_ROUTES_CACHE.get(key)
    if cached is not None and cached[:2] == versions:
        return cached[2:]

    # Native Sonic uses logical [E, N, K]. Direct callers retain the historical
    # [E, K, N] ABI; SonicMoEExperts(flydsl) stores native weights end-to-end.
    logical_w1 = (
        w1.detach().contiguous()
        if native_weight_layout
        else w1.detach().transpose(1, 2).contiguous()
    )
    logical_w2 = (
        w2.detach().contiguous()
        if native_weight_layout
        else w2.detach().transpose(1, 2).contiguous()
    )
    num_experts, projection_size, hidden_size = map(int, logical_w1.shape)
    intermediate_size = int(logical_w2.shape[2])
    exact_qwen3_e16 = (
        num_experts,
        hidden_size,
        projection_size,
        intermediate_size,
    ) == (16, 2048, 1536, 768)
    if exact_qwen3_e16:
        config = SonicMoEConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=1,
            tile_m=128,
            tile_n=128,
            tile_k=64,
            down_tile_m=64,
            down_tile_n=256,
            down_tile_k=64,
            renormalize=False,
            stage1_b_cache_mod=0,
            stage2_b_cache_mod=0,
            stage1_xcd_swizzle=8,
            stage1_k_wave=1,
            stage2_xcd_swizzle=0,
            stage2_pipeline_stages=2,
            stage1_write_padded_rows=True,
            stage1_lds_swizzle=True,
            activation="swiglu",
            compute_dtype="bf16",
        )
    else:
        # Conservative native fallback used by correctness/small-shape calls.
        # Production Qwen3 always takes the measured profile above.
        config = SonicMoEConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=1,
            tile_m=32,
            tile_n=_choose_tile(intermediate_size, (128, 64)),
            tile_k=_choose_tile(hidden_size, (128, 64)),
            down_tile_m=32,
            down_tile_n=_choose_tile(hidden_size, (128, 64)),
            down_tile_k=_choose_tile(intermediate_size, (128, 64)),
            renormalize=False,
            activation="swiglu",
            compute_dtype="bf16",
        )
    # One reusable workspace per layer. The FlyDSL default of 8, multiplied by
    # 48 Qwen3 layers and EP-overlap streams, retained ~100 GiB and OOM'd the
    # MBS=2 GBS=256 recipe during the first forward.
    operator = SonicMoE(
        config,
        prepare_sonic_bf16_weights(logical_w1, logical_w2, config),
        max_cached_workspaces=1,
    )
    value = (
        versions[0],
        versions[1],
        operator,
        config,
        logical_w1,
        logical_w2,
    )
    _NATIVE_ROUTES_CACHE[key] = value
    return value[2:]


class _FlyDSLNativeRoutes(torch.autograd.Function):
    """Expert-major route-training and retained-state backward adapter."""

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        scores,
        w1,
        w2,
        cu_seqlens,
        native_weight_layout,
    ):
        from kernels.moe.sonic_backward import sonic_moe_backward_routes

        del sonic_moe_backward_routes  # Imported here so missing ce7b755 fails early.
        hidden_states = hidden_states.contiguous()
        scores = scores.reshape(-1).float().contiguous()
        cu_seqlens = cu_seqlens.to(
            device=hidden_states.device, dtype=torch.int32
        ).contiguous()
        tokens = int(hidden_states.shape[0])
        num_experts = int(w1.shape[0])

        operator, config, logical_w1, logical_w2 = _native_routes_operator(
            w1,
            w2,
            native_weight_layout=bool(native_weight_layout),
        )
        # Workspace outputs are reusable; autograd must own this invocation's
        # output until Megatron finishes unpermutation and backward.
        output = torch.empty_like(hidden_states)
        # Megatron's permute plus EP all-to-all already produce expert-major
        # rows. Materialize the flat-route tensors while passing cu_seqlens so
        # FlyDSL can skip its generic counting sort and use direct output stores.
        token_indices = torch.arange(
            tokens, dtype=torch.int32, device=hidden_states.device
        )
        counts = cu_seqlens[1:] - cu_seqlens[:-1]
        expert_indices = torch.repeat_interleave(
            torch.arange(
                num_experts, dtype=torch.int32, device=hidden_states.device
            ),
            counts,
            output_size=tokens,
        )
        output, state = operator.forward_routes_training(
            hidden_states,
            token_indices,
            expert_indices,
            scores,
            out=output,
            expert_offsets=cu_seqlens,
            token_indices_identity=True,
        )
        retained_metadata = (
            state.sorted_token_ids,
            state.sorted_route_ids,
            state.sorted_weights,
            state.sorted_expert_ids,
            state.num_valid_ids,
            state.expert_frequency,
        )
        # EP8 makes the per-step route count move with expert load imbalance,
        # so the retained-state bucket is keyed on the weight shape only.
        retained_state_shape = (
            int(config.hidden_size),
            int(config.intermediate_size),
            num_experts,
        ) == (2048, 768, 16) and tokens >= 64
        if retained_state_shape and any(
            tensor is None for tensor in retained_metadata
        ):
            raise RuntimeError(
                "E16 route forward did not retain all six sorter tensors"
            )
        ctx.config = config
        ctx.native_weight_layout = bool(native_weight_layout)
        # Keep the invocation-owned preactivation, ready event and all six
        # sorter tensors alive until autograd consumes them.
        ctx.forward_state = state if retained_state_shape else None
        ctx.save_for_backward(
            hidden_states,
            scores,
            logical_w1,
            logical_w2,
            token_indices,
            expert_indices,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        from kernels.moe.sonic_backward import sonic_moe_backward_routes

        (
            hidden_states,
            scores,
            logical_w1,
            logical_w2,
            token_indices,
            expert_indices,
        ) = ctx.saved_tensors
        dx, dw1, dw2, dscores = sonic_moe_backward_routes(
            hidden_states,
            logical_w1,
            logical_w2,
            token_indices,
            expert_indices,
            scores,
            grad_output.contiguous(),
            ctx.config,
            forward_state=ctx.forward_state,
            token_indices_sorted=True,
        )
        dw1_out = dw1 if ctx.native_weight_layout else dw1.transpose(1, 2)
        dw2_out = dw2 if ctx.native_weight_layout else dw2.transpose(1, 2)
        return (
            dx,
            dscores,
            dw1_out,
            dw2_out,
            None,
            None,
        )


def flydsl_pre_routed(
    hidden_states: torch.Tensor,
    tokens_per_expert,
    scores: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    native_weight_layout: bool = False,
) -> torch.Tensor:
    """Pre-routed FlyDSL forward with selected FlyDSL or AITER backward."""
    hidden_states = hidden_states.contiguous()
    if hidden_states.shape[0] == 0:
        zero = (scores.sum() + w1.sum() + w2.sum()).to(hidden_states.dtype) * 0
        return hidden_states + zero
    cu_seqlens = _cu_seqlens_from_counts(
        tokens_per_expert, int(w1.shape[0]), hidden_states.device
    )
    native_routes_training = (
        os.environ.get("SONIC_MOE_GEMM_BACKEND", "triton") == "flydsl"
        and os.environ.get("SONIC_MOE_FLYDSL_NATIVE", "1") != "0"
        and hidden_states.dtype == torch.bfloat16
        and w1.dtype == torch.bfloat16
        and w2.dtype == torch.bfloat16
        and (
            (
                native_weight_layout
                and int(w1.shape[1]) == 2 * int(w2.shape[2])
                and int(w1.shape[2]) == int(w2.shape[1])
            )
            or (
                not native_weight_layout
                and int(w1.shape[2]) == 2 * int(w2.shape[1])
                and int(w1.shape[1]) == int(w2.shape[2])
            )
        )
    )
    if native_routes_training:
        return _FlyDSLNativeRoutes.apply(
            hidden_states,
            scores.reshape(-1),
            w1,
            w2,
            cu_seqlens,
            native_weight_layout,
        )
    if native_weight_layout:
        w1 = w1.transpose(1, 2).contiguous()
        w2 = w2.transpose(1, 2).contiguous()
    return _FlyDSLPreRouted.apply(
        hidden_states, scores.reshape(-1), w1, w2, cu_seqlens
    )


@functools.lru_cache(maxsize=64)
def _compile_grouped_nn(
    contraction_size: int,
    output_size: int,
    num_experts: int,
    block_m: int,
    block_n: int,
    block_k: int,
    compact_grid: bool,
    device_index: int,
):
    from kernels.moe.sonic_grouped_a16w16 import compile_sonic_grouped_a16w16_nn

    n_waves = 4 if block_n >= 256 else 2
    return compile_sonic_grouped_a16w16_nn(
        contraction_size=contraction_size,
        output_size=output_size,
        num_experts=num_experts,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=2,
        n_waves=n_waves,
        sorted_block_m=block_m,
        compact_grid=compact_grid,
        # ce7b755 requires route-slot stores for the legacy M-reuse schedule.
        # The production E16 retained-state path uses its native exact queue.
        expert_m_reuse=False,
        device_index=device_index,
    )


def _dgrad_tiles(contraction_size: int, output_size: int, num_experts: int) -> tuple[int, int, int]:
    override = _TILE_OVERRIDE
    if override is not None:
        if (
            override.dx is not None
            and (contraction_size, output_size) == (_QWEN3_I_FULL, _QWEN3_H)
        ):
            return override.dx
        if (
            override.da is not None
            and (contraction_size, output_size) == (_QWEN3_H, _QWEN3_I)
        ):
            return override.da
    if num_experts == _QWEN3_LOCAL_E and (contraction_size, output_size) == (
        _QWEN3_I_FULL,
        _QWEN3_H,
    ):
        return _QWEN3_NN_DX_TILE
    if num_experts == _QWEN3_LOCAL_E and (contraction_size, output_size) == (
        _QWEN3_H,
        _QWEN3_I,
    ):
        return _QWEN3_NN_DA_TILE
    if contraction_size % _DX_BLOCK_K or output_size % _DX_BLOCK_N:
        raise ValueError(
            "FlyDSL dgrad requires K divisible by "
            f"{_DX_BLOCK_K} and N divisible by {_DX_BLOCK_N}, got K={contraction_size} "
            f"N={output_size}"
        )
    if output_size % _DA_BLOCK_N == 0 and contraction_size % _DA_BLOCK_K == 0:
        if output_size != contraction_size and output_size <= contraction_size:
            return _DA_BLOCK_M, _DA_BLOCK_N, _DA_BLOCK_K
    return _DX_BLOCK_M, _DX_BLOCK_N, _DX_BLOCK_K


def _wgrad_tiles(output_m: int, output_n: int, num_experts: int):
    override = _TILE_OVERRIDE
    if override is not None:
        if override.dw1 is not None and (output_m, output_n) == (
            _QWEN3_H,
            _QWEN3_I_FULL,
        ):
            return override.dw1
        if override.dw2 is not None and (output_m, output_n) == (
            _QWEN3_I,
            _QWEN3_H,
        ):
            return override.dw2
    if num_experts == _QWEN3_LOCAL_E and (output_m, output_n) == (
        _QWEN3_H,
        _QWEN3_I_FULL,
    ):
        return _QWEN3_TN_DW1_TILE
    if num_experts == _QWEN3_LOCAL_E and (output_m, output_n) == (
        _QWEN3_I,
        _QWEN3_H,
    ):
        return _QWEN3_TN_DW2_TILE
    from kernels.moe.sonic_grouped_tn import grouped_tn_tuning

    return grouped_tn_tuning(output_m, output_n)


def _compact_group_state(cu_seqlens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One compact expert queue: ``[count, (expert, first_row) * count]``."""
    cu = cu_seqlens.to(dtype=torch.int32).contiguous()
    num_experts = int(cu.numel() - 1)
    frequency = torch.empty(num_experts, dtype=torch.int32, device=cu.device)
    queue = torch.empty(1 + 2 * num_experts, dtype=torch.int32, device=cu.device)
    if cu.is_cuda and num_experts <= 32:
        _compact_queue_kernel()[(1,)](
            cu, frequency, queue, NUM_EXPERTS=num_experts
        )
        return frequency, queue
    frequency.copy_(cu[1:] - cu[:-1])
    active = torch.nonzero(frequency, as_tuple=False).flatten().to(torch.int32)
    count = int(active.numel())
    queue.zero_()
    queue[0] = count
    if count:
        queue[1 : 1 + 2 * count : 2] = active
        queue[2 : 2 + 2 * count : 2] = cu[:-1].index_select(0, active)
    return frequency, queue


def _weight_as_kn(weight_nk: torch.Tensor) -> torch.Tensor:
    """Transpose ``[E, N, K]`` weights to ``[E, K, N]`` once per live tensor."""
    key = id(weight_nk)
    cached = _WEIGHT_KN_CACHE.get(key)
    if (
        cached is not None
        and cached[0] is weight_nk
        and cached[1] == int(weight_nk._version)
    ):
        return cached[2]
    kn = _as_bf16(weight_nk.transpose(1, 2).contiguous())
    _WEIGHT_KN_CACHE[key] = (weight_nk, int(weight_nk._version), kn)
    if len(_WEIGHT_KN_CACHE) > 8:
        _WEIGHT_KN_CACHE.pop(next(iter(_WEIGHT_KN_CACHE)))
    return kn


def _as_bf16(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.bfloat16:
        return tensor
    return tensor.to(dtype=torch.bfloat16)


def _pad_expert_rows(
    tensor: torch.Tensor,
    layout: PreRoutedFlyDSLLayout,
) -> torch.Tensor:
    padded = tensor.new_zeros(layout.padded_capacity, tensor.shape[1])
    padded.index_copy_(0, layout.dest_index.to(torch.int64), tensor)
    return padded


def _flydsl_grouped_dgrad_padded(
    activations: torch.Tensor,
    weight_nk: torch.Tensor,
    out: torch.Tensor,
    layout: PreRoutedFlyDSLLayout,
) -> torch.Tensor:
    """Run current metadata-direct NN ABI and gather compact expert rows."""
    from kernels.common.tensor_shim import _run_compiled

    activations = _as_bf16(activations.contiguous())
    weight = _weight_as_kn(weight_nk)
    contraction_size = int(weight.shape[1])
    output_size = int(weight.shape[2])
    num_experts = int(weight.shape[0])
    _block_m, block_n, block_k = _dgrad_tiles(
        contraction_size, output_size, num_experts
    )
    # The metadata-direct schedule emits one descriptor per sorter block.
    # Match its M quantum so every descriptor covers exactly one compute tile.
    block_m = layout.sort_unit
    padded_in = _pad_expert_rows(activations, layout)
    padded_out = out.new_zeros(layout.padded_capacity, output_size)
    grid = max(
        1,
        (layout.padded_capacity // block_m) * (output_size // block_n),
    )
    launcher = _compile_grouped_nn(
        contraction_size,
        output_size,
        num_experts,
        block_m,
        block_n,
        block_k,
        False,
        activations.device.index or 0,
    )
    stream = torch.cuda.current_stream(activations.device)
    _run_compiled(
        launcher,
        padded_in.data_ptr(),
        weight.data_ptr(),
        layout.num_valid_ids.data_ptr(),  # unused non-compact schedule
        layout.sorted_expert_ids.data_ptr(),
        layout.num_valid_ids.data_ptr(),
        padded_out.data_ptr(),
        grid,
        stream,
    )
    out.copy_(padded_out.index_select(0, layout.dest_index.to(torch.int64)))
    return out


def _flydsl_grouped_wgrad_padded(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
    layout: PreRoutedFlyDSLLayout,
) -> torch.Tensor:
    """Run the current TN queue ABI over sorter-padded expert segments."""
    lhs_padded = _pad_expert_rows(_as_bf16(lhs.contiguous()), layout)
    rhs_padded = _pad_expert_rows(_as_bf16(rhs.contiguous()), layout)
    return flydsl_grouped_wgrad(
        lhs_padded,
        rhs_padded,
        torch.empty(0, dtype=torch.int32, device=lhs.device),
        out,
        frequency=layout.frequency,
        queue=layout.active_queue,
    )


def flydsl_grouped_dgrad(
    activations: torch.Tensor,
    weight_nk: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    *,
    frequency: torch.Tensor | None = None,
    queue: torch.Tensor | None = None,
) -> torch.Tensor:
    """``out = activations @ weight`` per expert, ``weight`` stored as ``[E, N, K]``."""
    if activations.shape[0] == 0:
        return out.zero_()
    del frequency, queue
    weight = _weight_as_kn(weight_nk)
    block_m, _, _ = _dgrad_tiles(
        int(weight.shape[1]), int(weight.shape[2]), int(weight.shape[0])
    )
    layout = build_pre_routed_metadata(
        cu_seqlens.to(device=activations.device),
        num_tokens=int(activations.shape[0]),
        sort_unit=block_m,
    )
    work_out = (
        out
        if out.dtype == torch.bfloat16
        else torch.empty_like(out, dtype=torch.bfloat16)
    )
    _flydsl_grouped_dgrad_padded(activations, weight_nk, work_out, layout)
    if work_out is not out:
        out.copy_(work_out)
    return out


def flydsl_grouped_wgrad(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    *,
    frequency: torch.Tensor | None = None,
    queue: torch.Tensor | None = None,
) -> torch.Tensor:
    """``out[e] = lhs_e^T @ rhs_e``, matching AITER ``[E, K, N]`` wgrad layout."""
    from kernels.moe.sonic_grouped_tn import grouped_tn_from_queue_flydsl

    if lhs.shape[0] == 0:
        return out.zero_()
    if frequency is None or queue is None:
        frequency, queue = _compact_group_state(cu_seqlens.to(device=lhs.device))
    work_out = out if out.dtype == torch.bfloat16 else torch.empty_like(out, dtype=torch.bfloat16)
    work_out.zero_()
    tiles = _wgrad_tiles(int(lhs.shape[1]), int(rhs.shape[1]), int(work_out.shape[0]))
    grouped_tn_from_queue_flydsl(
        _as_bf16(lhs.contiguous()),
        _as_bf16(rhs.contiguous()),
        frequency,
        queue,
        work_out,
        block_m=tiles[0],
        block_n=tiles[1],
        block_k=tiles[2],
        k_padding=tiles[3],
        m_waves=tiles[4],
        n_waves=tiles[5],
    )
    if work_out is not out:
        out.copy_(work_out)
    return out


def flydsl_grouped_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    A_is_transposed: bool = False,
    B_is_transposed: bool = False,
) -> torch.Tensor:
    if A_is_transposed and B_is_transposed:
        raise ValueError("FlyDSL grouped GEMM does not support AT and BT together")
    if A_is_transposed:
        if out is None:
            out = torch.empty(
                (cu_seqlens.numel() - 1, A.shape[1], B.shape[1]),
                dtype=A.dtype,
                device=A.device,
            )
        return flydsl_grouped_wgrad(A, B, cu_seqlens, out)
    if B_is_transposed:
        if out is None:
            out = torch.empty(
                (A.shape[0], B.shape[1]),
                dtype=A.dtype,
                device=A.device,
            )
        return flydsl_grouped_dgrad(A, B, cu_seqlens, out)
    raise ValueError("FlyDSL grouped GEMM only implements dgrad and wgrad")


def _should_use_flydsl(
    A_is_transposed: bool,
    B_is_transposed: bool,
    A_scale,
    B_scale,
    bias,
    A_idx,
    scatter_idx,
    A: torch.Tensor,
) -> bool:
    if os.environ.get("SONIC_MOE_GEMM_BACKEND", "triton") != "flydsl":
        return False
    if A_scale is not None or B_scale is not None:
        return False
    if bias is not None or A_idx is not None or scatter_idx is not None:
        return False
    return bool(A_is_transposed) != bool(B_is_transposed)


def install_flydsl_grouped_gemm_backend() -> None:
    """Patch AITER Sonic ``grouped_gemm`` so backward GEMMs use FlyDSL."""
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        import kernels.moe.sonic_grouped_a16w16  # noqa: F401
        import kernels.moe.sonic_grouped_tn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SONIC_MOE_GEMM_BACKEND=flydsl requires FlyDSL on PYTHONPATH "
            "(kernels.moe). Mount FLYDSL_ROOT and add it to PYTHONPATH."
        ) from exc

    import aiter.ops.triton._triton_kernels.moe.sonicmoe as sonicmoe
    import aiter.ops.triton._triton_kernels.moe.sonicmoe.backward as sonic_bwd
    import aiter.ops.triton._triton_kernels.moe.sonicmoe.grouped_gemm_triton as ggt

    original = ggt.grouped_gemm

    def _grouped_gemm(
        A,
        B,
        cu_seqlens,
        out=None,
        bias=None,
        A_idx=None,
        scatter_idx=None,
        A_is_transposed=False,
        B_is_transposed=False,
        A_scale=None,
        B_scale=None,
        block_size=128,
        out_dtype=None,
    ):
        if _should_use_flydsl(
            A_is_transposed,
            B_is_transposed,
            A_scale,
            B_scale,
            bias,
            A_idx,
            scatter_idx,
            A,
        ):
            return flydsl_grouped_gemm(
                A,
                B,
                cu_seqlens,
                out,
                A_is_transposed=A_is_transposed,
                B_is_transposed=B_is_transposed,
            )
        return original(
            A,
            B,
            cu_seqlens,
            out=out,
            bias=bias,
            A_idx=A_idx,
            scatter_idx=scatter_idx,
            A_is_transposed=A_is_transposed,
            B_is_transposed=B_is_transposed,
            A_scale=A_scale,
            B_scale=B_scale,
            block_size=block_size,
            out_dtype=out_dtype,
        )

    ggt.grouped_gemm = _grouped_gemm
    sonicmoe.grouped_gemm = _grouped_gemm
    sonic_bwd.grouped_gemm = _grouped_gemm
    _INSTALLED = True
