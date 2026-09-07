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
    - **Triton MOE GEMM**: ``moe_gemm_a8w8`` (per-tensor delayed/dynamic)
    - **Sonic grouped GEMM**: blockwise FP8 with in-kernel 1×128 / 128×128 scales
    - **Triton MOE per-token**: ``moe_gemm_per_token`` (fused per-token scale)
    - **Triton MOE MXFP8**: ``moe_gemm_mxfp8`` (fused MXFP8 microscaling)
    - **CKTile DeepGEMM**: ``deepgemm`` (BF16/FP16/FP8 grouped flat MM)

Quantization modes:
    - ``none``       — BF16 GMM/ptgmm via AITER Triton
    - ``delayed``    — per-tensor FP8 MOE GEMM via AITER Triton
    - ``dynamic``    — per-tensor FP8 MOE GEMM via AITER Triton
    - ``per_token``  — per-token FP8 fused MOE GEMM via AITER Triton
    - ``blockwise`` / ``blockwise2d`` — official Qwen/DeepSeek fine-grained
      FP8: 128×128 weight tiles + dynamic 1×128 activations. Sonic grouped
      GEMM applies the same recipe on forward, dgrad, and wgrad.
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


def _default_fp8_dtype() -> torch.dtype:
    from lumen.quantize.config import _get_float8_e4m3

    return _get_float8_e4m3()


def _build_routing_data_from_group_sizes(group_sizes, total_tokens):
    """Build aiter RoutingData from Lumen's simple group_sizes tensor.

    Histogram, prefix sums, and the block-pid map stay on the GPU. ``block_m``
    and ``max_n_tiles`` are derived from Python shapes (``T``, ``E``), so this
    does not D2H ``group_sizes``.

    ``token_offs_pad[-1]`` is forced to ``max_n_tiles`` so the kernel's
    padding math matches the launch grid (``padding_m == 0``).
    """
    import triton
    from aiter.ops.triton.moe.moe_routing.routing import ExptData, RoutingData

    hist = group_sizes.to(dtype=torch.int32)
    if not hist.is_contiguous():
        hist = hist.contiguous()
    device = hist.device
    n_expts_tot = hist.shape[0]
    tokens_per_expert = max(1, total_tokens // n_expts_tot)
    block_m = max(16, min(triton.next_power_of_2(tokens_per_expert), 128))
    if total_tokens <= n_expts_tot:
        max_n_tiles = total_tokens
    else:
        max_n_tiles = n_expts_tot - 1 - ((n_expts_tot - total_tokens - 1) // block_m)

    n_tiles = torch.div(hist + (block_m - 1), block_m, rounding_mode="trunc")
    token_offs_raw = torch.empty(n_expts_tot + 1, dtype=torch.int32, device=device)
    token_offs_raw[0] = 0
    token_offs_raw[1:] = torch.cumsum(hist, dim=0)
    tile_offs = torch.empty(n_expts_tot + 1, dtype=torch.int32, device=device)
    tile_offs[0] = 0
    tile_offs[1:] = torch.cumsum(n_tiles, dim=0)
    token_offs_pad = tile_offs.clone()
    token_offs_pad[-1] = max_n_tiles

    block_pid_map = torch.full((max_n_tiles,), -1, dtype=torch.int32, device=device)
    max_tiles_one = max(1, (total_tokens + block_m - 1) // block_m)
    expert_ids = torch.arange(n_expts_tot, device=device, dtype=torch.int32).unsqueeze(1)
    block_ids = torch.arange(max_tiles_one, device=device, dtype=torch.int32).unsqueeze(0)
    valid = block_ids < n_tiles.unsqueeze(1)
    dest = tile_offs[:-1].unsqueeze(1) + block_ids
    packed = (block_ids << 16) + expert_ids
    if max_n_tiles > 0:
        in_range = valid & (dest < max_n_tiles)
        block_pid_map[dest[in_range].long()] = packed[in_range]

    expt_data = ExptData(
        hist=hist,
        token_offs_raw=token_offs_raw,
        token_offs_pad=token_offs_pad,
        block_pid_map=block_pid_map,
    )
    return RoutingData(
        block_m=block_m,
        gate_scal=None,
        expt_hist=hist,
        n_expts_tot=n_expts_tot,
        n_expts_act=1,
        expt_data=expt_data,
    )


# id(nn.Parameter) -> (version, fp8_dtype, block_size, shape, w_fp8, w_scales)
_weight_fp8_cache = {}


def _quant_blockwise_2d(w, fp8_dtype, block_size=128):
    """Quantize a 3D weight tensor [E, K, N] with 2D block scales.

    Uses a batched GPU kernel and caches the result until the BF16 master
    tensor's ``_version`` changes (optimizer in-place updates).

    Only ``nn.Parameter`` storage is cached. FSDP2 all-gather views are
    ephemeral tensors whose ``data_ptr`` is recycled; caching those leaks
    GPU memory and can serve stale scales.
    """
    from lumen.ops.quantize.ops import quant_fp8_blockwise_weight_3d

    if not isinstance(w, torch.nn.Parameter):
        return quant_fp8_blockwise_weight_3d(w, fp8_dtype, block_size)

    cache_key = id(w)
    version = int(w._version)
    shape = tuple(w.shape)
    cached = _weight_fp8_cache.get(cache_key)
    if (
        cached is not None
        and cached[0] == version
        and cached[1] == fp8_dtype
        and cached[2] == block_size
        and cached[3] == shape
    ):
        return cached[4], cached[5]

    w_fp8, w_scales = quant_fp8_blockwise_weight_3d(w, fp8_dtype, block_size)
    _weight_fp8_cache[cache_key] = (
        version,
        fp8_dtype,
        block_size,
        shape,
        w_fp8,
        w_scales,
    )
    return w_fp8, w_scales


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
    fp8_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Grouped GEMM dispatch with multi-backend fallback (all AITER).

    For BF16 (scaling_type="none"):
        out[g] = lhs[group_start:group_end] @ rhs[g].T + bias[g]

    For ``blockwise`` / ``blockwise2d``:
        Quantizes ``lhs`` as 1×128 and ``rhs`` as 128×128, then runs Sonic
        grouped GEMM. Sonic weights are ``[E, K, N]`` (``Y = X @ W``).

    Args:
        lhs: Activation tensor ``[total_tokens, K]``.
        rhs: Weight tensor. Sonic blockwise path uses ``[num_experts, K, N]``;
             other backends may use ``[num_experts, N, K]``.
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
    if fp8_dtype is None:
        fp8_dtype = _default_fp8_dtype()
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

    if scaling_type in ("blockwise", "blockwise2d"):
        backends = []

        def _sonic_blockscale():
            lhs_2d = lhs.reshape(-1, lhs.shape[-1]).contiguous()
            if lhs.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
                if x_scale is None or w_scale is None:
                    raise ValueError("Pre-quantized FP8 inputs require explicit x_scale and w_scale")
                lhs_fp8, lhs_scales = lhs_2d, x_scale
                rhs_fp8, rhs_scales = rhs, w_scale
            else:
                from lumen.ops.quantize.ops import quant_fp8_blockwise_impl

                lhs_fp8, lhs_scales = quant_fp8_blockwise_impl(
                    lhs_2d, fp8_dtype, axis=1, block_size=block_size
                )
                rhs_fp8, rhs_scales = _quant_blockwise_2d(rhs, fp8_dtype, block_size)

            from aiter.ops.triton._triton_kernels.moe.sonicmoe.grouped_gemm_triton import (
                grouped_gemm as sonic_grouped_gemm,
            )

            return sonic_grouped_gemm(
                lhs_fp8,
                rhs_fp8,
                _cu_seqlens_from_group_sizes(group_sizes),
                A_scale=lhs_scales,
                B_scale=rhs_scales,
                block_size=block_size,
                out_dtype=torch.bfloat16,
            )

        backends.append((Backend.TRITON, _sonic_blockscale))

        def _sequential_fallback():
            return _grouped_gemm_fp8_sequential(
                lhs,
                rhs,
                group_sizes,
                x_scale,
                w_scale,
                scaling_type,
                bias,
                block_size=block_size,
                fp8_dtype=fp8_dtype,
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


def _grouped_gemm_fp8_sequential(
    lhs,
    rhs,
    group_sizes,
    x_scale,
    w_scale,
    scaling_type,
    bias=None,
    block_size=128,
    fp8_dtype=None,
):
    """Sequential per-expert FP8 GEMM via AITER Triton GEMM backends."""
    if fp8_dtype is None:
        fp8_dtype = _default_fp8_dtype()
    from lumen.ops.quantize.linear import dispatch_gemm, quantize_input

    outputs = []
    offset = 0
    for g, size in enumerate(group_sizes):
        size = int(size)
        if size == 0:
            continue
        x_g = lhs[offset : offset + size]
        w_kn = rhs[g]
        if w_kn.shape[0] != x_g.shape[1]:
            raise ValueError(
                "grouped FP8 sequential expects rhs[e] shaped [K, N] matching "
                f"lhs K={x_g.shape[1]}, got {tuple(w_kn.shape)}"
            )
        w_nk = w_kn.transpose(0, 1).contiguous()
        xs = x_scale[g] if x_scale is not None and x_scale.dim() > 0 else x_scale
        ws = w_scale[g] if w_scale is not None and w_scale.dim() > 0 else w_scale
        b_g = bias[g] if bias is not None else None

        if xs is None and scaling_type != "none":
            desc = quantize_input(x_g, scaling_type, fp8_dtype, block_size)
            x_g, xs = desc.data, desc.scale
        if ws is None and scaling_type != "none":
            desc = quantize_input(w_nk, scaling_type, fp8_dtype, block_size)
            w_nk, ws = desc.data, desc.scale

        out_g = dispatch_gemm(x_g, w_nk, xs, ws, scaling_type, b_g)
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

    BF16 only (AITER ptgmm). Blockwise FP8 wgrad lives on
    ``grouped_quantized_linear`` / ``grouped_fp8_expert_mlp``.

    Args:
        grad_output: ``[total_tokens, N]``.
        input_tensor: ``[total_tokens, K]``.
        group_sizes: Expert token counts ``[num_experts]``.
        scaling_type: Must be ``"none"``.

    Returns:
        Weight gradient ``[num_experts, N, K]``.
    """
    if scaling_type != "none":
        raise ValueError(
            "grouped_gemm_wgrad is BF16-only; use grouped_quantized_linear or "
            "grouped_fp8_expert_mlp for blockwise FP8 wgrad"
        )
    backends = []
    if _probe_aiter_gmm():

        def _ptgmm():
            fn = _get_ptgmm()
            # ptgmm documents TRANS_LHS for a view of .t(), but large T
            # has produced GPU memory-access faults. Materialize (N, T).
            lhs = grad_output.transpose(0, 1).contiguous()
            rhs = input_tensor.contiguous()
            return fn(lhs, rhs, group_sizes)

        backends.append((Backend.TRITON, _ptgmm))
    return try_backends(backends, op_name="grouped_gemm_wgrad")


def _cu_seqlens_from_group_sizes(group_sizes: torch.Tensor) -> torch.Tensor:
    counts = group_sizes.to(dtype=torch.int32)
    zeros = torch.zeros(1, dtype=torch.int32, device=counts.device)
    return torch.cat([zeros, torch.cumsum(counts, dim=0).to(torch.int32)])


def _sonic_grouped_linear_backward(
    grad_output,
    inp,
    weight,
    group_sizes,
    cu_seqlens=None,
    scaling_type="none",
    fp8_dtype=None,
    block_size=128,
):
    """Dgrad/wgrad via SonicMoE grouped GEMM.

    ``blockwise`` / ``blockwise2d`` match the Qwen3-*-FP8 / DeepSeek recipe:
    dgrad is 1×128 activations against a transposed 128×128 weight;
    wgrad quantizes each expert's token segment independently (no scale
    crossover). Other scaling types stay BF16 grouped GEMM.
    """
    from aiter.ops.triton._triton_kernels.moe.sonicmoe.grouped_gemm_triton import (
        grouped_gemm as sonic_grouped_gemm,
    )

    if cu_seqlens is None:
        cu_seqlens = _cu_seqlens_from_group_sizes(group_sizes)
    if scaling_type in ("blockwise", "blockwise2d"):
        if fp8_dtype is None:
            fp8_dtype = _default_fp8_dtype()
        from lumen.ops.quantize.ops import (
            quant_fp8_blockwise_impl,
            quant_fp8_blockwise_segment_m_impl,
        )

        grad_row, grad_row_scale = quant_fp8_blockwise_impl(
            grad_output, fp8_dtype, axis=1, block_size=block_size
        )
        weight_fp8, weight_scale = _quant_blockwise_2d(
            weight, fp8_dtype, block_size
        )
        grad_input = sonic_grouped_gemm(
            grad_row,
            weight_fp8,
            cu_seqlens,
            B_is_transposed=True,
            A_scale=grad_row_scale,
            B_scale=weight_scale,
            block_size=block_size,
            out_dtype=torch.bfloat16,
        )

        scale_counts = torch.div(
            group_sizes.to(torch.int32) + block_size - 1,
            block_size,
            rounding_mode="trunc",
        )
        scale_cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=group_sizes.device),
                torch.cumsum(scale_counts, dim=0).to(torch.int32),
            ]
        )
        inp_col, inp_col_scale = quant_fp8_blockwise_segment_m_impl(
            inp,
            len(group_sizes),
            group_sizes,
            cu_seqlens,
            scale_cu_seqlens,
            fp8_dtype,
            block_size,
        )
        grad_col, grad_col_scale = quant_fp8_blockwise_segment_m_impl(
            grad_output,
            len(group_sizes),
            group_sizes,
            cu_seqlens,
            scale_cu_seqlens,
            fp8_dtype,
            block_size,
        )
        grad_weight = sonic_grouped_gemm(
            inp_col,
            grad_col,
            cu_seqlens,
            A_is_transposed=True,
            A_scale=inp_col_scale,
            B_scale=grad_col_scale,
            block_size=block_size,
            out_dtype=torch.bfloat16,
        )
        return grad_input, grad_weight

    grad_input = sonic_grouped_gemm(
        grad_output,
        weight,
        cu_seqlens,
        B_is_transposed=True,
    )
    grad_weight = sonic_grouped_gemm(
        inp,
        grad_output,
        cu_seqlens,
        A_is_transposed=True,
    )
    return grad_input, grad_weight


class _GroupedQuantizedLinear(torch.autograd.Function):
    """Pre-routed grouped GEMM with BF16 master weights and FP8 compute.

    Weight layout matches SonicMoE / AITER grouped GEMM: ``weight[e]`` is
    ``[K, N]`` and ``Y_e = X_e @ weight[e]``.

    ``blockwise`` / ``blockwise2d`` use FP8 grouped GEMM on forward, dgrad,
    and wgrad. Failures propagate; there is no BF16 backward fallback.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        weight: torch.Tensor,
        group_sizes: torch.Tensor,
        scaling_type: str,
        fp8_dtype: torch.dtype,
        block_size: int,
    ) -> torch.Tensor:
        group_sizes = group_sizes.to(device=inp.device, dtype=torch.int32)
        out = grouped_gemm(
            inp,
            weight,
            group_sizes,
            scaling_type=scaling_type,
            block_size=block_size,
            fp8_dtype=fp8_dtype,
        )
        ctx.save_for_backward(inp, weight, group_sizes)
        ctx.scaling_type = scaling_type
        ctx.fp8_dtype = fp8_dtype
        ctx.block_size = block_size
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inp, weight, group_sizes = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        inp = inp.contiguous()
        weight = weight.contiguous()
        grad_input, grad_weight = _sonic_grouped_linear_backward(
            grad_output,
            inp,
            weight,
            group_sizes,
            scaling_type=ctx.scaling_type,
            fp8_dtype=ctx.fp8_dtype,
            block_size=ctx.block_size,
        )
        return grad_input, grad_weight, None, None, None, None


def grouped_quantized_linear(
    inp: torch.Tensor,
    weight: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    scaling_type: str = "blockwise2d",
    fp8_dtype: Optional[torch.dtype] = None,
    block_size: int = 128,
) -> torch.Tensor:
    """Autograd grouped Linear for expert-sorted tokens.

    Default ``blockwise2d`` is the official Qwen3 FP8 recipe on this GEMM:
    E4M3 128×128 weights, dynamic 1×128 activations, FP8 dgrad/wgrad.

    Args:
        inp: ``[total_tokens, K]``.
        weight: ``[num_experts, K, N]``.
        group_sizes: per-expert token counts ``[num_experts]``.
    """
    if fp8_dtype is None:
        fp8_dtype = _default_fp8_dtype()
    if inp.numel() == 0:
        return inp.new_empty(inp.shape[0], weight.shape[-1])
    return _GroupedQuantizedLinear.apply(
        inp,
        weight,
        group_sizes,
        scaling_type,
        fp8_dtype,
        block_size,
    )


class _GroupedFp8ExpertMlp(torch.autograd.Function):
    """SwiGLU expert MLP: FP8 grouped w1/w2 GEMMs and FP8 grouped backward."""

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        group_sizes: torch.Tensor,
        scaling_type: str,
        fp8_dtype: torch.dtype,
        block_size: int,
        concat_layout: bool,
    ):
        from aiter.ops.triton._triton_kernels.moe.sonicmoe.activation_kernels import (
            activation_fwd,
        )

        group_sizes = group_sizes.to(device=hidden.device, dtype=torch.int32)
        fc1 = grouped_gemm(
            hidden,
            w1,
            group_sizes,
            scaling_type=scaling_type,
            block_size=block_size,
            fp8_dtype=fp8_dtype,
        )
        intermediate = w1.shape[-1] // 2
        hidden_act = activation_fwd(
            fc1, intermediate, "swiglu", concat_layout=concat_layout
        )
        output = grouped_gemm(
            hidden_act,
            w2,
            group_sizes,
            scaling_type=scaling_type,
            block_size=block_size,
            fp8_dtype=fp8_dtype,
        )
        ctx.save_for_backward(hidden, w1, w2, fc1, hidden_act, group_sizes)
        ctx.intermediate = intermediate
        ctx.scaling_type = scaling_type
        ctx.fp8_dtype = fp8_dtype
        ctx.block_size = block_size
        ctx.concat_layout = concat_layout
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from aiter.ops.triton._triton_kernels.moe.sonicmoe.activation_kernels import (
            activation_bwd,
        )

        hidden, w1, w2, fc1, hidden_act, group_sizes = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        hidden = hidden.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()
        cu_seqlens = _cu_seqlens_from_group_sizes(group_sizes)
        grad_act, grad_w2 = _sonic_grouped_linear_backward(
            grad_output,
            hidden_act,
            w2,
            group_sizes,
            cu_seqlens=cu_seqlens,
            scaling_type=ctx.scaling_type,
            fp8_dtype=ctx.fp8_dtype,
            block_size=ctx.block_size,
        )
        grad_fc1 = activation_bwd(
            fc1, grad_act, ctx.intermediate, "swiglu", concat_layout=ctx.concat_layout
        )
        grad_hidden, grad_w1 = _sonic_grouped_linear_backward(
            grad_fc1,
            hidden,
            w1,
            group_sizes,
            cu_seqlens=cu_seqlens,
            scaling_type=ctx.scaling_type,
            fp8_dtype=ctx.fp8_dtype,
            block_size=ctx.block_size,
        )
        return grad_hidden, grad_w1, grad_w2, None, None, None, None, None


def grouped_fp8_expert_mlp(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    scaling_type: str = "blockwise2d",
    fp8_dtype: Optional[torch.dtype] = None,
    block_size: int = 128,
    concat_layout: bool = True,
) -> torch.Tensor:
    """Pre-routed SwiGLU expert MLP with official blockwise FP8.

    Forward: grouped FP8 GEMM on fused ``w1`` (gate+up) and ``w2``, with
    Sonic SwiGLU in between. Backward: FP8 dgrad/wgrad on both GEMMs
    (activation backward stays BF16). Master ``w1``/``w2`` remain BF16.

    ``concat_layout`` must match how ``w1`` interleaves its gate and up
    halves, exactly as for the BF16 SonicMoE entry points: ``True`` for
    ``[gate | up]`` (Megatron ``linear_fc1``), ``False`` for per-column
    ``gate, up`` pairs (the FSDP expert slice). A mismatch silently computes
    ``silu(up) * gate`` on shuffled columns instead of raising.
    """
    if fp8_dtype is None:
        fp8_dtype = _default_fp8_dtype()
    if hidden.numel() == 0:
        return hidden.new_empty(hidden.shape[0], w2.shape[-1])
    if w1.shape[-1] % 2 != 0:
        raise ValueError(
            f"SwiGLU w1 last dim must be even, got {tuple(w1.shape)}"
        )
    return _GroupedFp8ExpertMlp.apply(
        hidden,
        w1,
        w2,
        group_sizes,
        scaling_type,
        fp8_dtype,
        block_size,
        concat_layout,
    )
