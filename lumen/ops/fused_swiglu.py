###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused SwiGLU activation dispatch.

Dispatches to AITER's Triton fused SwiGLU kernels which compute
``silu(y1) * y2`` (forward) and the full backward in single kernel
launches each, eliminating the ~6-8 separate elementwise launches that
Megatron's ``@jit_fuser`` swiglu/swiglu_back produce.

Enable globally via ``LUMEN_FUSED_SWIGLU=1`` (installed by megatron_patches).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _probe_aiter_swiglu() -> bool:
    """Return True if AITER fused SwiGLU kernels are importable."""
    from lumen.ops.dispatch import _probe_aiter_swiglu as _probe

    return _probe()


def fused_swiglu(y: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU forward via AITER Triton kernel."""
    from aiter.ops.triton.activation import swiglu_fwd

    return swiglu_fwd(y)


def fused_swiglu_backward(grad_output: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU backward via AITER Triton kernel."""
    from aiter.ops.triton.activation import swiglu_bwd

    return swiglu_bwd(grad_output, y)


@triton.jit
def _swiglu_pair_fwd_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    stride_gate_row,
    stride_up_row,
    stride_out_row,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < cols
    gate = tl.load(
        gate_ptr + row * stride_gate_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        up_ptr + row * stride_up_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    result = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + row * stride_out_row + offsets, result, mask=mask)


@triton.jit
def _swiglu_pair_bwd_kernel(
    grad_ptr,
    gate_ptr,
    up_ptr,
    dgate_ptr,
    dup_ptr,
    stride_grad_row,
    stride_gate_row,
    stride_up_row,
    stride_dgate_row,
    stride_dup_row,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < cols
    grad = tl.load(
        grad_ptr + row * stride_grad_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        gate_ptr + row * stride_gate_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        up_ptr + row * stride_up_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    sigmoid = tl.sigmoid(gate)
    silu = gate * sigmoid
    dsilu = sigmoid * (1.0 + gate * (1.0 - sigmoid))
    tl.store(
        dgate_ptr + row * stride_dgate_row + offsets,
        grad * dsilu * up,
        mask=mask,
    )
    tl.store(
        dup_ptr + row * stride_dup_row + offsets,
        grad * silu,
        mask=mask,
    )


def _swiglu_pair_block_size(cols: int) -> int:
    return min(triton.next_power_of_2(cols), 65536)


class _FusedSwiGLUPair(torch.autograd.Function):
    """Autograd-safe SwiGLU over separate HF gate/up projection outputs."""

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if gate.shape != up.shape:
            raise ValueError(
                f"SwiGLU gate/up shapes must match, got {gate.shape} and {up.shape}"
            )
        gate_2d = gate.reshape(-1, gate.shape[-1]).contiguous()
        up_2d = up.reshape(-1, up.shape[-1]).contiguous()
        out_2d = torch.empty_like(gate_2d)
        rows, cols = gate_2d.shape
        _swiglu_pair_fwd_kernel[(rows,)](
            gate_2d,
            up_2d,
            out_2d,
            gate_2d.stride(0),
            up_2d.stride(0),
            out_2d.stride(0),
            cols,
            BLOCK_SIZE=_swiglu_pair_block_size(cols),
        )
        ctx.save_for_backward(gate_2d, up_2d)
        ctx.input_shape = gate.shape
        return out_2d.reshape(gate.shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate_2d, up_2d = ctx.saved_tensors
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        dgate = torch.empty_like(gate_2d)
        dup = torch.empty_like(up_2d)
        rows, cols = gate_2d.shape
        _swiglu_pair_bwd_kernel[(rows,)](
            grad_2d,
            gate_2d,
            up_2d,
            dgate,
            dup,
            grad_2d.stride(0),
            gate_2d.stride(0),
            up_2d.stride(0),
            dgate.stride(0),
            dup.stride(0),
            cols,
            BLOCK_SIZE=_swiglu_pair_block_size(cols),
        )
        return dgate.reshape(ctx.input_shape), dup.reshape(ctx.input_shape)


def fused_swiglu_pair(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute ``silu(gate) * up`` without concatenating HF activations."""
    if not gate.is_cuda or not up.is_cuda:
        return torch.nn.functional.silu(gate) * up
    return _FusedSwiGLUPair.apply(gate, up)
