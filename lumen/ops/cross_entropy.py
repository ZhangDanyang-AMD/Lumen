###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""Vocab-parallel cross-entropy loss — thin wrapper over AITER kernels.

The actual Triton kernel implementations live in
``aiter.ops.triton._triton_kernels.cross_entropy`` and the Python
forward/backward orchestration lives in
``aiter.ops.triton.cross_entropy``.

When ``use_sdma=True`` the TP all-gather is routed through mori SDMA
instead of ``torch.distributed.all_gather_into_tensor``.

This module re-exports a single ``parallel_cross_entropy`` autograd
function for vocab-parallel cross-entropy computation.
"""

import os
from typing import Union

import torch
import torch.distributed as dist

try:
    from aiter.ops.triton._triton_kernels.cross_entropy import (
        _ce_fused_loss_grad_kernel,
        _ce_local_softmax_stats_kernel,
    )
    from aiter.ops.triton.cross_entropy import _block_size_and_warps

    _HAVE_SDMA_CE_KERNELS = True
except ImportError:
    _ce_fused_loss_grad_kernel = None
    _ce_local_softmax_stats_kernel = None
    _block_size_and_warps = None
    _HAVE_SDMA_CE_KERNELS = False
from aiter.ops.triton.cross_entropy import cross_entropy_backward
from aiter.ops.triton.cross_entropy import cross_entropy_forward as _aiter_ce_forward

# The target-range check is a host sync. Keep it for the first few calls, then
# stay on device. ``LUMEN_CE_VALIDATE_STEPS=0`` disables it immediately.
_CE_VALIDATE_STEPS = int(os.environ.get("LUMEN_CE_VALIDATE_STEPS", "2"))
_CE_CALLS = 0
_CE_WAVES_PER_EU = 2
_CE_NUM_STAGES = 2


def _consume_ce_validation() -> bool:
    global _CE_CALLS
    _CE_CALLS += 1
    return _CE_CALLS <= _CE_VALIDATE_STEPS


def _aiter_ce_forward_chunked(*args, **kwargs):
    # Lazy import: cross_entropy_forward_chunked was added to AITER after the
    # base image was built.  Importing at module load time would fail on older
    # images where the symbol doesn't exist yet.
    from aiter.ops.triton.cross_entropy import cross_entropy_forward_chunked
    return cross_entropy_forward_chunked(*args, **kwargs)

__all__ = ["parallel_cross_entropy"]


def _validate_ce_targets(target, ignore_idx, vocab_size, enabled: bool) -> None:
    if not enabled:
        return
    valid_target = (target == ignore_idx) | (
        (target >= 0) & (target < vocab_size)
    )
    if not valid_target.all():
        raise ValueError(
            f"target out of range: expected in [0, {vocab_size}) or == "
            f"ignore_idx ({ignore_idx})"
        )


def _cross_entropy_forward_sdma(
    _input: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
    reduce_loss: bool,
    dist_group: Union[dist.ProcessGroup, None],
    ignore_idx: int,
    chunk_rows: int = 0,
    validate_targets: bool = False,
):
    """SDMA-accelerated cross-entropy forward.

    Identical to the AITER ``cross_entropy_forward`` except the TP
    all-gather uses mori SDMA via :class:`SdmaTpComm`.

    When ``chunk_rows > 0`` the row dimension is split into chunks so that
    each chunk's allgather transfers only ``chunk_rows * 3`` floats instead
    of ``n_rows * 3``, reducing peak activation memory.
    """
    from lumen.modules.sdma_comm import SdmaTpComm

    if not _input.is_contiguous():
        _input = _input.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()

    B, SQ, V = _input.shape
    n_rows = B * SQ
    if target.numel() != n_rows:
        raise ValueError(f"target.numel() ({target.numel()}) != B*SQ ({n_rows})")
    block_size, num_warps = _block_size_and_warps(_input.element_size(), V)
    rank = 0 if dist_group is None else dist.get_rank(dist_group)
    world_size = 1 if dist_group is None else dist.get_world_size(dist_group)
    if label_smoothing > 0 and world_size > 1:
        raise ValueError(
            "label_smoothing > 0 is not supported with world_size > 1"
        )
    _validate_ce_targets(target, ignore_idx, V * world_size, validate_targets)
    if reduce_loss:
        n_valid = (target != ignore_idx).sum().clamp_min(1)
    else:
        n_valid = torch.ones((), dtype=torch.int64, device=_input.device)

    if chunk_rows > 0:
        return _cross_entropy_forward_sdma_chunked(
            _input, target, label_smoothing, reduce_loss,
            dist_group, ignore_idx, chunk_rows,
            B, SQ, V, n_rows, block_size, num_warps, rank, world_size, n_valid,
        )

    loss_1d = torch.zeros(n_rows, dtype=torch.float32, device=_input.device)
    m_d_Xy = torch.empty(n_rows * 3, dtype=torch.float32, device=_input.device)
    _launch_ce_chunk(
        _input, target, loss_1d, m_d_Xy, n_valid,
        rank, world_size, ignore_idx, V, n_rows, reduce_loss, label_smoothing,
        block_size, num_warps, dist_group,
    )
    loss = loss_1d.reshape(B, SQ) if not reduce_loss else (loss_1d.sum() / n_valid)
    return loss, _input


def _launch_ce_chunk(
    chunk_x, chunk_y, chunk_loss, stats, n_valid,
    rank, world_size, ignore_idx, vocab, n_rows, reduce_loss, label_smoothing,
    block_size, num_warps, dist_group,
):
    from lumen.modules.sdma_comm import SdmaTpComm

    rows = chunk_y.shape[0]
    _ce_local_softmax_stats_kernel[(rows,)](
        chunk_x,
        chunk_x.stride(-2) if chunk_x.dim() == 3 else chunk_x.stride(0),
        chunk_y,
        chunk_y.stride(-1),
        stats,
        1,
        rank,
        vocab,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        waves_per_eu=_CE_WAVES_PER_EU,
        num_stages=_CE_NUM_STAGES,
    )
    if world_size > 1:
        gathered = SdmaTpComm.get(dist_group)._ag(stats).reshape(-1)
    else:
        gathered = stats
    _ce_fused_loss_grad_kernel[(rows,)](
        chunk_x,
        chunk_x.stride(-2) if chunk_x.dim() == 3 else chunk_x.stride(0),
        chunk_y,
        chunk_y.stride(-1),
        chunk_loss,
        chunk_loss.stride(-1) if chunk_loss.dim() == 1 else chunk_loss.stride(0),
        gathered,
        1,
        rank,
        world_size,
        ignore_idx,
        vocab,
        n_rows,
        n_valid,
        reduce_loss=reduce_loss,
        label_smoothing=label_smoothing,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        waves_per_eu=_CE_WAVES_PER_EU,
        num_stages=_CE_NUM_STAGES,
    )


def _cross_entropy_forward_sdma_chunked(
    _input: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
    reduce_loss: bool,
    dist_group: Union[dist.ProcessGroup, None],
    ignore_idx: int,
    chunk_rows: int,
    B: int,
    SQ: int,
    V: int,
    n_rows: int,
    block_size: int,
    num_warps: int,
    rank: int,
    world_size: int,
    n_valid: torch.Tensor,
):
    """SDMA chunked cross-entropy: row-wise chunks to cap peak buffer size."""
    input_2d = _input.view(n_rows, V)
    target_1d = target.view(n_rows)
    loss_1d = torch.zeros(n_rows, dtype=torch.float32, device=_input.device)
    stats = torch.empty(chunk_rows * 3, dtype=torch.float32, device=_input.device)

    row = 0
    while row < n_rows:
        rows_this = min(chunk_rows, n_rows - row)
        _launch_ce_chunk(
            input_2d[row : row + rows_this],
            target_1d[row : row + rows_this],
            loss_1d[row : row + rows_this],
            stats[: rows_this * 3],
            n_valid,
            rank,
            world_size,
            ignore_idx,
            V,
            rows_this,
            reduce_loss,
            label_smoothing,
            block_size,
            num_warps,
            dist_group,
        )
        row += rows_this

    loss = loss_1d.reshape(B, SQ) if not reduce_loss else (loss_1d.sum() / n_valid)
    return loss, _input


class _ParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        _input,
        target,
        label_smoothing=0.0,
        reduce_loss=False,
        dist_process_group=None,
        ignore_idx=-100,
        is_cg_capturable=False,
        use_sdma=False,
        chunk_rows=0,
    ):
        # AITER Triton kernels only check stride(-1) == 1 before using
        # stride(-2) as the row stride.  A transposed-but-last-dim-contiguous
        # tensor (e.g. from .transpose(0,1)) can have a huge stride(-2) that
        # causes out-of-bounds GPU memory reads.  Ensure full contiguity here.
        if not _input.is_contiguous():
            _input = _input.contiguous()
        validate_targets = _consume_ce_validation()

        if use_sdma and dist_process_group is not None and _HAVE_SDMA_CE_KERNELS:
            loss, grad_input = _cross_entropy_forward_sdma(
                _input,
                target,
                label_smoothing,
                reduce_loss,
                dist_process_group,
                ignore_idx,
                chunk_rows,
                validate_targets,
            )
        elif chunk_rows > 0:
            loss, grad_input = _aiter_ce_forward_chunked(
                _input,
                target,
                label_smoothing,
                reduce_loss,
                dist_process_group,
                ignore_idx,
                chunk_rows,
                validate_targets=validate_targets,
            )
        else:
            loss, grad_input = _aiter_ce_forward(
                _input,
                target,
                label_smoothing,
                reduce_loss,
                dist_process_group,
                ignore_idx,
                validate_targets=validate_targets,
            )
        ctx.save_for_backward(grad_input.detach())
        ctx.is_cg_capturable = is_cg_capturable
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (grad_input,) = ctx.saved_tensors
        grad_input = cross_entropy_backward(grad_input, grad_output, ctx.is_cg_capturable)
        # extra None for chunk_rows
        return grad_input, None, None, None, None, None, None, None, None


parallel_cross_entropy = _ParallelCrossEntropy.apply
