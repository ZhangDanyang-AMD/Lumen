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

from functools import reduce
from operator import mul
from typing import Union

import torch
import torch.distributed as dist
import triton
from aiter.ops.triton._triton_kernels.cross_entropy import (
    cross_entropy_kernel,
    online_softmax_kernel,
)
from aiter.ops.triton.cross_entropy import (
    MAX_FUSED_SIZE,
    NUM_WARPS,
    cross_entropy_backward,
)
from aiter.ops.triton.cross_entropy import cross_entropy_forward as _aiter_ce_forward

__all__ = ["parallel_cross_entropy"]


def _cross_entropy_forward_sdma(
    _input: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
    reduce_loss: bool,
    dist_group: Union[dist.ProcessGroup, None],
    ignore_idx: int,
):
    """SDMA-accelerated cross-entropy forward.

    Identical to the AITER ``cross_entropy_forward`` except the TP
    all-gather uses mori SDMA via :class:`SdmaTpComm`.
    """
    from lumen.modules.sdma_comm import SdmaTpComm

    B, SQ, V = _input.shape
    n_rows = B * SQ
    assert reduce(mul, list(target.size())) == n_rows
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    loss_1d = torch.empty(n_rows, dtype=torch.float32, device=_input.device)
    m_d_Xy = torch.empty(n_rows * 3, dtype=torch.float32, device=_input.device)

    if _input.stride(-1) != 1:
        _input = _input.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    rank = 0 if dist_group is None else dist.get_rank(dist_group)

    online_softmax_kernel[(n_rows,)](
        _input,
        _input.stride(-2),
        target,
        target.stride(-1),
        m_d_Xy,
        m_d_Xy.stride(-1),
        rank,
        V,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )

    world_size = 1 if dist_group is None else dist.get_world_size(dist_group)
    if world_size > 1:
        comm = SdmaTpComm.get(dist_group)
        ag_result = comm._ag(m_d_Xy)  # (npes, n_rows * 3)
        gathered = ag_result.reshape(-1)  # (world_size * n_rows * 3,)
    else:
        gathered = m_d_Xy

    cross_entropy_kernel[(n_rows,)](
        _input,
        _input.stride(-2),
        target,
        target.stride(-1),
        loss_1d,
        loss_1d.stride(-1),
        gathered,
        gathered.stride(-1),
        rank,
        world_size,
        ignore_idx,
        V,
        n_rows,
        reduce_loss=reduce_loss,
        label_smoothing=label_smoothing,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )

    loss = loss_1d.reshape(B, SQ) if not reduce_loss else (loss_1d.sum() / n_rows)
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
    ):
        # AITER Triton kernels only check stride(-1) == 1 before using
        # stride(-2) as the row stride.  A transposed-but-last-dim-contiguous
        # tensor (e.g. from .transpose(0,1)) can have a huge stride(-2) that
        # causes out-of-bounds GPU memory reads.  Ensure full contiguity here.
        if not _input.is_contiguous():
            _input = _input.contiguous()

        if use_sdma and dist_process_group is not None:
            loss, grad_input = _cross_entropy_forward_sdma(
                _input,
                target,
                label_smoothing,
                reduce_loss,
                dist_process_group,
                ignore_idx,
            )
        else:
            loss, grad_input = _aiter_ce_forward(
                _input,
                target,
                label_smoothing,
                reduce_loss,
                dist_process_group,
                ignore_idx,
            )
        ctx.save_for_backward(grad_input.detach())
        ctx.is_cg_capturable = is_cg_capturable
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (grad_input,) = ctx.saved_tensors
        grad_input = cross_entropy_backward(grad_input, grad_output, ctx.is_cg_capturable)
        return grad_input, None, None, None, None, None, None, None


parallel_cross_entropy = _ParallelCrossEntropy.apply
