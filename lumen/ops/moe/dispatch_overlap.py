###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Communication helpers for overlapping MoE token dispatch metadata.

Variable-size all-to-all requires host split lists.  The count exchange can,
however, run asynchronously while the caller sorts tokens and builds the data
payload.  This mirrors Megatron's deferred synchronization without coupling the
operation to a specific model implementation.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist


@dataclass
class AsyncCountExchange:
    """Pending non-differentiable MoE count exchange."""

    send_counts: torch.Tensor
    recv_counts: torch.Tensor
    work: Optional[Any] = None

    def wait_for_splits(self) -> tuple[list[int], list[int]]:
        """Wait for communication and return host split-size lists."""
        if self.work is not None:
            self.work.wait()
        return self.send_counts.tolist(), self.recv_counts.tolist()


def begin_count_exchange(
    send_counts: torch.Tensor,
    group: Optional[dist.ProcessGroup],
) -> AsyncCountExchange:
    """Start an asynchronous all-to-all exchange of per-rank token counts.

    The returned object owns both tensors until :meth:`wait_for_splits` is
    called, so the caller can safely perform token permutation and payload
    construction while the collective is in flight.
    """
    if group is None or not dist.is_initialized() or dist.get_world_size(group) == 1:
        return AsyncCountExchange(send_counts, send_counts)

    recv_counts = torch.empty_like(send_counts)
    work = dist.all_to_all_single(
        recv_counts,
        send_counts,
        group=group,
        async_op=True,
    )
    return AsyncCountExchange(send_counts, recv_counts, work)


_SIDE_STREAM: Optional[torch.cuda.Stream] = None
_PINNED: dict[tuple, torch.Tensor] = {}
_DEVICE_A2A_LOGGED = False


def device_a2a_enabled() -> bool:
    """Match Megatron: on unless ``LUMEN_MOE_DEVICE_A2A=0``."""
    import os

    return os.environ.get("LUMEN_MOE_DEVICE_A2A", "1") != "0"


def _side_stream() -> torch.cuda.Stream:
    global _SIDE_STREAM
    if _SIDE_STREAM is None:
        _SIDE_STREAM = torch.cuda.Stream()
    return _SIDE_STREAM


def _pinned(slot: str, flat: torch.Tensor) -> torch.Tensor:
    device_index = flat.device.index if flat.device.index is not None else -1
    key = (slot, device_index, flat.numel(), flat.dtype)
    cached = _PINNED.get(key)
    if cached is None:
        cached = torch.empty(
            flat.numel(),
            dtype=flat.dtype,
            device="cpu",
            pin_memory=True,
        )
        _PINNED[key] = cached
    return cached


def host_split_pair(
    send_counts: torch.Tensor,
    recv_counts: torch.Tensor,
) -> tuple[list[int], list[int]]:
    """Return host split lists without draining the compute stream.

    RCCL still reads split sizes on the host. ``Tensor.tolist()`` synchronizes
    the current stream first, so a token gather queued there cannot overlap
    the copy. Sum and D2H run on a side stream that does not wait for that
    gather. A 1-D tensor is copied as-is; a 2-D tensor is summed over its
    last dimension.
    """
    if not device_a2a_enabled() or not send_counts.is_cuda:
        send_flat = send_counts if send_counts.dim() == 1 else send_counts.sum(dim=-1)
        recv_flat = recv_counts if recv_counts.dim() == 1 else recv_counts.sum(dim=-1)
        return send_flat.tolist(), recv_flat.tolist()

    global _DEVICE_A2A_LOGGED
    if not _DEVICE_A2A_LOGGED:
        _DEVICE_A2A_LOGGED = True
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(
                "FSDP MoE device all-to-all: split sizes D2H on a side stream",
                flush=True,
            )

    stream = _side_stream()
    send_counts.record_stream(stream)
    recv_counts.record_stream(stream)
    with torch.cuda.stream(stream):
        send_flat = send_counts if send_counts.dim() == 1 else send_counts.sum(dim=-1)
        recv_flat = recv_counts if recv_counts.dim() == 1 else recv_counts.sum(dim=-1)
        send_flat = send_flat.contiguous().reshape(-1)
        recv_flat = recv_flat.contiguous().reshape(-1)
        send_host = _pinned("send", send_flat)
        recv_host = _pinned("recv", recv_flat)
        send_host.copy_(send_flat, non_blocking=True)
        recv_host.copy_(recv_flat, non_blocking=True)
        send_flat.record_stream(stream)
        recv_flat.record_stream(stream)
    event = stream.record_event()
    if not event.query():
        event.synchronize()
    return (
        [int(value) for value in send_host.tolist()],
        [int(value) for value in recv_host.tolist()],
    )
