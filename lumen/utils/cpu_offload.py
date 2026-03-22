###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""CPU offload for autograd saved tensors in Lumen.

Uses ``torch.autograd.graph.saved_tensors_hooks`` to move tensors saved for
backward to pinned CPU memory, freeing GPU memory during forward. Dedicated
CUDA streams overlap D2H work with compute; unpack restores tensors on GPU
during backward.
"""

from contextlib import contextmanager

import torch


class CPUOffloadManager:
    """Offload autograd-saved tensors to pinned CPU memory."""

    def __init__(self, enabled=True, pin_memory=True):
        self.enabled = enabled
        self.pin_memory = pin_memory
        self._device = None
        self._d2h_stream = None
        self._h2d_stream = None
        self._offloaded_bytes = 0

    def _get_d2h_stream(self):
        if self._d2h_stream is None:
            self._d2h_stream = torch.cuda.Stream(device=self._device)
        return self._d2h_stream

    def _get_h2d_stream(self):
        if self._h2d_stream is None:
            self._h2d_stream = torch.cuda.Stream(device=self._device)
        return self._h2d_stream

    @property
    def memory_saved_bytes(self):
        return self._offloaded_bytes

    def _pack(self, tensor):
        if not tensor.is_cuda or not self.enabled:
            return tensor
        if tensor.nelement() * tensor.element_size() < 1024:
            return tensor
        self._device = tensor.device
        d2h = self._get_d2h_stream()
        event = torch.cuda.current_stream(self._device).record_event()
        with torch.cuda.stream(d2h):
            d2h.wait_event(event)
            cpu = torch.empty(
                tensor.shape,
                dtype=tensor.dtype,
                pin_memory=self.pin_memory,
            )
            cpu.copy_(tensor, non_blocking=True)
            cpu._d2h_event = d2h.record_event()
        self._offloaded_bytes += tensor.nelement() * tensor.element_size()
        return cpu

    def _unpack(self, packed):
        if not isinstance(packed, torch.Tensor) or packed.is_cuda:
            return packed
        if hasattr(packed, "_d2h_event"):
            packed._d2h_event.synchronize()
        h2d = self._get_h2d_stream()
        event = torch.cuda.current_stream(self._device).record_event()
        with torch.cuda.stream(h2d):
            h2d.wait_event(event)
            gpu = packed.to(self._device, non_blocking=True)
        h2d.synchronize()
        return gpu


@contextmanager
def lumen_cpu_offload_context(enabled=True, pin_memory=True):
    """Context manager wrapping ``saved_tensors_hooks`` for CPU offload.

    Usage::

        with lumen_cpu_offload_context(enabled=True) as mgr:
            output = model(input)
            loss = criterion(output, target)
        loss.backward()

    Args:
        enabled: When False, no hooks are installed.
        pin_memory: Allocate pinned CPU buffers for async copies.
    """
    mgr = CPUOffloadManager(enabled=enabled, pin_memory=pin_memory)
    if not enabled:
        yield mgr
        return
    with torch.autograd.graph.saved_tensors_hooks(mgr._pack, mgr._unpack):
        yield mgr
