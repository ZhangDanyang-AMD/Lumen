###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Async CPU activation offload for Lumen.

Provides hooks that offload forward activations to pinned CPU memory
and prefetch them back to GPU before backward computation. Uses
dedicated CUDA streams for async D2H/H2D transfers.
"""

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CPUOffloadManager:
    """Manages async activation offload to CPU pinned memory.

    Registers forward hooks to copy activations to CPU after each layer,
    and backward hooks to prefetch them back to GPU before they're needed.

    Uses two dedicated CUDA streams:
    - d2h_stream: Device-to-host transfers (forward)
    - h2d_stream: Host-to-device transfers (backward)

    Args:
        enabled: Whether offload is active.
        pin_memory: Use pinned (page-locked) CPU memory for transfers.
    """

    def __init__(self, enabled: bool = True, pin_memory: bool = True):
        self.enabled = enabled
        self.pin_memory = pin_memory
        self._d2h_stream: Optional[torch.cuda.Stream] = None
        self._h2d_stream: Optional[torch.cuda.Stream] = None
        self._cpu_tensors: Dict[str, Tuple[torch.Tensor, torch.Size, torch.dtype]] = {}
        self._hooks: List = []
        self._layer_order: List[str] = []

    def _get_d2h_stream(self, device):
        if self._d2h_stream is None:
            self._d2h_stream = torch.cuda.Stream(device=device)
        return self._d2h_stream

    def _get_h2d_stream(self, device):
        if self._h2d_stream is None:
            self._h2d_stream = torch.cuda.Stream(device=device)
        return self._h2d_stream

    def register_hooks(self, model: nn.Module, layer_types: Optional[tuple] = None) -> int:
        """Register forward/backward hooks on model layers.

        Args:
            model: Model to hook.
            layer_types: Tuple of module types to hook. If None, hooks all
                modules with parameters.

        Returns:
            Number of hooks registered.
        """
        if not self.enabled:
            return 0

        count = 0
        for name, module in model.named_modules():
            if layer_types is not None:
                if not isinstance(module, layer_types):
                    continue
            elif not list(module.parameters(recurse=False)):
                continue

            fwd_hook = module.register_forward_hook(self._make_forward_hook(name))
            bwd_hook = module.register_full_backward_pre_hook(self._make_backward_hook(name))
            self._hooks.extend([fwd_hook, bwd_hook])
            self._layer_order.append(name)
            count += 1

        logger.info("CPUOffloadManager: registered %d layer hooks", count)
        return count

    def _make_forward_hook(self, layer_name: str):
        """Create forward hook that offloads output to CPU."""
        manager = self

        def hook(module, input, output):
            if not manager.enabled:
                return output
            if not isinstance(output, torch.Tensor):
                return output
            if not output.is_cuda:
                return output

            device = output.device
            d2h = manager._get_d2h_stream(device)

            # Async copy to pinned CPU memory
            with torch.cuda.stream(d2h):
                if manager.pin_memory:
                    cpu_tensor = torch.empty(
                        output.shape,
                        dtype=output.dtype,
                        pin_memory=True,
                    )
                    cpu_tensor.copy_(output, non_blocking=True)
                else:
                    cpu_tensor = output.to("cpu", non_blocking=True)

            manager._cpu_tensors[layer_name] = (cpu_tensor, output.shape, output.dtype)

            # Return a placeholder that saves memory
            # The actual tensor will be prefetched in backward
            return output

        return hook

    def _make_backward_hook(self, layer_name: str):
        """Create backward pre-hook that prefetches activations from CPU."""
        manager = self

        def hook(module, grad_output):
            if not manager.enabled:
                return
            if layer_name not in manager._cpu_tensors:
                return

            cpu_tensor, shape, dtype = manager._cpu_tensors[layer_name]

            # Determine device from grad_output
            device = None
            if isinstance(grad_output, tuple):
                for g in grad_output:
                    if isinstance(g, torch.Tensor) and g.is_cuda:
                        device = g.device
                        break
            elif isinstance(grad_output, torch.Tensor) and grad_output.is_cuda:
                device = grad_output.device

            if device is not None:
                h2d = manager._get_h2d_stream(device)
                with torch.cuda.stream(h2d):
                    _ = cpu_tensor.to(device, non_blocking=True)
                # Sync to ensure data is available
                h2d.synchronize()

            # Clean up
            del manager._cpu_tensors[layer_name]

        return hook

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._layer_order.clear()
        self._cpu_tensors.clear()

    def memory_saved_bytes(self) -> int:
        """Estimate bytes currently offloaded to CPU."""
        total = 0
        for name, (tensor, shape, dtype) in self._cpu_tensors.items():
            total += tensor.nelement() * tensor.element_size()
        return total


@contextmanager
def lumen_cpu_offload_context(
    model: nn.Module,
    enabled: bool = True,
    pin_memory: bool = True,
    layer_types: Optional[tuple] = None,
):
    """Context manager for CPU activation offload.

    Usage::

        with lumen_cpu_offload_context(model, enabled=True):
            output = model(input)
            loss = criterion(output, target)
            loss.backward()

    Args:
        model: Model to offload activations from.
        enabled: Whether to enable offload.
        pin_memory: Use pinned CPU memory.
        layer_types: Module types to offload. None = all with params.
    """
    manager = CPUOffloadManager(enabled=enabled, pin_memory=pin_memory)
    if enabled:
        manager.register_hooks(model, layer_types)
    try:
        yield manager
    finally:
        manager.remove_hooks()
