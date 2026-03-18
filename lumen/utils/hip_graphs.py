###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""HIP/CUDA graph capture utilities for Lumen.

Provides graph-capture wrappers that record a training step (or sub-step)
into a replayable HIP graph, reducing kernel launch overhead. Handles
FP8 scaling manager state updates that must remain graph-safe.
"""

import logging
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LumenGraphedCallable:
    """Wraps a callable (forward pass) in a CUDA/HIP graph.

    1. Warm up the callable with sample inputs
    2. Record a graph capturing the computation
    3. Replay the graph on subsequent calls

    The graph is re-recorded if input shapes change.

    Args:
        callable_fn: The function to graph-capture.
        sample_args: Sample arguments for warmup and capture.
        sample_kwargs: Sample keyword arguments.
        num_warmup: Number of warmup iterations before capture.
        pool: Optional CUDA memory pool for graph allocation.
    """

    def __init__(
        self,
        callable_fn: Callable,
        sample_args: Tuple[torch.Tensor, ...],
        sample_kwargs: Optional[dict] = None,
        num_warmup: int = 3,
        pool: Optional[Any] = None,
    ):
        self._fn = callable_fn
        self._num_warmup = num_warmup
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_inputs: Optional[Tuple[torch.Tensor, ...]] = None
        self._static_output: Optional[torch.Tensor] = None
        self._input_shapes: Optional[Tuple[torch.Size, ...]] = None
        self._pool = pool

        self._capture(sample_args, sample_kwargs or {})

    def _capture(self, args: Tuple, kwargs: dict):
        """Warm up and capture the graph."""
        device = None
        for a in args:
            if isinstance(a, torch.Tensor) and a.is_cuda:
                device = a.device
                break

        if device is None:
            logger.warning("No CUDA tensors in args, skipping graph capture")
            self._graph = None
            return

        # Create static input buffers
        self._static_inputs = tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)
        self._input_shapes = tuple(a.shape if isinstance(a, torch.Tensor) else None for a in args)

        # Warmup
        s = torch.cuda.Stream(device=device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s):
            for _ in range(self._num_warmup):
                self._fn(*self._static_inputs, **kwargs)
        torch.cuda.current_stream(device).wait_stream(s)

        # Capture
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=self._pool, stream=s):
            self._static_output = self._fn(*self._static_inputs, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        if self._graph is None:
            return self._fn(*args, **kwargs)

        # Copy inputs into static buffers
        for static, new in zip(self._static_inputs, args):
            if isinstance(static, torch.Tensor) and isinstance(new, torch.Tensor):
                if static.shape != new.shape:
                    logger.info("Input shape changed, re-capturing graph")
                    self._capture(args, kwargs)
                    return self._static_output
                static.copy_(new)

        # Replay graph
        self._graph.replay()
        return self._static_output

    def reset(self):
        """Release the captured graph."""
        self._graph = None
        self._static_inputs = None
        self._static_output = None


class LumenGraphedModule(nn.Module):
    """Wrapper that graph-captures a module's forward pass.

    Args:
        module: The module to wrap.
        sample_input: Sample input for graph capture.
        num_warmup: Warmup iterations.
        enabled: Whether graph capture is active.
    """

    def __init__(
        self,
        module: nn.Module,
        sample_input: Optional[torch.Tensor] = None,
        num_warmup: int = 3,
        enabled: bool = True,
    ):
        super().__init__()
        self.module = module
        self.enabled = enabled
        self._graphed: Optional[LumenGraphedCallable] = None
        self._num_warmup = num_warmup

        if enabled and sample_input is not None:
            self._graphed = LumenGraphedCallable(module, (sample_input,), num_warmup=num_warmup)

    def forward(self, *args, **kwargs):
        if self._graphed is not None and self.enabled:
            return self._graphed(*args, **kwargs)
        return self.module(*args, **kwargs)

    def capture(self, sample_input: torch.Tensor):
        """Manually trigger graph capture with the given sample input."""
        self._graphed = LumenGraphedCallable(self.module, (sample_input,), num_warmup=self._num_warmup)

    def release_graph(self):
        """Release the captured graph."""
        if self._graphed is not None:
            self._graphed.reset()
            self._graphed = None


def lumen_make_graphed_callables(
    callables: List[Callable],
    sample_args: List[Tuple[torch.Tensor, ...]],
    num_warmup: int = 3,
) -> List[LumenGraphedCallable]:
    """Graph-capture multiple callables sharing a memory pool.

    Args:
        callables: List of functions to capture.
        sample_args: Corresponding sample arguments.
        num_warmup: Warmup iterations per callable.

    Returns:
        List of LumenGraphedCallable instances.
    """
    pool = getattr(torch.cuda, "graph_pool_handle", lambda: None)() if torch.cuda.is_available() else None
    graphed = []
    for fn, args in zip(callables, sample_args):
        gc = LumenGraphedCallable(fn, args, num_warmup=num_warmup, pool=pool)
        graphed.append(gc)
    return graphed
