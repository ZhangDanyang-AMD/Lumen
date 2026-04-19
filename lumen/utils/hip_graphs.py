###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""HIP/CUDA graph capture utilities for Lumen.

Provides graph-capture wrappers that record a training step (or sub-step)
into a replayable HIP graph, reducing kernel launch overhead. Handles
FP8 scaling manager state updates that must remain graph-safe.

Key classes:

- ``LumenGraphedCallable`` — wraps a single callable in a forward-only graph.
- ``LumenGraphedModule``   — wraps an ``nn.Module`` via ``LumenGraphedCallable``.
- ``LumenGraphedLayer``    — per-layer forward+backward graph capture with a
  custom ``autograd.Function`` that replays separate forward and backward
  CUDA graphs.  Modelled after TE's ``Graphed`` autograd function.
- ``capture_lumen_graphs`` — captures all transformer layers of a Megatron
  GPTModel and replaces their ``forward`` with graphed wrappers.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forward-only graph capture (unchanged from original)
# ---------------------------------------------------------------------------


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

        self._static_inputs = tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)
        self._input_shapes = tuple(a.shape if isinstance(a, torch.Tensor) else None for a in args)

        s = torch.cuda.Stream(device=device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s):
            for _ in range(self._num_warmup):
                self._fn(*self._static_inputs, **kwargs)
        torch.cuda.current_stream(device).wait_stream(s)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=self._pool, stream=s):
            self._static_output = self._fn(*self._static_inputs, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        if self._graph is None:
            return self._fn(*args, **kwargs)

        for static, new in zip(self._static_inputs, args):
            if isinstance(static, torch.Tensor) and isinstance(new, torch.Tensor):
                if static.shape != new.shape:
                    logger.info("Input shape changed, re-capturing graph")
                    self._capture(args, kwargs)
                    return self._static_output
                static.copy_(new)

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


# ---------------------------------------------------------------------------
# Per-layer forward+backward graph capture (P5)
# ---------------------------------------------------------------------------


class _GraphedLayerFn(torch.autograd.Function):
    """Custom autograd Function that replays pre-captured forward and backward
    CUDA/HIP graphs instead of re-executing the layer's ops.

    Static buffers for inputs/outputs/grads are created during capture and
    reused across iterations via ``copy_``.
    """

    @staticmethod
    def forward(
        ctx,
        static_fwd_input,
        static_fwd_output,
        fwd_graph,
        static_bwd_grad_output,
        static_bwd_grad_input,
        bwd_graph,
        real_input,
    ):
        static_fwd_input.copy_(real_input)
        fwd_graph.replay()

        ctx.static_bwd_grad_output = static_bwd_grad_output
        ctx.static_bwd_grad_input = static_bwd_grad_input
        ctx.bwd_graph = bwd_graph

        return static_fwd_output.detach().clone().requires_grad_()

    @staticmethod
    def backward(ctx, grad_output):
        ctx.static_bwd_grad_output.copy_(grad_output)
        ctx.bwd_graph.replay()
        return (
            None,  # static_fwd_input
            None,  # static_fwd_output
            None,  # fwd_graph
            None,  # static_bwd_grad_output
            None,  # static_bwd_grad_input
            None,  # bwd_graph
            ctx.static_bwd_grad_input.clone(),  # real_input grad
        )


class LumenGraphedLayer:
    """Captures a transformer layer's forward and backward into separate
    CUDA/HIP graphs and provides a ``__call__`` that replays them.

    Usage::

        graphed = LumenGraphedLayer(layer, sample_hidden, sample_kwargs)
        # Replace layer forward:
        layer._original_forward = layer.forward
        layer.forward = graphed

    Args:
        layer: The ``nn.Module`` transformer layer.
        sample_input: Hidden states tensor ``(S, B, H)`` with ``requires_grad=True``.
        sample_kwargs: Dict of non-grad keyword arguments (attention_mask, etc.).
        num_warmup: Warmup iterations before capture.
    """

    def __init__(
        self,
        layer: nn.Module,
        sample_input: torch.Tensor,
        sample_kwargs: Optional[Dict[str, Any]] = None,
        num_warmup: int = 3,
    ):
        self.layer = layer
        self._enabled = True
        device = sample_input.device
        kwargs = sample_kwargs or {}

        pool = getattr(torch.cuda, "graph_pool_handle", lambda: None)() if torch.cuda.is_available() else None

        # --- Static buffers ---
        self._static_input = sample_input.clone().requires_grad_(True)
        self._static_kwargs = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}

        # --- Warmup ---
        s = torch.cuda.Stream(device=device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s):
            for _ in range(num_warmup):
                out = layer(self._static_input, **self._static_kwargs)
                if isinstance(out, tuple):
                    out = out[0]
                grad = torch.ones_like(out)
                out.backward(grad)
                for p in layer.parameters():
                    p.grad = None
                self._static_input.grad = None
        torch.cuda.current_stream(device).wait_stream(s)

        # --- Capture forward ---
        self._fwd_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._fwd_graph, pool=pool, stream=s):
            fwd_out = layer(self._static_input, **self._static_kwargs)
            if isinstance(fwd_out, tuple):
                fwd_out = fwd_out[0]
        self._static_output = fwd_out

        # --- Capture backward ---
        self._static_grad_output = torch.ones_like(fwd_out)
        self._bwd_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._bwd_graph, pool=pool, stream=s):
            self._static_output.backward(self._static_grad_output, retain_graph=True)
        self._static_grad_input = (
            self._static_input.grad.clone()
            if self._static_input.grad is not None
            else torch.zeros_like(self._static_input)
        )
        for p in layer.parameters():
            p.grad = None
        self._static_input.grad = None

    def __call__(self, hidden_states, **kwargs):
        if not self._enabled:
            return self.layer(hidden_states, **kwargs)

        for k, v in kwargs.items():
            if k in self._static_kwargs and isinstance(v, torch.Tensor):
                self._static_kwargs[k].copy_(v)

        return _GraphedLayerFn.apply(
            self._static_input,
            self._static_output,
            self._fwd_graph,
            self._static_grad_output,
            self._static_grad_input,
            self._bwd_graph,
            hidden_states,
        )

    def disable(self):
        self._enabled = False

    def enable(self):
        self._enabled = True


def capture_lumen_graphs(
    model: nn.Module,
    seq_len: int,
    micro_batch_size: int,
    hidden_size: int,
    num_warmup: int = 3,
    device: str = "cuda",
) -> int:
    """Capture HIP/CUDA graphs for each transformer layer in a Megatron model.

    Iterates ``model.decoder.layers`` and replaces each layer's ``forward``
    with a :class:`LumenGraphedLayer` wrapper that replays pre-captured
    forward and backward graphs.

    Args:
        model: Megatron GPTModel (or the unwrapped model with ``.decoder.layers``).
        seq_len: Sequence length (post TP/SP/CP division).
        micro_batch_size: Per-GPU micro-batch size.
        hidden_size: Model hidden dimension.
        num_warmup: Warmup iterations before capture.
        device: CUDA device.

    Returns:
        Number of layers captured.
    """
    if not hasattr(model, "decoder") or model.decoder is None:
        logger.warning("Model has no decoder attribute, skipping graph capture")
        return 0
    if not hasattr(model.decoder, "layers"):
        logger.warning("Model decoder has no layers attribute, skipping graph capture")
        return 0

    layers = model.decoder.layers
    if len(layers) == 0:
        return 0

    torch.cuda.synchronize()

    sample_attn_mask = torch.ones(
        (1, 1, seq_len, seq_len),
        dtype=torch.bool,
        device=device,
    ).tril()

    captured = 0
    for l_no, layer in enumerate(layers):
        sample_input = torch.randn(
            (seq_len, micro_batch_size, hidden_size),
            dtype=torch.bfloat16,
            requires_grad=True,
            device=device,
        )

        sample_kwargs = {
            "attention_mask": sample_attn_mask,
        }

        if hasattr(layer, "self_attention") and hasattr(layer.self_attention, "config"):
            layer.self_attention.config.test_mode = False

        try:
            graphed = LumenGraphedLayer(
                layer,
                sample_input,
                sample_kwargs=sample_kwargs,
                num_warmup=num_warmup,
            )
            layer._original_forward = layer.forward
            layer.forward = graphed
            captured += 1
        except Exception as e:
            logger.warning(
                f"Failed to capture graph for layer {l_no}: {e}. " "Falling back to eager execution for this layer."
            )

    logger.info(f"Captured HIP graphs for {captured}/{len(layers)} transformer layers")
    return captured
