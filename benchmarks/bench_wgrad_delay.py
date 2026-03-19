###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Benchmark 5 — Wgrad delay improves overlap.

Exercises Lumen's ``_DeferredWgrad`` feature from
``lumen.modules.parallel_linear``:

  * **_DeferredWgrad API**: ``defer(weight, compute_fn)`` stores wgrad
    closure; ``execute()`` runs it later — overlapping with next-layer
    forward or communication.
  * **Pipeline overlap**: Two-layer pipeline where layer-2 dW overlaps
    with layer-1 dX on a secondary stream.
  * **Integration test**: Uses Lumen's ``_DeferredWgrad`` class directly.
  * **gradient_accumulation_fusion**: Tests the ``main_grad.add_`` pattern
    from ``quantized_linear`` backward.

Run::

    python -m benchmarks.bench_wgrad_delay
    pytest benchmarks/bench_wgrad_delay.py -v -s
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from benchmarks.bench_utils import (
    BenchResult,
    cuda_timer,
    print_report,
    require_cuda,
)
from benchmarks.conftest import CUDA

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
M = 4096  # tokens (B * S)
K = 4096  # hidden_dim
N = 11008  # FFN intermediate


# ---------------------------------------------------------------------------
# 1. Lumen _DeferredWgrad API
# ---------------------------------------------------------------------------


@CUDA
class TestDeferredWgradAPI:
    """Test Lumen's _DeferredWgrad class for wgrad deferral."""

    # Expected: Deferred wgrad (defer + execute) should have nearly identical
    # total latency to eager wgrad. The purpose of deferral is NOT to speed up
    # a single wgrad, but to allow scheduling it on a secondary stream later,
    # overlapping with other work. The defer() call itself is near-zero cost
    # (just stores a closure); execute() triggers the same GEMM.
    def test_defer_and_execute_latency(self):
        """Measure defer + execute vs eager wgrad."""
        from lumen.modules.parallel_linear import _DeferredWgrad

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
        grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

        # Eager: compute dW immediately
        def _eager():
            dw = grad_out.T @ x
            w.grad = dw

        r_eager = cuda_timer(_eager, label="eager wgrad (dW = gO^T @ X)")

        # Deferred: store closure, execute later
        dwg = _DeferredWgrad()

        def _deferred():
            dwg.defer(w, lambda: grad_out.T @ x)
            dwg.execute()

        r_deferred = cuda_timer(_deferred, label="deferred wgrad (defer + execute)")

        print_report("_DeferredWgrad: Eager vs Deferred", [r_eager, r_deferred])

    # Expected: Near-zero latency (~0.001 ms). defer() only captures a Python
    # closure (weight reference + compute lambda) without launching any GPU
    # work. This confirms the deferral mechanism adds negligible overhead to
    # the backward pass critical path.
    def test_defer_only_latency(self):
        """Measure just the defer call (no execute) — should be near-zero."""
        from lumen.modules.parallel_linear import _DeferredWgrad

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
        grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        dwg = _DeferredWgrad()

        r = cuda_timer(
            lambda: dwg.defer(w, lambda: grad_out.T @ x),
            label="defer only (closure capture)",
        )
        print_report("_DeferredWgrad: Defer Only", [r])
        print(f"  Defer overhead: {r.avg_ms:.4f} ms (should be ~0)")


# ---------------------------------------------------------------------------
# 2. Wgrad overlap with next-layer forward
# ---------------------------------------------------------------------------


@CUDA
class TestWgradOverlapWithForward:
    """Simulate deferring wgrad to overlap with the next layer's forward pass."""

    # Expected: Overlapped should be ~1.3-1.8x faster than sequential. In
    # sequential mode, dW1 and fwd2 execute serially (total = dW1 + fwd2).
    # In overlapped mode, dW1 runs on wgrad_stream concurrently with fwd2 on
    # the compute stream, so total ≈ max(dW1, fwd2). The speedup depends on
    # how well the GPU can parallelize two GEMMs on separate streams (limited
    # by SM occupancy and memory bandwidth contention).
    def test_sequential_vs_overlapped(self):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
        w2 = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.02
        grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

        wgrad_stream = torch.cuda.Stream()

        # Sequential: layer-1 dW, then layer-2 forward
        def _sequential():
            _ = grad_out.T @ x  # layer-1 dW
            torch.cuda.synchronize()
            _ = (x @ w1.T) @ w2.T  # layer-2 forward
            torch.cuda.synchronize()

        r_seq = cuda_timer(_sequential, label="sequential (dW1 then fwd2)")

        # Overlapped: layer-1 dW on wgrad_stream, layer-2 forward on compute
        def _overlapped():
            with torch.cuda.stream(wgrad_stream):
                _ = grad_out.T @ x  # layer-1 dW (deferred)
            _ = (x @ w1.T) @ w2.T  # layer-2 forward (concurrent)
            wgrad_stream.synchronize()

        r_ovl = cuda_timer(_overlapped, label="overlapped (dW1 || fwd2)")

        speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_ovl.extra["speedup"] = round(speedup, 2)
        print_report("Wgrad Overlap with Next-Layer Forward", [r_seq, r_ovl])


# ---------------------------------------------------------------------------
# 3. Two-layer backward pipeline
# ---------------------------------------------------------------------------


@CUDA
class TestWgradTwoLayerPipeline:
    """Two-layer pipeline: dX computed eagerly, dW deferred to overlap."""

    # Expected: Deferred pipeline should show speedup because layer-2's dW
    # overlaps with layer-1's dX on separate streams. In a real transformer,
    # dW is the largest GEMM in backward (M*N*K FLOPs) and dominates the
    # backward time. Deferring it to overlap with the next layer's dX
    # effectively hides the dW latency, reducing total backward time by
    # up to the cost of one dW GEMM per layer.
    def test_pipeline_eager_vs_deferred(self):
        w1 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
        w2 = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.02
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        wgrad_stream = torch.cuda.Stream()

        def _eager():
            # Forward
            o1 = x @ w1.T  # layer 1 fwd
            o2 = o1 @ w2.T  # layer 2 fwd
            g2 = torch.ones_like(o2)
            # Backward: all in sequence
            dx2 = g2 @ w2  # layer 2 dX
            _ = g2.T @ o1  # layer 2 dW
            _ = dx2 @ w1  # layer 1 dX
            _ = dx2.T @ x  # layer 1 dW
            torch.cuda.synchronize()

        def _deferred():
            # Forward
            o1 = x @ w1.T
            o2 = o1 @ w2.T
            g2 = torch.ones_like(o2)
            # Layer 2 backward: dX eager, dW deferred
            dx2 = g2 @ w2
            with torch.cuda.stream(wgrad_stream):
                _ = g2.T @ o1  # deferred — overlaps with layer-1 dX
            # Layer 1 backward: dX on compute stream (concurrent with layer-2 dW)
            _ = dx2 @ w1
            wgrad_stream.synchronize()
            # Layer 1 dW
            with torch.cuda.stream(wgrad_stream):
                _ = dx2.T @ x
            wgrad_stream.synchronize()

        r_eager = cuda_timer(_eager, label="2-layer eager backward")
        r_deferred = cuda_timer(_deferred, label="2-layer deferred wgrad")

        speedup = r_eager.avg_ms / max(r_deferred.avg_ms, 1e-6)
        r_deferred.extra["speedup"] = round(speedup, 2)
        print_report("Two-Layer Pipeline: Eager vs Deferred Wgrad", [r_eager, r_deferred])


# ---------------------------------------------------------------------------
# 4. Gradient accumulation fusion pattern
# ---------------------------------------------------------------------------


@CUDA
class TestGradAccumulationFusion:
    """Test the main_grad.add_ pattern used by quantized_linear backward."""

    # Expected: main_grad.add_(dw) should have similar or slightly higher
    # latency than w.grad = dw because add_ performs a fused read-add-write
    # while assignment just updates a pointer. However, the real benefit of
    # gradient_accumulation_fusion is in multi-microbatch training: add_
    # accumulates gradients in-place without allocating new tensors per
    # microbatch, saving memory allocation overhead and enabling pipelining.
    def test_add_vs_assign(self):
        """Compare w.grad = dw (assign) vs w.main_grad.add_(dw) (accumulate)."""
        w = nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
        w.main_grad = torch.zeros_like(w.data)
        grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        def _assign():
            dw = grad_out.T @ x
            w.grad = dw

        r_assign = cuda_timer(_assign, label="w.grad = dw (assign)")

        def _accumulate():
            dw = grad_out.T @ x
            w.main_grad.add_(dw)

        r_accum = cuda_timer(_accumulate, label="w.main_grad.add_(dw) (fused accum)")

        print_report("Gradient Accumulation: Assign vs Fused Add", [r_assign, r_accum])


# ---------------------------------------------------------------------------
# 5. Wgrad overlap with communication
# ---------------------------------------------------------------------------


@CUDA
class TestWgradOverlapWithComm:
    """Simulate deferring wgrad to overlap with TP reduce-scatter."""

    # Expected: Overlapped should be faster because the deferred dW GEMM runs
    # on wgrad_stream concurrently with the reduce-scatter (simulated as
    # buffer copy) on the compute stream. In real TP training, the reduce-scatter
    # uses the network fabric (SDMA/NCCL) while the dW GEMM uses compute SMs,
    # so they can run truly in parallel with minimal resource contention.
    def test_wgrad_comm_overlap(self):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        # Simulated comm buffer (reduce-scatter mock)
        comm_src = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        comm_dst = torch.empty(M // 2, K, device="cuda", dtype=torch.bfloat16)

        wgrad_stream = torch.cuda.Stream()

        # Sequential: dX → dW → reduce-scatter
        def _sequential():
            _ = grad_out @ w
            _ = grad_out.T @ x
            torch.cuda.synchronize()
            comm_dst.copy_(comm_src[: M // 2])
            torch.cuda.synchronize()

        r_seq = cuda_timer(_sequential, label="sequential (dX → dW → RS)")

        # Overlapped: dX → defer dW (overlaps with reduce-scatter on wgrad_stream)
        def _overlapped():
            _ = grad_out @ w
            with torch.cuda.stream(wgrad_stream):
                _ = grad_out.T @ x  # deferred wgrad
            comm_dst.copy_(comm_src[: M // 2])  # reduce-scatter on compute
            wgrad_stream.synchronize()

        r_ovl = cuda_timer(_overlapped, label="overlapped (dW || RS)")

        speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_ovl.extra["speedup"] = round(speedup, 2)
        print_report("Wgrad Overlap with Communication", [r_seq, r_ovl])


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def main():
    require_cuda()

    from lumen.modules.parallel_linear import _DeferredWgrad

    results: List[BenchResult] = []

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
    grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    # Eager
    r_eager = cuda_timer(lambda: grad_out.T @ x, label="eager wgrad")
    results.append(r_eager)

    # Deferred via Lumen API
    dwg = _DeferredWgrad()

    def _deferred():
        dwg.defer(w, lambda: grad_out.T @ x)
        dwg.execute()

    r_def = cuda_timer(_deferred, label="Lumen _DeferredWgrad")
    results.append(r_def)

    # Overlap simulation
    wgrad_stream = torch.cuda.Stream()
    w2 = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    def _overlapped():
        with torch.cuda.stream(wgrad_stream):
            _ = grad_out.T @ x
        _ = x @ w2.T  # next-layer forward
        wgrad_stream.synchronize()

    r_ovl = cuda_timer(_overlapped, label="deferred wgrad || next fwd")
    results.append(r_ovl)

    print_report("Lumen Wgrad Delay Overlap", results)


if __name__ == "__main__":
    main()
