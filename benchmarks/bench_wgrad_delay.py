###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Benchmark 5 — Wgrad delay improves overlap.

Tests **actual Lumen implementations** only:

  * **_DeferredWgrad API**: ``defer(weight, compute_fn)`` / ``execute()``
    from ``lumen.modules.parallel_linear``.
  * **_DeferredWgrad + stream overlap**: Executes the deferred wgrad on a
    secondary CUDA stream while the main stream does other work — this is
    how the API is *intended* to be used in a real training loop.
  * **_DeferredWgrad + real NCCL allreduce** (multi-GPU): Overlaps the
    deferred wgrad GEMM with a real NCCL allreduce on a separate stream,
    demonstrating true hardware parallelism (compute SMs vs NIC/RDMA).
  * **_DeferredWgrad + SDMA allreduce** (multi-GPU): Overlaps the deferred
    wgrad GEMM with Lumen's ``SdmaTpComm`` async allreduce.  SDMA uses
    dedicated DMA engines with zero SM contention — expected to yield
    higher overlap ratios than NCCL.
  * **gradient_accumulation_fusion**: The ``main_grad.add_(dw)`` path inside
    ``quantized_linear`` backward (``gradient_accumulation_fusion=True``).

Run single-GPU::

    python -m benchmarks.bench_wgrad_delay
    pytest benchmarks/bench_wgrad_delay.py -v -s

Run multi-GPU — NCCL::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k RealComm
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k RealComm

Run multi-GPU — SDMA (requires mori)::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k SdmaComm
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k SdmaComm
"""

from __future__ import annotations

import os
from typing import List

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from benchmarks.bench_utils import (
    BenchResult,
    cuda_timer,
    print_overlap_summary,
    print_report,
    require_cuda,
)
from benchmarks.conftest import AITER, CUDA

# ---------------------------------------------------------------------------
# Dimensions (Llama 3.1 8B)
# ---------------------------------------------------------------------------
M = 4096  # tokens (B * S)
K = 4096  # hidden_dim
N = 14336  # FFN intermediate


# ---------------------------------------------------------------------------
# 1. Lumen _DeferredWgrad API — basic overhead
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

        def _eager():
            dw = grad_out.T @ x
            w.grad = dw

        r_eager = cuda_timer(_eager, label="eager wgrad (dW = gO^T @ X)")

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

    # Expected: _DeferredWgrad.execute() accumulates into main_grad when it
    # exists (line 58-59 of parallel_linear.py). After execute(), the weight's
    # main_grad should contain the computed gradient, and _pending_wgrad should
    # be cleared. This validates the accumulation path that the distributed
    # optimizer relies on.
    def test_main_grad_accumulation(self):
        """Verify execute() writes to main_grad when present."""
        from lumen.modules.parallel_linear import _DeferredWgrad

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
        w.main_grad = torch.zeros(N, K, device="cuda", dtype=torch.bfloat16)
        grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

        dwg = _DeferredWgrad()
        dwg.defer(w, lambda: grad_out.T @ x)

        assert dwg.has_pending
        dwg.execute()
        assert not dwg.has_pending

        assert w.main_grad.abs().sum() > 0, "main_grad should be non-zero after execute()"
        print(f"\n  main_grad norm: {w.main_grad.norm().item():.4f}")


# ---------------------------------------------------------------------------
# 2. _DeferredWgrad with stream overlap (intended usage pattern)
# ---------------------------------------------------------------------------


@CUDA
class TestDeferredWgradStreamOverlap:
    """Use _DeferredWgrad.execute() on a secondary stream to overlap with
    other work on the compute stream — this is the actual usage pattern
    that produces the overlap benefit in Lumen training.
    """

    # NOTE: Two large GEMMs on the same GPU typically saturate HBM bandwidth,
    # so stream-level overlap yields little speedup (overlap_ratio ≈ 0).
    # This test establishes the baseline. See test_overlap_with_simulated_comm
    # for the realistic scenario where wgrad overlaps with communication.
    def test_overlap_vs_sequential(self):
        from lumen.modules.parallel_linear import _DeferredWgrad

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
        w.main_grad = torch.zeros(N, K, device="cuda", dtype=torch.bfloat16)
        w_next = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.02
        grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

        dwg = _DeferredWgrad()
        wgrad_stream = torch.cuda.Stream()

        # Sequential: execute deferred wgrad, then next-layer forward
        def _sequential():
            w.main_grad.zero_()
            dwg.defer(w, lambda: grad_out.T @ x)
            dwg.execute()
            torch.cuda.synchronize()
            _ = x @ w_next
            torch.cuda.synchronize()

        r_seq = cuda_timer(_sequential, label="sequential: execute() then fwd")

        # Measure individual components for overlap ratio
        r_dw = cuda_timer(lambda: grad_out.T @ x, label="dW alone")
        r_fwd = cuda_timer(lambda: x @ w_next, label="next-layer fwd alone")

        # Overlapped: execute deferred wgrad on wgrad_stream, fwd on compute
        def _overlapped():
            w.main_grad.zero_()
            dwg.defer(w, lambda: grad_out.T @ x)
            with torch.cuda.stream(wgrad_stream):
                dwg.execute()
            _ = x @ w_next  # concurrent on default stream
            wgrad_stream.synchronize()

        r_ovl = cuda_timer(_overlapped, label="overlapped: execute() || fwd")

        speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        T_parts = r_dw.avg_ms + r_fwd.avg_ms
        overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
        r_ovl.extra["speedup"] = round(speedup, 2)
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)
        print_report(
            "_DeferredWgrad Stream Overlap",
            [r_dw, r_fwd, r_seq, r_ovl],
        )
        print(f"  Overlap ratio: {overlap_ratio:.3f}")

    # Expected: overlap_ratio > 0.3. In real training, _DeferredWgrad.execute()
    # (a GEMM on compute SMs) runs concurrently with communication (allreduce /
    # reduce-scatter on SDMA/NIC hardware). Since they use different hardware
    # resources, true parallelism is achieved. We simulate the communication
    # latency with torch.cuda._sleep() which blocks the stream without using
    # compute SMs or HBM bandwidth, mimicking SDMA/NIC behaviour.
    def test_overlap_with_simulated_comm(self):
        from lumen.modules.parallel_linear import _DeferredWgrad

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
        w.main_grad = torch.zeros(N, K, device="cuda", dtype=torch.bfloat16)
        grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

        dwg = _DeferredWgrad()
        comm_stream = torch.cuda.Stream()

        # Calibrate: measure wgrad GEMM latency, then set sleep to ~same duration
        r_dw = cuda_timer(lambda: grad_out.T @ x, label="dW alone", iters=10)
        dw_ns = int(r_dw.avg_ms * 1e6)
        # _sleep() takes GPU clock cycles; approximate 1 ns ≈ 1-2 cycles at ~1.5 GHz
        sleep_cycles = max(dw_ns * 2, 100_000)

        r_comm = cuda_timer(
            lambda: torch.cuda._sleep(sleep_cycles),
            label="simulated comm (sleep)",
            iters=10,
        )

        # Sequential: wgrad then comm
        def _sequential():
            w.main_grad.zero_()
            dwg.defer(w, lambda: grad_out.T @ x)
            dwg.execute()
            torch.cuda.synchronize()
            torch.cuda._sleep(sleep_cycles)
            torch.cuda.synchronize()

        r_seq = cuda_timer(_sequential, label="sequential: wgrad then comm", iters=10)

        # Overlapped: wgrad on default stream, comm on comm_stream
        def _overlapped():
            w.main_grad.zero_()
            dwg.defer(w, lambda: grad_out.T @ x)
            with torch.cuda.stream(comm_stream):
                torch.cuda._sleep(sleep_cycles)
            dwg.execute()
            torch.cuda.current_stream().wait_stream(comm_stream)

        r_ovl = cuda_timer(_overlapped, label="overlapped: wgrad || comm", iters=10)

        T_parts = r_dw.avg_ms + r_comm.avg_ms
        overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
        speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_ovl.extra["speedup"] = round(speedup, 2)
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)
        print_report(
            "_DeferredWgrad + Simulated Comm Overlap",
            [r_dw, r_comm, r_seq, r_ovl],
        )
        print_overlap_summary(
            t_compute=r_dw.avg_ms,
            t_comm=r_comm.avg_ms,
            t_seq=r_seq.avg_ms,
            t_ovl=r_ovl.avg_ms,
            compute_label="wgrad",
            comm_label="sim comm",
        )

    # Expected: Running execute() on a secondary stream for each of N layers
    # in a pipeline should show cumulative speedup. Each layer's deferred dW
    # overlaps with the next layer's dX, hiding N-1 dW computations from
    # the critical path. The total time approaches T(all_dX) + T(one_dW)
    # instead of T(all_dX) + T(all_dW).
    def test_multi_layer_pipeline(self):
        from lumen.modules.parallel_linear import _DeferredWgrad

        n_layers = 4
        weights = [nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02) for _ in range(n_layers)]
        for w in weights:
            w.main_grad = torch.zeros_like(w.data)

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        grad_outs = [torch.randn(M, N, device="cuda", dtype=torch.bfloat16) for _ in range(n_layers)]

        dwg = _DeferredWgrad()
        wgrad_stream = torch.cuda.Stream()

        def _eager_pipeline():
            for i in range(n_layers):
                weights[i].main_grad.zero_()
                _ = grad_outs[i] @ weights[i]  # dX
                weights[i].main_grad.add_(grad_outs[i].T @ x)  # dW

        def _deferred_pipeline():
            for i in range(n_layers):
                weights[i].main_grad.zero_()
                if dwg.has_pending:
                    with torch.cuda.stream(wgrad_stream):
                        dwg.execute()
                _ = grad_outs[i] @ weights[i]  # dX on compute stream
                dwg.defer(weights[i], lambda i=i: grad_outs[i].T @ x)
            with torch.cuda.stream(wgrad_stream):
                dwg.execute()
            wgrad_stream.synchronize()

        r_eager = cuda_timer(_eager_pipeline, label=f"{n_layers}-layer eager")
        r_deferred = cuda_timer(_deferred_pipeline, label=f"{n_layers}-layer deferred")

        speedup = r_eager.avg_ms / max(r_deferred.avg_ms, 1e-6)
        r_deferred.extra["speedup"] = round(speedup, 2)
        print_report(
            f"{n_layers}-Layer Pipeline: Eager vs _DeferredWgrad",
            [r_eager, r_deferred],
        )


# ---------------------------------------------------------------------------
# 3. Multi-GPU: _DeferredWgrad + real NCCL allreduce overlap
# ---------------------------------------------------------------------------


def _init_dist():
    if dist.is_initialized():
        return
    if "RANK" not in os.environ:
        return
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl:gloo", device_id=torch.device(f"cuda:{local_rank}"))


@pytest.fixture(scope="session", autouse=True)
def _cleanup_dist():
    """Ensure NCCL process group is destroyed on exit to avoid SIGSEGV."""
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


_DIST = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Multi-GPU — run with torchrun --nproc_per_node=N",
)


@_DIST
class TestDeferredWgradRealComm:
    """Overlap _DeferredWgrad.execute() with real NCCL allreduce.

    This is the closest benchmark to the actual training loop: after each
    layer's backward, the weight gradient is deferred and then executed on
    the wgrad stream while the main stream launches an allreduce for the
    *previous* layer's gradient.  NCCL allreduce uses NIC/RDMA hardware
    while the wgrad GEMM uses compute SMs — true hardware parallelism.

    Run::

        torchrun --nproc_per_node=2 benchmarks/bench_wgrad_delay.py
        torchrun --nproc_per_node=8 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k RealComm
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    # Expected: The wgrad GEMM (grad_out^T @ x, on compute SMs) runs
    # concurrently with NCCL allreduce (on NIC/RDMA hardware).
    # Total time ≈ max(T_wgrad, T_allreduce) instead of their sum.
    # When allreduce >> wgrad (common with small TP-local matrices),
    # the overlap_ratio formula yields a low value because the hidden
    # wgrad is a small fraction of T_total. The key metric is:
    #   speedup = T_seq / T_ovl  (should be > 1.0)
    #   hidden_ms = T_seq - T_ovl  (should be close to T_wgrad)
    def test_wgrad_overlap_with_allreduce(self):
        from lumen.modules.parallel_linear import _DeferredWgrad

        x = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
        w = nn.Parameter(torch.randn(N, K, device=self.device, dtype=torch.bfloat16) * 0.02)
        w.main_grad = torch.zeros(N, K, device=self.device, dtype=torch.bfloat16)
        grad_out = torch.randn(M, N, device=self.device, dtype=torch.bfloat16)

        # Buffer for allreduce (simulates previous layer's gradient)
        ar_buf = torch.randn(N, K, device=self.device, dtype=torch.bfloat16)

        dwg = _DeferredWgrad()
        comm_stream = torch.cuda.Stream(device=self.device)

        # Warmup
        for _ in range(3):
            dist.all_reduce(ar_buf.clone())
            _ = grad_out.T @ x
        torch.cuda.synchronize()
        dist.barrier()

        # Measure components individually
        r_wgrad = cuda_timer(
            lambda: grad_out.T @ x,
            label="wgrad GEMM alone",
            iters=10,
        )

        def _allreduce():
            buf = ar_buf.clone()
            dist.all_reduce(buf)
            torch.cuda.synchronize()

        r_comm = cuda_timer(_allreduce, label="allreduce alone", iters=10)

        # Sequential: wgrad then allreduce
        def _sequential():
            w.main_grad.zero_()
            dwg.defer(w, lambda: grad_out.T @ x)
            dwg.execute()
            torch.cuda.synchronize()
            buf = ar_buf.clone()
            dist.all_reduce(buf)
            torch.cuda.synchronize()

        r_seq = cuda_timer(_sequential, label="sequential: wgrad then allreduce", iters=10)

        # Overlapped: allreduce on comm_stream, wgrad on default stream
        def _overlapped():
            w.main_grad.zero_()
            dwg.defer(w, lambda: grad_out.T @ x)
            with torch.cuda.stream(comm_stream):
                buf = ar_buf.clone()
                dist.all_reduce(buf)
            dwg.execute()
            torch.cuda.current_stream().wait_stream(comm_stream)

        r_ovl = cuda_timer(_overlapped, label="overlapped: wgrad || allreduce", iters=10)

        T_parts = r_wgrad.avg_ms + r_comm.avg_ms
        overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
        speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_ovl.extra["speedup"] = round(speedup, 2)
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)

        if self.rank == 0:
            print_report(
                f"_DeferredWgrad + Real NCCL AllReduce (world={self.world})",
                [r_wgrad, r_comm, r_seq, r_ovl],
            )
            print_overlap_summary(
                t_compute=r_wgrad.avg_ms,
                t_comm=r_comm.avg_ms,
                t_seq=r_seq.avg_ms,
                t_ovl=r_ovl.avg_ms,
                compute_label="wgrad",
                comm_label="allreduce",
            )

    # Expected: In a 4-layer pipeline, each layer's deferred wgrad overlaps
    # with the allreduce of the previous layer's gradient. Cumulative
    # savings ≈ N × T_wgrad (all wgrads hidden behind allreduces).
    def test_multi_layer_pipeline_with_allreduce(self):
        from lumen.modules.parallel_linear import _DeferredWgrad

        n_layers = 4
        weights = [
            nn.Parameter(torch.randn(N, K, device=self.device, dtype=torch.bfloat16) * 0.02) for _ in range(n_layers)
        ]
        for w_i in weights:
            w_i.main_grad = torch.zeros_like(w_i.data)

        x = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
        grad_outs = [torch.randn(M, N, device=self.device, dtype=torch.bfloat16) for _ in range(n_layers)]

        dwg = _DeferredWgrad()
        comm_stream = torch.cuda.Stream(device=self.device)

        # Warmup
        for _ in range(3):
            dist.all_reduce(weights[0].data.clone())
            _ = grad_outs[0].T @ x
        torch.cuda.synchronize()
        dist.barrier()

        # Eager: sequential wgrad + allreduce per layer
        def _eager():
            for i in range(n_layers):
                weights[i].main_grad.zero_()
                _ = grad_outs[i] @ weights[i]  # dX
                weights[i].main_grad.add_(grad_outs[i].T @ x)  # dW
                torch.cuda.synchronize()
                dist.all_reduce(weights[i].main_grad)
                torch.cuda.synchronize()

        r_eager = cuda_timer(_eager, label=f"{n_layers}-layer eager+allreduce", iters=10)

        # Deferred: overlap wgrad with allreduce of previous layer
        def _deferred():
            pending_ar = None
            for i in range(n_layers):
                weights[i].main_grad.zero_()

                if dwg.has_pending:
                    dwg.execute()

                if pending_ar is not None:
                    torch.cuda.current_stream().wait_stream(comm_stream)

                _ = grad_outs[i] @ weights[i]  # dX on compute stream
                dwg.defer(weights[i], lambda i=i: grad_outs[i].T @ x)

                if i > 0:
                    with torch.cuda.stream(comm_stream):
                        dist.all_reduce(weights[i - 1].main_grad)
                    pending_ar = i - 1

            # Final layer: execute last wgrad, allreduce last gradient
            if dwg.has_pending:
                dwg.execute()
            torch.cuda.synchronize()
            with torch.cuda.stream(comm_stream):
                dist.all_reduce(weights[-1].main_grad)
            torch.cuda.current_stream().wait_stream(comm_stream)

        r_deferred = cuda_timer(
            _deferred,
            label=f"{n_layers}-layer deferred+allreduce",
            iters=10,
        )

        speedup = r_eager.avg_ms / max(r_deferred.avg_ms, 1e-6)
        r_deferred.extra["speedup"] = round(speedup, 2)

        saved_ms = r_eager.avg_ms - r_deferred.avg_ms

        if self.rank == 0:
            print_report(
                f"{n_layers}-Layer Pipeline: Eager vs Deferred + AllReduce (world={self.world})",
                [r_eager, r_deferred],
            )
            print(f"  Speedup:  {speedup:.2f}x")
            print(f"  Saved:    {saved_ms:.3f} ms  " f"(≈ {n_layers} × {saved_ms / n_layers:.3f} ms per layer)")
            print()


# ---------------------------------------------------------------------------
# 4. Multi-GPU: _DeferredWgrad + SDMA allreduce overlap
# ---------------------------------------------------------------------------


def _sdma_available():
    try:
        import mori  # noqa: F401

        return True
    except ImportError:
        return False


_SDMA = pytest.mark.skipif(
    "RANK" not in os.environ or not _sdma_available(),
    reason="Multi-GPU + mori SDMA required",
)


@_SDMA
class TestDeferredWgradSdmaComm:
    """Overlap _DeferredWgrad.execute() with SDMA allreduce.

    SDMA uses dedicated hardware DMA engines that are completely independent
    of compute SMs, providing higher overlap ratios than NCCL which may
    contend for GPU resources.  This test validates the primary design goal
    of Lumen's comm-compute overlap: wgrad GEMM on compute SMs while
    gradient allreduce runs on SDMA hardware with zero contention.

    Run::

        torchrun --nproc_per_node=2 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k SdmaComm
        torchrun --nproc_per_node=8 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k SdmaComm
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    # Expected: SDMA DMA engines are fully independent of compute SMs —
    # the wgrad GEMM and SDMA allreduce share zero hardware resources.
    # When allreduce >> wgrad, overlap_ratio stays low (same formula
    # limitation as NCCL), but hidden_ms / T_wgrad should be higher
    # than NCCL because SDMA has no SM contention at all.
    def test_wgrad_overlap_with_sdma_allreduce(self):
        from lumen.modules.parallel_linear import _DeferredWgrad
        from lumen.modules.sdma_comm import SdmaTpComm

        x = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
        w = nn.Parameter(torch.randn(N, K, device=self.device, dtype=torch.bfloat16) * 0.02)
        w.main_grad = torch.zeros(N, K, device=self.device, dtype=torch.bfloat16)
        grad_out = torch.randn(M, N, device=self.device, dtype=torch.bfloat16)

        ar_buf = torch.randn(N, K, device=self.device, dtype=torch.bfloat16)

        comm = SdmaTpComm(dist.group.WORLD)
        sdma_stream = torch.cuda.Stream(device=self.device)
        dwg = _DeferredWgrad()

        # Warmup
        for _ in range(3):
            comm.allreduce_sum(ar_buf.clone())
            _ = grad_out.T @ x
        torch.cuda.synchronize()
        dist.barrier()

        r_wgrad = cuda_timer(lambda: grad_out.T @ x, label="wgrad GEMM alone", iters=10)

        def _sdma_ar():
            buf = ar_buf.clone()
            comm.allreduce_sum_inplace(buf)

        r_comm = cuda_timer(_sdma_ar, label="SDMA allreduce alone", iters=10)

        # Sequential: wgrad then SDMA allreduce
        def _sequential():
            w.main_grad.zero_()
            dwg.defer(w, lambda: grad_out.T @ x)
            dwg.execute()
            torch.cuda.synchronize()
            buf = ar_buf.clone()
            comm.allreduce_sum_inplace(buf)

        r_seq = cuda_timer(_sequential, label="sequential: wgrad then SDMA AR", iters=10)

        # Overlapped: SDMA allreduce async on sdma_stream, wgrad on default
        def _overlapped():
            w.main_grad.zero_()
            dwg.defer(w, lambda: grad_out.T @ x)
            buf = ar_buf.clone()
            comm.allreduce_sum_async(buf, stream=sdma_stream)
            dwg.execute()
            comm.wait_allreduce_sum(stream=sdma_stream)
            torch.cuda.current_stream().wait_stream(sdma_stream)

        r_ovl = cuda_timer(_overlapped, label="overlapped: wgrad || SDMA AR", iters=10)

        T_parts = r_wgrad.avg_ms + r_comm.avg_ms
        overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
        speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_ovl.extra["speedup"] = round(speedup, 2)
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)

        if self.rank == 0:
            print_report(
                f"_DeferredWgrad + SDMA AllReduce (world={self.world})",
                [r_wgrad, r_comm, r_seq, r_ovl],
            )
            print_overlap_summary(
                t_compute=r_wgrad.avg_ms,
                t_comm=r_comm.avg_ms,
                t_seq=r_seq.avg_ms,
                t_ovl=r_ovl.avg_ms,
                compute_label="wgrad",
                comm_label="SDMA AR",
            )

    # Expected: In a 4-layer pipeline, SDMA overlap should yield higher
    # speedup than NCCL (compare with TestDeferredWgradRealComm) because
    # the DMA engines never contend with the compute SMs running wgrad.
    def test_multi_layer_pipeline_with_sdma_allreduce(self):
        from lumen.modules.parallel_linear import _DeferredWgrad
        from lumen.modules.sdma_comm import SdmaTpComm

        n_layers = 4
        weights = [
            nn.Parameter(torch.randn(N, K, device=self.device, dtype=torch.bfloat16) * 0.02) for _ in range(n_layers)
        ]
        for w_i in weights:
            w_i.main_grad = torch.zeros_like(w_i.data)

        x = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
        grad_outs = [torch.randn(M, N, device=self.device, dtype=torch.bfloat16) for _ in range(n_layers)]

        comm = SdmaTpComm(dist.group.WORLD)
        sdma_stream = torch.cuda.Stream(device=self.device)
        dwg = _DeferredWgrad()

        # Warmup
        for _ in range(3):
            comm.allreduce_sum(weights[0].data.clone())
            _ = grad_outs[0].T @ x
        torch.cuda.synchronize()
        dist.barrier()

        # Eager: sequential wgrad + SDMA allreduce per layer
        def _eager():
            for i in range(n_layers):
                weights[i].main_grad.zero_()
                _ = grad_outs[i] @ weights[i]  # dX
                weights[i].main_grad.add_(grad_outs[i].T @ x)  # dW
                torch.cuda.synchronize()
                comm.allreduce_sum_inplace(weights[i].main_grad)

        r_eager = cuda_timer(_eager, label=f"{n_layers}-layer eager+SDMA AR", iters=10)

        # Deferred: overlap wgrad with SDMA allreduce of previous layer
        def _deferred():
            pending_ar = False
            for i in range(n_layers):
                weights[i].main_grad.zero_()

                if dwg.has_pending:
                    dwg.execute()

                if pending_ar:
                    comm.wait_allreduce_sum(stream=sdma_stream)
                    torch.cuda.current_stream().wait_stream(sdma_stream)

                _ = grad_outs[i] @ weights[i]  # dX on compute stream
                dwg.defer(weights[i], lambda i=i: grad_outs[i].T @ x)

                if i > 0:
                    comm.allreduce_sum_async(
                        weights[i - 1].main_grad,
                        stream=sdma_stream,
                    )
                    pending_ar = True

            # Final layer
            if dwg.has_pending:
                dwg.execute()
            torch.cuda.synchronize()
            comm.allreduce_sum_async(weights[-1].main_grad, stream=sdma_stream)
            comm.wait_allreduce_sum(stream=sdma_stream)
            torch.cuda.current_stream().wait_stream(sdma_stream)

        r_deferred = cuda_timer(
            _deferred,
            label=f"{n_layers}-layer deferred+SDMA AR",
            iters=10,
        )

        speedup = r_eager.avg_ms / max(r_deferred.avg_ms, 1e-6)
        r_deferred.extra["speedup"] = round(speedup, 2)

        saved_ms = r_eager.avg_ms - r_deferred.avg_ms

        if self.rank == 0:
            print_report(
                f"{n_layers}-Layer Pipeline: Eager vs Deferred + SDMA AR (world={self.world})",
                [r_eager, r_deferred],
            )
            print(f"  Speedup:  {speedup:.2f}x")
            print(f"  Saved:    {saved_ms:.3f} ms  " f"(≈ {n_layers} × {saved_ms / n_layers:.3f} ms per layer)")
            print()


# ---------------------------------------------------------------------------
# 5. gradient_accumulation_fusion via quantized_linear backward
# ---------------------------------------------------------------------------


@CUDA
@AITER
class TestGradAccumFusionQuantizedLinear:
    """Test the gradient_accumulation_fusion flag in Lumen's quantized_linear.

    When gradient_accumulation_fusion=True, the backward pass of
    quantized_linear directly does weight.main_grad.add_(grad_weight)
    instead of returning grad_weight. This avoids a separate accumulation
    kernel in multi-microbatch training.
    """

    # Expected: With gradient_accumulation_fusion=True, the backward should
    # accumulate directly into main_grad (grad_weight returned as None).
    # With =False, backward returns grad_weight as a separate tensor.
    # Latency should be similar, but the fused path saves one kernel launch
    # (the separate .add_ call) and one tensor allocation per microbatch.
    def test_fused_vs_separate_accumulation(self):
        from lumen.ops.quantize.linear import quantized_linear
        from lumen.quantize.config import _get_float8_e4m3

        fp8_dtype = _get_float8_e4m3()
        hidden = K
        ffn = 1024

        x = torch.randn(M, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = nn.Parameter(torch.randn(ffn, hidden, device="cuda", dtype=torch.bfloat16) * 0.02)
        w.main_grad = torch.zeros_like(w.data)

        def _without_fusion():
            x_in = x.detach().requires_grad_(True)
            out = quantized_linear(
                x_in,
                w,
                scaling_type="none",
                fp8_dtype=fp8_dtype,
                gradient_accumulation_fusion=False,
            )
            out.sum().backward()

        def _with_fusion():
            w.main_grad.zero_()
            x_in = x.detach().requires_grad_(True)
            out = quantized_linear(
                x_in,
                w,
                scaling_type="none",
                fp8_dtype=fp8_dtype,
                gradient_accumulation_fusion=True,
            )
            out.sum().backward()

        r_no_fuse = cuda_timer(_without_fusion, label="grad_accum_fusion=False")
        r_fuse = cuda_timer(_with_fusion, label="grad_accum_fusion=True")

        print_report(
            "quantized_linear: gradient_accumulation_fusion",
            [r_no_fuse, r_fuse],
        )

    # Expected: After backward with gradient_accumulation_fusion=True,
    # w.grad should be None (not returned) and w.main_grad should contain
    # the gradient. This verifies the fused path works correctly.
    def test_fusion_writes_to_main_grad(self):
        from lumen.ops.quantize.linear import quantized_linear
        from lumen.quantize.config import _get_float8_e4m3

        fp8_dtype = _get_float8_e4m3()
        hidden = 256
        ffn = 512

        x = torch.randn(32, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = nn.Parameter(torch.randn(ffn, hidden, device="cuda", dtype=torch.bfloat16) * 0.02)
        w.main_grad = torch.zeros(ffn, hidden, device="cuda", dtype=torch.bfloat16)

        out = quantized_linear(
            x,
            w,
            scaling_type="none",
            fp8_dtype=fp8_dtype,
            gradient_accumulation_fusion=True,
        )
        out.sum().backward()

        assert w.main_grad.abs().sum() > 0, "main_grad should be non-zero"
        print(f"\n  main_grad norm after fused backward: {w.main_grad.norm().item():.4f}")

    # Expected: With gradient_accumulation_fusion=True under FP8 scaling modes,
    # the backward should still correctly accumulate into main_grad. FP8
    # quantization in backward adds quant/dequant steps but the final
    # main_grad.add_(grad_weight) path should work identically.
    def test_fusion_with_fp8_scaling(self):
        from lumen.ops.quantize.linear import quantized_linear
        from lumen.quantize.config import _get_float8_e4m3

        fp8_dtype = _get_float8_e4m3()
        hidden = 256
        ffn = 512

        results: List[BenchResult] = []
        for scaling in ["none", "delayed", "dynamic"]:
            x = torch.randn(32, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            w = nn.Parameter(torch.randn(ffn, hidden, device="cuda", dtype=torch.bfloat16) * 0.02)
            w.main_grad = torch.zeros(ffn, hidden, device="cuda", dtype=torch.bfloat16)

            def _run(s=scaling):
                w.main_grad.zero_()
                x_in = x.detach().requires_grad_(True)
                out = quantized_linear(
                    x_in,
                    w,
                    scaling_type=s,
                    fp8_dtype=fp8_dtype,
                    gradient_accumulation_fusion=True,
                )
                out.sum().backward()

            r = cuda_timer(_run, label=f"fused backward scaling={scaling}")
            results.append(r)

        print_report("gradient_accumulation_fusion across FP8 modes", results)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def main():
    require_cuda()

    from lumen.modules.parallel_linear import _DeferredWgrad

    results: List[BenchResult] = []

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
    w.main_grad = torch.zeros(N, K, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    # 1. Eager vs deferred
    r_eager = cuda_timer(lambda: grad_out.T @ x, label="eager wgrad")
    results.append(r_eager)

    dwg = _DeferredWgrad()

    def _deferred():
        w.main_grad.zero_()
        dwg.defer(w, lambda: grad_out.T @ x)
        dwg.execute()

    r_def = cuda_timer(_deferred, label="Lumen _DeferredWgrad")
    results.append(r_def)

    # 2. Stream overlap: wgrad || simulated comm (shows true overlap)
    comm_stream = torch.cuda.Stream()
    r_dw = cuda_timer(lambda: grad_out.T @ x, label="dW alone", iters=10)
    dw_ns = int(r_dw.avg_ms * 1e6)
    sleep_cycles = max(dw_ns * 2, 100_000)

    r_comm = cuda_timer(lambda: torch.cuda._sleep(sleep_cycles), label="simulated comm", iters=10)

    def _seq_comm():
        w.main_grad.zero_()
        dwg.defer(w, lambda: grad_out.T @ x)
        dwg.execute()
        torch.cuda.synchronize()
        torch.cuda._sleep(sleep_cycles)
        torch.cuda.synchronize()

    r_seq = cuda_timer(_seq_comm, label="sequential: wgrad then comm", iters=10)

    def _ovl_comm():
        w.main_grad.zero_()
        dwg.defer(w, lambda: grad_out.T @ x)
        with torch.cuda.stream(comm_stream):
            torch.cuda._sleep(sleep_cycles)
        dwg.execute()
        torch.cuda.current_stream().wait_stream(comm_stream)

    r_ovl = cuda_timer(_ovl_comm, label="_DeferredWgrad || comm", iters=10)

    T_parts = r_dw.avg_ms + r_comm.avg_ms
    overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
    r_ovl.extra["speedup"] = round(r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6), 2)
    r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)
    results.extend([r_dw, r_comm, r_seq, r_ovl])

    print_report("Lumen Wgrad Delay Benchmarks", results)


if __name__ == "__main__":
    main()
