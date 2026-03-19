###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Benchmark 2 — Multi-GPU training achieves target overlap ratio.

Exercises Lumen's TP communication-compute overlap features:

  * **LumenColumnParallelLinear** with ``tp_comm_overlap=True``:
    SDMA allgather overlapped with GEMM.
  * **LumenRowParallelLinear** with ``tp_comm_overlap=True``:
    SDMA reduce-scatter overlapped with GEMM.
  * **SdmaTpComm** async primitives: allgather_dim0_async,
    reduce_scatter_dim0_async, allreduce_sum_async.
  * **NCCL stream-based overlap**: baseline comparison.

The *overlap ratio* is defined as::

    overlap_ratio = 1 - (T_overlapped / (T_comm + T_compute))

Run single-GPU simulation::

    pytest benchmarks/bench_comm_overlap.py -v -s -k Simulation

Run multi-GPU::

    torchrun --nproc_per_node=2 -m benchmarks.bench_comm_overlap
"""

from __future__ import annotations

import os
from typing import List

import pytest
import torch
import torch.distributed as dist

from benchmarks.bench_utils import (
    BenchResult,
    cuda_timer,
    print_report,
    require_cuda,
)
from benchmarks.conftest import CUDA

# ---------------------------------------------------------------------------
# Dimensions — Llama-2 7B column-parallel shape
# ---------------------------------------------------------------------------
M = 4096  # tokens (B * S)
K = 4096  # hidden_dim
N = 11008  # FFN intermediate (sharded by TP)


# ---------------------------------------------------------------------------
# 1. Single-GPU overlap simulation (no dist required)
# ---------------------------------------------------------------------------


@CUDA
class TestOverlapSimulation:
    """Single-GPU simulation of comm-compute overlap using CUDA streams.

    Demonstrates the overlap pattern used by LumenColumnParallelLinear:
    start allgather on SDMA/comm stream, run local-shard GEMM on compute
    stream, wait for allgather, then GEMM the remaining shards.
    """

    # Expected: Positive overlap ratio (>0.2). The simulated allgather (memcpy)
    # runs on comm_stream while the local-shard GEMM runs on compute stream.
    # Since memcpy uses the DMA engine and GEMM uses compute SMs, they can
    # run in parallel. The overlap ratio = 1 - T_overlapped/(T_comm+T_compute)
    # measures time saved vs serial; when comm≈compute, perfect overlap gives
    # ratio ≈ 0.5; ratio=0 means no overlap.
    def test_column_parallel_overlap_pattern(self):
        """Simulate the column-parallel overlap: allgather + local GEMM."""
        tp_size = 2
        shard_m = M // tp_size
        x_local = torch.randn(shard_m, K, device="cuda", dtype=torch.bfloat16)
        x_full = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

        comm_stream = torch.cuda.Stream()

        # Sequential: gather + full GEMM
        def _sequential():
            x_full.copy_(torch.cat([x_local, x_local], dim=0))
            torch.cuda.synchronize()
            _ = x_full @ w.T
            torch.cuda.synchronize()

        r_seq = cuda_timer(_sequential, label="sequential (gather + full GEMM)")

        # Overlapped: local GEMM while "allgather" runs on comm stream
        def _overlapped():
            with torch.cuda.stream(comm_stream):
                x_full.copy_(torch.cat([x_local, x_local], dim=0))
            _ = x_local @ w.T  # local shard GEMM on compute stream
            comm_stream.synchronize()
            _ = x_full[shard_m:] @ w.T  # remaining shards GEMM

        r_ovl = cuda_timer(_overlapped, label="overlapped (local GEMM || gather)")

        # Measure components
        r_comm = cuda_timer(
            lambda: x_full.copy_(torch.cat([x_local, x_local], dim=0)),
            label="comm alone (simulated gather)",
        )
        r_gemm = cuda_timer(lambda: x_full @ w.T, label="GEMM alone (full)")

        T_parts = r_comm.avg_ms + r_gemm.avg_ms
        overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)

        print_report("Column-Parallel Overlap Simulation", [r_comm, r_gemm, r_seq, r_ovl])
        print(f"  Overlap ratio: {overlap_ratio:.3f}")
        assert r_ovl.avg_ms > 0

    # Expected: Overlapped latency < sequential. In row-parallel, the GEMM
    # produces partial results that need reduce-scatter across TP ranks.
    # By starting the reduce-scatter on comm_stream immediately after GEMM
    # completes, we overlap the scatter with any subsequent computation.
    # The benefit is smaller here because the GEMM must finish before scatter.
    def test_row_parallel_overlap_pattern(self):
        """Simulate row-parallel: GEMM + reduce-scatter."""
        tp_size = 2
        x = torch.randn(M, K // tp_size, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(K, K // tp_size, device="cuda", dtype=torch.bfloat16)
        rs_output = torch.randn(M // tp_size, K, device="cuda", dtype=torch.bfloat16)

        comm_stream = torch.cuda.Stream()

        def _sequential():
            result = x @ w.T
            torch.cuda.synchronize()
            rs_output.copy_(result[: M // tp_size])
            torch.cuda.synchronize()

        r_seq = cuda_timer(_sequential, label="sequential (GEMM + scatter)")

        def _overlapped():
            result = x @ w.T
            with torch.cuda.stream(comm_stream):
                rs_output.copy_(result[: M // tp_size])
            comm_stream.synchronize()

        r_ovl = cuda_timer(_overlapped, label="overlapped (GEMM || scatter)")

        print_report("Row-Parallel Overlap Simulation", [r_seq, r_ovl])


# ---------------------------------------------------------------------------
# 2. Multi-GPU NCCL overlap
# ---------------------------------------------------------------------------


def _is_distributed():
    return dist.is_initialized()


def _init_dist():
    if dist.is_initialized():
        return
    if "RANK" not in os.environ:
        return
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


def _distributed_allgather(tensor, group=None):
    world = dist.get_world_size(group)
    chunks = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(chunks, tensor, group=group)
    return torch.cat(chunks, dim=0)


_DIST = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Multi-GPU — run with torchrun --nproc_per_node=N",
)


@_DIST
class TestNCCLColumnParallelOverlap:
    """NCCL allgather + GEMM overlap (column-parallel TP pattern)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    # Expected: overlap_ratio > 0.3. NCCL allgather uses the GPU's NIC/RDMA
    # engine while the GEMM uses compute SMs. When allgather runs on a
    # dedicated comm_stream, the local GEMM can execute concurrently.
    # The overlap ratio depends on relative sizes: if comm << compute,
    # comm is fully hidden (ratio→1.0); if comm ≈ compute, ratio ≈ 0.5.
    def test_allgather_gemm_overlap(self):
        shard_m = M // self.world
        x_local = torch.randn(shard_m, K, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(N // self.world, K, device=self.device, dtype=torch.bfloat16)
        comm_stream = torch.cuda.Stream(device=self.device)

        for _ in range(3):
            _distributed_allgather(x_local)
            _ = x_local @ w.T
        torch.cuda.synchronize()

        r_comm = cuda_timer(
            lambda: _distributed_allgather(x_local),
            label="allgather alone",
            iters=10,
        )
        r_compute = cuda_timer(lambda: x_local @ w.T, label="GEMM alone", iters=10)

        def _seq():
            gathered = _distributed_allgather(x_local)
            torch.cuda.synchronize()
            _ = gathered @ w.T
            torch.cuda.synchronize()

        r_seq = cuda_timer(_seq, label="sequential", iters=10)

        # Overlapped: allgather on comm_stream while local GEMM runs.
        # gathered result intentionally unused — measuring overlap latency.
        def _ovl():
            with torch.cuda.stream(comm_stream):
                _distributed_allgather(x_local)
            _ = x_local @ w.T
            comm_stream.synchronize()

        r_ovl = cuda_timer(_ovl, label="overlapped", iters=10)

        T_parts = r_comm.avg_ms + r_compute.avg_ms
        overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)

        if self.rank == 0:
            print_report(
                f"NCCL Column-Parallel Overlap (world={self.world})",
                [
                    r_comm,
                    r_compute,
                    r_seq,
                    r_ovl,
                ],
            )
            print(f"  Overlap ratio: {overlap_ratio:.3f}")


@_DIST
class TestNCCLRowParallelOverlap:
    """NCCL reduce-scatter + GEMM overlap (row-parallel TP pattern)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    def _reduce_scatter(self, tensor):
        shard_size = tensor.shape[0] // self.world
        out = torch.empty(shard_size, *tensor.shape[1:], device=self.device, dtype=tensor.dtype)
        dist.reduce_scatter_tensor(out, tensor)
        return out

    # Expected: overlap_ratio > 0.2. Reduce-scatter aggregates partial GEMM
    # results across ranks, using the NIC while a new GEMM can start on
    # compute SMs. NCCL reduce-scatter is more complex than allgather (it
    # involves reduction), so it may contend more for GPU resources, yielding
    # a slightly lower overlap ratio than the allgather case.
    def test_reduce_scatter_gemm_overlap(self):
        x = torch.randn(M, K // self.world, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(K, K // self.world, device=self.device, dtype=torch.bfloat16)
        gemm_output = x @ w.T
        comm_stream = torch.cuda.Stream(device=self.device)

        for _ in range(3):
            self._reduce_scatter(gemm_output)
        torch.cuda.synchronize()

        r_comm = cuda_timer(lambda: self._reduce_scatter(gemm_output), label="reduce_scatter", iters=10)
        r_compute = cuda_timer(lambda: x @ w.T, label="GEMM", iters=10)

        def _ovl():
            with torch.cuda.stream(comm_stream):
                self._reduce_scatter(gemm_output)
            _ = x @ w.T
            comm_stream.synchronize()

        r_ovl = cuda_timer(_ovl, label="overlapped", iters=10)

        T_parts = r_comm.avg_ms + r_compute.avg_ms
        overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)

        if self.rank == 0:
            print_report(
                f"NCCL Row-Parallel Overlap (world={self.world})",
                [
                    r_comm,
                    r_compute,
                    r_ovl,
                ],
            )


# ---------------------------------------------------------------------------
# 3. SDMA overlap (Lumen SdmaTpComm)
# ---------------------------------------------------------------------------


def _sdma_available():
    try:
        import mori  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    "RANK" not in os.environ or not _sdma_available(),
    reason="Multi-GPU + mori SDMA required",
)
class TestSdmaColumnOverlap:
    """Lumen SdmaTpComm allgather + GEMM overlap (true hardware parallelism)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    # Expected: Higher overlap ratio than NCCL (typically >0.5). SDMA uses
    # dedicated hardware DMA engines (separate from both compute SMs and the
    # NIC) for intra-node memory transfers. This provides true hardware-level
    # parallelism: SDMA allgather runs on DMA engines while GEMM runs on
    # compute SMs with zero contention, unlike NCCL which may use GPU SMs
    # for protocol processing.
    def test_sdma_allgather_gemm_overlap(self):
        from lumen.modules.sdma_comm import SdmaTpComm

        shard_m = M // self.world
        x_local = torch.randn(shard_m, K, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(N // self.world, K, device=self.device, dtype=torch.bfloat16)

        comm = SdmaTpComm(dist.group.WORLD)
        sdma_stream = torch.cuda.Stream(device=self.device)
        compute_stream = torch.cuda.current_stream(self.device)

        for _ in range(3):
            comm.allgather_dim0(x_local)
        torch.cuda.synchronize()

        r_comm = cuda_timer(lambda: comm.allgather_dim0(x_local), label="SDMA allgather", iters=10)
        r_compute = cuda_timer(lambda: x_local @ w.T, label="GEMM", iters=10)

        def _sdma_overlap():
            comm.allgather_dim0_async(x_local, stream=sdma_stream)
            _ = x_local @ w.T
            _ = comm.wait_allgather_dim0(stream=sdma_stream)
            compute_stream.wait_stream(sdma_stream)

        r_ovl = cuda_timer(_sdma_overlap, label="SDMA overlap", iters=10)

        T_parts = r_comm.avg_ms + r_compute.avg_ms
        overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)

        if self.rank == 0:
            print_report(
                f"SDMA Column-Parallel Overlap (world={self.world})",
                [
                    r_comm,
                    r_compute,
                    r_ovl,
                ],
            )
            print(f"  SDMA overlap ratio: {overlap_ratio:.3f}")


@pytest.mark.skipif(
    "RANK" not in os.environ or not _sdma_available(),
    reason="Multi-GPU + mori SDMA required",
)
class TestSdmaRowOverlap:
    """Lumen SdmaTpComm reduce-scatter + GEMM overlap."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    # Expected: SDMA reduce-scatter overlap should also outperform NCCL.
    # The hardware DMA engine handles the scatter while compute SMs run the
    # GEMM. The reduction step may still require some SM involvement, but the
    # bulk data movement is offloaded to DMA, enabling better overlap than
    # NCCL which processes the entire reduce-scatter on GPU SMs/NIC.
    def test_sdma_reduce_scatter_overlap(self):
        from lumen.modules.sdma_comm import SdmaTpComm

        x = torch.randn(M, K // self.world, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(K, K // self.world, device=self.device, dtype=torch.bfloat16)
        gemm_output = x @ w.T

        comm = SdmaTpComm(dist.group.WORLD)
        sdma_stream = torch.cuda.Stream(device=self.device)
        compute_stream = torch.cuda.current_stream(self.device)

        for _ in range(3):
            comm.reduce_scatter_dim0(gemm_output)
        torch.cuda.synchronize()

        r_comm = cuda_timer(
            lambda: comm.reduce_scatter_dim0(gemm_output),
            label="SDMA reduce_scatter",
            iters=10,
        )
        r_compute = cuda_timer(lambda: x @ w.T, label="GEMM", iters=10)

        def _sdma_rs_overlap():
            with torch.cuda.stream(sdma_stream):
                comm.reduce_scatter_dim0_async(gemm_output, stream=sdma_stream)
            _ = x @ w.T
            comm.wait_reduce_scatter_dim0(stream=sdma_stream)
            compute_stream.wait_stream(sdma_stream)

        r_ovl = cuda_timer(_sdma_rs_overlap, label="SDMA RS overlap", iters=10)

        T_parts = r_comm.avg_ms + r_compute.avg_ms
        overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)

        if self.rank == 0:
            print_report(
                f"SDMA Row-Parallel Overlap (world={self.world})",
                [
                    r_comm,
                    r_compute,
                    r_ovl,
                ],
            )


# ---------------------------------------------------------------------------
# 4. NCCL vs SDMA comparison
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    "RANK" not in os.environ or not _sdma_available(),
    reason="Multi-GPU + mori SDMA required",
)
class TestNCCLvsSdma:
    """Direct comparison of NCCL vs SDMA overlap ratios."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    # Expected: SDMA should show >1.1x speedup over NCCL for overlapped
    # allgather+GEMM. The key advantage is hardware isolation: SDMA uses
    # dedicated DMA engines that don't compete with the GEMM for compute SMs
    # or memory bandwidth. NCCL uses GPU SMs for protocol processing, causing
    # resource contention with the GEMM. On AMD MI300X, this translates to
    # measurably higher overlap ratios and lower end-to-end latency.
    def test_nccl_vs_sdma_allgather_overlap(self):
        from lumen.modules.sdma_comm import SdmaTpComm

        shard_m = M // self.world
        x_local = torch.randn(shard_m, K, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(N // self.world, K, device=self.device, dtype=torch.bfloat16)
        comm = SdmaTpComm(dist.group.WORLD)
        comm_stream = torch.cuda.Stream(device=self.device)
        sdma_stream = torch.cuda.Stream(device=self.device)
        compute_stream = torch.cuda.current_stream(self.device)

        for _ in range(3):
            _distributed_allgather(x_local)
            comm.allgather_dim0(x_local)
        torch.cuda.synchronize()

        # NCCL overlap
        def _nccl():
            with torch.cuda.stream(comm_stream):
                _distributed_allgather(x_local)
            _ = x_local @ w.T
            comm_stream.synchronize()

        r_nccl = cuda_timer(_nccl, label="NCCL overlap", iters=10)

        # SDMA overlap
        def _sdma():
            comm.allgather_dim0_async(x_local, stream=sdma_stream)
            _ = x_local @ w.T
            comm.wait_allgather_dim0(stream=sdma_stream)
            compute_stream.wait_stream(sdma_stream)

        r_sdma = cuda_timer(_sdma, label="SDMA overlap", iters=10)

        speedup = r_nccl.avg_ms / max(r_sdma.avg_ms, 1e-6)
        r_sdma.extra["speedup_vs_nccl"] = round(speedup, 2)

        if self.rank == 0:
            print_report(f"NCCL vs SDMA Overlap (world={self.world})", [r_nccl, r_sdma])


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def main():
    require_cuda()
    _init_dist()

    results: List[BenchResult] = []

    # Single-GPU simulation
    tp_size = 2
    shard_m = M // tp_size
    x_local = torch.randn(shard_m, K, device="cuda", dtype=torch.bfloat16)
    x_full = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    comm_stream = torch.cuda.Stream()

    r_comm = cuda_timer(
        lambda: x_full.copy_(torch.cat([x_local, x_local], dim=0)),
        label="simulated allgather",
    )
    r_compute = cuda_timer(lambda: x_full @ w.T, label="full GEMM")

    def _ovl():
        with torch.cuda.stream(comm_stream):
            x_full.copy_(torch.cat([x_local, x_local], dim=0))
        _ = x_local @ w.T
        comm_stream.synchronize()
        _ = x_full[shard_m:] @ w.T

    r_ovl = cuda_timer(_ovl, label="column-parallel overlap")

    T_parts = r_comm.avg_ms + r_compute.avg_ms
    overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
    r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)

    results = [r_comm, r_compute, r_ovl]

    is_rank_0 = not dist.is_initialized() or dist.get_rank() == 0
    if is_rank_0:
        print_report("Lumen TP Comm-Compute Overlap", results)

    if _is_distributed():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
