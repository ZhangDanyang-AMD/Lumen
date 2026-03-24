###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Benchmark 2 — Multi-GPU comm-compute overlap.

Measures real communication-compute overlap on multi-GPU setups using NCCL
and SDMA backends.  All tests require ``torchrun`` with at least 2 GPUs.

  * **Section 1 — Multi-GPU NCCL overlap**: Raw NCCL allgather / reduce-scatter
    overlapped with GEMM on separate streams.
  * **Section 2 — Multi-GPU SDMA overlap**: ``SdmaTpComm`` async primitives
    with dedicated hardware DMA engines.
  * **Section 3 — NCCL vs SDMA comparison**: Direct head-to-head.

The *overlap ratio* is defined as::

    overlap_ratio = 1 - (T_overlapped / (T_comm + T_compute))

Run NCCL overlap::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_comm_overlap.py -v -s -k NCCL
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_comm_overlap.py -v -s -k NCCL

Run SDMA overlap (requires mori)::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_comm_overlap.py -v -s -k Sdma
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_comm_overlap.py -v -s -k Sdma

Run NCCL vs SDMA head-to-head (requires mori)::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_comm_overlap.py -v -s -k NCCLvsSdma

Run all tests::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_comm_overlap.py -v -s
"""

from __future__ import annotations

import os
from typing import List

import pytest
import torch
import torch.distributed as dist

from benchmarks.bench_utils import (
    BenchResult,
    compute_bandwidth_gb_s,
    cuda_timer,
    print_bandwidth_summary,
    print_bench_warnings,
    print_overlap_summary,
    print_report_with_table,
)

# ---------------------------------------------------------------------------
# Dimensions — Llama 3.1 8B column-parallel shape
# ---------------------------------------------------------------------------
B, S = 2, 2048
H, D = 32, 128
HIDDEN = H * D  # 4096
FFN_HIDDEN = 14336

M = B * S  # tokens
K = HIDDEN  # hidden_dim
N = FFN_HIDDEN  # FFN intermediate (sharded by TP)

# Timing parameters — overridable via LUMEN_BENCH_WARMUP / LUMEN_BENCH_ITERS
_WARMUP = 10
_ITERS = 30
_TRIM = 10.0  # trim 10% outliers from both tails


# ---------------------------------------------------------------------------
# 1. Multi-GPU NCCL overlap
# ---------------------------------------------------------------------------


def _init_dist():
    if dist.is_initialized():
        return
    if "RANK" not in os.environ:
        return
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )


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
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_compute = cuda_timer(
            lambda: x_local @ w.T,
            label="GEMM alone",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
        )

        def _seq():
            gathered = _distributed_allgather(x_local)
            _ = gathered @ w.T

        r_seq = cuda_timer(
            _seq,
            label="sequential",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        def _ovl():
            with torch.cuda.stream(comm_stream):
                _distributed_allgather(x_local)
            _ = x_local @ w.T
            comm_stream.synchronize()

        r_ovl = cuda_timer(
            _ovl,
            label="overlapped",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        T_parts = r_comm.avg_ms + r_compute.avg_ms
        overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
        speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)
        r_ovl.extra["speedup"] = round(speedup, 2)

        if self.rank == 0:
            print_report_with_table(
                f"NCCL Column-Parallel Overlap (world={self.world})",
                [r_comm, r_compute, r_seq, r_ovl],
            )
            print_overlap_summary(
                t_compute=r_compute.avg_ms,
                t_comm=r_comm.avg_ms,
                t_seq=r_seq.avg_ms,
                t_ovl=r_ovl.avg_ms,
                compute_label="GEMM",
                comm_label="allgather",
            )


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

        r_comm = cuda_timer(
            lambda: self._reduce_scatter(gemm_output),
            label="reduce_scatter",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_compute = cuda_timer(
            lambda: x @ w.T,
            label="GEMM",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
        )

        def _seq():
            self._reduce_scatter(gemm_output)
            _ = x @ w.T

        r_seq = cuda_timer(
            _seq,
            label="sequential",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        def _ovl():
            with torch.cuda.stream(comm_stream):
                self._reduce_scatter(gemm_output)
            _ = x @ w.T
            comm_stream.synchronize()

        r_ovl = cuda_timer(
            _ovl,
            label="overlapped",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        T_parts = r_comm.avg_ms + r_compute.avg_ms
        overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
        speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)
        r_ovl.extra["speedup"] = round(speedup, 2)

        if self.rank == 0:
            print_report_with_table(
                f"NCCL Row-Parallel Overlap (world={self.world})",
                [r_comm, r_compute, r_seq, r_ovl],
            )
            print_overlap_summary(
                t_compute=r_compute.avg_ms,
                t_comm=r_comm.avg_ms,
                t_seq=r_seq.avg_ms,
                t_ovl=r_ovl.avg_ms,
                compute_label="GEMM",
                comm_label="reduce_scatter",
            )


# ---------------------------------------------------------------------------
# 2. SDMA overlap (Lumen SdmaTpComm)
# ---------------------------------------------------------------------------


def _sdma_available():
    try:
        os.environ.setdefault("MORI_ENABLE_SDMA", "1")
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
        os.environ["MORI_ENABLE_SDMA"] = "1"
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

        r_comm = cuda_timer(
            lambda: comm.allgather_dim0(x_local),
            label="SDMA allgather",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_compute = cuda_timer(
            lambda: x_local @ w.T,
            label="GEMM",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
        )

        def _seq():
            comm.allgather_dim0(x_local)
            _ = x_local @ w.T

        r_seq = cuda_timer(
            _seq,
            label="sequential",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        def _sdma_overlap():
            comm.allgather_dim0_async(x_local, stream=sdma_stream)
            _ = x_local @ w.T
            _ = comm.wait_allgather_dim0(stream=sdma_stream)
            compute_stream.wait_stream(sdma_stream)

        r_ovl = cuda_timer(
            _sdma_overlap,
            label="SDMA overlap",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        T_parts = r_comm.avg_ms + r_compute.avg_ms
        overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
        speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)
        r_ovl.extra["speedup"] = round(speedup, 2)

        if self.rank == 0:
            print_report_with_table(
                f"SDMA Column-Parallel Overlap (world={self.world})",
                [r_comm, r_compute, r_seq, r_ovl],
            )
            print_overlap_summary(
                t_compute=r_compute.avg_ms,
                t_comm=r_comm.avg_ms,
                t_seq=r_seq.avg_ms,
                t_ovl=r_ovl.avg_ms,
                compute_label="GEMM",
                comm_label="SDMA AG",
            )


@pytest.mark.skipif(
    "RANK" not in os.environ or not _sdma_available(),
    reason="Multi-GPU + mori SDMA required",
)
class TestSdmaRowOverlap:
    """Lumen SdmaTpComm reduce-scatter + GEMM overlap."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        os.environ["MORI_ENABLE_SDMA"] = "1"
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
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_compute = cuda_timer(
            lambda: x @ w.T,
            label="GEMM",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
        )

        def _seq():
            comm.reduce_scatter_dim0(gemm_output)
            _ = x @ w.T

        r_seq = cuda_timer(
            _seq,
            label="sequential",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        def _sdma_rs_overlap():
            with torch.cuda.stream(sdma_stream):
                comm.reduce_scatter_dim0_async(gemm_output, stream=sdma_stream)
            _ = x @ w.T
            comm.wait_reduce_scatter_dim0(stream=sdma_stream)
            compute_stream.wait_stream(sdma_stream)

        r_ovl = cuda_timer(
            _sdma_rs_overlap,
            label="SDMA RS overlap",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        T_parts = r_comm.avg_ms + r_compute.avg_ms
        overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
        speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)
        r_ovl.extra["speedup"] = round(speedup, 2)

        if self.rank == 0:
            print_report_with_table(
                f"SDMA Row-Parallel Overlap (world={self.world})",
                [r_comm, r_compute, r_seq, r_ovl],
            )
            print_overlap_summary(
                t_compute=r_compute.avg_ms,
                t_comm=r_comm.avg_ms,
                t_seq=r_seq.avg_ms,
                t_ovl=r_ovl.avg_ms,
                compute_label="GEMM",
                comm_label="SDMA RS",
            )


# ---------------------------------------------------------------------------
# 3. NCCL vs SDMA comparison
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    "RANK" not in os.environ or not _sdma_available(),
    reason="Multi-GPU + mori SDMA required",
)
class TestNCCLvsSdma:
    """Direct comparison of NCCL vs SDMA overlap ratios."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        os.environ["MORI_ENABLE_SDMA"] = "1"
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
        torch.cuda.synchronize()
        for _ in range(3):
            comm.allgather_dim0(x_local)
        torch.cuda.synchronize()

        # NCCL overlap
        def _nccl():
            with torch.cuda.stream(comm_stream):
                _distributed_allgather(x_local)
            _ = x_local @ w.T
            comm_stream.synchronize()

        r_nccl = cuda_timer(
            _nccl,
            label="NCCL overlap",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        # SDMA overlap
        def _sdma():
            comm.allgather_dim0_async(x_local, stream=sdma_stream)
            _ = x_local @ w.T
            comm.wait_allgather_dim0(stream=sdma_stream)
            compute_stream.wait_stream(sdma_stream)

        r_sdma = cuda_timer(
            _sdma,
            label="SDMA overlap",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_nccl.avg_ms / max(r_sdma.avg_ms, 1e-6)
        r_sdma.extra["speedup_vs_nccl"] = round(speedup, 2)

        if self.rank == 0:
            print_report_with_table(f"NCCL vs SDMA Overlap (world={self.world})", [r_nccl, r_sdma])

    @pytest.mark.parametrize(
        "gemm_n",
        [256, 1024, 4096, 7168, 14336, 28672],
        ids=lambda n: f"N={n}",
    )
    def test_nccl_vs_sdma_scaling(self, gemm_n):
        """Sweep GEMM output dimension to find SDMA crossover point.

        Keeps the allgather size fixed (shard_m x K) and varies the GEMM
        weight width.  Smaller gemm_n → tiny GEMM where SDMA overhead
        dominates; larger gemm_n → SDMA DMA engine parallelism wins.
        """
        from lumen.modules.sdma_comm import SdmaTpComm

        shard_m = M // self.world
        x_local = torch.randn(shard_m, K, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(gemm_n, K, device=self.device, dtype=torch.bfloat16)

        comm = SdmaTpComm(dist.group.WORLD)
        comm_stream = torch.cuda.Stream(device=self.device)
        sdma_stream = torch.cuda.Stream(device=self.device)
        compute_stream = torch.cuda.current_stream(self.device)

        for _ in range(3):
            _distributed_allgather(x_local)
            _ = x_local @ w.T
        torch.cuda.synchronize()
        for _ in range(3):
            comm.allgather_dim0(x_local)
            _ = x_local @ w.T
        torch.cuda.synchronize()

        gemm_flops = 2 * shard_m * K * gemm_n

        r_gemm = cuda_timer(
            lambda: x_local @ w.T,
            label=f"GEMM N={gemm_n}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
        )

        def _nccl():
            with torch.cuda.stream(comm_stream):
                _distributed_allgather(x_local)
            _ = x_local @ w.T
            comm_stream.synchronize()

        def _sdma():
            comm.allgather_dim0_async(x_local, stream=sdma_stream)
            _ = x_local @ w.T
            comm.wait_allgather_dim0(stream=sdma_stream)
            compute_stream.wait_stream(sdma_stream)

        r_nccl = cuda_timer(
            _nccl,
            label=f"NCCL N={gemm_n}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_sdma = cuda_timer(
            _sdma,
            label=f"SDMA N={gemm_n}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_nccl.avg_ms / max(r_sdma.avg_ms, 1e-6)
        winner = "SDMA" if speedup > 1.0 else "NCCL"
        r_sdma.extra["vs_nccl"] = f"{speedup:.2f}x"
        r_sdma.extra["winner"] = winner
        r_gemm.extra["GFLOPS"] = round(gemm_flops / max(r_gemm.avg_ms * 1e6, 1), 1)

        if self.rank == 0:
            print_report_with_table(
                f"NCCL vs SDMA  N={gemm_n}  (world={self.world})",
                [r_gemm, r_nccl, r_sdma],
            )

    def test_nccl_vs_sdma_scaling_summary(self):
        """Print a combined summary table after all scaling tests.

        Runs all GEMM sizes in one test so the results appear in a single
        table for easy comparison.
        """
        from lumen.modules.sdma_comm import SdmaTpComm

        shard_m = M // self.world
        x_local = torch.randn(shard_m, K, device=self.device, dtype=torch.bfloat16)
        comm = SdmaTpComm(dist.group.WORLD)
        comm_stream = torch.cuda.Stream(device=self.device)
        sdma_stream = torch.cuda.Stream(device=self.device)
        compute_stream = torch.cuda.current_stream(self.device)

        sizes = [256, 1024, 4096, 7168, 14336, 28672]
        all_results: List[BenchResult] = []

        for gemm_n in sizes:
            w = torch.randn(gemm_n, K, device=self.device, dtype=torch.bfloat16)

            for _ in range(3):
                _distributed_allgather(x_local)
                _ = x_local @ w.T
            torch.cuda.synchronize()
            for _ in range(3):
                comm.allgather_dim0(x_local)
                _ = x_local @ w.T
            torch.cuda.synchronize()

            def _nccl(w=w):
                with torch.cuda.stream(comm_stream):
                    _distributed_allgather(x_local)
                _ = x_local @ w.T
                comm_stream.synchronize()

            def _sdma(w=w):
                comm.allgather_dim0_async(x_local, stream=sdma_stream)
                _ = x_local @ w.T
                comm.wait_allgather_dim0(stream=sdma_stream)
                compute_stream.wait_stream(sdma_stream)

            r_nccl = cuda_timer(
                _nccl,
                label=f"NCCL  N={gemm_n:>5}",
                warmup=_WARMUP,
                iters=_ITERS,
                trim_pct=_TRIM,
                dist_barrier=True,
            )
            r_sdma = cuda_timer(
                _sdma,
                label=f"SDMA  N={gemm_n:>5}",
                warmup=_WARMUP,
                iters=_ITERS,
                trim_pct=_TRIM,
                dist_barrier=True,
            )

            speedup = r_nccl.avg_ms / max(r_sdma.avg_ms, 1e-6)
            tag = "SDMA wins" if speedup > 1.0 else "NCCL wins"
            r_sdma.extra["vs_nccl"] = f"{speedup:.2f}x ({tag})"
            all_results.extend([r_nccl, r_sdma])

        if self.rank == 0:
            print_report_with_table(
                f"NCCL vs SDMA Scaling Summary (world={self.world})",
                all_results,
            )

    def test_nccl_vs_sdma_reduce_scatter_overlap(self):
        """NCCL vs SDMA for GEMM + reduce-scatter overlap."""
        from lumen.modules.sdma_comm import SdmaTpComm

        K_tp = K // self.world
        x = torch.randn(M, K_tp, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(N, K_tp, device=self.device, dtype=torch.bfloat16)
        comm = SdmaTpComm(dist.group.WORLD)
        comm_stream = torch.cuda.Stream(device=self.device)
        sdma_stream = torch.cuda.Stream(device=self.device)
        compute_stream = torch.cuda.current_stream(self.device)

        def _rs_nccl(tensor):
            shard = tensor.shape[0] // self.world
            out = torch.empty(shard, *tensor.shape[1:], device=self.device, dtype=tensor.dtype)
            dist.reduce_scatter_tensor(out, tensor)
            return out

        for _ in range(3):
            gemm_out = x @ w.T
            _rs_nccl(gemm_out)
        torch.cuda.synchronize()
        for _ in range(3):
            gemm_out = x @ w.T
            comm.reduce_scatter_dim0(gemm_out)
        torch.cuda.synchronize()

        def _nccl():
            gemm_out = x @ w.T
            with torch.cuda.stream(comm_stream):
                _rs_nccl(gemm_out)
            comm_stream.synchronize()

        r_nccl = cuda_timer(
            _nccl,
            label="NCCL RS overlap",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        def _sdma():
            gemm_out = x @ w.T
            comm.reduce_scatter_dim0_async(gemm_out, stream=sdma_stream)
            comm.wait_reduce_scatter_dim0(stream=sdma_stream)
            compute_stream.wait_stream(sdma_stream)

        r_sdma = cuda_timer(
            _sdma,
            label="SDMA RS overlap",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_nccl.avg_ms / max(r_sdma.avg_ms, 1e-6)
        r_sdma.extra["speedup_vs_nccl"] = round(speedup, 2)
        rs_bytes = M * N * 2
        r_nccl.extra["rs_bw_gb_s"] = round(compute_bandwidth_gb_s(rs_bytes, r_nccl.avg_ms), 1)
        r_sdma.extra["rs_bw_gb_s"] = round(compute_bandwidth_gb_s(rs_bytes, r_sdma.avg_ms), 1)

        if self.rank == 0:
            print_report_with_table(f"NCCL vs SDMA RS Overlap (world={self.world})", [r_nccl, r_sdma])
            print_bandwidth_summary(label="RS (NCCL)", bytes_transferred=rs_bytes, time_ms=r_nccl.avg_ms)
            print_bandwidth_summary(label="RS (SDMA)", bytes_transferred=rs_bytes, time_ms=r_sdma.avg_ms)
            print_bench_warnings(result=r_sdma, speedup=speedup)

    @pytest.mark.parametrize(
        "gemm_n",
        [256, 1024, 4096, 7168, 14336, 28672],
        ids=lambda n: f"N={n}",
    )
    def test_nccl_vs_sdma_rs_scaling(self, gemm_n):
        """Sweep GEMM output dim for RS: SDMA vs NCCL crossover."""
        from lumen.modules.sdma_comm import SdmaTpComm

        K_tp = K // self.world
        x = torch.randn(M, K_tp, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(gemm_n, K_tp, device=self.device, dtype=torch.bfloat16)

        comm = SdmaTpComm(dist.group.WORLD)
        comm_stream = torch.cuda.Stream(device=self.device)
        sdma_stream = torch.cuda.Stream(device=self.device)
        compute_stream = torch.cuda.current_stream(self.device)

        def _rs_nccl(tensor):
            shard = tensor.shape[0] // self.world
            out = torch.empty(shard, *tensor.shape[1:], device=self.device, dtype=tensor.dtype)
            dist.reduce_scatter_tensor(out, tensor)
            return out

        for _ in range(3):
            gemm_out = x @ w.T
            _rs_nccl(gemm_out)
        torch.cuda.synchronize()
        for _ in range(3):
            gemm_out = x @ w.T
            comm.reduce_scatter_dim0(gemm_out)
        torch.cuda.synchronize()

        rs_bytes = M * gemm_n * 2

        r_gemm = cuda_timer(
            lambda: x @ w.T,
            label=f"GEMM N={gemm_n}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
        )

        def _nccl(w=w):
            gemm_out = x @ w.T
            with torch.cuda.stream(comm_stream):
                _rs_nccl(gemm_out)
            comm_stream.synchronize()

        def _sdma(w=w):
            gemm_out = x @ w.T
            comm.reduce_scatter_dim0_async(gemm_out, stream=sdma_stream)
            comm.wait_reduce_scatter_dim0(stream=sdma_stream)
            compute_stream.wait_stream(sdma_stream)

        r_nccl = cuda_timer(
            _nccl,
            label=f"NCCL RS N={gemm_n}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_sdma = cuda_timer(
            _sdma,
            label=f"SDMA RS N={gemm_n}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_nccl.avg_ms / max(r_sdma.avg_ms, 1e-6)
        winner = "SDMA" if speedup > 1.0 else "NCCL"
        r_sdma.extra["vs_nccl"] = f"{speedup:.2f}x"
        r_sdma.extra["winner"] = winner
        r_sdma.extra["rs_bw_gb_s"] = round(compute_bandwidth_gb_s(rs_bytes, r_sdma.avg_ms), 1)

        if self.rank == 0:
            print_report_with_table(
                f"NCCL vs SDMA RS  N={gemm_n}  (world={self.world})",
                [r_gemm, r_nccl, r_sdma],
            )
            print_bench_warnings(result=r_sdma, speedup=speedup)

    def test_nccl_vs_sdma_rs_scaling_summary(self):
        """Combined summary for RS scaling across all GEMM sizes."""
        from lumen.modules.sdma_comm import SdmaTpComm

        K_tp = K // self.world
        x = torch.randn(M, K_tp, device=self.device, dtype=torch.bfloat16)
        comm = SdmaTpComm(dist.group.WORLD)
        comm_stream = torch.cuda.Stream(device=self.device)
        sdma_stream = torch.cuda.Stream(device=self.device)
        compute_stream = torch.cuda.current_stream(self.device)

        sizes = [256, 1024, 4096, 7168, 14336, 28672]
        all_results: List[BenchResult] = []

        def _rs_nccl(tensor):
            shard = tensor.shape[0] // self.world
            out = torch.empty(shard, *tensor.shape[1:], device=self.device, dtype=tensor.dtype)
            dist.reduce_scatter_tensor(out, tensor)
            return out

        for gemm_n in sizes:
            w = torch.randn(gemm_n, K_tp, device=self.device, dtype=torch.bfloat16)

            for _ in range(3):
                gemm_out = x @ w.T
                _rs_nccl(gemm_out)
            torch.cuda.synchronize()
            for _ in range(3):
                gemm_out = x @ w.T
                comm.reduce_scatter_dim0(gemm_out)
            torch.cuda.synchronize()

            def _nccl(w=w):
                gemm_out = x @ w.T
                with torch.cuda.stream(comm_stream):
                    _rs_nccl(gemm_out)
                comm_stream.synchronize()

            def _sdma(w=w):
                gemm_out = x @ w.T
                comm.reduce_scatter_dim0_async(gemm_out, stream=sdma_stream)
                comm.wait_reduce_scatter_dim0(stream=sdma_stream)
                compute_stream.wait_stream(sdma_stream)

            r_nccl = cuda_timer(
                _nccl,
                label=f"NCCL RS N={gemm_n:>5}",
                warmup=_WARMUP,
                iters=_ITERS,
                trim_pct=_TRIM,
                dist_barrier=True,
            )
            r_sdma = cuda_timer(
                _sdma,
                label=f"SDMA RS N={gemm_n:>5}",
                warmup=_WARMUP,
                iters=_ITERS,
                trim_pct=_TRIM,
                dist_barrier=True,
            )

            speedup = r_nccl.avg_ms / max(r_sdma.avg_ms, 1e-6)
            r_sdma.extra["vs_nccl"] = f"{speedup:.2f}x"
            r_sdma.extra["winner"] = "SDMA" if speedup > 1.0 else "NCCL"
            rs_bytes = M * gemm_n * 2
            r_sdma.extra["rs_bw_gb_s"] = round(compute_bandwidth_gb_s(rs_bytes, r_sdma.avg_ms), 1)
            all_results.extend([r_nccl, r_sdma])

        if self.rank == 0:
            print_report_with_table(
                f"NCCL vs SDMA RS Scaling Summary (world={self.world})",
                all_results,
            )
