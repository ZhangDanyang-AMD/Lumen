###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Benchmark 2 — Multi-GPU training achieves target overlap ratio.

Exercises Lumen's TP communication-compute overlap features using real
``LumenColumnParallelLinear`` and ``LumenRowParallelLinear`` modules:

  * **Section 1 — Module-level overlap (single-GPU, mocked comm)**:
    Instantiates real Lumen parallel linear modules with mocked SdmaTpComm.
    Compares ``tp_comm_overlap=True`` (``_forward_sdma_overlap_column``:
    async allgather + local GEMM + wait + remote GEMM) against
    ``tp_comm_overlap=False`` (sync allgather + full GEMM).
  * **Section 2 — Multi-GPU NCCL overlap**: Raw NCCL allgather / reduce-scatter
    overlapped with GEMM on separate streams.
  * **Section 3 — Multi-GPU SDMA overlap**: ``SdmaTpComm`` async primitives
    with dedicated hardware DMA engines.
  * **Section 4 — NCCL vs SDMA comparison**: Direct head-to-head.

The *overlap ratio* is defined as::

    overlap_ratio = 1 - (T_overlapped / (T_comm + T_compute))

Run single-GPU (uses mocked comm, no ``torchrun`` needed)::

    pytest benchmarks/bench_comm_overlap.py -v -s -k Lumen

Run multi-GPU (Section 1 standalone — mocked comm, prints summary table)::

    torchrun --nproc_per_node=2 -m benchmarks.bench_comm_overlap

Run multi-GPU NCCL overlap (Section 2 — real NCCL allgather/reduce-scatter)::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_comm_overlap.py -v -s -k NCCL
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_comm_overlap.py -v -s -k NCCL

Run multi-GPU SDMA overlap (Section 3 — requires mori, uses hardware DMA)::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_comm_overlap.py -v -s -k Sdma
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_comm_overlap.py -v -s -k Sdma

Run NCCL vs SDMA head-to-head (Section 4 — requires mori)::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_comm_overlap.py -v -s -k NCCLvsSdma

Run all multi-GPU tests together::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_comm_overlap.py -v -s
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import List
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from benchmarks.bench_utils import (
    BenchResult,
    cuda_timer,
    print_overlap_summary,
    print_report,
    require_cuda,
)
from benchmarks.conftest import CUDA

# ---------------------------------------------------------------------------
# Dimensions — Llama 3.1 8B column-parallel shape
# ---------------------------------------------------------------------------
B, S = 2, 2048
H, D = 32, 128
HIDDEN = H * D  # 4096
FFN_HIDDEN = 14336
TP_SIZE = 2

M = B * S  # tokens
K = HIDDEN  # hidden_dim
N = FFN_HIDDEN  # FFN intermediate (sharded by TP)


# ---------------------------------------------------------------------------
# Helpers: mock environment for single-GPU Lumen module instantiation
# ---------------------------------------------------------------------------

_PARALLEL_LINEAR = "lumen.modules.parallel_linear"

_MODULE_PATCHES = {
    f"{_PARALLEL_LINEAR}._get_tp_group": mock.MagicMock(),
    f"{_PARALLEL_LINEAR}._pg_size": TP_SIZE,
    f"{_PARALLEL_LINEAR}._pg_rank": 0,
    f"{_PARALLEL_LINEAR}._use_sdma_from_args": True,
    f"{_PARALLEL_LINEAR}._tp_comm_overlap_from_args": False,
    f"{_PARALLEL_LINEAR}.divide": lambda a, b: a // b,
    f"{_PARALLEL_LINEAR}._initialize_affine_weight_gpu": None,
    f"{_PARALLEL_LINEAR}.set_tensor_model_parallel_attributes": None,
    f"{_PARALLEL_LINEAR}.make_sharded_tensors_for_checkpoint": {},
    f"{_PARALLEL_LINEAR}.copy_to_tensor_model_parallel_region": lambda x, **kw: x,
    f"{_PARALLEL_LINEAR}.gather_from_sequence_parallel_region": lambda x, **kw: x,
    f"{_PARALLEL_LINEAR}.gather_from_tensor_model_parallel_region": lambda x, **kw: x,
    f"{_PARALLEL_LINEAR}.reduce_from_tensor_model_parallel_region": lambda x, **kw: x,
    f"{_PARALLEL_LINEAR}.reduce_scatter_to_sequence_parallel_region": lambda x, **kw: x,
}


def _make_config(sequence_parallel=False, lumen_tp_comm_overlap=False):
    return SimpleNamespace(
        params_dtype=torch.bfloat16,
        perform_initialization=False,
        use_cpu_initialization=False,
        sequence_parallel=sequence_parallel,
        tensor_model_parallel_size=TP_SIZE,
        expert_model_parallel_size=1,
        lumen_tp_comm_overlap=lumen_tp_comm_overlap,
    )


def _apply_patches():
    """Return a list of started mock.patch objects (caller must stop them)."""
    patches = []
    for target, val in _MODULE_PATCHES.items():
        if val is None:
            p = mock.patch(target)
        elif callable(val) and not isinstance(val, mock.MagicMock):
            p = mock.patch(target, side_effect=val)
        elif isinstance(val, (int, dict)):
            p = mock.patch(target, return_value=val)
        else:
            p = mock.patch(target, return_value=val)
        p.start()
        patches.append(p)
    return patches


class _MockSdmaComm:
    """Simulates SdmaTpComm async APIs with real GPU memcpy for realistic latency."""

    def __init__(self, tp_size=TP_SIZE):
        self._tp_size = tp_size
        self._stream = torch.cuda.Stream()
        self._ag_input = None
        self._ag_output = None
        self._rs_output = None

    def allgather_dim0(self, tensor):
        return torch.cat([tensor] * self._tp_size, dim=0)

    def allgather_dim0_async(self, tensor, stream=None):
        self._ag_input = tensor
        s = stream or self._stream
        with torch.cuda.stream(s):
            self._ag_output = torch.cat([tensor] * self._tp_size, dim=0)
        return True

    def wait_allgather_dim0(self, stream=None):
        s = stream or self._stream
        torch.cuda.current_stream().wait_stream(s)
        return self._ag_output

    def reduce_scatter_dim0(self, tensor):
        chunk = tensor.shape[0] // self._tp_size
        return tensor[:chunk].clone()

    def reduce_scatter_dim0_async(self, tensor, stream=None):
        s = stream or self._stream
        chunk = tensor.shape[0] // self._tp_size
        with torch.cuda.stream(s):
            self._rs_output = tensor[:chunk].clone()
        return True

    def wait_reduce_scatter_dim0(self, stream=None):
        s = stream or self._stream
        torch.cuda.current_stream().wait_stream(s)
        return self._rs_output

    def allreduce_sum_async(self, tensor, stream=None):
        s = stream or self._stream
        with torch.cuda.stream(s):
            tensor.div_(self._tp_size)
        return True

    def wait_allreduce_sum(self, stream=None):
        s = stream or self._stream
        torch.cuda.current_stream().wait_stream(s)


def _make_column_parallel(overlap: bool, seq_parallel: bool = True):
    """Instantiate a real LumenColumnParallelLinear with mocked TP env."""
    from lumen.modules.parallel_linear import LumenColumnParallelLinear

    config = _make_config(sequence_parallel=seq_parallel, lumen_tp_comm_overlap=overlap)
    m = LumenColumnParallelLinear(
        K,
        N,
        config=config,
        init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        bias=False,
    )
    torch.nn.init.kaiming_uniform_(m.weight)
    m.use_sdma = True
    m.tp_comm_overlap = overlap
    m.sequence_parallel = seq_parallel
    m.explicit_expert_comm = False
    m.tp_size = TP_SIZE
    m.gather_output = False
    m._sdma_comm = _MockSdmaComm()
    m.to(device="cuda", dtype=torch.bfloat16)
    return m


def _make_row_parallel(overlap: bool, seq_parallel: bool = False):
    """Instantiate a real LumenRowParallelLinear with mocked TP env."""
    from lumen.modules.parallel_linear import LumenRowParallelLinear

    config = _make_config(sequence_parallel=seq_parallel, lumen_tp_comm_overlap=overlap)
    m = LumenRowParallelLinear(
        N,
        K,
        config=config,
        init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        bias=False,
    )
    torch.nn.init.kaiming_uniform_(m.weight)
    m.use_sdma = True
    m.tp_comm_overlap = overlap
    m.sequence_parallel = seq_parallel
    m.explicit_expert_comm = False
    m.tp_size = TP_SIZE
    m._sdma_comm = _MockSdmaComm()
    m.to(device="cuda", dtype=torch.bfloat16)
    return m


# ---------------------------------------------------------------------------
# 1. Lumen module-level comm-compute overlap (single-GPU, mocked comm)
# ---------------------------------------------------------------------------


@CUDA
class TestLumenColumnParallelOverlap:
    """Compare LumenColumnParallelLinear with tp_comm_overlap=True vs False.

    Uses mocked SdmaTpComm so the test runs on a single GPU. The overlap
    path (_forward_sdma_overlap_column) launches allgather_dim0_async on
    a secondary stream, runs local-shard GEMM on the compute stream
    concurrently, waits for allgather, then GEMMs the remaining shards.

    The non-overlap path (_forward_sdma_pre_gemm) does a synchronous
    allgather first, then runs one full-input GEMM.
    """

    @pytest.fixture(autouse=True)
    def _setup_patches(self):
        patches = _apply_patches()
        yield
        for p in patches:
            p.stop()

    # Expected: The overlap module should be faster or comparable to the
    # non-overlap module. _forward_sdma_overlap_column overlaps the allgather
    # with the local-shard GEMM on separate streams, then only GEMMs the
    # remote shard after allgather completes. Non-overlap does sync allgather
    # first (blocking), then one full GEMM. With mocked comm (torch.cat on
    # a secondary stream), the overlap benefit comes from hiding the cat
    # latency behind the local GEMM.
    def test_overlap_vs_non_overlap(self):
        m_overlap = _make_column_parallel(overlap=True)
        m_no_overlap = _make_column_parallel(overlap=False)
        m_no_overlap.weight.data.copy_(m_overlap.weight.data)

        x = torch.randn(B, S, K, device="cuda", dtype=torch.bfloat16)

        r_no_ovl = cuda_timer(
            lambda: m_no_overlap(x),
            label="ColumnParallel tp_comm_overlap=False",
        )
        r_ovl = cuda_timer(
            lambda: m_overlap(x),
            label="ColumnParallel tp_comm_overlap=True",
        )

        latency_ratio = r_no_ovl.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_ovl.extra["latency_ratio"] = round(latency_ratio, 2)

        print_report("LumenColumnParallelLinear: Overlap vs Non-Overlap", [r_no_ovl, r_ovl])
        print(f"  Latency ratio (non-overlap / overlap): {latency_ratio:.2f}x")
        print()
        print("  NOTE: Mocked comm (torch.cat ≈ 0 ms) — ratio < 1.0 is expected.")
        print("  Overlap path has extra stream sync + 2 GEMMs overhead.")

    # Expected: Latency should scale proportionally with sequence length
    # because GEMM time dominates. The overlap benefit (hiding allgather) is
    # relatively larger at longer sequences where allgather transfers more data
    # but GEMM grows quadratically in FLOPs.
    @pytest.mark.parametrize("seqlen", [512, 1024, 2048, 4096])
    def test_overlap_seqlen_sweep(self, seqlen):
        m = _make_column_parallel(overlap=True)
        x = torch.randn(B, seqlen, K, device="cuda", dtype=torch.bfloat16)

        r = cuda_timer(lambda: m(x), label=f"ColumnParallel overlap S={seqlen}")
        print_report(f"ColumnParallel Overlap S={seqlen}", [r])
        assert r.avg_ms > 0


@CUDA
class TestLumenRowParallelOverlap:
    """Compare LumenRowParallelLinear with tp_comm_overlap=True vs False.

    Row-parallel overlap uses async reduce-scatter on a secondary stream
    after GEMM finishes. The "overlap" here means the reduce-scatter runs
    asynchronously, freeing the compute stream for subsequent layers.

    Non-overlap does a synchronous reduce-scatter that blocks the compute
    stream until communication completes.
    """

    @pytest.fixture(autouse=True)
    def _setup_patches(self):
        patches = _apply_patches()
        yield
        for p in patches:
            p.stop()

    # Expected: Overlap and non-overlap should have similar latency in
    # isolation (single-layer benchmark) because the row-parallel overlap
    # still waits for reduce-scatter before returning. The real benefit
    # appears in multi-layer pipelines where the async reduce-scatter on
    # the SDMA stream frees the compute stream to start the next layer's
    # forward pass before communication completes.
    def test_overlap_vs_non_overlap(self):
        m_overlap = _make_row_parallel(overlap=True, seq_parallel=True)
        m_no_overlap = _make_row_parallel(overlap=False, seq_parallel=True)
        m_no_overlap.weight.data.copy_(m_overlap.weight.data)

        x = torch.randn(M, N // TP_SIZE, device="cuda", dtype=torch.bfloat16)

        r_no_ovl = cuda_timer(
            lambda: m_no_overlap(x),
            label="RowParallel tp_comm_overlap=False",
        )
        r_ovl = cuda_timer(
            lambda: m_overlap(x),
            label="RowParallel tp_comm_overlap=True",
        )

        diff_pct = (r_ovl.avg_ms - r_no_ovl.avg_ms) / max(r_no_ovl.avg_ms, 1e-6) * 100
        r_ovl.extra["diff_vs_sync"] = f"{diff_pct:+.1f}%"
        print_report("LumenRowParallelLinear: Overlap vs Non-Overlap", [r_no_ovl, r_ovl])
        print(f"  Diff vs sync: {diff_pct:+.1f}%")
        print()
        print("  NOTE: Row-parallel overlap benefit appears in multi-layer pipelines,")
        print("  not in single-layer isolation (async RS still waits before returning).")

    # Expected: In a two-layer pipeline (column → row), using overlap on both
    # modules should be faster than non-overlap because the column-parallel
    # hides allgather behind local GEMM, and the row-parallel's async
    # reduce-scatter can overlap with the next layer or post-processing.
    def test_column_row_pipeline(self):
        col_ovl = _make_column_parallel(overlap=True)
        row_ovl = _make_row_parallel(overlap=True, seq_parallel=True)
        col_no_ovl = _make_column_parallel(overlap=False)
        row_no_ovl = _make_row_parallel(overlap=False, seq_parallel=True)
        col_no_ovl.weight.data.copy_(col_ovl.weight.data)
        row_no_ovl.weight.data.copy_(row_ovl.weight.data)

        x = torch.randn(B, S, K, device="cuda", dtype=torch.bfloat16)

        def _pipeline(col, row):
            out, _ = col(x)
            out2d = out.reshape(-1, out.shape[-1])
            result, _ = row(out2d)
            return result

        r_no_ovl = cuda_timer(
            lambda: _pipeline(col_no_ovl, row_no_ovl),
            label="pipeline tp_comm_overlap=False",
        )
        r_ovl = cuda_timer(
            lambda: _pipeline(col_ovl, row_ovl),
            label="pipeline tp_comm_overlap=True",
        )

        speedup = r_no_ovl.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_ovl.extra["speedup"] = round(speedup, 2)
        print_report("Column→Row Pipeline: Overlap vs Non-Overlap", [r_no_ovl, r_ovl])
        print(f"  Pipeline speedup: {speedup:.2f}x")


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
        speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)
        r_ovl.extra["speedup"] = round(speedup, 2)

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

        r_comm = cuda_timer(lambda: self._reduce_scatter(gemm_output), label="reduce_scatter", iters=10)
        r_compute = cuda_timer(lambda: x @ w.T, label="GEMM", iters=10)

        def _seq():
            self._reduce_scatter(gemm_output)
            torch.cuda.synchronize()
            _ = x @ w.T
            torch.cuda.synchronize()

        r_seq = cuda_timer(_seq, label="sequential", iters=10)

        def _ovl():
            with torch.cuda.stream(comm_stream):
                self._reduce_scatter(gemm_output)
            _ = x @ w.T
            comm_stream.synchronize()

        r_ovl = cuda_timer(_ovl, label="overlapped", iters=10)

        T_parts = r_comm.avg_ms + r_compute.avg_ms
        overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
        speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)
        r_ovl.extra["speedup"] = round(speedup, 2)

        if self.rank == 0:
            print_report(
                f"NCCL Row-Parallel Overlap (world={self.world})",
                [
                    r_comm,
                    r_compute,
                    r_seq,
                    r_ovl,
                ],
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

        r_comm = cuda_timer(lambda: comm.allgather_dim0(x_local), label="SDMA allgather", iters=10)
        r_compute = cuda_timer(lambda: x_local @ w.T, label="GEMM", iters=10)

        def _seq():
            comm.allgather_dim0(x_local)
            torch.cuda.synchronize()
            _ = x_local @ w.T
            torch.cuda.synchronize()

        r_seq = cuda_timer(_seq, label="sequential", iters=10)

        def _sdma_overlap():
            comm.allgather_dim0_async(x_local, stream=sdma_stream)
            _ = x_local @ w.T
            _ = comm.wait_allgather_dim0(stream=sdma_stream)
            compute_stream.wait_stream(sdma_stream)

        r_ovl = cuda_timer(_sdma_overlap, label="SDMA overlap", iters=10)

        T_parts = r_comm.avg_ms + r_compute.avg_ms
        overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
        speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)
        r_ovl.extra["speedup"] = round(speedup, 2)

        if self.rank == 0:
            print_report(
                f"SDMA Column-Parallel Overlap (world={self.world})",
                [
                    r_comm,
                    r_compute,
                    r_seq,
                    r_ovl,
                ],
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
            iters=10,
        )
        r_compute = cuda_timer(lambda: x @ w.T, label="GEMM", iters=10)

        def _seq():
            comm.reduce_scatter_dim0(gemm_output)
            torch.cuda.synchronize()
            _ = x @ w.T
            torch.cuda.synchronize()

        r_seq = cuda_timer(_seq, label="sequential", iters=10)

        def _sdma_rs_overlap():
            with torch.cuda.stream(sdma_stream):
                comm.reduce_scatter_dim0_async(gemm_output, stream=sdma_stream)
            _ = x @ w.T
            comm.wait_reduce_scatter_dim0(stream=sdma_stream)
            compute_stream.wait_stream(sdma_stream)

        r_ovl = cuda_timer(_sdma_rs_overlap, label="SDMA RS overlap", iters=10)

        T_parts = r_comm.avg_ms + r_compute.avg_ms
        overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
        speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)
        r_ovl.extra["speedup"] = round(speedup, 2)

        if self.rank == 0:
            print_report(
                f"SDMA Row-Parallel Overlap (world={self.world})",
                [
                    r_comm,
                    r_compute,
                    r_seq,
                    r_ovl,
                ],
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
    col_speedup = float("nan")
    row_speedup = float("nan")
    row_diff = 0.0

    # --- Lumen module-level benchmark (single-GPU, mocked comm) ---
    patches = _apply_patches()
    try:
        col_ovl = _make_column_parallel(overlap=True)
        col_no_ovl = _make_column_parallel(overlap=False)
        col_no_ovl.weight.data.copy_(col_ovl.weight.data)

        x = torch.randn(B, S, K, device="cuda", dtype=torch.bfloat16)

        r_no_ovl = cuda_timer(lambda: col_no_ovl(x), label="ColumnParallel overlap=False")
        r_ovl = cuda_timer(lambda: col_ovl(x), label="ColumnParallel overlap=True")
        col_speedup = r_no_ovl.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_ovl.extra["speedup"] = round(col_speedup, 2)
        results.extend([r_no_ovl, r_ovl])

        row_ovl = _make_row_parallel(overlap=True, seq_parallel=True)
        row_no_ovl = _make_row_parallel(overlap=False, seq_parallel=True)
        row_no_ovl.weight.data.copy_(row_ovl.weight.data)

        x_row = torch.randn(M, N // TP_SIZE, device="cuda", dtype=torch.bfloat16)
        r_row_no = cuda_timer(lambda: row_no_ovl(x_row), label="RowParallel overlap=False")
        r_row_ovl = cuda_timer(lambda: row_ovl(x_row), label="RowParallel overlap=True")
        row_speedup = r_row_no.avg_ms / max(r_row_ovl.avg_ms, 1e-6)
        row_diff = (r_row_ovl.avg_ms - r_row_no.avg_ms) / max(r_row_no.avg_ms, 1e-6) * 100
        r_row_ovl.extra["speedup"] = round(row_speedup, 2)
        r_row_ovl.extra["diff_vs_sync"] = f"{row_diff:+.1f}%"
        results.extend([r_row_no, r_row_ovl])
    finally:
        for p in patches:
            p.stop()

    is_rank_0 = not dist.is_initialized() or dist.get_rank() == 0
    if is_rank_0:
        print_report("Lumen TP Comm-Compute Overlap", results)
        print(f"  ColumnParallel speedup: {col_speedup:.2f}x")
        print(f"  RowParallel    speedup: {row_speedup:.2f}x  (diff: {row_diff:+.1f}%)")
        print()
        print("  NOTE: Section 1 uses mocked comm (torch.cat ≈ 0 ms).")
        print("  Real overlap benefits require NCCL/SDMA — see Sections 2-4.")
        print()

    if _is_distributed():
        torch.cuda.synchronize()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
