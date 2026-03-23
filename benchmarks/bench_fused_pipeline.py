###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Benchmark — Fused pipelined comm-GEMM overlap.

Measures the latency and overlap ratio of the fused pipeline path
(``_FusedColumnParallelForward`` / ``_FusedRowParallelForward``) vs:

  * **Sequential baseline**: AG/RS then GEMM (no overlap).
  * **Fused pipeline**: Real AG+GEMM / GEMM+RS overlap in forward,
    dGEMM+RS / AG+dGEMM overlap in backward.

Both column-parallel and row-parallel are benchmarked.

Requires >= 2 GPUs with NCCL.

Run forward-only::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fused_pipeline.py -v -s -k fwd
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fused_pipeline.py -v -s -k fwd

Run forward + backward::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fused_pipeline.py -v -s -k fwd_bwd
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fused_pipeline.py -v -s -k fwd_bwd

Run chunk scaling sweep::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fused_pipeline.py -v -s -k chunk_sweep

Run all::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fused_pipeline.py -v -s
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from benchmarks.bench_utils import (
    cuda_timer,
    print_overlap_summary,
    print_report_with_table,
)

# Dimensions — Llama 3.1 8B shapes
B, S = 2, 2048
H = 4096
FFN = 14336
_WARMUP = 10
_ITERS = 30
_TRIM = 10.0

_DIST = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Multi-GPU — run with torchrun --nproc_per_node=N",
)


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


@_DIST
class TestFusedPipelineColumnFwd:
    """Column-parallel forward: AG + GEMM overlap."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    def test_fused_vs_sequential_fwd(self):
        from lumen.modules.comm_overlap import (
            NcclCommBackend,
            PipelinedAllgatherGemm,
            PipelinedGemmReduceScatter,
            fused_column_parallel_forward,
        )

        S_local = (B * S) // self.world
        x_local = torch.randn(S_local, H, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(FFN // self.world, H, device=self.device, dtype=torch.bfloat16)
        w_param = torch.nn.Parameter(w.clone())

        backend = NcclCommBackend(dist.group.WORLD)
        pipeline_ag = PipelinedAllgatherGemm(num_chunks=4, comm=backend)
        pipeline_rs = PipelinedGemmReduceScatter(num_chunks=4, comm=backend)

        for _ in range(3):
            gathered = _distributed_allgather(x_local)
            _ = F.linear(gathered, w)
        torch.cuda.synchronize()

        def _seq():
            gathered = _distributed_allgather(x_local)
            return F.linear(gathered, w)

        r_seq = cuda_timer(
            _seq,
            label="sequential (AG->GEMM)",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        r_comm = cuda_timer(
            lambda: _distributed_allgather(x_local),
            label="AG alone",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        gathered = _distributed_allgather(x_local)
        r_compute = cuda_timer(
            lambda: F.linear(gathered, w),
            label="GEMM alone",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
        )

        def _fused():
            return fused_column_parallel_forward(
                x_local.clone(),
                w.clone(),
                None,
                pipeline_ag,
                pipeline_rs,
                w_param,
                False,
                False,
                None,
                False,
            )

        r_fused = cuda_timer(
            _fused,
            label="fused pipeline (AG+GEMM)",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        T_parts = r_comm.avg_ms + r_compute.avg_ms
        overlap_ratio = 1 - (r_fused.avg_ms / max(T_parts, 1e-6))
        speedup = r_seq.avg_ms / max(r_fused.avg_ms, 1e-6)
        r_fused.extra["overlap_ratio"] = round(overlap_ratio, 3)
        r_fused.extra["speedup"] = round(speedup, 2)

        if self.rank == 0:
            print_report_with_table(
                f"Fused Column-Parallel Forward (world={self.world})",
                [r_comm, r_compute, r_seq, r_fused],
            )
            print_overlap_summary(
                t_compute=r_compute.avg_ms,
                t_comm=r_comm.avg_ms,
                t_seq=r_seq.avg_ms,
                t_ovl=r_fused.avg_ms,
                compute_label="GEMM",
                comm_label="allgather",
            )

    @pytest.mark.parametrize("num_chunks", [1, 2, 4, 8])
    def test_chunk_sweep_fwd(self, num_chunks):
        from lumen.modules.comm_overlap import (
            NcclCommBackend,
            PipelinedAllgatherGemm,
            PipelinedGemmReduceScatter,
            fused_column_parallel_forward,
        )

        S_local = (B * S) // self.world
        x_local = torch.randn(S_local, H, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(FFN // self.world, H, device=self.device, dtype=torch.bfloat16)
        w_param = torch.nn.Parameter(w.clone())

        if S_local * self.world % num_chunks != 0:
            pytest.skip(f"S={S_local * self.world} not divisible by {num_chunks}")

        backend = NcclCommBackend(dist.group.WORLD)
        pipeline_ag = PipelinedAllgatherGemm(num_chunks=num_chunks, comm=backend)
        pipeline_rs = PipelinedGemmReduceScatter(num_chunks=num_chunks, comm=backend)

        def _fused():
            return fused_column_parallel_forward(
                x_local.clone(),
                w.clone(),
                None,
                pipeline_ag,
                pipeline_rs,
                w_param,
                False,
                False,
                None,
                False,
            )

        r = cuda_timer(
            _fused,
            label=f"fused N={num_chunks}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        if self.rank == 0:
            print_report_with_table(
                f"Column Fwd Chunk Sweep N={num_chunks} (world={self.world})",
                [r],
            )


@_DIST
class TestFusedPipelineFwdBwd:
    """Full forward + backward cycle latency."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    def test_fused_column_fwd_bwd(self):
        from lumen.modules.comm_overlap import (
            NcclCommBackend,
            PipelinedAllgatherGemm,
            PipelinedGemmReduceScatter,
            fused_column_parallel_forward,
        )

        S_local = (B * S) // self.world
        w = torch.randn(FFN // self.world, H, device=self.device, dtype=torch.bfloat16)

        backend = NcclCommBackend(dist.group.WORLD)
        pipeline_ag = PipelinedAllgatherGemm(num_chunks=4, comm=backend)
        pipeline_rs = PipelinedGemmReduceScatter(num_chunks=4, comm=backend)
        w_param = torch.nn.Parameter(w.clone())

        def _seq_fwd_bwd():
            x = torch.randn(S_local, H, device=self.device, dtype=torch.bfloat16, requires_grad=True)
            gathered = _distributed_allgather(x)
            y = F.linear(gathered, w)
            y.sum().backward()
            return x.grad

        def _fused_fwd_bwd():
            x = torch.randn(S_local, H, device=self.device, dtype=torch.bfloat16, requires_grad=True)
            y = fused_column_parallel_forward(
                x,
                w_param,
                None,
                pipeline_ag,
                pipeline_rs,
                w_param,
                False,
                False,
                None,
                False,
            )
            y.sum().backward()
            return x.grad

        for _ in range(3):
            _seq_fwd_bwd()
            _fused_fwd_bwd()
        torch.cuda.synchronize()

        r_seq = cuda_timer(
            _seq_fwd_bwd,
            label="sequential fwd+bwd",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_fused = cuda_timer(
            _fused_fwd_bwd,
            label="fused fwd+bwd",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_seq.avg_ms / max(r_fused.avg_ms, 1e-6)
        r_fused.extra["speedup"] = round(speedup, 2)

        if self.rank == 0:
            print_report_with_table(
                f"Fused Column Fwd+Bwd (world={self.world})",
                [r_seq, r_fused],
            )


@_DIST
class TestFusedPipelineRowFwd:
    """Row-parallel forward: GEMM + RS overlap."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    def test_fused_row_vs_sequential_fwd(self):
        from lumen.modules.comm_overlap import (
            NcclCommBackend,
            PipelinedAllgatherGemm,
            PipelinedGemmReduceScatter,
            fused_row_parallel_forward,
        )

        S_full = B * S
        H_tp = H // self.world
        x = torch.randn(S_full, H_tp, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(FFN, H_tp, device=self.device, dtype=torch.bfloat16)
        w_param = torch.nn.Parameter(w.clone())

        backend = NcclCommBackend(dist.group.WORLD)
        pipeline_rs = PipelinedGemmReduceScatter(num_chunks=4, comm=backend)
        pipeline_ag = PipelinedAllgatherGemm(num_chunks=4, comm=backend)

        def _reduce_scatter(tensor):
            shard = tensor.shape[0] // self.world
            out = torch.empty(shard, *tensor.shape[1:], device=self.device, dtype=tensor.dtype)
            dist.reduce_scatter_tensor(out, tensor)
            return out

        for _ in range(3):
            gemm_out = F.linear(x, w)
            _reduce_scatter(gemm_out)
        torch.cuda.synchronize()

        def _seq():
            return _reduce_scatter(F.linear(x, w))

        r_seq = cuda_timer(
            _seq,
            label="sequential (GEMM->RS)",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        r_compute = cuda_timer(
            lambda: F.linear(x, w),
            label="GEMM alone",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
        )

        gemm_out = F.linear(x, w)
        r_comm = cuda_timer(
            lambda: _reduce_scatter(gemm_out),
            label="RS alone",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        def _fused():
            return fused_row_parallel_forward(
                x.clone(),
                w.clone(),
                pipeline_rs,
                pipeline_ag,
                w_param,
                False,
                False,
                None,
            )

        r_fused = cuda_timer(
            _fused,
            label="fused pipeline (GEMM+RS)",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        T_parts = r_comm.avg_ms + r_compute.avg_ms
        overlap_ratio = 1 - (r_fused.avg_ms / max(T_parts, 1e-6))
        speedup = r_seq.avg_ms / max(r_fused.avg_ms, 1e-6)
        r_fused.extra["overlap_ratio"] = round(overlap_ratio, 3)
        r_fused.extra["speedup"] = round(speedup, 2)

        if self.rank == 0:
            print_report_with_table(
                f"Fused Row-Parallel Forward (world={self.world})",
                [r_comm, r_compute, r_seq, r_fused],
            )
            print_overlap_summary(
                t_compute=r_compute.avg_ms,
                t_comm=r_comm.avg_ms,
                t_seq=r_seq.avg_ms,
                t_ovl=r_fused.avg_ms,
                compute_label="GEMM",
                comm_label="reduce_scatter",
            )
