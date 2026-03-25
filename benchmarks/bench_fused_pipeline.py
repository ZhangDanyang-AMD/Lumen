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

from benchmarks.bench_utils import compute_bandwidth_gb_s  # noqa: F401
from benchmarks.bench_utils import print_bandwidth_summary  # noqa: F401
from benchmarks.bench_utils import (
    cuda_timer,
    print_bench_warnings,
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


def _sdma_available():
    try:
        os.environ.setdefault("MORI_ENABLE_SDMA", "1")
        import mori  # noqa: F401

        return True
    except ImportError:
        return False


_SDMA_DIST = pytest.mark.skipif(
    "RANK" not in os.environ or not _sdma_available(),
    reason="Multi-GPU + mori SDMA required",
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


class _AllGatherFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group
        world = dist.get_world_size(group)
        ctx.world = world
        chunks = [torch.empty_like(tensor) for _ in range(world)]
        dist.all_gather(chunks, tensor.contiguous(), group=group)
        return torch.cat(chunks, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        shard = grad_output.shape[0] // ctx.world
        out = torch.empty(
            shard,
            *grad_output.shape[1:],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        dist.reduce_scatter_tensor(out, grad_output.contiguous(), group=ctx.group)
        return out, None


class _ReduceScatterFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, world, device, group):
        ctx.group = group
        shard = tensor.shape[0] // world
        out = torch.empty(shard, *tensor.shape[1:], device=device, dtype=tensor.dtype)
        dist.reduce_scatter_tensor(out, tensor.contiguous(), group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        world = dist.get_world_size(ctx.group)
        grad_output_contig = grad_output.contiguous()
        chunks = [torch.empty_like(grad_output_contig) for _ in range(world)]
        dist.all_gather(chunks, grad_output_contig, group=ctx.group)
        return torch.cat(chunks, dim=0), None, None, None


def _distributed_allgather(tensor, group=None):
    return _AllGatherFunc.apply(tensor, group)


def _reduce_scatter(tensor, world, device, group=None):
    return _ReduceScatterFunc.apply(tensor, world, device, group)


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

        for _ in range(3):
            gemm_out = F.linear(x, w)
            _reduce_scatter(gemm_out, self.world, self.device)
        torch.cuda.synchronize()

        def _seq():
            return _reduce_scatter(F.linear(x, w), self.world, self.device)

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
            lambda: _reduce_scatter(gemm_out, self.world, self.device),
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


@_SDMA_DIST
class TestFusedPipelineSdmaColumn:
    """NCCL vs SDMA backend for fused pipelined overlap."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        os.environ["MORI_ENABLE_SDMA"] = "1"
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    def _make_backends(self):
        from lumen.modules.comm_overlap import (
            NcclCommBackend,
            PipelinedAllgatherGemm,
            PipelinedGemmReduceScatter,
            SdmaCommBackend,
        )
        from lumen.modules.sdma_comm import SdmaTpComm

        nccl = NcclCommBackend(dist.group.WORLD)
        sdma_comm = SdmaTpComm(dist.group.WORLD)
        sdma = SdmaCommBackend(sdma_comm)

        ag_nccl = PipelinedAllgatherGemm(num_chunks=4, comm=nccl)
        rs_nccl = PipelinedGemmReduceScatter(num_chunks=4, comm=nccl)
        ag_sdma = PipelinedAllgatherGemm(num_chunks=4, comm=sdma)
        rs_sdma = PipelinedGemmReduceScatter(num_chunks=4, comm=sdma)
        return (ag_nccl, rs_nccl), (ag_sdma, rs_sdma)

    def test_nccl_vs_sdma_fused_column_fwd(self):
        from lumen.modules.comm_overlap import fused_column_parallel_forward

        S_local = (B * S) // self.world
        x_local = torch.randn(S_local, H, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(FFN // self.world, H, device=self.device, dtype=torch.bfloat16)
        w_param = torch.nn.Parameter(w.clone())
        (ag_nccl, rs_nccl), (ag_sdma, rs_sdma) = self._make_backends()

        def _run(ag, rs):
            return fused_column_parallel_forward(
                x_local.clone(),
                w.clone(),
                None,
                ag,
                rs,
                w_param,
                False,
                False,
                None,
                False,
            )

        for _ in range(3):
            _run(ag_nccl, rs_nccl)
            _run(ag_sdma, rs_sdma)
        torch.cuda.synchronize()

        r_nccl = cuda_timer(
            lambda: _run(ag_nccl, rs_nccl),
            label="NCCL fused column fwd",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_sdma = cuda_timer(
            lambda: _run(ag_sdma, rs_sdma),
            label="SDMA fused column fwd",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_nccl.avg_ms / max(r_sdma.avg_ms, 1e-6)
        r_sdma.extra["speedup_vs_nccl"] = round(speedup, 2)

        if self.rank == 0:
            print_report_with_table(
                f"Fused Column Fwd: NCCL vs SDMA (world={self.world})",
                [r_nccl, r_sdma],
            )
            print_bench_warnings(result=r_sdma, speedup=speedup)

    def test_nccl_vs_sdma_fused_row_fwd(self):
        from lumen.modules.comm_overlap import fused_row_parallel_forward

        S_full = B * S
        H_tp = H // self.world
        x = torch.randn(S_full, H_tp, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(FFN, H_tp, device=self.device, dtype=torch.bfloat16)
        w_param = torch.nn.Parameter(w.clone())
        (ag_nccl, rs_nccl), (ag_sdma, rs_sdma) = self._make_backends()

        def _run(rs, ag):
            return fused_row_parallel_forward(
                x.clone(),
                w.clone(),
                rs,
                ag,
                w_param,
                False,
                False,
                None,
            )

        for _ in range(3):
            _run(rs_nccl, ag_nccl)
            _run(rs_sdma, ag_sdma)
        torch.cuda.synchronize()

        r_nccl = cuda_timer(
            lambda: _run(rs_nccl, ag_nccl),
            label="NCCL fused row fwd",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_sdma = cuda_timer(
            lambda: _run(rs_sdma, ag_sdma),
            label="SDMA fused row fwd",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_nccl.avg_ms / max(r_sdma.avg_ms, 1e-6)
        r_sdma.extra["speedup_vs_nccl"] = round(speedup, 2)

        if self.rank == 0:
            print_report_with_table(
                f"Fused Row Fwd: NCCL vs SDMA (world={self.world})",
                [r_nccl, r_sdma],
            )
            print_bench_warnings(result=r_sdma, speedup=speedup)

    def test_nccl_vs_sdma_fused_column_fwd_bwd(self):
        from lumen.modules.comm_overlap import fused_column_parallel_forward

        S_local = (B * S) // self.world
        w = torch.randn(FFN // self.world, H, device=self.device, dtype=torch.bfloat16)
        w_param = torch.nn.Parameter(w.clone())
        (ag_nccl, rs_nccl), (ag_sdma, rs_sdma) = self._make_backends()

        def _run(ag, rs):
            x = torch.randn(S_local, H, device=self.device, dtype=torch.bfloat16, requires_grad=True)
            y = fused_column_parallel_forward(
                x,
                w_param,
                None,
                ag,
                rs,
                w_param,
                False,
                False,
                None,
                False,
            )
            y.sum().backward()
            return x.grad

        for _ in range(3):
            _run(ag_nccl, rs_nccl)
            _run(ag_sdma, rs_sdma)
        torch.cuda.synchronize()

        r_nccl = cuda_timer(
            lambda: _run(ag_nccl, rs_nccl),
            label="NCCL fused fwd+bwd",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_sdma = cuda_timer(
            lambda: _run(ag_sdma, rs_sdma),
            label="SDMA fused fwd+bwd",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_nccl.avg_ms / max(r_sdma.avg_ms, 1e-6)
        r_sdma.extra["speedup_vs_nccl"] = round(speedup, 2)

        if self.rank == 0:
            print_report_with_table(
                f"Fused Column Fwd+Bwd: NCCL vs SDMA (world={self.world})",
                [r_nccl, r_sdma],
            )
            print_bench_warnings(result=r_sdma, speedup=speedup)


@_DIST
class TestFusedPipelineBackwardOverlap:
    """Isolate backward overlap: dgrad+RS and AG+dgrad measured separately."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    def test_column_backward_dgrad_rs_overlap(self):
        """Column-parallel backward: dgrad + RS overlap."""
        from lumen.modules.comm_overlap import (
            NcclCommBackend,
            PipelinedAllgatherGemm,
            PipelinedGemmReduceScatter,
            fused_column_parallel_forward,
        )

        S_local = (B * S) // self.world
        w = torch.randn(FFN // self.world, H, device=self.device, dtype=torch.bfloat16)
        w_param = torch.nn.Parameter(w.clone())
        backend = NcclCommBackend(dist.group.WORLD)
        pipeline_ag = PipelinedAllgatherGemm(num_chunks=4, comm=backend)
        pipeline_rs = PipelinedGemmReduceScatter(num_chunks=4, comm=backend)

        def _seq_bwd():
            x = torch.randn(S_local, H, device=self.device, dtype=torch.bfloat16, requires_grad=True)
            gathered = _distributed_allgather(x)
            y = F.linear(gathered, w)
            y.sum().backward()
            return x.grad

        def _fused_bwd():
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
            _seq_bwd()
            _fused_bwd()
        torch.cuda.synchronize()

        r_seq = cuda_timer(
            _seq_bwd,
            label="sequential column bwd",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_fused = cuda_timer(
            _fused_bwd,
            label="fused column bwd",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_seq.avg_ms / max(r_fused.avg_ms, 1e-6)
        overlap_ratio = 1 - (r_fused.avg_ms / max(r_seq.avg_ms, 1e-6))
        r_fused.extra["speedup"] = round(speedup, 2)
        r_fused.extra["overlap_ratio"] = round(overlap_ratio, 3)

        if self.rank == 0:
            print_report_with_table(
                f"Column Backward Overlap (world={self.world})",
                [r_seq, r_fused],
            )
            print_overlap_summary(
                t_compute=r_seq.avg_ms,
                t_comm=0,
                t_seq=r_seq.avg_ms,
                t_ovl=r_fused.avg_ms,
                compute_label="seq bwd",
                comm_label="(fused)",
            )
            print_bench_warnings(result=r_fused, speedup=speedup, overlap_ratio=overlap_ratio)

    def test_row_backward_ag_dgrad_overlap(self):
        """Row-parallel backward: AG + dgrad overlap."""
        from lumen.modules.comm_overlap import (
            NcclCommBackend,
            PipelinedAllgatherGemm,
            PipelinedGemmReduceScatter,
            fused_row_parallel_forward,
        )

        S_full = B * S
        H_tp = H // self.world
        w = torch.randn(FFN, H_tp, device=self.device, dtype=torch.bfloat16)
        w_param = torch.nn.Parameter(w.clone())
        backend = NcclCommBackend(dist.group.WORLD)
        pipeline_rs = PipelinedGemmReduceScatter(num_chunks=4, comm=backend)
        pipeline_ag = PipelinedAllgatherGemm(num_chunks=4, comm=backend)

        def _seq_bwd():
            x = torch.randn(S_full, H_tp, device=self.device, dtype=torch.bfloat16, requires_grad=True)
            y = _reduce_scatter(F.linear(x, w), self.world, self.device)
            y.sum().backward()
            return x.grad

        def _fused_bwd():
            x = torch.randn(S_full, H_tp, device=self.device, dtype=torch.bfloat16, requires_grad=True)
            y = fused_row_parallel_forward(
                x,
                w_param,
                pipeline_rs,
                pipeline_ag,
                w_param,
                False,
                False,
                None,
            )
            y.sum().backward()
            return x.grad

        for _ in range(3):
            _seq_bwd()
            _fused_bwd()
        torch.cuda.synchronize()

        r_seq = cuda_timer(
            _seq_bwd,
            label="sequential row bwd",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_fused = cuda_timer(
            _fused_bwd,
            label="fused row bwd",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_seq.avg_ms / max(r_fused.avg_ms, 1e-6)
        overlap_ratio = 1 - (r_fused.avg_ms / max(r_seq.avg_ms, 1e-6))
        r_fused.extra["speedup"] = round(speedup, 2)
        r_fused.extra["overlap_ratio"] = round(overlap_ratio, 3)

        if self.rank == 0:
            print_report_with_table(
                f"Row Backward Overlap (world={self.world})",
                [r_seq, r_fused],
            )
            print_overlap_summary(
                t_compute=r_seq.avg_ms,
                t_comm=0,
                t_seq=r_seq.avg_ms,
                t_ovl=r_fused.avg_ms,
                compute_label="seq bwd",
                comm_label="(fused)",
            )
            print_bench_warnings(result=r_fused, speedup=speedup, overlap_ratio=overlap_ratio)
