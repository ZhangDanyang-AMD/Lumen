###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Benchmark — End-to-end transformer layer fusion strategies.

Measures the latency of a single transformer layer (column-parallel +
row-parallel) under three schemes:

  * **Naive**: Sequential AG -> GEMM_up -> GEMM_down -> RS.
  * **Fused NCCL**: Pipelined comm-GEMM overlap with NCCL backend.
  * **Fused SDMA**: Pipelined comm-GEMM overlap with SDMA backend.

Also includes a TP scaling sweep (TP=2, 4, 8) to show how fusion
benefits change with parallelism width.

Requires >= 2 GPUs with NCCL.  TP scaling requires 8 GPUs.

Run E2E layer benchmarks::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_e2e_fusion.py -v -s -k TransformerLayer
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_e2e_fusion.py -v -s -k TransformerLayer

Run TP scaling sweep (requires 8 GPUs)::

    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_e2e_fusion.py -v -s -k TPScaling

Run all::

    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_e2e_fusion.py -v -s
"""

from __future__ import annotations

import os
from typing import List

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from benchmarks.bench_utils import (
    compute_bandwidth_gb_s,
    cuda_timer,
    print_bandwidth_summary,
    print_bench_warnings,
    print_report_with_table,
)

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
        dist.all_gather(chunks, tensor, group=group)
        return torch.cat(chunks, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        shard = grad_output.shape[0] // ctx.world
        out = torch.empty(shard, *grad_output.shape[1:], device=grad_output.device, dtype=grad_output.dtype)
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
        chunks = [torch.empty_like(grad_output) for _ in range(world)]
        dist.all_gather(chunks, grad_output, group=ctx.group)
        return torch.cat(chunks, dim=0), None, None, None


def _distributed_allgather(tensor, group=None):
    return _AllGatherFunc.apply(tensor, group)


def _reduce_scatter(tensor, world, device, group=None):
    return _ReduceScatterFunc.apply(tensor, world, device, group)


@_DIST
class TestE2ETransformerLayerFusion:
    """E2E single transformer layer: Naive vs Fused NCCL (vs Fused SDMA)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    def _make_layer(self):
        w_up = torch.randn(FFN // self.world, H, device=self.device, dtype=torch.bfloat16)
        w_down = torch.randn(H, FFN // self.world, device=self.device, dtype=torch.bfloat16)
        return w_up, w_down

    def _naive_fwd(self, x_local, w_up, w_down):
        gathered = _distributed_allgather(x_local)
        up = F.linear(gathered, w_up)
        down = F.linear(up, w_down)
        return _reduce_scatter(down, self.world, self.device)

    def _make_pipelines(self, backend):
        from lumen.modules.comm_overlap import (
            PipelinedAllgatherGemm,
            PipelinedGemmReduceScatter,
        )

        ag = PipelinedAllgatherGemm(num_chunks=4, comm=backend)
        rs = PipelinedGemmReduceScatter(num_chunks=4, comm=backend)
        return ag, rs

    def _fused_fwd(self, x_local, w_up, w_down, ag, rs, w_up_param, w_down_param, deferred=None):
        from lumen.modules.comm_overlap import (
            fused_column_parallel_forward,
            fused_row_parallel_forward,
        )

        delay = deferred is not None
        up = fused_column_parallel_forward(
            x_local,
            w_up,
            None,
            ag,
            rs,
            w_up_param,
            False,
            delay,
            deferred,
            False,
        )
        return fused_row_parallel_forward(
            up,
            w_down,
            rs,
            ag,
            w_down_param,
            False,
            delay,
            deferred,
        )

    def _run_comparison(self, fwd_only=False, bwd_only=False):
        from lumen.modules.comm_overlap import NcclCommBackend
        from lumen.modules.parallel_linear import _DeferredWgrad

        S_local = (B * S) // self.world
        w_up, w_down = self._make_layer()
        w_up_param = torch.nn.Parameter(w_up.clone())
        w_down_param = torch.nn.Parameter(w_down.clone())
        nccl = NcclCommBackend(dist.group.WORLD)
        ag, rs = self._make_pipelines(nccl)
        deferred = _DeferredWgrad()

        ag_bytes = B * S * H * 2
        rs_bytes = B * S * H * 2
        needs_grad = not fwd_only

        if bwd_only:
            x_naive = torch.randn(S_local, H, device=self.device, dtype=torch.bfloat16, requires_grad=True)
            y_naive_saved = self._naive_fwd(x_naive, w_up_param, w_down_param)

            x_fused = torch.randn(S_local, H, device=self.device, dtype=torch.bfloat16, requires_grad=True)
            y_fused_saved = self._fused_fwd(
                x_fused,
                w_up,
                w_down,
                ag,
                rs,
                w_up_param,
                w_down_param,
                deferred=deferred,
            )

            def _naive():
                x_naive.grad = None
                w_up_param.grad = None
                w_down_param.grad = None
                y_naive_saved.sum().backward(retain_graph=True)

            def _fused_nccl():
                x_fused.grad = None
                w_up_param.grad = None
                w_down_param.grad = None
                y_fused_saved.sum().backward(retain_graph=True)
                deferred.execute()

        else:

            def _naive():
                w_up_param.grad = None
                w_down_param.grad = None
                x = torch.randn(S_local, H, device=self.device, dtype=torch.bfloat16, requires_grad=needs_grad)
                y = self._naive_fwd(x, w_up_param, w_down_param)
                if needs_grad:
                    y.sum().backward()
                return y

            def _fused_nccl():
                w_up_param.grad = None
                w_down_param.grad = None
                x = torch.randn(S_local, H, device=self.device, dtype=torch.bfloat16, requires_grad=needs_grad)
                y = self._fused_fwd(
                    x, w_up, w_down, ag, rs, w_up_param, w_down_param, deferred=deferred if needs_grad else None
                )
                if needs_grad:
                    y.sum().backward()
                    deferred.execute()
                return y

        for _ in range(3):
            _naive()
        torch.cuda.synchronize()
        for _ in range(3):
            _fused_nccl()
        torch.cuda.synchronize()

        label_suffix = "fwd" if fwd_only else ("bwd-only" if bwd_only else "fwd+bwd")

        r_naive = cuda_timer(
            _naive,
            label=f"naive {label_suffix}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_fused = cuda_timer(
            _fused_nccl,
            label=f"fused NCCL {label_suffix}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_naive.avg_ms / max(r_fused.avg_ms, 1e-6)
        r_fused.extra["speedup"] = round(speedup, 2)

        results = [r_naive, r_fused]

        if _sdma_available() and not bwd_only:
            from lumen.modules.comm_overlap import SdmaCommBackend
            from lumen.modules.sdma_comm import SdmaTpComm

            os.environ["MORI_ENABLE_SDMA"] = "1"
            sdma_comm = SdmaTpComm(dist.group.WORLD)
            sdma = SdmaCommBackend(sdma_comm)
            ag_s, rs_s = self._make_pipelines(sdma)

            def _fused_sdma():
                w_up_param.grad = None
                w_down_param.grad = None
                x = torch.randn(S_local, H, device=self.device, dtype=torch.bfloat16, requires_grad=needs_grad)
                y = self._fused_fwd(
                    x, w_up, w_down, ag_s, rs_s, w_up_param, w_down_param, deferred=deferred if needs_grad else None
                )
                if needs_grad:
                    y.sum().backward()
                    deferred.execute()
                return y

            for _ in range(3):
                _fused_sdma()
            torch.cuda.synchronize()

            r_sdma = cuda_timer(
                _fused_sdma,
                label=f"fused SDMA {label_suffix}",
                warmup=_WARMUP,
                iters=_ITERS,
                trim_pct=_TRIM,
                dist_barrier=True,
            )
            sdma_speedup = r_naive.avg_ms / max(r_sdma.avg_ms, 1e-6)
            r_sdma.extra["speedup"] = round(sdma_speedup, 2)
            results.append(r_sdma)

        if self.rank == 0:
            print_report_with_table(
                f"E2E Transformer Layer {label_suffix} (world={self.world})",
                results,
            )
            print_bandwidth_summary(label="AG", bytes_transferred=ag_bytes, time_ms=r_fused.avg_ms)
            print_bandwidth_summary(label="RS", bytes_transferred=rs_bytes, time_ms=r_fused.avg_ms)
            print_bench_warnings(result=r_fused, speedup=speedup)

    def test_single_layer_fwd_bwd(self):
        self._run_comparison(fwd_only=False, bwd_only=False)

    def test_single_layer_fwd_only(self):
        self._run_comparison(fwd_only=True, bwd_only=False)

    def test_single_layer_bwd_only(self):
        self._run_comparison(fwd_only=False, bwd_only=True)


@_SDMA_DIST
class TestE2ETransformerLayerSdmaOnly:
    """Isolated SDMA measurement — no NCCL comparison overhead."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        os.environ["MORI_ENABLE_SDMA"] = "1"
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    def _run_sdma(self, fwd_only=False, bwd_only=False):
        from lumen.modules.comm_overlap import (
            PipelinedAllgatherGemm,
            PipelinedGemmReduceScatter,
            SdmaCommBackend,
            fused_column_parallel_forward,
            fused_row_parallel_forward,
        )
        from lumen.modules.parallel_linear import _DeferredWgrad
        from lumen.modules.sdma_comm import SdmaTpComm

        S_local = (B * S) // self.world
        w_up = torch.randn(FFN // self.world, H, device=self.device, dtype=torch.bfloat16)
        w_down = torch.randn(H, FFN // self.world, device=self.device, dtype=torch.bfloat16)
        w_up_param = torch.nn.Parameter(w_up.clone())
        w_down_param = torch.nn.Parameter(w_down.clone())
        deferred = _DeferredWgrad()

        sdma_comm = SdmaTpComm(dist.group.WORLD)
        sdma = SdmaCommBackend(sdma_comm)
        ag = PipelinedAllgatherGemm(num_chunks=4, comm=sdma)
        rs = PipelinedGemmReduceScatter(num_chunks=4, comm=sdma)

        ag_bytes = B * S * H * 2
        rs_bytes = B * S * H * 2
        needs_grad = not fwd_only
        delay = needs_grad

        if bwd_only:
            x_saved = torch.randn(S_local, H, device=self.device, dtype=torch.bfloat16, requires_grad=True)
            up = fused_column_parallel_forward(
                x_saved,
                w_up,
                None,
                ag,
                rs,
                w_up_param,
                False,
                True,
                deferred,
                False,
            )
            y_saved = fused_row_parallel_forward(
                up,
                w_down,
                rs,
                ag,
                w_down_param,
                False,
                True,
                deferred,
            )

            def _fused():
                x_saved.grad = None
                w_up_param.grad = None
                w_down_param.grad = None
                y_saved.sum().backward(retain_graph=True)
                deferred.execute()

        else:

            def _fused():
                w_up_param.grad = None
                w_down_param.grad = None
                x = torch.randn(S_local, H, device=self.device, dtype=torch.bfloat16, requires_grad=needs_grad)
                up = fused_column_parallel_forward(
                    x,
                    w_up,
                    None,
                    ag,
                    rs,
                    w_up_param,
                    False,
                    delay,
                    deferred if delay else None,
                    False,
                )
                y = fused_row_parallel_forward(
                    up,
                    w_down,
                    rs,
                    ag,
                    w_down_param,
                    False,
                    delay,
                    deferred if delay else None,
                )
                if needs_grad:
                    y.sum().backward()
                    deferred.execute()
                return y

        for _ in range(3):
            _fused()
        torch.cuda.synchronize()

        label = "fwd" if fwd_only else ("bwd-only" if bwd_only else "fwd+bwd")
        r = cuda_timer(
            _fused,
            label=f"SDMA only {label}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        if self.rank == 0:
            print_report_with_table(
                f"E2E SDMA-Only {label} (world={self.world})",
                [r],
            )
            print_bandwidth_summary(label="AG", bytes_transferred=ag_bytes, time_ms=r.avg_ms)
            print_bandwidth_summary(label="RS", bytes_transferred=rs_bytes, time_ms=r.avg_ms)
            print_bench_warnings(result=r)

    def test_single_layer_fwd_bwd(self):
        self._run_sdma(fwd_only=False)

    def test_single_layer_fwd_only(self):
        self._run_sdma(fwd_only=True)

    def test_single_layer_bwd_only(self):
        self._run_sdma(bwd_only=True)


@_DIST
class TestTPScaling:
    """TP scaling sweep: measure fusion benefit at TP=2, 4, 8."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    @pytest.mark.skipif(
        "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) < 8,
        reason="Requires 8 GPUs for TP scaling sweep",
    )
    def test_tp_scaling_sweep(self):
        from lumen.modules.comm_overlap import (
            NcclCommBackend,
            PipelinedAllgatherGemm,
            PipelinedGemmReduceScatter,
            fused_column_parallel_forward,
            fused_row_parallel_forward,
        )
        from lumen.modules.parallel_linear import _DeferredWgrad

        tp_sizes = [2, 4, 8]
        sweep_results: List[dict] = []

        for tp_size in tp_sizes:
            group_start = (self.rank // tp_size) * tp_size
            group_ranks = list(range(group_start, group_start + tp_size))
            tp_group = dist.new_group(group_ranks)

            S_local = (B * S) // tp_size
            w_up = torch.randn(FFN // tp_size, H, device=self.device, dtype=torch.bfloat16)
            w_down = torch.randn(H, FFN // tp_size, device=self.device, dtype=torch.bfloat16)
            w_up_param = torch.nn.Parameter(w_up.clone())
            w_down_param = torch.nn.Parameter(w_down.clone())
            deferred = _DeferredWgrad()

            backend = NcclCommBackend(tp_group)
            ag = PipelinedAllgatherGemm(num_chunks=4, comm=backend)
            rs = PipelinedGemmReduceScatter(num_chunks=4, comm=backend)

            ag_bytes = B * S * H * 2
            rs_bytes = B * S * H * 2

            def _naive(s_local=S_local, w_u=w_up_param, w_d=w_down_param, tp_g=tp_group, tp_s=tp_size):
                w_u.grad = None
                w_d.grad = None
                x = torch.randn(s_local, H, device=self.device, dtype=torch.bfloat16, requires_grad=True)
                gathered = _distributed_allgather(x, group=tp_g)
                up = F.linear(gathered, w_u)
                down = F.linear(up, w_d)
                y = _reduce_scatter(down, tp_s, self.device, group=tp_g)
                y.sum().backward()
                return y

            def _fused(
                s_local=S_local,
                w_u=w_up,
                w_d=w_down,
                _ag=ag,
                _rs=rs,
                w_u_p=w_up_param,
                w_d_p=w_down_param,
                _def=deferred,
            ):
                w_u_p.grad = None
                w_d_p.grad = None
                x = torch.randn(s_local, H, device=self.device, dtype=torch.bfloat16, requires_grad=True)
                up = fused_column_parallel_forward(
                    x,
                    w_u,
                    None,
                    _ag,
                    _rs,
                    w_u_p,
                    False,
                    True,
                    _def,
                    False,
                )
                y = fused_row_parallel_forward(
                    up,
                    w_d,
                    _rs,
                    _ag,
                    w_d_p,
                    False,
                    True,
                    _def,
                )
                y.sum().backward()
                _def.execute()
                return y

            for _ in range(3):
                _naive()
            torch.cuda.synchronize()
            for _ in range(3):
                _fused()
            torch.cuda.synchronize()

            r_naive = cuda_timer(
                _naive,
                label=f"TP={tp_size} naive",
                warmup=_WARMUP,
                iters=_ITERS,
                trim_pct=_TRIM,
                dist_barrier=True,
            )
            r_fused = cuda_timer(
                _fused,
                label=f"TP={tp_size} fused",
                warmup=_WARMUP,
                iters=_ITERS,
                trim_pct=_TRIM,
                dist_barrier=True,
            )

            speedup = r_naive.avg_ms / max(r_fused.avg_ms, 1e-6)
            ag_bw = compute_bandwidth_gb_s(ag_bytes, r_fused.avg_ms)
            rs_bw = compute_bandwidth_gb_s(rs_bytes, r_fused.avg_ms)

            sweep_results.append(
                {
                    "tp": tp_size,
                    "naive_ms": r_naive.avg_ms,
                    "fused_ms": r_fused.avg_ms,
                    "speedup": speedup,
                    "ag_bw": ag_bw,
                    "rs_bw": rs_bw,
                }
            )

            dist.barrier()

        if self.rank == 0:
            print("\n" + "=" * 72)
            print("TP Scaling Summary (Llama 8B shapes, fwd+bwd)")
            print("=" * 72)
            print(
                f"{'TP':>4s}  {'Naive ms':>10s}  {'Fused ms':>10s}  " f"{'Speedup':>8s}  {'AG BW':>10s}  {'RS BW':>10s}"
            )
            print("-" * 72)
            for row in sweep_results:
                print(
                    f"{row['tp']:>4d}  {row['naive_ms']:>10.3f}  {row['fused_ms']:>10.3f}  "
                    f"{row['speedup']:>7.2f}x  {row['ag_bw']:>8.1f} GB/s  {row['rs_bw']:>8.1f} GB/s"
                )
            print("=" * 72)
            for row in sweep_results:
                if row["speedup"] < 1.0:
                    print(f"  \u26a0 WARNING: TP={row['tp']} — fused is slower than naive " f"({row['speedup']:.2f}x)")
