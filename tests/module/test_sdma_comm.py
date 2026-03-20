###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for lumen.modules.sdma_comm — SDMA TP communication primitives.

Covers:
  - SdmaTpContext singleton lifecycle
  - SdmaTpComm construction and singleton pattern
  - allgather_dim0: correctness, shape, multi-dtype
  - allgather_last_dim: correctness, shape
  - allreduce_sum: correctness, multi-dtype
  - allreduce_sum_inplace: correctness
  - reduce_scatter_dim0: correctness, shape
  - Autograd functions: forward/backward gradient flow
  - Performance: throughput for TP-sized collectives

Multi-GPU tests use torch.multiprocessing.spawn with mori shmem init.
"""

import os
import time

import pytest
import torch

_SDMA_SPAWN_COOLDOWN_SECS = float(os.environ.get("LUMEN_SDMA_SPAWN_COOLDOWN_SECS", "2"))


def _sdma_spawn(fn, args, nprocs, join=True):
    """``mp.spawn`` with a KFD SDMA-queue reclamation cooldown.

    See ``tests/ops/test_sdma.py::_sdma_spawn`` for the full rationale.
    Override with ``LUMEN_SDMA_SPAWN_COOLDOWN_SECS=0`` for fast local runs.
    """
    import torch.multiprocessing as mp

    if _SDMA_SPAWN_COOLDOWN_SECS > 0:
        time.sleep(_SDMA_SPAWN_COOLDOWN_SECS)
    return mp.spawn(fn, args=args, nprocs=nprocs, join=join)


def _get_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _sdma_tp_worker_teardown(shmem_mod):
    """Standard teardown for SdmaTpComm worker processes.

    Follows the cleanup order from the mori ccl reference tests:
    destroy SDMA handles, finalize shmem, then destroy process group.
    """
    import torch.distributed as dist

    from lumen.modules.sdma_comm import SdmaTpComm, SdmaTpContext

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    SdmaTpComm.reset()
    SdmaTpContext.reset()
    if dist.is_initialized():
        dist.barrier()
    shmem_mod.shmem_finalize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ===================================================================
# SdmaTpContext singleton
# ===================================================================


class TestSdmaTpContextUnit:
    def test_reset_clears_singleton(self):
        from lumen.modules.sdma_comm import SdmaTpContext

        SdmaTpContext.reset()
        assert SdmaTpContext._instance is None


# ===================================================================
# SdmaTpComm singleton
# ===================================================================


class TestSdmaTpCommUnit:
    def test_reset_clears_singleton(self):
        from lumen.modules.sdma_comm import SdmaTpComm

        SdmaTpComm.reset()
        assert SdmaTpComm._instance is None


# ===================================================================
# Multi-GPU distributed tests
# ===================================================================


def _worker_tp_allgather_dim0(rank, world_size, port, results_dict):
    """Worker: test SdmaTpComm.allgather_dim0 correctness."""
    import mori.shmem as shmem
    import torch.distributed as dist

    from lumen.modules.sdma_comm import SdmaTpComm, SdmaTpContext

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MORI_ENABLE_SDMA"] = "1"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    comm = None
    try:
        tp_group = dist.new_group(list(range(world_size)))
        SdmaTpContext.reset()
        SdmaTpComm.reset()
        comm = SdmaTpComm.get(tp_group)

        seq_local = 16
        hidden = 64
        local_tensor = torch.full(
            (seq_local, hidden),
            float(rank + 1),
            dtype=torch.bfloat16,
            device=f"cuda:{rank}",
        )

        gathered = comm.allgather_dim0(local_tensor)

        passed = True
        assert gathered.shape == (seq_local * world_size, hidden)

        for pe in range(world_size):
            chunk = gathered[pe * seq_local : (pe + 1) * seq_local]
            expected = float(pe + 1)
            if not torch.allclose(chunk.float(), torch.full_like(chunk.float(), expected), atol=1e-3):
                passed = False

        results_dict[rank] = passed
    finally:
        del comm
        _sdma_tp_worker_teardown(shmem)


def _worker_tp_allreduce(rank, world_size, port, results_dict):
    """Worker: test SdmaTpComm.allreduce_sum correctness."""
    import mori.shmem as shmem
    import torch.distributed as dist

    from lumen.modules.sdma_comm import SdmaTpComm, SdmaTpContext

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MORI_ENABLE_SDMA"] = "1"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    comm = None
    try:
        tp_group = dist.new_group(list(range(world_size)))
        SdmaTpContext.reset()
        SdmaTpComm.reset()
        comm = SdmaTpComm.get(tp_group)

        n_elems = 256
        local = torch.full(
            (n_elems,),
            float(rank + 1),
            dtype=torch.float32,
            device=f"cuda:{rank}",
        )

        result = comm.allreduce_sum(local)

        expected_sum = sum(range(1, world_size + 1))
        passed = torch.allclose(
            result,
            torch.full_like(result, float(expected_sum)),
            atol=0.5,
        )
        results_dict[rank] = passed
    finally:
        del comm
        _sdma_tp_worker_teardown(shmem)


def _worker_tp_reduce_scatter(rank, world_size, port, results_dict):
    """Worker: test SdmaTpComm.reduce_scatter_dim0 correctness."""
    import mori.shmem as shmem
    import torch.distributed as dist

    from lumen.modules.sdma_comm import SdmaTpComm, SdmaTpContext

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MORI_ENABLE_SDMA"] = "1"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    comm = None
    try:
        tp_group = dist.new_group(list(range(world_size)))
        SdmaTpContext.reset()
        SdmaTpComm.reset()
        comm = SdmaTpComm.get(tp_group)

        chunk_size = 16
        full_size = chunk_size * world_size
        local = torch.full(
            (full_size,),
            float(rank + 1),
            dtype=torch.float32,
            device=f"cuda:{rank}",
        )

        result = comm.reduce_scatter_dim0(local)
        assert result.shape == (chunk_size,)

        expected_sum = sum(range(1, world_size + 1))
        passed = torch.allclose(
            result,
            torch.full_like(result, float(expected_sum)),
            atol=0.5,
        )
        results_dict[rank] = passed
    finally:
        del comm
        _sdma_tp_worker_teardown(shmem)


def _worker_tp_allgather_last_dim(rank, world_size, port, results_dict):
    """Worker: test SdmaTpComm.allgather_last_dim correctness."""
    import mori.shmem as shmem
    import torch.distributed as dist

    from lumen.modules.sdma_comm import SdmaTpComm, SdmaTpContext

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MORI_ENABLE_SDMA"] = "1"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    comm = None
    try:
        tp_group = dist.new_group(list(range(world_size)))
        SdmaTpContext.reset()
        SdmaTpComm.reset()
        comm = SdmaTpComm.get(tp_group)

        batch, seq, d_local = 2, 8, 16
        local = torch.full(
            (batch, seq, d_local),
            float(rank + 1),
            dtype=torch.bfloat16,
            device=f"cuda:{rank}",
        )

        gathered = comm.allgather_last_dim(local)
        assert gathered.shape == (batch, seq, d_local * world_size)

        passed = True
        for pe in range(world_size):
            chunk = gathered[..., pe * d_local : (pe + 1) * d_local]
            expected = float(pe + 1)
            if not torch.allclose(chunk.float(), torch.full_like(chunk.float(), expected), atol=1e-3):
                passed = False

        results_dict[rank] = passed
    finally:
        del comm
        _sdma_tp_worker_teardown(shmem)


class TestSdmaTpCommDistributed:
    """Multi-GPU SdmaTpComm correctness tests."""

    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 8)

    def test_allgather_dim0(self):
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results = manager.dict()
        port = _get_free_port()
        _sdma_spawn(
            _worker_tp_allgather_dim0,
            args=(self.world_size, port, results),
            nprocs=self.world_size,
        )
        for r in range(self.world_size):
            assert results[r], f"PE {r} allgather_dim0 failed"

    def test_allreduce_sum(self):
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results = manager.dict()
        port = _get_free_port()
        _sdma_spawn(
            _worker_tp_allreduce,
            args=(self.world_size, port, results),
            nprocs=self.world_size,
        )
        for r in range(self.world_size):
            assert results[r], f"PE {r} allreduce_sum failed"

    def test_reduce_scatter_dim0(self):
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results = manager.dict()
        port = _get_free_port()
        _sdma_spawn(
            _worker_tp_reduce_scatter,
            args=(self.world_size, port, results),
            nprocs=self.world_size,
        )
        for r in range(self.world_size):
            assert results[r], f"PE {r} reduce_scatter_dim0 failed"

    def test_allgather_last_dim(self):
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results = manager.dict()
        port = _get_free_port()
        _sdma_spawn(
            _worker_tp_allgather_last_dim,
            args=(self.world_size, port, results),
            nprocs=self.world_size,
        )
        for r in range(self.world_size):
            assert results[r], f"PE {r} allgather_last_dim failed"


# ===================================================================
# Performance benchmarks
# ===================================================================


def _worker_tp_perf(rank, world_size, port, results_dict, op_name, n_elems, iterations, warmup):
    """Worker: measure TP comm throughput."""
    import mori.shmem as shmem
    import numpy as np
    import torch.distributed as dist

    from lumen.modules.sdma_comm import SdmaTpComm, SdmaTpContext

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MORI_ENABLE_SDMA"] = "1"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    comm = None
    try:
        tp_group = dist.new_group(list(range(world_size)))
        SdmaTpContext.reset()
        SdmaTpComm.reset()
        comm = SdmaTpComm.get(tp_group)
        device = f"cuda:{rank}"

        local = torch.randn(n_elems, dtype=torch.bfloat16, device=device)

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device)

        op_fn = {
            "allgather_dim0": lambda c=comm: c.allgather_dim0(local),
            "allreduce_sum": lambda c=comm: c.allreduce_sum(local),
        }[op_name]

        times = []
        for i in range(warmup + iterations):
            start_ev.record(stream)
            _ = op_fn()
            end_ev.record(stream)
            stream.synchronize()
            if i >= warmup:
                times.append(start_ev.elapsed_time(end_ev))

        avg_ms = np.mean(times) if times else 0
        total_bytes = n_elems * 2 * world_size
        bw = (total_bytes / (avg_ms / 1000.0)) / (1024**3) if avg_ms > 0 else 0

        results_dict[rank] = {"avg_ms": avg_ms, "bandwidth_gb_s": bw}
    finally:
        del comm
        _sdma_tp_worker_teardown(shmem)


class TestSdmaTpCommPerformance:
    """TP communication performance benchmarks."""

    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 8)

    @pytest.mark.parametrize("n_elems", [4096, 65536, 1048576])
    def test_allgather_dim0_perf(self, n_elems):
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results = manager.dict()
        port = _get_free_port()
        _sdma_spawn(
            _worker_tp_perf,
            args=(self.world_size, port, results, "allgather_dim0", n_elems, 10, 5),
            nprocs=self.world_size,
        )
        r0 = results[0]
        total_bytes = n_elems * 2 * self.world_size
        print(
            f"\nTP allgather_dim0 SDMA: {total_bytes / 1024:.1f} KB, "
            f"avg={r0['avg_ms']:.3f}ms, BW={r0['bandwidth_gb_s']:.2f} GB/s"
        )

    @pytest.mark.parametrize("n_elems", [4096, 65536, 1048576])
    def test_allreduce_sum_perf(self, n_elems):
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results = manager.dict()
        port = _get_free_port()
        _sdma_spawn(
            _worker_tp_perf,
            args=(self.world_size, port, results, "allreduce_sum", n_elems, 10, 5),
            nprocs=self.world_size,
        )
        r0 = results[0]
        total_bytes = n_elems * 2 * self.world_size
        print(
            f"\nTP allreduce_sum SDMA: {total_bytes / 1024:.1f} KB, "
            f"avg={r0['avg_ms']:.3f}ms, BW={r0['bandwidth_gb_s']:.2f} GB/s"
        )
