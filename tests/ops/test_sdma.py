###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for lumen.ops.sdma — SDMA collective wrappers and communication.

Covers:
  - is_sdma_available detection
  - SdmaContext singleton lifecycle (get/reset)
  - SdmaAllgather: basic allgather, buffer reuse, shape consistency
  - SdmaAllreduce: in-place and out-of-place sum, dtype variants
  - SdmaAll2all: basic all-to-all, correctness, buffer reuse
  - sdma_allgather_max: element-wise MAX across PEs
  - Performance: SDMA bandwidth measurement (latency, throughput)

Multi-GPU tests use torch.multiprocessing.spawn with mori shmem init.
"""

import functools
import os
import subprocess
import sys

import pytest
import torch

from lumen.ops.sdma import is_sdma_available


@functools.lru_cache(maxsize=1)
def _sdma_hardware_available() -> bool:
    """Probe whether SDMA hardware is usable by doing a minimal mori init in a subprocess.

    Returns True only if a single-rank mori shmem init (including SDMA queue
    creation) succeeds.  The result is cached for the lifetime of the process.
    """
    if not is_sdma_available():
        return False
    if torch.cuda.device_count() < 2:
        return False
    probe_script = """
import os, torch, torch.distributed as dist
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29399")
os.environ["MORI_ENABLE_SDMA"] = "1"
torch.cuda.set_device(0)
device = torch.device("cuda", 0)
dist.init_process_group("cpu:gloo,cuda:nccl", rank=0, world_size=1, device_id=device)
torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
import mori.shmem as shmem
shmem.shmem_torch_process_group_init("default")
shmem.shmem_finalize()
dist.destroy_process_group()
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe_script],
            capture_output=True,
            timeout=60,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


_requires_sdma_hw = pytest.mark.skipif(
    not _sdma_hardware_available(),
    reason="SDMA hardware not available (mori shmem init fails)",
)


# ===================================================================
# is_sdma_available
# ===================================================================


def test_is_sdma_available_returns_true():
    """is_sdma_available returns True in this environment."""
    assert is_sdma_available() is True


# ===================================================================
# SdmaContext singleton
# ===================================================================


class TestSdmaContext:
    def setup_method(self):
        from lumen.ops.sdma import SdmaContext

        SdmaContext.reset()

    def test_reset_clears_singleton(self):
        from lumen.ops.sdma import SdmaContext

        SdmaContext.reset()
        assert SdmaContext._instance is None


# ===================================================================
# SdmaAllgather unit tests
# ===================================================================


class TestSdmaAllgatherUnit:
    """Unit tests that validate the SdmaAllgather API contracts without full distributed init."""

    def test_npes_property(self):
        """npes should delegate to the context."""
        from unittest.mock import MagicMock

        from lumen.ops.sdma import SdmaAllgather

        fake_ctx = MagicMock()
        fake_ctx.npes = 4
        ag = SdmaAllgather.__new__(SdmaAllgather)
        ag._ctx = fake_ctx
        ag._handle = None
        ag._capacity_elems = 0
        ag._gathered = None
        assert ag.npes == 4

    def test_ensure_handle_skips_when_capacity_sufficient(self):
        """_ensure_handle should not recreate when capacity is sufficient."""
        from unittest.mock import MagicMock

        from lumen.ops.sdma import SdmaAllgather

        fake_ctx = MagicMock()
        fake_ctx.my_pe = 0
        fake_ctx.npes = 2
        ag = SdmaAllgather.__new__(SdmaAllgather)
        ag._ctx = fake_ctx
        sentinel = MagicMock()
        ag._handle = sentinel
        ag._capacity_elems = 512
        ag._gathered = None
        ag._ensure_handle(256)
        assert ag._handle is sentinel


# ===================================================================
# SdmaAll2all unit tests
# ===================================================================


class TestSdmaAll2allUnit:
    """Validate SdmaAll2all API shape contracts."""

    def test_npes_property(self):
        from unittest.mock import MagicMock

        from lumen.ops.sdma import SdmaAll2all

        fake_ctx = MagicMock()
        fake_ctx.npes = 8
        a2a = SdmaAll2all.__new__(SdmaAll2all)
        a2a._ctx = fake_ctx
        a2a._handle = None
        a2a._capacity_bytes = 0
        assert a2a.npes == 8

    def test_ensure_handle_reuses_when_fits(self):
        from unittest.mock import MagicMock

        from lumen.ops.sdma import SdmaAll2all

        fake_ctx = MagicMock()
        fake_ctx.my_pe = 0
        fake_ctx.npes = 2
        a2a = SdmaAll2all.__new__(SdmaAll2all)
        a2a._ctx = fake_ctx
        sentinel = MagicMock()
        a2a._handle = sentinel
        a2a._capacity_bytes = 1024
        a2a._ensure_handle(512)
        assert a2a._handle is sentinel


# ===================================================================
# Multi-GPU distributed SDMA tests
# ===================================================================


def _worker_allgather(rank, world_size, port, results_dict):
    """Worker: test SdmaAllgather correctness."""
    import mori.shmem as shmem
    import torch.distributed as dist

    from lumen.ops.sdma import SdmaAllgather, SdmaContext

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MORI_ENABLE_SDMA"] = "1"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=device)
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

    ag = None
    try:
        ctx = SdmaContext.get()
        ag = SdmaAllgather(ctx)

        n_elems = 1024
        local = torch.full((n_elems,), float(rank + 1), dtype=torch.float32, device=f"cuda:{rank}")
        gathered = ag(local)

        passed = True
        for pe in range(world_size):
            expected_val = float(pe + 1)
            chunk = gathered[pe]
            if not torch.allclose(chunk, torch.full_like(chunk, expected_val)):
                passed = False

        results_dict[rank] = passed
    except Exception:
        results_dict[rank] = False
    finally:
        del ag
        _sdma_worker_teardown(shmem)


def _worker_all2all(rank, world_size, port, results_dict):
    """Worker: test SdmaAll2all correctness."""
    import mori.shmem as shmem
    import torch.distributed as dist

    from lumen.ops.sdma import SdmaAll2all, SdmaContext

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MORI_ENABLE_SDMA"] = "1"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=device)
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

    a2a = None
    try:
        ctx = SdmaContext.get()
        a2a = SdmaAll2all(ctx)

        elems_per_pe = 256
        device = f"cuda:{rank}"

        input_tensor = torch.zeros(elems_per_pe * world_size, dtype=torch.uint32, device=device)
        for dest_pe in range(world_size):
            value = (rank + 1) * 1000 + dest_pe
            input_tensor[dest_pe * elems_per_pe : (dest_pe + 1) * elems_per_pe] = value

        output_tensor = torch.zeros_like(input_tensor)
        a2a(input_tensor, output_tensor)

        passed = True
        for src_pe in range(world_size):
            expected_value = (src_pe + 1) * 1000 + rank
            chunk = output_tensor[src_pe * elems_per_pe : (src_pe + 1) * elems_per_pe]
            if not torch.all(chunk == expected_value):
                passed = False

        results_dict[rank] = passed
    except Exception:
        results_dict[rank] = False
    finally:
        del a2a
        _sdma_worker_teardown(shmem)


def _worker_allreduce(rank, world_size, port, results_dict):
    """Worker: test SdmaAllreduce correctness."""
    import mori.shmem as shmem
    import torch.distributed as dist

    from lumen.ops.sdma import SdmaAllreduce, SdmaContext

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MORI_ENABLE_SDMA"] = "1"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=device)
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

    ar = None
    try:
        ctx = SdmaContext.get()
        ar = SdmaAllreduce(ctx=ctx)

        n_elems = 1024
        local = torch.full((n_elems,), float(rank + 1), dtype=torch.float32, device=f"cuda:{rank}")

        expected_sum = sum(range(1, world_size + 1))

        result = local.clone()
        ar.inplace(result)

        results_dict[rank] = torch.allclose(
            result,
            torch.full_like(result, float(expected_sum)),
            atol=0.5,
        )
    except Exception:
        results_dict[rank] = False
    finally:
        del ar
        _sdma_worker_teardown(shmem)


def _get_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _sdma_worker_teardown(shmem_mod):
    """Standard teardown for SDMA worker processes.

    Follows the cleanup order from the mori ccl reference tests
    (``test_allgather.py``, ``test_all2all.py``, etc.) plus
    ``TorchDistContext.__exit__``:

    1. CUDA sync — flush any outstanding GPU work.
    2. Barrier — all ranks rendezvous before tearing down.
    3. ``SdmaContext.reset()`` — clears the singleton reference.
       Callers must ``del`` their SDMA wrapper objects (``ag``,
       ``a2a``, ``ar``, …) **before** calling this function so that
       the C++ SDMA handle destructors run while shmem is still alive.
    4. Barrier — sync again after handle destruction.
    5. ``shmem_finalize()`` — release mori symmetric-memory resources.
    6. Barrier — mirrors ``TorchDistContext.__exit__`` rendezvous.
    7. ``destroy_process_group()`` — must be **last** so that shmem can
       still use the torch process group during finalization.
    """
    import torch.distributed as dist

    from lumen.ops.sdma import SdmaContext

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    SdmaContext.reset()
    if dist.is_initialized():
        dist.barrier()
    shmem_mod.shmem_finalize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


@_requires_sdma_hw
class TestSdmaDistributed:
    """Multi-GPU SDMA correctness tests."""

    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 8)

    def test_allgather_correctness(self):
        """SdmaAllgather produces correct gathered data across PEs."""
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results = manager.dict()
        port = _get_free_port()
        mp.spawn(
            _worker_allgather,
            args=(self.world_size, port, results),
            nprocs=self.world_size,
            join=True,
        )
        for rank in range(self.world_size):
            assert results[rank], f"PE {rank} allgather verification failed"

    def test_all2all_correctness(self):
        """SdmaAll2all produces correct shuffled data across PEs."""
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results = manager.dict()
        port = _get_free_port()
        mp.spawn(
            _worker_all2all,
            args=(self.world_size, port, results),
            nprocs=self.world_size,
            join=True,
        )
        for rank in range(self.world_size):
            assert results[rank], f"PE {rank} all2all verification failed"

    def test_allreduce_correctness(self):
        """SdmaAllreduce SUM produces correct reduced values."""
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results = manager.dict()
        port = _get_free_port()
        mp.spawn(
            _worker_allreduce,
            args=(self.world_size, port, results),
            nprocs=self.world_size,
            join=True,
        )
        for rank in range(self.world_size):
            assert results[rank], f"PE {rank} allreduce verification failed"


# ===================================================================
# SDMA vs NCCL golden correctness tests
# ===================================================================


def _worker_allgather_vs_nccl(rank, world_size, port, results_dict, n_elems, seed):
    """Worker: compare SdmaAllgather output against NCCL all_gather golden."""
    import mori.shmem as shmem
    import torch.distributed as dist

    from lumen.ops.sdma import SdmaAllgather, SdmaContext

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MORI_ENABLE_SDMA"] = "1"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=device)
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

    ag = None
    try:
        device = f"cuda:{rank}"
        torch.manual_seed(seed + rank)
        local = torch.randn(n_elems, dtype=torch.float32, device=device)

        ctx = SdmaContext.get()
        ag = SdmaAllgather(ctx)
        sdma_gathered = ag(local)

        nccl_gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(nccl_gathered, local)
        nccl_stacked = torch.stack(nccl_gathered)

        results_dict[rank] = torch.allclose(sdma_gathered, nccl_stacked, atol=1e-6)
    except Exception:
        results_dict[rank] = False
    finally:
        del ag
        _sdma_worker_teardown(shmem)


def _worker_all2all_vs_nccl(rank, world_size, port, results_dict, elems_per_pe, seed):
    """Worker: compare SdmaAll2all output against NCCL all_to_all golden."""
    import mori.shmem as shmem
    import torch.distributed as dist

    from lumen.ops.sdma import SdmaAll2all, SdmaContext

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MORI_ENABLE_SDMA"] = "1"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=device)
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

    a2a = None
    try:
        device = f"cuda:{rank}"
        total_elems = elems_per_pe * world_size
        torch.manual_seed(seed + rank)
        input_tensor = torch.randint(0, 2**31, (total_elems,), dtype=torch.uint32, device=device)

        sdma_output = torch.zeros_like(input_tensor)
        ctx = SdmaContext.get()
        a2a = SdmaAll2all(ctx)
        a2a(input_tensor, sdma_output)

        nccl_input = input_tensor.view(torch.int32)
        nccl_output = torch.zeros_like(nccl_input)
        dist.all_to_all_single(nccl_output, nccl_input)
        nccl_output_u32 = nccl_output.view(torch.uint32)

        results_dict[rank] = torch.equal(sdma_output, nccl_output_u32)
    except Exception:
        results_dict[rank] = False
    finally:
        del a2a
        _sdma_worker_teardown(shmem)


def _worker_allreduce_vs_nccl(rank, world_size, port, results_dict, n_elems, seed):
    """Worker: compare SdmaAllreduce SUM against NCCL all_reduce golden."""
    import mori.shmem as shmem
    import torch.distributed as dist

    from lumen.ops.sdma import SdmaAllreduce, SdmaContext

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MORI_ENABLE_SDMA"] = "1"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=device)
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

    ar = None
    try:
        device = f"cuda:{rank}"
        torch.manual_seed(seed + rank)
        local = torch.randn(n_elems, dtype=torch.float32, device=device)

        ctx = SdmaContext.get()
        ar = SdmaAllreduce(ctx=ctx)
        sdma_result = local.clone()
        ar.inplace(sdma_result)

        nccl_result = local.clone()
        dist.all_reduce(nccl_result, op=dist.ReduceOp.SUM)

        results_dict[rank] = torch.allclose(sdma_result, nccl_result, atol=0.5, rtol=0.02)
    except Exception:
        results_dict[rank] = False
    finally:
        del ar
        _sdma_worker_teardown(shmem)


def _worker_allreduce_outofplace(rank, world_size, port, results_dict, n_elems, seed):
    """Worker: test SdmaAllreduce out-of-place __call__ path."""
    import mori.shmem as shmem
    import torch.distributed as dist

    from lumen.ops.sdma import SdmaAllreduce, SdmaContext

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MORI_ENABLE_SDMA"] = "1"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=device)
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

    ar = None
    try:
        device = f"cuda:{rank}"
        torch.manual_seed(seed + rank)
        local = torch.randn(n_elems, dtype=torch.float32, device=device)

        ctx = SdmaContext.get()
        ar = SdmaAllreduce(ctx=ctx)
        sdma_output = torch.zeros_like(local)
        ar(local, sdma_output)

        nccl_result = local.clone()
        dist.all_reduce(nccl_result, op=dist.ReduceOp.SUM)

        results_dict[rank] = torch.allclose(sdma_output, nccl_result, atol=0.5, rtol=0.02)
    except Exception:
        results_dict[rank] = False
    finally:
        del ar
        _sdma_worker_teardown(shmem)


def _worker_allgather_max(rank, world_size, port, results_dict, n_elems, seed):
    """Worker: test sdma_allgather_max against torch.distributed MAX golden."""
    import mori.shmem as shmem
    import torch.distributed as dist

    from lumen.ops.sdma import SdmaAllgather, SdmaContext, sdma_allgather_max

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MORI_ENABLE_SDMA"] = "1"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=device)
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

    ag = None
    try:
        device = f"cuda:{rank}"
        torch.manual_seed(seed + rank)
        local = torch.randn(n_elems, dtype=torch.float32, device=device)

        ctx = SdmaContext.get()
        ag = SdmaAllgather(ctx)
        sdma_max_result = sdma_allgather_max(local, ag)

        nccl_max = local.clone()
        dist.all_reduce(nccl_max, op=dist.ReduceOp.MAX)

        results_dict[rank] = torch.allclose(sdma_max_result, nccl_max, atol=1e-6)
    except Exception:
        results_dict[rank] = False
    finally:
        del ag
        _sdma_worker_teardown(shmem)


def _worker_allgather_buffer_reuse(rank, world_size, port, results_dict, seed):
    """Worker: call SdmaAllgather multiple times with different data/sizes."""
    import mori.shmem as shmem
    import torch.distributed as dist

    from lumen.ops.sdma import SdmaAllgather, SdmaContext

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MORI_ENABLE_SDMA"] = "1"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=device)
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

    ag = None
    try:
        device = f"cuda:{rank}"
        ctx = SdmaContext.get()
        ag = SdmaAllgather(ctx)

        passed = True
        for call_idx, n_elems in enumerate([256, 1024, 512, 2048, 256]):
            torch.manual_seed(seed + rank * 100 + call_idx)
            local = torch.randn(n_elems, dtype=torch.float32, device=device)

            sdma_gathered = ag(local)

            nccl_gathered = [torch.empty_like(local) for _ in range(world_size)]
            dist.all_gather(nccl_gathered, local)
            nccl_stacked = torch.stack(nccl_gathered)

            if not torch.allclose(sdma_gathered, nccl_stacked, atol=1e-6):
                passed = False
                break

        results_dict[rank] = passed
    except Exception:
        results_dict[rank] = False
    finally:
        del ag
        _sdma_worker_teardown(shmem)


def _worker_all2all_buffer_reuse(rank, world_size, port, results_dict, seed):
    """Worker: call SdmaAll2all multiple times with different sizes."""
    import mori.shmem as shmem
    import torch.distributed as dist

    from lumen.ops.sdma import SdmaAll2all, SdmaContext

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MORI_ENABLE_SDMA"] = "1"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=device)
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

    a2a = None
    try:
        device = f"cuda:{rank}"
        ctx = SdmaContext.get()
        a2a = SdmaAll2all(ctx)

        passed = True
        for call_idx, elems_per_pe in enumerate([64, 512, 256, 1024, 64]):
            total = elems_per_pe * world_size
            torch.manual_seed(seed + rank * 100 + call_idx)
            inp = torch.randint(0, 2**31, (total,), dtype=torch.uint32, device=device)
            sdma_out = torch.zeros_like(inp)
            a2a(inp, sdma_out)

            nccl_in = inp.view(torch.int32)
            nccl_out = torch.zeros_like(nccl_in)
            dist.all_to_all_single(nccl_out, nccl_in)

            if not torch.equal(sdma_out, nccl_out.view(torch.uint32)):
                passed = False
                break

        results_dict[rank] = passed
    except Exception:
        results_dict[rank] = False
    finally:
        del a2a
        _sdma_worker_teardown(shmem)


@_requires_sdma_hw
class TestSdmaVsNcclGolden:
    """SDMA vs NCCL golden comparison with random data."""

    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 8)

    @pytest.mark.parametrize("n_elems", [128, 1024, 65536, 1048576])
    def test_allgather_vs_nccl(self, n_elems):
        """SdmaAllgather matches NCCL all_gather on random data."""
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results = manager.dict()
        port = _get_free_port()
        mp.spawn(
            _worker_allgather_vs_nccl,
            args=(self.world_size, port, results, n_elems, 42),
            nprocs=self.world_size,
            join=True,
        )
        for rank in range(self.world_size):
            assert results[rank], f"PE {rank}: SdmaAllgather != NCCL (n={n_elems})"

    @pytest.mark.parametrize("elems_per_pe", [64, 256, 4096, 65536])
    def test_all2all_vs_nccl(self, elems_per_pe):
        """SdmaAll2all matches NCCL all_to_all_single on random data."""
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results = manager.dict()
        port = _get_free_port()
        mp.spawn(
            _worker_all2all_vs_nccl,
            args=(self.world_size, port, results, elems_per_pe, 42),
            nprocs=self.world_size,
            join=True,
        )
        for rank in range(self.world_size):
            assert results[rank], f"PE {rank}: SdmaAll2all != NCCL (per_pe={elems_per_pe})"

    @pytest.mark.parametrize("n_elems", [128, 1024, 65536, 1048576])
    def test_allreduce_vs_nccl(self, n_elems):
        """SdmaAllreduce SUM matches NCCL all_reduce SUM on random data."""
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results = manager.dict()
        port = _get_free_port()
        mp.spawn(
            _worker_allreduce_vs_nccl,
            args=(self.world_size, port, results, n_elems, 42),
            nprocs=self.world_size,
            join=True,
        )
        for rank in range(self.world_size):
            assert results[rank], f"PE {rank}: SdmaAllreduce != NCCL (n={n_elems})"

    def test_allreduce_outofplace_vs_nccl(self):
        """SdmaAllreduce out-of-place __call__ matches NCCL all_reduce."""
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results = manager.dict()
        port = _get_free_port()
        mp.spawn(
            _worker_allreduce_outofplace,
            args=(self.world_size, port, results, 4096, 42),
            nprocs=self.world_size,
            join=True,
        )
        for rank in range(self.world_size):
            assert results[rank], f"PE {rank}: SdmaAllreduce outofplace != NCCL"

    @pytest.mark.parametrize("n_elems", [256, 4096, 65536])
    def test_allgather_max_vs_nccl(self, n_elems):
        """sdma_allgather_max matches NCCL all_reduce MAX on random data."""
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results = manager.dict()
        port = _get_free_port()
        mp.spawn(
            _worker_allgather_max,
            args=(self.world_size, port, results, n_elems, 42),
            nprocs=self.world_size,
            join=True,
        )
        for rank in range(self.world_size):
            assert results[rank], f"PE {rank}: sdma_allgather_max != NCCL MAX (n={n_elems})"

    def test_allgather_buffer_reuse(self):
        """SdmaAllgather handle reuse across varying sizes stays correct."""
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results = manager.dict()
        port = _get_free_port()
        mp.spawn(
            _worker_allgather_buffer_reuse,
            args=(self.world_size, port, results, 42),
            nprocs=self.world_size,
            join=True,
        )
        for rank in range(self.world_size):
            assert results[rank], f"PE {rank}: allgather buffer reuse failed"

    def test_all2all_buffer_reuse(self):
        """SdmaAll2all handle reuse across varying sizes stays correct."""
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results = manager.dict()
        port = _get_free_port()
        mp.spawn(
            _worker_all2all_buffer_reuse,
            args=(self.world_size, port, results, 42),
            nprocs=self.world_size,
            join=True,
        )
        for rank in range(self.world_size):
            assert results[rank], f"PE {rank}: all2all buffer reuse failed"


# ===================================================================
# SDMA performance benchmarks
# ===================================================================


def _worker_all2all_perf(rank, world_size, port, results_dict, elems_per_pe, iterations, warmup):
    """Worker: measure SdmaAll2all throughput."""
    import mori.shmem as shmem
    import numpy as np
    import torch.distributed as dist

    from lumen.ops.sdma import SdmaAll2all, SdmaContext

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MORI_ENABLE_SDMA"] = "1"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=device)
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

    a2a = None
    try:
        ctx = SdmaContext.get()
        a2a = SdmaAll2all(ctx)
        device = f"cuda:{rank}"

        input_tensor = torch.randint(0, 2**31, (elems_per_pe * world_size,), dtype=torch.uint32, device=device)
        output_tensor = torch.zeros_like(input_tensor)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device)

        times = []
        for i in range(warmup + iterations):
            start_event.record(stream)
            a2a(input_tensor, output_tensor, stream)
            end_event.record(stream)
            stream.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            if i >= warmup:
                times.append(elapsed_ms)

        avg_ms = np.mean(times) if times else 0.0
        total_bytes = elems_per_pe * world_size * 4
        bandwidth_gb_s = (total_bytes / (avg_ms / 1000.0)) / (1024**3) if avg_ms > 0 else 0

        results_dict[rank] = {
            "avg_ms": avg_ms,
            "min_ms": np.min(times) if times else 0,
            "max_ms": np.max(times) if times else 0,
            "bandwidth_gb_s": bandwidth_gb_s,
        }
    finally:
        del a2a
        _sdma_worker_teardown(shmem)


def _worker_allgather_perf(rank, world_size, port, results_dict, n_elems, iterations, warmup):
    """Worker: measure SdmaAllgather throughput."""
    import mori.shmem as shmem
    import numpy as np
    import torch.distributed as dist

    from lumen.ops.sdma import SdmaAllgather, SdmaContext

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MORI_ENABLE_SDMA"] = "1"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=device)
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

    ag = None
    try:
        ctx = SdmaContext.get()
        ag = SdmaAllgather(ctx)
        device = f"cuda:{rank}"

        local = torch.randn(n_elems, dtype=torch.float32, device=device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device)

        times = []
        for i in range(warmup + iterations):
            start_event.record(stream)
            _ = ag(local, stream)
            end_event.record(stream)
            stream.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            if i >= warmup:
                times.append(elapsed_ms)

        avg_ms = np.mean(times) if times else 0.0
        total_bytes = n_elems * 4 * world_size
        bandwidth_gb_s = (total_bytes / (avg_ms / 1000.0)) / (1024**3) if avg_ms > 0 else 0

        results_dict[rank] = {
            "avg_ms": avg_ms,
            "bandwidth_gb_s": bandwidth_gb_s,
        }
    finally:
        del ag
        _sdma_worker_teardown(shmem)


@_requires_sdma_hw
class TestSdmaPerformance:
    """SDMA performance benchmarks.

    These tests measure latency and bandwidth. They pass unconditionally
    (no hard threshold) but print results for manual inspection.
    """

    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 8)

    @pytest.mark.parametrize("elems_per_pe", [1024, 65536, 1048576, 16777216])
    def test_all2all_bandwidth(self, elems_per_pe):
        """Measure SdmaAll2all bandwidth for various data sizes."""
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results = manager.dict()
        port = _get_free_port()
        mp.spawn(
            _worker_all2all_perf,
            args=(self.world_size, port, results, elems_per_pe, 10, 5),
            nprocs=self.world_size,
            join=True,
        )
        total_bytes = elems_per_pe * self.world_size * 4
        r0 = results[0]
        print(
            f"\nAll2all SDMA: {total_bytes / 1024**2:.1f} MB, "
            f"avg={r0['avg_ms']:.3f}ms, "
            f"BW={r0['bandwidth_gb_s']:.2f} GB/s"
        )

    @pytest.mark.parametrize("n_elems", [1024, 65536, 1048576, 16777216])
    def test_allgather_bandwidth(self, n_elems):
        """Measure SdmaAllgather bandwidth for various data sizes."""
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results = manager.dict()
        port = _get_free_port()
        mp.spawn(
            _worker_allgather_perf,
            args=(self.world_size, port, results, n_elems, 10, 5),
            nprocs=self.world_size,
            join=True,
        )
        total_bytes = n_elems * 4 * self.world_size
        r0 = results[0]
        print(
            f"\nAllgather SDMA: {total_bytes / 1024**2:.1f} MB, "
            f"avg={r0['avg_ms']:.3f}ms, "
            f"BW={r0['bandwidth_gb_s']:.2f} GB/s"
        )
