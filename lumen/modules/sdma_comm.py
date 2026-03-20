###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""SDMA-based TP communication primitives for Lumen modules.

When ``--use-sdma`` is enabled, these autograd functions replace Megatron's
``torch.distributed``-based mappings with mori's SDMA engine.  The SDMA
engine offloads data movement to a dedicated hardware copy engine, freeing
the compute units for overlapping work.

Supported primitives (each mirrors the corresponding Megatron mapping):

- :func:`sdma_gather_from_sequence_parallel_region`
  (fwd: all-gather dim 0, bwd: reduce-scatter dim 0)
- :func:`sdma_reduce_scatter_to_sequence_parallel_region`
  (fwd: reduce-scatter dim 0, bwd: all-gather dim 0)
- :func:`sdma_reduce_from_tensor_model_parallel_region`
  (fwd: all-reduce SUM, bwd: identity)
- :func:`sdma_gather_from_tensor_model_parallel_region`
  (fwd: all-gather last dim, bwd: split last dim)
- :func:`sdma_copy_to_tensor_model_parallel_region`
  (fwd: identity, bwd: all-reduce SUM)
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SDMA TP Context — manages mori shmem init for the TP process group
# ---------------------------------------------------------------------------


class SdmaTpContext:
    """Process-level singleton that registers the TP group with mori shmem."""

    _instance: Optional["SdmaTpContext"] = None

    def __init__(self, tp_group: torch.distributed.ProcessGroup):
        import mori.shmem as shmem

        group_name = "lumen_tp"
        torch._C._distributed_c10d._register_process_group(group_name, tp_group)
        shmem.shmem_torch_process_group_init(group_name)
        self.my_pe: int = shmem.shmem_mype()
        self.npes: int = shmem.shmem_npes()
        self._tp_group = tp_group
        logger.info("SdmaTpContext: PE %d / %d", self.my_pe, self.npes)

    @classmethod
    def get(cls, tp_group: torch.distributed.ProcessGroup) -> "SdmaTpContext":
        if cls._instance is None:
            cls._instance = cls(tp_group)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None


# ---------------------------------------------------------------------------
# Cached SDMA handle wrappers
# ---------------------------------------------------------------------------


class _TpSdmaAllgather:
    """Cached ``AllgatherSdma`` handle that works with any dtype by
    reinterpreting tensor bytes as float32 for the SDMA transport.
    """

    def __init__(self, ctx: SdmaTpContext):
        self._ctx = ctx
        self._handle = None
        self._capacity_f32: int = 0
        self._gathered_buf: Optional[torch.Tensor] = None

    def _ensure(self, n_f32: int) -> None:
        if self._handle is not None and n_f32 <= self._capacity_f32:
            return
        from mori.ccl import AllgatherSdma

        elem_size = 4  # float32
        padding_elems = 64
        self._handle = AllgatherSdma(
            self._ctx.my_pe,
            self._ctx.npes,
            input_buffer_size=(n_f32 + padding_elems) * elem_size,
            output_buffer_size=self._ctx.npes * (n_f32 + padding_elems) * elem_size,
            copy_output_to_user=True,
        )
        self._capacity_f32 = n_f32
        logger.debug(
            "_TpSdmaAllgather: (re)alloc for %d f32 elems (PE %d/%d)",
            n_f32,
            self._ctx.my_pe,
            self._ctx.npes,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Flat all-gather. Returns ``(npes, n_elems)`` in *tensor.dtype*."""
        tensor = tensor.contiguous()
        orig_dtype = tensor.dtype
        orig_numel = tensor.numel()
        flat = tensor.view(-1)

        if orig_dtype != torch.float32:
            flat_f32 = flat.view(torch.float32)
        else:
            flat_f32 = flat
        n_f32 = flat_f32.numel()

        self._ensure(n_f32)
        npes = self._ctx.npes

        if (
            self._gathered_buf is None
            or self._gathered_buf.numel() < n_f32 * npes
            or self._gathered_buf.device != tensor.device
        ):
            self._gathered_buf = torch.empty(
                n_f32 * npes,
                dtype=torch.float32,
                device=tensor.device,
            )

        stream = torch.cuda.current_stream(tensor.device)
        self._handle(flat_f32, self._gathered_buf, n_f32, stream)
        stream.synchronize()

        gathered = self._gathered_buf[: n_f32 * npes]
        if orig_dtype != torch.float32:
            gathered = gathered.view(orig_dtype)

        return gathered[: orig_numel * npes].reshape(npes, orig_numel)

    def start_async(self, tensor: torch.Tensor, stream=None) -> bool:
        """Start async all-gather. Call :meth:`wait_async` to complete."""
        tensor = tensor.contiguous()
        orig_dtype = tensor.dtype
        orig_numel = tensor.numel()
        flat = tensor.view(-1)

        if orig_dtype != torch.float32:
            flat_f32 = flat.view(torch.float32)
        else:
            flat_f32 = flat
        n_f32 = flat_f32.numel()

        self._ensure(n_f32)
        npes = self._ctx.npes

        if (
            self._gathered_buf is None
            or self._gathered_buf.numel() < n_f32 * npes
            or self._gathered_buf.device != tensor.device
        ):
            self._gathered_buf = torch.empty(
                n_f32 * npes,
                dtype=torch.float32,
                device=tensor.device,
            )

        s = stream if stream is not None else torch.cuda.current_stream(tensor.device)
        self._async_meta = {
            "orig_dtype": orig_dtype,
            "orig_numel": orig_numel,
            "n_f32": n_f32,
            "npes": npes,
        }
        return self._handle.start_async(flat_f32, self._gathered_buf, n_f32, s)

    def wait_async(self, stream=None) -> torch.Tensor:
        """Wait for async all-gather and return ``(npes, n_elems)`` in original dtype."""
        meta = getattr(self, "_async_meta", None)
        if meta is None:
            raise RuntimeError("wait_async called without prior start_async")
        s = stream if stream is not None else torch.cuda.current_stream(self._gathered_buf.device)
        self._handle.wait_async(s)
        gathered = self._gathered_buf[: meta["n_f32"] * meta["npes"]]
        if meta["orig_dtype"] != torch.float32:
            gathered = gathered.view(meta["orig_dtype"])
        result = gathered[: meta["orig_numel"] * meta["npes"]].reshape(meta["npes"], meta["orig_numel"])
        del self._async_meta
        return result


class _TpSdmaAllreduce:
    """Cached ``AllreduceSdma`` handle (SUM) for a specific dtype.

    The mori SDMA allreduce kernel natively supports uint32, int32,
    float16, and bfloat16.  When ``dtype`` is ``torch.float32``, this
    wrapper transparently casts to bfloat16 for the reduction and casts
    the result back.
    """

    _NATIVE_DTYPES = frozenset({torch.uint32, torch.int32, torch.float16, torch.bfloat16})

    def __init__(self, ctx: SdmaTpContext, dtype: torch.dtype = torch.float32):
        if dtype != torch.float32 and dtype not in self._NATIVE_DTYPES:
            raise ValueError(
                f"_TpSdmaAllreduce: unsupported dtype {dtype}. "
                f"Supported: float32 (via bf16), {sorted(str(d) for d in self._NATIVE_DTYPES)}"
            )
        self._ctx = ctx
        self._user_dtype = dtype
        self._wire_dtype = torch.bfloat16 if dtype == torch.float32 else dtype
        self._handle = None
        self._capacity: int = 0

    def _ensure(self, n_elems: int) -> None:
        if self._handle is not None and n_elems <= self._capacity:
            return
        from mori.ccl import AllreduceSdma

        elem_size = torch.tensor([], dtype=self._wire_dtype).element_size()
        npes = self._ctx.npes
        self._handle = AllreduceSdma(
            self._ctx.my_pe,
            npes,
            input_buffer_size=n_elems * elem_size,
            output_buffer_size=npes * (n_elems // npes + 64) * elem_size,
            copy_output_to_user=True,
            dtype=self._wire_dtype,
        )
        self._capacity = n_elems
        logger.debug(
            "_TpSdmaAllreduce: (re)alloc for %d elems, " "user_dtype=%s wire_dtype=%s (PE %d/%d)",
            n_elems,
            self._user_dtype,
            self._wire_dtype,
            self._ctx.my_pe,
            self._ctx.npes,
        )

    @property
    def _needs_cast(self) -> bool:
        return self._user_dtype != self._wire_dtype

    def inplace(self, tensor: torch.Tensor) -> None:
        tensor = tensor.contiguous()
        self._ensure(tensor.numel())
        stream = torch.cuda.current_stream(tensor.device)
        if self._needs_cast:
            buf = tensor.to(self._wire_dtype)
            self._handle.allreduce_inplace(buf, buf.numel(), stream)
            stream.synchronize()
            tensor.copy_(buf.to(self._user_dtype))
        else:
            self._handle.allreduce_inplace(tensor, tensor.numel(), stream)
            stream.synchronize()

    def start_async_inplace(self, tensor: torch.Tensor, stream=None) -> bool:
        """Start async in-place allreduce.  Caller must call :meth:`wait_async`."""
        tensor = tensor.contiguous()
        self._ensure(tensor.numel())
        if self._needs_cast:
            self._async_buf = tensor.to(self._wire_dtype)
            self._async_tensor = tensor
            return self._handle.start_async_inplace(self._async_buf, self._async_buf.numel(), stream)
        return self._handle.start_async_inplace(tensor, tensor.numel(), stream)

    def wait_async(self, stream=None) -> None:
        """Wait for a previously started async allreduce."""
        self._handle.wait_async(stream)
        if self._needs_cast and hasattr(self, "_async_buf"):
            self._async_tensor.copy_(self._async_buf.to(self._user_dtype))
            del self._async_buf, self._async_tensor


# ---------------------------------------------------------------------------
# SdmaTpComm — high-level TP communication manager
# ---------------------------------------------------------------------------


class SdmaTpComm:
    """Manages SDMA handles for TP communication.

    Provides high-level methods that match the semantics of Megatron's
    TP mapping primitives but route through mori SDMA.

    Use :meth:`get` to obtain the (cached) singleton for a given TP group.
    """

    _instance: Optional["SdmaTpComm"] = None

    def __init__(self, tp_group: torch.distributed.ProcessGroup):
        self._ctx = SdmaTpContext.get(tp_group)
        self._ag = _TpSdmaAllgather(self._ctx)
        self._ar_handles: dict = {}
        self._rs_output_buf: Optional[torch.Tensor] = None

    @property
    def npes(self) -> int:
        return self._ctx.npes

    @property
    def my_pe(self) -> int:
        return self._ctx.my_pe

    def _get_ar(self, dtype: torch.dtype) -> _TpSdmaAllreduce:
        if dtype not in self._ar_handles:
            self._ar_handles[dtype] = _TpSdmaAllreduce(self._ctx, dtype)
        return self._ar_handles[dtype]

    # -- Primitives --

    def allgather_dim0(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather along dim 0. ``[S/TP, ...] → [S, ...]``."""
        shape = tensor.shape
        gathered = self._ag(tensor)  # (npes, numel_per_pe)
        return gathered.reshape(self.npes * shape[0], *shape[1:])

    def allgather_last_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather along last dim. ``[*, D/TP] → [*, D]``."""
        shape = tensor.shape
        batch_shape = shape[:-1]
        d_local = shape[-1]
        gathered = self._ag(tensor)  # (npes, numel_per_pe)
        reshaped = gathered.reshape(self.npes, *batch_shape, d_local)
        return reshaped.movedim(0, -2).reshape(*batch_shape, self.npes * d_local)

    def allreduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """Out-of-place all-reduce SUM."""
        result = tensor.clone()
        ar = self._get_ar(tensor.dtype)
        ar.inplace(result)
        return result

    def allreduce_sum_inplace(self, tensor: torch.Tensor) -> None:
        """In-place all-reduce SUM."""
        ar = self._get_ar(tensor.dtype)
        ar.inplace(tensor)

    def reduce_scatter_dim0(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce-scatter along dim 0. ``[S, ...] → [S/TP, ...]``.

        Implemented as all-reduce(SUM) + local chunk extraction.
        """
        reduced = self.allreduce_sum(tensor)
        chunk_size = tensor.shape[0] // self.npes
        start = self.my_pe * chunk_size
        return reduced[start : start + chunk_size].contiguous()

    # -- Async primitives (for comm-GEMM overlap) --

    def allgather_dim0_async(self, tensor: torch.Tensor, stream=None) -> bool:
        """Start async all-gather along dim 0. Call :meth:`wait_allgather_dim0` to complete."""
        shape = tensor.shape
        self._ag_async_shape = shape
        return self._ag.start_async(tensor, stream)

    def wait_allgather_dim0(self, stream=None) -> torch.Tensor:
        """Wait for async all-gather and return ``[S, ...]``."""
        gathered = self._ag.wait_async(stream)  # (npes, numel_per_pe)
        shape = getattr(self, "_ag_async_shape", None)
        if shape is None:
            raise RuntimeError("wait_allgather_dim0 called without prior allgather_dim0_async")
        del self._ag_async_shape
        return gathered.reshape(self.npes * shape[0], *shape[1:])

    def allreduce_sum_async(self, tensor: torch.Tensor, stream=None) -> bool:
        """Start async in-place all-reduce SUM. Call :meth:`wait_allreduce_sum` to complete."""
        tensor = tensor.contiguous()
        self._ar_async_dtype = tensor.dtype
        ar = self._get_ar(tensor.dtype)
        return ar.start_async_inplace(tensor, stream)

    def wait_allreduce_sum(self, stream=None) -> None:
        """Wait for async all-reduce to complete."""
        dtype = getattr(self, "_ar_async_dtype", None)
        if dtype is None:
            raise RuntimeError("wait_allreduce_sum called without prior allreduce_sum_async")
        ar = self._get_ar(dtype)
        ar.wait_async(stream)
        del self._ar_async_dtype

    def reduce_scatter_dim0_async(self, tensor: torch.Tensor, stream=None) -> bool:
        """Start async reduce-scatter along dim 0.

        Call :meth:`wait_reduce_scatter_dim0` to complete.  The caller
        must keep *tensor* alive until wait returns (mori holds a raw
        pointer to the device buffer).
        """
        tensor = tensor.contiguous()
        ar = self._get_ar(tensor.dtype)
        wire_dtype = ar._wire_dtype
        wire_tensor = tensor.to(wire_dtype) if ar._needs_cast else tensor
        n_elems = wire_tensor.numel()
        if (
            self._rs_output_buf is None
            or self._rs_output_buf.numel() < n_elems
            or self._rs_output_buf.device != wire_tensor.device
            or self._rs_output_buf.dtype != wire_dtype
        ):
            self._rs_output_buf = torch.empty_like(wire_tensor)
        self._rs_async_shape = tensor.shape
        self._rs_user_dtype = tensor.dtype
        self._rs_input_buf = wire_tensor
        return ar._handle.start_async(wire_tensor, self._rs_output_buf, n_elems, stream)

    def wait_reduce_scatter_dim0(self, stream=None) -> torch.Tensor:
        """Wait for async reduce-scatter and return ``[S/TP, ...]``."""
        shape = getattr(self, "_rs_async_shape", None)
        if shape is None:
            raise RuntimeError("wait_reduce_scatter_dim0 called without prior reduce_scatter_dim0_async")
        user_dtype = self._rs_user_dtype
        ar = self._get_ar(user_dtype)
        ar._handle.wait_async(stream)
        chunk_size = shape[0] // self.npes
        start = self.my_pe * chunk_size
        result = self._rs_output_buf[start : start + chunk_size].contiguous()
        if ar._needs_cast:
            result = result.to(user_dtype)
        del self._rs_async_shape, self._rs_user_dtype, self._rs_input_buf
        return result

    @classmethod
    def get(cls, tp_group: torch.distributed.ProcessGroup) -> "SdmaTpComm":
        if cls._instance is None:
            cls._instance = cls(tp_group)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None


# ---------------------------------------------------------------------------
# Autograd functions
# ---------------------------------------------------------------------------


class _SdmaGatherFromSequenceParallelRegion(torch.autograd.Function):
    """FWD: all-gather dim 0 via SDMA.  BWD: reduce-scatter dim 0."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, comm: SdmaTpComm):
        ctx.comm = comm
        return comm.allgather_dim0(input_)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return ctx.comm.reduce_scatter_dim0(grad_output), None


class _SdmaReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """FWD: reduce-scatter dim 0 via SDMA.  BWD: all-gather dim 0."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, comm: SdmaTpComm):
        ctx.comm = comm
        return comm.reduce_scatter_dim0(input_)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return ctx.comm.allgather_dim0(grad_output), None


class _SdmaReduceFromTensorModelParallelRegion(torch.autograd.Function):
    """FWD: all-reduce SUM via SDMA.  BWD: identity (pass-through)."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, comm: SdmaTpComm):
        return comm.allreduce_sum(input_)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class _SdmaGatherFromTensorModelParallelRegion(torch.autograd.Function):
    """FWD: all-gather last dim via SDMA.  BWD: split last dim (local chunk)."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, comm: SdmaTpComm):
        ctx.comm = comm
        ctx.d_local = input_.shape[-1]
        return comm.allgather_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        d_local = ctx.d_local
        start = ctx.comm.my_pe * d_local
        return grad_output[..., start : start + d_local].contiguous(), None


class _SdmaCopyToTensorModelParallelRegion(torch.autograd.Function):
    """FWD: identity (pass-through).  BWD: all-reduce SUM via SDMA."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, comm: SdmaTpComm):
        ctx.comm = comm
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return ctx.comm.allreduce_sum(grad_output), None


# ---------------------------------------------------------------------------
# Public API — SDMA-accelerated Megatron mapping functions
# ---------------------------------------------------------------------------


def sdma_gather_from_sequence_parallel_region(
    input_: torch.Tensor,
    comm: SdmaTpComm,
) -> torch.Tensor:
    return _SdmaGatherFromSequenceParallelRegion.apply(input_, comm)


def sdma_reduce_scatter_to_sequence_parallel_region(
    input_: torch.Tensor,
    comm: SdmaTpComm,
) -> torch.Tensor:
    return _SdmaReduceScatterToSequenceParallelRegion.apply(input_, comm)


def sdma_reduce_from_tensor_model_parallel_region(
    input_: torch.Tensor,
    comm: SdmaTpComm,
) -> torch.Tensor:
    return _SdmaReduceFromTensorModelParallelRegion.apply(input_, comm)


def sdma_gather_from_tensor_model_parallel_region(
    input_: torch.Tensor,
    comm: SdmaTpComm,
) -> torch.Tensor:
    return _SdmaGatherFromTensorModelParallelRegion.apply(input_, comm)


def sdma_copy_to_tensor_model_parallel_region(
    input_: torch.Tensor,
    comm: SdmaTpComm,
) -> torch.Tensor:
    return _SdmaCopyToTensorModelParallelRegion.apply(input_, comm)
