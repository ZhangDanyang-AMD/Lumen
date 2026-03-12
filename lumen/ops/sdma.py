###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Lumen SDMA collective wrappers over ``mori.ccl``.

Provides thin, lazy-initialized wrappers around mori's SDMA-based collective
primitives (AllreduceSdma, AllgatherSdma) so that the rest of Lumen never
imports ``mori`` directly.  This module also centralizes shmem lifecycle
management (``shmem_torch_process_group_init`` / ``shmem_finalize``) so that
the one-time-per-process initialization is done exactly once regardless of
how many call-sites use SDMA.

Availability check::

    from lumen.ops.sdma import is_sdma_available
    if is_sdma_available():
        from lumen.ops.sdma import sdma_allgather_max, SdmaContext
"""

import functools
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def is_sdma_available() -> bool:
    """Return True if mori SDMA primitives can be imported."""
    try:
        import mori.shmem  # noqa: F401
        from mori.ccl import AllgatherSdma, AllreduceSdma  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Shmem lifecycle (process-level singleton)
# ---------------------------------------------------------------------------


class SdmaContext:
    """Process-level singleton managing mori shmem initialization and PE info.

    Usage::

        ctx = SdmaContext.get()   # lazy init on first call
        print(ctx.my_pe, ctx.npes)
    """

    _instance: Optional["SdmaContext"] = None

    def __init__(self):
        import mori.shmem as shmem

        shmem.shmem_torch_process_group_init("default")
        self.my_pe: int = shmem.shmem_mype()
        self.npes: int = shmem.shmem_npes()
        logger.info(
            "SdmaContext initialized: PE %d / %d",
            self.my_pe,
            self.npes,
        )

    @classmethod
    def get(cls) -> "SdmaContext":
        """Return the singleton, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the singleton (for testing or process-group re-init)."""
        cls._instance = None


# ---------------------------------------------------------------------------
# AllgatherSdma handle cache
# ---------------------------------------------------------------------------


class SdmaAllgather:
    """Reusable AllgatherSdma handle with automatic buffer management.

    Wraps ``mori.ccl.AllgatherSdma`` and caches the handle so that
    repeated calls with the same (or smaller) tensor size reuse the
    same symmetric-memory buffers.

    Args:
        ctx: A :class:`SdmaContext` (defaults to the process singleton).
    """

    def __init__(self, ctx: Optional[SdmaContext] = None):
        self._ctx = ctx or SdmaContext.get()
        self._handle = None
        self._capacity_elems: int = 0
        self._gathered: Optional[torch.Tensor] = None

    @property
    def npes(self) -> int:
        return self._ctx.npes

    def _ensure_handle(self, n_elems: int) -> None:
        """(Re-)create the AllgatherSdma handle if the current buffer is
        too small for *n_elems* float32 elements per PE.
        """
        if self._handle is not None and n_elems <= self._capacity_elems:
            return

        from mori.ccl import AllgatherSdma

        elem_size = 4  # float32
        input_buf_bytes = n_elems * elem_size
        output_buf_bytes = self._ctx.npes * (n_elems + 64) * elem_size

        self._handle = AllgatherSdma(
            self._ctx.my_pe,
            self._ctx.npes,
            input_buffer_size=input_buf_bytes,
            output_buffer_size=output_buf_bytes,
            copy_output_to_user=True,
        )
        self._capacity_elems = n_elems
        logger.debug(
            "SdmaAllgather: (re-)allocated handle for %d elems " "(PE %d/%d)",
            n_elems,
            self._ctx.my_pe,
            self._ctx.npes,
        )

    def __call__(
        self,
        tensor: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """Allgather *tensor* across all PEs.  Returns a 2-D tensor
        ``(npes, n_elems)`` on the same device as *tensor*.
        """
        tensor = tensor.contiguous()
        n_elems = tensor.numel()
        npes = self._ctx.npes
        self._ensure_handle(n_elems)

        if self._gathered is None or self._gathered.numel() < n_elems * npes or self._gathered.device != tensor.device:
            self._gathered = torch.empty(
                n_elems * npes,
                dtype=torch.float32,
                device=tensor.device,
            )

        if stream is None:
            stream = torch.cuda.current_stream(tensor.device)
        self._handle(tensor, self._gathered, n_elems, stream)
        stream.synchronize()
        return self._gathered[: n_elems * npes].reshape(npes, n_elems)


# ---------------------------------------------------------------------------
# AllreduceSdma handle cache
# ---------------------------------------------------------------------------


class SdmaAllreduce:
    """Reusable AllreduceSdma handle with automatic buffer management.

    Wraps ``mori.ccl.AllreduceSdma``.  The reduction operation is
    element-wise **SUM** (the only op mori's SDMA kernel supports).

    Args:
        dtype: Element type (default ``torch.float32``).
        ctx: A :class:`SdmaContext` (defaults to the process singleton).
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        ctx: Optional[SdmaContext] = None,
    ):
        self._ctx = ctx or SdmaContext.get()
        self._dtype = dtype
        self._handle = None
        self._capacity_elems: int = 0

    @property
    def npes(self) -> int:
        return self._ctx.npes

    def _ensure_handle(self, n_elems: int) -> None:
        if self._handle is not None and n_elems <= self._capacity_elems:
            return

        from mori.ccl import AllreduceSdma

        elem_size = torch.tensor([], dtype=self._dtype).element_size()
        input_buf_bytes = n_elems * elem_size
        npes = self._ctx.npes
        output_buf_bytes = npes * (n_elems // npes + 64) * elem_size

        self._handle = AllreduceSdma(
            self._ctx.my_pe,
            npes,
            input_buffer_size=input_buf_bytes,
            output_buffer_size=output_buf_bytes,
            copy_output_to_user=True,
            dtype=self._dtype,
        )
        self._capacity_elems = n_elems
        logger.debug(
            "SdmaAllreduce: (re-)allocated handle for %d elems, dtype=%s " "(PE %d/%d)",
            n_elems,
            self._dtype,
            self._ctx.my_pe,
            self._ctx.npes,
        )

    def inplace(
        self,
        tensor: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """In-place SUM allreduce on *tensor*."""
        tensor = tensor.contiguous()
        n_elems = tensor.numel()
        self._ensure_handle(n_elems)
        if stream is None:
            stream = torch.cuda.current_stream(tensor.device)
        self._handle.allreduce_inplace(tensor, n_elems, stream)
        stream.synchronize()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Out-of-place SUM allreduce: ``output = sum(input across PEs)``."""
        input_tensor = input_tensor.contiguous()
        n_elems = input_tensor.numel()
        self._ensure_handle(n_elems)
        if stream is None:
            stream = torch.cuda.current_stream(input_tensor.device)
        self._handle(input_tensor, output_tensor, n_elems, stream)
        stream.synchronize()


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------


def sdma_allgather_max(
    packed: torch.Tensor,
    allgather: SdmaAllgather,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Allgather *packed* via SDMA, then return element-wise MAX across PEs.

    This emulates ``torch.distributed.all_reduce(op=MAX)`` using SDMA
    transport: the data path goes through the SDMA engine (allgather),
    and only the final MAX reduction is computed locally on-GPU.

    Args:
        packed: 1-D float32 tensor of values on this rank.
        allgather: A pre-created :class:`SdmaAllgather` handle.
        stream: CUDA stream (defaults to current stream).

    Returns:
        1-D float32 tensor with the global element-wise MAX.
    """
    gathered = allgather(packed, stream=stream)  # (npes, n_elems)
    return gathered.amax(dim=0)
