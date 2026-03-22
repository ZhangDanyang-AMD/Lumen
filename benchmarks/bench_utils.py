###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Shared utilities for Lumen benchmarks."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Generator, List, Optional

import torch

TRACE_DIR = os.path.join(os.path.dirname(__file__), "traces")

# ---------------------------------------------------------------------------
# Hardware guards
# ---------------------------------------------------------------------------


def require_cuda():
    if not torch.cuda.is_available():
        print("CUDA not available — skipping benchmark.", file=sys.stderr)
        sys.exit(0)


def require_aiter():
    try:
        import aiter  # noqa: F401
    except ImportError:
        print("AITER not installed — skipping benchmark.", file=sys.stderr)
        sys.exit(0)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    """Container for a single benchmark measurement."""

    name: str
    avg_ms: float
    min_ms: float = 0.0
    max_ms: float = 0.0
    iters: int = 0
    extra: Dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        s = f"{self.name:>45s}  avg={self.avg_ms:8.3f} ms"
        if self.min_ms > 0:
            s += f"  min={self.min_ms:8.3f} ms  max={self.max_ms:8.3f} ms"
        for k, v in self.extra.items():
            if isinstance(v, float):
                s += f"  {k}={v:.4f}"
            else:
                s += f"  {k}={v}"
        return s


def cuda_timer(
    fn: Callable[[], object],
    warmup: int = 5,
    iters: int = 20,
    label: str = "",
    sync_before: bool = True,
) -> BenchResult:
    """Time *fn* on GPU using CUDA events.

    Returns a :class:`BenchResult` with average, min, and max latencies.
    """
    for _ in range(warmup):
        fn()
    if sync_before:
        torch.cuda.synchronize()

    times: List[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg = sum(times) / len(times)
    return BenchResult(
        name=label or "unnamed",
        avg_ms=avg,
        min_ms=min(times),
        max_ms=max(times),
        iters=iters,
    )


def cuda_timer_batch(
    fn: Callable[[], object],
    warmup: int = 5,
    iters: int = 20,
    label: str = "",
) -> BenchResult:
    """Time *fn* over *iters* invocations with a single pair of CUDA events.

    Useful when per-iteration event overhead matters (very fast kernels).
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    return BenchResult(
        name=label or "unnamed",
        avg_ms=total_ms / iters,
        iters=iters,
    )


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def track_cuda_memory(device: Optional[torch.device] = None):
    """Context manager that yields a dict updated with peak memory stats."""
    dev = device or torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(dev)
    torch.cuda.synchronize(dev)
    mem_before = torch.cuda.memory_allocated(dev)
    info: Dict[str, int] = {}
    yield info
    torch.cuda.synchronize(dev)
    info["peak_bytes"] = torch.cuda.max_memory_allocated(dev)
    info["allocated_before"] = mem_before
    info["allocated_after"] = torch.cuda.memory_allocated(dev)
    info["peak_delta"] = info["peak_bytes"] - mem_before


def format_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(title: str, results: List[BenchResult]) -> None:
    """Pretty-print a table of benchmark results."""
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    for r in results:
        print(f"  {r}")
    print(sep)
    print()


def dump_json(results: List[BenchResult], path: str) -> None:
    """Write results to a JSON file for CI integration."""
    data = []
    for r in results:
        d = {"name": r.name, "avg_ms": r.avg_ms, "iters": r.iters}
        d.update(r.extra)
        data.append(d)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Tracing helpers (torch.profiler → Chrome / Perfetto JSON)
# ---------------------------------------------------------------------------


def _ensure_trace_dir(trace_path: str) -> None:
    d = os.path.dirname(trace_path)
    if d:
        os.makedirs(d, exist_ok=True)


def trace_fn(
    fn: Callable[[], object],
    trace_path: str,
    warmup: int = 3,
    active: int = 1,
    label: str = "",
    record_shapes: bool = True,
    with_stack: bool = False,
    with_modules: bool = False,
) -> str:
    """Run *fn* under ``torch.profiler`` and export a Chrome trace JSON.

    Args:
        fn: Zero-arg callable to profile.
        trace_path: Output ``.json`` path (directories created automatically).
        warmup: Profiler warmup iterations (not recorded).
        active: Iterations recorded in the trace.
        label: Optional label printed in the summary header.
        record_shapes: Record tensor shapes in the trace.
        with_stack: Capture Python call stacks (slower but more detail).
        with_modules: Record ``nn.Module`` hierarchy.

    Returns:
        The absolute path to the exported trace file.
    """
    _ensure_trace_dir(trace_path)

    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=record_shapes,
        with_stack=with_stack,
        with_modules=with_modules,
    ) as prof:
        for _ in range(active):
            fn()
            torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)

    header = label or os.path.basename(trace_path)
    print(f"\n--- Trace: {header} ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    size = os.path.getsize(trace_path)
    print(f"  Saved: {trace_path} ({format_bytes(size)})")
    return os.path.abspath(trace_path)


@contextlib.contextmanager
def trace_context(
    trace_path: str,
    record_shapes: bool = True,
    with_stack: bool = False,
    label: str = "",
) -> Generator[torch.profiler.profile, None, None]:
    """Context manager that wraps a code block in ``torch.profiler``.

    Use this instead of :func:`trace_fn` when the profiled code is not a
    single callable (e.g. multi-stream overlap benchmarks).

    .. code-block:: python

        with trace_context("traces/my_trace.json") as prof:
            # ... code to profile ...
        # trace JSON written automatically on exit

    Yields:
        The ``torch.profiler.profile`` object (rarely needed directly).
    """
    _ensure_trace_dir(trace_path)

    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=record_shapes,
        with_stack=with_stack,
    )
    prof.__enter__()
    try:
        yield prof
    finally:
        prof.__exit__(None, None, None)
        prof.export_chrome_trace(trace_path)
        header = label or os.path.basename(trace_path)
        print(f"\n--- Trace: {header} ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        size = os.path.getsize(trace_path)
        print(f"  Saved: {trace_path} ({format_bytes(size)})")
