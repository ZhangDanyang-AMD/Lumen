###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Benchmark 5 — Wgrad delay improves overlap.

Tests **actual Lumen implementations** only:

  * **_DeferredWgrad API**: ``defer(weight, compute_fn)`` / ``execute()``
    from ``lumen.modules.parallel_linear``.
  * **_DeferredWgrad + stream overlap**: Executes the deferred wgrad on a
    secondary CUDA stream while the main stream does other work — this is
    how the API is *intended* to be used in a real training loop.
  * **gradient_accumulation_fusion**: The ``main_grad.add_(dw)`` path inside
    ``quantized_linear`` backward (``gradient_accumulation_fusion=True``).

Run::

    python -m benchmarks.bench_wgrad_delay
    pytest benchmarks/bench_wgrad_delay.py -v -s
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from benchmarks.bench_utils import (
    BenchResult,
    cuda_timer,
    print_report,
    require_cuda,
)
from benchmarks.conftest import AITER, CUDA

# ---------------------------------------------------------------------------
# Dimensions (Llama 3.1 8B)
# ---------------------------------------------------------------------------
M = 4096  # tokens (B * S)
K = 4096  # hidden_dim
N = 14336  # FFN intermediate


# ---------------------------------------------------------------------------
# 1. Lumen _DeferredWgrad API — basic overhead
# ---------------------------------------------------------------------------


@CUDA
class TestDeferredWgradAPI:
    """Test Lumen's _DeferredWgrad class for wgrad deferral."""

    # Expected: Deferred wgrad (defer + execute) should have nearly identical
    # total latency to eager wgrad. The purpose of deferral is NOT to speed up
    # a single wgrad, but to allow scheduling it on a secondary stream later,
    # overlapping with other work. The defer() call itself is near-zero cost
    # (just stores a closure); execute() triggers the same GEMM.
    def test_defer_and_execute_latency(self):
        """Measure defer + execute vs eager wgrad."""
        from lumen.modules.parallel_linear import _DeferredWgrad

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
        grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

        def _eager():
            dw = grad_out.T @ x
            w.grad = dw

        r_eager = cuda_timer(_eager, label="eager wgrad (dW = gO^T @ X)")

        dwg = _DeferredWgrad()

        def _deferred():
            dwg.defer(w, lambda: grad_out.T @ x)
            dwg.execute()

        r_deferred = cuda_timer(_deferred, label="deferred wgrad (defer + execute)")

        print_report("_DeferredWgrad: Eager vs Deferred", [r_eager, r_deferred])

    # Expected: Near-zero latency (~0.001 ms). defer() only captures a Python
    # closure (weight reference + compute lambda) without launching any GPU
    # work. This confirms the deferral mechanism adds negligible overhead to
    # the backward pass critical path.
    def test_defer_only_latency(self):
        """Measure just the defer call (no execute) — should be near-zero."""
        from lumen.modules.parallel_linear import _DeferredWgrad

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
        grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        dwg = _DeferredWgrad()

        r = cuda_timer(
            lambda: dwg.defer(w, lambda: grad_out.T @ x),
            label="defer only (closure capture)",
        )
        print_report("_DeferredWgrad: Defer Only", [r])
        print(f"  Defer overhead: {r.avg_ms:.4f} ms (should be ~0)")

    # Expected: _DeferredWgrad.execute() accumulates into main_grad when it
    # exists (line 58-59 of parallel_linear.py). After execute(), the weight's
    # main_grad should contain the computed gradient, and _pending_wgrad should
    # be cleared. This validates the accumulation path that the distributed
    # optimizer relies on.
    def test_main_grad_accumulation(self):
        """Verify execute() writes to main_grad when present."""
        from lumen.modules.parallel_linear import _DeferredWgrad

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
        w.main_grad = torch.zeros(N, K, device="cuda", dtype=torch.bfloat16)
        grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

        dwg = _DeferredWgrad()
        dwg.defer(w, lambda: grad_out.T @ x)

        assert dwg.has_pending
        dwg.execute()
        assert not dwg.has_pending

        assert w.main_grad.abs().sum() > 0, "main_grad should be non-zero after execute()"
        print(f"\n  main_grad norm: {w.main_grad.norm().item():.4f}")


# ---------------------------------------------------------------------------
# 2. _DeferredWgrad with stream overlap (intended usage pattern)
# ---------------------------------------------------------------------------


@CUDA
class TestDeferredWgradStreamOverlap:
    """Use _DeferredWgrad.execute() on a secondary stream to overlap with
    other work on the compute stream — this is the actual usage pattern
    that produces the overlap benefit in Lumen training.
    """

    # Expected: Overlapped should be ~1.3-1.8x faster than sequential because
    # _DeferredWgrad.execute() (which runs the deferred GEMM) happens on
    # wgrad_stream concurrently with the next layer's forward on the compute
    # stream. Total ≈ max(dW, fwd) instead of dW + fwd.
    def test_overlap_vs_sequential(self):
        from lumen.modules.parallel_linear import _DeferredWgrad

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
        w.main_grad = torch.zeros(N, K, device="cuda", dtype=torch.bfloat16)
        w_next = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.02
        grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

        dwg = _DeferredWgrad()
        wgrad_stream = torch.cuda.Stream()

        # Sequential: execute deferred wgrad, then next-layer forward
        def _sequential():
            w.main_grad.zero_()
            dwg.defer(w, lambda: grad_out.T @ x)
            dwg.execute()
            torch.cuda.synchronize()
            _ = x @ w_next
            torch.cuda.synchronize()

        r_seq = cuda_timer(_sequential, label="sequential: execute() then fwd")

        # Measure individual components for overlap ratio
        r_dw = cuda_timer(lambda: grad_out.T @ x, label="dW alone")
        r_fwd = cuda_timer(lambda: x @ w_next, label="next-layer fwd alone")

        # Overlapped: execute deferred wgrad on wgrad_stream, fwd on compute
        def _overlapped():
            w.main_grad.zero_()
            dwg.defer(w, lambda: grad_out.T @ x)
            with torch.cuda.stream(wgrad_stream):
                dwg.execute()
            _ = x @ w_next  # concurrent on default stream
            wgrad_stream.synchronize()

        r_ovl = cuda_timer(_overlapped, label="overlapped: execute() || fwd")

        speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        T_parts = r_dw.avg_ms + r_fwd.avg_ms
        overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
        r_ovl.extra["speedup"] = round(speedup, 2)
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)
        print_report(
            "_DeferredWgrad Stream Overlap",
            [r_dw, r_fwd, r_seq, r_ovl],
        )
        print(f"  Overlap ratio: {overlap_ratio:.3f}")

    # Expected: Running execute() on a secondary stream for each of N layers
    # in a pipeline should show cumulative speedup. Each layer's deferred dW
    # overlaps with the next layer's dX, hiding N-1 dW computations from
    # the critical path. The total time approaches T(all_dX) + T(one_dW)
    # instead of T(all_dX) + T(all_dW).
    def test_multi_layer_pipeline(self):
        from lumen.modules.parallel_linear import _DeferredWgrad

        n_layers = 4
        weights = [nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02) for _ in range(n_layers)]
        for w in weights:
            w.main_grad = torch.zeros_like(w.data)

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        grad_outs = [torch.randn(M, N, device="cuda", dtype=torch.bfloat16) for _ in range(n_layers)]

        dwg = _DeferredWgrad()
        wgrad_stream = torch.cuda.Stream()

        def _eager_pipeline():
            for i in range(n_layers):
                weights[i].main_grad.zero_()
                _ = grad_outs[i] @ weights[i]  # dX
                weights[i].main_grad.add_(grad_outs[i].T @ x)  # dW

        def _deferred_pipeline():
            for i in range(n_layers):
                weights[i].main_grad.zero_()
                if dwg.has_pending:
                    with torch.cuda.stream(wgrad_stream):
                        dwg.execute()
                _ = grad_outs[i] @ weights[i]  # dX on compute stream
                dwg.defer(weights[i], lambda i=i: grad_outs[i].T @ x)
            with torch.cuda.stream(wgrad_stream):
                dwg.execute()
            wgrad_stream.synchronize()

        r_eager = cuda_timer(_eager_pipeline, label=f"{n_layers}-layer eager")
        r_deferred = cuda_timer(_deferred_pipeline, label=f"{n_layers}-layer deferred")

        speedup = r_eager.avg_ms / max(r_deferred.avg_ms, 1e-6)
        r_deferred.extra["speedup"] = round(speedup, 2)
        print_report(
            f"{n_layers}-Layer Pipeline: Eager vs _DeferredWgrad",
            [r_eager, r_deferred],
        )


# ---------------------------------------------------------------------------
# 3. gradient_accumulation_fusion via quantized_linear backward
# ---------------------------------------------------------------------------


@CUDA
@AITER
class TestGradAccumFusionQuantizedLinear:
    """Test the gradient_accumulation_fusion flag in Lumen's quantized_linear.

    When gradient_accumulation_fusion=True, the backward pass of
    quantized_linear directly does weight.main_grad.add_(grad_weight)
    instead of returning grad_weight. This avoids a separate accumulation
    kernel in multi-microbatch training.
    """

    # Expected: With gradient_accumulation_fusion=True, the backward should
    # accumulate directly into main_grad (grad_weight returned as None).
    # With =False, backward returns grad_weight as a separate tensor.
    # Latency should be similar, but the fused path saves one kernel launch
    # (the separate .add_ call) and one tensor allocation per microbatch.
    def test_fused_vs_separate_accumulation(self):
        from lumen.ops.quantize.linear import quantized_linear
        from lumen.quantize.config import _get_float8_e4m3

        fp8_dtype = _get_float8_e4m3()
        hidden = K
        ffn = 1024

        x = torch.randn(M, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = nn.Parameter(torch.randn(ffn, hidden, device="cuda", dtype=torch.bfloat16) * 0.02)
        w.main_grad = torch.zeros_like(w.data)

        def _without_fusion():
            x_in = x.detach().requires_grad_(True)
            out = quantized_linear(
                x_in,
                w,
                scaling_type="none",
                fp8_dtype=fp8_dtype,
                gradient_accumulation_fusion=False,
            )
            out.sum().backward()

        def _with_fusion():
            w.main_grad.zero_()
            x_in = x.detach().requires_grad_(True)
            out = quantized_linear(
                x_in,
                w,
                scaling_type="none",
                fp8_dtype=fp8_dtype,
                gradient_accumulation_fusion=True,
            )
            out.sum().backward()

        r_no_fuse = cuda_timer(_without_fusion, label="grad_accum_fusion=False")
        r_fuse = cuda_timer(_with_fusion, label="grad_accum_fusion=True")

        print_report(
            "quantized_linear: gradient_accumulation_fusion",
            [r_no_fuse, r_fuse],
        )

    # Expected: After backward with gradient_accumulation_fusion=True,
    # w.grad should be None (not returned) and w.main_grad should contain
    # the gradient. This verifies the fused path works correctly.
    def test_fusion_writes_to_main_grad(self):
        from lumen.ops.quantize.linear import quantized_linear
        from lumen.quantize.config import _get_float8_e4m3

        fp8_dtype = _get_float8_e4m3()
        hidden = 256
        ffn = 512

        x = torch.randn(32, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = nn.Parameter(torch.randn(ffn, hidden, device="cuda", dtype=torch.bfloat16) * 0.02)
        w.main_grad = torch.zeros(ffn, hidden, device="cuda", dtype=torch.bfloat16)

        out = quantized_linear(
            x,
            w,
            scaling_type="none",
            fp8_dtype=fp8_dtype,
            gradient_accumulation_fusion=True,
        )
        out.sum().backward()

        assert w.main_grad.abs().sum() > 0, "main_grad should be non-zero"
        print(f"\n  main_grad norm after fused backward: {w.main_grad.norm().item():.4f}")

    # Expected: With gradient_accumulation_fusion=True under FP8 scaling modes,
    # the backward should still correctly accumulate into main_grad. FP8
    # quantization in backward adds quant/dequant steps but the final
    # main_grad.add_(grad_weight) path should work identically.
    def test_fusion_with_fp8_scaling(self):
        from lumen.ops.quantize.linear import quantized_linear
        from lumen.quantize.config import _get_float8_e4m3

        fp8_dtype = _get_float8_e4m3()
        hidden = 256
        ffn = 512

        results: List[BenchResult] = []
        for scaling in ["none", "delayed", "dynamic"]:
            x = torch.randn(32, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            w = nn.Parameter(torch.randn(ffn, hidden, device="cuda", dtype=torch.bfloat16) * 0.02)
            w.main_grad = torch.zeros(ffn, hidden, device="cuda", dtype=torch.bfloat16)

            def _run(s=scaling):
                w.main_grad.zero_()
                x_in = x.detach().requires_grad_(True)
                out = quantized_linear(
                    x_in,
                    w,
                    scaling_type=s,
                    fp8_dtype=fp8_dtype,
                    gradient_accumulation_fusion=True,
                )
                out.sum().backward()

            r = cuda_timer(_run, label=f"fused backward scaling={scaling}")
            results.append(r)

        print_report("gradient_accumulation_fusion across FP8 modes", results)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def main():
    require_cuda()

    from lumen.modules.parallel_linear import _DeferredWgrad

    results: List[BenchResult] = []

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
    w.main_grad = torch.zeros(N, K, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    # 1. Eager vs deferred
    r_eager = cuda_timer(lambda: grad_out.T @ x, label="eager wgrad")
    results.append(r_eager)

    dwg = _DeferredWgrad()

    def _deferred():
        w.main_grad.zero_()
        dwg.defer(w, lambda: grad_out.T @ x)
        dwg.execute()

    r_def = cuda_timer(_deferred, label="Lumen _DeferredWgrad")
    results.append(r_def)

    # 2. Stream overlap using _DeferredWgrad
    wgrad_stream = torch.cuda.Stream()
    w_next = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    def _overlapped():
        w.main_grad.zero_()
        dwg.defer(w, lambda: grad_out.T @ x)
        with torch.cuda.stream(wgrad_stream):
            dwg.execute()
        _ = x @ w_next
        wgrad_stream.synchronize()

    r_ovl = cuda_timer(_overlapped, label="_DeferredWgrad || next fwd")
    results.append(r_ovl)

    print_report("Lumen Wgrad Delay Benchmarks", results)


if __name__ == "__main__":
    main()
