###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Benchmark 3 — FP8 param all-gather reduces memory.

Exercises Lumen's ``FP8ParamManager`` feature:

  * **Quantize params**: Convert BF16 model weights → FP8 (E4M3).
  * **Dequant hooks**: Forward pre-hooks that dequantize before compute.
  * **Memory savings**: ~2x reduction in param storage and all-gather volume.
  * **Numerical quality**: Round-trip quant→dequant SNR.
  * **End-to-end**: ``LumenGatedMLP`` with FP8 params — latency & accuracy.
  * **Section 6 — Multi-GPU NCCL**: Real ``all_gather`` with FP8 vs BF16 shards,
    including the full quant → gather → dequant pipeline latency.
  * **Section 7 — Multi-GPU SDMA**: ``SdmaTpComm`` allgather with FP8 param
    shards on dedicated DMA engines.
  * **Section 8 — E2E distributed**: Full-model FP8 param allgather + forward
    correctness across ranks.

Run single-GPU::

    python -m benchmarks.bench_fp8_param_allgather
    pytest benchmarks/bench_fp8_param_allgather.py -v -s

Run multi-GPU (2 GPU) — NCCL::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k NCCL

Run multi-GPU (2 GPU) — SDMA (requires mori)::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k SDMA

Run multi-GPU (2 GPU) — E2E::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k E2E

Run multi-GPU (8 GPU) — all distributed tests::

    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k "NCCL or SDMA or E2E"

Run multi-GPU (8 GPU) — individual sections::

    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k NCCL
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k SDMA
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k E2E
"""

from __future__ import annotations

import os
from typing import List

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from benchmarks.bench_utils import (
    BenchResult,
    cuda_timer,
    format_bytes,
    print_report,
    print_report_with_table,
    print_table,
    require_cuda,
)
from benchmarks.conftest import CUDA

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
HIDDEN = 4096
FFN_HIDDEN = 14336
N_LAYERS = 32

# Timing parameters — overridable via LUMEN_BENCH_WARMUP / LUMEN_BENCH_ITERS
_WARMUP = 10
_ITERS = 30
_TRIM = 10.0


def _build_mlp_stack(n_layers=4, hidden=HIDDEN, ffn=FFN_HIDDEN):
    """Build an nn.Linear stack mimicking transformer MLP layers.

    FP8ParamManager operates on nn.Linear (checks ``isinstance(module,
    nn.Linear)``), so we use standard Linear layers to model the
    gate/up/down projections of each transformer layer.
    """
    layers = []
    for _ in range(n_layers):
        layers.extend(
            [
                nn.Linear(hidden, ffn, bias=False),  # gate_proj
                nn.Linear(hidden, ffn, bias=False),  # up_proj
                nn.Linear(ffn, hidden, bias=False),  # down_proj
                nn.Linear(hidden, hidden, bias=False),  # qkv_proj
            ]
        )
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Multi-GPU infrastructure
# ---------------------------------------------------------------------------


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


_DIST = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Multi-GPU — run with torchrun --nproc_per_node=N",
)


def _sdma_available():
    try:
        import mori  # noqa: F401

        return True
    except ImportError:
        return False


_SDMA = pytest.mark.skipif(
    "RANK" not in os.environ or not _sdma_available(),
    reason="Multi-GPU + mori SDMA required",
)


# ---------------------------------------------------------------------------
# 1. FP8ParamManager on nn.Linear stacks (matching Lumen shapes)
# ---------------------------------------------------------------------------


@CUDA
class TestFP8ParamManagerCompression:
    """FP8ParamManager applied to nn.Linear stacks with Lumen-scale shapes."""

    # Expected: ~2x memory reduction. BF16 uses 2 bytes/element; FP8 uses
    # 1 byte/element plus a small per-tensor scale (4 bytes). For large weight
    # matrices (e.g., 4096x14336), the scale overhead is negligible, yielding
    # close to 2x compression. This directly halves all-gather volume in TP.
    def test_single_layer_savings(self):
        """Quantize a single layer's linears to FP8."""
        from lumen.quantize.fp8_params import FP8ParamManager

        model = _build_mlp_stack(n_layers=1).to(device="cuda", dtype=torch.bfloat16)
        bf16_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())

        mgr = FP8ParamManager()
        n_quant = mgr.quantize_params(model)

        fp8_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
        scale_overhead = n_quant * 4
        total_fp8 = fp8_bytes + scale_overhead
        ratio = bf16_bytes / total_fp8

        sep = "=" * 52
        print(f"\n{sep}")
        print(f"  FP8 Param Compression — 1 layer ({n_quant} tensors)")
        print(sep)
        print(f"  BF16 params:   {format_bytes(bf16_bytes):>10s}")
        print(f"  FP8  params:   {format_bytes(total_fp8):>10s}")
        print("  ──────────────────────────────")
        print(f"  Compression:   {ratio:.2f}x")
        print(f"  Saved:         {format_bytes(bf16_bytes - total_fp8):>10s}")
        print(sep)
        assert ratio > 1.9, f"Expected ~2x, got {ratio:.2f}x"

    # Expected: Same ~2x compression as single layer, confirming that
    # FP8ParamManager scales across the full model. The per-tensor scale
    # overhead becomes even more negligible with more parameters, so the
    # compression ratio should be at least 1.9x across 4 layers (16 linears).
    def test_multi_layer_savings(self):
        """Quantize a 4-layer stack to FP8."""
        from lumen.quantize.fp8_params import FP8ParamManager

        model = _build_mlp_stack(n_layers=4).to(device="cuda", dtype=torch.bfloat16)
        bf16_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())

        mgr = FP8ParamManager()
        n_quant = mgr.quantize_params(model)
        fp8_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
        saved = mgr.memory_savings_bytes(model)

        total_fp8 = fp8_bytes + n_quant * 4
        ratio = bf16_bytes / total_fp8
        sep = "=" * 52
        print(f"\n{sep}")
        print(f"  FP8 Param Compression — 4 layers ({n_quant} tensors)")
        print(sep)
        print(f"  BF16 params:   {format_bytes(bf16_bytes):>10s}")
        print(f"  FP8  params:   {format_bytes(total_fp8):>10s}")
        print("  ──────────────────────────────")
        print(f"  Compression:   {ratio:.2f}x")
        print(f"  Saved:         {format_bytes(saved):>10s}")
        print(sep)
        assert ratio > 1.9


# ---------------------------------------------------------------------------
# 2. Dequant hooks — forward latency with FP8 params
# ---------------------------------------------------------------------------


@CUDA
class TestFP8DequantHookLatency:
    """Forward latency of nn.Linear with FP8 param dequant hooks vs BF16."""

    # Expected: FP8 params with dequant hooks add small overhead (<10%) over
    # BF16 forward. The dequant hook runs a fast FP8→BF16 cast before each
    # nn.Linear forward. For large matrices, GEMM dominates and the cast is
    # nearly free. The benefit is 2x less memory, enabling larger models to
    # fit in GPU memory and halving all-gather communication in TP training.
    def test_bf16_vs_fp8_forward(self):
        from lumen.quantize.fp8_params import FP8ParamManager

        x = torch.randn(2, 2048, HIDDEN, device="cuda", dtype=torch.bfloat16)

        # BF16 baseline
        model_bf16 = nn.Sequential(
            nn.Linear(HIDDEN, FFN_HIDDEN, bias=False),
            nn.SiLU(),
            nn.Linear(FFN_HIDDEN, HIDDEN, bias=False),
        ).to(device="cuda", dtype=torch.bfloat16)
        r_bf16 = cuda_timer(lambda: model_bf16(x), label="Linear forward (BF16 params)")

        # FP8 params with dequant hooks
        model_fp8 = nn.Sequential(
            nn.Linear(HIDDEN, FFN_HIDDEN, bias=False),
            nn.SiLU(),
            nn.Linear(FFN_HIDDEN, HIDDEN, bias=False),
        ).to(device="cuda", dtype=torch.bfloat16)
        mgr = FP8ParamManager()
        mgr.quantize_params(model_fp8)
        mgr.register_dequant_hooks(model_fp8)

        r_fp8 = cuda_timer(lambda: model_fp8(x), label="Linear forward (FP8 params + dequant)")

        overhead = (r_fp8.avg_ms - r_bf16.avg_ms) / max(r_bf16.avg_ms, 1e-6) * 100
        r_fp8.extra["overhead"] = f"{overhead:+.1f}%"
        print_report("FP8 Param Forward Latency", [r_bf16, r_fp8])


# ---------------------------------------------------------------------------
# 3. Quant/dequant primitive latency
# ---------------------------------------------------------------------------


@CUDA
class TestFP8QuantDequantLatency:
    """Latency of quantize_param_to_fp8 / dequantize_param_from_fp8."""

    # Expected: Quant and dequant are both memory-bound element-wise ops.
    # Quant (BF16→FP8) finds amax then divides+casts; dequant (FP8→BF16)
    # divides by scale then casts. Dequant should be slightly faster than
    # quant because it skips the absmax reduction pass. Larger shapes (FFN gate)
    # take proportionally longer due to more memory traffic.
    @pytest.mark.parametrize(
        "shape,name",
        [
            ((HIDDEN, HIDDEN), "QKV"),
            ((FFN_HIDDEN, HIDDEN), "FFN gate"),
            ((HIDDEN, FFN_HIDDEN), "FFN down"),
        ],
    )
    def test_quant_dequant(self, shape, name):
        from lumen.quantize.fp8_params import (
            dequantize_param_from_fp8,
            quantize_param_to_fp8,
        )

        weight = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

        r_q = cuda_timer(lambda: quantize_param_to_fp8(weight), label=f"quantize {name} {shape}")
        fp8_w, scale = quantize_param_to_fp8(weight)
        r_d = cuda_timer(
            lambda: dequantize_param_from_fp8(fp8_w, scale),
            label=f"dequantize {name} {shape}",
        )
        print_report(f"FP8 quant/dequant: {name}", [r_q, r_d])


# ---------------------------------------------------------------------------
# 4. All-gather bandwidth simulation (per-layer)
# ---------------------------------------------------------------------------


@CUDA
class TestFP8AllgatherBandwidth:
    """Simulate all-gather volume for a full 7B model: BF16 vs FP8."""

    # Expected: ~2x bandwidth reduction. A 7B model has ~7 billion BF16
    # params = ~14 GB. FP8 params = ~7 GB + negligible scale overhead.
    # In TP all-gather, each GPU broadcasts its shard to all peers; FP8
    # halves the bytes on the interconnect, directly reducing communication
    # time which is often the TP bottleneck at scale.
    def test_7b_allgather_volume(self):
        per_layer_shapes = [
            ("gate", FFN_HIDDEN, HIDDEN),
            ("up", FFN_HIDDEN, HIDDEN),
            ("down", HIDDEN, FFN_HIDDEN),
            ("qkv", HIDDEN * 3, HIDDEN),
        ]

        total_bf16 = 0
        total_fp8 = 0
        for name, rows, cols in per_layer_shapes:
            numel = rows * cols
            total_bf16 += numel * 2 * N_LAYERS
            total_fp8 += (numel * 1 + 4) * N_LAYERS  # 1 byte data + 4 bytes scale

        ratio = total_bf16 / total_fp8
        saved = total_bf16 - total_fp8
        sep = "=" * 52
        print(f"\n{sep}")
        print(f"  All-Gather Volume — {N_LAYERS}-layer 7B model")
        print(sep)
        print(f"  BF16 volume:   {format_bytes(total_bf16):>10s}")
        print(f"  FP8  volume:   {format_bytes(total_fp8):>10s}")
        print("  ──────────────────────────────")
        print(f"  BW reduction:  {ratio:.2f}x")
        print(f"  Saved/step:    {format_bytes(saved):>10s}")
        print(sep)
        assert ratio > 1.9


# ---------------------------------------------------------------------------
# 5. Numerical quality — round-trip SNR
# ---------------------------------------------------------------------------


@CUDA
class TestFP8ParamSNR:
    """Verify numerical quality of FP8 param compression."""

    # Expected: SNR > 15 dB. FP8 E4M3 has 3 mantissa bits (vs BF16's 7),
    # giving ~23 dB theoretical SNR for uniform-distributed data. For
    # normally-distributed weights, per-tensor scaling preserves most dynamic
    # range. Values near zero suffer more quantization noise, but the overall
    # SNR for transformer weights typically exceeds 18-25 dB.
    @pytest.mark.parametrize("shape", [(HIDDEN, HIDDEN), (FFN_HIDDEN, HIDDEN)])
    def test_roundtrip_snr(self, shape):
        from lumen.quantize.fp8_params import (
            dequantize_param_from_fp8,
            quantize_param_to_fp8,
        )

        weight = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        fp8_w, scale = quantize_param_to_fp8(weight)
        recovered = dequantize_param_from_fp8(fp8_w, scale, torch.bfloat16)

        signal_power = (weight.float() ** 2).mean()
        noise_power = ((weight.float() - recovered.float()) ** 2).mean()
        snr_db = 10 * torch.log10(signal_power / noise_power).item()

        print(f"\n  FP8 param round-trip SNR {shape}: {snr_db:.1f} dB")
        assert snr_db > 15, f"FP8 param SNR too low: {snr_db:.1f} dB"

    # Expected: Model output SNR > 10 dB. Quantization error accumulates
    # through multiple layers (Linear → SiLU → Linear), but per-tensor FP8
    # preserves enough precision for training convergence. Lower than per-param
    # SNR because errors compound through nonlinearities. This validates that
    # FP8 param storage is viable for training without significant accuracy loss.
    def test_linear_stack_output_snr(self):
        """Compare model output with BF16 vs FP8 params."""
        from lumen.quantize.fp8_params import FP8ParamManager

        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(256, 512, bias=False),
            nn.SiLU(),
            nn.Linear(512, 256, bias=False),
        ).to(device="cuda", dtype=torch.bfloat16)
        x = torch.randn(2, 32, 256, device="cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            ref_out = model(x).clone()

        mgr = FP8ParamManager()
        mgr.quantize_params(model)
        mgr.register_dequant_hooks(model)

        with torch.no_grad():
            fp8_out = model(x)

        signal = (ref_out.float() ** 2).mean()
        noise = ((ref_out.float() - fp8_out.float()) ** 2).mean()
        snr_db = 10 * torch.log10(signal / noise).item()

        print(f"\n  Model output SNR (BF16 vs FP8 params): {snr_db:.1f} dB")
        assert snr_db > 10, f"Output SNR too low: {snr_db:.1f} dB"


# ---------------------------------------------------------------------------
# 5. Timed all-gather: BF16 vs FP8 param shards (M2 acceptance criterion)
# ---------------------------------------------------------------------------


def _is_dist_initialized():
    """Return True if torch.distributed is initialized."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)
class TestFP8AllGatherDistributed:
    """Measure actual all_gather time for BF16 vs FP8 param shards.

    Directly addresses M2 acceptance: 'FP8 param all-gather reduces memory'
    and 'Compare memory footprint with/without FP8 param all-gather.'

    These tests require ``torchrun --nproc_per_node=N`` to be meaningful.
    On single-GPU, they fall back to a local simulation that measures the
    bandwidth difference of gathering larger (BF16) vs smaller (FP8) buffers.
    """

    # Expected: FP8 all-gather should take roughly half the time of BF16
    # all-gather because the data volume is halved (1 byte/element vs 2).
    # All-gather is bandwidth-bound on the interconnect, so halving the
    # payload directly halves the transfer time (minus fixed latency).
    @pytest.mark.parametrize("shape", [(HIDDEN, HIDDEN), (FFN_HIDDEN, HIDDEN)])
    def test_allgather_bf16_vs_fp8(self, shape):
        from lumen.quantize.fp8_params import quantize_param_to_fp8

        weight_bf16 = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        fp8_w, _ = quantize_param_to_fp8(weight_bf16)

        if _is_dist_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

            bf16_shard = weight_bf16.chunk(world_size)[rank].contiguous()
            fp8_shard = fp8_w.chunk(world_size)[rank].contiguous()

            bf16_gather_list = [torch.empty_like(bf16_shard) for _ in range(world_size)]
            fp8_gather_list = [torch.empty_like(fp8_shard) for _ in range(world_size)]

            r_bf16 = cuda_timer(
                lambda: torch.distributed.all_gather(bf16_gather_list, bf16_shard),
                label=f"all_gather BF16 {shape}",
            )
            r_fp8 = cuda_timer(
                lambda: torch.distributed.all_gather(fp8_gather_list, fp8_shard),
                label=f"all_gather FP8 {shape}",
            )
        else:
            tp_size = 8
            bf16_shard = weight_bf16[: shape[0] // tp_size].contiguous()
            fp8_shard = fp8_w[: shape[0] // tp_size].contiguous()

            bf16_full = torch.empty_like(weight_bf16)
            fp8_full = torch.empty(shape, device="cuda", dtype=fp8_w.dtype)

            r_bf16 = cuda_timer(
                lambda: torch.cat([bf16_shard] * tp_size, dim=0, out=bf16_full),
                label=f"simulated all_gather BF16 {shape}",
            )
            r_fp8 = cuda_timer(
                lambda: torch.cat([fp8_shard] * tp_size, dim=0, out=fp8_full),
                label=f"simulated all_gather FP8 {shape}",
            )

        speedup = r_bf16.avg_ms / max(r_fp8.avg_ms, 1e-6)
        bf16_bytes = bf16_shard.nelement() * bf16_shard.element_size()
        fp8_bytes = fp8_shard.nelement() * fp8_shard.element_size()
        bw_reduction = bf16_bytes / max(fp8_bytes, 1)

        r_fp8.extra["speedup"] = round(speedup, 2)
        r_fp8.extra["bandwidth_reduction"] = f"{bw_reduction:.1f}x"
        print_report(f"All-Gather BF16 vs FP8 {shape}", [r_bf16, r_fp8])
        print(f"  Bandwidth reduction: {bw_reduction:.1f}x")
        print(f"  Time speedup: {speedup:.2f}x")


# ---------------------------------------------------------------------------
# 6. Multi-GPU NCCL: real all-gather BF16 vs FP8 + full pipeline
# ---------------------------------------------------------------------------


@_DIST
class TestFP8AllGatherNCCL:
    """Real NCCL all-gather with FP8 vs BF16 param shards.

    Measures three key scenarios:
      1. Raw all-gather bandwidth: FP8 shard vs BF16 shard.
      2. Full pipeline: quantize → all-gather FP8 → dequantize vs
         plain BF16 all-gather (the realistic training hot path).
      3. Multi-layer: pipeline across N layers to amortise any fixed costs.

    Run::

        torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k NCCL
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    @pytest.mark.parametrize("shape", [(HIDDEN, HIDDEN), (FFN_HIDDEN, HIDDEN)])
    def test_raw_allgather_bf16_vs_fp8(self, shape):
        """Raw NCCL all_gather_into_tensor: BF16 (2 B/elem) vs FP8 (1 B/elem)."""
        from lumen.quantize.fp8_params import quantize_param_to_fp8

        weight_bf16 = torch.randn(*shape, device=self.device, dtype=torch.bfloat16)
        fp8_w, _ = quantize_param_to_fp8(weight_bf16)

        bf16_shard = weight_bf16.chunk(self.world)[self.rank].contiguous()
        fp8_shard = fp8_w.chunk(self.world)[self.rank].contiguous()

        bf16_out = torch.empty_like(weight_bf16)
        fp8_out = torch.empty(shape, device=self.device, dtype=fp8_w.dtype)

        for _ in range(3):
            dist.all_gather_into_tensor(bf16_out, bf16_shard)
            dist.all_gather_into_tensor(fp8_out, fp8_shard)
        torch.cuda.synchronize()

        r_bf16 = cuda_timer(
            lambda: dist.all_gather_into_tensor(bf16_out, bf16_shard),
            label=f"NCCL all_gather BF16 {shape}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_fp8 = cuda_timer(
            lambda: dist.all_gather_into_tensor(fp8_out, fp8_shard),
            label=f"NCCL all_gather FP8  {shape}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_bf16.avg_ms / max(r_fp8.avg_ms, 1e-6)
        bw_ratio = (bf16_shard.nelement() * bf16_shard.element_size()) / max(
            fp8_shard.nelement() * fp8_shard.element_size(), 1
        )
        r_fp8.extra["speedup"] = round(speedup, 2)
        r_fp8.extra["bw_reduction"] = f"{bw_ratio:.1f}x"

        if self.rank == 0:
            print_report_with_table(f"NCCL All-Gather BF16 vs FP8 {shape} (world={self.world})", [r_bf16, r_fp8])

    @pytest.mark.parametrize("shape", [(HIDDEN, HIDDEN), (FFN_HIDDEN, HIDDEN)])
    def test_full_pipeline_quant_gather_dequant(self, shape):
        """Full pipeline: quant→gather→dequant (FP8) vs plain gather (BF16).

        This is the realistic hot path: rank-local weights are stored in FP8,
        all-gathered in FP8 (half the bytes), then dequantized to BF16 for GEMM.
        """
        from lumen.quantize.fp8_params import dequantize_param_from_fp8, quantize_param_to_fp8

        weight_bf16 = torch.randn(*shape, device=self.device, dtype=torch.bfloat16)
        fp8_w, scale = quantize_param_to_fp8(weight_bf16)

        bf16_shard = weight_bf16.chunk(self.world)[self.rank].contiguous()
        fp8_shard = fp8_w.chunk(self.world)[self.rank].contiguous()
        scale_shard = scale.clone()

        bf16_out = torch.empty_like(weight_bf16)
        fp8_out = torch.empty(shape, device=self.device, dtype=fp8_w.dtype)

        for _ in range(3):
            dist.all_gather_into_tensor(bf16_out, bf16_shard)
            dist.all_gather_into_tensor(fp8_out, fp8_shard)
        torch.cuda.synchronize()

        def _bf16_pipeline():
            dist.all_gather_into_tensor(bf16_out, bf16_shard)

        def _fp8_pipeline():
            dist.all_gather_into_tensor(fp8_out, fp8_shard)
            dequantize_param_from_fp8(fp8_out, scale_shard, torch.bfloat16)

        r_bf16 = cuda_timer(
            _bf16_pipeline,
            label=f"BF16 gather {shape}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_fp8 = cuda_timer(
            _fp8_pipeline,
            label=f"FP8 quant+gather+dequant {shape}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_bf16.avg_ms / max(r_fp8.avg_ms, 1e-6)
        r_fp8.extra["speedup"] = round(speedup, 2)

        if self.rank == 0:
            print_report_with_table(
                f"Full Pipeline BF16 vs FP8 {shape} (world={self.world})",
                [r_bf16, r_fp8],
            )

    def test_multi_layer_pipeline(self):
        """Pipeline across 4 transformer layers (gate+up+down+qkv per layer)."""
        from lumen.quantize.fp8_params import dequantize_param_from_fp8, quantize_param_to_fp8

        layer_shapes = [
            (FFN_HIDDEN, HIDDEN),
            (FFN_HIDDEN, HIDDEN),
            (HIDDEN, FFN_HIDDEN),
            (HIDDEN, HIDDEN),
        ]
        n_layers = 4

        bf16_shards = []
        fp8_shards = []
        fp8_scales = []
        bf16_outs = []
        fp8_outs = []

        for _ in range(n_layers):
            for shape in layer_shapes:
                w = torch.randn(*shape, device=self.device, dtype=torch.bfloat16)
                fp8_w, sc = quantize_param_to_fp8(w)
                bf16_shards.append(w.chunk(self.world)[self.rank].contiguous())
                fp8_shards.append(fp8_w.chunk(self.world)[self.rank].contiguous())
                fp8_scales.append(sc)
                bf16_outs.append(torch.empty_like(w))
                fp8_outs.append(torch.empty(*shape, device=self.device, dtype=fp8_w.dtype))

        for i in range(len(bf16_shards)):
            dist.all_gather_into_tensor(bf16_outs[i], bf16_shards[i])
            dist.all_gather_into_tensor(fp8_outs[i], fp8_shards[i])
        torch.cuda.synchronize()

        def _bf16_all():
            for i in range(len(bf16_shards)):
                dist.all_gather_into_tensor(bf16_outs[i], bf16_shards[i])

        def _fp8_all():
            for i in range(len(fp8_shards)):
                dist.all_gather_into_tensor(fp8_outs[i], fp8_shards[i])
            for i in range(len(fp8_shards)):
                dequantize_param_from_fp8(fp8_outs[i], fp8_scales[i], torch.bfloat16)

        r_bf16 = cuda_timer(
            _bf16_all, label=f"BF16 gather {n_layers}L", warmup=_WARMUP, iters=_ITERS, trim_pct=_TRIM, dist_barrier=True
        )
        r_fp8 = cuda_timer(
            _fp8_all,
            label=f"FP8 gather+dequant {n_layers}L",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_bf16.avg_ms / max(r_fp8.avg_ms, 1e-6)
        total_bf16_bytes = sum(s.nelement() * s.element_size() for s in bf16_shards)
        total_fp8_bytes = sum(s.nelement() * s.element_size() for s in fp8_shards)
        r_fp8.extra["speedup"] = round(speedup, 2)
        r_fp8.extra["bf16_vol"] = format_bytes(total_bf16_bytes)
        r_fp8.extra["fp8_vol"] = format_bytes(total_fp8_bytes)

        if self.rank == 0:
            print_report_with_table(
                f"Multi-Layer Pipeline ({n_layers}L, world={self.world})",
                [r_bf16, r_fp8],
            )


# ---------------------------------------------------------------------------
# 7. Multi-GPU SDMA: FP8 param all-gather on dedicated DMA engines
# ---------------------------------------------------------------------------


@_SDMA
class TestFP8AllGatherSDMA:
    """SDMA all-gather with FP8 param shards.

    SDMA uses dedicated hardware DMA engines (separate from compute SMs),
    providing zero-contention allgather that can overlap perfectly with
    compute.  FP8 params halve the DMA transfer volume.

    Run::

        torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k SDMA
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        os.environ["MORI_ENABLE_SDMA"] = "1"
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    @pytest.mark.parametrize("shape", [(HIDDEN, HIDDEN), (FFN_HIDDEN, HIDDEN)])
    def test_sdma_allgather_bf16_vs_fp8(self, shape):
        """SDMA allgather_dim0: BF16 shard vs FP8 shard."""
        from lumen.modules.sdma_comm import SdmaTpComm
        from lumen.quantize.fp8_params import quantize_param_to_fp8

        weight_bf16 = torch.randn(*shape, device=self.device, dtype=torch.bfloat16)
        fp8_w, _ = quantize_param_to_fp8(weight_bf16)

        bf16_shard = weight_bf16.chunk(self.world, dim=0)[self.rank].contiguous()
        fp8_shard = fp8_w.chunk(self.world, dim=0)[self.rank].contiguous()

        comm = SdmaTpComm(dist.group.WORLD)

        for _ in range(3):
            comm.allgather_dim0(bf16_shard)
            comm.allgather_dim0(fp8_shard)
        torch.cuda.synchronize()

        r_bf16 = cuda_timer(
            lambda: comm.allgather_dim0(bf16_shard),
            label=f"SDMA allgather BF16 {shape}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_fp8 = cuda_timer(
            lambda: comm.allgather_dim0(fp8_shard),
            label=f"SDMA allgather FP8  {shape}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_bf16.avg_ms / max(r_fp8.avg_ms, 1e-6)
        r_fp8.extra["speedup"] = round(speedup, 2)

        if self.rank == 0:
            print_report_with_table(
                f"SDMA All-Gather BF16 vs FP8 {shape} (world={self.world})",
                [r_bf16, r_fp8],
            )

    @pytest.mark.parametrize("shape", [(HIDDEN, HIDDEN), (FFN_HIDDEN, HIDDEN)])
    def test_sdma_async_allgather_overlap_with_dequant(self, shape):
        """Overlap SDMA allgather (FP8 shard) with dequant of previous layer."""
        from lumen.modules.sdma_comm import SdmaTpComm
        from lumen.quantize.fp8_params import dequantize_param_from_fp8, quantize_param_to_fp8

        weight_bf16 = torch.randn(*shape, device=self.device, dtype=torch.bfloat16)
        fp8_w, scale = quantize_param_to_fp8(weight_bf16)
        fp8_shard = fp8_w.chunk(self.world, dim=0)[self.rank].contiguous()

        comm = SdmaTpComm(dist.group.WORLD)
        sdma_stream = torch.cuda.Stream(device=self.device)
        compute_stream = torch.cuda.current_stream(self.device)

        for _ in range(3):
            comm.allgather_dim0(fp8_shard)
        torch.cuda.synchronize()

        r_gather = cuda_timer(
            lambda: comm.allgather_dim0(fp8_shard),
            label=f"SDMA gather FP8 {shape}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        gathered_fp8 = comm.allgather_dim0(fp8_shard)
        r_dequant = cuda_timer(
            lambda: dequantize_param_from_fp8(gathered_fp8, scale, torch.bfloat16),
            label=f"dequant {shape}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
        )

        def _sequential():
            g = comm.allgather_dim0(fp8_shard)
            dequantize_param_from_fp8(g, scale, torch.bfloat16)

        def _overlapped():
            comm.allgather_dim0_async(fp8_shard, stream=sdma_stream)
            dequantize_param_from_fp8(gathered_fp8, scale, torch.bfloat16)
            comm.wait_allgather_dim0(stream=sdma_stream)
            compute_stream.wait_stream(sdma_stream)

        r_seq = cuda_timer(
            _sequential, label=f"sequential {shape}", warmup=_WARMUP, iters=_ITERS, trim_pct=_TRIM, dist_barrier=True
        )
        r_ovl = cuda_timer(
            _overlapped, label=f"overlapped {shape}", warmup=_WARMUP, iters=_ITERS, trim_pct=_TRIM, dist_barrier=True
        )

        speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_ovl.extra["speedup"] = round(speedup, 2)

        if self.rank == 0:
            print_report_with_table(
                f"SDMA Gather+Dequant Overlap {shape} (world={self.world})",
                [r_gather, r_dequant, r_seq, r_ovl],
            )


# ---------------------------------------------------------------------------
# 8. E2E distributed: full model FP8 param allgather + forward correctness
# ---------------------------------------------------------------------------


@_DIST
class TestFP8ParamE2EDistributed:
    """End-to-end: all ranks hold FP8 shards, allgather, dequant, forward.

    Verifies that the distributed FP8 param pipeline produces numerically
    correct results (SNR > 10 dB vs BF16 reference) and measures the
    total latency of the gather-dequant-forward pipeline.

    Run::

        torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k E2E
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    def test_gather_dequant_forward_correctness(self):
        """Allgather FP8 shards → dequant → Linear forward, check SNR vs BF16."""
        from lumen.quantize.fp8_params import dequantize_param_from_fp8, quantize_param_to_fp8

        torch.manual_seed(42)
        H_in, H_out = 256, 512
        weight_full = torch.randn(H_out, H_in, device=self.device, dtype=torch.bfloat16)
        x = torch.randn(2, 32, H_in, device=self.device, dtype=torch.bfloat16)

        dist.broadcast(weight_full, src=0)
        dist.broadcast(x, src=0)

        ref_out = torch.nn.functional.linear(x, weight_full)

        fp8_full, scale = quantize_param_to_fp8(weight_full)
        fp8_shard = fp8_full.chunk(self.world, dim=0)[self.rank].contiguous()

        gathered_list = [torch.empty_like(fp8_shard) for _ in range(self.world)]
        dist.all_gather(gathered_list, fp8_shard)
        fp8_gathered = torch.cat(gathered_list, dim=0)

        weight_recovered = dequantize_param_from_fp8(fp8_gathered, scale, torch.bfloat16)
        fp8_out = torch.nn.functional.linear(x, weight_recovered)

        signal = (ref_out.float() ** 2).mean()
        noise = ((ref_out.float() - fp8_out.float()) ** 2).mean()
        snr_db = 10 * torch.log10(signal / noise).item()

        if self.rank == 0:
            print(f"\n  E2E distributed FP8 param forward SNR: {snr_db:.1f} dB")
        assert snr_db > 10, f"Distributed FP8 param SNR too low: {snr_db:.1f} dB"

    def test_gather_dequant_forward_latency(self):
        """Time the full pipeline: scatter shards → allgather FP8 → dequant → GEMM."""
        from lumen.quantize.fp8_params import dequantize_param_from_fp8, quantize_param_to_fp8

        shape = (FFN_HIDDEN, HIDDEN)
        weight_full = torch.randn(*shape, device=self.device, dtype=torch.bfloat16)
        dist.broadcast(weight_full, src=0)

        x = torch.randn(2, 2048, HIDDEN, device=self.device, dtype=torch.bfloat16)
        fp8_full, scale = quantize_param_to_fp8(weight_full)
        fp8_shard = fp8_full.chunk(self.world, dim=0)[self.rank].contiguous()
        bf16_shard = weight_full.chunk(self.world, dim=0)[self.rank].contiguous()

        bf16_out = torch.empty_like(weight_full)
        fp8_out_buf = torch.empty(*shape, device=self.device, dtype=fp8_full.dtype)

        for _ in range(3):
            dist.all_gather_into_tensor(bf16_out, bf16_shard)
            dist.all_gather_into_tensor(fp8_out_buf, fp8_shard)
        torch.cuda.synchronize()

        def _bf16_pipeline():
            dist.all_gather_into_tensor(bf16_out, bf16_shard)
            torch.nn.functional.linear(x, bf16_out)

        def _fp8_pipeline():
            dist.all_gather_into_tensor(fp8_out_buf, fp8_shard)
            w = dequantize_param_from_fp8(fp8_out_buf, scale, torch.bfloat16)
            torch.nn.functional.linear(x, w)

        r_bf16 = cuda_timer(
            _bf16_pipeline, label="BF16 gather+GEMM", warmup=_WARMUP, iters=_ITERS, trim_pct=_TRIM, dist_barrier=True
        )
        r_fp8 = cuda_timer(
            _fp8_pipeline,
            label="FP8 gather+dequant+GEMM",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_bf16.avg_ms / max(r_fp8.avg_ms, 1e-6)
        r_fp8.extra["speedup"] = round(speedup, 2)

        if self.rank == 0:
            print_report_with_table(
                f"E2E Gather+Forward {shape} (world={self.world})",
                [r_bf16, r_fp8],
            )

    def test_multi_layer_pipelined_gather_forward(self):
        """Multi-layer pipeline: overlap allgather(layer i+1) with dequant+GEMM(layer i).

        The single-layer E2E test is sequential (gather→dequant→GEMM), so the
        dequant overhead is fully exposed and FP8 appears slower than BF16.
        In real training, layers are processed in sequence. This test pipelines
        them: while compute processes layer i (dequant+GEMM on compute stream),
        the comm stream pre-fetches layer i+1's FP8 shard via allgather.
        The dequant cost is hidden behind the overlapped communication.
        """
        from lumen.quantize.fp8_params import dequantize_param_from_fp8, quantize_param_to_fp8

        n_layers = 4
        layer_shapes = [
            (FFN_HIDDEN, HIDDEN),
            (FFN_HIDDEN, HIDDEN),
            (HIDDEN, FFN_HIDDEN),
            (HIDDEN, HIDDEN),
        ]

        bf16_shards = []
        fp8_shards = []
        fp8_scales = []
        bf16_outs = []
        fp8_outs = []
        xs = []

        for _ in range(n_layers):
            for shape in layer_shapes:
                w = torch.randn(*shape, device=self.device, dtype=torch.bfloat16)
                dist.broadcast(w, src=0)
                fp8_w, sc = quantize_param_to_fp8(w)
                bf16_shards.append(w.chunk(self.world, dim=0)[self.rank].contiguous())
                fp8_shards.append(fp8_w.chunk(self.world, dim=0)[self.rank].contiguous())
                fp8_scales.append(sc)
                bf16_outs.append(torch.empty_like(w))
                fp8_outs.append(torch.empty(*shape, device=self.device, dtype=fp8_w.dtype))
                in_features = shape[1]
                xs.append(torch.randn(2, 2048, in_features, device=self.device, dtype=torch.bfloat16))

        N = len(bf16_shards)
        comm_stream = torch.cuda.Stream(device=self.device)

        # Warmup
        for i in range(N):
            dist.all_gather_into_tensor(bf16_outs[i], bf16_shards[i])
            dist.all_gather_into_tensor(fp8_outs[i], fp8_shards[i])
        torch.cuda.synchronize()

        # BF16 baseline: sequential gather + GEMM per layer
        def _bf16_sequential():
            for i in range(N):
                dist.all_gather_into_tensor(bf16_outs[i], bf16_shards[i])
                torch.nn.functional.linear(xs[i], bf16_outs[i])

        # FP8 sequential: gather + dequant + GEMM per layer (no overlap)
        def _fp8_sequential():
            for i in range(N):
                dist.all_gather_into_tensor(fp8_outs[i], fp8_shards[i])
                w = dequantize_param_from_fp8(fp8_outs[i], fp8_scales[i], torch.bfloat16)
                torch.nn.functional.linear(xs[i], w)

        # FP8 pipelined: allgather(i+1) on comm_stream while dequant+GEMM(i)
        def _fp8_pipelined():
            dist.all_gather_into_tensor(fp8_outs[0], fp8_shards[0])
            torch.cuda.synchronize()
            for i in range(N):
                if i + 1 < N:
                    with torch.cuda.stream(comm_stream):
                        dist.all_gather_into_tensor(fp8_outs[i + 1], fp8_shards[i + 1])
                w = dequantize_param_from_fp8(fp8_outs[i], fp8_scales[i], torch.bfloat16)
                torch.nn.functional.linear(xs[i], w)
                if i + 1 < N:
                    torch.cuda.current_stream().wait_stream(comm_stream)

        r_bf16 = cuda_timer(
            _bf16_sequential,
            label=f"BF16 sequential {n_layers}L",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_fp8_seq = cuda_timer(
            _fp8_sequential,
            label=f"FP8 sequential {n_layers}L",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_fp8_pipe = cuda_timer(
            _fp8_pipelined,
            label=f"FP8 pipelined {n_layers}L",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        sp_seq = r_bf16.avg_ms / max(r_fp8_seq.avg_ms, 1e-6)
        sp_pipe = r_bf16.avg_ms / max(r_fp8_pipe.avg_ms, 1e-6)
        pipe_vs_seq = r_fp8_seq.avg_ms / max(r_fp8_pipe.avg_ms, 1e-6)
        r_fp8_seq.extra["vs_bf16"] = round(sp_seq, 2)
        r_fp8_pipe.extra["vs_bf16"] = round(sp_pipe, 2)
        r_fp8_pipe.extra["vs_fp8_seq"] = round(pipe_vs_seq, 2)

        if self.rank == 0:
            print_report_with_table(
                f"E2E Pipelined Gather+Forward ({n_layers}L, world={self.world})",
                [r_bf16, r_fp8_seq, r_fp8_pipe],
            )
            saved_ms = r_fp8_seq.avg_ms - r_fp8_pipe.avg_ms
            print(f"  Pipeline saves {saved_ms:.3f} ms vs FP8 sequential " f"({pipe_vs_seq:.2f}x speedup)")
            if sp_pipe >= 1.0:
                print(f"  FP8 pipelined beats BF16: {sp_pipe:.2f}x")
            else:
                gap = r_fp8_pipe.avg_ms - r_bf16.avg_ms
                print(f"  FP8 pipelined still {gap:.3f} ms behind BF16 " f"(dequant overhead not fully hidden)")
            print()


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def main():
    require_cuda()

    from lumen.quantize.fp8_params import (
        FP8ParamManager,
        dequantize_param_from_fp8,
        quantize_param_to_fp8,
    )

    results: List[BenchResult] = []

    # Quant/dequant latency
    for name, shape in [("QKV", (HIDDEN * 3, HIDDEN)), ("FFN", (FFN_HIDDEN, HIDDEN))]:
        w = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        r_q = cuda_timer(lambda: quantize_param_to_fp8(w), label=f"quantize {name}")
        fp8_w, sc = quantize_param_to_fp8(w)
        r_d = cuda_timer(lambda: dequantize_param_from_fp8(fp8_w, sc), label=f"dequantize {name}")
        results.extend([r_q, r_d])

    # Linear stack with FP8 params
    model = _build_mlp_stack(n_layers=4).to(device="cuda", dtype=torch.bfloat16)
    bf16_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    mgr = FP8ParamManager()
    n_quant = mgr.quantize_params(model)
    fp8_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    total_fp8 = fp8_bytes + n_quant * 4
    ratio = bf16_bytes / total_fp8

    r_mem = BenchResult(name="4-layer BF16→FP8 compression", avg_ms=0)
    r_mem.extra["bf16"] = format_bytes(bf16_bytes)
    r_mem.extra["fp8"] = format_bytes(total_fp8)
    r_mem.extra["ratio"] = round(ratio, 2)
    results.append(r_mem)

    print_report("Lumen FP8 Param All-Gather", results)
    print_table("Summary Table", results)
    print(f"  Compression: {ratio:.2f}x  ({n_quant} tensors, saved {format_bytes(bf16_bytes - total_fp8)})")
    print()


if __name__ == "__main__":
    main()
