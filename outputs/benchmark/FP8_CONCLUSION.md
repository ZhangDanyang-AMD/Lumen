# Lumen FP8 Training — Consolidated Conclusions

**Date**: 2026-04-08
**Hardware**: 8x AMD MI300X (256 GB VRAM each), ROCm 7.0
**Frameworks**: TRL 0.29.1 + FSDP1, VERL 0.7.1 + FSDP2 + SGLang 0.5.9

---

## Executive Summary

After comprehensive testing across 6 model/framework configurations, 3 architectural fixes, and 20+ benchmark runs, the conclusion is:

**Lumen FP8 via `quant.enable()` does NOT save memory or improve speed on MI300X for GRPO training at any tested model scale (0.5B through 32B).** The overhead of quantize/dequantize operations and FP8 GEMM compute exceeds any potential savings.

The one exception: **FP8 Attention (dpa)** reduces memory by 11% on TRL+FSDP1, but with a 13x speed penalty.

---

## Complete Benchmark Matrix

### Memory Impact (Peak GPU Memory)

| Model | Framework | BF16 Baseline | FP8 | Delta | Verdict |
|-------|-----------|--------------|-----|-------|---------|
| Qwen2-0.5B | TRL+FSDP1 | 7.11 GB | 7.11 GB | 0% | NEUTRAL |
| Llama-3.1-8B | TRL+FSDP1 | 34.57 GB | 38.85 GB | **+12%** | WORSE |
| Llama-3.1-8B | VERL+FSDP2 | 11.76 GB (alloc) / 18.91 GB (res) | 12.22 GB / 18.91 GB | +3.9% / 0% | WORSE/SAME |
| Qwen2.5-32B | TRL+FSDP1 | 124.18 GB | 127.13 GB | +2% | WORSE |
| Qwen2.5-32B | VERL+FSDP2 | 38.43 GB (alloc) / 42.97 GB (res) | 38.43 GB / 44.29 GB | 0% / +3.1% | SAME/WORSE |

### Speed Impact (Actor Update Time)

| Model | Framework | BF16 | FP8 | Slowdown |
|-------|-----------|------|-----|----------|
| Llama-3.1-8B | TRL+FSDP1 | 11.1s | 126.7s | **11.5x** |
| Llama-3.1-8B | VERL+FSDP2 | 7.39s | 11.23s | **1.52x** |
| Qwen2.5-32B | TRL+FSDP1 | 13.4s | 47.7s | **3.6x** |
| Qwen2.5-32B | VERL+FSDP2 | 9.04s | 16.76s | **1.85x** |

### Throughput Impact

| Model | Framework | BF16 | FP8 | Delta |
|-------|-----------|------|-----|-------|
| Llama-3.1-8B | VERL+FSDP2 | 774 tok/s | 593 tok/s | **-23.4%** |
| Qwen2.5-32B | VERL+FSDP2 | 128 tok/s | 87 tok/s | **-32.4%** |

---

## Root Cause Analysis

### Why FP8 is Counterproductive

1. **`quant.enable()` only quantizes GEMM compute** — it does NOT reduce parameter storage, optimizer state, or activation memory.

2. **Quantize/dequantize overhead exceeds savings**: Each forward+backward pass requires additional quantize (BF16→FP8) and dequantize (FP8→BF16) operations per linear layer. At 8B-32B scales, these GEMMs are not large enough for FP8 compute savings to offset this overhead.

3. **FSDP1 Triton fallback (10-20x penalty)**: FSDP1's mixed precision upcasts params to FP32. AITER FP8 kernels only support BF16/FP16 input, causing fallback to Python-based Triton — catastrophically slow.

4. **FSDP2 is better but still slower**: FSDP2 keeps activations in BF16, avoiding Triton fallback. But FP8 GEMM is intrinsically slower than BF16 GEMM at 8B/32B matrix sizes on MI300X (CK kernels active, no fallback, still slower).

5. **FP8 buffers add memory overhead**: `quant.enable()` creates FP8 quantize/dequantize buffers per linear layer, adding ~4 GB/GPU at 8B scale.

### FSDP1 vs FSDP2 Speed Comparison

| Metric | FSDP1 (TRL) | FSDP2 (VERL) | Improvement |
|--------|-------------|--------------|-------------|
| FP8 slowdown (8B) | 11.5x | 1.52x | **7.6x better** |
| FP8 slowdown (32B) | 3.6x | 1.85x | **1.9x better** |

FSDP2 eliminates the Triton fallback issue, making FP8 overhead far more reasonable — but still net-negative.

---

## Architectural Fixes Applied (2026-04-08)

Three previously-blocked features were fixed and benchmarked:

| Fix | What Was Done | Result |
|-----|---------------|--------|
| **FP8 Weight Cache** | Wired `store_weights_fp8()` into LumenConfig; caches FP8 weights to skip per-forward re-quantization | Actor -2.0%, Throughput +3.3% |
| **FP8 Activation Storage** | Extended `_apply_pre_quant` to set flag on `nn.Linear` (not just Lumen-native types) | No measurable effect (full FP8 path already saves activations as FP8) |
| **FP8 Param Gather** | Fixed AITER kernel crash by adding `.contiguous()` guard + `TORCH_CHECK` | Previously crashed; now works. Actor -1.7%, Throughput +1.4% |
| **All Three Combined** | Weight Cache + Act Store + Param Gather | **Actor -2.4%, Throughput +2.8%** vs FP8-only |

These fixes improve FP8 speed by ~3% but do not change the fundamental conclusion: FP8 is still slower than BF16.

---

## What Actually Works for Memory Optimization

| Strategy | Memory Saving | Speed Impact | Recommendation |
|----------|---------------|-------------|----------------|
| **LoRA r=16** (BF16) | **-48%** (34.57→17.83 GB at 8B) | **1.16x faster** | **BEST** — use for memory-constrained scenarios |
| **VERL+FSDP2** (vs TRL+FSDP1) | **-66%** (34.57→11.76 GB at 8B) | Comparable | **BEST** — async rollout architecture |
| **FP8 Attention (dpa)** | **-11%** (34.57→30.92 GB at 8B) | 13x slower | Use only if memory is the absolute constraint |
| **Lumen Norm** | **-3.9%** reserved memory | +5% throughput | Small but free improvement |
| FP8 Linear (quant.enable) | **+4-12%** (worse) | 1.5-11.5x slower | **DO NOT USE** |

---

## vs Expected Results (train_target.md)

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| FP8 memory (7B) | -5% to -10% | **+12%** (TRL) / +3.9% (VERL) | FAIL |
| FP8 memory (32B) | -10% to -15% | **0% to +3%** | FAIL |
| FP8 training speedup (7B) | small difference | **1.5-11.5x slower** | FAIL |
| FP8 training speedup (32B) | 1.3-1.5x | **1.5-1.85x slower** | FAIL |
| FP8 correctness | consistent reward | consistent reward | PASS |
| LoRA memory savings | ~78% (70B est.) | **48%** (8B) | PARTIAL |
| VERL integration | working | working (15+ patches) | PASS |

---

## Files Changed

### Code Fixes
- `lumen/config.py` — FP8 weight cache support, activation store for nn.Linear, Lumen Norm FSDP2 compat
- `lumen/ops/quantize/linear.py` — Weight contiguity guard
- `lumen/rl/verl/verl_entry.py` — New env vars, optimizer hook wiring
- `lumen/rl/verl/config.py` — `lumen_fp8_weight_cache` field
- `third_party/aiter/csrc/kernels/quant_kernels.cu` — Contiguity TORCH_CHECK
- `examples/rl/verl/run_grpo_fsdp2.sh` — New env vars

### Environment Variables for VERL Launcher
```bash
LUMEN_FP8=1                    # Enable FP8 linear quantization
LUMEN_FP8_ATTN=dpa             # FP8 attention (none/dpa/mha)
LUMEN_NORM=1                   # Lumen norm replacement
LUMEN_FP8_WEIGHT_CACHE=1       # FP8 weight caching (skip re-quant)
LUMEN_FP8_ACTIVATION_STORE=1   # FP8 activation storage
LUMEN_FP8_PARAM_GATHER=1       # FP8 parameter gather
```

---

## Recommendations

1. **Do NOT use `quant.enable()` / `apply_fp8_training()` for GRPO** on MI300X — it is counterproductive at all tested scales.
2. **Use LoRA + VERL+FSDP2** for maximum memory efficiency.
3. **FP8 Attention (dpa)** is the only FP8 feature that saves memory, but at a severe speed cost.
4. **For FP8 to become beneficial**, the following architectural changes would be needed:
   - True FP8 weight storage in FSDP (not just GEMM compute quantization) — blocked by PyTorch
   - FP8 GEMM performance improvement on MI300X (currently slower than BF16 GEMM at 8B/32B)
   - FP8 optimizer state (e.g., 8-bit Adam) to halve optimizer memory
   - FP8 communication (all-reduce/all-gather in FP8) to reduce FSDP comm volume
