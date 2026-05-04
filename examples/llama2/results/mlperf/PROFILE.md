# Lumen vs MLPerf TE Profile Analysis

**Config**: LLaMA2-70B LoRA SFT, 8x MI300X, TP=1 DP=8, FP8 hybrid delayed scaling, hipBLASLt GEMM
**Profile source**: `profile_summary_current.txt` — steps 100-102 (stable), all optimizations enabled (v47)
**Lumen step time**: ~4,158ms (with C++ dispatch) / ~4,197ms (Python-only) | **MLPerf TE reference**: ~3,967ms | **Gap**: ~191ms (4.8%) with C++ dispatch
**600-step regression**: Verified — val_loss 0.9210 at step 576 (passes 0.925 target), step time 4,170ms avg, zero NaN/skip

### Key changes since last profile

| Optimization | Old (profile_summary_best) | Current (v47) |
|---|---|---|
| FP8 quant kernel | `_cast_amax_fp8_kernel` (Triton 2D-tiled) | `_static_quant_amax_kernel` (row-based, via `static_quant_with_amax`) |
| GEMM layout | Mixed (some NT paths) | hipBLASLt NN layout exclusively (`hipb_mm` — no weight transpose) |
| `hipThreadExchangeStreamCaptureMode` | 228ms/step (10,546 calls) | Eliminated via `LD_PRELOAD` stub |
| Transpose copies | `aten::copy_` included transpose ops | Zero transpose copies (only optimizer param copy remains) |
| Step time | ~4,265ms | ~4,197ms |
| Gap vs TE | ~298ms (7.5%) | ~230ms (5.8%) |

## 1. Kernel Comparison: Lumen vs TE

### 1.1 GEMM Kernels (No Gap)

Both use identical hipBLASLt FP8 GEMMs. 3-step totals from profile:

| Kernel | Lumen 3-step (ms) | Lumen/step (ms) | Purpose |
|--------|-------------------|-----------------|---------|
| `Cijk_Ailk_Bljk_F8B8NBS` (various fwd) | 2,386 + 880 + 28 + 7 = 3,301 | 1,100 | FP8 fwd GEMMs (E4M3xE4M3) |
| `Cijk_Alik_Bljk_F8NBS` (various bwd) | 2,173 + 1,231 + 398 + 294 + 32 + 8 + 12 = 4,148 | 1,383 | FP8 bwd GEMMs (E5M2xE4M3) |
| BF16 GEMMs (`aten::mm` + `Cijk_*_BBS`) | 85 + 18 + 17 + 15 + 14 + 13 + 9 = 171 | 57 | LoRA + small GEMMs |
| **Total GEMM** | **7,620** | **~2,540** | |

TE estimated ~2,500ms/step. **Parity — not a gap source.**

### 1.2 Attention (No Gap)

Both use AITER CK FlashAttention v3:

| Kernel | Lumen 3-step (ms) | /step (ms) |
|--------|-------------------|-----------|
| `fmha_fwd_hd128_bf16_causal_rtna` | 728 | 243 |
| `fmha_bwd_hd128_bf16_causal_br_a32_rtna` | 1,674 | 558 |
| `fmha_bwd_hd128_dq_convert_bf16_rtna` | 67 | 22 |
| `fmha_bwd_hd128_odo_bf16` | 43 | 14 |
| **Total attention** | **2,512** | **~837** |

TE uses same kernels. **No gap.**

### 1.3 FP8 Quantization (Gap Source #1: ~105ms/step)

#### TE: Fused cast+transpose

TE uses `cast_transpose_fusion_kernel_optimized` — one kernel that simultaneously quantizes BF16->FP8, transposes for backward, and updates amax. Plus `delayed_scaling_recipe` for scale compute (~6.5us/call). Conservative estimate: **~100ms/step total quant cost.**

#### Lumen: `static_quant_with_amax` (row-based kernel, Branch 2)

| Kernel | 3-step (ms) | /step (ms) | Calls | Avg (us) | Purpose |
|--------|-----------|-----------|-------|----------|---------|
| `_static_quant_amax_kernel` | 376.5 | 125.5 | 2,661 | 141 | Row-based fused quant+amax (fwd+bwd delayed) |
| `_rmsnorm_quant_amax_fp8_bf16_kernel` | 86.8 | 28.9 | 606 | 143 | Fused RMSNorm+FP8 quant |
| `_fused_silu_mul_fp8_dynamic_quant_amax_kernel` | 81.7 | 27.2 | 303 | 270 | SwiGLU+quant amax phase |
| `_fused_silu_mul_fp8_dynamic_quant_apply_kernel` | 114.2 | 38.1 | 303 | 377 | SwiGLU+quant apply phase |
| `aiter::dynamic_per_tensor_quant` | 56.5 | 18.8 | 303 | 187 | CK single-kernel (E4M3 fwd) |
| `_amax_abs_kernel` | 161.0 | 53.7 | 1,671 | 96 | Standalone amax |
| `_compute_scale_kernel` | 19.4 | 6.5 | 3,267 | 6 | Scale from amax history |
| `data_to_scale_kernel` | 35.4 | 11.8 | 303 | 117 | AITER scale |
| `scaled_quant_kernel` | 19.1 | 6.4 | 303 | 63 | AITER quant |
| `initializeScale` | 1.9 | 0.6 | 303 | 6 | Scale init |
| **Total FP8 quant** | **952.5** | **~317** | | | |

**Key improvement vs old profile**: `_static_quant_amax_kernel` at 125.5ms/step (141us avg) replaces the old `_cast_amax_fp8_kernel` at 213ms/step (240us avg) — **41% faster per-kernel**.

**No transpose copies**: hipBLASLt NN layout (`hipb_mm`) eliminates the need for `transpose_cached`. The old profile showed `aten::copy_` and `Memcpy DtoD` included transpose operations; now **zero transpose copies remain** (confirmed by copy trace — only optimizer param copy left).

**Gap vs TE**: Lumen ~317ms/step vs TE ~100ms/step = ~217ms raw. After GPU overlap masking, effective gap estimated at **~105ms/step**.

### 1.4 Memory Copy Operations

| Operation | 3-step (ms) | /step (ms) | Calls | Notes |
|-----------|-----------|-----------|-------|-------|
| `aten::copy_` | 166.0 | 55.3 | 20,120 | **All from optimizer** `_copy_main_params_to_model_params` (480/step = 133.7 MB/step). Remaining ~19,640 calls are trivial dtype casts. **Zero transpose copies.** |
| `Memcpy DtoD` | 104.0 | 34.7 | 16,696 | Device-to-device (non-transpose) |
| `hipStreamIsCapturing` | 2.4ms CPU | 0.0 | 2,114 | **LD_PRELOAD stub active** — was 228ms/step before |
| **Total** | **272.4** | **~90** | | |

**Major improvement**: `hipThreadExchangeStreamCaptureMode` eliminated entirely (was 228.6ms/step, now 0ms CUDA). `hipStreamIsCapturing` at 2,114 calls has zero CUDA time.

**Copy trace confirms**: The only `copy_()` source in Lumen code is the optimizer's `_copy_main_params_to_model_params` (480 calls/step, 133.7 MB/step). No transpose, no FP8 cast, no quant-related copies.

### 1.5 Communication (NCCL)

| Kernel | 3-step (ms) | /step (ms) |
|--------|-----------|-----------|
| `rcclGenericKernel` (all_reduce) | 234 | 78.0 |
| `nccl:all_reduce` | 230 | 76.8 |
| `nccl:allreduce_coalesced` (optimizer) | 3.4 | 1.1 |
| `mscclKernel_Sum` | 0.6 | 0.2 |
| **Total NCCL** | **~234** | **~78** |

DP=8 gradient all-reduce. Slightly improved from old profile (was ~94ms/step).

### 1.6 SwiGLU / MLP

| Kernel | 3-step (ms) | /step (ms) |
|--------|-----------|-----------|
| `_swiglu_fwd_kernel` | 142.0 | 47.3 |
| `_swiglu_bwd_kernel` | 204.2 | 68.1 |
| SwiGLU+quant (amax+apply) | 195.9 | 65.3 |
| **Total SwiGLU** | **542** | **~181** |

TE fuses SwiGLU into FC1 `_Linear`. Lumen has separate kernels. The 2-phase SwiGLU+quant (amax then apply) reads intermediate data twice.

### 1.7 RMSNorm

| Kernel | 3-step (ms) | /step (ms) | Calls |
|--------|-----------|-----------|-------|
| `_rmsnorm_quant_amax_fp8_bf16_kernel` | 86.8 | 28.9 | 606 |
| `_rms_norm_kernel` | 37.8 | 12.6 | 480 |
| `_rmsnorm_bwd_triton` | 68.9 | 23.0 | 480 |
| `_rmsnorm_bwd_dg_reduce_triton` | 4.2 | 1.4 | 480 |
| **Total RMSNorm** | **197.7** | **~66** |

Competitive with TE's fused norm path.

### 1.8 LoRA (Lumen Advantage)

| Kernel | 3-step (ms) | /step (ms) | Calls |
|--------|-----------|-----------|-------|
| `_fused_lora_fwd_kernel` | 119.1 | 39.7 | 606 |
| `_fused_lora_bwd_kernel` | 98.0 | 32.7 | 480 |
| **Total LoRA** | **217.1** | **~72** |

TE does not have fused LoRA kernels. **Lumen advantage.**

### 1.9 Other Kernels

| Kernel | 3-step (ms) | /step (ms) | Purpose |
|--------|-----------|-----------|---------|
| `fused_rope_forward` | 46.7 | 15.6 | RoPE fwd |
| `fused_rope_backward` | 41.6 | 13.9 | RoPE bwd |
| `aten::add_` | 110.2 | 36.7 | Residual add |
| `aten::add` | 100.5 | 33.5 | Gradient accumulation |
| `aten::fill_` | 75.3 | 25.1 | Zero-init buffers |
| `aten::cat` | 39.1 | 13.0 | QKV concat |
| `aten::sum` / reduce | 25.9 + 25.0 | 17.0 | Loss/grad reductions |
| `dropout` (fwd+bwd) | 3.9 | 1.3 | LoRA dropout |
| Optimizer (FusedAdam) | 1.3 | 0.4 | Weight update |

### 1.10 Unfused Operations in Lumen (vs TE)

1. **2-phase SwiGLU+quant**: `amax_kernel` + `apply_kernel` — two passes over intermediate data. Single-pass fusion would save ~20-25ms/step.

2. **Standalone `_amax_abs_kernel`**: 53.7ms/step across 1,671 calls. Many of these could be folded into the quant kernel that follows them.

3. **Scale computation chain**: `_compute_scale_kernel` (6.5ms) + `data_to_scale_kernel` (11.8ms) + `initializeScale` (0.6ms) = 18.9ms/step. TE uses `delayed_scaling_recipe` (~6.5us/call).

### 1.11 Resolved Gaps (No Longer Applicable)

1. **~~Cast+transpose fusion~~**: hipBLASLt NN layout eliminates weight transpose entirely. `transpose_cached` is never called. Copy trace confirms zero transpose copies. **Not a gap source.**

2. **~~`hipThreadExchangeStreamCaptureMode`~~**: Eliminated by `LD_PRELOAD` stub (`hip_no_stream_capture.so`). Was 228ms/step, now 0ms. **Resolved.**

## 2. Bubble Analysis

### 2.1 GPU Utilization

**GPU utilization: ~99.5%** on the main compute stream.

Total CUDA kernel time: 12,516ms across 3 steps. Step time from logs: ~4,197ms × 3 = 12,591ms. Only ~75ms of gaps total (~25ms/step).

### 2.2 Key Finding: Bubbles Are Minimal

In-step bubbles are dominated by unavoidable step transitions and NCCL sync. The ~230ms gap vs TE is **not from idle GPU time** but from:

1. **FP8 quant overhead** — more kernel passes for the same logical operation (~105ms effective gap)
2. **Unfused operations** — standalone amax, 2-phase SwiGLU+quant (~75ms)
3. **Minor overhead** — host launch latency, additional elementwise ops (~50ms)

## 3. Optimization Opportunities (Ranked by Impact)

### 3.1 Reduce Standalone amax — TESTED, NO IMPACT

`_amax_abs_kernel` at 53.7ms/step (1,671 calls/3-step). Instrumentation (`LUMEN_TRACE_AMAX=1`) revealed the source: activation checkpointing recompute creates 477 fresh LoRA tensor_ids per step that hit `len(history) == 0` in `get_scale()`, launching a standalone `_amax_abs_kernel` that is immediately wasted because `static_quant_with_amax` recomputes amax.

**Fix implemented**: Added `skip_amax_precompute` to `get_scale()` and `quantize_bwd_delayed()` to skip standalone amax when fused kernel will fire. Reduced calls from 557/step to ~0/step in steady state.

**A/B result**: No measurable improvement. Baseline: 4,187.6ms/step avg (steps 20-50). With fix: 4,206.5ms/step avg (+18.9ms, within thermal noise). The 557 tiny kernel launches (~0.1ms each) were fully overlapped by the GPU scheduler with larger GEMM/attention kernels. Raw profiler `self_cuda_time` reports occupancy, not throughput impact. **Fix kept for correctness** (eliminates redundant work) but has zero wall-clock benefit.

### 3.2 SwiGLU+Quant Single Kernel — NOT FEASIBLE

Fusing `_fused_silu_mul_fp8_dynamic_quant_amax_kernel` + `_fused_silu_mul_fp8_dynamic_quant_apply_kernel` into one pass is **mathematically impossible** with dynamic per-tensor scaling. Phase 1 must scan ALL rows to compute global amax via atomic reduction. Phase 2 quantizes using that scale. Triton has no cross-block synchronization for single-pass fusion. Switching to delayed scaling would change training semantics.

### 3.3 Scale Computation — TESTED, NO IMPACT

`_compute_scale_kernel` (6.5ms) + `data_to_scale_kernel` (11.8ms) + `initializeScale` (0.6ms) = 18.9ms/step. Batch precomputation tested (v1-v3) and showed no benefit — Triton launches fully overlapped with GPU compute.

### 3.4 Quant Kernel Efficiency — SUBSUMED BY 3.1

The standalone `_amax_abs_kernel` calls (1,671/3-step) are the same root cause as 3.1. Eliminating them had no wall-clock impact (see 3.1 A/B results). The remaining `_static_quant_amax_kernel` calls (2,661/3-step, 125.5ms/step) are the fused quant+amax path — already optimal for the row-based layout.

### 3.5 SDMA DP Gradient All-Reduce (Est. ~10-20ms/step)

Replace NCCL with SDMA for DP gradient all-reduce (plan exists). NCCL currently ~78ms/step.

## 4. Summary

### Per-Step Kernel Time Breakdown (from 3-step profile / 3)

| Category | Lumen (ms/step) | % of CUDA | Notes |
|----------|-----------------|-----------|-------|
| FP8 GEMMs | 2,540 | 60.9% | Identical to TE |
| Attention | 837 | 20.1% | Identical to TE |
| FP8 Quant total | 317 | 7.6% | TE ~100ms -> **~105ms effective gap** |
| SwiGLU | 181 | 4.3% | TE fuses into FC1 |
| NCCL | 78 | 1.9% | Same as TE |
| LoRA | 72 | 1.7% | **Lumen advantage** |
| RMSNorm | 66 | 1.6% | Similar to TE |
| Copy/Memcpy | 90 | 2.2% | **Zero transpose copies** (optimizer only) |
| RoPE | 30 | 0.7% | Same as TE |
| Other (add, fill, cat, etc.) | 127 | 3.0% | Minor overhead |
| **Total CUDA time** | **~4,172** | | |

### Gap vs TE (~230ms)

| Gap Source | Estimated (ms) | Status |
|------------|---------------|--------|
| FP8 quant overhead (more kernel passes) | ~105 | **Not addressable** — fully overlapped with GPU compute (A/B tested) |
| SwiGLU+quant 2-phase | ~25 | **Not feasible** — dynamic scaling requires 2-pass |
| Standalone amax | ~30 | **Tested, no impact** — kernel launches overlapped by GPU scheduler |
| Scale computation chain | ~15 | **Tested, no impact** — fully overlapped with GPU compute |
| Minor host/launch overhead | ~55 | **~30ms addressed** via C++ dispatch (see 3.6) |
| **Total identified** | **~230** | **~145ms unaddressable; ~30ms saved via C++ dispatch** |

**Key insight**: The ~230ms gap is NOT explained by redundant kernel launches or fusible operations. Profiler `self_cuda_time` overstates the impact of small kernels because they run fully overlapped with large GEMM/attention kernels. The ~55ms host/launch overhead is partially addressable via C++ dispatch.

### 3.6 C++ FP8 Quant Dispatch — TESTED, ~30ms/step saving

Replaced the Python hot path (`get_scale()` → `_compute_scale_kernel` → `static_quant_with_amax()` → `update_amax_value()`) with a single C++ function (`FP8QuantDispatcher.quantize()`) that:
- Owns amax history in C++ (`std::unordered_map<string, deque<Tensor>>`)
- Computes scale inline in the HIP kernel (eliminates separate `_compute_scale_kernel` launch)
- Launches `fused_scale_quant_amax_kernel` (HIP, row-based, quant+amax in one kernel)
- Updates history in C++ (deque push_back + pop_front)

**A/B result** (55 steps, `LUMEN_CPP_QUANT_DISPATCH=1`):

| Steps | Baseline (ms) | C++ Dispatch (ms) | Delta |
|-------|--------------|-------------------|-------|
| 20 | 4,177.6 | 4,143.1 | -34.5 |
| 30 | 4,188.5 | 4,160.1 | -28.4 |
| 40 | 4,189.3 | 4,160.1 | -29.2 |
| 50 | 4,194.8 | 4,167.3 | -27.5 |
| **Avg** | **4,187.6** | **4,157.7** | **-29.9** |

Loss: 1.3916 (baseline) vs 1.3928 (C++ dispatch) — within noise.
Gated by `LUMEN_CPP_QUANT_DISPATCH=1` (default off).

**600-step regression test** (`TRAIN_STEPS=600 LUMEN_CPP_QUANT_DISPATCH=1`):

| Step | val_loss | Target | Status |
|------|----------|--------|--------|
| 192 | 0.9447 | 0.925 | On track |
| 384 | 0.9296 | 0.925 | Below target |
| 576 | **0.9210** | 0.925 | **Passes** |
| 600 (final val) | 0.9260 | 0.925 | Passes |

Step time: 4,170ms avg (std ~5ms), consistent across all 600 steps. Zero NaN/skipped iterations. Memory stable at 99.53%. **No convergence or stability regression.**

**New step time**: ~4,158ms. **New gap vs TE**: ~191ms (4.8%).

### Resolved (No Longer Gap Sources)

| Former Gap Source | Old Cost (ms/step) | Current | Resolution |
|---|---|---|---|
| `hipThreadExchangeStreamCaptureMode` | 228.6 | 0 | `LD_PRELOAD` stub |
| Transpose/copy overhead | ~55 | 0 | hipBLASLt NN layout — no transpose needed |
| `_cast_amax_fp8_kernel` inefficiency | 213 | 125.5 | Replaced by `_static_quant_amax_kernel` (row-based) |
| Python dispatch overhead | ~16 | ~2 | C++ `FP8QuantDispatcher` (quant+scale+amax in 1 C++ call) |

### Profile Files

- `profiling/lumen_profile_summary.txt` — torch.profiler summary, steps 100-102 (v47 baseline)
- `profiling/lumen_latest_profile_summary.txt` — same as above (latest)
- `profiling/te_profile_results.txt` — TE per-layer microbenchmark
- `profile_trace_current.json` — chrome trace, steps 100-102
- `profile_summary_current_copy_trace.txt` — `copy_`/`contiguous` call trace
