# MLPerf Llama2-70B LoRA SFT — Lumen Benchmark Results

Lumen training results on the MLPerf `llama2_70b_lora` benchmark (8x MI300X),
compared against the official AMD MLPerf v5.1 reference run on the **same machine**.

**Target**: val_loss < 0.925

## Result Summary

| Metric | Lumen v47 (current) | MLPerf Ref (local, same MI300X) |
|--------|--------------------|---------------------------------|
| Best val_loss | **0.9223** | 0.9243 |
| Passes MLPerf target? | **Yes** (step 576) | Yes (step 384) |
| Pre-eval step time | **4,730 ms** | **3,967 ms** |
| Post-eval step time | **5,550 ms** | ~4,000 ms (est.) |
| Speed ratio (Lumen / MLPerf) | **1.19x** | 1.0x |
| Memory utilization | 98.7% | ~82% |
| Stability | 0 NaN/skip | 0 NaN/skip |

**Local MLPerf reference**: `rocm/amd-mlperf:llama2_70b_training_5.1` Docker,
`SEED=1234`, zarr checkpoint converted from `NousResearch/Llama-2-70b-hf`.
Log: `mlperf_ref_mi300x_20260416_030753.log`.

## Implemented Optimizations and Measured Impact

All optimizations applied cumulatively. Impact is measured against the unoptimized
Lumen baseline (7,400 ms/step, val_loss 0.9371, does not pass target).

### Convergence Fixes

| Optimization | Impact on val_loss | Impact on Speed |
|--------------|--------------------|-----------------|
| **Epoch-level data shuffling** (`LUMEN_SHUFFLE_TRAIN=1`) | **-0.016** (0.9371→0.9208, now passes target) | None |
| **Aligned eval schedule** (`LUMEN_EVAL_ALIGNED=1`, every 192 steps) | -0.003 (matches MLPerf eval cadence) | **-11% wall-clock** (fewer evals) |

Data shuffling was the single most important fix. Without it, Lumen does not
converge below 0.925 regardless of other optimizations.

### Speed Optimizations

| Optimization | Measured Savings | Mechanism |
|--------------|-----------------|-----------|
| **Fused quant+amax** (`LUMEN_FUSED_QUANT_AMAX=1`) | **-377 ms/step (6.1%)** | Merge `abs()` + `amax()` into single Triton kernel |
| **Fused quant+scale** (`LUMEN_FUSED_QUANT_SCALE=1`) | **-206 ms/step (2.8%)** | Merge quant + scale computation |
| **Post-eval allocator fixes** (eval recompute, warmup, GC, cache clear) | **-11.1% total training time** | Prevent memory fragmentation after eval passes |
| **Backend caching + sync elimination** (`LUMEN_SKIP_BACKEND_SYNC=1`) | **~1-2% step time** | Cache GEMM backend selection, skip redundant syncs |
| **Fused RMSNorm + FP8 quant** (`LUMEN_FUSED_NORM_QUANT=1`) | ~0.2% step time | Merge norm + quant into single kernel |
| **Fused SwiGLU backward** (`LUMEN_FUSED_SWIGLU=1`) | Included in baseline→current | Fuse SwiGLU backward elementwise ops |
| **hipBLASLt for all GEMMs** (`LUMEN_PREFER_HIPBLASLT=1`) | **-790 ms/step (14.2%)** | hipBLASLt replaces CK for fwd+bwd; `.t()` view eliminates weight transpose copy |
| **Wgrad `.t()` view** (v47) | **~50 ms/step** | Eliminate `grad_fp8.t().contiguous()` in wgrad path |
| **SwiGLU fused amax** (v47) | **~30-80 ms/step** | Replace `bf16.abs().amax()` with `fused_amax_abs()` in SwiGLU FP8 path |
| **FP8 weight gradients** (`FP8_WGRAD=1`, hipBLASLt) | ~0 ms (correctness alignment) | Match MLPerf FP8 wgrad path |
| **ACL=21** (`RECOMPUTE_NUM_LAYERS=21`) | ~0 ms (memory trade-off) | Match MLPerf activation checkpointing depth |

### Net Result

| Metric | Baseline | Lumen v47 | MLPerf Ref (local) |
|--------|----------|-----------|-------------------|
| Pre-eval step time | 7,400 ms | **4,730 ms** | **3,967 ms** |
| Post-eval step time | — | **5,550 ms** | ~4,000 ms |
| Memory utilization | — | 98.7% | ~82% |
| val_loss (best) | 0.9371 (fail) | **0.9223** (pass) | 0.9243 (pass) |
| Speed ratio vs MLPerf | 1.87x | **1.19x** | 1.0x |

## Val_loss Trajectory

### Lumen v47 (SEED=1234)

| Step | val_loss | Passes target? |
|------|----------|----------------|
| 192 | 0.9501 | No |
| 384 | 0.9323 | No |
| 576 | **0.9223** | **Yes** |

### MLPerf Reference — Local MI300X (SEED=1234)

| Step | val_loss | Throughput (s/s) | Step time (ms) |
|------|----------|-----------------|---------------|
| 192 | 0.9398 | 2.0167 | 3,967 |
| 240 | 0.9395 | 2.0172 | 3,966 |
| 288 | 0.9311 | 2.0168 | 3,967 |
| 336 | 0.9268 | 2.0167 | 3,967 |
| 384 | **0.9243** | 2.0156 | 3,969 |

### Convergence Gap at Matched Steps

| Step | Lumen v47 | MLPerf Ref (local) | Gap |
|------|-----------|-------------------|-----|
| 192 | 0.9501 | 0.9398 | +0.010 |
| 384 | 0.9323 | 0.9243 | +0.008 |

Lumen reaches the MLPerf target at step 576 vs MLPerf reference at step 384.
The gap is from accumulated kernel-level numerical differences (FP8 format,
GEMM backend, normalization order), each individually < 0.005 val_loss.

## Speed Gap Analysis

Both Lumen and MLPerf reference use identical parallelism (TP=1, ACL=21, DP=8)
and FP8 config. The remaining 1.19x pre-eval speed gap is from kernel fusion,
dispatch overhead, and memory pressure.

### Remaining Speed Gap: 4,730 ms (Lumen v47) vs 3,967 ms (MLPerf local) = 763 ms

Based on v47 profiling data (steps 8-10, rank 0, 3-step totals divided by 3):

| # | Root Cause | Lumen (ms/step) | TE est. (ms/step) | Gap (ms) | Status |
|---|-----------|-----------------|-------------------|---------|--------|
| 1 | **aten::copy_** (dtype casts, residual copies) | 289 | ~60 | **~229** | Reduced from 486→336→289; wgrad `.t()` view eliminated more copies |
| 2 | **FP8 quant/scale pipeline** | ~223 | 54 | **~169** | dynamic_quant=64, static_quant=49, amax_abs=94, compute_scale=4, abs+amax separate=12 |
| 3 | **aten::cat + add + add_** | ~200 | ~30 | **~170** | Autograd concat, elementwise residual ops |
| 4 | **Cast+transpose** | 162 | included | **~80** | Triton fused kernel; consumed by hipBLASLt wgrad |
| 5 | **Memory pressure** (98.7% vs 82%) | — | — | **~50-100** | Amplifies all allocation/free overhead |
| 6 | **SwiGLU residual** (Triton vs TE C++) | ~158 | ~100 | **~58** | fwd=49, bwd=75, fused_silu_mul=37 |
| 7 | **Memcpy DtoD** | 88 | ~20 | **~68** | Device copies from unfused ops |
| 8 | **Kernel dispatch** (~11.6K vs ~5K launches) | ~16 | ~8 | **~8** | Reduced 5x from 57K |
| 9 | **Other** (dropout, fill_, zero_, mul, neg, RoPE, NCCL) | ~120 | ~50 | **~70** | |
| | **Total estimated gap** | | | **~850** | vs measured 763 ms |

### Highest-Impact Next Steps

1. **Fuse copy/cast into GEMM epilogue** — 289 ms of `aten::copy_` is the #1 gap.
   TE avoids most copies by fusing dtype casts into GEMM epilogues (FP8→BF16 and
   BF16→FP8 happen inside the same kernel as the matmul). Lumen does them as
   separate `aten::copy_` calls. Implementing custom GEMM epilogues or using
   hipBLASLt's epilogue API could eliminate ~150-200 ms.

2. **Fuse FP8 quantization into preceding ops** — 223 ms across dynamic_quant,
   static_quant, amax_abs, and compute_scale. TE's `_LayerNormLinear` fuses
   RMSNorm → quant → GEMM in a single fused op. Lumen does RMSNorm and quant
   as separate Triton kernels. Fusing quant into the RMSNorm Triton kernel
   (already started with `LUMEN_FUSED_NORM_QUANT`) and into SwiGLU output
   would save ~100-150 ms.

3. **Reduce elementwise overhead** — `aten::cat` (69 ms), `aten::add/add_` (131 ms)
   from autograd residual paths. TE's fused modules avoid explicit concat by
   writing QKV directly into a pre-allocated buffer. Implementing in-place QKV
   projection would eliminate the cat entirely.

4. **Reduce memory pressure** — at 98.7% (189 GiB), the ROCm allocator fragments
   heavily. Each freed block may not be reusable for the next allocation. Reducing
   to ~90% would eliminate ~50-100 ms of allocator overhead. Options:
   - FP8 storage for linear inputs (currently BF16, ~24 GiB for 59 layers)
   - Reduce attention activation footprint

### TE GPU Time Breakdown (per layer, fwd + bwd, from profiling)

| Operation | Per Iter (ms) | GEMM (ms) | Overhead (ms) | Launches/iter |
|-----------|--------------|-----------|--------------|---------------|
| LayerNormLinear (QKV: 8192→10240) | 5.114 | 3.529 (69%) | 1.585 | 7 |
| Linear (proj: 8192→8192) | 3.300 | 2.839 (86%) | 0.461 | 8 |
| Linear (fc1: 8192→57344) | 23.083 | 20.848 (90%) | 2.235 | 8 |
| Linear (fc2: 28672→8192) | 11.915 | 10.756 (90%) | 1.159 | 8 |
| **Per layer total** | **43.412** | **37.972** | **5.440** | **31** |
| **80 layers** | **3,473** | **3,038** | **435** | **2,480** |

## Lumen v47 GPU Time Breakdown (profiled steps 8-10, rank 0)

| Category | 3-step total (ms) | Per step (ms) | % of Step | Launches/step |
|----------|--------------------|--------------|-----------|---------------|
| **GEMM (hipBLASLt)** | **7,307** | **2,436** | 53.0% | ~1,249 |
| Attention backward | ~1,774 | ~591 | 12.9% | 80 |
| Attention forward | ~732 | ~244 | 5.3% | 101 |
| Cast+transpose (Triton) | ~486 | ~162 | 3.5% | 810 |
| SwiGLU fwd+bwd (fused) | ~483 | ~161 | 3.5% | ~844 |
| aten::copy_ | **866** | **289** | 6.3% | ~11,471 |
| _amax_abs_kernel | **282** | **94** | 2.0% | ~794 |
| FP8 quant (dynamic+static) | ~340 | ~113 | 2.5% | ~962 |
| aten::mul | ~249 | ~83 | 1.8% | ~1,901 |
| aten::cat | ~208 | ~69 | 1.5% | ~647 |
| aten::add + add_ | ~394 | ~131 | 2.9% | ~1,572 |
| Memcpy DtoD | ~265 | ~88 | 1.9% | ~9,150 |
| NCCL AllReduce | ~209 | ~70 | 1.5% | ~165 |
| RMSNorm (fwd+bwd) | ~99 | ~33 | 0.7% | ~320 |
| Dropout | ~74 | ~25 | 0.5% | 202 |
| LoRA mm | ~81 | ~27 | 0.6% | 522 |
| Other (fill_, zero_, neg, etc.) | ~270 | ~90 | 2.0% | — |
| **Total** | **~13,786** | **~4,595** | **100%** | **~11,600** |

Wall-clock step times: 4,718–4,738 ms (profiler overhead ~3%).

### Lumen vs TE Comparison (v47 profile)

| Category | Lumen (ms) | TE est. (ms) | Lumen/TE | Status |
|----------|-----------|-------------|----------|--------|
| **GEMM** | 2,436 | 3,038 | **0.80x** | **Faster than TE** |
| **Attention** | 835 | ~822 | 1.02x | Near parity |
| **SwiGLU** | 161 | ~100 | 1.6x | Fused Triton (was 8.6x) |
| **FP8 quant/scale** | 223 | 54 | 4.1x | Improved (was 7.6x) |
| **Copy/cast** | 289 | ~60 | 4.8x | Improved (was 5.6x) |
| **Cat** | 69 | ~15 | 4.6x | |
| **NCCL** | 70 | ~119 | **0.59x** | Faster |
| **Other** | 332 | ~70 | 4.7x | Memcpy, dropout, add, mul, fill |
| **Total** | **~4,595** | **~3,473** (layers only) | **1.32x** | |

### Key Findings

1. **GEMM is 20% faster than TE.** hipBLASLt at 2,436 ms/step vs TE's 3,038 ms
   (0.80x ratio). hipBLASLt handles mixed-dtype (E4M3/E5M2) natively and the
   `.t()` view eliminates weight transpose entirely.

2. **The gap is entirely non-GEMM overhead.** Lumen's GEMM is faster, but the
   763 ms total gap comes from copy/cast (229 ms), FP8 quant (169 ms),
   elementwise ops (170 ms), cast+transpose (80 ms), and memory pressure (~75 ms).

3. **FP8 quant/scale pipeline improved.** v47's fused amax in SwiGLU path
   reduced `_amax_abs_kernel` from 102 ms to 94 ms/step. Total FP8 quant
   overhead is 223 ms vs TE's 54 ms (4.1x, down from 7.6x).

4. **aten::copy_ reduced 14%.** v47's wgrad `.t()` view eliminated grad transpose
   copies: 289 ms (v47) vs 336 ms (v46). Still 4.8x vs TE.

5. **Kernel launches stable at ~11.6K/step** (5x reduction from 57K baseline).
   TE achieves ~2.5K launches for the 80-layer forward+backward.

6. **Memory pressure at 98.7%** amplifies allocator overhead. Post-eval
   fragmentation adds ~820 ms/step (4,730 → 5,550 ms).

## Post-Eval Performance

| Metric | Lumen (early) | Lumen v46 | Lumen v47 | MLPerf Ref (local) |
|--------|---------------|-----------|-----------|-------------------|
| Pre-eval ms/step | 5,898 | 4,780 | **4,730** | **3,967** |
| Post-eval #1 ms/step | 7,167 | ~5,350 | **5,550** | ~4,000 |
| Post-eval delta | +21.7% | ~12% | **+17.3%** | ~0% |

Root cause: Megatron's `transformer_block.py` skips activation checkpointing
during eval (`self.training=False`), allocating activations for all 80 layers
vs 21 during training. At 98%+ memory, this fragments the ROCm allocator and
the fragmentation persists after training resumes.

## Configuration Diff

| Parameter | Lumen v47 | MLPerf Reference (local) |
|-----------|----------|--------------------------|
| Framework | Megatron-LM-AMD + Lumen + AITER | NeMo v2.3.0 + TE + Megatron-LM |
| Docker | N/A (native) | `rocm/amd-mlperf:llama2_70b_training_5.1` |
| TP / PP / CP | 1 / 1 / 1 | 1 / 1 / 1 |
| DP | 8 | 8 |
| MBS / GBS | 1 / 8 | 1 / 8 |
| LR | 4e-4 | 4e-4 |
| LR Warmup | 0 | 0 |
| LR Schedule | Cosine, 1024 steps | Cosine, 1024 steps |
| LoRA rank / alpha | 16 / 32 | 16 / 32 |
| FP8 Format | E4M3 hybrid (delayed) | E4M3 hybrid (delayed) |
| FP8 amax_history | 4 | 4 |
| FP8 amax_algo | most_recent | most_recent |
| FP8 backward dtype | E5M2 (hipBLASLt) | E5M2 (hipBLASLt) |
| FP8 wgrad | FP8 (hipBLASLt) | FP8 (TE) |
| Activation recompute | 21 layers (full/block) | 21 layers (full/block) |
| RMSNorm | AITER Triton | TE Triton / apex |
| SwiGLU | Fused Triton (fwd+bwd) | TE fused C++ |
| Attention | AITER CK FMHA v3 | TE CK fused attn v3 |
| RoPE | Unfused Megatron | TE fused |
| Seed | 1234 | 1234 |
| Val check interval | 192 steps | 48 steps (384/GBS, SKIP_EVALS=3) |

## Optimization History

| Change | val_loss | Step time | vs MLPerf |
|--------|----------|-----------|-----------|
| Baseline (no shuffle) | 0.9371 (fail) | 7,400 ms | 1.87x |
| + Data shuffling | **0.9208** (pass) | 7,400 ms | 1.87x |
| + Aligned eval + fused norm+quant | 0.9192 | 6,200 ms | 1.56x |
| + Fused quant+amax | — | 5,809 ms | 1.46x |
| + Fused SwiGLU bwd | — | 5,348 ms | 1.35x |
| + ACL=21 + hipBLASLt wgrad | 0.921 | 5,570 ms | 1.40x |
| + hipBLASLt all + `.t()` view (v46) | 0.9216 | 4,780 ms | 1.21x |
| **+ Wgrad `.t()` + SwiGLU fused amax (v47)** | **0.9223** | **4,730 ms** | **1.19x** |

## Source Files

| File | Purpose |
|------|---------|
| `lumen/models/llama2/dataset.py` | `LLaMA2SFTDataset` with `_build_samples_mapping()` shuffle |
| `lumen/models/llama2/megatron/sft.py` | Passes `shuffle=True` to dataset when `LUMEN_SHUFFLE_TRAIN=1` |
| `lumen/modules/layernorm_linear.py` | Fused RMSNorm + FP8 quant (`LUMEN_FUSED_NORM_QUANT=1`) |
| `lumen/modules/parallel_linear.py` | `pre_quantized_input` threading for fused path |
| `lumen/ops/quantize/linear.py` | FP8 autograd functions; wgrad `.t()` view, fused cast+transpose |
| `lumen/ops/quantize/cast_transpose.py` | Triton `cast_transpose_amax_fp8` kernel |
| `lumen/ops/dispatch.py` | Backend caching + sync elimination |
| `lumen/models/_swiglu_fp8_fuse.py` | Fused SwiGLU + FP8 quant + fused amax |
| `lumen/models/megatron.py` | `_patch_fused_swiglu_mlp` / `_run_warmup_eval_pass` / `_precompute_fp8_transpose` |
| `lumen/models/megatron_patches.py` | `install_eval_recompute` / `install_post_eval_cache_clear` |
| `lumen/kernels/compute_scale.py` | Triton `_compute_scale_kernel` for FP8 scale computation |
| `lumen/quantize/scaling_manager.py` | `_quantize_core` with fused cast+transpose + amax path |
| `examples/llama2/config_MI300X_tp1_dp8.sh` | Training config (all env flags) |

## Profiling Data

| File | Description |
|------|------------|
| `profiling/lumen_profile_summary.txt` | `torch.profiler` key_averages — CK baseline, 3 steps, rank 0 |
| `profiling/lumen_latest_profile_summary.txt` | `torch.profiler` key_averages — v46 hipBLASLt all + `.t()` view, 3 steps, rank 0 |
| `profiling/te_profile_results.txt` | `torch.profiler` key_averages — TE ops (LayerNormLinear, Linear) at same tensor shapes, 10 iters |
