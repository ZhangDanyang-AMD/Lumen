# MLPerf Llama2-70B LoRA SFT — Lumen Benchmark Results

Lumen training results on the MLPerf `llama2_70b_lora` benchmark (8x MI300X),
compared against the official AMD MLPerf v5.1 reference run on the **same machine**.

**Target**: val_loss < 0.925

## Result Summary

| Metric | Lumen (current) | Lumen v48 | MLPerf Ref (local, same MI300X) |
|--------|-----------------|-----------|--------------------------------|
| Best val_loss | **0.9190** | 0.9211 | 0.9243 |
| Passes MLPerf target? | **Yes** (step 576) | Yes (step 576) | Yes (step 384) |
| Pre-eval step time | **4,699 ms** | 4,747 ms | **3,967 ms** |
| Post-eval step time | **5,569 ms** | 5,640 ms | ~4,000 ms (est.) |
| Speed ratio (Lumen / MLPerf) | **1.18x** | 1.20x | 1.0x |
| Memory utilization | 98.5% | 98.4% | ~82% |
| Stability | 0 NaN/skip | 0 NaN/skip | 0 NaN/skip |
| Total time (1024 steps) | **6,330 s** | — | — |

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
| **hipBLASLt for all GEMMs** (`LUMEN_PREFER_HIPBLASLT=1`) | **-790 ms/step (14.2%)** | hipBLASLt replaces CK for fwd+bwd; `.t()` view eliminates weight transpose copy |
| **Fused quant+amax** (`LUMEN_FUSED_QUANT_AMAX=1`) | **-377 ms/step (6.1%)** | Merge `abs()` + `amax()` into single Triton kernel |
| **Fused quant+scale** (`LUMEN_FUSED_QUANT_SCALE=1`) | **-206 ms/step (2.8%)** | Merge quant + scale computation |
| **Post-eval allocator fixes** (eval recompute, warmup, GC, cache clear) | **-11.1% total training time** | Prevent memory fragmentation after eval passes |
| **Fused NQG norm+quant** (`LUMEN_FUSED_NORM_QUANT_GEMM=1`) | **-51 ms/step (-1.1%)** | Fuse RMSNorm + FP8 quant in single AITER Triton kernel; rsigma output eliminates extra FP32 reduction |
| **Fused residual+norm** (`LUMEN_FUSED_RESIDUAL_NORM=1`) | **~0 ms** (net zero) | Skip BDA, fuse add into norm input; 1 fewer kernel launch per layer |
| **Backend caching + sync elimination** (`LUMEN_SKIP_BACKEND_SYNC=1`) | **~1-2% step time** | Cache GEMM backend selection, skip redundant syncs |
| **Fused RMSNorm + FP8 quant** (`LUMEN_FUSED_NORM_QUANT=1`) | ~0.2% step time | Merge norm + quant into single kernel (V1 path) |
| **Fused SwiGLU backward** (`LUMEN_FUSED_SWIGLU=1`) | Included in baseline→current | Fuse SwiGLU backward elementwise ops |
| **Wgrad `.t()` view** (v47) | **~50 ms/step** | Eliminate `grad_fp8.t().contiguous()` in wgrad path |
| **SwiGLU fused amax** (v47) | **~30-80 ms/step** | Replace `bf16.abs().amax()` with `fused_amax_abs()` in SwiGLU FP8 path |
| **FP8 weight gradients** (`FP8_WGRAD=1`, hipBLASLt) | ~0 ms (correctness alignment) | Match MLPerf FP8 wgrad path |
| **ACL=21** (`RECOMPUTE_NUM_LAYERS=21`) | ~0 ms (memory trade-off) | Match MLPerf activation checkpointing depth |

### Net Result

| Metric | Baseline | Lumen v47 | Lumen v48 | Lumen (current) | MLPerf Ref (local) |
|--------|----------|-----------|-----------|-----------------|-------------------|
| Pre-eval step time | 7,400 ms | 4,730 ms | 4,747 ms | **4,699 ms** | **3,967 ms** |
| Post-eval step time | — | 5,550 ms | 5,640 ms | **5,569 ms** | ~4,000 ms |
| Memory utilization | — | 98.7% | 98.4% | 98.5% | ~82% |
| val_loss (best) | 0.9371 (fail) | 0.9223 (pass) | 0.9211 (pass) | **0.9190** (pass) | 0.9243 (pass) |
| Speed ratio vs MLPerf | 1.87x | 1.19x | 1.20x | **1.18x** | 1.0x |

## Val_loss Trajectory

### Lumen (current) — rsigma optimization (SEED=1234)

| Step | val_loss | Passes target? |
|------|----------|----------------|
| 192 | 0.9510 | No |
| 384 | 0.9351 | No |
| 576 | **0.9244** | **Yes** |
| 768 | 0.9246 | Yes |
| 960 | 0.9206 | Yes |
| 1024 | **0.9190** | **Yes** |

### Lumen v48 (SEED=1234)

| Step | val_loss | Passes target? |
|------|----------|----------------|
| 192 | 0.9485 | No |
| 384 | 0.9321 | No |
| 576 | **0.9211** | **Yes** |

### MLPerf Reference — Local MI300X (SEED=1234)

| Step | val_loss | Throughput (s/s) | Step time (ms) |
|------|----------|-----------------|---------------|
| 192 | 0.9398 | 2.0167 | 3,967 |
| 240 | 0.9395 | 2.0172 | 3,966 |
| 288 | 0.9311 | 2.0168 | 3,967 |
| 336 | 0.9268 | 2.0167 | 3,967 |
| 384 | **0.9243** | 2.0156 | 3,969 |

### Convergence Gap at Matched Steps

| Step | Lumen (current) | MLPerf Ref (local) | Gap |
|------|-----------------|-------------------|-----|
| 192 | 0.9510 | 0.9398 | +0.011 |
| 384 | 0.9351 | 0.9243 | +0.011 |

Lumen reaches the MLPerf target at step 576 vs MLPerf reference at step 384.
The gap is from accumulated kernel-level numerical differences (FP8 format,
GEMM backend, normalization order), each individually < 0.005 val_loss.

## Speed Gap Analysis

Both Lumen and MLPerf reference use identical parallelism (TP=1, ACL=21, DP=8)
and FP8 config. The remaining 1.18x pre-eval speed gap is from kernel fusion,
dispatch overhead, and memory pressure.

### Remaining Speed Gap: 4,699 ms (Lumen) vs 3,967 ms (MLPerf local) = 732 ms

Based on v47 profiling data (steps 8-10, rank 0, 3-step totals divided by 3):

| # | Root Cause | Lumen (ms/step) | TE est. (ms/step) | Gap (ms) | Status |
|---|-----------|-----------------|-------------------|---------|--------|
| 1 | **aten::copy_** (dtype casts, residual copies) | 289 | ~60 | **~229** | Reduced from 486→336→289; wgrad `.t()` view eliminated more copies |
| 2 | **FP8 quant/scale pipeline** | ~223 | 54 | **~169** | dynamic_quant=64, static_quant=49, amax_abs=94, compute_scale=4, abs+amax separate=12 |
| 3 | **aten::cat + add + add_** | ~200 | ~30 | **~170** | Autograd concat, elementwise residual ops |
| 4 | **Cast+transpose** | 162 | included | **~80** | Triton fused kernel; consumed by hipBLASLt wgrad |
| 5 | **Memory pressure** (98.5% vs 82%) | — | — | **~50-100** | Amplifies all allocation/free overhead |
| 6 | **SwiGLU residual** (Triton vs TE C++) | ~158 | ~100 | **~58** | fwd=49, bwd=75, fused_silu_mul=37 |
| 7 | **Memcpy DtoD** | 88 | ~20 | **~68** | Device copies from unfused ops |
| 8 | **Kernel dispatch** (~11.6K vs ~5K launches) | ~16 | ~8 | **~8** | Reduced 5x from 57K |
| 9 | **Other** (dropout, fill_, zero_, mul, neg, RoPE, NCCL) | ~120 | ~50 | **~70** | |
| | **Total estimated gap** | | | **~850** | vs measured 732 ms |

### Highest-Impact Next Steps

1. **Fuse copy/cast into GEMM epilogue** — 289 ms of `aten::copy_` is the #1 gap.
   TE avoids most copies by fusing dtype casts into GEMM epilogues (FP8→BF16 and
   BF16→FP8 happen inside the same kernel as the matmul). Implementing custom GEMM
   epilogues or using hipBLASLt's epilogue API could eliminate ~150-200 ms.

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

4. **Reduce memory pressure** — at 98.5% (189 GiB), the ROCm allocator fragments
   heavily. Each freed block may not be reusable for the next allocation. Reducing
   to ~90% would eliminate ~50-100 ms of allocator overhead. Options:
   - FP8 storage for linear inputs (currently BF16, ~24 GiB for 59 layers)
   - Reduce attention activation footprint

## Post-Eval Performance

| Metric | Lumen v47 | Lumen v48 | Lumen (current) | MLPerf Ref (local) |
|--------|-----------|-----------|-----------------|-------------------|
| Pre-eval ms/step | 4,730 | 4,747 | **4,699** | **3,967** |
| Post-eval #1 ms/step | 5,550 | 5,640 | **5,569** | ~4,000 |
| Post-eval delta | +17.3% | +18.8% | **+18.5%** | ~0% |

Root cause: Megatron's `transformer_block.py` skips activation checkpointing
during eval (`self.training=False`), allocating activations for all 80 layers
vs 21 during training. At 98%+ memory, this fragments the ROCm allocator and
the fragmentation persists after training resumes.

## Configuration Diff

| Parameter | Lumen (current) | MLPerf Reference (local) |
|-----------|-----------------|--------------------------|
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
| + Wgrad `.t()` + SwiGLU fused amax (v47) | 0.9223 | 4,730 ms | 1.19x |
| + Fused residual+norm + v48 kernel opts (v48) | 0.9211 | 4,747 ms | 1.20x |
| **+ NQG fusion + rsigma opt (current)** | **0.9190** | **4,699 ms** | **1.18x** |

## Source Files

### Core — Required for NQG Fusion and Training

| File | Purpose |
|------|---------|
| `lumen/ops/fused_norm_quant.py` | **NEW**: Fused RMSNorm + FP8 quant + `_FusedNQGNorm` autograd Function + rsigma output |
| `lumen/ops/fused_residual_norm.py` | **NEW**: Deferred BDA + autograd-aware RMSNorm + fused add+rmsnorm |
| `lumen/models/megatron_patches.py` | **MODIFIED**: `install_fused_residual_norm()` — NQG + deferred BDA patches on `TransformerLayer` |
| `lumen/quantize/__init__.py` | **MODIFIED**: Thread-local `_set_pre_quantized_activation` / `_pop_pre_quantized_activation` for NQG bypass |
| `lumen/ops/quantize/linear.py` | **MODIFIED**: `pre_quantized_input` param in `QuantizedLinearFunction` and `FP8StoredLinearFunction` |

### AITER Triton Kernel Changes

| File | Purpose |
|------|---------|
| `third_party/aiter/aiter/ops/triton/_triton_kernels/quant/fused_fp8_quant.py` | **MODIFIED**: `_rmsmorm_op` returns `(norm, rsigma)` tuple; kernel outputs rsigma via `OUTPUT_RSIGMA` flag |
| `third_party/aiter/aiter/ops/triton/quant/fused_fp8_quant.py` | **MODIFIED**: `output_rsigma=False` param; allocates rsigma tensor; returns 5-tuple when enabled |

### AITER C++ Fused Module (optional — JIT compiled, not used in current path)

| File | Purpose |
|------|---------|
| `third_party/aiter/aiter/ops/fused_norm_quant_gemm.py` | **NEW**: Python wrapper for C++ fused norm+quant+GEMM |
| `third_party/aiter/csrc/kernels/fused_norm_quant_gemm.cu` | **NEW**: HIP C++ host function (rmsnorm_quant + hipBLASLt) |
| `third_party/aiter/csrc/include/fused_norm_quant_gemm.h` | **NEW**: Header |
| `third_party/aiter/csrc/pybind/fused_norm_quant_gemm_pybind.cu` | **NEW**: pybind11 module |
| `third_party/aiter/csrc/include/rocm_ops.hpp` | **MODIFIED**: `FUSED_NORM_QUANT_GEMM_PYBIND` macro |
| `third_party/aiter/aiter/jit/optCompilerConfig.json` | **MODIFIED**: `module_fused_norm_quant_gemm` entry |
| `third_party/aiter/aiter/jit/core.py` | **MODIFIED**: added module to build list |

### Other Modified Files

| File | Purpose |
|------|---------|
| `lumen/modules/layernorm_linear.py` | `_FusedResidualRMSNormFP8Quant` / `_FusedRMSNormFP8QuantV2` autograd Functions |
| `lumen/modules/parallel_linear.py` | `is_contiguous()` guards |
| `lumen/ops/normalization/rmsnorm.py` | `rmsnorm_from_module` + `fused_add_rmsnorm` with autograd guard |
| `lumen/ops/quantize/cast_transpose.py` | `cast_amax_fp8` (no-transpose), `rmsnorm_quant_amax_fp8` kernels |
| `lumen/ops/quantize/quant_amax_fused.py` | `dequant_fp8_to_bf16` fused dequant kernel |
| `lumen/ops/dispatch.py` | `_probe_aiter_fused_quant` + backend caching |
| `lumen/ops/__init__.py` | Module exports |
| `lumen/ops/gemm/__init__.py` | GEMM epilogue exports |
| `lumen/quantize/scaling_manager.py` | `_quantize_core` with fused cast+transpose+amax path |
| `lumen/models/_swiglu_fp8_fuse.py` | `is_contiguous()` guard |
| `lumen/utils/hip_graphs.py` | HIP graph capture fixes (not actively used) |
| `examples/llama2/run_tp1_dp8.sh` | All env flags including `LUMEN_FUSED_NORM_QUANT_GEMM=1` |
| `examples/llama2/run_profile.sh` | Profiling script additions |
| `examples/llama2/scripts/patch_mlp_fp8_store.py` | SwiGLU pre-alloc fix |

### Obsolete Patch Scripts (integrated into megatron_patches.py)

| File | Status |
|------|--------|
| `examples/llama2/scripts/patch_fused_nqg.py` | **OBSOLETE** — functionality in `install_fused_residual_norm()` |
| `examples/llama2/scripts/patch_fused_residual_norm.py` | **OBSOLETE** — functionality in `install_fused_residual_norm()` |

### Non-Essential (generated data, debug logs)

| File | Status |
|------|--------|
| `examples/llama2/tunableop_results{0-7}.csv` | TunableOp cache — machine-specific, not needed for code submission |
| `.cursor/tmp-training-bugs.md` | Debug notes — not production code |
| `lumen/ops/gemm/epilogue.py` | NEW but **unused** (v49 FP8 output GEMM — OOM blocked) |
| `lumen/ops/gemm/fp8_output.py` | NEW but **unused** (v49 FP8 output GEMM — OOM blocked) |

## Profiling Data

| File | Description |
|------|------------|
| `profiling/lumen_profile_summary.txt` | `torch.profiler` key_averages — CK baseline, 3 steps, rank 0 |
| `profiling/lumen_latest_profile_summary.txt` | `torch.profiler` key_averages — v46 hipBLASLt all + `.t()` view, 3 steps, rank 0 |
| `profiling/te_profile_results.txt` | `torch.profiler` key_averages — TE ops (LayerNormLinear, Linear) at same tensor shapes, 10 iters |
