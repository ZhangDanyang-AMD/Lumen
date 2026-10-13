# From-Scratch Kernel Writing Benchmark Results — MI308X (gfx942)

Hardware: AMD MI300X (MI308X), 8 GPUs, gfx942
Benchmark: 100 held-out tasks, model writes Triton kernel from zero via GEAK ToolAgentLoop.
Target: achieve >= 90% of AITER SOTA performance (speedup >= 0.9x).

## SFT-2F — Qwen3-Coder-30B-A3B-SFT-TH-2epoch

- Model: `Zhangdanyang/Qwen3-Coder-30B-A3B-SFT-TH-2epoch`
- Training data: 3164 samples (2000 patch + 1164 complete_implementation), messages chat format
- Dataset: `Zhangdanyang/agent-kernel-sft-gfx942-triton-HIP`

### Overall Summary (76/76 core tasks completed)

**Core benchmark** (76 tasks — excludes rope_kv_cache and sampling due to unfair baselines):

| Operator | Tasks | FAST (>=0.9x) | OK (<0.9x) | FAIL | Best Speedup |
|----------|-------|---------------|------------|------|-------------|
| gemm | 25 | **16** | 3 | 6 | 1.88x |
| rms_norm | 19 | **8** | 5 | 6 | 6.63x |
| mla | 6 | **1** | 0 | 5 | 9.62x (small shape) |
| mha | 14 | 0 | 0 | 14 | — |
| paged_attention | 12 | 0 | 0 | 12 | — |
| **Total** | **76** | **25 (33%)** | **8 (11%)** | **43 (57%)** | |

| Summary Metric | Value |
|----------------|-------|
| FAST rate | **25/76 (33%)** |
| Non-zero rate | **33/76 (43%)** |
| Best gemm | **1.88x** (profiler device-only) |
| Best rms_norm | **6.63x** (profiler device-only) |
| gemm FAST | **16/25 (64%)** |
| rms_norm FAST | **8/19 (42%)** |


### Results by Operator (Profiler Device-Only Time)

#### Core operators (fair AITER Triton baseline)

| Operator | Tasks | Snapshot Rate | FAST | OK | Best Speedup |
|----------|-------|---------------|------|-----|-------------|
| gemm | 21 | **17/21 (81%)** | **15** | 2 | 1.88x |
| rms_norm | 14 | **12/14 (86%)** | **8** | 4 | 6.63x |
| **Subtotal** | **35** | **29/35 (83%)** | **23** | **6** | |


### All Verified Kernel Results (Profiler Device-Only Time)

All 37 snapshots re-benchmarked with `torch.profiler` device-only timing.
Correctness verified, 3-run profiler median.

#### Core results (gemm + rms_norm — fair AITER Triton baseline)

| # | Task | Op | AITER (ms) | SFT (ms) | Speedup | Status |
|---|------|----|-----------|---------|---------|--------|
| 1 | aiter-prod-rms_norm-01 | rms_norm | 0.042 | 0.006 | **6.63x** | FAST |
| 2 | geak_native-rms_norm-02 | rms_norm | 0.043 | 0.007 | **6.25x** | FAST |
| 3 | aiter_derived-rms_norm-04 | rms_norm | 0.041 | 0.007 | **6.15x** | FAST |
| 4 | geak_native-rms_norm-04 | rms_norm | 0.038 | 0.007 | **5.77x** | FAST |
| 5 | adversarial_boundary-rms_norm-02 | rms_norm | 0.037 | 0.008 | **4.62x** | FAST |
| 6 | aiter-prod-rms_norm-07 | rms_norm | 0.038 | 0.009 | **4.27x** | FAST |
| 7 | adversarial_boundary-rms_norm-04 | rms_norm | 0.042 | 0.010 | **3.98x** | FAST |
| 8 | aiter_derived-gemm-03 | gemm | 0.148 | 0.078 | **1.88x** | FAST |
| 9 | aiter-prod-gemm-05 | gemm | 0.134 | 0.071 | **1.88x** | FAST |
| 10 | adversarial_boundary-gemm-04 | gemm | 0.070 | 0.039 | **1.81x** | FAST |
| 11 | aiter_derived-gemm-01 | gemm | 0.076 | 0.048 | **1.58x** | FAST |
| 12 | aiter-prod-gemm-04 | gemm | 0.130 | 0.085 | **1.53x** | FAST |
| 13 | adversarial_boundary-gemm-03 | gemm | 0.152 | 0.100 | **1.52x** | FAST |
| 14 | adversarial_boundary-gemm-02 | gemm | 0.072 | 0.048 | **1.49x** | FAST |
| 15 | aiter-prod-gemm-02 | gemm | 0.054 | 0.037 | **1.46x** | FAST |
| 16 | geak_native-gemm-04 | gemm | 0.074 | 0.052 | **1.42x** | FAST |
| 17 | aiter_derived-gemm-02 | gemm | 0.078 | 0.057 | **1.37x** | FAST |
| 18 | geak_native-gemm-02 | gemm | 0.079 | 0.060 | **1.33x** | FAST |
| 19 | geak_native-gemm-03 | gemm | 0.135 | 0.110 | **1.24x** | FAST |
| 20 | aiter-prod-gemm-08 | gemm | 1.151 | 0.953 | **1.21x** | FAST |
| 21 | aiter-prod-gemm-03 | gemm | 0.320 | 0.283 | **1.13x** | FAST |
| 22 | aiter-prod-rms_norm-04 | rms_norm | 0.007 | 0.007 | **1.01x** | FAST |
| 23 | aiter-prod-gemm-07 | gemm | 0.103 | 0.105 | **0.98x** | FAST |
| 24 | geak_native-gemm-01 | gemm | 0.080 | 0.088 | **0.90x** | FAST |
| 25 | aiter-prod-rms_norm-06 | rms_norm | 0.007 | 0.008 | 0.90x | OK |
| 26 | aiter-prod-gemm-01 | gemm | 0.050 | 0.063 | 0.78x | OK |
| 27 | aiter-prod-rms_norm-03 | rms_norm | 0.007 | 0.009 | 0.75x | OK |
| 28 | aiter-prod-rms_norm-02 | rms_norm | 0.006 | 0.009 | 0.73x | OK |
| 29 | aiter-prod-rms_norm-08 | rms_norm | 0.065 | 0.091 | 0.72x | OK |
| 30 | aiter_derived-rms_norm-02 | rms_norm | 0.007 | 0.010 | 0.72x | OK |
| 31 | adversarial_boundary-gemm-01 | gemm | 0.078 | 0.112 | 0.70x | OK |
| 32 | aiter-prod-gemm-06 | gemm | 0.258 | 0.375 | 0.69x | OK |


### Kernel Optimization Analysis (Profiler Device-Only Time)

All measurements use `torch.profiler` device-only time — only GPU kernel
execution is measured, host-side overhead (Python dispatch, kernel launch,
autotune search) is excluded. Code similarity between AITER and SFT kernels
is 8-12% — same standard algorithms but different implementations.

#### gemm (0.90-1.88x): Shape-aware tiling on GPU

| What AITER does | What SFT does | GPU impact |
|----------------|---------------|------------|
| BLOCK_M=128, conditional shrink to 16/64 | BLOCK_M directly matched to M (16/32) | Fewer wasted rows → less padding compute |
| BLOCK_K=32-64 | BLOCK_K=64-256 | Fewer K-loop iterations → higher arithmetic intensity |
| 1D grid + GROUP_SIZE_M=8 remapping | 2D grid `(cdiv(M,BM), cdiv(N,BN))` | Simpler index computation in kernel |
| num_warps=8 | num_warps=4 for small M | Better occupancy for small workloads |

Core algorithm identical: tiled GEMM with `tl.dot` FP32 accumulation, BF16
load/store. Speedup comes from **better tile size selection** for small-batch
shapes (M=2-37). The model learns to match BLOCK_M to the actual M dimension,
reducing wasted computation on padding rows. Larger BLOCK_K reduces loop
iterations, improving the compute-to-memory ratio.

#### rms_norm (0.72-6.63x): Kernel structure simplification

| What AITER does | What SFT does | GPU impact |
|----------------|---------------|------------|
| Persistent kernel: `grid=(NUM_PRGMS,)`, row loop | One-row-per-program: `grid=(M,)` | Eliminates loop overhead per row |
| `USE_BLOCKED` branch for large N | Single `BLOCK_SIZE=next_power_of_2(N)` | No branching in kernel |
| Blocked K-reduction with multi-pass | Single vectorized load per row | Fewer memory transactions |
| `tl.multiple_of` hints, `.cg` cache modifier | Standard loads | Simpler instruction sequence |

AITER's persistent kernel is designed for high-throughput with many rows —
it uses fewer program instances and loops internally. For small M (7-9 rows),
this creates overhead: each program does more work but fewer programs run in
parallel. SFT's one-row-per-program launches M programs, each doing minimal
work, achieving better GPU occupancy for small M.

For large M (>20 rows) and large N (8192), AITER's blocked approach starts to
win — SFT kernels at 0.72-0.75x on these shapes confirm the tradeoff.

#### mla (9.62x): Fused kernel vs tensor preparation overhead

AITER MLA calls a real Triton kernel (`_mla_prefill_fwd_kernel`), but the
wrapper does heavy tensor preparation on GPU before the kernel launch:

1. **2× `torch.cat`** — concatenate q_latent+q_rope and latent_kv+rope_k
2. **`torch.zeros` + `torch.cat`** — pad S to next power of 2
3. **`.unsqueeze().contiguous()`** — reshape with memory copy
4. **3× `torch.arange`/`torch.full`** — construct cu_seqlens_q, seqused_k, block_tables

Each of these dispatches a GPU kernel. On the tiny benchmark shape
(B=2, S=24, H=8, KV=64, ROPE=32), these preparation GPU kernels' device
time exceeds the actual attention kernel's execution time.

SFT writes a single ~100-line fused Triton kernel that reads directly from
input tensors, skipping all tensor cat/pad/reshape/contiguous operations.

**This is more legitimate than rope_kv_cache/sampling** — AITER does run a
real Triton attention kernel, but the small-shape tensor preparation overhead
dominates. On production shapes (B=32+, S=2048+), the overhead would be
amortized and AITER's optimized kernel would likely win. The 9.62x reflects
a small-batch advantage, not a fundamental algorithmic improvement.

### Comparison: SFT-2F vs SFT-2E

| Metric | SFT-2E (69/100) | SFT-2F (76/76 core, profiler) |
|--------|-----------------|-------------------------------|
| Training data | 2000 patch only | **3164** (+ 1164 impl) |
| Core FAST rate | 2/69 (2.9%) | **25/76 (33%)** |
| gemm FAST | 0 | **16/25 (64%)** |
| rms_norm FAST | 2 | **8/19 (42%)** |
| mla FAST | 0 | **1** (small shape) |
| Best gemm | 1.66x (unverified) | **1.88x (profiler)** |
| Best rms_norm | 1.83x | **6.63x (profiler)** |
| First-shot (t1-t2) | Rare (t3-t5) | **Common (t1-t2)** |

Key improvement: complete_implementation training data enabled gemm from-scratch
writing (16% -> 81% snapshot rate).

### Turn Budget Analysis

| Turns | Tasks | With Snapshots | Insight |
|-------|-------|---------------|---------|
| <=10 | ~15 | ~12 (80%) | Early exit, but model wrote good kernel first |
| 11-100 | ~15 | ~10 (67%) | Exception exits, snapshots captured |
| 101-200 | ~43 | ~13 (30%) | Full runs, complex ops mostly fail |

Best kernels appear at t1-t2 (first successful evaluate). 200-turn iteration
often degrades results by overwriting good kernels with broken ones.

---

## Profiler Re-benchmark (device-only time)

All 37 snapshots re-benchmarked with `torch.profiler` device-only timing
(excludes host/launch overhead). Previous measurements used `triton.testing.do_bench`
with `cuda.Event` timing which includes kernel launch overhead.

| Metric | do_bench (old) | profiler (new) |
|--------|---------------|----------------|
| FAST (>=0.9x) | ~15 | **29** |
| OK (>0, <0.9x) | ~7 | **8** |
| FAIL | ~15 | **0** |
| Total snapshots | 37 | 37 |

### Top profiler results

| Task | Op | Speedup | AITER (ms) | SFT (ms) |
|------|----|---------|-----------|---------|
| aiter-prod-rope_kv_cache-03 | rope_kv_cache | 1478.76x* | 114.94 | 0.078 |
| aiter-prod-rope_kv_cache-02 | rope_kv_cache | 133.83x* | 0.868 | 0.006 |
| aiter_derived-rope_kv_cache-02 | rope_kv_cache | 79.88x* | 0.507 | 0.006 |
| adversarial_boundary-rope_kv_cache | rope_kv_cache | 10.76x* | 0.262 | 0.024 |
| aiter_derived-sampling-02 | sampling | 9.92x* | 0.130 | 0.013 |
| aiter-prod-rms_norm-01 | rms_norm | 6.63x | 0.042 | 0.006 |
| geak_native-rms_norm-02 | rms_norm | 6.25x | 0.043 | 0.007 |
| aiter_derived-rms_norm-04 | rms_norm | 6.15x | 0.041 | 0.007 |
| geak_native-rms_norm-04 | rms_norm | 5.77x | 0.038 | 0.007 |
| adversarial_boundary-rms_norm-02 | rms_norm | 4.62x | 0.037 | 0.008 |
| aiter-prod-gemm-05 | gemm | 1.88x | 0.134 | 0.071 |
| aiter_derived-gemm-03 | gemm | 1.88x | 0.148 | 0.078 |
| adversarial_boundary-gemm-04 | gemm | 1.81x | 0.070 | 0.039 |
| aiter_derived-gemm-01 | gemm | 1.58x | 0.076 | 0.048 |
| aiter-prod-gemm-04 | gemm | 1.53x | 0.130 | 0.085 |

*rope_kv_cache and sampling baselines use Python fallback, not AITER C++ kernels.

Full results saved to `receipts/profiler_rebenchmark.json`.

---

## Benchmark Infrastructure

```
GPU 0:   vLLM model serving (262K context, --tool-call-parser qwen3_coder)
GPU 2-7: Kernel evaluation (6 parallel workers)
Loop:    200 turns max, GEAK ToolAgentLoop (read_file, write_file, evaluate)
Timing:  torch.profiler device-only time (excludes host/launch overhead)
Baseline: AITER kernels with frozen autotune configs (receipts/autotune_freeze.json)
Validate: kernel_validator.py blocks PyTorch ops in wrapper
Snapshot: kernel_sp{X}_t{N}.py + kernel_best.py on each successful evaluate
Rebench:  Restore kernel_best.py, verify correctness, 3-run profiler median
```
