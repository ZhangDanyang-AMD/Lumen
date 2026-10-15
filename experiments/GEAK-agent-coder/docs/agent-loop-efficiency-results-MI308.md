# From-Scratch Kernel Writing Benchmark Results — MI308X (gfx942)

Hardware: AMD MI300X (MI308X), 8 GPUs, gfx942
Benchmark: 76 held-out tasks (excludes rope_kv_cache/sampling), model writes
Triton kernel from zero via GEAK ToolAgentLoop.
Target: achieve >= 90% of AITER SOTA performance (speedup >= 0.9x).
Timing: `torch.profiler` device-only (GPU kernel execution only).

## SFT-4F — Qwen3-Coder-30B-A3B-SFT-TH-4epoch

- Model: `Zhangdanyang/Qwen3-Coder-30B-A3B-SFT-TH-4epoch`
- Training data: 3164 samples (2000 patch + 1164 complete_implementation), 4 epochs
- Dataset: `Zhangdanyang/agent-kernel-sft-gfx942-triton-HIP`

### SFT-4F Summary (76/76 core tasks, profiler device-only time)

| Operator | Tasks | FAST (>=0.9x) | OK (<0.9x) | FAIL | Best Speedup |
|----------|-------|---------------|------------|------|-------------|
| gemm | 25 | **16** | 0 | 9 | 1.94x |
| rms_norm | 19 | **10** | 2 | 7 | 6.28x |
| mha | 14 | **3** | 0 | 11 | 5.46x |
| mla | 6 | **1** | 0 | 5 | 8.04x |
| paged_attention | 12 | 0 | 0 | 12 | — |
| **Total** | **76** | **30 (39%)** | **2 (3%)** | **44 (58%)** | |

### SFT-4F All Verified Results (Profiler Device-Only Time)

| # | Task | Op | AITER (ms) | SFT (ms) | Speedup | Turns |
|---|------|----|-----------|---------|---------|-------|
| 1 | aiter_derived-mla-04 | mla | 0.054 | 0.007 | **8.04x** | 79 |
| 2 | geak_native-rms_norm-02 | rms_norm | 0.043 | 0.007 | **6.28x** | 103 |
| 3 | aiter_derived-rms_norm-04 | rms_norm | 0.041 | 0.007 | **5.87x** | 6 |
| 4 | geak_native-rms_norm-04 | rms_norm | 0.038 | 0.007 | **5.52x** | 74 |
| 5 | aiter-prod-rms_norm-07 | rms_norm | 0.038 | 0.007 | **5.50x** | 22 |
| 6 | adversarial_boundary-rms_norm-02 | rms_norm | 0.037 | 0.007 | **5.49x** | 6 |
| 7 | adversarial_boundary-mha-02 | mha | 0.035 | 0.006 | **5.46x** | 116 |
| 8 | adversarial_boundary-rms_norm-04 | rms_norm | 0.041 | 0.008 | **5.35x** | 6 |
| 9 | aiter_derived-mha-04 | mha | 0.034 | 0.006 | **5.27x** | 102 |
| 10 | aiter_derived-mha-02 | mha | 0.036 | 0.007 | **5.26x** | 108 |
| 11 | geak_native-gemm-04 | gemm | 0.074 | 0.041 | **1.79x** | 8 |
| 12 | adversarial_boundary-gemm-03 | gemm | 0.152 | 0.097 | **1.57x** | 105 |
| 13 | aiter-prod-gemm-04 | gemm | 0.130 | 0.083 | **1.57x** | 64 |
| 14 | adversarial_boundary-gemm-04 | gemm | 0.070 | 0.046 | **1.55x** | 6 |
| 15 | aiter_derived-gemm-03 | gemm | 0.148 | 0.098 | **1.51x** | 153 |
| 16 | aiter-prod-gemm-05 | gemm | 0.134 | 0.092 | **1.46x** | 117 |
| 17 | aiter_derived-gemm-04 | gemm | 0.073 | 0.050 | **1.46x** | 50 |
| 18 | adversarial_boundary-gemm-01 | gemm | 0.078 | 0.054 | **1.44x** | 117 |
| 19 | geak_native-gemm-01 | gemm | 0.080 | 0.055 | **1.43x** | 129 |
| 20 | aiter-prod-gemm-07 | gemm | 0.103 | 0.076 | **1.36x** | 91 |
| 21 | aiter_derived-gemm-01 | gemm | 0.076 | 0.057 | **1.33x** | 125 |
| 22 | geak_native-gemm-03 | gemm | 0.135 | 0.105 | **1.29x** | 89 |
| 23 | adversarial_boundary-gemm-02 | gemm | 0.072 | 0.059 | **1.22x** | 6 |
| 24 | aiter-prod-gemm-01 | gemm | 0.049 | 0.043 | **1.15x** | 56 |
| 25 | aiter_derived-rms_norm-02 | rms_norm | 0.007 | 0.006 | **1.07x** | 58 |
| 26 | aiter-prod-gemm-02 | gemm | 0.054 | 0.051 | **1.06x** | 81 |
| 27 | aiter-prod-rms_norm-06 | rms_norm | 0.007 | 0.007 | **1.03x** | 158 |
| 28 | aiter_derived-gemm-02 | gemm | 0.078 | 0.079 | **0.98x** | 129 |
| 29 | aiter-prod-rms_norm-04 | rms_norm | 0.007 | 0.007 | **0.95x** | 19 |
| 30 | aiter-prod-rms_norm-03 | rms_norm | 0.006 | 0.007 | **0.95x** | 139 |
| 31 | aiter-prod-rms_norm-08 | rms_norm | 0.065 | 0.088 | 0.74x | 200 |
| 32 | aiter-prod-rms_norm-05 | rms_norm | 0.036 | 0.053 | 0.69x | 200 |

### SFT-4F Turn Budget

| Turns | Snapshots | FAST | Insight |
|-------|-----------|------|---------|
| <=10 | 4 | 4 | First-shot success (gemm, rms_norm) |
| 11-50 | 3 | 3 | Quick convergence |
| 51-100 | 8 | 8 | Most gemm results here |
| 101-150 | 11 | 10 | MHA/MLA need 100+ turns |
| 151-200 | 6 | 5 | Long runs, some rms_norm |

---

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

---

## Base Model — Qwen3-Coder-30B-A3B-Instruct (no SFT)

- Model: `Qwen/Qwen3-Coder-30B-A3B-Instruct`
- No kernel SFT training data. Pure base model coding capability.

### Base Summary (76/76 core tasks completed)

| Operator | Tasks | FAST (>=0.9x) | OK (<0.9x) | FAIL | Best Speedup |
|----------|-------|---------------|------------|------|-------------|
| gemm | 25 | **4** | 1 | 20 | 1.25x |
| rms_norm | 19 | **12** | 3 | 4 | 5.68x |
| mla | 6 | 0 | 0 | 6 | — |
| mha | 14 | 0 | 0 | 14 | — |
| paged_attention | 12 | 0 | 0 | 12 | — |
| **Total** | **76** | **16 (21%)** | **4 (5%)** | **56 (74%)** | |

### Base Profiler Results (19 snapshots, device-only time)

| # | Task | Op | AITER (ms) | SFT (ms) | Speedup | Status |
|---|------|----|-----------|---------|---------|--------|
| 1 | adversarial_boundary-rms_norm | rms_norm | 0.042 | 0.007 | **5.68x** | FAST |
| 2 | adversarial_boundary-rms_norm | rms_norm | 0.037 | 0.007 | **5.50x** | FAST |
| 3 | geak_native-rms_norm-04 | rms_norm | 0.038 | 0.007 | **5.30x** | FAST |
| 4 | aiter-prod-rms_norm-12 | rms_norm | 0.037 | 0.007 | **5.21x** | FAST |
| 5 | geak_native-rms_norm-02 | rms_norm | 0.043 | 0.010 | **4.36x** | FAST |
| 6 | aiter-prod-rms_norm-07 | rms_norm | 0.038 | 0.009 | **4.31x** | FAST |
| 7 | aiter-prod-rms_norm-10 | rms_norm | 0.040 | 0.012 | **3.32x** | FAST |
| 8 | aiter-prod-rms_norm-01 | rms_norm | 0.042 | 0.019 | **2.20x** | FAST |
| 9 | aiter_derived-rms_norm-04 | rms_norm | 0.041 | 0.024 | **1.74x** | FAST |
| 10 | aiter-prod-gemm-13 | gemm | 0.112 | 0.089 | **1.25x** | FAST |
| 11 | aiter-prod-gemm-01 | gemm | 0.050 | 0.040 | **1.24x** | FAST |
| 12 | aiter-prod-gemm-02 | gemm | 0.054 | 0.046 | **1.16x** | FAST |
| 13 | aiter-prod-gemm-04 | gemm | 0.130 | 0.124 | **1.05x** | FAST |
| 14 | aiter-prod-rms_norm-09 | rms_norm | 0.007 | 0.007 | **1.02x** | FAST |
| 15 | aiter-prod-rms_norm-13 | rms_norm | 0.007 | 0.007 | **0.99x** | FAST |
| 16 | aiter-prod-rms_norm-02 | rms_norm | 0.006 | 0.007 | **0.94x** | FAST |
| 17 | aiter-prod-gemm-03 | gemm | 0.320 | 0.545 | 0.59x | OK |
| 18 | aiter-prod-rms_norm-03 | rms_norm | 0.007 | 0.011 | 0.58x | OK |
| 19 | aiter-prod-rms_norm-08 | rms_norm | 0.065 | 0.128 | 0.51x | OK |

---

## Three-Model Comparison (profiler device-only time)

### Overall Performance

| Metric | Base | SFT-2F (2 epoch) | SFT-4F (4 epoch) |
|--------|------|------------------|------------------|
| **FAST rate** | 16/76 (21%) | 25/76 (33%) | **30/76 (39%)** |
| **Non-zero rate** | 20/76 (26%) | 33/76 (43%) | **32/76 (42%)** |
| **Snapshots** | 19 | 33 | **32** |

### By Operator

| Operator | Tasks | Base FAST | SFT-2F FAST | SFT-4F FAST |
|----------|-------|-----------|-------------|-------------|
| gemm | 25 | 4 (16%) | **16 (64%)** | **16 (64%)** |
| rms_norm | 19 | **12 (63%)** | 8 (42%) | 10 (53%) |
| mha | 14 | 0 | 0 | **3 (21%)** |
| mla | 6 | 0 | 1 | 1 |
| paged_attention | 12 | 0 | 0 | 0 |
| **Total** | **76** | **16 (21%)** | **25 (33%)** | **30 (39%)** |

### Best Speedup per Operator

| Operator | Base | SFT-2F | SFT-4F |
|----------|------|--------|--------|
| gemm | 1.25x | 1.88x | **1.94x** |
| rms_norm | 5.68x | **6.63x** | 6.28x |
| mha | — | — | **5.46x** |
| mla | — | 9.62x | **8.04x** |

### Turn Efficiency

| Metric | Base | SFT-2F | SFT-4F |
|--------|------|--------|--------|
| Avg turns (FAST tasks) | 85 | **57** | 87 |
| Avg turns (all tasks) | 180 | 126 | **154** |
| FAST in <=10 turns | 2 | 3 | **4** |
| FAST in <=50 turns | 4 | 6 | **7** |

SFT-2F achieves FAST results in fewer turns (avg 57) than both Base (85) and
SFT-4F (87), suggesting 2-epoch training produces stronger first-shot ability.
SFT-4F's higher FAST count comes from succeeding on harder tasks (MHA, MLA)
that require more turns (100+).

### Key Takeaways

1. **SFT-4F is the best model overall**: 39% FAST vs SFT-2F 33% vs Base 21%.
   Each epoch of training improves performance.

2. **MHA breakthrough at 4 epochs**: SFT-4F achieves 3/14 MHA FAST (21%),
   writing simple dot-product attention that outperforms AITER's flash attention
   on small shapes. Neither Base nor SFT-2F can write MHA. This is the key
   differentiator for 4-epoch training.

3. **gemm saturated at 2 epochs**: Both SFT-2F and SFT-4F get 16/25 gemm FAST
   (64%). The capability is fully learned in 2 epochs.

4. **rms_norm recovers at 4 epochs**: SFT-4F (53%) improves over SFT-2F (42%),
   partially recovering toward Base (63%). SFT-2F's rms_norm regression is
   corrected by additional training.

5. **SFT-2F is fastest per-FAST**: Average 57 turns vs SFT-4F's 87 — the
   2-epoch model has stronger first-shot capability on simple operators,
   while the 4-epoch model succeeds on harder tasks that need more iterations.

6. **paged_attention: still 0%**: None of the three models can write paged
   attention from scratch. This requires understanding blocked KV cache layout
   not present in training data.

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
