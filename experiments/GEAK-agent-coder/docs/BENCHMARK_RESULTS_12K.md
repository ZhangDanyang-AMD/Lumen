# Benchmark Results: Base vs SFT on AMD GPU Kernel Optimization

**Date:** 2026-10-04 (V8 final with speedup analysis)
**Hardware:** 8x AMD MI308X (gfx942), GPU 0 for model serving, GPU 1-7 for kernel eval
**Held-out suite:** 99 tasks (54 Triton + 45 HIP), 10 operator families × 3 source suites, verified no training data overlap
**Models:** Base (Qwen3-Coder-30B-A3B-Instruct), SFT-2epoch (val_loss 0.178), SFT-4epoch (val_loss 0.173)
**Benchmark version:** V8 (fuzzy patch + full-code fallback + non-patch retry + multi-turn 5-turn error recovery + agent loop metrics)

---

## 0. Executive Summary

**Key finding: SFT improves generation capability but not optimization quality.**

- Patch mode measures "edit without breaking" — high pass rate but almost no speedup (geomean 0.74-1.00x)
- Generation mode measures "write kernel from scratch" — SFT improves compile/correct rate but generated kernels are slower than baseline (geomean 0.20-0.36x)
- **The meaningful optimization metric is Fast@1.2** (patches that achieve ≥20% speedup): SFT-2e leads at 26%, vs Base 13% and SFT-4e 11%
- SFT-4e shows overfitting: more patches apply but fewer produce speedup compared to SFT-2e

---

## 1. Speedup Analysis (Primary Metric)

### 1.1 Patch Mode — Speedup vs Baseline

Given parent kernel source, model generates a unified diff to optimize it. Speedup = baseline_ms / patched_ms.

| Metric | Base | SFT-2epoch | SFT-4epoch |
|--------|------|-----------|-----------|
| Correct patches | 48/99 | 42/99 | 37/99 |
| **Speedup geomean** | 0.762x | **1.001x** | 0.742x |
| Faster@1.0 (any speedup) | 16/48 (33%) | **19/42 (45%)** | 14/37 (38%) |
| **Fast@1.2 (≥20% speedup)** | 6/48 (13%) | **11/42 (26%)** | 4/37 (11%) |
| Fast@1.5 (≥50% speedup) | 5/48 (10%) | **6/42 (14%)** | 3/37 (8%) |
| Fast@2.0 (≥100% speedup) | 0/48 | 0/42 | 0/37 |

**SFT-2epoch is the best optimizer**: only model with geomean ≥1.0x, highest Fast@1.2 rate (26%), and highest Faster@1.0 rate (45%).

SFT-4epoch regressed on speedup despite having comparable pass rate, suggesting memorization of patch patterns rather than optimization reasoning.

#### Per-Language Breakdown (Triton vs HIP)

**Triton kernels:**

| Metric | Base | SFT-2epoch | SFT-4epoch |
|--------|------|-----------|-----------|
| Correct patches | 35 | 35 | 27 |
| Speedup geomean | 0.886x | 0.984x | 0.898x |
| Faster@1.0 | 12/35 (34%) | 14/35 (40%) | 10/27 (37%) |
| **Fast@1.2** | 4/35 (11%) | **7/35 (20%)** | 3/27 (11%) |
| Fast@1.5 | 4/35 (11%) | **5/35 (14%)** | 3/27 (11%) |

**HIP kernels:**

| Metric | Base | SFT-2epoch | SFT-4epoch |
|--------|------|-----------|-----------|
| Correct patches | 13 | 7 | 10 |
| Speedup geomean | 0.508x | **1.092x** | 0.445x |
| Faster@1.0 | 4/13 (30%) | **5/7 (71%)** | 4/10 (40%) |
| **Fast@1.2** | 2/13 (15%) | **4/7 (57%)** | 1/10 (10%) |
| Fast@1.5 | 1/13 (7%) | 1/7 (14%) | 0/10 (0%) |

**Key finding:** SFT-2epoch 在 HIP kernel 上优化效果最突出 — Fast@1.2 达 57%（vs Base 15%），geomean 1.09x（唯一 >1.0x 的组合）。Triton kernel 优化效果较均匀（Fast@1.2 11-20%）。HIP kernel 因为用 C++ inline extension 编写，patch 空间更大（loop unroll、memory coalescing、warp-level primitives），SFT 学到了有效的 HIP 优化 pattern。

### 1.2 Generation Mode — Speedup vs Baseline

Model writes kernel.py from scratch given only the operator contract. Speedup = baseline_ms / generated_ms.

| Metric | Base | SFT-2epoch | SFT-4epoch |
|--------|------|-----------|-----------|
| Correct kernels | 7/99 | 4/99 | 12/99 |
| **Speedup geomean** | 0.333x | 0.356x | 0.199x |
| Faster@1.0 | 2/7 | 1/4 | 1/12 |
| Fast@1.2 | 0/7 | 1/4 | 0/12 |

**No model produces competitive from-scratch kernels.** Generated kernels are 3-5x slower than expert-written baselines. This is expected — cold-start generation without profiling data cannot match tuned implementations.

### 1.3 Interpretation

Patch mode with speedup filtering is the meaningful benchmark:
- **Pass@1 (compile+correct)** measures "can the model edit a kernel without breaking it" — necessary but not sufficient
- **Fast@1.2** measures "can the model actually make kernels faster" — this is the optimization metric
- Models that pass more patches but with lower speedup are just making safe no-op edits, not optimizing

---

## 2. Correctness Benchmark (V8 Multi-Turn)

### 2.1 Patch Mode (5-turn error recovery, fuzzy patch apply)

| Metric | Base | SFT-2epoch | SFT-4epoch |
|--------|------|-----------|-----------|
| Total PASS (patch+gen combined) | **55/99** | 46/99 | 49/99 |
| Patch correct | **48/99** | 42/99 | 37/99 |
| Gen correct | 7/99 | 4/99 | **12/99** |
| Recovered via multi-turn | 35 | 27 | **42** |
| Patch apply rate | 100% | 100% | 100% |

### 2.2 Generation Mode (compile + correctness, no speedup gate)

| Metric | Base | SFT-2epoch | SFT-4epoch |
|--------|------|-----------|-----------|
| Code generated | 99/99 | 99/99 | 99/99 |
| **Compiled** | 27/99 (27%) | 27/99 (27%) | **40/99 (40%)** |
| **Correct** | 7/99 (7%) | 4/99 (4%) | **12/99 (12%)** |

SFT-4epoch compile rate is 1.5x base (40% vs 27%). This is the clearest SFT signal for code generation.

---

## 3. Agent Loop Efficiency (V8)

| Metric | Base | SFT-2epoch | SFT-4epoch |
|--------|------|-----------|-----------|
| First-turn pass rate (patch) | 48% | 42% | 37% |
| Turns to pass (mean) | 2.1 | 2.4 | 2.9 |
| Error recovery rate (multi-turn) | 35/55 (64%) | 27/46 (59%) | 42/49 (86%) |
| Compile error rate (gen T1) | 73% | 73% | 60% |

SFT-4epoch has the highest error recovery rate (86%) — SFT training improved the model's ability to fix its own mistakes when given error feedback.

---

## 4. Per-Operator Breakdown (Patch Mode, Speedup)

### Fast@1.2 by operator family (SFT-2epoch, best optimizer)

| Operator | Correct | Fast@1.2 | Notable |
|----------|---------|----------|---------|
| rms_norm | 7/12 | 3/7 | Best optimization target |
| paged_attention | 2/6 | 2/2 | 1.75-1.87x speedups |
| rope_kv_cache | 3/12 | 2/3 | 1.82-1.85x from unrolling |
| mha | 2/12 | 1/2 | 1.77x on aiter_derived |
| gemm | 4/12 | 1/4 | Inconsistent |
| fused_moe | 2/12 | 1/2 | 1.21x |
| blockscale_gemm | 0/6 | 0/0 | — |
| mla | 0/6 | 0/0 | — |
| sampling | 3/9 | 0/3 | Correct but no speedup |

Attention-family operators (paged_attention, rope_kv_cache, mha) show the highest speedup potential.

---

## 5. Training Data Gap Analysis

### Current state (2000 gfx942 kernel SFT samples)

| Operator | Samples | Patch Fast@1.2 (SFT-2e) | Assessment |
|----------|---------|--------------------------|------------|
| rms_norm | 19 | 43% | **Good optimization signal** |
| paged_attention | 13 | 100% (2/2 correct) | **High potential, need more data** |
| mha | 13 | 50% (1/2 correct) | Need more data |
| mla | 22 | 0% (0 correct) | **Severely insufficient** |
| gemm | 19 | 25% | Needs shape diversity |
| fused_moe | 7 | 50% (1/2 correct) | **Severely insufficient** |
| rope_kv_cache | 11 | 67% (2/3 correct) | Good signal, need more |
| sampling | 17 | 0% (3 correct, 0 fast) | Model edits are safe but not faster |

### Priority data additions for optimization quality

1. **paged_attention** (+137 samples): 100% Fast@1.2 when correct — highest ROI
2. **mha** (+137 samples): 50% Fast@1.2 when correct, only 2 correct — volume needed
3. **fused_moe** (+143 samples): MoE routing + expert GEMM fusion, critical for production
4. **mla** (+128 samples): 0% compile rate — model cannot produce these at all
5. **rope_kv_cache** (+89 samples): 67% Fast@1.2 when correct, strong signal

---

## 6. Key Conclusions

1. **Patch mode speedup is the right metric.** Pass@1 (compile+correct) without speedup is misleading — Base has the highest pass rate but worst optimization quality.

2. **SFT-2epoch > SFT-4epoch for optimization.** 2-epoch model produces fewer correct patches but more of them actually speed things up (26% vs 11% Fast@1.2). 4-epoch overfits to template patterns.

3. **From-scratch generation is not competitive.** No model produces kernels anywhere close to expert baseline performance. Generation mode is useful for testing code synthesis, not optimization.

4. **RL training should optimize for speedup, not just correctness.** The GRPO reward function (`reward = 1.0 + clip(log(speedup), 0, log3)`) correctly prioritizes speedup. Target: speedup geomean >1.2x, Fast@1.2 >40%.

5. **Attention operators are the highest-value training targets.** paged_attention and rope_kv_cache show 67-100% Fast@1.2 when correct, but have very few training samples (11-13).

---

## 7. Methodology

### V8 Benchmark Infrastructure

| Feature | Description |
|---------|-------------|
| Fuzzy patch apply | Python context-insensitive hunk matching (`fuzzy_patch.py`) — 100% apply rate |
| Full-code fallback | If model outputs complete code instead of patch, use directly |
| Non-patch retry | If SFT model outputs explanation, ask for regeneration |
| Multi-turn recovery | Up to 5 turns of error feedback + retry |
| Arch gate removal | Auto-remove broken gfx942/NVIDIA architecture checks |
| Function aliasing | Auto-add expected function name alias |
| Signature in prompt | Full reference function signature from initial_source.py |
| Agent loop metrics | First-turn pass, turns-to-pass, error recovery rate |
| **Speedup measurement** | baseline_ms/candidate_ms from GEAK performance harness |

### Held-out dataset

- 120 tasks from `Zhangdanyang/agent-phase1-held-out-private`
- 99 pass baseline (12 all_reduce excluded, 9 adversarial shape failures)
- 10 operator families × 3 source suites × 4 shape variants
- SHA-256 verified harnesses, zero training data overlap

### Model serving

vLLM 0.15.0+rocm700, TP=1, 32K context, enforce-eager, qwen3_coder tool parser
