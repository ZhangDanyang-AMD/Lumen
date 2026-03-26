# Deferred Wgrad Overlap Benchmark Analysis - 8 GPU

**Source log**: `bench_wgrad_delay_8GPU.log:1-401`
**Benchmark**: `benchmarks/bench_wgrad_delay.py`
**Scope**: latency and overlap summary only. This cleaned log contains single-layer overlap numbers, 4-layer pipeline comparisons, Megatron-style realism, a shape sweep, and two profile experiments, but not pytest/runtime metadata.

The benchmark covers:

- isolated real-module deferred-wgrad overlap with `NCCL` and `SDMA`
- default-shape 4-layer sequential vs pipelined comparisons
- Tier 2 Megatron-style realism for `NCCL` and `SDMA`
- a `N=1024..57344` 4-layer scaling sweep
- two additional profiles: `backend_gap` and `pipeline_gain`
- both per-shape runs and a dedicated scaling-summary rerun

---

## Executive Summary

| Category | Key Finding |
|---|---|
| Overall backend ranking | the comparison blocks usually favor `NCCL`, but some standalone pipeline blocks are still mixed and must be read separately |
| Best comparison-block pipeline result | side-by-side 4-layer `deferred+NCCL` reaches **1.10x** speedup and saves **2.075 ms** |
| Standalone 4-layer caveat | earlier per-backend 4-layer blocks are still negative for both backends: **`NCCL 0.97x`**, **`SDMA 0.96x`** |
| Strongest microcase | isolated single-layer `NCCL` overlap reaches **1.13x** speedup and hides about **84%** of standalone allreduce time |
| SDMA default picture | isolated `0.98x`; standalone 4-layer `0.96x`; side-by-side 4-layer `1.00x` |
| Scaling sweep | Larger `N` reduces the loss, but the dedicated summary still tops out at only **0.96x** for `NCCL` and **0.95x** for `SDMA`, with `N=28672` effectively tied |
| Stability caveat | the one-off `N=57344` `NCCL` run reaches **1.01x**, but the later summary rerun falls back below break-even |
| Megatron-style realism | `NCCL` beats `SDMA` at **3.475 ms vs 3.865 ms** |

**Bottom line**: this log does not support a broad "SDMA is better" claim for deferred-wgrad overlap. On these 8-GPU tests, the direct comparison blocks usually favor `NCCL`, but some standalone pipeline blocks are still negative for both backends. Overlap gains are selective, and the shape sweep still does not show a stable crossover into clearly positive territory.

---

## 1. Default Profile Results

Default profile:

- `B=2`
- `S=2048`
- `H=4096`
- `FFN=14336`
- `chunks=4`
- `world=8`

### 1.1 Main Default Numbers

| Test | NCCL Result | SDMA Result | Main Reading |
|---|---:|---:|---|
| isolated single-layer overlap | `5.180 ms`, `1.13x` | `5.958 ms`, `0.98x` | only `NCCL` shows a clear gain in the simplest overlap microcase |
| standalone 4-layer pipeline block | `23.436 ms`, `0.97x` | `23.778 ms`, `0.96x` | when each backend is measured in its own default 4-layer block, both regress |
| side-by-side single-layer comparison | `6.037 ms`, `0.99x` | `6.120 ms`, `0.99x` | when rerun head-to-head, both backends are nearly flat |
| side-by-side 4-layer comparison | `20.850 ms`, `1.10x` | `23.364 ms`, `1.00x` | only this comparison block gives a clear pipeline win, and only for `NCCL` |
| Megatron-style realism | `3.475 ms` | `3.865 ms` | `NCCL` is about `1.11x` faster |

### 1.2 What the Default Profile Says

- In the isolated `NCCL` microcase, sequential minus overlapped latency is `0.665 ms`, which is about **84%** of the standalone `0.796 ms` allreduce time. So the default single-layer `NCCL` path can hide most of the comm cost when measured alone.
- `SDMA` fails the same isolated test. Its overlapped latency is `0.093 ms` slower than sequential, so there is no usable single-layer win in that default setup.
- The standalone per-backend 4-layer blocks are negative for both backends: `NCCL 22.844 -> 23.436 ms` (`0.97x`) and `SDMA 22.741 -> 23.778 ms` (`0.96x`).
- The direct side-by-side single-layer comparison is much less optimistic: both `NCCL` and `SDMA` fall to about `0.99x`. That means the isolated `NCCL` win is real evidence, but not yet robust across measurement setups.
- A later side-by-side 4-layer comparison is more favorable to `NCCL`. There, `NCCL` cuts `22.925 -> 20.850 ms` and saves `2.075 ms`, while `SDMA` stays flat at `23.272 -> 23.364 ms`.
- The Megatron-style realism case reinforces the same ranking. `NCCL` has lower latency than `SDMA`, so the overall default-shape evidence points to `NCCL > SDMA` for this benchmark.

This means the default shape already shows the core pattern of the log:

1. In the default-profile comparison blocks, `NCCL` is the stronger backend for this deferred-wgrad schedule.
2. Overlap wins are possible, but they are selective and not uniformly reproducible.

---

## 2. Scaling Sweep Results

### 2.1 Dedicated Scaling Summary

The dedicated summary rerun is the safer top-level view than any single standalone shape run, because several standalone rows are already flagged `!NOISY` or `~unstable`.

| N | NCCL (ms) | NCCL Speedup | SDMA (ms) | SDMA Speedup | Main Reading |
|---:|---:|---:|---:|---:|---|
| `1024` | `4.112` | `0.90x` | `4.270` | `0.86x` | both negative, `NCCL` better |
| `4096` | `4.484` | `0.91x` | `4.640` | `0.88x` | both negative, `NCCL` better |
| `7168` | `4.969` | `0.90x` | `5.110` | `0.87x` | both negative, `NCCL` better |
| `14336` | `6.187` | `0.89x` | `6.231` | `0.88x` | nearly tied, still negative |
| `28672` | `8.568` | `0.92x` | `8.561` | `0.92x` | practical tie, still below break-even |
| `57344` | `13.318` | `0.96x` | `13.481` | `0.95x` | closest to break-even, still negative in the summary rerun |

### 2.2 Main Scaling Takeaways

- Larger `N` clearly helps. `NCCL` improves from `0.90x` at `N=1024` to `0.96x` at `N=57344`, while `SDMA` improves from `0.86x` to `0.95x`.
- `NCCL` is equal or better at most summary points. `N=28672` is a practical tie, with `SDMA` lower by only `0.007 ms`.
- The dedicated summary still shows negative `ovl_eff` across the full sweep, so larger shapes amortize the pipeline tax but do not remove it.
- The small-shape end is the worst regime. At `N=1024`, `SDMA` drops to `0.86x` with `ovl_eff=-93%`, which means the fixed scheduling overhead dominates when useful compute is small.
- Some standalone scaling rows are explicitly noisy. The standalone `N=4096` sequential row is tagged `[!NOISY]`, and standalone `N=14336` rows are tagged `[~unstable]`, which is another reason to prefer the dedicated summary for top-level interpretation.

### 2.3 Largest-Shape Stability Note

- The standalone `N=57344` run briefly reports a narrow `NCCL` win: `12.891 -> 12.743 ms`, `1.01x`, `ovl_eff=8%`.
- But the later dedicated scaling-summary rerun reports `13.318 ms`, `0.96x`, `ovl_eff=-29%`.
- So the current data does **not** support claiming a stable crossover at the largest shape. The safe reading is: **approaching break-even, but not yet reproducibly past it**.

---

## 3. Profile Experiment Results

### 3.1 Additional Profiles

| Profile | NCCL Overlapped | NCCL Speedup | SDMA Overlapped | SDMA Speedup | Main Reading |
|---|---:|---:|---:|---:|---|
| `backend_gap` | `8.437 ms` | `1.12x` | `9.371 ms` | `1.01x` | widened backend gap creates a clear `NCCL` win but barely helps `SDMA` |
| `pipeline_gain` | `18.254 ms` | `1.03x` | `18.406 ms` | `1.02x` | more compute makes both backends slightly positive, with `NCCL` still better |

### 3.2 What These Experiments Say

- `backend_gap` is the clearest positive `NCCL` case in the log. `NCCL` saves `1.045 ms`, while `SDMA` saves only `0.067 ms`. So the added backend-side gap seems to expose overlap opportunity that `NCCL` can exploit much better.
- The `backend_gap` `SDMA` sequential row is explicitly tagged `[~unstable]`, so the weak `SDMA` result there should be treated with extra caution rather than as a perfectly clean measurement.
- `pipeline_gain` increases the compute side of the equation. Both backends improve slightly, but the speedups shrink to `1.03x` and `1.02x` because allreduce is now a smaller fraction of total time.
- In both experiments, `NCCL` remains at least as good as `SDMA`, so the backend ranking does not flip even in the "better overlap" profiles.

---

## 4. Overall Interpretation

This cleaned 8-GPU log supports four conclusions.

### 4.1 NCCL Is Usually Better In This Benchmark

`NCCL` wins or ties in the main comparison-driven places that matter:

- default 4-layer pipeline: `1.10x` vs `1.00x`
- Megatron-style realism: `3.475 ms` vs `3.865 ms`
- scaling summary: `NCCL` is equal or better at most `N`, with `N=28672` effectively tied

So for the deferred-wgrad schedule in `bench_wgrad_delay`, the current evidence points to `NCCL` as the stronger backend.

### 4.2 Overlap Wins Are Fragile

The log contains both:

- an isolated `NCCL` microcase with a real `1.13x` gain
- a direct side-by-side single-layer rerun where both backends fall back to about `0.99x`

That means overlap wins exist, but they are sensitive to harness details and schedule context. This is not yet the kind of result that can be treated as universally stable.

### 4.3 Larger Shapes Help, But The Crossover Is Not Stable

The shape sweep clearly trends in the right direction:

- penalties shrink as `N` grows
- the largest standalone `NCCL` point briefly turns positive

But the dedicated summary rerun falls back below `1.00x`, and several standalone rows are already flagged for noise or instability. So the correct interpretation is not "large shapes solve it"; it is "large shapes are the most promising regime, but the win is not yet reproducible."

### 4.4 This Log Is Latency-Only

Unlike `bench_e2e_fusion`, this log does not include bandwidth or memory tables. So this document can support only a latency/overlap conclusion:

- which backend is faster in this schedule
- where overlap helps or hurts
- how that changes with shape

It does **not** support any claim about hidden memory savings or bandwidth-driven explanations by itself.

---

## 5. Cross-Benchmark Reconciliation With `bench_e2e_fusion`

The statement "`NCCL` is usually better than `SDMA`" in this document is
**scope-limited**. It refers to the deferred-wgrad overlap and 4-layer pipeline
schedule inside `bench_wgrad_delay`, not to every fused layer schedule in
Lumen.

That distinction matters because `bench_e2e_fusion` measures a different unit
under test.

| Benchmark | Unit under test | What backend ranking means |
|-----------|-----------------|----------------------------|
| `bench_wgrad_delay` | deferred-wgrad overlap / 4-layer wgrad+allreduce pipeline | which backend hides delayed wgrad better in this schedule |
| `bench_e2e_fusion` | full single-layer pure-pipeline schedule | which backend gives lower end-to-end fused-layer latency |

This is why the two files can both be correct:

1. This file shows that the current deferred-wgrad pipeline usually favors
   `NCCL`.
2. `bench_e2e_fusion` shows that the current fused pure-pipeline layer backend
   favors `SDMA`.
3. Both files imply the same higher-level lesson: backend preference is
   **schedule-specific**, not a universal property of the transport.

So the safe reading is:

- **deferred wgrad overlap benchmark**: `NCCL >= SDMA` at the benchmark level, with one near-tie summary point around `N=28672`
- **pure-pipeline fused-layer benchmark**: `SDMA > NCCL`
- **against the respective baseline/reference**: wins are selective rather than universal

---

## Final Conclusion

The current 8-GPU `bench_wgrad_delay` performance summary says:

> **For this deferred-wgrad overlap benchmark, NCCL is usually the better backend, but overlap gains are fragile: NCCL gets a real win in some pipeline cases, SDMA rarely does, and the scaling sweep still stops short of a stable break-even point.**

That makes the next optimization questions narrower:

1. treat `NCCL` as the current default backend for deferred-wgrad pipeline work
2. focus on stabilizing the largest-shape `NCCL` gains and understanding why the direct comparison collapses the isolated single-layer win
3. do not transfer backend conclusions blindly from `bench_e2e_fusion` into `bench_wgrad_delay`, because the ranking depends on the schedule being tested
