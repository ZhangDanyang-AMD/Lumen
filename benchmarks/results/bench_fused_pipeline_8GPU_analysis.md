# Fused Pipeline Backend Matrix Benchmark Analysis — 8 GPU

**Source log**: `bench_fused_pipeline_8GPU.log`
**Benchmark**: `benchmarks/bench_fused_pipeline.py`
**Scope**: performance summary only. This cleaned log contains RCCL-vs-SDMA fused-pipeline timings and the final backend matrix table, but not pytest/runtime metadata.

The benchmark covers:

- `column_fwd`
- `row_fwd`
- `column_fwd_bwd`
- `row_fwd_bwd`
- `chunks=1/2/4/8`
- RCCL vs SDMA on `world=8`

---

## Executive Summary

| Category | Key Finding |
|---|---|
| Overall backend winner | `SDMA` is faster than `RCCL` in **13 / 16** measured phase-chunk points |
| Most consistent win | `column_fwd` and `column_fwd_bwd` are always faster with SDMA |
| Weakest area | `row_fwd` is worse with SDMA at `chunks=1/2/4`, and only barely wins at `chunks=8` |
| Best SDMA gain | `column_fwd` at `chunks=8` reaches **1.85x** vs RCCL |
| Best end-to-end gain | `row_fwd_bwd` at `chunks=8` reaches **1.67x** vs RCCL |
| Main negative trend | Increasing `num_chunks` makes the **absolute latency worse for both backends** in every phase |
| Stability risk | Several RCCL points are marked `unstable`, especially `*_fwd_bwd` at `chunks=1/8` |

**Bottom line**: transport choice clearly matters, because SDMA is usually the better fused backend on this 8-GPU run. But the matrix also shows a second, equally important result: making the pipeline more chunked increases fixed overhead fast enough that absolute latency keeps rising with `chunks`, even when SDMA beats RCCL.

---

## 1. Important Interpretation Note

This log does **not** contain a separate "naive / no-communication" baseline.

In this file:

- `speedup_vs_rccl` means **RCCL fused path is the reference baseline**
- `1.00x` on RCCL rows is the reference value
- SDMA rows above `1.0x` mean SDMA is faster than RCCL for the same phase and chunk count

So when this analysis says "worse than baseline", there are two possible meanings:

1. **Worse than the RCCL reference row** for the same phase/chunk
2. **Worse than the coarser chunk baseline** such as `chunks=1`

This distinction matters because the log supports both kinds of conclusions:

- `row_fwd` at `chunks=1/2/4` is worse than the RCCL reference
- almost every phase gets worse in absolute time as chunk count increases, compared with `chunks=1`

---

## 2. Backend Matrix Summary

### 2.1 Final Matrix

| Phase | Chunks | RCCL (ms) | SDMA (ms) | SDMA vs RCCL | Main Reading |
|---|---:|---:|---:|---:|---|
| `column_fwd` | 1 | 0.616 | 0.454 | 1.36x | SDMA wins |
| `column_fwd` | 2 | 0.933 | 0.622 | 1.50x | SDMA wins |
| `column_fwd` | 4 | 1.524 | 0.914 | 1.67x | SDMA wins |
| `column_fwd` | 8 | 2.749 | 1.486 | 1.85x | SDMA wins, both unstable |
| `row_fwd` | 1 | 0.916 | 1.216 | 0.75x | SDMA loses |
| `row_fwd` | 2 | 1.023 | 1.314 | 0.78x | SDMA loses |
| `row_fwd` | 4 | 1.330 | 1.697 | 0.78x | SDMA loses, SDMA unstable |
| `row_fwd` | 8 | 2.035 | 1.971 | 1.03x | Essentially break-even, slight SDMA win |
| `column_fwd_bwd` | 1 | 2.126 | 1.809 | 1.18x | SDMA wins, RCCL unstable |
| `column_fwd_bwd` | 2 | 3.077 | 2.305 | 1.34x | SDMA wins |
| `column_fwd_bwd` | 4 | 4.640 | 3.229 | 1.44x | SDMA wins |
| `column_fwd_bwd` | 8 | 8.720 | 5.355 | 1.63x | SDMA wins, RCCL unstable |
| `row_fwd_bwd` | 1 | 3.234 | 3.014 | 1.07x | Small SDMA win, RCCL unstable |
| `row_fwd_bwd` | 2 | 4.194 | 3.713 | 1.13x | SDMA wins, RCCL unstable |
| `row_fwd_bwd` | 4 | 5.469 | 4.527 | 1.21x | SDMA wins |
| `row_fwd_bwd` | 8 | 11.061 | 6.617 | 1.67x | SDMA wins, RCCL unstable |

### 2.2 Main Takeaways

- `SDMA` is the better backend almost everywhere.
- The only clear exception is `row_fwd`, where SDMA is slower until `chunks=8`.
- `column_*` phases benefit the most from SDMA, especially as chunk count grows.
- `row_fwd_bwd` eventually shows a strong SDMA gain, even though `row_fwd` alone does not.

That last point is important. It says the row-parallel transport path is not uniformly bad on SDMA. Instead, the shorter forward-only case is too small to amortize SDMA-side chunking overhead, while the longer forward+backward case has enough total work to expose a backend advantage.

---

## 3. Chunk Sweep Behavior

### 3.1 Absolute Latency Trend

For every phase, increasing `chunks` increases measured latency for **both** backends:

| Phase | RCCL `chunks=1` | RCCL `chunks=8` | SDMA `chunks=1` | SDMA `chunks=8` |
|---|---:|---:|---:|---:|
| `column_fwd` | 0.616 | 2.749 | 0.454 | 1.486 |
| `row_fwd` | 0.916 | 2.035 | 1.216 | 1.971 |
| `column_fwd_bwd` | 2.126 | 8.720 | 1.809 | 5.355 |
| `row_fwd_bwd` | 3.234 | 11.061 | 3.014 | 6.617 |

This is the clearest structural message in the whole log:

- finer chunking improves the **relative** position of SDMA in many cases,
- but finer chunking still hurts **absolute** latency.

So the current pipeline is paying more in chunk-management overhead than it gets back from additional overlap.

### 3.2 Relative Backend Trend

Even though absolute latency gets worse, SDMA's **relative** advantage usually gets larger:

- `column_fwd`: `1.36x -> 1.50x -> 1.67x -> 1.85x`
- `column_fwd_bwd`: `1.18x -> 1.34x -> 1.44x -> 1.63x`
- `row_fwd_bwd`: `1.07x -> 1.13x -> 1.21x -> 1.67x`

This means chunking is exposing a real backend difference:

- RCCL degrades faster than SDMA as the pipeline becomes more finely chunked
- but SDMA still does not escape the fixed-cost problem of the schedule itself

---

## 4. Why "Adding Communication" Can Be Worse Than Baseline

There are two different "baseline" comparisons in this log, and both need explanation.

### 4.1 Why Some SDMA rows are worse than the RCCL baseline

The obvious example is `row_fwd`:

- `chunks=1`: `0.75x`
- `chunks=2`: `0.78x`
- `chunks=4`: `0.78x`

This means SDMA is slower than RCCL in the row-forward-only phase for most chunk counts.

The most likely reason is that `row_fwd` is the least amortized case in the matrix:

1. It is a short forward-only path, so there is less total compute to hide transport setup.
2. It uses the row-parallel `GEMM + reduce_scatter` overlap path, where the communication chunks become small quickly as `num_chunks` increases.
3. Small chunked collectives are sensitive to fixed software cost:
   - chunk split / stitch overhead
   - staging buffer traffic
   - extra stream/event synchronization
   - per-chunk launch and scheduling overhead
4. In this short phase, those fixed costs can dominate the actual transport advantage.

That explains why SDMA loses in `row_fwd` but wins in `row_fwd_bwd`: once more work is present, the transport advantage finally has enough runtime to amortize its setup cost.

### 4.2 Why More Chunks are worse than the coarse-chunk baseline

If you compare `chunks=8` against `chunks=1`, all four phases get slower.

Example:

- `column_fwd` RCCL: `0.616 -> 2.749 ms`
- `column_fwd` SDMA: `0.454 -> 1.486 ms`
- `row_fwd_bwd` RCCL: `3.234 -> 11.061 ms`
- `row_fwd_bwd` SDMA: `3.014 -> 6.617 ms`

This is the classic "overlap did not amortize its own machinery" result.

Each additional chunk adds more fixed work:

- more chunk boundary bookkeeping
- more AG / RS launches
- more waits between comm and compute streams
- more staging-buffer use and reordering
- less payload per chunk, which lowers the useful work done by each collective

So even if chunking makes the backend comparison look better for SDMA, the schedule still gets more expensive in absolute terms.

### 4.3 Practical reading

This log does **not** say "communication is always bad".

It says:

- the current **chunked fused schedule** has significant fixed overhead
- that overhead is especially visible in short or row-forward-only cases
- SDMA often reduces the transport portion of the cost, but cannot erase the chunk-management cost by itself

---

## 5. Phase-by-Phase Interpretation

### 5.1 `column_fwd`

This is the cleanest SDMA win in the file.

- SDMA beats RCCL at every chunk count.
- The relative gain grows steadily with chunk count.
- But absolute latency still grows strongly with chunk count.

Interpretation:

- the column all-gather path benefits from SDMA transport,
- but the current chunked schedule still over-pays for fine granularity.

### 5.2 `row_fwd`

This is the outlier.

- SDMA loses at `chunks=1/2/4`
- only reaches a weak `1.03x` win at `chunks=8`
- `chunks=4` is also very noisy on the SDMA side (`CV=19.1%`)

Interpretation:

- forward-only row-parallel overlap is the least amortized path in the matrix
- the reduce-scatter-oriented chunked path appears more sensitive to fixed overhead than the column path

### 5.3 `column_fwd_bwd`

This is a stronger version of `column_fwd`.

- SDMA wins everywhere
- the gain scales from `1.18x` to `1.63x`
- RCCL instability becomes more visible at the small and largest chunk counts

Interpretation:

- once backward work is included, the transport difference is easier to see
- but chunk count still pushes both backends upward in absolute time

### 5.4 `row_fwd_bwd`

This is the most interesting reconciliation case.

- unlike `row_fwd`, SDMA wins at every chunk count
- the win grows from `1.07x` to `1.67x`

Interpretation:

- row-parallel SDMA is not fundamentally slower
- it just needs enough total work for its transport benefit to amortize setup and synchronization overhead

---

## 6. Stability Notes

Several points should be treated as directionally correct but not numerically exact:

- `RCCL column_fwd_bwd` at `chunks=1`
- `RCCL row_fwd_bwd` at `chunks=1`
- `RCCL column_fwd` at `chunks=2`
- `RCCL row_fwd_bwd` at `chunks=2`
- `SDMA row_fwd` at `chunks=4`
- both `column_fwd` rows at `chunks=8`
- `RCCL column_fwd_bwd` at `chunks=8`
- `RCCL row_fwd_bwd` at `chunks=8`

These do not invalidate the overall pattern, because the same qualitative story appears repeatedly:

- SDMA usually beats RCCL
- row-forward-only is the weak spot
- more chunks increase absolute latency

But they do mean the exact headline ratios at those points should not be over-interpreted.

---

## 7. Overall Interpretation

This cleaned 8-GPU fused pipeline log supports four conclusions.

### 7.1 Backend Choice Matters

SDMA is the better backend for most of the fused pipeline matrix:

- all `column_fwd` points
- all `column_fwd_bwd` points
- all `row_fwd_bwd` points
- and even `row_fwd` at the finest chunk count

So there is a genuine backend win here.

### 7.2 The Schedule Cost Still Dominates

The stronger result is not just "SDMA > RCCL". It is:

> **The chunked fused schedule gets slower as chunk count rises, for both backends.**

That means the benchmark is dominated by fixed pipeline overhead faster than it is helped by additional overlap.

### 7.3 Row Forward Is the Stress Case

`row_fwd` is the only phase where SDMA mostly loses. That makes it the best place to study why communication-enhanced paths can be worse than baseline:

- short runtime
- less work to amortize fixed cost
- more sensitivity to small reduce-scatter chunks

So if you want to reduce "optimized path slower than baseline" cases, `row_fwd` is the first place to investigate.

### 7.4 End-to-End Work Helps Amortization

The fact that `row_fwd_bwd` wins even though `row_fwd` loses is a useful sign:

- transport/backend improvement is real
- but it only becomes obvious when enough end-to-end work exists to hide setup cost

This strongly suggests the next optimization target is not the raw backend choice alone. It is the fixed cost of chunked scheduling, staging, and synchronization.

---

## Final Conclusion

The current 8-GPU `bench_fused_pipeline` backend-matrix summary says:

> **SDMA is usually the better fused backend than RCCL, but the current chunked fused schedule still becomes more expensive as chunk count increases, and short row-forward cases are still vulnerable to losing against the RCCL reference because fixed communication/synchronization overhead is not yet amortized.**

That narrows the next optimization question to:

1. keep treating **SDMA as the preferred fused backend** for most fused paths
2. focus on **reducing fixed per-chunk overhead** rather than only changing transport backend
3. use **`row_fwd`** as the primary diagnostic case for "optimized path slower than baseline"
4. treat the current chunk sweep as evidence that **more chunks are not automatically better** unless the overlap machinery itself becomes cheaper
