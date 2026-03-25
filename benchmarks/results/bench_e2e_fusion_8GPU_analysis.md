# E2E Transformer Layer Pure-Pipeline Benchmark Analysis — 8 GPU

**Source log**: `bench_e2e_fusion_8GPU.log:1-150`
**Benchmark**: `benchmarks/bench_e2e_fusion.py`
**Scope**: performance summary only. This cleaned log contains timing tables, bandwidth, memory, and TP-scaling results, but not pytest/runtime metadata.

The benchmark covers:

- single-layer pure-pipeline `fwd+bwd`, `fwd`, and `bwd-only`
- two additional end-to-end profiles: `backend_gap` and `pipeline_gain`
- a six-profile shape sweep
- isolated `SDMA only` reference numbers
- a `TP=2/4/8` scaling sweep

---

## Executive Summary

| Category | Key Finding |
|---|---|
| Fastest path | `naive` is the fastest path in every measured end-to-end case |
| Best fused backend | `SDMA` consistently outperforms `NCCL` inside the fused path |
| Best fused result | The best SDMA case is `backend_gap` at **0.42x** speedup vs naive, which is still a loss |
| Shape sweep result | All six shape-sweep profiles are labeled **negative optimization** for both NCCL and SDMA |
| Memory result | There is **no memory-saving case** in this log; fused peak delta is always higher than naive |
| TP scaling result | Fused speedup degrades from **0.34x -> 0.22x -> 0.14x** as TP increases from 2 to 8 |

**Bottom line**: transport choice matters, because SDMA is clearly better than NCCL, but the main conclusion does not change. On this 8-GPU pure-pipeline benchmark, the fused schedule itself is still slower than the naive baseline, and the shape sweep does not show a compensating memory win.

---

## 1. Default Profile Results

Default profile:

- `B=2`
- `S=2048`
- `H=4096`
- `FFN=14336`
- `chunks=4`
- `world=8`

### 1.1 Main End-to-End Numbers

| Mode | Naive (ms) | Fused NCCL (ms) | Fused SDMA (ms) | Main Reading |
|---|---:|---:|---:|---|
| `fwd+bwd` | 1.772 | 11.895 | 6.284 | SDMA is better than NCCL, but both lose badly to naive |
| `fwd` | 0.772 | 2.808 | 2.002 | Same pattern: SDMA helps, but naive is still much faster |
| `bwd-only` | 1.088 | 4.769 | n/a | NCCL backward-only remains clearly slower than naive |

### 1.2 SDMA-Only Reference Numbers

| Mode | SDMA only (ms) | Relative to Naive |
|---|---:|---:|
| `fwd+bwd` | 5.409 | 3.05x slower than naive `fwd+bwd` |
| `fwd` | 1.621 | 2.10x slower than naive `fwd` |
| `bwd-only` | 3.538 | 3.25x slower than naive `bwd-only` |

### 1.3 What the Default Profile Says

- For `fwd+bwd`, SDMA is about **1.89x faster than NCCL** inside the fused path (`11.895 / 6.284`), but `fused SDMA` is still about **3.55x slower than naive**.
- For `fwd`, SDMA is about **1.40x faster than NCCL** (`2.808 / 2.002`), but `fused SDMA` is still about **2.59x slower than naive**.
- For `bwd-only`, the mixed benchmark only exposes the NCCL fused path, and that path is still about **4.38x slower than naive**.
- The isolated `SDMA only bwd-only` result (`3.538 ms`) is better than `fused NCCL bwd-only` (`4.769 ms`), but it still does not approach the naive baseline.

This means the default shape already shows the core pattern of the entire log:

1. `SDMA > NCCL` inside the fused implementation.
2. `naive > fused` even after switching to the better transport backend.

---

## 2. Shape Sweep Results

### 2.1 Performance Table

| Profile | Tokens | FFN | Naive (ms) | NCCL (ms) | NCCL Speedup | SDMA (ms) | SDMA Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `comm_bound_small` | 4096 | 8192 | 1.584 | 13.913 | 0.11x | 5.548 | 0.29x |
| `default` | 4096 | 14336 | 1.861 | 14.314 | 0.13x | 6.096 | 0.31x |
| `compute_bound_small` | 4096 | 28672 | 2.427 | 17.309 | 0.14x | 8.557 | 0.28x |
| `comm_bound_large` | 8192 | 8192 | 2.401 | 13.360 | 0.18x | 6.862 | 0.35x |
| `backend_gap` | 8192 | 14336 | 3.042 | 13.273 | 0.23x | 7.190 | 0.42x |
| `pipeline_gain` | 8192 | 28672 | 4.279 | 16.032 | 0.27x | 11.165 | 0.38x |

### 2.2 Main Sweep Takeaways

- All six profiles are classified as **negative optimization** for both NCCL and SDMA.
- The best SDMA case is `backend_gap` at **0.42x**, which still means the fused path is about **2.38x slower** than naive.
- The best NCCL case is `pipeline_gain` at **0.27x**, which still means the fused path is about **3.70x slower** than naive.
- The worst case is `comm_bound_small`, where NCCL drops to **0.11x** and SDMA to **0.29x**.

### 2.3 How Shape Changes Affect the Result

Two trends stand out:

1. **Larger token count helps relative fused efficiency, but not enough to win.**
   - With `Tokens=4096`, SDMA speedup stays in the `0.28x-0.31x` range.
   - With `Tokens=8192`, SDMA improves to `0.35x-0.42x`.
   - NCCL also improves when tokens increase, from `0.11x-0.14x` to `0.18x-0.27x`.

2. **SDMA remains better than NCCL across the whole sweep.**
   - At every profile, the SDMA latency is lower than the NCCL latency.
   - The transport backend matters, but it does not reverse the sign of the optimization.

So the sweep does show a real amortization effect: larger shapes make the fused pipeline less bad. But even the best shape in this log still loses to the naive baseline.

---

## 3. Bandwidth and Memory Behavior

### 3.1 Effective Communication Bandwidth

| Profile | NCCL AG/RS BW (GB/s) | SDMA AG/RS BW (GB/s) | Better Backend |
|---|---:|---:|---|
| `comm_bound_small` | 2.4 / 2.4 | 6.0 / 6.0 | SDMA |
| `default` | 2.3 / 2.3 | 5.5 / 5.5 | SDMA |
| `compute_bound_small` | 1.9 / 1.9 | 3.9 / 3.9 | SDMA |
| `comm_bound_large` | 5.0 / 5.0 | 9.8 / 9.8 | SDMA |
| `backend_gap` | 5.1 / 5.1 | 9.3 / 9.3 | SDMA |
| `pipeline_gain` | 4.2 / 4.2 | 6.0 / 6.0 | SDMA |

The communication story is straightforward:

- SDMA delivers **higher effective AG/RS bandwidth than NCCL in every profile**.
- Larger-token profiles generally improve bandwidth for both backends.
- However, higher bandwidth alone does **not** produce an end-to-end latency win.

That means communication transport is only part of the cost. The remaining loss must come from the structure of the pure-pipeline schedule itself: chunking, staging, synchronization, and other fixed overheads that are not eliminated by faster transport.

### 3.2 Peak Working-Set Memory

| Profile | Naive Peak Delta (MiB) | NCCL Peak Delta (MiB) | SDMA Peak Delta (MiB) |
|---|---:|---:|---:|
| `comm_bound_small` | 100 | 120 | 120 |
| `default` | 94 | 132 | 132 |
| `compute_bound_small` | 100 | 184 | 184 |
| `comm_bound_large` | 216 | 240 | 240 |
| `backend_gap` | 216 | 264 | 264 |
| `pipeline_gain` | 228 | 340 | 340 |

This table gives a very important negative result:

- There is **no "memory win only" case** in this log.
- The fused path always uses **more** peak working-set memory than the naive path.
- NCCL and SDMA show the **same peak delta** for every profile.

That last point is especially useful. It suggests the extra peak memory is not caused by NCCL vs SDMA backend choice. It is coming from the shared fused-pipeline structure itself, such as chunk staging and additional temporary buffers.

The largest penalty appears at `pipeline_gain`:

- naive peak delta: `228 MiB`
- fused peak delta: `340 MiB`
- extra peak working set: **+112 MiB**

So this sweep does **not** support the claim that the current pure-pipeline implementation can trade latency loss for memory savings on these shapes.

---

## 4. TP Scaling Sweep

### 4.1 Summary

| TP | Naive (ms) | Fused (ms) | Speedup | Fused Slower Than Naive | AG BW | RS BW |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 4.972 | 14.809 | 0.34x | 2.98x slower | 2.3 GB/s | 2.3 GB/s |
| 4 | 2.768 | 12.482 | 0.22x | 4.51x slower | 2.7 GB/s | 2.7 GB/s |
| 8 | 1.788 | 12.486 | 0.14x | 6.98x slower | 2.7 GB/s | 2.7 GB/s |

### 4.2 Interpretation

The TP sweep shows the clearest structural problem in the benchmark:

- The naive path improves strongly with higher TP: `4.972 -> 2.768 -> 1.788 ms`.
- The fused path improves only once, then becomes almost flat: `14.809 -> 12.482 -> 12.486 ms`.
- Relative fused efficiency gets steadily worse: `0.34x -> 0.22x -> 0.14x`.

So as TP increases, local useful work shrinks faster than the fused overhead shrinks. In other words, the benchmark is increasingly dominated by fixed pipeline cost rather than by communication bandwidth.

The bandwidth numbers support the same reading:

- AG/RS bandwidth rises only from `2.3` to `2.7 GB/s` and then stops improving.
- End-to-end fused latency does not meaningfully benefit from the highest TP setting.

This means higher TP is not turning the pure pipeline into a better overlap schedule for this workload. It is mainly exposing how expensive the fixed chunked schedule becomes when per-rank work gets smaller.

---

## 5. Overall Interpretation

This cleaned 8-GPU log supports four conclusions.

### 5.1 Transport Choice Matters

SDMA is consistently better than NCCL:

- lower latency in the default single-layer benchmarks
- better latency across the full shape sweep
- higher effective AG/RS bandwidth in every profile

So there is a genuine backend advantage here.

### 5.2 The Schedule Still Loses

Even after switching to SDMA:

- the default `fwd+bwd` path is still much slower than naive
- all six shape-sweep cases remain negative optimization
- TP scaling gets worse rather than better

So the dominant limitation is not just "NCCL is slow". The dominant limitation is that this single-layer pure-pipeline schedule still carries too much fixed overhead for these shapes.

### 5.3 There Is No Memory Compensation in This Log

The shape sweep was designed to reveal either:

- a latency win, or
- a memory-only advantage when latency does not improve

This log shows neither:

- no latency win
- no memory-only win

The fused pipeline increases peak working-set memory for every measured profile.

### 5.4 Larger Shapes Help, But Do Not Flip the Result

The best relative numbers appear at the larger-token profiles, especially `backend_gap`, which indicates that more work can amortize part of the pipeline cost. But the improvement is only from "very bad" to "still clearly negative". The log does not contain a crossover point where the fused path becomes faster than naive.

---

## 6. Cross-Benchmark Reconciliation With `bench_comm_overlap`

The statement "SDMA is consistently better than NCCL" in this document is
**scope-limited**. It refers to the current **fused pure-pipeline backend**
inside `bench_e2e_fusion`, not to every SDMA collective in isolation.

That distinction matters because `bench_comm_overlap` measures individual
collective paths and overlap patterns, while this benchmark measures the total
single-layer schedule.

| Benchmark | Unit under test | What "SDMA better/worse" means |
|-----------|-----------------|--------------------------------|
| `bench_comm_overlap` | raw AG/RS path or overlap micro-benchmark | the collective path itself is faster/slower |
| `bench_e2e_fusion` | full chunked layer schedule | the backend gives lower **end-to-end fused layer latency** |

This is why the two files can both be correct:

1. `bench_comm_overlap` shows that the **raw 8-GPU RS path** is faster with
   NCCL than with SDMA.
2. This file shows that the **fused pure-pipeline backend** is faster with
   SDMA than with NCCL.
3. Both files still show that the **naive layer path** is faster than either
   fused backend.

So the safe ranking is:

- **raw RS primitive at 8 GPUs**: `NCCL > SDMA`
- **fused single-layer pure-pipeline backend**: `SDMA > NCCL`
- **against the naive E2E baseline**: `naive > SDMA > NCCL`

One more reporting detail is easy to miss: this document's AG/RS bandwidth
columns are derived from the **fused layer latency**, not from standalone AG/RS
timings. They should be read as schedule-level effective bandwidth, not raw
collective bandwidth.

That is why these tables should be interpreted as:

- transport/backend behavior **inside the current fused layer schedule**
- not proof that every SDMA collective is faster than its NCCL counterpart

---

## Final Conclusion

The current 8-GPU `bench_e2e_fusion` performance summary says:

> **SDMA is the better fused backend, but the fused pure-pipeline schedule is still slower than the naive baseline across all measured shapes, and it also increases peak working-set memory.**

That makes the next optimization question much narrower:

1. keep treating **SDMA as the preferred fused backend**
2. focus future work on **reducing or amortizing the fixed cost of the pure-pipeline schedule itself**
3. do not assume the current fused path provides a hidden memory advantage, because this log does not show one
