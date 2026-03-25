# E2E Transformer Layer Pure-Pipeline Benchmark Analysis — 8 GPU

**Source log**: `bench_e2e_fusion_8GPU.log`
**Benchmark**: `benchmarks/bench_e2e_fusion.py`
**Configuration**: 8 GPUs, BF16, `B=2`, `S=2048`, `H=4096`, `FFN=14336`, `num_chunks=4`
**Test suite**: 7 benchmark tests, all PASSED
**Runtime**: about 20 seconds total

This benchmark measures the **pure pipeline** path only:

- **Naive**: monolithic all-gather -> GEMM_up -> GEMM_down -> reduce-scatter
- **Fused NCCL**: chunked `PipelinedAllgatherGemm` / `PipelinedGemmReduceScatter` over NCCL
- **Fused SDMA**: the same chunked pipeline over mori SDMA

It intentionally does **not** include delayed wgrad scheduling. For delayed-wgrad overlap, the relevant benchmark is `bench_wgrad_delay.py`. For more isolated pipeline micro-benchmarks, use `bench_fused_pipeline.py`.

---

## Executive Summary

| Category | Key Finding |
|---|---|
| Functional status | **All 7 tests passed**; the earlier 8-GPU SDMA timeout no longer appears in this log |
| Fastest end-to-end path | **Naive** is still the fastest in every measured end-to-end case |
| Best fused backend | **SDMA beats NCCL** in single-layer `fwd` and `fwd+bwd` |
| Main performance result | Pure pipeline fusion is **not profitable** at these 8-GPU Llama-8B shapes |
| Backward-only signal | `fused NCCL bwd-only` is the weakest path; isolated `SDMA only bwd-only` is better, but still much slower than naive |
| TP scaling | Fused speedup degrades from **0.37x -> 0.26x -> 0.16x** as TP increases from 2 to 8 |

**Bottom line**: the stability issue appears resolved, but the performance story is unchanged. On this single-layer pure-pipeline benchmark, **SDMA is the better fused backend**, yet **the fused pipeline itself still loses to the naive baseline**. That means the dominant limitation is the chunked single-layer schedule, not just the transport choice.

---

## 1. Stability and Correctness

The most important functional result is that the full 8-GPU run now completes:

- `TestE2ETransformerLayerFusion`: `fwd+bwd`, `fwd-only`, and `bwd-only` all pass
- `TestE2ETransformerLayerSdmaOnly`: `fwd+bwd`, `fwd-only`, and `bwd-only` all pass
- `TestTPScaling::test_tp_scaling_sweep` also passes for `TP=2,4,8`

There are **no** `AllGather timeout waiting for peer ...` lines anywhere in this log. That matters because the earlier failing 8-GPU run was a transport-transition problem; this log shows that the benchmark is now stable enough to finish all timed sections and the TP scaling sweep.

The repeated `============================== 7 passed ... ==============================` footer appears once per torchrun rank, not because the suite ran 56 distinct tests. The actual suite still contains 7 benchmark tests.

---

## 2. Single-Layer End-to-End Results

### 2.1 Forward + Backward

| Variant | Avg (ms) | CV (%) | Relative to Naive |
|---|---:|---:|---:|
| naive fwd+bwd | 1.486 | 1.4 | 1.00x |
| fused NCCL fwd+bwd | 8.599 | 1.3 | **5.79x slower** |
| fused SDMA fwd+bwd | 5.381 | 0.3 | **3.62x slower** |

**Key comparisons**

- SDMA is **1.60x faster than NCCL** within the fused path (`8.599 / 5.381`)
- But even the better fused backend is still **far slower than naive**

**Interpretation**

This is the clearest result in the log: **backend choice matters, but it does not change the overall conclusion**. Switching from NCCL to SDMA reduces fused latency substantially, yet the fused pipeline still loses badly to the simple monolithic baseline.

That tells us the main bottleneck is not just collective implementation quality. The deeper issue is that, for this workload, the single-layer chunked pipeline adds more overhead than it hides.

---

### 2.2 Forward Only

| Variant | Avg (ms) | CV (%) | Relative to Naive |
|---|---:|---:|---:|
| naive fwd | 0.529 | 0.9 | 1.00x |
| fused NCCL fwd | 1.621 | 1.9 | **3.06x slower** |
| fused SDMA fwd | 1.248 | 1.1 | **2.36x slower** |

**Key comparisons**

- SDMA is **1.30x faster than NCCL** within the fused forward path
- The naive forward path is still the clear winner

**Interpretation**

The forward-only numbers confirm that the loss is not coming only from backward. Even before backward enters the picture, the 8-GPU pure pipeline path is already slower than the naive path.

This strongly suggests that the chunking/control overhead is too large for this shape:

- global sequence: `B * S = 4096`
- at `world=8`, each rank holds `S_local = 512`
- with `num_chunks=4`, each rank processes only `128` local rows per chunk

That means the pipeline is paying the fixed cost of:

- four all-gather launches instead of one
- four reduce-scatter launches instead of one
- stream/event coordination
- staging-buffer traffic
- chunk layout reorder

but each chunk is small enough that there is not much compute to hide behind the communication.

---

## 3. Backward-Only Behavior

### 3.1 Main Mixed Benchmark

| Variant | Avg (ms) | CV (%) | Relative to Naive |
|---|---:|---:|---:|
| naive bwd-only | 0.876 | 0.4 | 1.00x |
| fused NCCL bwd-only | 6.333 | 4.1 | **7.23x slower** |

The main `TestE2ETransformerLayerFusion` benchmark does **not** emit a `fused SDMA bwd-only` row. That is expected from the benchmark code: the mixed SDMA comparison is only run when `not bwd_only`.

The backward-only result is therefore especially important for NCCL:

- it is the slowest mixed-path result in the file
- it carries the highest variability (`CV=4.1%`, marked `~unstable`)

This indicates that backward is where the pure pipeline schedule is most fragile on 8 GPUs.

---

### 3.2 Isolated SDMA Backward

| Variant | Avg (ms) | CV (%) | Relative to Naive |
|---|---:|---:|---:|
| SDMA only bwd-only | 4.214 | 1.0 | **4.81x slower** |

Compared with `fused NCCL bwd-only`, the isolated SDMA backward path is:

- **1.50x faster than fused NCCL backward** (`6.333 / 4.214`)
- still **far slower than naive backward**

**Interpretation**

Again, SDMA improves the fused path, but does not rescue the benchmark. The transport backend is a secondary factor; the primary factor is the cost structure of the single-layer chunked backward path itself.

---

## 4. SDMA-Only Results

The standalone SDMA measurements are:

| Variant | Avg (ms) | CV (%) |
|---|---:|---:|
| SDMA only fwd+bwd | 5.902 | 1.2 |
| SDMA only fwd | 1.281 | 1.2 |
| SDMA only bwd-only | 4.214 | 1.0 |

These numbers are directionally consistent with the mixed benchmark:

- SDMA forward is around `1.25-1.28 ms`
- SDMA full forward+backward is around `5.4-5.9 ms`
- SDMA backward dominates the total cost

The absolute numbers differ slightly from the mixed `fused SDMA` rows, but the conclusion does not change:

1. the SDMA path is **stable**
2. the SDMA path is **faster than fused NCCL**
3. the SDMA path is **still slower than naive**

---

## 5. TP Scaling Sweep

### 5.1 Summary Table

| TP | Naive (ms) | Fused (ms) | Fused vs Naive | AG BW | RS BW |
|---|---:|---:|---:|---:|---:|
| 2 | 3.916 | 10.585 | **0.37x** | 3.2 GB/s | 3.2 GB/s |
| 4 | 1.970 | 7.444 | **0.26x** | 4.5 GB/s | 4.5 GB/s |
| 8 | 1.439 | 8.938 | **0.16x** | 3.8 GB/s | 3.8 GB/s |

### 5.2 What the Sweep Means

The naive path behaves as expected: as TP increases, each rank has less local work and the end-to-end time drops:

- `3.916 ms` at TP=2
- `1.970 ms` at TP=4
- `1.439 ms` at TP=8

The fused path behaves differently:

- it improves from TP=2 to TP=4 (`10.585 -> 7.444 ms`)
- then regresses again at TP=8 (`8.938 ms`)

The relative slowdown also gets steadily worse:

- fused is **2.70x slower** than naive at TP=2
- **3.78x slower** at TP=4
- **6.21x slower** at TP=8

**Interpretation**

As TP increases, the local compute per rank shrinks, but the benchmark keeps the same `num_chunks=4`. That makes the fixed per-chunk overhead increasingly expensive relative to useful compute.

For this benchmark:

- `TP=2` -> `S_local = 2048`
- `TP=4` -> `S_local = 1024`
- `TP=8` -> `S_local = 512`

At `TP=8`, each rank handles only `128` local rows per chunk. That is too little work to amortize the pipeline machinery, so the fused path loses the most precisely where TP is highest.

The bandwidth numbers reinforce that point. The best fused bandwidth appears at `TP=4` (`4.5 GB/s`), not at `TP=8`. So the highest TP setting does not turn the extra chunking into better communication efficiency; it mostly amplifies overhead.

---

## 6. Why Pure Pipeline Loses Here

This log is a good example of why **single-layer pure-pipeline** benchmarks can understate the value of overlap:

1. **The benchmark is intentionally single-layer**
   - there is no downstream layer to hide overhead behind
   - delayed wgrad is intentionally excluded

2. **The chunk size is small at 8 GPUs**
   - `S_local = 512`
   - `num_chunks = 4`
   - only `128` local rows per chunk

3. **The pipeline overhead is fixed**
   - extra chunk launches
   - extra synchronization
   - extra staging/copy/reorder work

4. **The benchmark asks one layer to pay all overhead immediately**
   - there is no multi-layer schedule to amortize setup cost
   - there is no delayed-wgrad schedule to recover backward time elsewhere

So the right conclusion is **not** that SDMA is bad. The right conclusion is:

- **SDMA is the better fused backend**
- **but this benchmark configuration does not provide enough useful overlap opportunity for the fused path to beat naive**

---

## 7. Practical Takeaways

### What this log proves

- The 8-GPU benchmark is now **functionally stable**
- SDMA is consistently **better than NCCL** inside the fused path
- The current single-layer pure-pipeline benchmark is **not performance-positive** on Llama-8B shapes at 8 GPUs

### What this log does not prove

- It does **not** prove that chunked overlap is useless in general
- It does **not** measure delayed-wgrad schedules
- It does **not** represent a multi-layer training pipeline where communication and compute from neighboring layers can overlap more effectively

### Recommended interpretation

Use this log as:

- a **correctness/stability checkpoint** for the pure pipeline implementation
- a **backend comparison** showing SDMA > NCCL for the fused path
- a warning that **single-layer pure-pipeline E2E latency is not enough, by itself, to justify the fused schedule**

For deeper performance questions:

- use `bench_fused_pipeline.py` to isolate chunked AG/RS pipeline behavior
- use `bench_wgrad_delay.py` to evaluate schedules where overlap has a real chance to pay off

---

## Final Conclusion

The 8-GPU `bench_e2e_fusion` run is now healthy and complete, but its performance message is clear:

> On this benchmark, **SDMA is the best fused backend, but the pure pipeline schedule itself is still slower than the naive baseline**.

That makes this log valuable for two reasons:

1. it confirms that the 8-GPU benchmark is stable again
2. it shows that the next optimization question is **not "NCCL or SDMA?"**, but rather **"what schedule gives the pipeline enough real work to hide its own overhead?"**
