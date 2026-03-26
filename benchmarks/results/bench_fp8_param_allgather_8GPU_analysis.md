# FP8 Param All-Gather Benchmark Analysis — 8 GPU Chunk Sweep

**Source logs**: `bench_fp8_param_allgather_chunk1.log`, `bench_fp8_param_allgather_chunk2.log`, `bench_fp8_param_allgather_chunk4.log`
**Benchmark**: `benchmarks/bench_fp8_param_allgather.py`
**Scope**: performance summary only. These cleaned logs compare `num_chunks=1`, `2`, and `4` for the 8-GPU FP8 param all-gather path, including raw all-gather latency, single-layer gather+forward, 4-layer pipelined overlap, tail latency, and scaling efficiency.

Rows marked `unstable` in the source logs are treated as directional only. The main conclusions below rely on patterns that repeat across multiple sections, not on any single unstable cell.

The benchmark covers:

- raw FP8 vs BF16 all-gather latency for two representative shapes
- full single-layer `quant -> gather -> dequant -> GEMM` pipeline timing
- three single-layer end-to-end profiles: `default`, `backend_gap`, and `pipeline_gain`
- four-layer pipelined gather+forward timing
- six-profile shape sweeps for both single-layer and pipelined paths
- tail latency and `TP=2/4/8` scaling sweeps

---

## Executive Summary

| Category | Key Finding |
|---|---|
| Fastest raw collective | `chunks=2` is the fastest isolated FP8 all-gather configuration in both raw-latency shapes |
| Fastest end-to-end chunk | `chunks=1` is the fastest FP8 choice in every end-to-end table in these logs |
| Single-layer overlap result | Overlap is a loss in the single-layer full-pipeline microbench for every measured shape and chunk count |
| Multi-layer overlap result | Four-layer pipelining usually recovers only a small `1.01x-1.08x` gain vs FP8 sequential, but it can also regress (`0.98x` in `backend_gap`, `chunks=4`) |
| Shape sweep result | All six shape-sweep profiles remain negative optimization vs BF16 for `chunks=1`, `2`, and `4` |
| Scaling result | `chunks=4` is the worst scaling point, especially for pipelined FP8 at 8 GPUs (`14.141 ms`, `68.1%` efficiency) |

**Bottom line**: FP8's bandwidth reduction is real in the raw all-gather section, and `chunks=2` is the best isolated communication setting. But once the benchmark includes dequantization, chunk staging, synchronization, and GEMM, extra chunking does not pay off. For this workload, the best practical FP8 choice is `chunks=1`, and `chunks=4` is consistently the worst end-to-end option.

---

## 1. Raw All-Gather Results

### 1.1 Latency Table

| Shape | BF16 ms | FP8 ms (`c=1`) | FP8 ms (`c=2`) | FP8 ms (`c=4`) | Best Chunk | Main Reading |
|---|---:|---:|---:|---:|---:|---|
| `4096, 4096` | `0.172-0.179` | `0.197` | `0.124` | `0.129` | `2` | Small-shape FP8 flips from a slight loss at `c=1` to a clear win once chunking is introduced, but this row is marked unstable |
| `14336, 4096` | `0.404-0.408` | `0.241` | `0.238` | `0.246` | `2` | All FP8 variants beat BF16 here; `c=2` is the best measured point, with `c=4` slightly behind |

### 1.2 What The Raw Section Says

- This is the only section where higher chunk count helps cleanly.
- `chunks=2` is the fastest FP8 raw all-gather configuration in both measured shapes.
- The benefit is modest between `c=2` and `c=4`, so the raw collective is not strongly sensitive past two chunks.
- These numbers isolate communication. They do **not** include the dequantization or GEMM work that dominates the later sections.

So the raw communication story is positive for FP8, and mildly positive for chunking. The rest of the benchmark shows that this raw win does not survive the full schedule.

---

## 2. Single-Layer Results

### 2.1 Full Pipeline Microbench

The single-layer full-pipeline microbench shows the clearest disconnect between raw communication wins and end-to-end behavior.

For the larger shape `14336 x 4096`:

- BF16 latency rises from `1.396 -> 1.512 -> 1.680 ms` as chunks increase from `1 -> 2 -> 4`
- FP8 sequential rises from `1.811 -> 2.093 -> 2.580 ms`
- FP8 overlapped rises from `3.181 -> 3.434 -> 4.034 ms`
- The overlapped single-layer path is always worse than FP8 sequential, with `ovl/seq = 0.57x`, `0.61x`, and `0.64x`

For the smaller shape `4096 x 4096`, the same trend appears:

- BF16: `0.494 -> 0.576 -> 0.691 ms`
- FP8 sequential: `0.783 -> 1.068 -> 1.681 ms`
- FP8 overlapped: `2.191 -> 2.509 -> 3.314 ms`

This means the raw FP8 all-gather win is already gone by the time the benchmark measures one complete `gather -> dequant -> linear` step. More chunking only increases the fixed schedule cost.

### 2.2 End-to-End Gather+Forward — Single Layer

| Profile | BF16 ms (`c=1`) | FP8 ms (`c=1`) | FP8 sp (`c=1`) | BF16 ms (`c=2`) | FP8 ms (`c=2`) | FP8 sp (`c=2`) | BF16 ms (`c=4`) | FP8 ms (`c=4`) | FP8 sp (`c=4`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `default` | `1.395` | `1.828` | `0.76x` | `1.509` | `2.108` | `0.72x` | `1.688` | `2.652` | `0.64x` |
| `backend_gap` | `2.120` | `2.741` | `0.77x` | `2.288` | `2.979` | `0.77x` | `2.488` | `3.410` | `0.73x` |
| `pipeline_gain` | `4.229` | `5.071` | `0.83x` | `4.366` | `5.344` | `0.82x` | `4.536` | `6.049` | `0.75x` |

### 2.3 Main Single-Layer Takeaways

- `chunks=1` is the fastest FP8 choice in all three single-layer E2E profiles.
- `chunks=2` never wins a single E2E profile, even though it is the best raw all-gather point.
- `chunks=4` is always the slowest option.
- In the three-profile single-layer E2E table above, the best FP8 case is `pipeline_gain` at `chunks=1`, but it still reaches only `0.83x` vs BF16.
- Some of the individual rows are marked `unstable` in the source logs, but the chunk ordering itself stays consistent across all three profiles.

So the single-layer part of the benchmark already gives a strong practical rule: if the goal is end-to-end latency rather than raw collective speed, `chunks=1` is the only reasonable choice among these three settings.

---

## 3. Four-Layer Pipeline And Shape Sweep

### 3.1 End-to-End Pipelined Gather+Forward — 4 Layers

| Profile | FP8 seq ms (`c=1`) | FP8 pipe ms (`c=1`) | pipe/seq (`c=1`) | FP8 seq ms (`c=2`) | FP8 pipe ms (`c=2`) | pipe/seq (`c=2`) | FP8 seq ms (`c=4`) | FP8 pipe ms (`c=4`) | pipe/seq (`c=4`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `default` | `24.447` | `23.692` | `1.03x` | `29.221` | `28.708` | `1.02x` | `34.414` | `33.929` | `1.01x` |
| `backend_gap` | `36.122` | `34.428` | `1.05x` | `41.268` | `38.221` | `1.08x` | `47.447` | `48.431` | `0.98x` |
| `pipeline_gain` | `66.678` | `62.589` | `1.07x` | `69.406` | `65.396` | `1.06x` | `79.198` | `73.109` | `1.08x` |

The four-layer pipeline does show real overlap, but the gain is limited:

- The improvement over FP8 sequential is only a few milliseconds in absolute terms
- The relative gain stays small, usually around `1.01x-1.08x`
- `chunks=4` does **not** create a stronger overlap regime; it only starts from a much slower sequential baseline
- `backend_gap` at `chunks=4` is the clearest warning sign, because the pipelined path is actually slower than FP8 sequential (`0.98x`)

Most importantly, BF16 still wins every one of these cases:

- `default`: BF16 vs FP8-pipe = `18.572 / 23.692 = 0.78x`, `20.016 / 28.708 = 0.70x`, `22.211 / 33.929 = 0.65x`
- `backend_gap`: `0.82x`, `0.82x`, `0.73x`
- `pipeline_gain`: `0.87x`, `0.86x`, `0.83x`

So multi-layer pipelining helps FP8 a little, but it does not change the winner, and it does not justify higher chunk counts.

### 3.2 Shape Sweep Results

The six-profile sweep repeats the same ranking.

The `chunks=4` source log also includes an explicit note that its sweep sections currently report fixed profile `chunks=4`. The tables below use the printed `Chunks` values as reported in the logs.

#### Single-layer sweep

- All six profiles remain below `1.0x` vs BF16 for `chunks=1`, `2`, and `4`
- `chunks=1` is the fastest FP8 point in all six profiles
- Average FP8/BF16 speedup across the six profiles is roughly:
  - `0.78x` at `chunks=1`
  - `0.77x` at `chunks=2`
  - `0.70x` at `chunks=4`

Representative examples:

| Profile | FP8 ms (`c=1`) | FP8 sp (`c=1`) | FP8 ms (`c=2`) | FP8 sp (`c=2`) | FP8 ms (`c=4`) | FP8 sp (`c=4`) |
|---|---:|---:|---:|---:|---:|---:|
| `default` | `1.784` | `0.80x` | `2.097` | `0.79x` | `2.516` | `0.65x` |
| `backend_gap` | `2.988` | `0.71x` | `3.014` | `0.77x` | `3.442` | `0.73x` |
| `pipeline_gain` | `4.989` | `0.85x` | `5.393` | `0.82x` | `6.039` | `0.75x` |

#### Pipelined sweep

- Pipelining helps FP8 sequential within a fixed chunk count, but not enough to beat BF16
- `chunks=1` is again the fastest FP8-pipe point in all six profiles
- Average BF16/FP8-pipe ratio across the six profiles is roughly:
  - `0.82x` at `chunks=1`
  - `0.78x` at `chunks=2`
  - `0.73x` at `chunks=4`

Representative examples:

| Profile | FP8 pipe ms (`c=1`) | pipe/BF16 (`c=1`) | FP8 pipe ms (`c=2`) | pipe/BF16 (`c=2`) | FP8 pipe ms (`c=4`) | pipe/BF16 (`c=4`) |
|---|---:|---:|---:|---:|---:|---:|
| `default` | `22.193` | `0.84x` | `28.861` | `0.70x` | `32.155` | `0.69x` |
| `backend_gap` | `34.895` | `0.81x` | `38.877` | `0.81x` | `46.796` | `0.76x` |
| `pipeline_gain` | `63.282` | `0.86x` | `64.823` | `0.87x` | `73.165` | `0.83x` |

### 3.3 What The Sweep Says

The shape sweep is important because it rules out the idea that more chunks might only help on certain shapes.

That does not happen here:

1. `chunks=1` is the best FP8 setting in every sweep row.
2. `chunks=2` is never meaningfully better than `chunks=1` in end-to-end latency.
3. `chunks=4` is uniformly worse and often much worse.

So the chunk-count conclusion is not specific to one profile. It holds across both smaller and larger token/FFN combinations in the reused six-profile sweep.

---

## 4. Tail Latency And Scaling

### 4.1 Tail Latency

The tail-latency section splits into two very different stories.

For isolated raw all-gather cases:

- FP8 average and p95 are better than BF16 in the visible `4096x4096` and `14336x4096` tables for all chunked runs
- `chunks=2` is generally the cleanest raw tail result, especially for the larger shape where it has the lowest visible FP8 average (`0.226 ms`) and max (`0.271 ms`), but the max column varies sharply across runs and should be read cautiously
- The small-shape case remains marked unstable, so it should not be overinterpreted

For the full pipeline:

- `chunks=1`: BF16 p95 `0.397 ms`, FP8 p95 `2.622 ms`
- `chunks=2`: BF16 p95 `0.405 ms`, FP8 p95 `2.522 ms`
- `chunks=4`: BF16 p95 `2.444 ms`, FP8 p95 `3.811 ms`

This means:

- FP8 raw collective tail can look good
- FP8 full-pipeline tail still looks bad
- `chunks=4` does not represent a real tail win for FP8; it mostly reflects BF16 tail becoming worse too

So the tail section reinforces the same message as the average-latency sections: the communication payload reduction is real, but the full schedule remains the problem.

### 4.2 Scaling Efficiency

| Chunks | FP8 sp @ 2 GPUs | FP8 sp @ 4 GPUs | FP8 sp @ 8 GPUs | Pipe ms @ 8 GPUs | Pipe eff @ 8 GPUs | Main Reading |
|---|---:|---:|---:|---:|---:|---|
| `1` | `1.11x` | `0.90x` | `0.76x` | `5.757` | `111.3%` | Best overall chunk; FP8 sequential wins only at 2 GPUs, but the pipelined path stays relatively flat |
| `2` | `1.07x` | `0.88x` | `0.71x` | `6.998` | `95.1%` | Same trend, slightly worse than `c=1` |
| `4` | `1.05x` | `0.84x` | `0.64x` | `14.141` | `68.1%` | Clear regression; 8-GPU pipelined latency jumps sharply |

### 4.3 Interpretation

The scaling section exposes the structural cost of extra chunking most clearly:

- FP8 sequential is only competitive at `2 GPUs`
- At `4` and `8 GPUs`, FP8 sequential loses to BF16 for every chunk count
- Higher chunk count steadily weakens FP8 sequential scaling
- `chunks=4` is especially problematic because the pipelined path nearly doubles from `7.833 ms` at `4 GPUs` to `14.141 ms` at `8 GPUs`

That last point is the strongest anti-`chunks=4` signal in the entire log set. Even if larger chunk counts were meant to create more overlap opportunities, this benchmark shows the opposite outcome at scale.

---

## 5. Reconciling The Sections

These logs can look contradictory at first glance:

1. The raw all-gather section says FP8 is faster than BF16, and `chunks=2` is best.
2. The end-to-end sections say FP8 is slower than BF16, and `chunks=1` is best.

Both statements are correct, because they measure different things.

| Section | Unit under test | What the result means |
|---|---|---|
| Raw all-gather | communication only | FP8 payload reduction helps, and a little chunking improves collective latency |
| Single-layer / 4-layer E2E | full `gather -> dequant -> compute` schedule | dequantization, chunk management, synchronization, and pipeline overhead dominate |

So the safe ranking is:

- **raw collective only**: `FP8 c=2` is best, while `c=1` and `c=4` trade places depending on shape
- **end-to-end FP8 latency**: `FP8 c=1 > FP8 c=2 > FP8 c=4`
- **against BF16 end-to-end**: `BF16 > FP8 c=1 > FP8 c=2 > FP8 c=4`

This is the main result of the chunk sweep.

---

## 6. Overall Interpretation

This 8-GPU chunk sweep supports four conclusions.

### 6.1 FP8 Communication Savings Are Real

The raw all-gather section is not a false signal:

- FP8 really does reduce the communication payload
- the raw latency tables do show real benefit
- `chunks=2` is the best measured isolated communication point

So the benchmark does confirm the intended low-level effect.

### 6.2 End-to-End Overhead Dominates

As soon as the benchmark includes dequantization and GEMM:

- single-layer FP8 is slower than BF16
- four-layer pipelining remains slower than BF16
- every shape-sweep row remains a negative optimization

So the main bottleneck is not raw all-gather bandwidth. The main bottleneck is the cost of integrating FP8 gather into the full layer schedule.

### 6.3 More Chunks Mostly Add Cost

The chunk trend is consistent:

- `chunks=2` helps the raw collective
- `chunks=2` hurts or fails to help every end-to-end table
- `chunks=4` is the slowest FP8 choice everywhere that matters

That means extra chunking is adding more scheduling overhead than overlap benefit for this workload.

### 6.4 The Practical Default Should Stay At `chunks=1`

Among the tested values, `chunks=1` is the only chunk count that is consistently least-bad for real end-to-end latency:

- fastest FP8 single-layer E2E profile in all three profiles
- fastest FP8 four-layer pipelined result in all three profiles
- fastest FP8 point in all six single-layer sweep rows
- fastest FP8-pipe point in all six pipelined sweep rows
- best scaling behavior among the three chunk counts

So the sweep does not support moving the default from `1` to `2` or `4`.

---

## Final Conclusion

The current 8-GPU `bench_fp8_param_allgather` chunk comparison says:

> **`chunks=2` is the best raw FP8 all-gather point, but `chunks=1` is the best practical FP8 configuration for every end-to-end path in these logs, while `chunks=4` is consistently the worst.**

That makes the next optimization question much narrower:

1. keep `chunks=1` as the default for end-to-end FP8 param gather experiments
2. treat `chunks=2` as a raw-collective-only tuning point, not as proof of end-to-end benefit
3. focus future work on reducing dequantization and schedule overhead rather than increasing chunk count
4. avoid using `chunks=4` as a recommended default on 8 GPUs until the scaling regression is understood
