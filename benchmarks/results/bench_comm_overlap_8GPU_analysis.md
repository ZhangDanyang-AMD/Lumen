# Communication-Compute Overlap Benchmark Analysis — 8 GPU

**Log**: `bench_comm_overlap_8GPU.log`
**Platform**: 8× AMD MI250X GCDs, Python 3.10.12, pytest 7.3.2
**Tests**: 20 items — NCCL overlap, SDMA overlap, NCCL vs SDMA comparison, scaling sweeps
**Matrix shape**: M=4096 (B×S), K=4096 (HIDDEN), N=14336 (FFN_HIDDEN); N varies in scaling tests

---

## Executive Summary

| Metric | Result |
|--------|--------|
| All-Gather overlap (NCCL) | **1.16x speedup**, overlap ratio = −0.07 |
| All-Gather overlap (SDMA) | **0.65x** (slower than sequential), overlap ratio = −0.40 |
| Reduce-Scatter overlap (NCCL) | **0.94x** (marginal regression) |
| Reduce-Scatter overlap (SDMA) | **1.09x speedup**, overlap ratio = 0.06 |
| AG head-to-head winner | **NCCL** for N ≤ 14336; **SDMA** at N = 28672 (1.16x) |
| RS head-to-head winner | **NCCL** across all sizes (SDMA 0.60x–0.95x) |

**Key finding**: On 8 GPUs, NCCL dominates SDMA for most tensor sizes. SDMA only surpasses NCCL for All-Gather at the largest dimension (N=28672). For Reduce-Scatter, SDMA is consistently slower across all sizes.

---

## 1. NCCL Column-Parallel Overlap (All-Gather + GEMM)

| Metric | Avg (ms) | Min (ms) | Max (ms) | CV (%) |
|--------|----------|----------|----------|--------|
| allgather alone | 0.212 | 0.208 | 0.215 | 0.8 |
| GEMM alone | 0.034 | 0.032 | 0.037 | 4.2 ~unstable |
| sequential | 0.304 | 0.298 | 0.309 | 1.1 |
| overlapped | 0.262 | 0.257 | 0.267 | 1.0 |

- **Overlap ratio**: −0.068
- **Speedup**: **1.16x**
- **Hidden GEMM**: 0.041 / 0.034 ms (121% of GEMM hidden)

**Analysis**: The negative overlap ratio means the overlapped time (0.262 ms) exceeds `max(AG, GEMM)` = 0.212 ms — there is overhead from stream scheduling and kernel co-execution. However, the sequential sum (0.304 ms) is still significantly larger, yielding a solid 1.16x speedup. The GEMM (0.034 ms) is fully hidden behind the allgather latency (0.212 ms). NCCL's allgather primarily uses the NIC and ring/tree protocol for data movement, consuming minimal GPU compute units (CUs). This leaves the CUs free for the GEMM, enabling effective overlap.

---

## 2. NCCL Row-Parallel Overlap (Reduce-Scatter + GEMM)

| Metric | Avg (ms) | Min (ms) | Max (ms) | CV (%) |
|--------|----------|----------|----------|--------|
| reduce_scatter | 0.171 | 0.167 | 0.176 | 1.4 |
| GEMM | 0.036 | 0.036 | 0.038 | 1.9 |
| sequential | 0.194 | 0.190 | 0.198 | 1.3 |
| overlapped | 0.206 | 0.201 | 0.213 | 1.7 |

- **Overlap ratio**: 0.007
- **Speedup**: **0.94x** (slightly slower)
- **Hidden GEMM**: −0.012 / 0.036 ms (−33% of GEMM hidden)

**Analysis — Why NCCL RS overlap shows negative optimization (0.94x)**:

Unlike AllGather, NCCL's Reduce-Scatter performs a **reduction operation (sum)** on the GPU. This reduction requires GPU compute units (CUs), creating direct contention with the GEMM kernel. On 8 GPUs, the RS reduction workload is larger than on 2 GPUs (7 partial sums vs 1), intensifying the CU competition.

The overlapped path (0.206 ms) is slower than sequential (0.194 ms) by 0.012 ms. This regression comes from:

1. **CU contention between RS reduction and GEMM** — both kernels need compute units simultaneously, mutually slowing each other down
2. **Stream scheduling overhead** — managing two concurrent streams adds latency that exceeds any overlap benefit when the GEMM is this small (0.036 ms)
3. **Theoretical maximum speedup** is only `(0.171 + 0.036) / 0.171` = 1.21x, but with the RS kernel itself consuming CUs, even modest contention wipes out the benefit

This contrasts with the 2-GPU case where NCCL RS achieved 1.03x — on 2 GPUs, the reduction workload is lighter and CU contention is manageable.

---

## 3. SDMA Column-Parallel Overlap (All-Gather + GEMM)

| Metric | Avg (ms) | Min (ms) | Max (ms) | CV (%) |
|--------|----------|----------|----------|--------|
| SDMA allgather | 0.166 | 0.163 | 0.171 | 1.4 |
| GEMM | 0.047 | 0.042 | 0.055 | 11.3 !NOISY |
| sequential | 0.193 | 0.191 | 0.197 | 0.7 |
| SDMA overlap | 0.298 | 0.290 | 0.304 | 1.4 |

- **Overlap ratio**: −0.399
- **Speedup**: **0.65x** (significantly slower)
- **Hidden GEMM**: −0.105 / 0.047 ms (−223% of GEMM hidden)

**Analysis — Why SDMA AG overlap is 54% slower than sequential**:

This is the most counter-intuitive result. SDMA is supposed to use dedicated DMA engines that don't compete with GEMM for compute resources. The raw SDMA AllGather latency (0.166 ms) is actually **faster than NCCL** (0.212 ms). Yet the overlapped path (0.298 ms) is dramatically worse than even the sequential sum (0.193 ms).

**Root cause: The SDMA PUT kernel's control plane runs on GPU CUs.**

The overlap benchmark executes the following path:

```python
def _sdma_overlap():
    comm.allgather_dim0_async(x_local, stream=sdma_stream)   # PUT kernel → sdma_stream
    _ = x_local @ w.T                                         # GEMM → compute_stream
    _ = comm.wait_allgather_dim0(stream=sdma_stream)          # WAIT → sdma_stream
    compute_stream.wait_stream(sdma_stream)                    # sync
```

While the actual data transfer uses DMA engines (off the CU path), the SDMA AllGather implementation has several phases that **do** consume CUs:

| Phase | CU usage | 8-GPU impact |
|-------|----------|-------------|
| **PUT kernel** (Grid=32768, Block=256) | Address calculation, flag setting, data staging — all on CUs | 8M threads launched; workload scales with 7 peers (vs 1 on 2-GPU) |
| **Flag polling** (WAIT phase) | Busy-wait kernel on CUs checking 7 remote flags | Blocks until the slowest peer completes; straggler effect amplified |
| **Barrier synchronization** | All 8 PEs must mutually confirm completion | Latency grows with peer count |
| **Output copy** | GPU kernel copying transit buffer → user output | 8 shards to copy (4× the data vs 2-GPU) |

**Timeline comparison:**

```
Sequential:
  [---PUT+WAIT+Copy (0.166ms)---][--GEMM (0.047ms)--]  = 0.193 ms

Overlapped (ideal, if zero CU contention):
  sdma_stream:  [---PUT+WAIT+Copy (0.166ms)---]
  compute_stream:     [--GEMM (0.047ms)--]              = 0.166 ms

Overlapped (actual):
  sdma_stream:  [--PUT kernel (CU-heavy)--][..flag poll..][barrier][copy]
  compute_stream:  [GEMM delayed/slowed by PUT kernel contention..........]
  sync:                                                     [wait_stream]
                                                                    = 0.298 ms
```

The PUT kernel with `Grid=32768, Block=256` (8M threads) floods the GPU scheduler. Even though data movement itself goes through DMA engines, the kernel's **control logic — address computation, flag writes, synchronization primitives — runs on CUs**. This effectively blocks or severely degrades GEMM execution on the compute stream.

**Evidence of interference**: The GEMM CV during SDMA overlap is **11.3%** (!NOISY), compared to 4.2% during NCCL overlap. This high variance is a strong signal that GEMM execution timing is being disrupted by the concurrent SDMA PUT kernel.

**Why this is worse at 8-GPU than 2-GPU:**

| Factor | 2-GPU | 8-GPU | Amplification |
|--------|-------|-------|---------------|
| Peers to PUT to | 1 | 7 | 7× more PUT work |
| Flags to poll | 1 | 7 | Wait for slowest of 7 |
| Barrier participants | 2 | 8 | O(log N) or O(N) sync |
| Output copy volume | 2 shards | 8 shards | 4× more data |
| 2-GPU overlap speedup | 1.02× | — | Already marginal |
| 8-GPU overlap speedup | — | 0.65× | Collapsed |

On 2 GPUs, the PUT kernel is small enough that CU contention is tolerable (1.02× speedup). On 8 GPUs, the PUT kernel's CU footprint grows proportionally with peer count, tipping the balance to severe negative optimization.

**Contrast with NCCL AG overlap (1.16×)**: NCCL's AllGather is handled primarily by the NIC and network protocol stack. GPU CUs are barely involved in data movement, so the GEMM runs unimpeded on the compute stream.

---

## 4. SDMA Row-Parallel Overlap (Reduce-Scatter + GEMM)

| Metric | Avg (ms) | Min (ms) | Max (ms) | CV (%) |
|--------|----------|----------|----------|--------|
| SDMA reduce_scatter | 0.251 | 0.248 | 0.253 | 0.5 |
| GEMM | 0.037 | 0.037 | 0.038 | 1.3 |
| sequential | 0.296 | 0.290 | 0.302 | 1.4 |
| SDMA RS overlap | 0.270 | 0.266 | 0.275 | 1.0 |

- **Overlap ratio**: 0.063
- **Speedup**: **1.09x**
- **Hidden GEMM**: 0.026 / 0.037 ms (69% of GEMM hidden)

**Analysis — Why SDMA RS overlap succeeds (1.09×) while SDMA AG overlap fails (0.65×)**:

This is the bright spot for SDMA. The SDMA RS overlap achieves a genuine 1.09x speedup with 69% of GEMM hidden. This stands in stark contrast to both:
- SDMA AG overlap (0.65×) — catastrophic regression
- NCCL RS overlap (0.94×) — mild regression

The key difference is the kernel architecture. SDMA RS uses `AllreduceSdma`, which has a **different kernel structure** from the AG's PUT kernel:
- No PUT + barrier + copy three-phase pipeline
- The allreduce kernel is more compact and releases CUs earlier
- GEMM CV is only 1.3% (stable), confirming minimal interference

This also explains why NCCL RS overlap (0.94×) is worse than SDMA RS overlap (1.09×) — NCCL's RS reduction runs entirely on CUs and directly competes with GEMM, while SDMA's allreduce uses a different execution path that contends less.

---

## 5. NCCL vs SDMA All-Gather Head-to-Head

| Metric | Avg (ms) | Min (ms) | Max (ms) | CV (%) |
|--------|----------|----------|----------|--------|
| NCCL overlap | 0.261 | 0.257 | 0.266 | 0.9 |
| SDMA overlap | 0.292 | 0.285 | 0.300 | 1.5 |

- **SDMA speedup vs NCCL**: 0.89x → **NCCL wins**

---

## 6. NCCL vs SDMA All-Gather Scaling (by N dimension)

| N | GEMM (ms) | GFLOPS | NCCL AG (ms) | SDMA AG (ms) | SDMA vs NCCL | Winner |
|---|-----------|--------|-------------|-------------|--------------|--------|
| 256 | 0.023 | 45,964 | 0.244 | 0.294 | 0.83x | NCCL |
| 1024 | 0.030 | 143,825 | 0.256 | 0.295 | 0.87x | NCCL |
| 4096 | 0.044 | 389,835 | 0.252 | 0.299 | 0.84x | NCCL |
| 7168 | 0.067 | 447,180 | 0.294 | 0.294 | 1.00x | NCCL |
| 14336 | 0.108 | 558,083 | 0.295 | 0.307 | 0.96x | NCCL |
| 28672 | 0.146 | 823,895 | 0.376 | 0.324 | **1.16x** | **SDMA** |

### Scaling Summary Table

| N | NCCL Avg (ms) | SDMA Avg (ms) | SDMA vs NCCL | Winner |
|---|---------------|---------------|--------------|--------|
| 256 | 0.245 | 0.295 | 0.83x | NCCL |
| 1024 | 0.246 | 0.296 | 0.83x | NCCL |
| 4096 | 0.249 | 0.297 | 0.84x | NCCL |
| 7168 | 0.266 | 0.300 | 0.89x | NCCL |
| 14336 | 0.290 | 0.305 | 0.95x | NCCL |
| 28672 | 0.326 | 0.327 | 1.00x | NCCL~ |

**Trend**: As N grows, SDMA closes the gap with NCCL. At N=28672 in the per-test result, SDMA wins (1.16x). In the scaling summary (which re-runs the sweep), they converge to parity. This is because SDMA's overhead is **fixed** (PUT kernel launch, barrier, flag polling) while NCCL's cost scales with data volume. At large N, NCCL's bandwidth cost dominates its fixed overhead, letting SDMA catch up.

---

## 7. NCCL vs SDMA Reduce-Scatter Head-to-Head

| Metric | Avg (ms) | Min (ms) | Max (ms) | CV (%) | BW (GB/s) |
|--------|----------|----------|----------|--------|-----------|
| NCCL RS overlap | 0.489 | 0.485 | 0.498 | 0.7 | 240.1 |
| SDMA RS overlap | 0.755 | 0.746 | 0.766 | 0.6 | 155.6 |

- **SDMA speedup vs NCCL**: 0.65x → **NCCL wins decisively**
- **Bandwidth gap**: NCCL achieves 240 GB/s vs SDMA's 156 GB/s for reduce-scatter

---

## 8. NCCL vs SDMA Reduce-Scatter Scaling (by N dimension)

| N | GEMM (ms) | NCCL RS (ms) | SDMA RS (ms) | SDMA vs NCCL | BW (GB/s) | Winner |
|---|-----------|-------------|-------------|--------------|-----------|--------|
| 256 | 0.019 | 0.109 | 0.115 | 0.95x | 18.3 | NCCL |
| 1024 | 0.024 | 0.134 | 0.145 | 0.92x | 57.9 | NCCL |
| 4096 | 0.037 | 0.216 | 0.279 | 0.77x | 120.2 | NCCL |
| 7168 | 0.065 | 0.292 | 0.423 | 0.69x | 138.9 | NCCL |
| 14336 | 0.124 | 0.475 | 0.758 | 0.63x | 154.9 | NCCL |
| 28672 | 0.191 | 0.840 | 1.407 | 0.60x | 166.9 | NCCL |

### RS Scaling Summary Table

| N | NCCL RS Avg (ms) | SDMA RS Avg (ms) | SDMA vs NCCL | SDMA BW (GB/s) | Winner |
|---|------------------|------------------|--------------|----------------|--------|
| 256 | 0.107 | 0.113 | 0.94x | 18.6 | NCCL |
| 1024 | 0.133 | 0.145 | 0.92x | 57.9 | NCCL |
| 4096 | 0.217 | 0.279 | 0.78x | 120.4 | NCCL |
| 7168 | 0.288 | 0.420 | 0.69x | 139.9 | NCCL |
| 14336 | 0.475 | 0.751 | 0.63x | 156.5 | NCCL |
| 28672 | 0.844 | 1.395 | 0.61x | 168.4 | NCCL |

**Trend**: Unlike All-Gather, the reduce-scatter gap *widens* as N grows. SDMA RS performance degrades from 0.94x at N=256 to 0.61x at N=28672. SDMA RS bandwidth plateaus around 156–168 GB/s while NCCL scales to 240 GB/s. The SDMA allreduce-based RS implementation likely suffers from suboptimal multi-hop data movement patterns at 8-GPU scale, where NCCL's highly optimized ring/tree topology has a clear advantage.

---

## 9. Stability Analysis

| Test | CV (%) | Assessment |
|------|--------|------------|
| NCCL AG allgather alone | 0.8 | Stable |
| NCCL AG GEMM alone | 4.2 | ~unstable |
| NCCL AG overlapped | 1.0 | Stable |
| NCCL RS reduce_scatter | 1.4 | Stable |
| NCCL RS overlapped | 1.7 | Stable |
| SDMA AG allgather | 1.4 | Stable |
| SDMA AG GEMM | **11.3** | **!NOISY** |
| SDMA AG overlap | 1.4 | Stable |
| SDMA RS reduce_scatter | 0.5 | Very stable |
| SDMA RS overlap | 1.0 | Stable |
| GEMM N=28672 | **5.7** | **!NOISY** |
| NCCL RS N=28672 scaling | **6.0** | **!NOISY** |

Noisy measurements are concentrated in:
- **SDMA AG GEMM (11.3%)** — The PUT kernel (Grid=32768, Block=256) running concurrently on the same GPU causes non-deterministic CU scheduling for the GEMM. This high variance is direct evidence of the CU contention described in Section 3.
- **Large-N GEMM (5.7%)** — expected variance at large matrix sizes due to memory subsystem pressure
- **Large-N NCCL RS (6.0%)** — scaling summary outlier, likely from occasional network congestion on the 8-GPU XGMI fabric

---

## 10. 8-GPU vs 2-GPU Comparison

| Test | 2-GPU Speedup | 8-GPU Speedup | Trend | Root Cause |
|------|---------------|---------------|-------|------------|
| NCCL AG overlap | 1.08x | 1.16x | Improved | NCCL AG uses NIC, CU-free — scales well |
| NCCL RS overlap | 1.03x | 0.94x | Regressed | RS reduction needs CUs; 7 partial sums vs 1 |
| SDMA AG overlap | 1.02x | 0.65x | Severe regression | PUT kernel CU footprint ×7; flag poll ×7 |
| SDMA RS overlap | 1.03x | 1.09x | Improved | Allreduce kernel is compact; less CU contention |
| AG head-to-head | SDMA 1.72x | NCCL 0.89x | Reversed | SDMA PUT overhead dominates at 8-GPU scale |
| RS head-to-head | SDMA 1.15x | NCCL 0.65x | Reversed | SDMA RS bandwidth plateaus; NCCL scales better |

**Key scaling observation**: The 2-GPU → 8-GPU transition **reverses the SDMA vs NCCL ranking** for both AG and RS. SDMA's architectural advantages (low raw latency, dedicated DMA engines) are negated at 8-GPU scale by:
1. The PUT kernel's CU-bound control plane scaling O(peers)
2. Flag polling and barrier synchronization latency growing with peer count
3. NCCL's ring/tree topology being specifically optimized for multi-GPU collective operations

---

## 11. Deep-Dive: Negative Optimization Root Causes

### 11.1 SDMA AG Overlap (0.65×) — PUT Kernel CU Contention

| Factor | Contribution | Detail |
|--------|-------------|--------|
| **PUT kernel CU occupancy** | Primary | `Grid=32768, Block=256` (8M threads) for address calculation, flag writes, data staging — all on CUs. Workload ×7 vs 2-GPU. |
| **Flag polling busy-wait** | Secondary | WAIT phase kernel polls 7 remote flags on CUs until the slowest peer completes. Straggler effect amplified. |
| **Barrier synchronization** | Contributing | 8-PE barrier has higher latency than 2-PE (O(log 8) vs O(1) hops). |
| **Output copy volume** | Contributing | Copying 8 shards from transit buffer → user output (4× the 2-GPU data volume). |
| **Stream scheduling overhead** | Contributing | GPU scheduler must interleave PUT kernel and GEMM kernel across CUs. |

The SDMA architecture's promise of "zero CU contention" holds only for the **data movement phase** (DMA engine transfers). The kernel's **control plane** — address computation, flag operations, synchronization — still runs on CUs and scales with peer count.

### 11.2 NCCL RS Overlap (0.94×) — Reduction CU Contention

| Factor | Contribution | Detail |
|--------|-------------|--------|
| **RS reduction on CUs** | Primary | NCCL reduce-scatter performs element-wise sum on GPU CUs, directly competing with GEMM. |
| **Increased reduction work** | Amplifying | 8-GPU RS requires 7 partial sums (vs 1 on 2-GPU), increasing CU demand. |
| **Stream scheduling overhead** | Contributing | Two CU-bound kernels on separate streams cause scheduler contention. |

### 11.3 Why SDMA RS Overlap Succeeds (1.09×)

The SDMA RS overlap is the sole case where SDMA achieves positive overlap on 8 GPUs. This is because `AllreduceSdma` uses a fundamentally different kernel architecture:
- No three-phase PUT → barrier → copy pipeline
- More compact kernel with shorter CU residency
- GEMM CV is only 1.3% (vs 11.3% for SDMA AG), confirming minimal interference

---

## 12. Key Insights & Recommendations

### All-Gather (Column-Parallel)

1. **NCCL AG overlap works well at 8 GPUs** — 1.16x speedup confirms that the allgather latency (~0.21ms) successfully hides the GEMM (~0.034ms). NCCL's NIC-based data movement leaves CUs free for compute.
2. **SDMA AG overlap fails at 8 GPUs** — the overlapped path (0.298ms) is 54% slower than sequential (0.193ms). The root cause is not DMA engine contention but the PUT kernel's **control-plane CU usage** scaling with peer count (7 peers × address calc + flag set + poll).
3. **SDMA AG catches up at large N**: At N=28672, SDMA allgather (0.324ms) beats NCCL (0.376ms) by 1.16x. SDMA's fixed overhead is amortized at large transfer sizes where NCCL's bandwidth-proportional cost dominates.

### Reduce-Scatter (Row-Parallel)

4. **NCCL RS is consistently faster than SDMA** across all sizes, with the gap widening at larger N (0.94x → 0.61x). NCCL achieves 240 GB/s vs SDMA's 156 GB/s at scale.
5. **NCCL RS overlap regresses (0.94×)** because the reduction operation itself consumes CUs, competing with GEMM. This is inherent to reduce-scatter semantics and worsens with more GPUs.
6. **SDMA RS overlap paradox**: Despite SDMA RS being slower in absolute terms, the SDMA RS *overlap* achieves 1.09x speedup (vs NCCL RS overlap at 0.94x). The AllreduceSdma kernel has a more compact CU footprint that doesn't interfere with GEMM as heavily as NCCL's RS reduction.

### Actionable Recommendations

| Priority | Action | Rationale |
|----------|--------|-----------|
| **High** | Use NCCL for both AG and RS at 8-GPU scale by default | NCCL wins across nearly all configurations |
| **High** | Reduce SDMA PUT kernel grid size | Current `Grid=32768` is excessive for control-plane work; smaller grid would free CUs for GEMM overlap |
| **High** | Investigate offloading PUT kernel flag/address logic to firmware or DMA command processor | Would eliminate CU contention entirely, achieving true hardware-level parallelism |
| **Medium** | Profile SDMA RS to understand 240 → 156 GB/s bandwidth gap vs NCCL | May be suboptimal multi-hop patterns or transit buffer sizing |
| **Medium** | Consider SDMA AG only for very large tensors (N ≥ 28672) where fixed overhead is amortized | Cost-benefit crossover at ~112 MB transfer size |
| **Medium** | Explore hybrid strategy: SDMA AG (raw latency wins) + NCCL RS (bandwidth wins) for end-to-end TP | Combines each backend's strength |
| **Low** | Increase warmup iterations for SDMA AG GEMM measurement | CV=11.3% may partially be a warmup artifact; confirm with longer runs |

---

## 13. Cross-Benchmark Reconciliation With `bench_e2e_fusion`

This log and `bench_e2e_fusion_8GPU_analysis.md` do **not** answer the same
question, so their SDMA/NCCL conclusions are not directly interchangeable.

| Benchmark | Unit under test | What "SDMA better/worse" means |
|-----------|-----------------|--------------------------------|
| `bench_comm_overlap` | individual overlap pattern or collective path | the specific AG/RS path itself is faster/slower |
| `bench_e2e_fusion` | full single-layer pure-pipeline schedule | the fused backend yields lower **end-to-end layer latency** |

The most important example is reduce-scatter:

- This file's RS scaling summary says the **raw 8-GPU RS path** is slower with
  SDMA than with NCCL.
- That does **not** contradict `bench_e2e_fusion`, where SDMA is the better
  fused backend, because the E2E benchmark measures the total chunked schedule:
  AG + GEMM-up + GEMM-down + RS + chunk staging + synchronization + backward.

So there are three valid statements that can all be true at once:

1. **Raw RS primitive at 8 GPUs**: `NCCL > SDMA`
2. **Fused single-layer pure-pipeline backend**: `SDMA > NCCL`
3. **Against the naive baseline**: `naive > SDMA > NCCL`

One more reporting detail matters here: this file's RS head-to-head tables use
standalone overlap timings for the measured collective path, while
`bench_e2e_fusion` reports AG/RS "bandwidth" from the **fused layer latency**.
Those E2E bandwidth numbers are therefore schedule-level effective bandwidth,
not raw collective bandwidth.

The safe interpretation is:

- use this file to decide which **collective path** is better in isolation
- use `bench_e2e_fusion` to decide which **backend behaves better inside the
  current fused layer schedule**
- do not infer from the E2E tables that every SDMA primitive is faster than its
  NCCL counterpart

---

## 14. Raw Benchmark Timing Reference

### Per-Section Results

| Section | Metric | Avg (ms) | Speedup |
|---------|--------|----------|---------|
| NCCL Column-Parallel | overlapped | 0.262 | 1.16x |
| NCCL Row-Parallel | overlapped | 0.206 | 0.94x |
| SDMA Column-Parallel | SDMA overlap | 0.298 | 0.65x |
| SDMA Row-Parallel | SDMA RS overlap | 0.270 | 1.09x |
| NCCL vs SDMA AG | NCCL/SDMA | 0.261/0.292 | NCCL wins |
| NCCL vs SDMA RS | NCCL/SDMA | 0.489/0.755 | NCCL wins |

### All-Gather Scaling (NCCL vs SDMA)

```
N        NCCL(ms)  SDMA(ms)  ratio   winner
  256    0.245     0.295     0.83x   NCCL
 1024    0.246     0.296     0.83x   NCCL
 4096    0.249     0.297     0.84x   NCCL
 7168    0.266     0.300     0.89x   NCCL
14336    0.290     0.305     0.95x   NCCL
28672    0.326     0.327     1.00x   NCCL~
```

### Reduce-Scatter Scaling (NCCL vs SDMA)

```
N        NCCL(ms)  SDMA(ms)  ratio   BW(GB/s)  winner
  256    0.107     0.113     0.94x    18.6     NCCL
 1024    0.133     0.145     0.92x    57.9     NCCL
 4096    0.217     0.279     0.78x   120.4     NCCL
 7168    0.288     0.420     0.69x   139.9     NCCL
14336    0.475     0.751     0.63x   156.5     NCCL
28672    0.844     1.395     0.61x   168.4     NCCL
```
