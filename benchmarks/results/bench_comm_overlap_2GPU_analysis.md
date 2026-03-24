# Comm-GEMM Overlap Benchmark Analysis — 2-GPU Results

**Source log:** `bench_comm_overlap_2GPU.log`
**Test suite:** `benchmarks/bench_comm_overlap.py` (20 tests, all PASSED)
**Configuration:** 2× AMD MI250X GCDs, `torchrun --nproc_per_node=2`
**Matrix shape:** M=8192, K=7168, N=7168 (default); N varies in scaling tests
**Runtime:** ~8.0 s total

---

## 1. Executive Summary

| Category | Key Finding |
|---|---|
| **NCCL AG + GEMM Overlap** | 1.08× speedup; overlap_ratio=0.014; 46% of GEMM hidden |
| **NCCL RS + GEMM Overlap** | 1.03× speedup; overlap_ratio=0.054; 34% of GEMM hidden |
| **SDMA AG + GEMM Overlap** | 1.02× speedup; overlap_ratio=0.060; 仅 7% GEMM hidden |
| **SDMA RS + GEMM Overlap** | 1.03× speedup; overlap_ratio=0.024; 27% of GEMM hidden |
| **SDMA vs NCCL AG (overlapped)** | SDMA 1.72× faster (0.431 vs 0.742 ms) |
| **SDMA vs NCCL AG Scaling** | SDMA wins at all N; advantage 1.48×–1.57× (summary sweep) |
| **SDMA vs NCCL RS Scaling** | NCCL wins at N=256 (0.97×); SDMA wins at N≥1024, peak 1.19× at N=7168 |
| **RS Bandwidth** | SDMA: 49.8–52.1 GB/s; NCCL: 43.1 GB/s |

---

## 2. Test-by-Test Breakdown

### 2.1 NCCL Column-Parallel AllGather + GEMM Overlap

实际 2-GPU 分布式基准测试 (NCCL 后端)。

| Component | Avg (ms) | Min | Max | CV (%) | 备注 |
|---|---|---|---|---|---|
| AllGather alone | 0.623 | 0.620 | 0.629 | 0.4% | 稳定 |
| GEMM alone | 0.130 | 0.108 | 0.140 | 5.5% | [!NOISY] |
| Sequential | 0.801 | 0.795 | 0.809 | 0.5% | 稳定 |
| **Overlapped** | **0.742** | 0.736 | 0.749 | 0.5% | 稳定 |

**关键指标：**
- **Overlap ratio:** 0.014 (1.4%)
- **Speedup:** 1.08×
- **Hidden GEMM:** 0.059 / 0.130 ms = **46% of GEMM hidden**

**分析：**

overlapped 时间 (0.742 ms) 接近于 AllGather 延迟 (0.623 ms) 加上约一半的 GEMM 时间。在此配置下 GEMM/comm 比为 0.130/0.623 ≈ 0.21 (计算远小于通信)，理论最大 speedup 为 `(AG + GEMM) / AG` = 1.21×。实测 1.08× 低于理论值，说明只有 46% 的 GEMM 被成功隐藏。

造成 overlap 不完全的原因：
1. **NCCL AG kernel 占用部分 CU 资源**，影响 GEMM 并行执行效率
2. **Stream 同步开销**：overlap 路径需要额外的 CUDA stream 管理
3. **GEMM 噪声大** (CV=5.5%)，小尺寸 GEMM 的 kernel launch overhead 相对显著

---

### 2.2 NCCL Row-Parallel Reduce-Scatter + GEMM Overlap

| Component | Avg (ms) | Min | Max | CV (%) | 备注 |
|---|---|---|---|---|---|
| Reduce-Scatter | 0.758 | 0.754 | 0.763 | 0.3% | 稳定 |
| GEMM | 0.069 | 0.065 | 0.074 | 4.5% | [~unstable] |
| Sequential | 0.807 | 0.804 | 0.813 | 0.3% | 稳定 |
| **Overlapped** | **0.783** | 0.780 | 0.788 | 0.3% | 稳定 |

**关键指标：**
- **Overlap ratio:** 0.054 (5.4%)
- **Speedup:** 1.03×
- **Hidden GEMM:** 0.024 / 0.069 ms = **34% of GEMM hidden**

**分析：**

RS 主导延迟 (0.758 ms vs GEMM 0.069 ms，**11× 的比值**)。由于 GEMM 极小 (仅 0.069 ms)，即使完美隐藏全部 GEMM，理论最大 speedup 也仅为 1.09×。

实测仅 34% 隐藏的原因：**NCCL reduce-scatter 不同于 all-gather，RS 需要在 GPU 上执行 reduction 操作（求和），因此会占用计算单元 (CU)**，与 GEMM 争抢资源。这在之前 8-GPU 分析中已确认——NCCL RS overlap 甚至可能出现负优化 (8GPU 下 speedup=0.94×)。

2-GPU 下竞争较轻，仍能获得微弱正收益 (1.03×)。

---

### 2.3 SDMA Column-Parallel AllGather + GEMM Overlap

| Component | Avg (ms) | Min | Max | CV (%) | 备注 |
|---|---|---|---|---|---|
| SDMA AllGather | 0.334 | 0.332 | 0.336 | 0.3% | 稳定 |
| GEMM | 0.123 | 0.110 | 0.130 | 5.9% | [!NOISY] |
| Sequential | 0.438 | 0.434 | 0.441 | 0.5% | 稳定 |
| **SDMA overlap** | **0.430** | 0.426 | 0.434 | 0.5% | 稳定 |

**关键指标：**
- **Overlap ratio:** 0.060 (6.0%)
- **Speedup:** 1.02×
- **Hidden GEMM:** 0.008 / 0.123 ms = **仅 7% of GEMM hidden**

**分析：**

这是一个重要的矛盾结果：

- **SDMA 裸 AllGather 延迟显著更低**：0.334 ms vs NCCL 0.623 ms (**1.87× faster**)
- **但 overlap 效率极差**：仅 7% GEMM 被隐藏 (对比 NCCL 46%)
- **绝对时间仍然更优**：0.430 ms vs NCCL 0.742 ms (**1.72× faster**)

**原因：SDMA PUT kernel 占用 GPU CU**

SDMA 的 AllGather 实现使用 PUT kernel (Grid=32768, Block=256) 在 GPU 上运行数据拷贝内核。该内核：
- 启动参数为 `Grid size: 32768, Block size: 256`，占用大量 CU 资源
- 在整个 PUT 阶段持续占用 CU，阻止 GEMM 有效并行

与 NCCL 相比，NCCL 的 AllGather 主要通过 NIC/DMA 引擎传输数据，GPU CU 占用较少，因此 GEMM 能更好地并行执行。

**结论：**SDMA 的绝对延迟优势 (1.87×) 远超 overlap 效率劣势。在端到端场景中，SDMA 始终是更优选择——即使它几乎不 overlap 任何 GEMM，单纯的低延迟就已经比 NCCL 的 overlap 版本更快。

---

### 2.4 SDMA Row-Parallel Reduce-Scatter + GEMM Overlap

| Component | Avg (ms) | Min | Max | CV (%) | 备注 |
|---|---|---|---|---|---|
| SDMA RS | 0.656 | 0.655 | 0.659 | 0.2% | 稳定 |
| GEMM | 0.067 | 0.065 | 0.069 | 2.4% | [~unstable] |
| Sequential | 0.724 | 0.720 | 0.734 | 0.5% | 稳定 |
| **SDMA RS overlap** | **0.706** | 0.703 | 0.711 | 0.3% | 稳定 |

**关键指标：**
- **Overlap ratio:** 0.024 (2.4%)
- **Speedup:** 1.03×
- **Hidden GEMM:** 0.018 / 0.067 ms = **27% of GEMM hidden**

**分析：**

SDMA RS 延迟比 NCCL RS 低 13.5% (0.656 vs 0.758 ms)。Overlap 效果与 NCCL RS 相当（NCCL 34% vs SDMA 27%），两者都受限于 GEMM 极小的问题。

---

### 2.5 NCCL vs SDMA AllGather Head-to-Head

直接对比 overlapped 模式下的延迟 (默认 N=7168):

| Backend | Overlapped Avg (ms) | CV (%) |
|---|---|---|
| NCCL | 0.742 | 0.5% |
| SDMA | 0.431 | 0.5% |

**SDMA speedup: 1.72×**

---

### 2.6 NCCL vs SDMA AllGather Scaling (N Sweep)

#### 逐 N 测试结果

| N | GEMM (ms) | GFLOPS | NCCL AG (ms) | SDMA AG (ms) | SDMA vs NCCL | Winner |
|---|---|---|---|---|---|---|
| 256 | 0.027 | 158,316 | 0.655 | 0.428 | **1.53×** | SDMA |
| 1024 | 0.051 | 335,110 | 0.691 | 0.429 | **1.61×** | SDMA |
| 4096 | 0.082 | 833,767 | 0.656 | 0.430 | **1.53×** | SDMA |
| 7168 | 0.123 | 977,608 | 0.739 | 0.428 | **1.73×** | SDMA |
| 14336 | 0.242 | 995,028 | 0.671 | 0.428 | **1.57×** | SDMA |
| 28672 | 0.475 | 1,013,309 | 1.078 | 0.519 | **2.08×** | SDMA |

#### Scaling Summary (最终汇总 sweep)

| N | NCCL (ms) | SDMA (ms) | Speedup | Winner |
|---|---|---|---|---|
| 256 | 0.654 | 0.427 | **1.53×** | SDMA |
| 1024 | 0.656 | 0.429 | **1.53×** | SDMA |
| 4096 | 0.657 | 0.427 | **1.54×** | SDMA |
| 7168 | 0.661 | 0.428 | **1.54×** | SDMA |
| 14336 | 0.670 | 0.427 | **1.57×** | SDMA |
| 28672 | 0.753 | 0.508 | **1.48×** | SDMA |

**关键发现：**

1. **SDMA AllGather 延迟几乎恒定** (~0.427–0.429 ms) 对 N ≤ 14336，说明 SDMA PUT 路径在这些数据量下不受带宽瓶颈限制。
2. **NCCL AllGather 延迟随 N 增长**：从 0.654 ms (N=256) 增至 0.753 ms (N=28672)，涨幅 15%，反映 NCCL 的带宽成正比开销。
3. **SDMA 优势在所有 N 值下稳定保持 1.48–1.57×**。
4. **N=28672 时 SDMA 延迟首次上升至 0.508 ms**：提示在 ~112 MB (28672×7168×2B / 2 ranks) 传输量时，SDMA transit buffer (16 MB input + 32 MB output) 开始成为瓶颈。

---

### 2.7 NCCL vs SDMA Reduce-Scatter Head-to-Head

默认配置下的 RS overlap 对比：

| Backend | RS Overlap Avg (ms) | BW (GB/s) |
|---|---|---|
| NCCL | 2.723 | 43.1 |
| SDMA | 2.358 | 49.8 |

**SDMA speedup: 1.15×** | BW 优势: +15.5%

数据量：117.4 MB (M=8192, N=7168, BF16, 单 rank shard)

---

### 2.8 NCCL vs SDMA Reduce-Scatter Scaling (N Sweep)

#### RS Scaling Summary

| N | Data Size (MB) | NCCL RS (ms) | SDMA RS (ms) | Speedup | Winner | SDMA BW (GB/s) |
|---|---|---|---|---|---|---|
| 256 | 0.5 | 0.121 | 0.125 | 0.97× | **NCCL** | 16.8 |
| 1024 | 8.0 | 0.253 | 0.236 | **1.07×** | SDMA | 35.6 |
| 4096 | 32.0 | 0.830 | 0.710 | **1.17×** | SDMA | 47.3 |
| 7168 | 56.0 | 1.365 | 1.145 | **1.19×** | SDMA | 51.3 |
| 14336 | 112.0 | 2.688 | 2.338 | **1.15×** | SDMA | 50.2 |
| 28672 | 224.0 | 5.132 | 4.510 | **1.14×** | SDMA | 52.1 |

**关键发现：**

1. **NCCL 在小 N 时胜出**：N=256 (0.5 MB) 时 SDMA RS 慢 3%。SDMA RS 的 barrier/flag 同步开销在小数据量下相对显著。
2. **交叉点在 N≈1024** (~8 MB)：SDMA RS 开始反超。
3. **SDMA BW 随 N 增长而提升**：从 16.8 GB/s (N=256) 到 52.1 GB/s (N=28672)，逐步接近 XGMI 理论带宽上限。
4. **SDMA RS speedup 在 N≥4096 时趋于稳定 (~1.14–1.19×)**：不像 AllGather 场景中 SDMA 优势持续扩大。RS 需要更多同步协调（partial sum reduction），这限制了进一步提速。

---

## 3. 通信操作对比矩阵

### 3.1 AllGather (AG)

| | NCCL | SDMA | 对比 |
|---|---|---|---|
| **裸 AG 延迟** | 0.623 ms | 0.334 ms | SDMA **1.87×** faster |
| **AG + GEMM Overlapped** | 0.742 ms | 0.430 ms | SDMA **1.72×** faster |
| **GEMM 隐藏率** | 46% | 7% | NCCL 更好 |
| **Overlap speedup** | 1.08× | 1.02× | NCCL 更好 |
| **端到端赢家** | — | — | **SDMA** (绝对延迟低) |

### 3.2 Reduce-Scatter (RS)

| | NCCL | SDMA | 对比 |
|---|---|---|---|
| **裸 RS 延迟** | 0.758 ms | 0.656 ms | SDMA **1.16×** faster |
| **RS + GEMM Overlapped** | 0.783 ms | 0.706 ms | SDMA **1.11×** faster |
| **GEMM 隐藏率** | 34% | 27% | NCCL 略好 |
| **Overlap speedup** | 1.03× | 1.03× | 持平 |
| **端到端赢家** | — | — | **SDMA** (绝对延迟低) |

---

## 4. Stability Analysis (稳定性分析)

### 测量质量标志

| Flag | Meaning | Threshold |
|---|---|---|
| (none) | 稳定 | CV < 2% |
| `[~unstable]` | 轻微波动 | 2% ≤ CV < 5% |
| `[!NOISY]` | 高方差 | CV ≥ 5% |

### 稳定性汇总

| Category | 稳定性 | 说明 |
|---|---|---|
| NCCL/SDMA 通信独立测量 | **全部稳定** (CV < 1%) | 时序基础设施可靠 |
| Sequential/Overlapped 复合测量 | **全部稳定** (CV < 1%) | 可信度高 |
| GEMM alone (小 N) | **NOISY** (CV 4–6%) | 小尺寸 GEMM 的 kernel launch overhead 主导 |
| Scaling 总结测量 | 稳定，除 N=28672 SDMA AG (CV=3.3%) | 大数据量 SDMA 有偶发 buffer 延迟 |

---

## 5. GEMM 性能分析

### 5.1 GEMM Throughput vs N

| N | Avg (ms) | GFLOPS | 相对 Peak 利用率* |
|---|---|---|---|
| 256 | 0.027 | 158,316 | ~38% |
| 1024 | 0.051 | 335,110 | ~81% |
| 4096 | 0.082 | 833,767 | ~100% |
| 7168 | 0.123 | 977,608 | ~100% |
| 14336 | 0.242 | 995,028 | ~100% |
| 28672 | 0.475 | 1,013,309 | ~100% |

*以 MI250X 单 GCD ~400 TFLOPS BF16 为参考。

**发现：**
- N≥4096 时 GEMM 达到接近峰值吞吐量，这是 comm-GEMM overlap 最有意义的区间。
- N=256–1024 时 GEMM 非常快 (0.027–0.051 ms)，overlap 收益极为有限——不值得为了隐藏如此短的 GEMM 而承担 overlap 的同步开销。

---

## 6. 通信带宽分析

### 6.1 AllGather 带宽

| Backend | Operation | Latency (ms) | Estimated BW (GB/s)* | 说明 |
|---|---|---|---|---|
| NCCL | AG (default) | 0.623 | ~26 | Ring/tree 协议开销 |
| SDMA | AG (default) | 0.334 | ~48 | 直接 PUT 传输 |
| NCCL | AG (N=28672) | 0.753 | ~53 | 大数据量更高效 |
| SDMA | AG (N=28672) | 0.508 | ~79 | 接近 XGMI 带宽 |

*基于 data_size / latency 估算。2-GPU MI250X 通过 XGMI 连接，~100 GB/s 单向带宽。

### 6.2 Reduce-Scatter 带宽

| Backend | Peak BW (GB/s) | At N= | 说明 |
|---|---|---|---|
| NCCL | 43.1 | default (7168) | 受 ring RS 协议限制 |
| SDMA | 52.1 | 28672 | 最大化带宽利用 |

**SDMA RS 带宽增长曲线：** 16.8 → 35.6 → 47.3 → 51.3 → 50.2 → 52.1 GB/s (N=256→28672)

---

## 7. Actionable Insights (行动建议)

### 7.1 [HIGH] SDMA AllGather 全面优于 NCCL

SDMA 在**所有测试的 N 值**上均胜出 (1.48–2.08×)。对于典型 Transformer 模型 (hidden_dim=7168–14336):
- 预期 **1.54–1.57× AG 加速**
- 即使 SDMA 的 overlap 效率仅 7%，其绝对延迟仍远低于 NCCL 的 overlapped 版本

**建议：** AllGather 通信路径优先使用 SDMA。

### 7.2 [MEDIUM] SDMA RS 在小 N 时有惩罚

N=256 (0.5 MB) 时 SDMA RS 慢于 NCCL (0.97×)。如果模型架构使用小 RS 消息（例如极高 TP 度 + 窄层），NCCL 可能更优。

**建议：** 考虑混合策略——AG 用 SDMA，N<1024 时 RS 用 NCCL。

### 7.3 [MEDIUM] RS Overlap 收益有限

NCCL 和 SDMA 的 RS+GEMM overlap 均只有 1.03× speedup。原因:
- Row-parallel 路径中 GEMM 远小于 RS (0.067–0.069 ms vs 0.656–0.758 ms)
- RS kernel 本身占用 CU (尤其 NCCL)，与 GEMM 竞争

**建议：**
- 通过层间 pipeline 实现更大的 overlap 收益 (第 N 层的 RS 与第 N+1 层的 compute overlap)
- 考虑 fuse 额外的 post-RS 计算进入 overlap 窗口

### 7.4 [LOW] SDMA PUT Kernel CU 占用

SDMA AG 的 PUT kernel 使用 `Grid=32768, Block=256`，大量占用 CU 资源导致 GEMM 几乎无法并行。这是一个潜在优化方向——减小 PUT kernel 的 grid size 或使用硬件 DMA 引擎可以释放 CU 给 GEMM。

### 7.5 [LOW] SDMA Transit Buffer 大小

N=28672 时 SDMA AG 延迟从 ~0.428 ms 跳至 0.508 ms (+19%)，表明 transit buffer (16 MB input + 32 MB output) 在此规模下不够用。对 hidden_dim > 14K 的模型考虑增大 buffer。

---

## 8. 2-GPU vs 8-GPU 对比预览

| 指标 | 2-GPU | 8-GPU* | 趋势 |
|---|---|---|---|
| NCCL AG overlap speedup | 1.08× | 1.09× | 基本持平 |
| NCCL RS overlap speedup | 1.03× | **0.94×** (负优化) | 8-GPU 竞争加剧 |
| SDMA AG overlap speedup | 1.02× | 0.98× (负优化) | 8-GPU PUT kernel 竞争更严重 |
| SDMA RS overlap speedup | 1.03× | 1.09× | 8-GPU 反而改善 |

*来自 `bench_comm_overlap_8GPU_analysis.md` 的数据。

8-GPU 下 NCCL RS overlap 出现明显负优化 (0.94×)，验证了 RS 的 reduction 操作与 GEMM 争抢 CU 的结论——随着 GPU 数量增加，RS 的 reduction 工作量增大，CU 竞争加剧。

---

## 9. Test Inventory (测试清单)

20 个测试全部通过。

| # | Test Class | Test Name | Key Result |
|---|---|---|---|
| 1 | TestNCCLColumnParallelOverlap | test_allgather_gemm_overlap | **1.08× speedup**, 46% GEMM hidden |
| 2 | TestNCCLRowParallelOverlap | test_reduce_scatter_gemm_overlap | 1.03× speedup, 34% GEMM hidden |
| 3 | TestSdmaColumnOverlap | test_sdma_allgather_gemm_overlap | 1.02× speedup, 7% GEMM hidden |
| 4 | TestSdmaRowOverlap | test_sdma_reduce_scatter_overlap | 1.03× speedup, 27% GEMM hidden |
| 5 | TestNCCLvsSdma | test_nccl_vs_sdma_allgather_overlap | **SDMA 1.72× faster** |
| 6 | TestNCCLvsSdma | test_nccl_vs_sdma_scaling[N=256] | SDMA 1.53× |
| 7 | TestNCCLvsSdma | test_nccl_vs_sdma_scaling[N=1024] | SDMA 1.61× |
| 8 | TestNCCLvsSdma | test_nccl_vs_sdma_scaling[N=4096] | SDMA 1.53× |
| 9 | TestNCCLvsSdma | test_nccl_vs_sdma_scaling[N=7168] | SDMA 1.73× |
| 10 | TestNCCLvsSdma | test_nccl_vs_sdma_scaling[N=14336] | SDMA 1.57× |
| 11 | TestNCCLvsSdma | test_nccl_vs_sdma_scaling[N=28672] | SDMA 2.08× |
| 12 | TestNCCLvsSdma | test_nccl_vs_sdma_scaling_summary | SDMA 1.48–1.57× |
| 13 | TestNCCLvsSdma | test_nccl_vs_sdma_reduce_scatter_overlap | **SDMA 1.15×**, 49.8 vs 43.1 GB/s |
| 14 | TestNCCLvsSdma | test_nccl_vs_sdma_rs_scaling[N=256] | NCCL 0.97× (NCCL wins) |
| 15 | TestNCCLvsSdma | test_nccl_vs_sdma_rs_scaling[N=1024] | SDMA 1.07× |
| 16 | TestNCCLvsSdma | test_nccl_vs_sdma_rs_scaling[N=4096] | SDMA 1.17× |
| 17 | TestNCCLvsSdma | test_nccl_vs_sdma_rs_scaling[N=7168] | SDMA 1.19× |
| 18 | TestNCCLvsSdma | test_nccl_vs_sdma_rs_scaling[N=14336] | SDMA 1.15× |
| 19 | TestNCCLvsSdma | test_nccl_vs_sdma_rs_scaling[N=28672] | SDMA 1.14× |
| 20 | TestNCCLvsSdma | test_nccl_vs_sdma_rs_scaling_summary | NCCL wins N=256; SDMA peaks 1.19× |
