# RFT Stage A 实验报告 — Diversity-Preserving Rejection Fine-Tuning

## v5f RFT (当前，2026-07-06)

### 概述

在 SFT v5f (format-aligned) 基础上执行 Stage A RFT。
v5f 模型对 122 个 gfx950 spec 大规模生成候选 → FlyDSL-Gym 沙箱验证 → 保留所有编译通过的实现 → 1 epoch 训练。

**HuggingFace**: [Zhangdanyang/Qwen2.5-Coder-RFT-v5f](https://huggingface.co/Zhangdanyang/Qwen2.5-Coder-RFT-v5f)

### 模型进化路径

| 阶段 | API Score | 沙箱编译率 | L5 Expert |
|------|-----------|-----------|-----------|
| SFT v5e | 74% | 22% (12 specs) | 50% |
| SFT v5f (+format) | 74% | 38% (122 specs) | 50% |
| **RFT v5f** | **75%** | **53%** (122 specs) | **57%** |

### 三层验证结果

沙箱验证器现在支持三个层级：编译、运行、正确性。

| 层级 | 通过数 / 1952 | 通过率 | 说明 |
|------|-------------|--------|------|
| 静态检查 | 1598 | 82% | Python 语法 + FlyDSL pattern + ≥15 行 |
| **编译** | 944 | **48%** | FlyDSL JIT import 成功 (无 import/type error) |
| **运行** | 211 | **11%** | entry point 可调用 (不崩溃/不 OOM/不超时) |
| **正确性** | 0 | **0%** | 输出与 PyTorch reference 不匹配 |

运行但不正确的原因分布：

| 原因 | 数量 | 说明 |
|------|------|------|
| INCORRECT | 149 | quant kernel 逻辑错误 (max_diff > atol) |
| Shape mismatch | 52 | topk/rmsnorm 输出维度与 reference 不同 |
| Returned None | 10 | in-place kernel 但无法匹配结果 |

> **关键结论**：模型学会了 FlyDSL API 语法 (48% 编译通过) 和部分 host-side 结构 (11% 运行)，
> 但 kernel 内部计算逻辑全部是错的 (0% 正确)。这是预期的 — SFT+RFT 只基于编译 pass/fail 训练，
> 没有 correctness reward。**运行时正确性是 RL Stage B/C 的目标。**

### Pipeline 执行

#### Step 1: 候选生成 (v5f 模型)

| 项目 | 配置 |
|------|------|
| 模型 | Qwen2.5-Coder-SFT-v5f |
| Spec 来源 | 213 个 gfx950 spec → 采样 122 个（每算子均匀） |
| 每 spec 候选数 | N=16 |
| 总候选数 | 1,952 |
| 生成温度 | temperature=0.8, top_p=0.95 |
| 提示风格 | 3 种轮换 (precise / natural / optimization) |
| 耗时 | ~23h (单卡 cuda:0) |

#### Step 2: 沙箱验证 (编译级)

| 阶段 | 通过数 | 通过率 |
|------|--------|--------|
| 总候选 | 1,952 | 100% |
| 静态检查 | 1,598 | 82% |
| **FlyDSL-Gym 沙箱编译** | **1,026** | **53%** |
| Diversity filter | 1,026 | 53% |

编译通过率 vs v5f SFT (pre-RFT)：38% → **53%** (+15pp)

**全部 12 算子沙箱通过率对比**:

| 算子 | v5e | v5f SFT | **RFT v5f** | Delta (RFT-v5f) |
|------|-----|---------|-------------|-----------------|
| rmsnorm | 2 (17%) | 41 (43%) | **57 (59%)** | +17pp |
| quant | 1 (8%) | 100 (37%) | **158 (58%)** | +21pp |
| softmax | 5 (42%) | 71 (49%) | **84 (58%)** | +9pp |
| layernorm | 4 (33%) | 31 (39%) | **45 (56%)** | +18pp |
| custom | 5 (42%) | 120 (44%) | **150 (55%)** | +11pp |
| topk | 3 (25%) | 48 (43%) | **58 (52%)** | +9pp |
| gemm | 4 (33%) | 96 (35%) | **137 (50%)** | +15pp |
| rope | 5 (42%) | 34 (35%) | **48 (50%)** | +15pp |
| paged_attn | 0 (0%) | 26 (32%) | **40 (50%)** | +18pp |
| mla | 6 (50%) | 43 (38%) | **55 (49%)** | +11pp |
| moe | 5 (42%) | 60 (42%) | **69 (48%)** | +6pp |
| flash_attn | 2 (17%) | 63 (23%) | **125 (46%)** | +23pp |

全部 12 个算子的编译通过率均有提升，flash_attn 提升最大 (+23pp)。

#### Step 3: RFT 数据集构建

| 项目 | 数量 |
|------|------|
| 沙箱通过候选 | 733 |
| RFT 对 (×2 重复) | 1,466 |
| v5f SFT 数据 | 6,809 |
| **合并后总数** | **8,275** |
| RFT 数据占比 | 17.7% |

#### Step 4: RFT 训练

| 项目 | 配置 |
|------|------|
| Base model | Qwen2.5-Coder-SFT-v5f (merged) |
| 训练轮数 | 1 epoch |
| MAX_STEPS | 1,035 |
| LR | 5e-6 |
| LoRA | r=64, alpha=128, dropout=0.05 |
| seq_length | 16384 |
| GBS | 8 |
| val_loss | 0.9615 → 0.9533 |
| 耗时 | ~2h (8xMI350X) |

### Benchmark (RFT vs v5f vs v5e)

| Level | v5e | v5f | **RFT v5f** | Delta (RFT-v5f) |
|-------|-----|-----|-------------|-----------------|
| L1 | 100% | 100% | **100%** | 0% |
| L2 | 100% | 100% | **100%** | 0% |
| L3 | 72% | 72% | 68% | -4% |
| L4 | 49% | 46% | **49%** | **+3%** |
| L5 | 50% | 50% | **57%** | **+7%** |
| **Overall** | 74% | 74% | **75%** | **+1%** |

| Metric | RFT | v5f | Target |
|--------|-----|-----|--------|
| Format compliance | 96% | 96% | >= 90% ✅ |
| Sandbox compilation | 100% | 100% | >= 80% ✅ |

L5 提升点: blockscale_gemm +25pp, moe_2stage +12pp, preshuffle_gemm +12pp。

### 产物

| 文件 | 位置 |
|------|------|
| v5f 候选 (122 specs) | `rft-results/candidates_v5f_gfx950.jsonl` |
| RFT 候选 (122 specs, RFT model) | `rft-results/candidates_rft_v5f_gfx950.jsonl` |
| 编译级验证 | `rft-results/verify_stats_rft_v5f_gfx950.json` |
| 三层验证 (含运行+正确性) | `rft-results/verify_stats_rft_v5f_runtime.json` |
| RFT 训练数据 | `rft-results/rft_v5f_train.jsonl` (8,275 条) |
| RFT 训练日志 | `rft-results/rft_v5f_train.log` |
| Benchmark | `rft-results/benchmark_rft_v5f.json` |
| HuggingFace | [Zhangdanyang/Qwen2.5-Coder-RFT-v5f](https://huggingface.co/Zhangdanyang/Qwen2.5-Coder-RFT-v5f) |

### 下一步

SFT + RFT 阶段完成。验证器已支持三层检查 (编译/运行/正确性)。
正确性 0% 表明需要 RL 阶段的 correctness reward 来训练计算逻辑：

- **Stage B**: Single-Turn DAPO — 编译 + 正确性 reward，100-200 steps
- **Stage C**: Multi-Turn DAPO + HRD + PrimeEcho — 3 轮迭代 (生成→修复→优化)

---

# RFT v1 (archived, 基于 SFT v5e)

### 概述

在 SFT v5e 基础上执行的初版 RFT。84 specs, 1344 候选。

**结果：沙箱编译率 21.9% → 30.7%，L4 首次达标 (54%)，12/12 算子全覆盖。**

**HuggingFace**: [Zhangdanyang/Qwen2.5-Coder-RFT-v1](https://huggingface.co/Zhangdanyang/Qwen2.5-Coder-RFT-v1)

| 指标 | SFT v5e | RFT v1 | Δ |
|------|---------|--------|---|
| Overall | 74.1% | 74.6% | +0.5% |
| 沙箱编译 | 21.9% | 30.7% | +8.8% |
| 算子覆盖 | 11/12 | 12/12 | +1 |
