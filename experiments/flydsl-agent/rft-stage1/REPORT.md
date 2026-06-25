# RFT Stage 1 实验报告

## 实验目标

按 plan.md Stage A: Diversity-Preserving RFT 的设计，用 SFT v2 模型对 RL spec 生成候选 kernel 代码，通过 FlyDSL-Gym 沙箱编译验证筛选正确实现。

## 实验设置

| 项目 | 配置 |
|------|------|
| SFT 模型 | Qwen2.5-Coder-SFT-v2 (LoRA r=64, 3 epochs, 59% kernel data) |
| RL specs | 13 specs (每种 operator 1 个) |
| 每 spec 候选数 | N=16 |
| 总候选数 | 208 |
| 生成温度 | temperature=0.8, top_p=0.95 |
| 提示风格 | 3 种轮换 (precise/natural/optimization) |
| 沙箱 | flydsl-gym:latest (rocm/vllm-dev:nightly + FlyDSL) |

## 结果

### 沙箱编译通过率: 0/208 (0%)

| 阶段 | 通过数 | 通过率 | 说明 |
|------|--------|--------|------|
| 总候选 | 208 | 100% | |
| Python 语法正确 | ~100 | ~48% | 52% 包含未清理的 markdown code block |
| 有 FlyDSL pattern | 203 | 98% | 几乎所有候选都包含 @flyc 或 import flydsl |
| **FlyDSL 编译通过** | **0** | **0%** | **全部失败** |

### 根因分析

**核心问题：模型学会了 FlyDSL 的概念但幻想了 API 路径。**

模型生成的 import 语句 vs 真实 FlyDSL API：

| 模型幻想的 import | 出现次数 | 真实 API |
|-------------------|---------|----------|
| `from flydsl.allocators import SharedAllocator` | 466 | `from flydsl.utils.smem_allocator import SmemAllocator` |
| `from flydsl.ops import ...` | 367 | 不存在，应用 `flydsl.expr` 下的子模块 |
| `from flydsl.core.fx import ...` | 318 | `import flydsl.expr as fx` |
| `from flydsl.gpu.wmma.mma.mma_config import ...` | 189 | `from flydsl.expr.rocdl import ...` |
| `import flydsl.flyc as flyc` | ~170 | `import flydsl.compiler as flyc` |

真实 FlyDSL 的标准 import pattern：
```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.expr import arith, buffer_ops, rocdl, range_constexpr
```

### 其他问题

1. **Markdown code block 未清理** — 52% 的候选代码被 ` ```python ... ``` ` 包裹，导致语法错误
2. **FIM token 泄漏** — 部分候选包含 `<|fim_suffix|>`、`<|fim_prefix|>` 等 token
3. **相对 import** — `from .mma import ...` 在独立文件中不可用
4. **幻想的类/函数** — `FxVector`, `get_block_info2`, `fx.allreduce` 等不存在

## 结论

### 0% 通过率是预期的

这不是沙箱 bug。SFT v2 模型虽然在 benchmark 上 `@flyc.kernel` 使用率达到 92%、`fx.*` API 92%，但这些是**模式匹配**指标 — 检查的是生成代码中是否出现这些关键词，而不是这些关键词是否用在正确的 import path 和 API 签名上。

模型知道 FlyDSL 使用 `@flyc.kernel`、`SmemAllocator`、`fx.make_layout`，但不知道这些 API 的正确模块路径。这是因为：
- SFT 训练数据中的 kernel 代码来自 CPT 语料（加了 `<|doc_start|>` 等标签），import 路径被 metadata header 干扰
- 增强的 SFT 数据虽然提高了 kernel 代码比例，但没有教会模型正确的 import pattern

### 下一步建议

1. **修复 SFT 数据中的 import pattern** — 确保所有 kernel 代码样本使用标准的 `import flydsl.compiler as flyc; import flydsl.expr as fx`
2. **添加 import-focused SFT 数据** — 专门生成一批只训练正确 import 的样本
3. **在 system prompt 中加入正确 import 模板** — 让模型在生成时有参考
4. **候选生成时 strip markdown code block** — 预处理清除 ``` 标记
5. **RFT 暂缓** — 等 SFT v3 解决 import 问题后再做

## 产物

| 文件 | 位置 |
|------|------|
| 候选代码 | `/home/danyzhan/rft-results/candidates.jsonl` (13 specs, 208 candidates) |
| 静态验证 | `/home/danyzhan/rft-results/verified_static.jsonl` |
| 沙箱验证 | `/home/danyzhan/rft-results/verified.jsonl` (0 passed) |
| 验证统计 | `/home/danyzhan/rft-results/verify_stats.json` |
