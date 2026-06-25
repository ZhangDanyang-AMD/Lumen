# SFT 实验报告 — v1 → v5e 完整演进

## 概述

九轮 SFT 迭代（v1→v5e）。Overall 50% → **74.1%**，沙箱编译率 0% → **21.9% (42/192)**。
关键转折：v5d 证明负例策略是打地鼠（消灭旧幻觉冒出新幻觉），v5e 转向正例洪水
（kernel 代码占比 35%→52%），沙箱编译率一举从 9.9% 翻倍到 21.9%。
SFT 阶段完成，模型已具备转入 RL (RFT → DAPO) 的充足基线能力。

**HuggingFace**: https://huggingface.co/Zhangdanyang/Qwen2.5-Coder-SFT-v5e

## 版本演进总表

| 版本 | 样本数 | Kernel率 | Val Loss | Overall | 沙箱编译 | 核心问题 |
|------|--------|---------|----------|---------|---------|---------|
| Base | — | — | — | 50.4% | — | 不懂 FlyDSL |
| v1 | 2,808 | 18% | 1.030 | 56.3% | — | 82% 非代码数据 |
| v2 | 2,916 | 59% | 0.985 | 72.4% | 0% | 37.6% 截断 + r=32 |
| v3 | 3,096 | 60% | 0.974 | 未测 | 0% | 81% import 幻觉 |
| v4 | 2,344 | 55% | 0.984 | 60.3% | 0.5% | 数据量减少致退化 |
| v5 | 2,596 | 58% | 待测 | 69.3% | 0% | 二级 API 幻觉 |
| v5b | 2,792 | 60% | 待测 | 76.5% | 4.7% (9/192) | 新变种幻觉 + 语法残留 |
| v5c | 2,943 | 62% | 待测 | 73.6% | 10.4% (20/192) | import 写法 + flyc API 幻觉 |
| v5d | 3,102 | 63% | 待测 | 待测 | 9.9% (19/192) | 打地鼠失败：旧幻觉消灭，新幻觉冒出 |
| **v5e** | **3,889** | **52%** | **待测** | **74.1%** | **21.9% (42/192)** | **SFT 完成，转入 RL** |

## v1: 基线 — 82% 非代码数据

### 问题

SFT 训练数据中 82% 不含 FlyDSL kernel 代码：

```
@flyc.kernel 出现率:  354/2808 (13%)
fx.* API:             471/2808 (17%)
实际 kernel 代码:     500/2808 (18%)
QA/文档/拒答:        2308/2808 (82%)

按来源:
  documentation_qa (669):        8% 含 kernel
  ai_annotated_instruction (375): 2% 含 kernel
  refusal_boundary (135):         0% 含 kernel
  augmentation_tile (105):       75% 含 kernel  ← 高质量但太少
```

### 结果

- Overall 56.3% (base 50.4%，仅 +5.9%)
- `@flyc.kernel` 使用率从 base 96% 降到 40% — 模型学会了写文档回答而不是 kernel
- L4-L5 几乎没有改善

### 脚本

无特殊脚本，使用原始 `flydsl-agent-dataset`。

---

## v2: 数据重采样 + kernel 提取 + r=64 + seq=16384

### 问题

v1 的两个根因：
1. **数据比例失衡** — kernel 代码只占 18%
2. **训练能力受限** — r=32 容量不足，seq=8192 截断了 37.6% 的数据

### 解决

1. **从 CPT 数据提取 kernel 代码**：54 个 FlyDSL kernel 文件 → 108 条 SFT 对
2. **加权重采样**：kernel 高源 5x，QA 0.3x，拒答 0.1x
3. **LoRA r=32 → r=64**：可训练参数 0.81% → 1.61%
4. **seq=8192 → 16384**：截断率 37.6% → 19.2%

```
脚本: experiments/flydsl-agent/dataprocess/enhance_sft_data.py
```

### 结果

| 指标 | v1 | v2 | Δ |
|------|-----|-----|-----|
| Overall | 56.3% | **72.4%** | +16.1% |
| @flyc.kernel | 40% | **92%** | +52% |
| fx.* API | 68% | **92%** | +24% |
| import flydsl | 96% | **100%** | +4% |
| L1 | 87% | **100%** | +13% |
| L3 GEMM | 64% | **76%** | +12% |
| L4 FP8 GEMM | 26% | **49%** | +23% |

### 遗留问题

RFT Stage 1 用 v2 模型生成 208 候选 → **沙箱编译通过率 0%**。

---

## v3: Import 幻觉修复

### 问题

RFT Stage 1 发现 v2 模型生成的代码 **81% import 路径是幻想的**：

```
v2 模型生成的 import 路径 vs 真实 FlyDSL API:

  454x  × from flydsl.allocators import SharedAllocator
          → ✓ from flydsl.utils.smem_allocator import SmemAllocator

  299x  × from flydsl.core.fx import fx_make_variable_value_type_impl
          → ✓ import flydsl.expr as fx

  261x  × import flydsl as fx
          → ✓ import flydsl.expr as fx

  188x  × from flydsl.gpu.wmma.mma.mma_config import WmmaMmaConfig
          → ✓ from flydsl.expr.rocdl import cluster

  99x   × import flydsl as flyc
          → ✓ import flydsl.compiler as flyc

正确 import 比例: 18.6% (995 / 5349)
```

**矛盾**：SFT 训练数据中 `import flydsl.compiler as flyc` 出现了 1375 次、
`import flydsl.expr as fx` 出现了 1570 次，但模型仍然生成错误 import。
原因是这些正确 import 被埋在长 kernel 代码中间，模型没有独立学会它们。

### 解决

新增 60 条 import-focused SFT 数据 × 3 重复 = 180 条：

| 类型 | 数量 | 内容 |
|------|------|------|
| Import 模板 | 24 | "写 FlyDSL 标准 import" → 正确 import block |
| Kernel 骨架 | 6 | kernel 框架含正确 import + @flyc.kernel + @flyc.jit |
| Import 纠错 | 30 | 15 种常见错误 import → 正确 import + 解释 |

```
脚本: experiments/flydsl-agent/dataprocess/fix_import_sft.py
```

### 结果

Val loss: 0.985 → 0.974（改善）。Benchmark 和 RFT 因 v4 数据问题发现而未完整测试。

---

## v4: Hardware-Feature 错配清洗

### 问题（最严重）

RFT Stage 1 持续 0% 编译通过率。深入分析发现 **76% 的 gfx950 训练样本**
在 assistant 代码中使用了 gfx950 硬件上不存在的特性：

```
2741 个 gfx950 相关样本中, 2071 个 (76%) 有错配:
  × mxfp4 (FP4):    1051 条 — FP4 是 gfx1250 (MI450) 独有
  × tdm (TDM ops):   591 条 — TDM 是 gfx1250 独有
  × wmma (WMMA):     429 条 — gfx950 用 MFMA, 不用 WMMA
```

### 根因追溯

数据管线 `process_all_v2.py` 的 `augmentation_hardware` 逻辑有 bug：
把 gfx1250 kernel（含 WMMA/TDM/mxfp4）原封不动改个硬件名适配为 gfx950，
但没有去掉 gfx1250 独有的特性。

```
来源分析:
  augmentation_hardware:     415 条错误 ← 最多
  augmentation_tile:         354 条
  augmentation_pipeline:     226 条
  kernel_reverse_annotation: 159 条

实际来源文件:
  kernels/wmma_gemm_gfx1250.py         → 标记为 gfx950 (错!)
  kernels/moe_gemm_2stage_wmma_gfx1250.py → 标记为 gfx950 (错!)
  kernels/gemm_fp8fp4_gfx1250.py       → 标记为 gfx950 (错!)
```

### Hardware-Feature 兼容性矩阵

| Feature | gfx942 (MI300X) | gfx950 (MI350X) | gfx1250 (MI450) |
|---------|:---:|:---:|:---:|
| MFMA | ✅ | ✅ | ❌ |
| WMMA | ❌ | ❌ | ✅ |
| TDM | ❌ | ❌ | ✅ |
| mxfp4 (FP4) | ❌ | ❌ | ✅ |
| preshuffle | ✅ | ✅ | ❌ |
| swizzle_xor16 | ✅ | ✅ | ❌ |
| pipeline | ✅ | ✅ | ✅ |
| SmemAllocator | ✅ | ✅ | ✅ |
| split_k | ✅ | ✅ | ✅ |

### 解决

丢弃所有 assistant 代码含 hardware-feature 错配的 SFT 样本：

```
第一版 regex (过宽):  3096 → 1532 (丢弃 50%) — 误杀 812 条
  \bfp4\b 匹配了注释中的 "FP4" 文字 (如 "MFMA + FP4/MFMA-scale")
  \bwmma\b 匹配了 arch dispatch 代码中的 WMMA 分支

第二版 regex (精准):  3096 → 2344 (丢弃 24%) — 只匹配代码调用
  wmma_\w+( / WmmaAtom / from.*wmma import
  tdm_ops. / TdmCopy / tdm_load
  mxfp4_quant / fp4_gemm / wfp4 / afp4

脚本: experiments/flydsl-agent/dataprocess/clean_hw_features.py
```

### v4 候选错误详细分析

RFT Stage 1 用 v4 模型生成 192 候选，错误分布：

| 错误类型 | 数量 | 占比 | 说明 |
|---------|------|------|------|
| **语法错误** | 174 | 90.6% | |
| └ markdown 未清理 | 98 | 51% | 代码包裹在 ``` 中 |
| └ invalid syntax | 110 | 57% | 含注释中的非法字符 |
| └ unterminated string | 18 | 9% | |
| **无 @flyc 装饰器** | 7 | 3.6% | 有 import 但没写 kernel |
| **正确结构但 API 错** | 6 | 3.1% | 见下方案例 |
| **import 仍然错误** | 3 | 1.6% | v3 修复大部分,仍有残余 |
| **部分正确** | 2 | 1.0% | |
| **✅ 编译通过** | **1** | **0.5%** | paged_attn/gfx950 |

#### 语法错误详解 (174 个, 90.6%)

最大问题是**模型输出包含 markdown code block**（104/192 = 54%）:
```
模型输出:
  ```python
  import flydsl.compiler as flyc
  ...
  ```

verify.py 收到的:
  ```python           ← 这行导致 SyntaxError
  import flydsl.compiler as flyc
```

verify 脚本已有 strip markdown 逻辑（v2 修复），但匹配不到所有格式。
部分候选的 markdown 嵌套在输出中间而不是开头。

#### 正确结构但 API 错误 (6 个, 3.1%)

案例 1 — 幻想的 API:
```python
import flydsl.compiler as flyc  # ✓ 正确
import flydsl.expr as fx        # ✓ 正确
from flydsl.allocators import SmemAllocator  # × 不存在

@flyc.kernel(
    grid_dim=(fx.ceil_div(fx.Arg("B"), 8), ...),  # × fx.Arg 不存在
    smem_bytes=fx.Arg("T") * 4 * 32,              # × fx.Arg 不存在
)
def rmsnorm(
    x: fx.DeviceArray(3, dtype=torch.float32),    # × fx.DeviceArray 不存在
```
正确 API: `fx.Tensor`, `fx.Constexpr[int]`, `from flydsl.utils.smem_allocator import SmemAllocator`

案例 2 — 混入 JAX:
```python
import flydsl.compiler as flyc  # ✓
import flydsl.expr as fx        # ✓
import jax                      # × FlyDSL 不用 JAX
import jax.numpy as jnp         # × 幻觉
```

案例 3 — 幻想的类型系统:
```python
import flydsl.compiler as flyc  # ✓
import flydsl.expr as fx        # ✓
import flydsl.types as ft       # × flydsl.types 不存在

@flyc.kernel
def softmax_fma(x: ft.Tensor, y: ft.Tensor):  # × ft.Tensor 不存在
```
正确: `x: fx.Tensor`

#### 通过编译的唯一候选

```python
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.kernel
def paged_attn_fwd(
    q: fx.Tensor,
    k: fx.Tensor,
    v: fx.Tensor,
    logsumexp: fx.Tensor,
    y: fx.Tensor,
    num_heads: int,
    split_k: int,
):
    num_seqs = q.layout[0]
    ...
```
正确使用了 `import flydsl.compiler as flyc`, `import flydsl.expr as fx`, `@flyc.kernel`, `fx.Tensor`。

### v4 结果

| 指标 | v2 | v4 | Δ | 原因 |
|------|-----|-----|-----|------|
| Overall | 72.4% | 60.3% | -12.1% | 数据量减少 |
| L1 | 100% | 100% | = | |
| L2 | 100% | 80% | -20% | layernorm/rope 退化 |
| L3 | 76% | 76% | = | |
| L4 | 49% | 26% | -23% | FP8 GEMM 从 49%→14% |
| L5 | 38% | 20% | -18% | preshuffle 退化 |
| @flyc.kernel | 92% | 64% | -28% | 复杂 kernel 样本减少 |
| Pipeline | 12% | 0% | -12% | pipeline 样本被清洗掉 |
| Swizzle | 4% | 0% | -4% | swizzle 样本被清洗掉 |
| **沙箱编译** | **0%** | **0.5%** | **+0.5%** | **首次突破！** |

---

---

## v5: 补充 gfx950 kernel + Gluon 教程 + 二级幻觉发现

### 改进

1. **+156 条真实 gfx950 kernel 代码** — 从 FlyDSL 仓库 27 个 gfx950 兼容 kernel 生成 52 对 ×3
2. **+18 条 no-markdown 模板** — 强调 "output raw Python, no markdown"
3. **+18 条 API 类型纠错** — `fx.DeviceArray`→`fx.Tensor` 等 9 种纠错 ×2
4. **+60 条 Gluon GEMM 优化教程** — v0→v9 渐进式优化 (520→1489 TFLOPS)

### Benchmark 结果

| 指标 | v4 | **v5** | Δ |
|------|-----|--------|-----|
| L1 | 100% | **100%** | = |
| L2 | 80% | **100%** | +20% |
| L3 | 76% | **72%** | -4% |
| L4 | 26% | **37%** | +11% |
| L5 | 20% | **38%** | +18% |
| **Overall** | **60%** | **69%** | **+9%** |

v5 整体大幅改善：L2 恢复到 100%，L4 +11%，L5 +18%。

### 沙箱编译 0% — 双重根因分析

**原始分析 (clean_code 修复前)**：192 候选中仅 18 个通过静态检查，其余因语法错误失败。

**clean_code 修复后重新分析**：

```
语法错误根因拆解 (192 候选):
  Special token 泄漏:     126 (66%) ← <|fim_middle|>, <|file_sep|> 未被 strip
  Markdown 包裹:           54 (28%) ← ``` 包裹
  代码截断:                19 (10%) ← max_new_tokens=4096 不够，括号未闭合
  System prompt 泄漏:       5 (3%)  ← 模型把 system prompt 当代码输出

修复 clean_code() 后:
  增强正则: strip <|fim_*|>, <|file_sep|>, <|endoftext|>, <|im_start/end|>
  增强 markdown: strip 中间位置的 ```
  截断处理: 遇到残余 <| 时截断到该位置

效果:
  语法通过: 18/192 (9%) → 64/192 (33%)  ← +46 候选
  沙箱通过: 0/18 → 0/64               ← 全部因 ImportError 失败
```

**关键发现：语法问题可以在代码层面解决，不需要重新训练。**

沙箱编译失败的真正瓶颈是**二级 API 幻觉** — 64 个语法正确的候选全部在 `from flydsl.expr import Expr` 等不存在的 import 上失败：

```
模型生成的代码（顶层 import 正确，二级 import 幻觉）:

  import flydsl.compiler as flyc    # ✓ 正确 
  import flydsl.expr as fx          # ✓ 正确
  from flydsl.expr import Expr      # × 不存在！应用 fx.Tensor
  from flydsl.expr import ArithOp   # × 不存在！应用 from flydsl.expr import arith
  from flydsl.expr import dtypes    # × 不存在！类型是 fx.Float32 等直接属性
  from flydsl.utils import Layout   # × Layout 在 fx.Layout，不在 utils

根因: 模型不知道 FlyDSL 的模块结构树:
  flydsl/
  ├── compiler (flyc) — @flyc.kernel, @flyc.jit
  ├── expr (fx) — 222 个公开名 (类型/函数/布局代数)
  │   ├── arith — 算术运算 (addf, mulf...)
  │   ├── buffer_ops — 缓冲区操作
  │   ├── rocdl — ROCDL/MFMA 指令
  │   └── typing — T, Vector
  ├── utils.smem_allocator — SmemAllocator, SmemPtr
  └── runtime.device — get_rocm_arch

模型知道要从 flydsl.expr 导入东西，但不知道哪些名字存在、哪些不存在。
需要教会模型整棵 import 链条树——"需要 X → 在哪个模块 → 怎么导入"。
```

**代码修复** (已完成，不需重训):
- `verify_candidates.py::clean_code()` — 增强 special token / markdown / 截断处理
- `generate_candidates.py` — `max_new_tokens` 4096 → 6144

脚本: `experiments/flydsl-agent/rft-stage1/verify_candidates.py`

### 解决方案（v5b 方向）

已生成 import 链条导航数据（25 对 ×4 = 100 条），教模型三种推理：
1. **正向导航**: "我需要 SmemAllocator" → flydsl → utils → smem_allocator → SmemAllocator
2. **模块参考卡**: "flydsl.expr 有什么？" → 完整列表 (222 个名字)
3. **反向否定**: "flydsl.expr.Expr 不存在" → 应该用 fx.Tensor

脚本: `experiments/flydsl-agent/dataprocess/fix_import_chain.py`

---

---

## v5b: Import 链条导航 + API 参考 — 沙箱编译突破

### 改进

1. **+100 条 import 链条导航** — "需要 X → 哪个模块 → 怎么 import"（25 对 ×4）
2. **+96 条 API 参考 + 幻觉纠错** — 模块完整 API 列表 + 17 种错误 import 纠正（32 对 ×3）
3. **clean_code() 代码修复** — 增强 special token / markdown strip（不需重训）
4. **max_new_tokens 4096→6144** — 避免长 kernel 截断

### Benchmark 结果

| 指标 | v5 | **v5b** | Δ | Target |
|------|-----|---------|------|--------|
| L1 | 100% | **100%** | = | 90% ✅ |
| L2 | 100% | **100%** | = | 85% ✅ |
| L3 | 72% | **76%** | +4% | 70% ✅ |
| L4 | 37% | **51%** | +14% | 50% ✅ |
| L5 | 38% | **55%** | +17% | 20% ✅ |
| **Overall** | **69%** | **76.5%** | **+7.5%** | **60% ✅** |

**全 5 级首次全部达标。** L4 从 37%→51% 首次超过 50% 目标。

### 沙箱编译: 0% → 4.7% (9/192)

9 个 kernel 通过 FlyDSL 真实编译，覆盖 9/12 算子：

| 算子 | 通过数 | 代码行数 | import 模式 |
|------|--------|---------|------------|
| layernorm | 1 | 53 | flyc + fx + arith,buffer_ops + SmemAllocator |
| gemm | 1 | 97 | flyc + fx + flyrt |
| mla | 2 | 90, 116 | flyc + fx + arith,buffer_ops |
| softmax | 1 | 34 | flyc + fx |
| flash_attn | 1 | 65 | flyc + fx |
| topk | 1 | 29 | flyc + fx |
| custom | 1 | 75 | flyc + fx + typing.Tensor |
| paged_attn | 1 | 118 | flyc + fx + arith,buffer_ops,rocdl |

未通过的算子: moe (0), rmsnorm (0), quant (0), rope (0)

### 剩余错误分析

```
192 候选错误分布 (v5b):
  语法错误 (clean_code 后仍有):  98 (51%)  ← 需要改进生成阶段
  静态通过但沙箱 ImportError:    85 (44%)  ← 新变种幻觉
  沙箱编译通过:                   9 (4.7%) ← 突破!

新变种幻觉 (85 个候选):
  5x  from flydsl.expr import fx              → 应该用 import flydsl.expr as fx
  5x  import flydsl.expr.ops as ops           → flydsl.expr.ops 不存在
  6x  from flydsl.expr.types import F16/BF16  → flydsl.expr.types 不存在, 用 fx.Float16
  2x  from flydsl.expr import dtypes          → 老问题残留
  2x  from flydsl.runtime import rocdl        → rocdl 在 flydsl.expr.rocdl
  2x  from flydsl.utils import div_up         → div_up 不在 utils
```

### v5c 方向

1. **生成阶段修复**: 加 stop tokens 避免 special token 泄漏, 减少 51% 语法错误
2. **新变种纠错数据**: 针对 `flydsl.expr.types`, `flydsl.expr.ops`, `from flydsl.expr import fx` 加纠错
3. **增大候选数 N=32**: 提高每 spec 的编译通过概率

---

## 各版本关键教训

| 版本 | 教训 |
|------|------|
| v1 | 训练数据的内容比例直接决定模型能力方向 |
| v2 | LoRA rank 和 seq_length 是硬约束，不够就是不够 |
| v3 | 模型能学会概念但不一定学会正确的调用路径 — 需要专门的纠错数据 |
| v4 | 错误数据比没有数据更有害 — 76% 错配数据教会了模型不可能的代码模式 |
| v4 | 清洗后数据量减半导致高级能力退化 — 需要用正确数据补回 |
| v5 | 幻觉分层次：v3 修了一级 import 路径，v5 发现还有二级 API 名幻觉 |
| v5 | 模块结构知识无法从代码示例中隐式学会——需要显式的链条导航教学 |
| v5b | Import 链条导航数据直接有效——沙箱通过率 0%→4.7%，证明显式教模块结构可行 |
| v5b | 幻觉是长尾问题——修一批旧的（Expr/dtypes）会冒出新的（expr.types/expr.ops） |
| v5b | 语法错误中 66% 是后处理问题，可以在代码层面修复不需要重训 |

---

## v5c: 完整模块结构摘要 (30 模块) — 沙箱编译再次突破

### 改进

1. **+111 条完整模块结构摘要** — 覆盖全部 30 个 FlyDSL 模块（compiler/*, expr/*, runtime/*, utils/*）
   - 12 条 kernel 生成（system prompt 嵌入完整模块树）×3 重复
   - 5 条模块 QA 参考（per-module API 列表）×3 重复
   - 20 条负例（高频幻觉路径显式否定）×3 重复
2. **MODULE_DIGEST 扩展** — 从 flydsl.expr 扩展到 compiler/expr/runtime/utils 全部子模块
3. **"DOES NOT EXIST" 列表扩展** — 20 条高频幻觉路径（含 `from flydsl.expr import fx`, `flydsl.expr.types` 等）

### Benchmark 结果

| 指标 | v5b | **v5c** | Δ | Target |
|------|------|---------|------|--------|
| L1 | 100% | **100%** | = | 90% ✅ |
| L2 | 100% | **100%** | = | 85% ✅ |
| L3 | 76% | **80%** | +4% | 70% ✅ |
| L4 | 51% | **43%** | -8% | 50% ❌ |
| L5 | 55% | **45%** | -10% | 20% ✅ |
| **Overall** | **76.5%** | **73.6%** | **-2.9%** | **60% ✅** |

L3 提升至 80%，但 L4/L5 退化。模块摘要数据可能稀释了高级 kernel 样本的权重。

### 沙箱编译: 4.7% → 10.4% (20/192)

**沙箱编译通过率翻倍！** 20 个 kernel 通过编译，覆盖 9/12 算子：

| 算子 | v5b | **v5c** | Δ |
|------|-----|---------|---|
| gemm | 1 | **3** | +2 |
| layernorm | 1 | **3** | +2 |
| flash_attn | 1 | **3** | +2 |
| paged_attn | 1 | **3** | +2 |
| mla | 2 | **2** | = |
| moe | 0 | **2** | +2 |
| topk | 1 | **2** | +1 |
| softmax | 1 | **1** | = |
| custom | 1 | **1** | = |
| rmsnorm | 0 | **0** | = |
| quant | 0 | **0** | = |
| rope | 0 | **0** | = |

### 剩余错误分析 (172 失败候选)

#### 静态失败 (53/192, 27.6%)

| 原因 | 数量 |
|------|------|
| 语法错误 | 50 |
| 代码过短 (<15 行) | 3 |

#### 沙箱失败 — 详细分类 (119/192, 62%)

**类别 1: Import 写法错误 (42 处)**

| 错误模式 | 次数 | 正确写法 |
|---------|------|---------|
| `from flydsl.expr import fx` | 17 | `import flydsl.expr as fx` |
| `from flydsl.compiler import flyc` | 7 | `import flydsl.compiler as flyc` |
| `flydsl.expr.func` (不存在) | 6 | 无此模块 |
| `from flydsl import X` | 4 | `import flydsl.compiler as flyc` |
| `flydsl.expr._expr` (不存在) | 3 | 无此模块 |
| `flydsl.expr.ops` (不存在) | 3 | `from flydsl.expr import arith` |
| `flydsl.runtime.rocm` (不存在) | 1 | `from flydsl.expr import rocdl` |
| `flydsl.utils.gemm_test_utils` (不存在) | 1 | 无此模块 |

**类别 2: API 名称幻觉 (97 处，最严重)**

| 错误模式 | 次数 | 正确用法 |
|---------|------|---------|
| `flyc.kernel_context` | 13 | flyc 只有 `.kernel/.jit/.compile` |
| `from flydsl.expr import types` | 6 | 类型是 fx.* 直接属性 |
| `from flydsl.expr import expr` | 4 | 不存在 |
| `from flydsl.expr import utils` | 4 | 不存在 |
| `from flydsl.expr import dtypes` | 4 | 用 fx.Float32 等 |
| `from flydsl.expr import memory` | 3 | 不存在 |
| `from flydsl.expr import atomics` | 3 | 不存在 |
| `from flydsl.expr import type_traits` | 3 | 不存在 |
| `flyc.SmemAllocator` | 3 | `from flydsl.utils.smem_allocator import SmemAllocator` |
| `flyc.get_shared_memory` | 2 | 不存在 |
| `flyc.launch` | 2 | 不存在 |
| 其他 (`enums`, `ir`, `f32`, `smem`) | 各2 | 不存在 |

**类别 3: 外部依赖/其他 (10 处)**

| 错误模式 | 次数 |
|---------|------|
| 使用 flyc.* 但未 import | 6 |
| `import jax` | 3 |
| `import triton` | 1 |

### 核心结论

1. **模块摘要有效** — 沙箱编译率 4.7%→10.4%（翻倍），证明教模型模块结构可行
2. **新幻觉不断涌现** — 修一批旧幻觉（types/ops），冒出新幻觉（atomics/enums/type_traits/kernel_context）
3. **两大核心错误模式未解决**:
   - `from flydsl.expr import fx` (17次) — 应为 `import flydsl.expr as fx`
   - `from flydsl.compiler import flyc` (7次) — 应为 `import flydsl.compiler as flyc`
4. **flyc API 幻觉** — 模型认为 flyc 有 `kernel_context`/`SmemAllocator`/`launch` 等属性

### v5d 方向

1. **扩展 DOES NOT EXIST 列表** — 加入 v5c 新发现的所有幻觉名称
2. **增加正例重复** — 更多 `import flydsl.expr as fx` / `import flydsl.compiler as flyc` 的正确用法
3. **flyc API 纠错** — 明确教 flyc 只有 `.kernel/.jit/.compile` 三个属性
4. **Import 写法纠错** — 专门针对 `from X import Y` vs `import X as Y` 的模式

---

## v5d: 扩展负例 + 纠错 — 打地鼠失败

### 改进

1. **负例扩展 20→58 条** — 新增 atomics/enums/type_traits/func/ir/kernel_context/rocm 等
2. **Import 纠错 +7 对** — `from flydsl.expr import fx` → `import flydsl.expr as fx` 等
3. **flyc API 边界 +QA** — 明确 flyc 只有 kernel/jit/compile
4. **正确 kernel 骨架 +6** — gemm/softmax/rmsnorm/rope/flash_attn/topk 完整代码

### 沙箱结果: 19/192 (9.9%)

| 算子 | v5c | v5d | Δ |
|------|-----|-----|---|
| rmsnorm | 0 | **4** | +4 ✅ |
| rope | 0 | **2** | +2 ✅ |
| quant | 0 | **1** | +1 ✅ |
| topk | 2 | **3** | +1 |
| paged_attn | 3 | 2 | -1 |
| gemm | 3 | 1 | -2 |
| layernorm | 3 | 1 | -2 |
| flash_attn | 3 | 1 | -2 |

三个之前完全不通过的算子（rmsnorm/rope/quant）首次通过，但其他算子退化。

### 关键发现 — 打地鼠效应

| 幻觉模式 | v5c | v5d | 说明 |
|---------|-----|-----|------|
| `flyc.kernel_context` | 13 | **0** | 消灭 ✅ |
| `flyc.load` | 0 | **16** | 新冒出 ❌ |
| `flyc.barrier` | 0 | **3** | 新冒出 ❌ |
| `from flydsl.expr import kernel` | 0 | **5** | 新冒出 ❌ |
| `from flydsl.expr import fx` | 17 | **17** | 顽固不变 ❌ |

**结论：负例策略本质是打地鼠——消灭一个已知幻觉，模型就发明一个新的。**
`from flydsl.expr import fx` (17次) 完全不受纠错数据影响，说明模型偏好
`from X import Y` 这个通用 Python 模式，数据量不够无法翻转。

### 策略转向 (v5e)

不再增加负例（无底洞），改为大幅提升正确 kernel 代码的占比：
- 提取 247 个去重正确 kernel，各 3 种 prompt 变体 → +741 条
- 8 个 mini-kernel 模板 ×5 → +40 条（短而精准，聚焦正确 import）
- flyc 边界 QA ×3 → +6 条
- 正确 kernel 比例：35% → **52%**

脚本: `experiments/flydsl-agent/dataprocess/boost_correct_kernels.py`

---

## 各版本关键教训

| 版本 | 教训 |
|------|------|
| v1 | 训练数据的内容比例直接决定模型能力方向 |
| v2 | LoRA rank 和 seq_length 是硬约束，不够就是不够 |
| v3 | 模型能学会概念但不一定学会正确的调用路径 — 需要专门的纠错数据 |
| v4 | 错误数据比没有数据更有害 — 76% 错配数据教会了模型不可能的代码模式 |
| v4 | 清洗后数据量减半导致高级能力退化 — 需要用正确数据补回 |
| v5 | 幻觉分层次：v3 修了一级 import 路径，v5 发现还有二级 API 名幻觉 |
| v5 | 模块结构知识无法从代码示例中隐式学会——需要显式的链条导航教学 |
| v5b | Import 链条导航数据直接有效——沙箱通过率 0%→4.7%，证明显式教模块结构可行 |
| v5b | 幻觉是长尾问题——修一批旧的（Expr/dtypes）会冒出新的（expr.types/expr.ops） |
| v5b | 语法错误中 66% 是后处理问题，可以在代码层面修复不需要重训 |
| v5c | 模块结构摘要让沙箱编译率翻倍 (4.7%→10.4%)，但幻觉是无底洞 |
| v5c | 核心 import 模式 (`import X as Y` vs `from X import Y`) 需要大量重复正例来覆盖 |
| v5d | 负例策略是打地鼠——消灭旧幻觉(kernel_context)，冒出新幻觉(flyc.load/barrier) |
| v5d | `from flydsl.expr import fx` 顽固不变——模型偏好通用 Python 模式，需用正例洪水翻转 |
| **v5e** | **正例洪水有效——kernel 比例 35%→52% 让沙箱编译率从 9.9% 翻倍到 21.9%** |
| **v5e** | **`from flydsl.expr import fx` 仍顽固 (17次)，但已证明这些候选即使修了 import 也全因其他幻觉失败——SFT 到此天花板** |
| **v5e** | **21.9% 沙箱通过率为 RFT 提供充足正向信号，SFT 阶段完成** |

---

## v5e: 正例洪水策略 — SFT 阶段最终版

### 改进

策略转变：从穷举负例（打地鼠）转为大幅增加正确 kernel 代码占比。

1. **+741 条 boosted kernel** — 247 个去重正确 kernel ×3 种 prompt 变体
2. **+40 条 mini-kernel** — 8 个精简模板 ×5 重复（聚焦正确 import block）
3. **+6 条 flyc 边界 QA** — 明确 flyc 只有 kernel/jit/compile
4. **+270 条模块摘要** — 沿用 v5d 的完整负例列表 + QA

数据集：2832 (v5b base) + 787 (boost) + 270 (module digest) = **3889 条**
正确 kernel 比例：35% → **52%**

### Benchmark 结果

| 指标 | v5b | v5c | v5d | **v5e** | Target |
|------|------|------|------|---------|--------|
| L1 | 100% | 100% | — | **100%** | 90% ✅ |
| L2 | 100% | 100% | — | **100%** | 85% ✅ |
| L3 | 76% | 80% | — | **72%** | 70% ✅ |
| L4 | 51% | 43% | — | **49%** | 50% ❌ (差1%) |
| L5 | 55% | 45% | — | **50%** | 20% ✅ |
| **Overall** | **76.5%** | **73.6%** | — | **74.1%** | **60% ✅** |

### 沙箱编译: 9.9% → 21.9% (42/192)

**沙箱编译率翻倍再翻倍！** 覆盖 **11/12 算子**：

| 算子 | v5c | v5d | **v5e** |
|------|-----|-----|---------|
| mla | 2 | 1 | **6** |
| moe | 2 | 1 | **5** |
| softmax | 1 | 1 | **5** |
| rope | 0 | 2 | **5** |
| custom | 1 | 1 | **5** |
| gemm | 3 | 1 | **4** |
| layernorm | 3 | 1 | **4** |
| topk | 2 | 3 | **3** |
| rmsnorm | 0 | 4 | **2** |
| flash_attn | 3 | 1 | **2** |
| quant | 0 | 1 | **1** |
| paged_attn | 3 | 2 | **0** |
| **总计** | **20** | **19** | **42** |

### 剩余错误分析

```
192 候选分布:
  静态通过:    134 (69.8%)  ← 比 v5d (124) 改善
  沙箱通过:     42 (21.9%)  ← v5d (19) 的 2.2 倍
  沙箱失败:     92 (47.9%)

顽固 import 模式 (v5c/v5d/v5e 均为 17 次):
  from flydsl.expr import fx: 17 次
  → 已验证: 即使修了这行，这 17 个候选全部因后续幻觉 (import rocdl / import arith 等) 失败
  → 结论: 不值得继续在 SFT 层面修复

flyc 属性幻觉 (新一批):
  flyc.Stage: 13, flyc.SmemAllocator: 12, flyc.stage: 7
  flyc.if_: 6, flyc.build: 5, flyc.launch: 3
  → 打地鼠效应继续：旧的 (kernel_context/load) 消灭，新的 (Stage/if_) 冒出
```

### SFT 阶段结论

**SFT 阶段到此完成**。关键数据点：
- Overall 74.1%（稳定在 73-76% 区间，继续 SFT 迭代收益递减）
- 沙箱编译 21.9%（超过 10% 目标的 2 倍，为 RFT 提供充足正向信号）
- 11/12 算子覆盖
- 幻觉问题已触及 SFT 天花板——需要 RL（编译反馈）来进一步突破

**下一步**: 转入 RL 阶段 (RFT → DAPO)
- Stage A: 用 v5e 模型生成 N=16 候选 → 沙箱验证 → diversity-preserving RFT
- Stage B: Single-Turn DAPO (编译/正确性 reward)
- Stage C: Multi-Turn DAPO + PrimeEcho (性能优化)

脚本: `experiments/flydsl-agent/dataprocess/boost_correct_kernels.py`

## 文件清单

| 文件 | 说明 |
|------|------|
| `sft/eval_sft.py` | 25 题 5 级 benchmark |
| `sft/train_sft.py` | FSDP2 SFT 训练器 |
| `sft/dataset.py` | SFT 数据集 + answer-only loss masking |
| `sft/config_sft.sh` | 训练超参配置 |
| `sft/run_sft.sh` | Docker 训练启动脚本 |
| `cpt/export_hf.py` | DCP→HF 格式导出 |
| `dataprocess/enhance_sft_data.py` | v2: kernel 提取 + 重采样 |
| `dataprocess/fix_import_sft.py` | v3: import 纠错数据 |
| `dataprocess/clean_hw_features.py` | v4: hw-feature 错配清洗 |
| `dataprocess/enhance_sft_v5.py` | v5: gfx950 kernel + no-markdown + API 纠正 |
| `dataprocess/add_gluon_tutorials.py` | v5: Gluon GEMM 教程 |
| `dataprocess/fix_import_chain.py` | v5b: import 链条导航 |
| `dataprocess/fix_api_hallucination.py` | v5b: API 参考 + 幻觉纠错 |
| `dataprocess/add_module_digest.py` | v5c/v5d: 完整模块结构摘要 (30 模块) |
| `dataprocess/boost_correct_kernels.py` | v5e: 正例洪水 (247 kernel ×3 + 8 mini ×5) |
| `rft-stage1/generate_candidates.py` | RFT 候选生成器 |
| `rft-stage1/verify_candidates.py` | 沙箱验证器 |
| `sandbox/Dockerfile` | FlyDSL-Gym 沙箱 |
