# FlyDSL Agent Dataset Pipeline (v5e)

> 九轮 SFT 迭代 → Overall 74.1%, 沙箱编译 21.9% → 转入 RFT

## 1. 整体架构

```
FlyDSL + aiter + gpu-docs
         │
    ┌────┴────┐
    ▼         ▼
 manifest  gpu-docs (机密)
    │         │
    ▼         ▼
 CPT/SFT/RL  拒答 SFT
    │
    ├── v2: kernel extraction + resampling
    ├── v3: import correction
    ├── v4: hw-feature cleanup
    ├── v5/v5b: gfx950 kernels + import chains + API reference
    ├── v5c/v5d: module structure digest + negative list
    └── v5e: correct kernel boost (positive flooding)
         │
         ▼
    SFT v5e (3,889 samples, 52% kernel, sandbox 21.9%)
         │
         ▼
    RFT Stage A (100 specs × 16 candidates → verified → short SFT)
```

| 产出 | 数量 | 说明 |
|------|------|------|
| SFT v5e | 3,889 train + 264 val | 52% 正确 kernel 代码，Overall 74.1% |
| RL specs | 2,563 train + 287 val | gfx950 213 个 spec |
| Sandbox | 21.9% (42/192) | 11/12 算子通过 |

---

## 2. SFT 九轮迭代总结

### 数据质量问题与解决

| 版本 | 发现的问题 | 解决方法 | 效果 |
|------|-----------|---------|------|
| v1→v2 | 82% 非代码数据 | kernel 提取 + 加权采样 | Overall 56%→72% |
| v2→v3 | 81% import 幻觉 | 60 条 import 纠错 ×3 | val_loss 改善 |
| v3→v4 | 76% hw-feature 错配 | 丢弃 gfx1250 特性 on gfx950 | 沙箱 0%→0.5% |
| v4→v5 | 数据量减少致退化 | +156 真实 gfx950 kernel | Overall 60%→69% |
| v5→v5b | 二级 API 幻觉 | import 链条导航 + API 参考 | 沙箱 0%→4.7% |
| v5b→v5c | 模块结构未知 | 30 模块完整摘要 | 沙箱 4.7%→10.4% |
| v5c→v5d | 新幻觉不断冒出 | 扩展负例 20→58 | 打地鼠失败 (9.9%) |
| **v5d→v5e** | **负例无底洞** | **正例洪水：kernel 比例 35%→52%** | **沙箱 9.9%→21.9%** |

### 关键教训

1. **正例 > 负例** — v5d 证明穷举 "does not exist" 是打地鼠（消灭 `flyc.kernel_context`，冒出 `flyc.load`）。v5e 用正确 kernel 代码洪水（247 ×3 + mini ×5）让沙箱编译率翻倍。
2. **`from flydsl.expr import fx` 是 SFT 天花板** — 横跨 v5c/v5d/v5e 均为 17 次，模型从预训练继承的 Python 惯性模式，SFT 数据量级无法翻转。但已验证这 17 个候选即使修了 import 也全部因后续幻觉失败——说明不值得继续修。
3. **错误数据比没有数据更有害** — v4 删除了 24% 数据，关键指标反而改善。

---

## 3. v5e 数据组成

### 来源分布

| 来源 | 数量 | 说明 |
|------|------|------|
| augmentation_tile | ~266 | tile size 变体 |
| augmentation_pipeline | ~195 | pipeline 深度变体 |
| boost_correct_kernel | **741** | **247 去重正确 kernel ×3 prompt 变体 (v5e 新增)** |
| module_digest_negative | 174 | 58 种幻觉 ×3 (v5d) |
| import_fix_template/correction | ~120 | import 纠错 |
| kernel_reverse_annotation | ~97 | 代码→指令 |
| kernel_code_synthesis | ~79 | CPT 提取的 kernel |
| boost_mini_kernel | **40** | **8 个精简模板 ×5 (v5e 新增)** |
| module_digest_kernel/qa | ~57 | 模块摘要 kernel + QA |
| 其他 | ~2120 | 文档QA、gfx950 kernel、Gluon 教程等 |
| **总计** | **3,889** | |

### 正确 import 模式覆盖

```
import flydsl.compiler as flyc    — ~1800 条 assistant 中包含
import flydsl.expr as fx          — ~1800 条
@flyc.kernel                      — ~1800 条
@flyc.jit                         — ~1600 条
from flydsl.expr import arith     — ~800 条
SmemAllocator                     — ~600 条
rocdl                             — ~700 条
```

---

## 4. 沙箱编译演进

```
v2:  0/208 = 0%      — 81% import 幻觉
v4:  1/192 = 0.5%    — 首次突破
v5b: 9/192 = 4.7%    — import 链条有效
v5c: 20/192 = 10.4%  — 模块摘要翻倍 ✓ (>10% 目标)
v5d: 19/192 = 9.9%   — 负例无效
v5e: 42/192 = 21.9%  — 正例洪水再翻倍 ✓✓

按算子 (v5e):
  mla: 6, moe: 5, softmax: 5, rope: 5, custom: 5
  gemm: 4, layernorm: 4, topk: 3, rmsnorm: 2
  flash_attn: 2, quant: 1, paged_attn: 0
```

---

## 5. RFT 阶段 (进行中)

```
v5e model (sandbox 21.9%)
    │
    ▼ generate_candidates.py (84 specs × 16 = 1344 candidates)
    │
    ▼ verify_candidates.py --use-sandbox (~290 pass)
    │
    ▼ build_rft_dataset.py (merge with v5e SFT data, ×2 repeat)
    │
    ▼ train_sft.py (1 epoch, lr=5e-6, on v5e merged model)
    │
    ▼ RFT v1 model → benchmark + sandbox eval
```

目标: 沙箱编译率 > 30%

---

## 6. 脚本参考

| 阶段 | 脚本 | 职责 |
|------|------|------|
| Base | `process_all_v2.py` | 主管线：扫描→manifest→CPT/SFT/RL |
| v2 | `enhance_sft_data.py` | kernel 提取 + 加权采样 |
| v3 | `fix_import_sft.py` | import 纠错对 |
| v4 | `clean_hw_features.py` | hw-feature 错配清洗 |
| v5 | `enhance_sft_v5.py` | gfx950 kernel + API 纠正 |
| v5 | `add_gluon_tutorials.py` | Gluon GEMM 教程 |
| v5b | `fix_import_chain.py` | import 链条导航数据 |
| v5b | `fix_api_hallucination.py` | API 参考 + 幻觉纠错 |
| v5c/d | `add_module_digest.py` | 30 模块结构摘要 + 负例列表 |
| v5e | `boost_correct_kernels.py` | 正确 kernel 模式增强 |
| RFT | `rft-stage1/generate_candidates.py` | 候选生成 |
| RFT | `rft-stage1/verify_candidates.py` | 沙箱验证 |
| RFT | `rft-stage1/build_rft_dataset.py` | 构建 RFT 数据集 |

---

*v5e · 2026-06-25 · SFT 阶段完成，转入 RFT*
