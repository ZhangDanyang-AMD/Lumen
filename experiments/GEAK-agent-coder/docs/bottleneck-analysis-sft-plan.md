# perf-turns → GEAK OPUS SFT 完整方案

## 1. 背景与目标

从 `/home/danyzhan/perf-turns` 数据集中提取两种能力，通过 LoRA SFT 训练进
Qwen3-Coder-30B-A3B 模型，并扩展 GEAK 支持 OPUS backend：

1. **Bottleneck 分析能力** — 静态 ISA 级 bottleneck 诊断，输出结构化 JSON
2. **OPUS kernel 生成能力** — 根据优化方向生成 gfx950 OPUS HIP C++ 代码 patch

同时扩展 GEAK 的 OPUS backend，使 GEAK 能在 gfx950 上运行完整的 OPUS kernel
优化闭环。

**数据来源：**
- perf-turns：9 个 campaign（D380-D450），87 个 kept turn，7 种 bottleneck 类型
- 每个 campaign 有 base/new 的 ISA `.s` 文件和 kernel source（SHA256 验证）
- GEAK `perf_knowledge/` 中已有六类 bottleneck 分类体系和优化策略映射
- AITER 中的 OPUS 库：`Lumen/third_party/aiter/csrc/include/opus/opus.hpp`

---

# Part A：训练数据

## 2. Bottleneck 分析数据 — `geak_bottleneck_sft_v1`

### 2.1 Schema

**Input（模型看到的）：**

```json
{
  "kernel_source": "// kz_core_v2_s2f_AW4.cu 完整源码...",
  "isa_census": {
    "symbols": {
      "kz_core_k_s2f_aw4<32,64,256>": {
        "instructions": 1241,
        "mfma_scale": 160,
        "buffer_load_dwordx4": 92,
        "ds_read_b128": 142,
        "ds_write_b128": 48,
        "v_accvgpr_read_b32": 64,
        "v_accvgpr_write_b32": 32,
        "s_waitcnt": 28
      }
    },
    "resources": {
      "vgpr": 316, "agpr": 144, "sgpr": 96,
      "lds_bytes": 118272, "spills": 0
    }
  },
  "isa_excerpt": "// 相关 ISA 片段（截断到 ~2048 行）...",
  "performance_gap": {
    "base_snapshot": "d424ladder-0abbda7d",
    "gain_pct": 14.9,
    "gain_type": "hot",
    "shapes": ["#3 (128,7168,16384)"],
    "device": "gfx950"
  },
  "device": "MI355X / gfx950 / CDNA4, 256 CU, 160KB LDS/CU"
}
```

**Output（模型要生成的）— 扩展 GEAK PROFILE_SCHEMA 的结构化 JSON：**

```json
{
  "bottleneck": "LDS-bound",
  "bottleneck_subtype": "bank_conflict",
  "profiler_used": "isa_static_analysis",
  "dispatch_count": 18,
  "device": "MI355X / gfx950",
  "key_metrics": {
    "lds_pitch_bytes": 128,
    "bank_aliasing_factor": 32,
    "mfma_count": 160,
    "vgpr_count": 316,
    "spills": 0
  },
  "evidence": [
    {
      "type": "isa_pattern",
      "detail": "lds_sa 行以 pitch=128 字节存储，32 行映射到 2 个 dword bank"
    },
    {
      "type": "census_diff",
      "detail": "pad 变体仅增加 7 条地址重算指令，其余 symbol 指令完全一致"
    }
  ],
  "top_opportunities": [
    "P0: 将 LDS 行 pitch 从 128 pad 到 132 字节，打破 32-bank aliasing",
    "P1: 若 LDS 预算紧张，考虑 XOR swizzle 实现零浪费 de-aliasing"
  ],
  "shift_note": "静态 ISA 分析，无 PMC counters；分类基于结构性证据"
}
```

### 2.2 七种 Bottleneck 子类型

| 子类型 | 来源 Campaign | GEAK 六分类 | 核心证据 |
|---|---|---|---|
| `register_temporal_alias` | D380, D397 | latency-bound | AGPR staging 写入覆盖未被 MFMA 消费的 accumulator |
| `vmem_wait_serialization` | D400, D425 | latency-bound | 单个 vmcnt(0) 阻塞所有 global→LDS 传输 |
| `wave_occupancy_imbalance` | D393 | occupancy-limited | consumer wave 过少，tile size 限制并行度 |
| `lds_bank_conflict` | D438 | LDS-bound | scale panel pitch 导致 32 行 alias 到 2 个 bank |
| `redundant_kernel_launch` | D441-A | overhead-bound | 多余的 host-side SA transpose kernel |
| `wasted_partial_tile_compute` | D441-B, D444 | compute-bound | partial tile 中一半 MFMA 和 B-half load 浪费 |
| `scale_path_memory_traffic` | D444, D450 | memory-bound | 单独的 scale transpose 或 LDS scale read 开销 |

### 2.3 Bottleneck 样本数量

| 层级 | 构造方式 | 数量 |
|---|---|---|
| Tier 1 | 每 campaign 一个"诊断 base kernel" | 9 |
| Tier 2 | Before/after 对（new kernel → "已解决"或新 bottleneck） | 9 |
| Tier 3 | 多轮调查的子分析分解 | 15-20 |
| Tier 4 | Shape 变体 | ~27 |
| Tier 5 | 负例（fixed kernel → "balanced"） | 9 |
| Tier 6 | 从 `perf_knowledge/` bottleneck→lever 映射合成 | ~90 |
| Tier 7 | 模板扰动 | ~40 |
| **Bottleneck 小计** | | **~200** |

---

## 3. OPUS Kernel 生成数据 — `geak_kernel_sft_v1`（复用现有 schema）

### 3.1 数据来源

perf-turns 的 9 个 campaign 每个都有 base/new kernel snapshot（SHA256 验证），
可以直接 `diff base → new` 生成 unified diff。

| Campaign | 主文件 | diff 规模 | 优化内容 | 可用性 |
|---|---|---|---|---|
| D393 | `kz_core_v2_s2f.cu` | 95 行 | 4-consumer wave body | ✓ |
| D397 | `_kernel_template.hpp` + `_common.h` | 502 行 | AGPR alias fix | ✓ |
| D425 | `kz_core_v2_s2f_AW4.cu` | 136 行 | vmcnt ladder | ✓ |
| D438 | `kz_core_v2_s2f_AW4.cu` | 59 行 | LDS bank padding | ✓ 最干净 |
| D441 | `_kernel_template.hpp` + `_wrap.cc` | 3594 行 | N64 pruned tile | ⚠ 需拆分 |
| D444 | `_wrap.cc` + `_common.h` + 新 `.hpp` | 86 行 + 新文件 | scale fold | ✓ |
| D450 | `kz_skv1_p1a_s0.cu` | 26 行 | SA preload | ✓ |
| D380 | `kz_skv1_p1a_s2.cu → s0.cu` | 2 行 | scale 切换 | ✗ 太小 |
| D400 | `kz_skv1_p1a_s0.cu` | 0 行 kernel diff | driver-side | ✗ 无 kernel 改动 |

**可用 kernel 生成样本：7 条**（去掉 D380 太小、D400 无 kernel diff）。

### 3.2 Schema（复用 `geak_kernel_sft_v1`）

```json
{
  "schema_version": "geak_kernel_sft_v1",
  "task_type": "direction_conditioned",
  "input": {
    "contract": {
      "operator": "mxfp8_blockscale_gemm",
      "architecture": "gfx950",
      "m": 128, "n": 7168, "k": 16384,
      "input_dtype": "fp8_e4m3",
      "output_dtype": "bf16",
      "layout": "A[M,K] @ W[K,N], SA[M,K/128] uint8, SB[K/128,N] uint8",
      "language": "hip",
      "language_version": "hipcc + pinned clang",
      "backend_version": "opus 2026-10"
    },
    "parent_source": "// base kz_core_v2_s2f_AW4.cu 完整源码...",
    "direction": "S0 SA LDS bank de-aliasing: scale panel lds_sa 以 pitch G=128 存储，32 行映射到 2 个 dword bank。将物理行 pitch pad 到 132 字节打破 aliasing，保持逻辑 G 和 ABI 不变。仅改 gather store 和 consumer ds_read_u8 地址。"
  },
  "output": {
    "patch": "<unified diff: base → new>"
  },
  "labels": {
    "patch_applies": true,
    "compile_pass": true,
    "correctness_pass": true,
    "benchmark_valid": true,
    "verified_speedup": 1.149
  },
  "provenance": {
    "source_hash": "<base SHA256>",
    "patch_hash": "<diff SHA256>",
    "gpu": "gfx950",
    "gpu_sku": "MI355X",
    "campaign_id": "D438-SA-pad",
    "verify_source": "campaign_decision_gate"
  }
}
```

### 3.3 Labels 来源

| Label | 来源 | 说明 |
|---|---|---|
| `patch_applies` | CPU 验证 | `diff base new` 可逆，`patch --dry-run` 验证 |
| `compile_pass` | 已验证 | new snapshot 含编译后的 `.so`（SHA256 验证） |
| `correctness_pass` | Campaign decision gate | 通过了 Lead + verifier 验证 |
| `benchmark_valid` | Campaign timing | INDEX.md 记录了 gain |
| `verified_speedup` | Campaign gain | 如 D438: hot +14.9% = 1.149 |

Campaign 级 labels 成立是因为 base 和 new 都是 **冻结的、SHA256 验证的** snapshot，
且 campaign 经过了 decision gate（Lead + verifier + profiler 独立确认）。

### 3.4 Direction 提取

从每个 campaign 的第一个 prompt.md 中的 Lead 消息提取结构化 direction。
已确认每个 campaign 的 Lead 消息都包含明确的优化指令。

### 3.5 D441 特殊处理

D441 的 diff 达 3594 行（`_kernel_template.hpp` 大幅重写），超出 output token
预算。处理方式：
- 拆分为 Rung A（wrapper 优化，82 行）和 Rung B（template 重写）两个独立样本
- Rung B 按子文件拆分：`_common.h`（8 行）+ `_kernel_template.hpp`（截断到核心改动区域）

### 3.6 Kernel 生成样本数量

| 来源 | 数量 |
|---|---|
| Campaign 级 base→new diff | 7 |
| D441 拆分为 Rung A + Rung B | +1（原 1 条拆为 2 条） |
| **Kernel 生成小计** | **8** |

---

## 4. 数据构造管线

### Step A：手动标注（一次性，~1 天）

1. 为 9 个 campaign 各写 `ground_truth.json`（bottleneck 类型 + ISA 证据 + gain）
2. 为 7 个可用 campaign 提取 Lead direction 文本

### Step B：自动化提取脚本

| 脚本 | 功能 |
|---|---|
| `extract_census.py` | 解析 `kernels/{base,new}/*.s` → 结构化 ISA census JSON |
| `extract_isa_excerpt.py` | 按 bottleneck 类型提取相关 ISA 区域（截断 ~2048 行） |
| `extract_kernel_diff.py` | `diff base new` → unified diff，验证 `patch_applies` |
| `build_bottleneck_samples.py` | 组装 bottleneck 分析样本 |
| `build_kernel_samples.py` | 组装 kernel 生成样本 |

### Step C：合成扩展

Bottleneck 从 ~70 扩展到 ~200（Tier 6-7：perf_knowledge 合成 + 模板扰动）。
Kernel 生成保持 8 条（无法合成——需要真实 verified 代码）。

### 总训练数据

| 数据集 | 样本数 | task_type |
|---|---|---|
| 现有 GEAK kernel coding（gfx942 Triton/HIP） | 2,000 | 五类混合 |
| 新增 bottleneck 分析（gfx950） | ~200 | `bottleneck_analysis` |
| 新增 OPUS kernel 生成（gfx950） | 8 | `direction_conditioned` |
| **总计** | **~2,208** |

Bottleneck 占 ~9%，OPUS kernel 生成占 ~0.4%。

---

# Part B：训练代码改动

## 5. SFT 训练管线改动

所有训练代码位于 `/home/danyzhan/Lumen/experiments/GEAK-agent-coder/`。

### 5.1 Schema 注册 — `contracts.py`

```
文件：src/geak_agent_coder/data/contracts.py
位置：line 12, SUPPORTED_SCHEMAS

改动：
1. 添加 "geak_bottleneck_sft_v1" 到 SUPPORTED_SCHEMAS
2. 在 line 108 的 if-elif 链中添加 bottleneck schema 验证分支：
   - 要求 labels.bottleneck_correct (bool)
   - 要求 labels.evidence_grounded (bool)
   - 要求 labels.opportunities_actionable (bool)

注意：OPUS kernel 生成样本复用 "geak_kernel_sft_v1"，无需新 schema
```

### 5.2 格式化 — `formatting.py`

```
文件：src/geak_agent_coder/data/formatting.py
位置：line 121, canonical_qwen_messages() 的 schema_version 路由

改动：
添加 elif sample.get("schema_version") == "geak_bottleneck_sft_v1" 分支：
- System prompt：专用 bottleneck 分析 prompt（见下方）
- User message：JSON 序列化的 input（kernel_source + isa_census + performance_gap）
- Assistant message：output.response（结构化 JSON）

Bottleneck 专用 system prompt：
"You are an expert GPU kernel performance analyst for AMD CDNA4 (gfx950).
Given kernel source code, ISA assembly census data, and a performance gap
description, identify the primary bottleneck and return a structured JSON
analysis following GEAK PROFILE_SCHEMA."
```

### 5.3 数据配置 — YAML

```
文件：configs/data/qwen3_30b_a3b_with_opus.yaml（新建，基于 qwen3_30b_a3b_full2000.yaml）

添加两个数据源：
1. bottleneck_train:
   type: local_jsonl
   path: /home/danyzhan/geak_sft_dataset/bottleneck-analysis-v1/processed/train.jsonl

2. opus_kernel_train:
   type: local_jsonl
   path: /home/danyzhan/geak_sft_dataset/opus-kernel-v1/processed/train.jsonl
```

### 5.4 采样 — `sampling.py`

无需改动。现有 `_stratum` 函数按 `(sample_domain, lane, task_type, implementation_family_id)`
分层，新数据自动获得独立 stratum。

---

# Part C：GEAK 改动

## 6. GEAK OPUS Backend 扩展

### 6.1 核心发现：GEAK 架构对扩展友好

GEAK **不硬编码** 编译器或语言。编译/验证/benchmark 由 **task dir（oracle）** 定义：
- `baseline_src/` — 冻结 baseline 实现
- `unittest.py` — correctness 脚本
- `meta.json` — contract、shapes、baseline_callable
- `COMMANDMENT.md` — benchmark_engineer 生成的规则

`kernel_lane.js` 的核心逻辑不需要修改。

### 6.2 创建 OPUS Task Dir

```
位置：/home/danyzhan/phase1_control/fused-canonical/candidates/opus-mxfp8-blockscale-gemm/

需要创建的文件：

1. baseline_src/
   └── kz_core_v2_s2f_AW4.cu     ← 从 perf-turns D438 base snapshot 复制
   └── kz_skv1_p1a_s0.cu          ← t16 body
   └── gemm_a8w8_mxfp8_scale_*.hpp/cc  ← template 文件
   └── Makefile                    ← 用 pinned clang 编译

2. unittest.py
   ← 改写 perf-turns 的 kz_d374.py scorer 为 GEAK 标准格式
   功能：
   - 加载 E4M3 A/W + uint8 SA/SB 测试数据
   - 跑 candidate kernel
   - 对比 torch reference (dequant → fp32 matmul → bf16)
   - W14 tolerance 验证
   - 输出 GEAK 标准 correctness JSON

3. meta.json
   {
     "operator": "mxfp8_blockscale_gemm",
     "architecture": "gfx950",
     "language": "hip",
     "backend": "opus",
     "shapes": [
       {"M": 128, "N": 7168, "K": 16384, "label": "#3"},
       {"M": 128, "N": 4096, "K": 7168,  "label": "#4"},
       {"M": 128, "N": 2048, "K": 7168,  "label": "#1"},
       {"M": 128, "N": 2176, "K": 7168,  "label": "#9"}
     ],
     "dtype": {
       "input": "fp8_e4m3", "weight": "fp8_e4m3",
       "scale_a": "uint8_mxfp8", "scale_b": "uint8_mxfp8",
       "output": "bf16"
     },
     "baseline_callable": "kz_core_k_s2f",
     "build_cmd": "make -j CXX=/path/to/pinned-clang++ OPUS_INCLUDE=/path/to/opus",
     "modifiable_files": ["kz_core_v2_s2f_AW4.cu"]
   }
```

### 6.3 Build 环境

```
在 gfx950 MI355X 机器上准备：

1. Pinned clang（perf-turns 使用的版本）
   路径：/mnt/.../toolchain/pin-agpr/build/bin/clang++

2. OPUS headers
   来源：Lumen/third_party/aiter/csrc/include/opus/opus.hpp
   安装到：/path/to/opus/include/

3. HIP runtime
   来源：机器上的 ROCm 安装

4. Build script — build.sh
   hipcc/clang++ --offload-arch=gfx950 -I/path/to/opus \
     kz_core_v2_s2f_AW4.cu -o libkz.so -shared -fPIC
```

### 6.4 Catalog 注册

```
文件：/home/danyzhan/phase1_control/fused-canonical/catalog.yaml

添加条目：
- id: phase1-hip-gfx950-opus-mxfp8-blockscale-gemm
  type: opus_kernel
  kernel_path: /home/danyzhan/phase1_control/fused-canonical/candidates/opus-mxfp8-blockscale-gemm
  direction: 'Optimize an OPUS MXFP8 block-scaled GEMM kernel (E4M3 A/W, UE8M0
    uint8 scales, BF16 out) on gfx950 MI355X. Use opus:: MFMA intrinsic wrappers.
    Do not call AITER, CK, or external libraries at runtime.'
  operator: mxfp8_blockscale_gemm
  backend: hip
  architecture: gfx950
  provenance:
    source: perf-turns-v3
    base_snapshot: d424ladder-0abbda7d
    license: proprietary
```

### 6.5 `kernel_lane.js` — 添加 `staticAnalysisProfile()`

```
文件：/home/danyzhan/GEAK/kernel_workflow/kernel_lane.js
位置：line 850-856（Profile 阶段，profile_engineer 之后）

添加函数 staticAnalysisProfile(kernelSource, isaPath, perfGap)：
1. 调用 census.py 解析 ISA → isa_census JSON
2. 读取 kernel source
3. 按 geak_bottleneck_sft_v1 训练格式构造 prompt
4. 调用本地 fine-tuned 模型 endpoint（/v1/chat/completions）
5. 解析响应、校验 PROFILE_SCHEMA
6. 返回 profileSummary 对象

触发条件（line ~855 之后）：
  if (!profileSummary && isaExists) {
    profileSummary = await staticAnalysisProfile(kernelSource, isaPath, perfGap);
    log(`Static ISA analysis: ${profileSummary?.bottleneck || '?'}`);
  }

输出直接被后续代码消费：
  - line 878: BOTTLENECK = profileSummary.bottleneck
  - line 958: history.bottleneck_now = profileSummary.bottleneck
  - optimize loop 中的 direction 生成
```

### 6.6 不需要改的部分

| 组件 | 原因 |
|---|---|
| `kernel_lane.js` 核心 orchestration | task-dir-driven，不硬编码语言 |
| verify/profile agent 逻辑 | 从 task dir 推导命令 |
| SFT 数据 schema (`geak_kernel_sft_v1`) | OPUS kernel 样本复用现有 schema |
| 采样策略 | 自动按 stratum 分配 |

---

# Part D：验证

## 7. 验证策略

### 7.1 数据验证

1. **Schema roundtrip** — 每个样本通过 `validate_sample()` + `tokenize_sample()`
2. **Patch applies** — 7 条 kernel 样本全部 `patch --dry-run` 通过
3. **Evidence 一致性** — bottleneck 样本的 `evidence[].detail` 与 `key_metrics` 交叉验证
4. **Token 长度** — 全部样本 < 12288 tokens（max_sequence_length）

### 7.2 模型验证

5. **Bottleneck 分类准确性** — 留出 D438 + D441 作为 dev set，验证分类正确
6. **Kernel coding 无退化** — 在合并模型上跑现有 gfx942 SFT benchmark
7. **OPUS 代码质量** — 对 D438 base source 推理，检查生成的 patch 是否合理

### 7.3 GEAK 集成验证

8. **Task dir smoke** — OPUS task dir 在 gfx950 上 `unittest.py` 通过
9. **staticAnalysisProfile 测试** — D438 ISA 输入 → assert `bottleneck: "LDS-bound"`
10. **端到端 GEAK run** — 在 gfx950 上跑一轮 OPUS task，验证完整闭环

---

## 8. 执行顺序与时间估算

### Phase 1：数据构造（~3 天，无需 GPU）

```
1. 手动标注 9 个 ground_truth.json + 7 个 direction 提取      — 1 天
2. 实现 extract_census.py + extract_isa_excerpt.py             — 半天
3. 实现 extract_kernel_diff.py（base→new unified diff）        — 半天
4. 实现 build_bottleneck_samples.py → Tier 1-5（~70 条）       — 半天
5. 合成 Tier 6-7（perf_knowledge 合成 + 模板扰动，~130 条）    — 半天
6. 实现 build_kernel_samples.py → 8 条 OPUS kernel 样本        — 半天
```

### Phase 2：训练管线适配（~1 天，无需 GPU）

```
7. contracts.py 添加 geak_bottleneck_sft_v1 schema + 验证
8. formatting.py 添加 bottleneck 路由 + system prompt
9. 新建 configs/data/qwen3_30b_a3b_with_opus.yaml
10. 数据校验 + tokenize 测试（CPU）
```

### Phase 3：GEAK OPUS Backend（~1 周，需要 gfx950 机器）

```
11. 创建 OPUS task dir（baseline_src/ + unittest.py + meta.json）  — 2-3 天
12. 配置 gfx950 build 环境（pinned clang + opus headers）         — 1 天
13. catalog.yaml 注册 OPUS task                                    — 半天
14. Task dir smoke test（unittest.py 在 gfx950 上通过）            — 半天
15. kernel_lane.js 添加 staticAnalysisProfile()                   — 1 天
```

### Phase 4：训练 + 评测（需要 8×MI308X 训练机 + gfx950 评测机）

```
16. LoRA SFT 训练（~2208 条，1-2 epoch）
17. Dev set 评测（bottleneck 分类 + kernel coding 回归）
18. 端到端 GEAK run on gfx950（OPUS task 完整闭环）
```

---

## 9. 模型学到的能力总结

| 能力类型 | 具体学到的 | 在 GEAK 中的作用 |
|---|---|---|
| **ISA 静态分析** | 从 .s assembly census 提取性能信号 | Profile 阶段 fallback |
| **Bottleneck 分类** | 映射到 GEAK 六类 + 7 种子类型 | 指导 optimize loop 方向 |
| **证据链构建** | 每个诊断关联到具体 ISA 模式 | 可解释的分析结果 |
| **优化方向排序** | P0-P5 优先级建议 | research phase 的 question 生成 |
| **OPUS 代码模式** | opus:: MFMA wrapper、template 特化、LDS 优化 | engineer 生成 OPUS patch |
| **gfx950 微架构** | 160KB LDS、AGPR 管理、bank conflict pattern | 架构感知的代码生成 |
| **判断"已修复"** | 分析 new kernel 识别 bottleneck 已解决 | 优化循环停止判断 |
