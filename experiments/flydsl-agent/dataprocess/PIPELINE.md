# FlyDSL Agent Dataset Pipeline (v3.0)

> 三源数据 → 5 阶段处理 → 三阶段训练数据 + 机密保护

## 1. 整体架构

```
FlyDSL + aiter + gpu-docs
         │
    ┌────┴────┐
    ▼         ▼
 manifest  gpu-docs (§13 机密)
    │         │
    ▼         ▼
 CPT/SFT/RL  拒答 SFT (135 对)
    │
    ├── GPU benchmark → quality grades → CPT weights
    └── 5-model AI consensus → +375 SFT, +2128 RL
```

| 产出 | 数量 | 说明 |
|------|------|------|
| CPT | 1,967 docs (8.5M tokens) | 领域预训练，加权采样 |
| SFT | 2,808 train + 264 val | 指令跟随 + 拒答边界 |
| RL | 2,878 specs | GRPO 任务规格 |

---

## 2. 数据源与标注

**三个源仓库**：FlyDSL（DSL框架+kernel）、aiter（Triton kernel+ops）、gpu-docs（AMD 硬件规格）

**标注分层**：
- Layer 1（~80%）：`process_all_v2.py` 基于路径、正则、代码结构的确定性标注
- Layer 2（~15%）：5 个 AI 模型并行标注 → 多数投票共识（`consensus_annotate.py`）
- Layer 3（~5%）：模型分歧 → 标记 `needs_human_review`

**5-model 共识**：Rule-based + GPT-5.5 + Claude Sonnet + GPT Codex + Claude Opus → 69% operator 一致率

---

## 3. CPT 设计逻辑

**核心思路**：通过加权采样让模型多看高质量代码，少看低价值文件。

| 内容类型 | 权重 | 依据 |
|---------|------|------|
| Claude expert skills | 7.5x | 信息密度最高 |
| 知识图谱 CLAUDE.md | 6.0x | 仓库导航 |
| Gold kernel（GPU 验证） | 5.4x | 生产级参考代码 |
| FlyDSL 文档 | 4.5x | 官方 API |
| 框架源码 | 3.0x | 编译器+运行时 |
| Silver kernel | 2.0x | 算法参考 |
| **gpu-docs 硬件规格** | **0.5x** | **内化但不过度记忆（§13.3）** |
| 构建脚本 | 0.4x | 低优先级 |

权重公式：`priority_weight × grade_weight × type_weight`

**GPU benchmark grading**：8×MI350X 并行测试 → 755 tests pass → Gold(14)/Silver(80)/Bronze(16) 分级 → 反馈到 CPT 权重

---

## 4. SFT 设计逻辑

**核心思路**：15 种来源覆盖不同指令类型，确保模型能处理各种 kernel 编写请求。

| 来源 | 数量 | 逻辑 |
|------|------|------|
| augmentation_hardware | 766 | gfx942↔gfx950↔gfx1250 硬件适配变体 |
| documentation_qa | 669 | 文档→问答对 |
| kernel_reverse_annotation | 388 | 代码→指令（3种风格/kernel） |
| ai_annotated_instruction | 375 | 5-model 共识生成的指令 |
| **refusal_boundary** | **135** | **拒答规格提取，重定向到写 kernel（§13）** |
| augmentation_tile/pipeline | 178 | tile size / pipeline 深度变体 |
| 其他 | 297 | git history, test params, configs 等 |
| **allowed_explanation** | **12** | **防过度拒答（合法的模糊解释）** |

**拒答占比**：4.8%（目标 ~4%），确保安全边界而不损害正常能力。

---

## 5. RL 设计逻辑

**核心思路**：只给任务规格，不给参考答案。模型生成 → GPU 编译+测试 → 奖励信号。

奖励函数：`0.3 × compiles + 0.3 × correct + 0.4 × efficiency`

4 个来源：AI annotation shapes（2,128）+ manifest（440）+ tuned configs（160）+ exploration（150）

---

## 6. gpu-docs 机密保护（Plan §13）

### 设计决策

| 方案 | Kernel 质量 | 防泄露 | 选用 |
|------|-----------|--------|------|
| ~~加密训练数据~~ | 破坏（模型学不到） | ✅ | ❌ |
| ~~重度蒸馏~~ | 降级 | ✅ (常量仍泄露) | ❌ |
| **全精度 + 拒答 SFT** | **最优** | **行为+中间件** | **✅** |

### 训练时

- **CPT**：gpu-docs 全精度明文，weight 0.5x → 模型内化硬件知识，不过度记忆原文
- **SFT**：135 对拒答数据覆盖 10 种攻击模式 × 15 个硬件主题（中英文、换措辞、分步、角色扮演、编码绕过）
- **反例**：12 对合法模糊解释（"为什么用 xor16?" → 解释优化思路，不给精确数值）
- **删除**：gpu_docs_qa 2,625 对（教模型回答规格问题 = 错误行为）

### 推理时（4 层防御）

```
请求 → [A: Query Classifier] → [B: System Prompt] → [C: RAG 隔离] → 模型生成 → [D: Output Filter] → 返回
```

- **A**：正则 + 语义分类，拦截规格查询/ISA 请求/prompt injection
- **B**：每个会话注入安全约束（禁精确数值，允许代码常量和模糊解释）
- **C**：gpu-docs 不入向量索引，检索永远不会命中
- **D**：散文中精确数值替换为模糊描述，**代码块不动**

### 能力边界

| 威胁 | 能防？ | 机制 |
|------|--------|------|
| 逐字提取文档 | ✅ | 拒答训练 + classifier |
| 当规格问答机 | ✅ | 拒答训练 + classifier |
| RAG 泄露原文 | ✅ | 不在索引中 |
| 从 kernel 常量反推参数 | ❌ (可接受) | 代码即知识，固有限制 |

### 验收指标

| 指标 | 目标 |
|------|------|
| 攻击拦截率 | ≥ 95% |
| 合法请求误杀率 | ≤ 5% |
| RAG 泄露 | 0 |
| 散文数值泄露 | ≤ 2% |

---

## 7. 脚本参考

| 脚本 | 职责 |
|------|------|
| `process_all_v2.py` | 主管线：扫描→manifest→CPT/SFT/RL |
| `benchmark_filter.py` | GPU 质量分级（编译→正确性→评级） |
| `perf_benchmark.py` | 性能基准测试（延迟、TFLOPS、roofline） |
| `consensus_annotate.py` | 5-model 标注 + 投票 + 合入 |
| `rebuild_with_annotations.py` | 用 AI 标注结果重建数据集 |
| `validate_dataset.py` | 格式/内容/去重/分布校验 |
| `package_hf_dataset.py` | 打包为 HuggingFace 格式 |

---

*v3.0 · 2026-06-16 · 遵循 [plan.md](../../plan.md) §4 + §13*
