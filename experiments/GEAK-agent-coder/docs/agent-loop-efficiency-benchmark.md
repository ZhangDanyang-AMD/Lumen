# Agent Loop Efficiency Benchmark: Evaluating Model Quality in Fixed-Agent Kernel Generation

## Motivation

Given a fixed agent framework (GEAK), we want to benchmark how a **trained model** compares to a **baseline model** (e.g., Claude Code) in terms of kernel optimization efficiency. The key question: does our model not only produce better kernels, but do so **more efficiently** within the agent loop?

This document defines the metrics, experimental setup, and reporting format.

---

## 1. Result Quality Metrics (What)

These measure the final kernel quality — the standard baselines everyone reports.

| Metric | Definition | Reference |
|--------|-----------|-----------|
| **Pass@k** | Probability that at least 1 of k samples is a correct kernel | [KernelBench (Ouyang et al., 2025)](https://arxiv.org/abs/2502.10517) |
| **Fast@p** | Fraction of kernels that are correct AND achieve >p× speedup over baseline (p=1, 1.2, 1.5, 2) | [KernelBench](https://arxiv.org/abs/2502.10517) |
| **Speedup (geomean)** | Geometric mean speedup vs `torch.compile` across all correct kernels | [CUDA Agent (Dai et al., 2026)](https://arxiv.org/abs/2602.24286) |
| **Faster Rate** | Percentage of tasks where the generated kernel beats baseline | [CUDA Agent](https://arxiv.org/abs/2602.24286) |

---

## 2. Process Efficiency Metrics (How)

These are the core metrics for our evaluation — they differentiate models that achieve the same result quality but with different efficiency.

### 2.1 Turn Efficiency

How many agent loop iterations does the model need?

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| **First-pass rate** | Fraction of tasks where Turn 1 produces a correct, faster-than-baseline kernel | Zero-shot capability; higher = less reliance on iterative refinement |
| **Turns to pass** | Average number of turns to produce the first correct kernel | [Dr. Kernel (2025)](https://arxiv.org/abs/2602.05885) found turn 3 is typically optimal for baselines |
| **Turns to best** | Average number of turns to reach the maximum speedup | Optimization convergence speed |
| **Convergence rate** | Fraction of tasks where the model reaches within 95% of its best speedup by turn T | [AdaExplore (2025)](https://arxiv.org/abs/2604.16625) reports ~40% fewer iterations to converge |

### 2.2 Token Efficiency

How much compute does the model consume?

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| **Cost-of-Pass** | Total tokens (or USD) consumed per successfully resolved task | [SWE-Bench+ (2024)](https://arxiv.org/abs/2410.06992): same pass rate can cost 10× difference |
| **Total tokens per task** | Sum of input + output tokens from start to best solution | [AgentLens (2025)](https://arxiv.org/abs/2605.12925): same pass masks **40×** token cost difference |
| **Output token ratio** | Output tokens / Total tokens | Agent loops are input-dominated (>99%); higher ratio = more efficient per turn |
| **Token-normalized speedup** | Speedup achieved / Total tokens consumed | Single efficiency number |

### 2.3 Trajectory Quality

How "clean" is the model's optimization path?

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| **Compile error rate** | Fraction of turns that fail to compile | Lower = model generates syntactically valid code, doesn't waste turns |
| **Correctness error rate** | Fraction of turns that compile but produce wrong results | Lower = model understands semantics, not just syntax |
| **Monotonic improvement rate** | Fraction of consecutive turn pairs where speedup strictly increases | Higher = model has a clear optimization direction, no wasted iterations |
| **Regression rate** | Fraction of turns where performance degrades vs previous turn | Lower = model doesn't undo its own progress |
| **Lucky Pass classification** | C1 (minimal, 34K tok) / C2 (brute-force, 850K tok) / C4 (excessive, 2.6M tok) | [AgentLens](https://arxiv.org/abs/2605.12925): binary pass/fail hides massive efficiency differences |

---

## 3. Composite Metrics (Single-Number Summary)

| Metric | Definition | Advantage |
|--------|-----------|-----------|
| **Speedup-AUC** | Area under the (Speedup vs Turns) curve, normalized by max turns | One number capturing "how fast to how good"; directly sortable |
| **Pareto efficiency score** | Distance from the Pareto frontier on (Pass Rate vs Cost) plot | [Artificial Analysis (2026)](https://artificialanalysis.ai/agents/coding-agents) uses this for coding agent ranking |
| **Efficiency@k** | Maximum speedup achieved within a budget of k turns or k tokens | Controls for cost; answers "given the same budget, who gets more?" |
| **Effective TFLOPS/token** | (Kernel TFLOPS achieved) / (Tokens consumed to generate it) | Hardware-aware efficiency; useful for GPU kernel domain specifically |

---

## 4. Experimental Setup

### Fixed Variables

- **Agent framework**: GEAK (generator + evaluator + reflector + optimizer)
- **Benchmark suite**: KernelBench Level 1-3 (250 tasks) or GEAK's TritonBench-revised (214 tasks)
- **Hardware**: MI308X (or target GPU)
- **Max turns**: N (e.g., 10)
- **Baselines**: PyTorch Eager, `torch.compile`

### Independent Variable

- **Model A**: Baseline (e.g., Claude Code / Seed1.6 / Qwen-Coder)
- **Model B**: Your RL-trained model

### Per-Turn Data Collection

For each task × each turn, record:

```json
{
  "task_id": "kernelbench_L2_042",
  "model": "model_b",
  "turn": 3,
  "compiled": true,
  "correct": true,
  "speedup_vs_eager": 2.45,
  "speedup_vs_compile": 1.82,
  "tokens_input": 12840,
  "tokens_output": 1536,
  "wall_time_s": 4.2,
  "compile_error": null,
  "runtime_error": null
}
```

---

## 5. Reporting Format

### 5.1 Summary Table

| Metric | Model A (baseline) | Model B (yours) | Delta | Note |
|--------|-------------------|-----------------|-------|------|
| Pass@1 (Turn 1) | | | | Zero-shot |
| Pass@1 (Turn N) | | | | After full loop |
| Fast@1.2 | | | | Non-trivial speedup |
| Fast@2.0 | | | | Significant speedup |
| Geomean Speedup vs compile | | | | |
| Avg turns to pass | | | | Lower is better |
| Avg turns to best | | | | Lower is better |
| Cost-of-Pass (K tokens) | | | | Lower is better |
| Compile error rate | | | | Lower is better |
| Monotonic improvement rate | | | | Higher is better |
| Speedup-AUC | | | | Higher is better |

### 5.2 Visualization: Speedup vs Turns Curve

The most important visualization. Shows zero-shot capability (intercept), learning speed (slope), and ceiling (asymptote) in one plot.

```
   Speedup vs torch.compile
     ^
 2.5 |                                          ___..---""""  Model B (yours)
     |                                   _.--""
 2.0 |                             _.--""
     |                       _.--""          .....--------  Model A (baseline)
 1.5 |                 _.--""       ....-----
     |           _.--""    ....-----
 1.0 |     _.--"" ...------
     |_.--""-----                        <-- torch.compile baseline (1.0×)
 0.5 |
     |
 0.0 +----+----+----+----+----+----+----+----+----+----> Turn
     0    1    2    3    4    5    6    7    8    9   10

   Key observations from this curve:
   - Y-intercept (Turn 1)  = model's zero-shot kernel generation ability
   - Slope (Turn 1→3)      = model's ability to learn from agent feedback
   - Asymptote (Turn 8+)   = model's optimization ceiling
   - AUC (shaded area)     = composite "fast AND good" efficiency score
   - Gap between curves    = your model's advantage at each turn
```

### 5.3 Visualization: Pass Rate vs Token Cost (Pareto Plot)

Shows the accuracy-efficiency tradeoff. Upper-left is better (more correct, less cost).

```
   Pass Rate (%)
     ^
 100 |                                              * Model B (yours)
     |
  90 |                            * Model A (baseline)
     |
  80 |
     |                 * Gemini 3 Pro
  70 |
     |         * GLM 4.6
  60 |
     |   * Kimi K2
  50 |
     |
  40 +----+--------+--------+--------+--------+---------> Avg Tokens per Task (K)
          50      100      150      200      250      300

   Ideal position: upper-left (high pass rate, low token cost)
   Pareto frontier: the convex hull connecting the best tradeoff points
```

### 5.4 Visualization: Per-Turn Trajectory Quality Heatmap

Shows the distribution of outcomes at each turn across all tasks.

```
   Turn │  Compiled  │  Correct  │  Faster  │  Regressed
   ─────┼────────────┼───────────┼──────────┼───────────
     1  │ ██████░░░░ │ █████░░░░ │ ███░░░░░ │
     2  │ ████████░░ │ ███████░░ │ █████░░░ │ █░░░░░░░░
     3  │ █████████░ │ ████████░ │ ██████░░ │ █░░░░░░░░
     4  │ █████████░ │ █████████ │ ███████░ │ ░░░░░░░░░
     5  │ █████████░ │ █████████ │ ████████ │ ░░░░░░░░░

   Model B (yours):
     1  │ █████████░ │ ████████░ │ ██████░░ │
     2  │ █████████░ │ █████████ │ ████████ │ ░░░░░░░░░
     3  │ ██████████ │ █████████ │ █████████│ ░░░░░░░░░

   Key: model B reaches Turn 5 quality of model A by Turn 2
```

### 5.5 Visualization: Lucky Pass Distribution

Shows how "clean" each model's successes are.

```
                    Model A (baseline)          Model B (yours)
                    ┌─────────────────┐         ┌─────────────────┐
   C1 Minimal       │ ████░░░░░░ 35%  │         │ ████████░░ 72%  │
   (34K tok avg)    │                 │         │                 │
                    ├─────────────────┤         ├─────────────────┤
   C2 Brute-force   │ █████░░░░░ 40%  │         │ ██░░░░░░░░ 18%  │
   (850K tok avg)   │                 │         │                 │
                    ├─────────────────┤         ├─────────────────┤
   C3 Guided        │ ██░░░░░░░░ 15%  │         │ █░░░░░░░░░  8%  │
   (200K tok avg)   │                 │         │                 │
                    ├─────────────────┤         ├─────────────────┤
   C4 Excessive     │ █░░░░░░░░░ 10%  │         │ ░░░░░░░░░░  2%  │
   (2.6M tok avg)   │                 │         │                 │
                    └─────────────────┘         └─────────────────┘

   Takeaway: Model B's passes are overwhelmingly "clean" (C1),
   while Model A often brute-forces its way to a solution (C2).
```

---

## 6. Key Analysis Angles

### 6.1 Per-Level Breakdown

Report all metrics split by KernelBench Level 1/2/3 difficulty. Your model might excel at simple operators (L1) but struggle with fused ops (L2) or full architectures (L3), or vice versa.

### 6.2 Per-Operator-Type Breakdown

Group tasks by operator category (matmul, conv2d, attention, fused ops, reductions) to find the model's strengths and weaknesses.

### 6.3 Failure Mode Analysis

For tasks where the model never passes within N turns:
- What fraction are compile errors vs correctness errors vs timeout?
- Does the model get stuck in loops (same error repeated)?
- How does this compare to the baseline model's failure modes?

### 6.4 Budget-Constrained Comparison

Plot Pass@1 and Fast@1.2 as a function of turn budget (1, 2, 3, ..., N):
- At what budget does your model match baseline's full-budget performance?
- This directly translates to inference cost savings.

---

## 7. References

- [KernelBench: Can LLMs Write Efficient GPU Kernels?](https://arxiv.org/abs/2502.10517) — Ouyang et al., 2025
- [CUDA Agent: Large-Scale Agentic RL for CUDA Kernel Generation](https://arxiv.org/abs/2602.24286) — Dai et al., 2026
- [Dr. Kernel: RL Done Right for Triton Kernel Generations](https://arxiv.org/abs/2602.05885) — 2025
- [AdaExplore: Failure-Driven Adaptation for Efficient Kernel Generation](https://arxiv.org/abs/2604.16625) — 2025
- [GEAK: Triton Kernel AI Agent & Evaluation Benchmarks](https://arxiv.org/abs/2507.23194) — Wang et al., 2025
- [AgentLens: The Lucky Pass Problem in SWE-Agent Evaluation](https://arxiv.org/abs/2605.12925) — 2025
- [SWE-Bench+: Enhanced Coding Benchmark for LLMs](https://arxiv.org/abs/2410.06992) — 2024
- [EET: Experience-Driven Early Termination for Cost-Efficient SE Agents](https://arxiv.org/abs/2601.05777) — 2025
- [AI Coding Cost Analysis: Agent Token Spend](https://www.augmentcode.com/guides/ai-coding-cost-analysis-agent-token-spend) — Augment Code, 2026
- [Coding Agent Index](https://artificialanalysis.ai/agents/coding-agents) — Artificial Analysis, 2026
- [Reducing Cost of LLM Agents with Trajectory Reduction](https://conf.researchr.org/details/fse-2026/fse-2026-research-papers/137/) — FSE 2026
- [Afterburner: RL for Self-Improving Code Efficiency](https://arxiv.org/abs/2505.23387) — 2025
- [KernelBenchX: Comprehensive GPU Kernel Benchmark](https://arxiv.org/abs/2605.04956) — 2026
- [KernelSkill: Multi-Agent Framework for GPU Kernel Optimization](https://arxiv.org/abs/2603.10085) — 2025
- [Token Economics for LLM Agents](https://arxiv.org/abs/2605.09104) — 2026

---

## Appendix A: GPU Kernel Generation Benchmarks

可用于独立评测模型 kernel 生成能力的 benchmark 汇总。筛选标准：有公开代码/数据、支持接入自定义模型、自动化评测 correctness + performance。

### A.1 通用 Kernel 生成 Benchmark

#### KernelBench — 最广泛使用，即插即用

- **论文**: [KernelBench: Can LLMs Write Efficient GPU Kernels?](https://arxiv.org/abs/2502.10517) (Ouyang et al., 2025)
- **GitHub**: [ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench)
- **规模**: 250 题 (L1: 100 单算子, L2: 100 融合算子, L3: 50 完整模型架构, L4: HuggingFace 模型)
- **目标语言**: CUDA
- **任务格式**: 给 PyTorch `Model` class → 模型生成 `ModelNew` class（含自定义 CUDA kernel）
- **评测流程**: 自动编译 → correctness check (随机输入 n 次) → 计时 speedup vs PyTorch eager / `torch.compile`
- **指标**: Pass@k, Fast@p (p=1, 1.2, 1.5, 2), Speedup (geomean)
- **模型接入方式**: 使用 **litellm** 作为 LLM backend，支持 OpenAI API 兼容的任何模型（vLLM serve、SGLang、本地模型均可），配 `.env` 即跑
- **额外能力**:
  - Modal 云 GPU 评测（无本地 GPU 也能跑）
  - [kernelbench-tinker](https://github.com/ScalingIntelligence/KernelBench) RL 集成（generation → eval → reward pipeline）
  - `scripts/run_and_check.py` 可单独评测一个 kernel 文件
- **可对标论文**: CUDA Agent, Claude Opus 4.5, Gemini 3 Pro, Dr. Kernel, ConCuR 等全部主流工作
- **局限**: CUDA only，不直接支持 AMD GPU

#### KernelBenchX — Triton 专项，category-aware 细粒度分析

- **论文**: [KernelBenchX: A Comprehensive Benchmark for Evaluating LLM-Generated GPU Kernels](https://arxiv.org/abs/2605.04956) (Wang et al., 2026)
- **GitHub**: [BonnieW05/KernelBenchX](https://github.com/BonnieW05/KernelBenchX)
- **规模**: 176 题, 15 类 (Activation, Conv, Fusion, Index, LinearAlgebra, Loss, Math, MatMul, Normalization, Optimizer, Pooling, Quantization, Random, Reduce, SpatialOps)
- **目标语言**: Triton
- **评测流程**: 两阶段 correctness (compile → execution) + efficiency (runtime ms, TFLOPS, memory BW, speedup) + 代码质量 (radon maintainability index)
- **指标**: Compilation Accuracy, Call Accuracy, Execution Accuracy, Speedup, Code Quality
- **模型接入方式**: 提供 reproducible evaluation harness，支持插入自定义 generation method
- **核心发现**: task structure 比 method 影响更大（category 解释 9.4% 方差 vs method 3.3%）；iterative refinement 提升 correctness 但 speedup 反而下降
- **优势**: 按算子类别细分，能精确定位模型在哪类 kernel 上强/弱
- **AMD 兼容**: 可以（Triton 跨平台）

#### TritonBench — 真实 GitHub Triton kernel

- **论文**: [TritonBench: Benchmarking LLM Capabilities for Generating Triton Operators](https://aclanthology.org/2025.findings-acl.1340/) (Li et al., ACL 2025 Findings)
- **规模**: TritonBench-G (184 GitHub 真实 kernel, >100 star repos) + TritonBench-T (166 PyTorch 对齐 kernel)
- **目标语言**: Triton
- **评测流程**: correctness (vs reference Triton 或 PyTorch) + speedup
- **模型接入方式**: 标准 prompt → generation → eval pipeline，论文附评测代码
- **优势**: 来源是真实 GitHub repo，kernel 多样性最好；覆盖矩阵运算、卷积、注意力、自定义计算等
- **可对标论文**: Dr. Kernel, TritonRL, GEAK, AutoTriton

#### MultiKernelBench — 跨硬件平台

- **论文**: [MultiKernelBench: A Multi-Platform Benchmark for Kernel Generation](https://arxiv.org/abs/2507.13028) (Wen et al., 2025)
- **GitHub**: [wzzll123/MultiKernelBench](https://github.com/wzzll123/MultiKernelBench)
- **目标语言**: CUDA / Triton / AscendC / TileLang / Pallas / SYCL
- **覆盖硬件**: NVIDIA GPU, AMD GPU, Huawei NPU, Google TPU, Intel GPU
- **优势**: 唯一覆盖所有主流 AI 硬件的 kernel benchmark

### A.2 面向生产/真实 Workload

#### FastKernels — 生产级推理 kernel

- **论文**: [FastKernels: Benchmarking GPU Kernel Generation in Production](https://arxiv.org/abs/2605.23215) (2026)
- **规模**: 46 架构, 8 类, 4 个 level
- **目标语言**: CUDA
- **特点**: **所有 kernel 来自真实推理 workload**（非合成）；Level 1 (primitive: attention, norm, activation, PE) → Level 2 (fused: residual+RMSNorm+quant, attention+output_proj, MoE gate+dispatch+expert) → Level 3/4 更高级
- **优势**: 可直接部署，性能对标 vLLM/SGLang；发现现有 benchmark 的 sandbox kernel 在生产中常出现接口不兼容和 silent correctness degradation

#### SOL-ExecBench — 对标硬件理论上限

- **论文**: [SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels](https://arxiv.org/abs/2603.19173) (Microsoft, 2026)
- **规模**: 235 题, 从 124 个真实 AI 模型 (LLM/diffusion/vision/audio/video/hybrid) 提取
- **目标语言**: CUDA
- **目标硬件**: NVIDIA Blackwell
- **核心创新**: 基准不是 PyTorch eager 而是 **SOLAR 推导的硬件理论上限 (Speed-of-Light)**；SOL Score = kernel 缩小了多少 baseline-to-SOL gap
- **覆盖精度**: BF16, FP8, NVFP4
- **防作弊**: GPU clock locking, L2 cache clearing, isolated subprocess, static analysis 检测 reward hacking
- **优势**: 最严格的评测标准；如果想发论文拉高 bar，用这个

#### RealisticTritonBench — 真实框架级 Triton kernel

- **论文**: [RealisticTritonBench: A Benchmark for Triton-Kernel Generation in Real-World AI Frameworks](https://arxiv.org/abs/2608.12004) (2026)
- **数据来源**: PyTorch, vLLM, SGLang 中实际使用的 Triton kernel
- **特点**: 覆盖性能优化、bug 修复、功能扩展等真实开发场景

### A.3 面向特定能力

#### GEAK TritonBench-revised — AMD GPU 原生适配

- **论文**: [GEAK: Introducing Triton Kernel AI Agent & Evaluation Benchmarks](https://arxiv.org/abs/2507.23194) (Wang et al., 2025)
- **规模**: 214 题 (184 from TritonBench-G + 30 ROCm repo kernel)
- **目标语言**: Triton
- **AMD 适配**: 修复了 37 个 AMD GPU 兼容问题（shared memory errors, invalid HIP arguments, ModuleNotFound 等）
- **模型接入**: GEAK 框架支持替换 backbone model
- **优势**: **唯一直接在 AMD GPU (MI300X/MI250) 上验证过的 benchmark**

#### BackendBench — PyTorch ATen Ops 级别

- **GitHub**: [meta-pytorch/BackendBench](https://github.com/meta-pytorch/BackendBench)
- **规模**: 271 ops (correctness), 124 ops (performance)
- **目标语言**: Triton (作为 PyTorch backend)
- **评测流程**: 模型生成 Triton kernel → 自动注册为 PyTorch backend → OpInfo test suite 验证
- **优势**: 最接近真实 upstream 场景（目标是生成的 kernel 直接合进 PyTorch）

#### KernelCraft — 新兴硬件极限测试

- **论文**: [KernelCraft: Benchmarking for Agentic Close-to-Metal Kernel Generation on Emerging Hardware](https://arxiv.org/abs/2603.08721) (2026)
- **特点**: 测 close-to-metal 能力；DeepSeek-V3.2 和 DeepSeek-R1 在 KernelBench 上能跑，在 KernelCraft 上**完全失败**
- **优势**: 测模型对新硬件的泛化能力

#### TritonGym — OOD + DSL 扩展

- **论文**: [TritonGym](https://arxiv.org/abs/2502.08694) (2025)
- **特点**: 含 mutated operator semantics、out-of-distribution kernel、DSL extension task
- **优势**: 测模型对非标准 kernel 的泛化能力

### A.4 Benchmark 对比总结

| Benchmark | 语言 | 规模 | GitHub | 自定义模型接入 | AMD 兼容 | 基准对比对象 | 可对标论文 |
|-----------|------|------|--------|--------------|---------|------------|----------|
| **KernelBench** | CUDA | 250 | [link](https://github.com/ScalingIntelligence/KernelBench) | litellm API (最简单) | 否 | eager / compile | CUDA Agent, Claude, Gemini 等全部 |
| **KernelBenchX** | Triton | 176 | [link](https://github.com/BonnieW05/KernelBenchX) | eval harness | 是 | PyTorch / reference Triton | KernelBenchX |
| **TritonBench** | Triton | 350 | 论文附代码 | prompt→gen→eval | 是 | reference Triton / PyTorch | Dr. Kernel, TritonRL, GEAK |
| **MultiKernelBench** | 多语言 | 多平台 | [link](https://github.com/wzzll123/MultiKernelBench) | 标准接口 | 是 | 各平台原生实现 | MultiKernelBench |
| **FastKernels** | CUDA | 46 架构 | 论文附代码 | — | 否 | vLLM / SGLang | FastKernels |
| **SOL-ExecBench** | CUDA | 235 | 论文附代码 | sandbox harness | 否 | **硬件理论上限 (SOL)** | SOL-ExecBench |
| **GEAK-revised** | Triton | 214 | GEAK repo | 替换 backbone | **原生 AMD** | reference Triton | GEAK |
| **BackendBench** | Triton | 271 | [link](https://github.com/meta-pytorch/BackendBench) | PyTorch backend | 是 | PyTorch OpInfo | BackendBench |
| **KernelCraft** | — | — | — | — | — | — | KernelCraft |
| **TritonGym** | Triton | — | — | — | 是 | — | TritonGym |

### A.5 推荐评测方案

针对 GEAK agent + AMD MI308X + RL 训练模型的场景：

1. **主力评测 (横向对标)**: **KernelBench** — 结果可直接与 CUDA Agent、Claude Opus 4.5、Gemini 3 Pro 横向比较
2. **Triton 细分分析**: **KernelBenchX** — 15 类算子拆分，精确定位模型在哪类 kernel 上强/弱
3. **AMD 验证**: **GEAK TritonBench-revised** — 唯一直接在 AMD GPU 上验证过的 benchmark，与目标硬件直接匹配
4. **如果想发顶会拉高 bar**: **SOL-ExecBench** — vs 硬件理论上限，比 vs torch.compile 更有说服力
