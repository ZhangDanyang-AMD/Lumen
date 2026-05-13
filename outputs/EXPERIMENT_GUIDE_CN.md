# Lumen TRL GRPO 实验指南 — 复现手册

## 硬件环境

| 项目 | 规格 |
|---|---|
| GPU | 8× AMD Instinct MI300X (192 GB HBM3, CDNA3 架构) |
| 互联 | xGMI 全连接, 每链路理论 64 GB/s |
| CPU | AMD EPYC |
| 内存 | ≥ 512 GB DDR5 |
| 存储 | NVMe SSD（模型加载建议放 `/dev/shm` 以加速） |

## 软件环境

| 组件 | 版本要求 |
|---|---|
| ROCm | ≥ 6.x |
| PyTorch | ≥ 2.4（FSDP2 需要） |
| TRL | ≥ 0.16 |
| Accelerate | ≥ 1.6.0 |
| Transformers | ≥ 4.50 |
| Lumen | 当前仓库 HEAD |

> **注**：实际测试时的精确版本信息会由 benchmark runner 自动记录到
> `env_info.json`（见 `run_grpo_benchmark.py` 的 `_collect_env_info()`）。

## 实验总览

本目录下包含 4 组已完成的实验，均在上述硬件上运行：

| 目录 | 实验内容 | FSDP | Actor 构建 | 对比基准 |
|---|---|---|---|---|
| `trl-grpo-70b/` | **Lumen + FSDP1** — 主实验 | v1 | Lumen `build_actor_model()` | — |
| `trl-grpo-70b-baseline/` | **纯 TRL + FSDP1** — BF16 基线 | v1 | TRL 默认 | vs `trl-grpo-70b/` |
| `trl-grpo-70b-fsdp2/` | **Lumen + FSDP2** — FSDP 版本对比 | v2 | Lumen `build_actor_model()` | vs `trl-grpo-70b/` |
| `trl-grpo-70b-baseline-fsdp2/` | **纯 TRL + FSDP2** — FSDP2 基线 | v2 | TRL 默认 | vs `trl-grpo-70b-fsdp2/` |

---

## 共用训练参数

所有 4 组实验使用相同的训练超参数：

| 参数 | 值 |
|---|---|
| 模型 | `NousResearch/Llama-2-70b-hf` (70B) |
| 训练步数 | 30 |
| 随机种子 | 1234 |
| GPU 数量 | 8 |
| Micro Batch Size | 1 |
| Gradient Accumulation | 1 |
| Num Generations | 8 |
| Max Completion Length | 256 tokens |
| Max Prompt Length | 512 tokens |
| Learning Rate | 5e-6 (linear decay) |
| Beta (KL penalty) | 0.0 |
| Gradient Checkpointing | ON |
| 数据集 | `trl-lib/Capybara` |
| Reward Function | Word-count conciseness reward（见下文） |

**Reward Function 定义**（位于 `examples/rl/trl/run_grpo_fsdp.py`）：

```python
def reward_fn(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        n_words = len(completion.split())
        if n_words < 5:
            r = 0.1
        elif n_words <= 60:
            r = min(1.0, 0.3 + 0.7 * n_words / 60)
        else:
            r = max(0.0, 1.0 - (n_words - 60) / 120)
        rewards.append(round(r, 4))
    return rewards
```

---

## 脚本文件说明

### 训练脚本

| 文件 | 用途 |
|---|---|
| `examples/rl/trl/run_grpo_fsdp.sh` | Shell 启动器，接受 FSDP 版本参数（1 或 2），通过环境变量配置所有超参 |
| `examples/rl/trl/run_grpo_fsdp.py` | Python 训练脚本，构建 actor model + TRL GRPOTrainer + 训练循环 |
| `examples/rl/trl/accelerate/fsdp1.yaml` | Accelerate FSDP1 配置（FULL_SHARD, BACKWARD_PRE） |
| `examples/rl/trl/accelerate/fsdp2.yaml` | Accelerate FSDP2 配置（fully_shard API, reshard_after_forward） |

### Benchmark 脚本（用于更系统化的 FP8 性能对比）

| 文件 | 用途 |
|---|---|
| `examples/rl/trl/benchmark/run_grpo_benchmark.sh` | Benchmark 启动器，支持 R1-R5 五种配置 |
| `examples/rl/trl/benchmark/run_grpo_benchmark.py` | Benchmark runner，自动记录环境信息、性能指标 |

Benchmark run ID 定义：

| Run ID | 配置 | FP8 | LoRA | 用途 |
|---|---|---|---|---|
| R1 | 纯 TRL baseline | OFF | OFF | BF16 性能基线 |
| R2 | Lumen BF16 | OFF | OFF | 验证 Lumen 框架开销 |
| R3 | Lumen FP8 Linear | Linear only | OFF | FP8 GEMM 加速 |
| R4 | Lumen FP8 Full | act+wgrad+reduce | OFF | FP8 全套优化 |
| R5 | Lumen FP8 + LoRA | act+wgrad+reduce | r=32 | FP8 + LoRA 组合 |

### 可视化脚本

| 文件 | 用途 | 输出 |
|---|---|---|
| `lumen/rl/trl/plot_curves.py` | 绘制单个 run 的 6 面板训练曲线 | `grpo_curves.png` |
| `examples/rl/trl/compare_runs.py` | 绘制两个 run 的对比曲线 + 生成 COMPARISON.md | `compare_curves.png` + `COMPARISON.md` |

### 回调与工具

| 文件 | 用途 |
|---|---|
| `lumen/rl/trl/eval_callback.py` | 训练中每步记录 metrics 到 `grpo_eval_log.jsonl` |
| `lumen/rl/trl/perf_callback.py` | Benchmark 用性能回调，记录 step time、内存、吞吐 |
| `lumen/rl/trl/patched_trainer.py` | Patched GRPOTrainer，支持 Lumen 的 FP8 lifecycle |

---

## 复现步骤

### Step 1：环境准备

```bash
# 确认 ROCm 和 PyTorch 可用
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"

# 确认 TRL 和 Accelerate 版本
python -c "import trl; print(trl.__version__)"
python -c "import accelerate; print(accelerate.__version__)"
```

### Step 2：模型下载

```bash
# 下载模型到 /dev/shm 加速加载（推荐）
huggingface-cli download NousResearch/Llama-2-70b-hf --local-dir /dev/shm/model/llama-2-70b
```

### Step 3：运行训练

所有实验均从 Lumen 仓库根目录执行。

#### 实验 A：Lumen + FSDP1（对应 `trl-grpo-70b/`）

```bash
cd /path/to/Lumen

MODEL_NAME=NousResearch/Llama-2-70b-hf \
OUTPUT_DIR=outputs/trl-grpo-70b \
NUM_PROCESSES=8 \
MAX_STEPS=30 \
GRAD_ACCUM=1 \
MICRO_BATCH_SIZE=1 \
NUM_GENERATIONS=8 \
MAX_COMPLETION_LENGTH=256 \
MAX_PROMPT_LENGTH=512 \
LR=5e-6 \
SEED=1234 \
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  bash examples/rl/trl/run_grpo_fsdp.sh 1
```

`run_grpo_fsdp.sh` 内部调用 `examples/rl/trl/run_grpo_fsdp.py`，
该脚本通过 Lumen 的 `build_actor_model()` + `run_grpo()` 构建 actor 并启动训练。

#### 实验 B：纯 TRL + FSDP1 基线（对应 `trl-grpo-70b-baseline/`）

基线实验使用 benchmark runner 的 **R1** 模式（不导入 Lumen，直接传模型名给 TRL GRPOTrainer）：

```bash
cd /path/to/Lumen

MODEL_DIR=/dev/shm/model/llama-2-70b \
OUTPUT_BASE=outputs/trl-grpo-70b-baseline \
NUM_PROCESSES=8 \
MAX_STEPS=30 \
GRAD_ACCUM=1 \
MICRO_BATCH_SIZE=1 \
NUM_GENERATIONS=8 \
MAX_COMPLETION_LENGTH=256 \
MAX_PROMPT_LENGTH=512 \
SEED=1234 \
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  bash examples/rl/trl/benchmark/run_grpo_benchmark.sh R1
```

> **R1 vs Lumen 的区别**：R1 不导入任何 Lumen 代码，直接传 model name string
> 给 `GRPOTrainer`，让 TRL 自行处理模型加载和 FSDP wrapping。

#### 实验 C：Lumen + FSDP2（对应 `trl-grpo-70b-fsdp2/`）

```bash
cd /path/to/Lumen

MODEL_NAME=NousResearch/Llama-2-70b-hf \
OUTPUT_DIR=outputs/trl-grpo-70b-fsdp2 \
NUM_PROCESSES=8 \
MAX_STEPS=30 \
GRAD_ACCUM=1 \
MICRO_BATCH_SIZE=1 \
NUM_GENERATIONS=8 \
MAX_COMPLETION_LENGTH=256 \
MAX_PROMPT_LENGTH=512 \
LR=5e-6 \
SEED=1234 \
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  bash examples/rl/trl/run_grpo_fsdp.sh 2
```

#### 实验 D：纯 TRL + FSDP2 基线（对应 `trl-grpo-70b-baseline-fsdp2/`）

```bash
cd /path/to/Lumen

MODEL_DIR=/dev/shm/model/llama-2-70b \
OUTPUT_BASE=outputs/trl-grpo-70b-baseline-fsdp2 \
NUM_PROCESSES=8 \
FSDP_VERSION=2 \
MAX_STEPS=30 \
GRAD_ACCUM=1 \
MICRO_BATCH_SIZE=1 \
NUM_GENERATIONS=8 \
MAX_COMPLETION_LENGTH=256 \
MAX_PROMPT_LENGTH=512 \
SEED=1234 \
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  bash examples/rl/trl/benchmark/run_grpo_benchmark.sh R1
```

### Step 4：生成可视化

训练完成后，输出目录中会有 `grpo_eval_log.jsonl`。运行以下命令生成图表：

```bash
# 生成单 run 训练曲线
python -m lumen.rl.trl.plot_curves outputs/trl-grpo-70b
python -m lumen.rl.trl.plot_curves outputs/trl-grpo-70b-baseline
python -m lumen.rl.trl.plot_curves outputs/trl-grpo-70b-fsdp2
python -m lumen.rl.trl.plot_curves outputs/trl-grpo-70b-baseline-fsdp2
```

```bash
# 对比 1：Lumen + FSDP1 vs 纯 TRL + FSDP1（基线）
# 输出到 dir_b = outputs/trl-grpo-70b-baseline/
python examples/rl/trl/compare_runs.py \
    outputs/trl-grpo-70b \
    outputs/trl-grpo-70b-baseline \
    --label-a "Lumen" --label-b "Baseline (TRL)"

# 对比 2：Lumen + FSDP1 vs Lumen + FSDP2
# 输出到 dir_b = outputs/trl-grpo-70b-fsdp2/
python examples/rl/trl/compare_runs.py \
    outputs/trl-grpo-70b \
    outputs/trl-grpo-70b-fsdp2 \
    --label-a "Lumen + FSDP1" --label-b "Lumen + FSDP2"

# 对比 3：Lumen + FSDP2 vs 纯 TRL + FSDP2（基线）
# 输出到 dir_b = outputs/trl-grpo-70b-baseline-fsdp2/
python examples/rl/trl/compare_runs.py \
    outputs/trl-grpo-70b-fsdp2 \
    outputs/trl-grpo-70b-baseline-fsdp2 \
    --label-a "Lumen + FSDP2" --label-b "Baseline + FSDP2"
```

> **注意**：`compare_runs.py` 将 `compare_curves.png` 和 `COMPARISON.md` 写到 **第二个目录**（`dir_b`）。

### Step 5：运行 Benchmark（可选，更系统化的性能测试）

```bash
# R1: BF16 baseline
bash examples/rl/trl/benchmark/run_grpo_benchmark.sh R1

# R4: Lumen FP8 Full
bash examples/rl/trl/benchmark/run_grpo_benchmark.sh R4

# 或一次跑完 R1-R4
bash examples/rl/trl/benchmark/run_grpo_benchmark.sh ALL
```

---

## 输出文件说明

每个实验目录包含以下文件：

| 文件 | 内容 | 生成方式 |
|---|---|---|
| `grpo_eval_log.jsonl` | 每步训练 metrics（reward, entropy, loss, grad_norm, length, win_rate） | 训练时由 `GRPOEvalCallback` 自动写入 |
| `grpo_curves.png` | 6 面板训练曲线图 | `python -m lumen.rl.trl.plot_curves <dir>` |
| `compare_curves.png` | 两个 run 的对比曲线图（仅对比目录有） | `python examples/rl/trl/compare_runs.py <dir_a> <dir_b>` |
| `COMPARISON.md` | 对比统计报告（Pearson r, 均值, 标准差） | 同上，自动生成 |
| `ANALYSIS.md` | 详细实验分析（仅 `trl-grpo-70b/`） | 手工撰写 |

Benchmark 实验额外输出：

| 文件 | 内容 |
|---|---|
| `env_info.json` | 硬件/软件环境快照（GPU 名称、ROCm 版本、PyTorch 版本、Lumen commit 等） |
| `grpo_perf_log.jsonl` | 每步性能 metrics（step_time, peak_memory, tokens） |
| `perf_summary.json` | 性能汇总统计（均值 ± 标准差，排除 warmup 步） |
| `run.log` | 完整的训练 stdout/stderr 日志 |

---

## 环境变量参考

`run_grpo_fsdp.sh` 支持的所有环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL_NAME` | `hf-internal-testing/tiny-random-LlamaForCausalLM` | 模型路径或 HuggingFace ID |
| `OUTPUT_DIR` | `outputs/trl-grpo-smoke` | 输出目录 |
| `NUM_PROCESSES` | 2 | GPU 数量 |
| `MICRO_BATCH_SIZE` | 1 | 每 GPU 每步的 micro batch |
| `GRAD_ACCUM` | 1 | 梯度累积步数 |
| `MAX_STEPS` | 2 | 训练总步数 |
| `LR` | 5e-6 | 学习率 |
| `LR_WARMUP_STEPS` | 0 | 学习率预热步数 |
| `MAX_PROMPT_LENGTH` | 1024 | 最大 prompt 长度 |
| `MAX_COMPLETION_LENGTH` | 512 | 最大生成长度 |
| `NUM_GENERATIONS` | 4 | 每 prompt 生成数 |
| `SEED` | 1234 | 随机种子 |
| `LINEAR_FP8` | 0 | 1=启用 FP8 Linear |
| `LORA_RANK` | 0 | LoRA rank（0=禁用） |
| `LORA_ALPHA` | 32 | LoRA alpha |
| `DATASET_NAME` | `trl-lib/Capybara` | HuggingFace 数据集名 |
| `TRAIN_DATA_PATH` | (空) | 本地 JSONL 数据路径（优先于 DATASET_NAME） |
| `TOKENIZER_NAME_OR_PATH` | (空) | 自定义 tokenizer 路径（默认使用模型自带） |
| `WARMUP_STEPS` | 0 | Lumen synthetic FP8 warmup 步数 |
| `LOG_INTERVAL` | 1 | 日志记录间隔（步） |
| `SAVE_INTERVAL` | 0 | checkpoint 保存间隔（0=不保存） |
| `SEQ_LENGTH` | (空) | 序列长度（默认 = MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH） |
| `LORA_DROPOUT` | 0.1 | LoRA dropout |

---

## 测试流程总结

```
Step 1: 环境准备
  ├── 验证 ROCm / PyTorch / TRL / Accelerate
  └── 确认 GPU 可用 (8× MI300X)

Step 2: 模型下载
  └── huggingface-cli download → /dev/shm/model/

Step 3: 训练（4 组实验，相同超参数）
  ├── A: Lumen + FSDP1  →  outputs/trl-grpo-70b/
  ├── B: Baseline + FSDP1  →  outputs/trl-grpo-70b-baseline/
  ├── C: Lumen + FSDP2  →  outputs/trl-grpo-70b-fsdp2/
  └── D: Baseline + FSDP2  →  outputs/trl-grpo-70b-baseline-fsdp2/
      每组约 2 小时 (~240s/step × 30 steps)

Step 4: 可视化
  ├── plot_curves.py  →  每个目录生成 grpo_curves.png
  └── compare_runs.py →  对比目录生成 compare_curves.png + COMPARISON.md

Step 5 (可选): Benchmark (R1-R5)
  └── 更系统化的 FP8 性能对比
```

## `grpo_eval_log.jsonl` 字段说明

每行一条 JSON，每个 training step 记录一条。主要字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `step` | int | 训练步编号 |
| `reward` / `rewards/reward_fn/mean` | float | 当步 8 个 completions 的平均 reward |
| `reward_std` / `rewards/reward_fn/std` | float | Reward 标准差 |
| `loss` | float | GRPO policy loss |
| `grad_norm` | float | 梯度范数 |
| `entropy` | float | Token 分布 entropy |
| `completions/mean_length` | float | 平均生成长度（tokens） |
| `completions/clipped_ratio` | float | 被 max_length 截断的比例 |
| `step_time` | float | 当步耗时（秒） |
| `num_tokens` | float | 累计处理 token 数 |
| `learning_rate` | float | 当前学习率 |
| `win_rate` | float | 滚动 5 步窗口 vs 前 3 步基线的胜率 |

---

## 已知问题与注意事项

1. **内存配置**：70B 模型需要设置 `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` 以避免内存碎片化导致 OOM。

2. **FSDP2 Entropy 溢出**：TRL 的 entropy 计算在 FSDP2 下偶发 BF16 数值溢出，
   表现为极端异常值（~1e23）。这是 TRL/BF16 问题，非 Lumen 引起。
   `plot_curves.py` 和 `compare_runs.py` 已内置 MAD-based 异常值过滤。

3. **Step 18 Collapse**：30 步训练中 step 18 可能出现模型生成极短 completion（<5 tokens）
   的 collapse 现象。这是 GRPO + length-based reward 的随机特性，两个 FSDP 版本均会出现，
   模型在下一步即恢复。

4. **基线实验的 Actor 构建差异**：Lumen 实验通过 `build_actor_model()` 构建 actor（加载 + bf16 + sdpa + gradient checkpointing），
   基线实验直接传模型名 string 给 TRL。两种方式的训练动态在统计上等价
   （Pearson r > 0.77，见各 COMPARISON.md）。
