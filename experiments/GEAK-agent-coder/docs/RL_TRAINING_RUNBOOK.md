# GEAK Kernel RL Training Runbook

## 0. 目标

在 SFT checkpoint 基础上，通过 Lumen-RL 的 GRPO 框架进一步提升 kernel optimization policy。

**两组实验并行对比：**

| Experiment | Base Model | Rationale |
|-----------|-----------|-----------|
| **RL-from-SFT2** | `Qwen3-Coder-30B-A3B-SFT-TH-2epoch` | SFT-2e 是当前最佳优化器 (speedup geomean 1.001x, Fast@1.2 26%) |
| **RL-from-SFT4** | `Qwen3-Coder-30B-A3B-SFT-TH-4epoch` | SFT-4e compile rate 最高 (40%) 但 speedup 差 (geomean 0.742x)，RL 能否修复？ |

**Hardware**: 8x AMD MI308X (gfx942, 192GB HBM3)
**Framework**: Lumen-RL (`lumenrl/algorithms/grpo.py` + `lumenrl/trainer/rl_trainer.py`)
**Reward**: GEAK sandbox compile → correctness → speedup

---

## 1. SFT Baseline Speedup 分析（V8 Benchmark）

### 1.1 Patch Mode Speedup（核心优化指标）

| Metric | Base | SFT-2epoch | SFT-4epoch |
|--------|------|-----------|-----------|
| Correct patches | 48/99 | 42/99 | 37/99 |
| **Speedup geomean** | 0.762x | **1.001x** | 0.742x |
| Faster@1.0 | 16/48 (33%) | **19/42 (45%)** | 14/37 (38%) |
| **Fast@1.2** | 6/48 (13%) | **11/42 (26%)** | 4/37 (11%) |
| Fast@1.5 | 5/48 (10%) | 6/42 (14%) | 3/37 (8%) |

**关键发现：**
- SFT-2e 是唯一 speedup geomean ≥1.0x 的模型，Fast@1.2 最高
- SFT-4e 过拟合：pass 更多但 speedup 差，学到的是安全编辑模式而非优化策略
- Base 虽 pass 最多但大部分 patch 让 kernel 变慢

### 1.2 Generation Mode Speedup

| Metric | Base | SFT-2epoch | SFT-4epoch |
|--------|------|-----------|-----------|
| Correct | 7/99 | 4/99 | 12/99 |
| Speedup geomean | 0.333x | 0.356x | 0.199x |

生成模式产生的 kernel 全面慢于 baseline，不适合做优化指标。

### 1.3 RL 训练目标

| Metric | SFT-2e (best) | RL Target | Rationale |
|--------|-------------|-----------|-----------|
| Speedup geomean | 1.001x | **>1.20x** | 从不退步 → 实际加速 |
| Fast@1.2 | 26% | **>40%** | 更多 patch 产生有意义的加速 |
| Fast@1.5 | 14% | **>25%** | 大幅加速比例翻倍 |
| Patch correct | 42/99 | **>50/99** | 不以牺牲 pass rate 为代价 |
| Gen compile | 40% | **>55%** | 代码合成能力提升 |

---

## 2. 方法与 Reward 设计

### 2.1 为什么选 GRPO

- 不需要 value model：用 group 内样本的相对排名代替 value baseline，省 50% GPU 内存
- 天然适合 verifiable rewards：kernel correctness 和 speedup 可以自动验证
- 文献验证：Dr. Kernel, DRTriton, AMDKernelVault, LEAP 均使用 GRPO

### 2.2 Single-Turn Focus

每个 prompt 采样 G=8 个 response，用 compile/correctness/speedup 作为 reward。Multi-turn RL 留作后续。

### 2.3 Reward Function

```python
def kernel_reward(response, task) -> float:
    # Stage 1: Apply patch/code to workspace
    # Stage 2: Compile on GPU
    if not compile_ok:
        return -1.0
    # Stage 3: Correctness check
    if not correct_ok:
        return -0.5
    # Stage 4: Performance measurement
    speedup = baseline_ms / patched_ms
    if speedup <= 1.0:
        return 0.0  # correct but no speedup
    # Logarithmic reward capped at 3x speedup
    return min(math.log(speedup), math.log(3.0))
```

Reward 范围：[-1.0, 1.1] — compile fail 最差，3x speedup 最好。

---

## 3. Lumen-RL 基建

### 3.1 核心组件

| 组件 | 路径 | 说明 |
|------|------|------|
| GRPO 算法 | `lumenrl/algorithms/grpo.py` | Asymmetric clip, KL penalty |
| Advantage | `lumenrl/algorithms/advantage_estimators.py` | grpo, trloo, rloo, dapo |
| RL Trainer | `lumenrl/trainer/rl_trainer.py` | rollout → reward → advantage → update |
| FSDP2 Engine | `lumenrl/engine/training/fsdp_engine.py` | Single-node FSDP2 training |
| Profiling Reward | `lumenrl/rewards/profiling_reward.py` | rocprof hardware counters |

### 3.2 Multi-Tune Agent 组件

| 组件 | 路径 | 说明 |
|------|------|------|
| Kernel Reward | `experiments/multi-tune-agent/rewards/kernel_reward.py` | GEAK sandbox batch reward |
| Prompt Loader | `experiments/multi-tune-agent/rewards/kernel_prompt_loader.py` | SFT data → GRPO prompts |
| RL Trajectory | `experiments/multi-tune-agent/rewards/rl_trajectory.py` | Per-rollout/step metrics |
| Fuzzy Patch | `experiments/multi-tune-agent/scripts/fuzzy_patch.py` | Context-insensitive patch apply |

### 3.3 需要的修改

**rl_trainer.py 的 reward 路径需要改为可插拔：**
- 当前 `_compute_rewards` 和 `_compute_rewards_full` 硬编码 `compute_math_reward`
- 需要：根据 `config.reward.function` 动态导入 reward function
- kernel reward 需要额外元数据（task_id, task_dir, baseline_ms）通过 dataset 传递

---

## 4. 训练配置

### 4.1 GPU 分配

```
8x MI308X (192GB each):
  GPU 0: vLLM model serving (TP=1, ~60GB for A3B model)
  GPU 1-7: 7x evaluation sandbox (~20GB each for kernel eval)
  GPU 0: FSDP2 training (colocated with vLLM, swap via optimizer offload)
```

### 4.2 GRPO Config (SFT-2epoch)

```yaml
# experiments/multi-tune-agent/configs/rl/grpo_kernel_sft2.yaml
cluster:
  num_nodes: 1
  gpus_per_node: 8

policy:
  model_name: /home/danyzhan/Lumen/experiments/GEAK-agent-coder/outputs/qwen3-coder-full2000-2epoch-merged
  training_backend: fsdp2
  generation_backend: vllm
  max_total_sequence_length: 16384
  max_response_length: 8192
  train_global_batch_size: 32
  gen_batch_size: 8
  train_micro_batch_size: 1
  learning_rate: 5.0e-7

algorithm:
  name: grpo
  grpo:
    num_generations: 8
    kl_coeff: 0.0
    clip_ratio: 0.2
    clip_ratio_high: 0.28

reward:
  type: function
  function: experiments.multi_tune_agent.rewards.kernel_reward.kernel_reward_batch
  dataset: /home/danyzhan/geak_sft_dataset/phase1-production-wave-2000-v1/processed/rl_prompts.jsonl

num_training_steps: 300
```

### 4.3 GRPO Config (SFT-4epoch)

Same as above but with:
```yaml
policy:
  model_name: /home/danyzhan/Lumen/experiments/GEAK-agent-coder/outputs/qwen3-coder-full2000-4epoch-merged
```

---

## 5. 启动流程

### 5.1 环境准备

```bash
cd /home/danyzhan/Lumen-RL
pip install -e .

cd experiments/multi-tune-agent
pip install -e .

export GEAK_ROOT=/home/danyzhan/GEAK
export HELD_OUT_ROOT=/home/danyzhan/held-out-benchmark
export EVAL_GPU_IDS="1,2,3,4,5,6,7"
```

### 5.2 准备 RL Prompt 数据

```bash
cd /home/danyzhan/Lumen-RL/experiments/multi-tune-agent
python -c "
from rewards.kernel_prompt_loader import load_kernel_prompts
import json
prompts = load_kernel_prompts(
    '/home/danyzhan/geak_sft_dataset/phase1-production-wave-2000-v1/processed/train.jsonl'
)
with open('/home/danyzhan/geak_sft_dataset/phase1-production-wave-2000-v1/processed/rl_prompts.jsonl', 'w') as f:
    for p in prompts:
        f.write(json.dumps(p) + '\n')
print(f'Wrote {len(prompts)} prompts')
"
```

### 5.3 验证 Reward Pipeline

```bash
python -c "
from rewards.kernel_reward import eval_single_kernel
from pathlib import Path
result = eval_single_kernel(
    response='No changes needed',
    task_id='heldv5-adversarial_boundary-rms_norm-triton-02',
    task_dir=Path('/home/danyzhan/held-out-benchmark/artifacts/kernel/heldv5-adversarial_boundary-rms_norm-triton-02/initial'),
    baseline_ms=0.05, family='rms_norm', gpu_id=4,
)
print(f'Reward: {result[\"reward\"]}, Stage: {result[\"stage\"]}')
"
```

### 5.4 运行 GRPO 训练

```bash
cd /home/danyzhan/Lumen-RL

# Train RL from SFT-2epoch (better optimizer)
python -m lumenrl.trainer.main \
    --config experiments/multi-tune-agent/configs/rl/grpo_kernel_sft2.yaml \
    2>&1 | tee experiments/multi-tune-agent/logs/rl_sft2_$(date +%Y%m%d_%H%M).log

# Train RL from SFT-4epoch (better compiler, worse optimizer)
python -m lumenrl.trainer.main \
    --config experiments/multi-tune-agent/configs/rl/grpo_kernel_sft4.yaml \
    2>&1 | tee experiments/multi-tune-agent/logs/rl_sft4_$(date +%Y%m%d_%H%M).log
```

### 5.5 监控

```bash
# Step summaries
tail -f experiments/multi-tune-agent/outputs/rl-grpo-sft2/step_summaries.jsonl

# Training reward curve
python -c "
import json
with open('experiments/multi-tune-agent/outputs/rl-grpo-sft2/step_summaries.jsonl') as f:
    for line in f:
        s = json.loads(line)
        print(f'step={s[\"step\"]:3d} reward={s[\"mean_reward\"]:.3f} compile={s[\"compile_rate\"]:.0%} correct={s[\"correct_rate\"]:.0%} speedup={s[\"mean_speedup\"]:.2f}x')
"
```

---

## 6. 评估

### 6.1 Merge RL Checkpoint

```bash
python /home/danyzhan/Lumen/experiments/GEAK-agent-coder/scripts/merge_to_hf.py \
    --checkpoint experiments/multi-tune-agent/outputs/rl-grpo-sft2/step-300 \
    --output experiments/multi-tune-agent/outputs/rl-grpo-sft2-merged
```

### 6.2 Run V8 Benchmark

```bash
cd /home/danyzhan/Lumen-RL/experiments/multi-tune-agent
export PYTHONPATH="src:.:scripts:${PYTHONPATH}"

# Serve RL model with vLLM
python scripts/run_held_out_benchmark.py rl-sft2 both
python scripts/run_held_out_benchmark.py rl-sft4 both
```

### 6.3 对比指标

| Metric | SFT-2e | RL-from-SFT2 | SFT-4e | RL-from-SFT4 |
|--------|--------|-------------|--------|-------------|
| Speedup geomean | 1.001x | ? | 0.742x | ? |
| Fast@1.2 | 26% | ? | 11% | ? |
| Fast@1.5 | 14% | ? | 8% | ? |
| Patch correct | 42/99 | ? | 37/99 | ? |
| Gen compile | 27% | ? | 40% | ? |
| Gen correct | 4% | ? | 12% | ? |

### 6.4 成功标准

- RL-from-SFT2: speedup geomean >1.2x, Fast@1.2 >40%
- RL-from-SFT4: speedup geomean >1.0x (修复过拟合), Fast@1.2 >20%
- 两者都不能让 patch correct 下降超过 5pp

---

## 7. 不可违反的规则

1. RL reward 必须来自真实 GPU 执行，不能用模型预测替代
2. Held-out 数据不能用于训练
3. Speedup 必须在与 SFT 相同的 GPU SKU 上测量
4. 不能通过修改 harness/oracle/test 来获得更高 reward
5. 每个 checkpoint 必须验证 patch correctness 不下降
6. GPU 内存不足时降低 batch/group size，不能跳过 eval

---

## 8. 参考文献

- [Dr. Kernel (ICML 2026)](https://arxiv.org/abs/2602.05885) — GRPO + TRLOO
- [Kevin (ICLR 2026)](https://arxiv.org/abs/2507.11948) — Multi-turn GRPO
- [DRTriton (2026)](https://arxiv.org/abs/2603.21465) — Synthetic data + GRPO
- [AMDKernelVault (2026)](https://arxiv.org/abs/2609.12471) — AMD GPU kernel GRPO
- [LEAP (2026)](https://arxiv.org/abs/2608.01804) — Adaptive pruning for code RL
- [Afterburner (2025)](https://arxiv.org/abs/2505.23387) — RL for code efficiency
