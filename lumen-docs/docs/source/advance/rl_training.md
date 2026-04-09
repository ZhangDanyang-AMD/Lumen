# RL Training

Lumen supports reinforcement learning post-training (RLHF) for LLMs by integrating with **TRL** (Hugging Face) and **VERL** (volcengine) as RL frameworks, while providing FP8 quantized training for the Actor and Reference models.

## Overview

RL post-training workloads (GRPO, PPO, DAPO) are more memory-intensive than SFT or pre-training because they maintain multiple model copies (Actor, Reference, optionally Critic) and perform multiple forward passes per step. Lumen addresses this by:

- **FP8 weights** for Actor and Reference models — ~50% parameter memory reduction
- **FP8 all-gather** — ~50% communication bandwidth saving during FSDP parameter gathering
- **FP8 attention** — ~11% additional memory saving on attention activations
- **Non-invasive integration** — `quant.enable(model)` works on TRL/VERL Actor models without framework modifications

## Supported Algorithms

| Algorithm | Framework | Status |
|-----------|-----------|--------|
| **GRPO** (Group Relative Policy Optimization) | TRL | Validated |
| **PPO** (Proximal Policy Optimization) | TRL / VERL | Supported |
| **DAPO** (Decoupled Alignment Policy Optimization) | TRL | Supported |

## Framework Integration

### TRL (Hugging Face)

Lumen integrates with TRL's `GRPOTrainer` via Hugging Face Accelerate + FSDP/FSDP2. The integration is in `lumen/rl/trl/`:

| Component | File | Description |
|-----------|------|-------------|
| **Runner** | `lumen/rl/trl/runner.py` | High-level `run_grpo()` entry point |
| **Args** | `lumen/rl/trl/args.py` | Argument translation to `GRPOConfig` |
| **Modeling** | `lumen/rl/trl/modeling.py` | `build_actor_model` with FP8, LoRA, gradient checkpointing |
| **Eval Callback** | `lumen/rl/trl/eval_callback.py` | Per-step JSONL logging (reward, length, entropy) |
| **Perf Callback** | `lumen/rl/trl/perf_callback.py` | Step-time and memory profiling |

#### Quick Start (TRL + GRPO)

```python
from lumen.rl.trl.runner import run_grpo

run_grpo(
    model_name="meta-llama/Llama-2-70b-hf",
    quant_format="fp8_e4m3",
    scaling="delayed",
    fsdp_strategy="FULL_SHARD",
    lora_r=16,
)
```

### VERL

Lumen integrates with VERL's worker architecture via FSDP2 and Megatron backends:

| Backend | Integration Path | Use Case |
|---------|-----------------|----------|
| **FSDP2** | VERL FSDPWorker + `quant.enable` | Default path, ≤70B models |
| **Megatron** | VERL MegatronWorker + Lumen Megatron stack | 32B+ dense models, MoE with EP |

```{mermaid}
flowchart LR
  subgraph VERL["VERL Controller"]
    R["Rollout"]
    L["Loss / Advantage"]
    S["Step Orchestration"]
  end

  subgraph PathA["Path A: FSDP2"]
    FA["FSDP2 Worker"]
    FQ["quant.enable(actor)"]
  end

  subgraph PathB["Path B: Megatron"]
    MA["Megatron Worker"]
    MQ["Lumen Megatron FP8"]
  end

  VERL --> PathA
  VERL --> PathB
  PathA --> HW["MI300X + AITER + MORI"]
  PathB --> HW
```

## Validated Results

### LLaMA-2-70B GRPO (8× MI300X, 30 steps)

Correlation against an external baseline run (same recipe), using Pearson *r* on per-step aggregates:

| Metric | Lumen mean | Baseline mean | Pearson *r* |
|--------|------------|---------------|-------------|
| Reward | 0.364 | 0.347 | 0.776 |
| Response length | 162.6 | 166.3 | 0.835 |
| Entropy | 1.935 | 1.915 | 0.906 |

### VERL vs TRL Memory Comparison (BF16)

Peak memory per GPU — VERL + FSDP2 + SGLang vs TRL + FSDP1:

| Model | TRL + FSDP1 | VERL + FSDP2 + SGLang | Reduction |
|-------|-------------|------------------------|-----------|
| Llama-3.1-8B | 34.57 GB/GPU | 11.76 GB/GPU | ~66% |
| Qwen2.5-32B | 124.18 GB/GPU | 38.43 GB/GPU | ~69% |

## Training Backends

| Configuration | Backend | Best For |
|---------------|---------|----------|
| TRL + FSDP1 | `quant.enable` + Accelerate | Quick experiments, small-medium models |
| TRL + FSDP2 | `quant.enable` + FSDP2 | Equivalent dynamics, newer PyTorch |
| VERL + FSDP2 | VERL FSDPWorker + `quant.enable` | Large-scale RL, significant memory savings |
| VERL + Megatron | VERL MegatronWorker + Lumen | MoE models, TP/PP/EP parallelism |

## Monitoring

Lumen's TRL integration includes two callbacks for training diagnostics:

### GRPOEvalCallback

Logs per-step metrics to JSONL files for offline analysis:
- Reward aggregates (mean, min, max)
- Response length distribution
- Policy entropy
- Other configured scalars (TRL version dependent)

### GRPOPerfCallback

Profiles training performance for backend comparison:
- Per-step wall time
- Peak GPU memory
- Useful for comparing FSDP1 vs FSDP2, BF16 vs FP8
