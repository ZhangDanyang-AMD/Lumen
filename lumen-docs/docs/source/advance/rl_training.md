# RL Training

Last updated: 04/09/2026.

Lumen supports reinforcement learning post-training (RLHF) for LLMs by integrating with **TRL** (Hugging Face) and **VERL** (volcengine) as RL frameworks, while providing FP8 quantized training for the Actor and Reference models. Both **SGLang** and **vLLM** are supported as rollout engines on ROCm.

## Overview

RL post-training workloads (GRPO, PPO, DAPO) are more memory-intensive than SFT or pre-training because they maintain multiple model copies (Actor, Reference, optionally Critic) and perform multiple forward passes per step. Lumen addresses this by:

- **FP8 Param Manager** — in-place FP8 parameter storage via `FP8_PARAM_MANAGER=1`, yielding **-25% peak VRAM** (no offload) or **-5%** (with offload) vs BF16
- **FP8 all-gather** — ~50% communication bandwidth saving during FSDP parameter gathering
- **FP8 attention** — ~11% additional memory saving on attention activations
- **Non-invasive integration** — `quant.enable(model)` works on TRL/VERL Actor models without framework modifications
- **Dual rollout engines** — SGLang and vLLM V1 both validated on ROCm MI300X

## Supported Algorithms

| Algorithm | Framework | Rollout | Status |
|-----------|-----------|---------|--------|
| **GRPO** | TRL | Built-in | Validated |
| **GRPO** | VERL (FSDP2) | SGLang | Validated |
| **GRPO** | VERL (FSDP2) | vLLM | Validated |
| **GRPO** | VERL (Megatron) | SGLang | Validated |
| **GRPO** | VERL (Megatron) | vLLM | Validated |
| **PPO** | TRL / VERL | SGLang / vLLM | Supported |
| **DAPO** | TRL | Built-in | Supported |

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

Lumen integrates with VERL's worker architecture via FSDP2 and Megatron backends, with **SGLang** or **vLLM** as the rollout engine:

| Backend | Rollout | Integration Path | Use Case |
|---------|---------|-----------------|----------|
| **FSDP2** | SGLang | VERL FSDPWorker + `quant.enable` | Default path, ≤70B models |
| **FSDP2** | vLLM | VERL FSDPWorker + `quant.enable` | vLLM-based inference pipelines |
| **Megatron** | SGLang | VERL MegatronWorker + Lumen Megatron stack | 32B+ dense models, MoE with EP |
| **Megatron** | vLLM | VERL MegatronWorker + Lumen Megatron stack | vLLM rollout with TP (TP=1 on ROCm) |

```{mermaid}
flowchart LR
  subgraph VERL["VERL Controller"]
    R["Rollout (SGLang / vLLM)"]
    L["Loss / Advantage"]
    S["Step Orchestration"]
  end

  subgraph PathA["Path A: FSDP2"]
    FA["FSDP2 Worker"]
    FQ["quant.enable(actor)\n+ FP8 Param Manager"]
  end

  subgraph PathB["Path B: Megatron"]
    MA["Megatron Worker"]
    MQ["Lumen Megatron FP8\n(on-the-fly quantization)"]
  end

  VERL --> PathA
  VERL --> PathB
  PathA --> HW["MI300X + AITER + MORI"]
  PathB --> HW
```

## Benchmark: FSDP2 vs Megatron on MI300X (ROCm 7.0)

Tested with Qwen2.5-0.5B-Instruct, 4× MI300X, GRPO, 2 steps, micro_batch=2, seq_len=512+256. Container: `rocm/sgl-dev:v0.5.9-rocm700` with VERL 0.8.0.dev.

### SGLang Rollout

| Configuration | Offload | Batch | Peak VRAM | Throughput | vs BF16 |
|---|---|---|---|---|---|
| FSDP2 BF16 | Yes | 16 | 48.06 GB | 797 tok/s | baseline |
| **FSDP2 FP8PM** | **Yes** | **16** | **45.50 GB** | **942 tok/s** | **-5% VRAM** |
| FSDP2 BF16 | No | 16 | 73.49 GB | ~900 tok/s | baseline |
| **FSDP2 FP8PM** | **No** | **16** | **54.87 GB** | **913 tok/s** | **-25% VRAM** |
| Megatron BF16 | Yes | 16 | 70.52 GB | 704 tok/s | N/A |
| **Megatron FP8PM** | **Yes** | **16** | **50.06 GB** | **369 tok/s** | **-29% VRAM** |

### vLLM Rollout

| Configuration | Offload | gpu_util | Peak VRAM | Throughput | Weight Update |
|---|---|---|---|---|---|
| FSDP2 BF16 (4 GPU) | Yes | 0.4 | 125.75 GB | 1,766 tok/s | 0.47–0.58s |
| **FSDP2 FP8PM (4 GPU)** | **Yes** | **0.4** | **125.78 GB** | **1,871 tok/s** | **0.55–0.70s** |
| **FSDP2 BF16 (4 GPU)** | **No** | **0.1** | **34.26 GB** | **445 tok/s** | **1.28s** |
| **FSDP2 FP8PM (4 GPU)** | **No** | **0.1** | **29.80 GB** | **477 tok/s** | **1.04s** |
| Megatron BF16 (4 GPU, TP=2) | Yes | 0.3 | 85.01 GB | 338 tok/s | 2.03s |
| Megatron FP8PM (4 GPU, TP=2) | Yes | 0.3 | 86.89 GB | 291 tok/s | 2.57s |

With `gpu_memory_utilization=0.1` (minimal vLLM reservation), FP8PM shows **29.80 GB** vs **34.26 GB** (BF16) — **13% less peak VRAM** and **7% faster throughput**.

### Key Findings

1. **FSDP2 outperforms Megatron** — 1.3× higher throughput (797 vs 704 tok/s with SGLang) and 32% lower peak memory (48 vs 71 GB) for BF16 with offloading.

2. **FP8PM saves memory in both offload modes** — After the `save_for_backward` clone fix in `lumen/quantize/fp8_params.py`, FP8PM correctly reclaims FSDP2's allgathered weight buffers. Savings: -5% (offload) to -25% (no offload).

3. **Megatron FP8PM uses on-the-fly quantization** — Parameters stay BF16 for optimizer/DDP compatibility; `_FP8MegatronLinearFunc` quantizes during forward only. In-place quantization crashes Megatron's distributed optimizer.

4. **vLLM V1 works on ROCm** — After a one-line `@with_amdsmi_context` fix on `RocmPlatform.get_device_uuid()` in vLLM. Without it, mismatched device UUIDs cause weight-transfer deadlock.

5. **vLLM FP8PM savings depend on gpu_memory_utilization** — At `gpu_util=0.4`, vLLM KV cache (~101 GB) masks actor savings. At `gpu_util=0.1`, savings become visible (13% VRAM, 7% throughput).

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

| Configuration | Rollout | Backend | Best For |
|---------------|---------|---------|----------|
| TRL + FSDP1 | Built-in | `quant.enable` + Accelerate | Quick experiments, small-medium models |
| TRL + FSDP2 | Built-in | `quant.enable` + FSDP2 | Equivalent dynamics, newer PyTorch |
| VERL + FSDP2 | SGLang | VERL FSDPWorker + `quant.enable` | Large-scale RL (recommended) |
| VERL + FSDP2 | vLLM | VERL FSDPWorker + `quant.enable` | vLLM-based inference pipelines |
| VERL + Megatron | SGLang | VERL MegatronWorker + Lumen | MoE models, TP/PP/EP parallelism |
| VERL + Megatron | vLLM | VERL MegatronWorker + Lumen | vLLM rollout (TP=1 on ROCm) |

## Recommended Configuration

For VERL GRPO on MI300X (ROCm) with FP8 memory savings:

```bash
# Best configuration: FSDP2 + SGLang + FP8PM (no offloading)
FP8_PARAM_MANAGER=1 \
NUM_GPUS=4 ROLLOUT_TP=2 ROLLOUT_GPU_UTIL=0.1 \
TRAIN_BSZ=16 MAX_STEPS=2 \
bash examples/rl/verl/run_grpo_fsdp2.sh
```

Expected savings: **-25% peak VRAM** vs BF16 (no offload), **-5%** with offload.

For vLLM rollout, additionally set:

```bash
rollout.tensor_model_parallel_size=1   # TP>=2 may hang on ROCm
rollout.free_cache_engine=false
rollout.enforce_eager=true
```

## ROCm-Specific Fixes

### vLLM V1 `get_device_uuid` Fix

vLLM V1 hangs on ROCm due to a missing `@with_amdsmi_context` decorator on `RocmPlatform.get_device_uuid()` in `vllm/platforms/rocm.py`. Without it, `amdsmi` is uninitialized in VERL worker processes, producing a fallback UUID that doesn't match the EngineCore's real hardware UUID, causing a ZMQ socket path mismatch and deadlock.

```python
# In vllm/platforms/rocm.py:
@classmethod
@with_amdsmi_context        # <-- ADD THIS LINE
def get_device_uuid(cls, device_id: int = 0) -> str:
    ...
```

### FP8PM FSDP2 Offload Fix

`_FP8LinearFunc.forward` previously called `ctx.save_for_backward(fp8_weight, scale)`, which pinned FSDP2's allgathered weight tensors on GPU and prevented offload reclamation (+44% memory regression). Fixed by cloning:

```python
# In lumen/quantize/fp8_params.py:
ctx.save_for_backward(fp8_weight.clone(), scale)
```

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
