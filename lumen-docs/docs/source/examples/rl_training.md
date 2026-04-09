# RL Training Examples

Last updated: 04/09/2026.

Reinforcement Learning training examples using Lumen FP8 optimizations on AMD MI300X GPUs.

---

## Supported Frameworks

| Directory | Framework | Backend | Description |
|-----------|-----------|---------|-------------|
| `examples/rl/trl/` | [TRL](https://github.com/huggingface/trl) | FSDP1/FSDP2/DDP | GRPO training via HuggingFace TRL GRPOTrainer |
| `examples/rl/verl/` | [VERL](https://github.com/volcengine/verl) | FSDP2 + sglang | GRPO training via VERL with async sglang rollout |

---

## Quick Start: TRL (Single-Node)

### 1. Build and launch the Docker container

```bash
cd examples/rl/trl
docker build -t lumen-trl -f Dockerfile ../../..

docker run --rm -it --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --shm-size 128G \
  -v /dev/shm/model:/dev/shm/model \
  lumen-trl bash
```

### 2. Run BF16 baseline (8-GPU DDP, Qwen2-0.5B)

```bash
cd examples/rl/trl/benchmark/qwen2.5-0.5b
bash run.sh R1
```

### 3. Run FP8 memory benchmark (single GPU, Llama-3.1-8B)

```bash
cd examples/rl/trl/benchmark
python test_fp8_memory.py --model /dev/shm/model/llama-3.1-8b --configs all
```

---

## Quick Start: VERL (Async Rollout with sglang)

### BF16 baseline

```bash
bash examples/rl/verl/run_grpo_fsdp2.sh
```

### With Lumen FP8

```bash
LUMEN_FP8=1 bash examples/rl/verl/run_grpo_fsdp2.sh
```

### With FP8 + LoRA + FP8 Param Manager

```bash
LUMEN_FP8=1 FP8_PARAM_MANAGER=1 bash examples/rl/verl/run_grpo_fsdp2.sh
```

### VERL + Megatron + sglang (multi-node, MoE/TP/PP)

```bash
bash examples/rl/verl/run_grpo_megatron_sglang.sh
```

---

## Lumen FP8 Modes for RL

| Mode | API | Memory Saving | Speed Impact |
|------|-----|---------------|--------------|
| FP8 Linear GEMM | `quant.enable(model, ...)` | None (weights stay BF16) | 3-4x slower (per-forward quantize) |
| FP8 Weight Cache | `quant.store_weights_fp8(model)` | +7.5 GB overhead | 2.5x slower (cached quantize) |
| FP8 Attention (dpa) | `LumenConfig(fp8_attn="dpa")` | None with grad ckpt | Marginal |
| **FP8ParamManager** | `FP8ParamManager.quantize_params(model)` | **-62% peak** | Needs validation |
| **FP8Param + 8-bit Adam** | FP8ParamManager + `bnb.optim.Adam8bit` | **-64% peak** | Needs validation |

---

## RL Algorithms

| Algorithm | Framework | LoRA + FP8 Status |
|-----------|-----------|-------------------|
| GRPO | TRL | Validated |
| GRPO | VERL (FSDP2) | Validated |
| GRPO | VERL (Megatron) | Supported |
| PPO | TRL / VERL | Supported |
| DAPO | TRL | Supported |

---

## Directory Layout

```
examples/rl/
├── README.md
├── trl/
│   ├── README.md
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── run_grpo_baseline.py
│   ├── compare_runs.py
│   ├── accelerate/
│   │   ├── fsdp1.yaml
│   │   └── fsdp2.yaml
│   └── benchmark/
│       ├── README.md
│       ├── test_fp8_memory.py
│       ├── run_grpo_benchmark.py
│       ├── run_grpo_benchmark.sh
│       ├── analyze_benchmark.py
│       ├── eval_checkpoints_gsm8k.py
│       ├── llama-3.1-8b/
│       └── qwen2.5-0.5b/
└── verl/
    └── run_grpo_fsdp2.sh
```

---

## See Also

- {doc}`/advance/rl_training` -- RL training architecture and configuration
- {doc}`/advance/rl_lora` -- RL Training with LoRA and FP8
- {doc}`/advance/feature_catalog` -- Complete Lumen feature inventory
