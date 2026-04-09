# RL Training Examples

Last updated: 04/09/2026.

Reinforcement Learning training examples using Lumen FP8 optimizations on AMD MI300X GPUs with SGLang and vLLM rollout engines.

---

## Supported Frameworks

| Directory | Framework | Backend | Rollout | Description |
|-----------|-----------|---------|---------|-------------|
| `examples/rl/trl/` | [TRL](https://github.com/huggingface/trl) | FSDP1/FSDP2/DDP | Built-in | GRPO training via HuggingFace TRL GRPOTrainer |
| `examples/rl/verl/` | [VERL](https://github.com/volcengine/verl) | FSDP2 | SGLang / vLLM | GRPO training via VERL with async rollout |
| `examples/rl/verl/` | [VERL](https://github.com/volcengine/verl) | Megatron | SGLang / vLLM | GRPO training with Megatron parallel (TP/PP/EP) |

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

## Quick Start: VERL + FSDP2 + SGLang

### BF16 baseline

```bash
bash examples/rl/verl/run_grpo_fsdp2.sh
```

### With FP8 Param Manager (recommended, -25% VRAM)

```bash
FP8_PARAM_MANAGER=1 \
NUM_GPUS=4 ROLLOUT_TP=2 ROLLOUT_GPU_UTIL=0.1 \
TRAIN_BSZ=16 MAX_STEPS=2 \
bash examples/rl/verl/run_grpo_fsdp2.sh
```

### With FP8 + LoRA + FP8 Param Manager

```bash
LUMEN_FP8=1 FP8_PARAM_MANAGER=1 bash examples/rl/verl/run_grpo_fsdp2.sh
```

---

## Quick Start: VERL + FSDP2 + vLLM

vLLM V1 is now supported as a rollout engine on ROCm (requires `get_device_uuid` fix — see {doc}`/advance/rl_training`).

### BF16 baseline (4 GPU)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NUM_GPUS=4 ROLLOUT_TP=1 ROLLOUT_GPU_UTIL=0.4 \
TRAIN_BSZ=64 MAX_STEPS=2 \
bash examples/rl/verl/run_grpo_fsdp2_vllm.sh
```

### With FP8 Param Manager (4 GPU, no offload, best throughput)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FP8_PARAM_MANAGER=1 \
NUM_GPUS=4 ROLLOUT_TP=1 ROLLOUT_GPU_UTIL=0.4 \
TRAIN_BSZ=64 MAX_STEPS=2 \
bash examples/rl/verl/run_grpo_fsdp2_vllm.sh
```

**Note:** For vLLM rollout on ROCm, use `rollout.tensor_model_parallel_size=1` (TP≥2 may hang), `rollout.free_cache_engine=false`, and `rollout.enforce_eager=true`.

---

## Quick Start: VERL + Megatron

### Megatron + SGLang (BF16)

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NUM_GPUS=4 ACTOR_TP=2 ROLLOUT_TP=2 ROLLOUT_GPU_UTIL=0.1 \
TRAIN_BSZ=16 MAX_STEPS=2 \
PARAM_OFFLOAD=true OPTIMIZER_OFFLOAD=true \
bash examples/rl/verl/run_grpo_megatron_sglang.sh
```

### Megatron + SGLang (FP8PM, on-the-fly quantization)

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FP8_PARAM_MANAGER=1 \
NUM_GPUS=4 ACTOR_TP=2 ROLLOUT_TP=2 ROLLOUT_GPU_UTIL=0.1 \
TRAIN_BSZ=16 MAX_STEPS=2 \
PARAM_OFFLOAD=true OPTIMIZER_OFFLOAD=true \
bash examples/rl/verl/run_grpo_megatron_sglang.sh
```

### Megatron + vLLM (BF16, rollout TP=1 on ROCm)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NUM_GPUS=4 ACTOR_TP=2 ROLLOUT_TP=1 ROLLOUT_GPU_UTIL=0.3 \
TRAIN_BSZ=16 MAX_STEPS=2 \
bash examples/rl/verl/run_grpo_megatron_vllm.sh
```

### Megatron + vLLM (FP8PM)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FP8_PARAM_MANAGER=1 \
NUM_GPUS=4 ACTOR_TP=2 ROLLOUT_TP=1 ROLLOUT_GPU_UTIL=0.3 \
TRAIN_BSZ=16 MAX_STEPS=2 \
PARAM_OFFLOAD=true OPTIMIZER_OFFLOAD=true \
bash examples/rl/verl/run_grpo_megatron_vllm.sh
```

**Note:** Megatron FP8PM uses on-the-fly BF16→FP8 quantization (in-place crashes the distributed optimizer). Requires patched `verl/workers/engine/megatron/transformer_impl.py`. Rollout TP must be 1 on ROCm for vLLM.

---

## Lumen FP8 Modes for RL

| Mode | API | Memory Saving | Speed Impact | Status |
|------|-----|---------------|--------------|--------|
| FP8 Linear GEMM | `quant.enable(model, ...)` | None (weights stay BF16) | 3-4× slower (per-forward quantize) | Validated |
| FP8 Weight Cache | `quant.store_weights_fp8(model)` | +7.5 GB overhead | 2.5× slower (cached quantize) | Validated |
| FP8 Attention (dpa) | `LumenConfig(fp8_attn="dpa")` | None with grad ckpt | Marginal | Validated |
| **FP8ParamManager (FSDP2)** | `FP8_PARAM_MANAGER=1` | **-25% peak (no offload)** | Neutral | **Validated** |
| **FP8ParamManager (FSDP2, offload)** | `FP8_PARAM_MANAGER=1` | **-5% peak (with offload)** | +18% throughput | **Validated** |
| **FP8ParamManager (Megatron)** | `FP8_PARAM_MANAGER=1` | **-29% peak** | -48% throughput (on-the-fly) | **Validated** |

---

## RL Algorithms

| Algorithm | Framework | Rollout | LoRA + FP8 Status |
|-----------|-----------|---------|-------------------|
| GRPO | TRL | Built-in | Validated |
| GRPO | VERL (FSDP2) | SGLang | Validated |
| GRPO | VERL (FSDP2) | vLLM | Validated |
| GRPO | VERL (Megatron) | SGLang | Validated |
| GRPO | VERL (Megatron) | vLLM | Validated |
| PPO | TRL / VERL | SGLang / vLLM | Supported |
| DAPO | TRL | Built-in | Supported |

**Note:** LoRA with Megatron is supported via Lumen's custom `MegatronLoraAdapter` — set `LORA_RANK=32` as an environment variable.

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
    ├── run_grpo_fsdp2.sh
    ├── run_grpo_fsdp2_vllm.sh
    ├── run_grpo_megatron_sglang.sh
    └── run_grpo_megatron_vllm.sh
```

---

## See Also

- {doc}`/advance/rl_training` -- RL training architecture, benchmarks, and ROCm fixes
- {doc}`/advance/rl_lora` -- RL Training with LoRA and FP8
- {doc}`/advance/feature_catalog` -- Complete Lumen feature inventory
