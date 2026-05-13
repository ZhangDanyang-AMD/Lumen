# Lumen RL Examples

Reinforcement Learning training examples using Lumen FP8 optimizations on AMD MI300X GPUs.

## Frameworks

| Directory | Framework | Backend | Description |
|---|---|---|---|
| [`trl/`](trl/) | [TRL](https://github.com/huggingface/trl) | FSDP1/FSDP2/DDP | GRPO training via HuggingFace TRL GRPOTrainer |
| [`verl/`](verl/) | [VERL](https://github.com/volcengine/verl) | FSDP2 + sglang | GRPO training via VERL with async sglang rollout |

## Quick Start

### TRL (recommended for single-node)

```bash
# Build the Docker image
cd examples/rl/trl
docker build -t lumen-trl -f Dockerfile ../../..

# Run container
docker run --rm -it --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --shm-size 128G \
  -v /dev/shm/model:/dev/shm/model \
  lumen-trl bash

# Inside container — BF16 baseline (8-GPU DDP, Qwen2-0.5B)
cd examples/rl/trl/benchmark/qwen2.5-0.5b
bash run.sh R1

# Inside container — FP8 memory benchmark (single GPU, Llama-3.1-8B)
cd examples/rl/trl/benchmark
python test_fp8_memory.py --model /dev/shm/model/llama-3.1-8b --configs all
```

### VERL (for async rollout with sglang)

```bash
# Inside the Lumen Docker container with VERL + sglang installed
bash examples/rl/verl/run_grpo_fsdp2.sh

# With FP8
LUMEN_FP8=1 bash examples/rl/verl/run_grpo_fsdp2.sh
```

## Lumen FP8 Modes

| Mode | API | Memory Saving | Speed Impact |
|---|---|---|---|
| FP8 Linear GEMM | `quant.enable(model, ...)` | None (weights stay BF16) | 3-4x slower (per-forward quantize) |
| FP8 Weight Cache | `quant.store_weights_fp8(model)` | +7.5 GB overhead | 2.5x slower (cached quantize) |
| FP8 Attention (dpa) | `LumenConfig(fp8_attn="dpa")` | None with grad ckpt | Marginal |
| **FP8ParamManager** | `FP8ParamManager.quantize_params(model)` | **-62% peak** | Needs validation |
| **FP8Param + 8-bit Adam** | FP8ParamManager + `bnb.optim.Adam8bit` | **-64% peak** | Needs validation |

See [`outputs/benchmark/FP8_MEMORY_BENCHMARK.md`](../../outputs/benchmark/FP8_MEMORY_BENCHMARK.md) for detailed numbers.

## Directory Layout

```
examples/rl/
├── README.md                          ← this file
├── trl/
│   ├── README.md                      ← TRL guide
│   ├── Dockerfile                     ← Full ROCm + Lumen + TRL image
│   ├── requirements.txt               ← TRL pip dependencies
│   ├── run_grpo_baseline.py           ← Pure TRL baseline (no Lumen)
│   ├── compare_runs.py                ← Compare Lumen vs baseline logs
│   ├── accelerate/
│   │   ├── fsdp1.yaml                 ← FSDP1 accelerate config
│   │   └── fsdp2.yaml                 ← FSDP2 accelerate config
│   └── benchmark/
│       ├── README.md                  ← Benchmark suite guide
│       ├── test_fp8_memory.py         ← FP8 memory savings benchmark
│       ├── run_grpo_benchmark.py      ← Multi-run benchmark driver
│       ├── run_grpo_benchmark.sh      ← Shared benchmark launcher
│       ├── analyze_benchmark.py       ← Post-hoc analysis
│       ├── eval_checkpoints_gsm8k.py  ← GSM8K checkpoint evaluation
│       ├── llama-3.1-8b/             ← 8B model configs + results
│       └── qwen2.5-0.5b/            ← 0.5B model configs + results
└── verl/
    └── run_grpo_fsdp2.sh             ← VERL FSDP2 + sglang launcher
```
