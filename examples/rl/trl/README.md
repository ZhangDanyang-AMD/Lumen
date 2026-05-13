# Lumen TRL Examples

GRPO reinforcement learning training using [TRL GRPOTrainer](https://github.com/huggingface/trl)
with Lumen FP8 optimizations on AMD MI300X GPUs.

## Prerequisites

- AMD MI300X GPUs (tested on 8-GPU nodes)
- ROCm 7.0+
- Docker (recommended) or manual installation of Lumen + TRL + AITER

## Docker Setup

Build the image from the Lumen repo root:

```bash
docker build -t lumen-trl -f examples/rl/trl/Dockerfile .
```

Run with GPU access:

```bash
docker run --rm -it --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --shm-size 128G \
  -v /dev/shm/model:/dev/shm/model \
  -v $(pwd)/outputs:/workspace/Lumen/outputs \
  lumen-trl bash
```

The image includes: ROCm PyTorch, AITER (AMD FP8 kernels), Mori (SDMA comm),
Megatron-LM-AMD, Lumen, TRL, and all benchmark dependencies.

## Scripts

| Script | Description |
|---|---|
| `run_grpo_fsdp.sh` | Main launcher for TRL GRPO with Lumen (FSDP1/FSDP2/DDP) |
| `run_grpo_fsdp.py` | Accelerate entrypoint for TRL GRPO |
| `run_grpo_baseline.py` | Pure TRL GRPO baseline (no Lumen), for comparison |
| `compare_runs.py` | Compare Lumen vs baseline GRPO runs from JSONL logs |

## Accelerate Configs

| Config | Description |
|---|---|
| `accelerate/fsdp1.yaml` | FSDP1 with `FULL_SHARD`, LlamaDecoderLayer wrapping |
| `accelerate/fsdp2.yaml` | FSDP2 with `reshard_after_forward`, CPU-efficient loading |
| `accelerate/ddp.yaml` | DDP (`MULTI_GPU`) — required for FP8ParamManager |

## FP8 Memory Savings

FP8ParamManager replaces `nn.Linear` weights with FP8 tensors (1 byte vs 2 bytes),
reducing peak memory by **61-62%** on DDP. See
[outputs/benchmark/FP8_MEMORY_SAVINGS_REPORT.md](../../outputs/benchmark/FP8_MEMORY_SAVINGS_REPORT.md)
for full results.

```bash
# BF16 baseline (DDP, 8-GPU)
ACCEL_CONFIG=examples/rl/trl/accelerate/ddp.yaml \
MODEL_NAME=/dev/shm/model/llama-3.1-8b NUM_PROCESSES=8 MAX_STEPS=20 \
bash examples/rl/trl/run_grpo_fsdp.sh 1

# FP8ParamManager (DDP, 8-GPU) — 61% memory savings
ACCEL_CONFIG=examples/rl/trl/accelerate/ddp.yaml \
FP8_PARAM_MANAGER=1 MODEL_NAME=/dev/shm/model/llama-3.1-8b NUM_PROCESSES=8 \
bash examples/rl/trl/run_grpo_fsdp.sh 1

# FP8ParamManager + 8-bit Adam (DDP, 8-GPU) — 62% memory savings
ACCEL_CONFIG=examples/rl/trl/accelerate/ddp.yaml \
FP8_PARAM_MANAGER=1 USE_8BIT_ADAM=1 MODEL_NAME=/dev/shm/model/llama-3.1-8b \
NUM_PROCESSES=8 bash examples/rl/trl/run_grpo_fsdp.sh 1
```

**Important**: FP8ParamManager requires DDP (not FSDP) because FSDP cannot handle
mixed FP8/BF16 parameter dtypes. FP8ParamManager freezes all `nn.Linear` weights;
only norms, embeddings, and lm_head are trainable.

## Benchmarks

See [`benchmark/README.md`](benchmark/README.md) for the full benchmark suite.

Quick examples:

```bash
# FP8 memory savings benchmark (single GPU)
cd benchmark
python test_fp8_memory.py --model /dev/shm/model/llama-3.1-8b --configs all

# Full GRPO training benchmark (8-GPU DDP, Llama-3.1-8B)
cd benchmark/llama-3.1-8b
bash run.sh ALL

# Full GRPO training benchmark (8-GPU DDP, Qwen2-0.5B)
cd benchmark/qwen2.5-0.5b
bash run.sh ALL
```

## Lumen Integration Points

The Lumen TRL integration lives in `lumen/rl/trl/` and provides:

| Module | Purpose |
|---|---|
| `lumen.rl.trl.args` | `TrlLumenArgs` dataclass — stable contract for model/data/FSDP/FP8 flags |
| `lumen.rl.trl.modeling` | `build_actor_model`, `build_reference_model`, `build_reward_model` with FP8 support |
| `lumen.rl.trl.runner` | `run_grpo(args, reward_fn=...)` — end-to-end GRPO runner |
| `lumen.rl.trl.warmup` | Synthetic warmup steps to stabilize FP8 scaling |
| `lumen.rl.trl.patched_trainer` | `PatchedGRPOTrainer` with GPU-synced stage timings |
| `lumen.rl.trl.perf_callback` | `GRPOPerfCallback` — JSONL perf logging |
| `lumen.rl.trl.eval_callback` | `GRPOEvalCallback` — JSONL eval logging with win-rate proxy |
| `lumen.rl.trl.plot_curves` | `python -m lumen.rl.trl.plot_curves <output_dir>` — plot training curves |

## Key Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TORCHDYNAMO_DISABLE` | `1` (recommended) | Disable torch.compile (ROCm Triton issues) |
| `HIP_VISIBLE_DEVICES` | `0,1,...,7` | GPU visibility |
| `MORI_ENABLE_SDMA` | `0` | Enable SDMA communication backend |
| `USE_HIPBLASLT` | `1` | Use hipBLASLt for BF16 GEMM |
| `FP8_PARAM_MANAGER` | `0` | Enable FP8ParamManager (true memory savings, requires DDP) |
| `USE_8BIT_ADAM` | `0` | Use bitsandbytes Adam8bit optimizer |
| `ACCEL_CONFIG` | auto | Override accelerate config path (use `ddp.yaml` for FP8PM) |
| `LINEAR_FP8` | `0` | Enable FP8 compute via `quant.enable` (not memory savings) |
| `LORA_RANK` | `0` | LoRA rank (0 = disabled) |
