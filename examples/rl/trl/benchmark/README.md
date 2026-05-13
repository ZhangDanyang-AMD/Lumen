# Lumen GRPO Benchmark Suite

Benchmarks for Lumen FP8 optimizations in TRL GRPO training on AMD MI300X GPUs.

## Benchmark Scripts

### FP8 Memory Savings (`test_fp8_memory.py`)

Measures **actual GPU memory** for different FP8 configurations on a single GPU with a
short training loop (3 steps by default). Tests four configs:

1. **BF16 Baseline** — standard `AdamW` optimizer, `bfloat16` weights
2. **FP8ParamManager** — true FP8 weight storage (`float8_e4m3fn`) with dequant hooks
3. **FP8ParamManager + 8-bit Adam** — combines FP8 weights with `bitsandbytes.optim.Adam8bit`
4. **FP8 Attention (dpa)** — FP8 attention via `LumenConfig(fp8_attn="dpa")`

```bash
# Run all configs (single process)
python test_fp8_memory.py --model /dev/shm/model/llama-3.1-8b --configs all \
  --output fp8_memory_results.json

# For accurate per-config numbers, run each config separately:
python test_fp8_memory.py --model /dev/shm/model/llama-3.1-8b --configs bf16
python test_fp8_memory.py --model /dev/shm/model/llama-3.1-8b --configs fp8params
python test_fp8_memory.py --model /dev/shm/model/llama-3.1-8b --configs fp8_8bit
python test_fp8_memory.py --model /dev/shm/model/llama-3.1-8b --configs fp8attn
```

**Latest results** (Llama-3.1-8B, single MI300X, process-isolated):

| Config | Peak Alloc | vs BF16 |
|---|---|---|
| BF16 Baseline | 76,861 MB | baseline |
| FP8ParamManager | 28,909 MB | **-62.4%** |
| FP8Param + 8-bit Adam | 27,923 MB | **-63.7%** |
| FP8 Attention (dpa) | 76,860 MB | -0.0% |

See `outputs/benchmark/FP8_MEMORY_BENCHMARK.md` for the full report.

### GRPO Training Benchmark (`run_grpo_benchmark.py` / `run_grpo_benchmark.sh`)

Multi-run benchmark driver for full GRPO training on 8 GPUs. Supports five run configs:

| Run ID | Config |
|---|---|
| R1 | BF16 baseline |
| R2 | Lumen BF16 (with Lumen model building, no FP8) |
| R3 | FP8 linear only (`quant.enable`, dynamic scaling) |
| R4 | Full FP8 suite (linear + attention + activation store) |
| R5 | Full FP8 + LoRA |

```bash
# Run all configs for a model
cd llama-3.1-8b && bash run.sh ALL

# Run a specific config
cd llama-3.1-8b && bash run.sh R1

# Or directly
cd qwen2.5-0.5b && bash run.sh R1
```

### Post-Hoc Analysis (`analyze_benchmark.py`)

Reads `grpo_perf_log.jsonl` from training runs and prints performance tables,
memory stats, and generates figures.

```bash
python analyze_benchmark.py --log-dir outputs/benchmark/llama-3.1-8b/bf16_preloaded
```

### GSM8K Evaluation (`eval_checkpoints_gsm8k.py`)

Evaluate saved checkpoints on GSM8K for accuracy-vs-step curves.

```bash
python eval_checkpoints_gsm8k.py \
  --output-dir outputs/benchmark/llama-3.1-8b/bf16_preloaded \
  --base-model /dev/shm/model/llama-3.1-8b \
  --num-samples 200
```

## Model Configs

### Llama-3.1-8B (`llama-3.1-8b/`)

- 8.03B parameters, 32 layers, 4096 hidden
- Dataset: nvidia/OpenMathInstruct-2 (train_1M split)
- See [`llama-3.1-8b/README.md`](llama-3.1-8b/README.md) for results

Training scripts:

| Script | Description |
|---|---|
| `train_grpo_bf16_preloaded.py` | BF16 baseline |
| `train_grpo_lumen_fp8.py` | FP8 dynamic scaling |
| `train_grpo_lumen_fp8_stored.py` | FP8 cached weights |
| `train_grpo_lumen_fp8_blockwise.py` | FP8 blockwise scaling |

### Qwen2-0.5B (`qwen2.5-0.5b/`)

- 0.49B parameters, small model for fast iteration
- Dataset: trl-lib/DeepMath-103K
- See [`qwen2.5-0.5b/README.md`](qwen2.5-0.5b/README.md) for results

Training scripts:

| Script | Description |
|---|---|
| `train_grpo_bf16_preloaded.py` | BF16 baseline |
| `train_grpo_lumen_fp8.py` | FP8 dynamic scaling |
| `train_grpo_lumen_fp8_delayed.py` | FP8 delayed scaling |
| `train_grpo_lumen_fp8_blockwise.py` | FP8 blockwise scaling |
| `train_grpo_lumen_bf16.py` | Lumen model build, BF16 (env validation) |
| `train_grpo_deepmath_bf16.py` | Tuned BF16 on DeepMath |
| `train_grpo_deepmath_fp8.py` | Tuned FP8 on DeepMath |
| `train_grpo_doc_align.py` | TRL GRPO doc reference reproduction |

## Output Structure

Benchmark outputs go to `outputs/benchmark/<model>/`:

```
outputs/benchmark/
├── FP8_MEMORY_BENCHMARK.md            ← FP8 memory savings report
└── llama-3.1-8b/
    ├── bf16_preloaded_1000steps.jsonl  ← BF16 perf log
    ├── fp8_dynamic_1000steps.jsonl     ← FP8 dynamic perf log
    └── fp8_memory_test_all.json       ← FP8 memory benchmark raw data
```

## Environment Requirements

All benchmarks run inside the Lumen TRL Docker container:

```bash
docker build -t lumen-trl -f examples/rl/trl/Dockerfile ../../..
docker run --rm -it --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --shm-size 128G \
  -v /dev/shm/model:/dev/shm/model \
  lumen-trl bash
```

Key environment variables:

```bash
export TORCHDYNAMO_DISABLE=1     # Required: disable torch.compile
export CUDA_VISIBLE_DEVICES=0    # For single-GPU memory benchmarks
```

For 8-GPU training benchmarks, the `run.sh` scripts handle GPU configuration automatically.
