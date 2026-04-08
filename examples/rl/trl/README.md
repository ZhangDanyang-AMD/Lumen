###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# TRL GRPO with Lumen

GRPO-first `TRL` integration for Lumen on HuggingFace + `FSDP/FSDP2`.

Current v1 scope:

- `GRPO` only
- LLaMA-family HuggingFace causal LMs
- Actor-side optional LoRA and linear FP8
- `beta=0.0` (no standalone reference-model path in v1)
- Built-in evaluation callback with per-step JSONL logging and curve plotting
- Validated end-to-end on LLaMA-2-70B across 8 AMD MI GPUs

## Install

```bash
python -m pip install -r examples/rl/trl/requirements.txt
```

Dependencies: `trl==0.29.1`, `transformers==5.3.0`, `accelerate==1.13.0`,
`datasets==4.8.4`, `peft==0.18.1`, `sentencepiece>=0.1.99`, `matplotlib>=3.7`.

## Quick Start

The shell launcher selects the Accelerate config from the trailing argument:

- `1` -> `examples/rl/trl/accelerate/fsdp1.yaml`
- `2` -> `examples/rl/trl/accelerate/fsdp2.yaml`

The smoke path defaults to `NUM_PROCESSES=2`, so it expects 2 visible GPUs.

```bash
# FSDP1 smoke
MODEL_NAME=hf-internal-testing/tiny-random-LlamaForCausalLM \
MAX_STEPS=2 \
bash examples/rl/trl/run_grpo_fsdp.sh 1

# FSDP2 smoke
MODEL_NAME=hf-internal-testing/tiny-random-LlamaForCausalLM \
MAX_STEPS=2 \
bash examples/rl/trl/run_grpo_fsdp.sh 2
```

The default dataset for the smoke launcher is `trl-lib/Capybara`.

## Local Prompt Dataset

For local data, point `TRAIN_DATA_PATH` at a JSON / JSONL file that `datasets`
can load with the `json` builder. The training dataset may expose a `prompt`
column or a `messages` column (chat-messages format). If only `messages` is
present, the first user message is automatically extracted as the prompt.

```bash
MODEL_NAME=hf-internal-testing/tiny-random-LlamaForCausalLM \
TRAIN_DATA_PATH=/data/rl_prompts.jsonl \
OUTPUT_DIR=./outputs/trl-grpo-local \
MAX_STEPS=2 \
bash examples/rl/trl/run_grpo_fsdp.sh 1
```

## Docker

A Dockerfile is provided for reproducible runs on ROCm/AMD GPUs. It builds on
`rocm/7.0` and installs all Lumen dependencies (AITER, mori, torchao,
Megatron-LM-AMD) plus TRL-specific requirements.

```bash
# Build the image
cd Lumen
docker build -t lumen-trl-test -f examples/rl/trl/Dockerfile .

# Run unit tests (default CMD)
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  lumen-trl-test
```

For training, mount the model directory and output volume:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --shm-size=128g \
  --security-opt seccomp=unconfined \
  -v /dev/shm/model:/dev/shm/model:ro \
  -v $(pwd)/outputs/trl-grpo-70b:/workspace/Lumen/outputs/trl-grpo-70b \
  -e NCCL_TIMEOUT=3600000 \
  -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 \
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  lumen-trl-test \
  python -m accelerate.commands.launch \
    --config_file examples/rl/trl/accelerate/fsdp1.yaml \
    --num_processes 8 \
    examples/rl/trl/run_grpo_fsdp.py \
    --model-name-or-path /dev/shm/model \
    --dataset-name trl-lib/Capybara \
    --output-dir outputs/trl-grpo-70b \
    --max-steps 30 --micro-batch-size 1 \
    --gradient-accumulation-steps 1 --num-generations 8 \
    --max-completion-length 256 --lr 5e-6 \
    --log-interval 1 --seed 1234
```

## LLaMA-2-70B Training

Start with `FSDP1` first, then try `FSDP2` after the stack is stable on your
target machine. The validated configuration below ran 30 steps in ~2 hours on
8 AMD MI GPUs.

### Lumen Path (recommended)

Uses `run_grpo_fsdp.py` which builds the actor model via Lumen's
`build_actor_model` (explicit `torch_dtype=bf16`, `attn_implementation="sdpa"`,
gradient checkpointing) and runs rollout through Transformers' default
`model.generate()`:

```bash
docker run ... lumen-trl-test \
  python -m accelerate.commands.launch \
    --config_file examples/rl/trl/accelerate/fsdp1.yaml \
    --num_processes 8 \
    examples/rl/trl/run_grpo_fsdp.py \
    --model-name-or-path /dev/shm/model \
    --dataset-name trl-lib/Capybara \
    --output-dir outputs/trl-grpo-70b \
    --max-steps 30 --micro-batch-size 1 \
    --gradient-accumulation-steps 1 --num-generations 8 \
    --max-completion-length 256 --lr 5e-6 \
    --log-interval 1 --seed 1234
```

### Baseline Path (pure TRL, no Lumen)

Uses `run_grpo_baseline.py` which passes the model name string directly to
`GRPOTrainer`, letting TRL handle model construction. Useful for A/B comparison
against the Lumen path:

```bash
docker run ... lumen-trl-test \
  python -m accelerate.commands.launch \
    --config_file examples/rl/trl/accelerate/fsdp1.yaml \
    --num_processes 8 \
    examples/rl/trl/run_grpo_baseline.py \
    --model-name-or-path /dev/shm/model \
    --dataset-name trl-lib/Capybara \
    --output-dir outputs/trl-grpo-70b-baseline \
    --max-steps 30 --micro-batch-size 1 \
    --gradient-accumulation-steps 1 --num-generations 8 \
    --max-completion-length 256 --lr 5e-6 \
    --log-interval 1 --seed 1234
```

### Optional Actor-Side Knobs

```bash
--linear-fp8          # Enable FP8 quantisation for linear layers
--lora-rank 16        # Apply LoRA with the given rank
--lora-alpha 32
--lora-dropout 0.1
```

## Evaluation and Metrics

The training loop registers a `GRPOEvalCallback` (defined in
`lumen/rl/trl/eval_callback.py`) that logs per-step metrics to
`grpo_eval_log.jsonl` in the output directory. Tracked metrics:

| Metric | Description |
|---|---|
| `reward` | Mean reward across the generation group |
| `entropy` | Token-level entropy (KL divergence proxy when `beta=0`) |
| `completions/mean_length` | Average completion length in tokens |
| `win_rate` | Fraction of recent steps (window=5) exceeding initial baseline reward |
| `loss` | GRPO policy gradient loss |
| `grad_norm` | Global gradient norm |

### Plotting

```bash
python lumen/rl/trl/plot_curves.py outputs/trl-grpo-70b
```

Generates `grpo_curves.png` in the output directory with 6-panel regression
curves (reward, entropy, length, win rate, loss, grad norm).

### Comparing Runs

After running both Lumen and baseline paths:

```bash
python examples/rl/trl/compare_runs.py \
  outputs/trl-grpo-70b outputs/trl-grpo-70b-baseline
```

Generates `compare_curves.png` (side-by-side 6-panel plot) and
`COMPARISON.md` (statistical summary with Pearson correlations) in the
baseline output directory.

## Validated Results

### LLaMA-2-70B Lumen vs Baseline (30 steps)

Both runs used identical configuration (seed=1234, 8 GPUs, FSDP1).

| Metric | Lumen Mean | Baseline Mean | Pearson r |
|---|---|---|---|
| Reward | 0.364 | 0.347 | 0.776 |
| Response Length | 162.6 | 166.3 | 0.835 |
| Entropy | 1.935 | 1.915 | 0.906 |
| Grad Norm | 5.081 | 4.925 | 0.826 |
| Win Rate | 0.730 | 0.728 | 0.865 |

Strong correlations across all metrics (r > 0.7 except loss) confirm that
Lumen's integration layer produces equivalent training dynamics to pure TRL.

### Llama-3.1-8B Memory Matrix (8x MI300X, 10 steps)

4-config comparison on `trl-lib/DeepMath-103K`, seed=1234, num_generations=4:

| Config | Distributed | Peak Mem/GPU | vs A | Avg Step | Reward |
|---|---|---|---|---|---|
| A) BF16 full | FSDP1 | 34.57 GB | baseline | 11.07s | 0.42→0.79 |
| B) BF16 LoRA r=16 | DDP | 17.83 GB | **-48%** | 9.56s | 0.41→0.43 |
| C) FP8 full | FSDP1 | 38.85 GB | +12% | 126.67s | 0.51→0.74 |
| D) FP8 LoRA r=16 | DDP | 20.64 GB | -40% | 185.28s | 0.48→0.47 |

**Key findings**: LoRA provides 48% memory savings. FP8 via `quant.enable()`
increases memory (+12%) and latency (11.5x) due to AITER Triton fallback when
FSDP1 upcasts params to FP32.

### Llama-3.1-8B Extended Lumen Optimizations (8x MI300X, 10 steps)

Additional Lumen FP8 features tested on FSDP1:

| Config | Peak Mem/GPU | vs BF16 | Speed |
|---|---|---|---|
| BF16 full (baseline) | 34.57 GB | — | 1x |
| FP8 Linear only | 38.85 GB | +12% | 0.10x |
| FP8 Linear + FP8 Attn (dpa) | **30.92 GB** | **-11%** | 0.08x |
| FP8 Linear + FP8 Attn + Act Store | 30.89 GB | -11% | 0.08x |
| FP8 Linear + FP8 Attn + Param Gather | CRASH | — | — |
| FP8 Linear + FP8 Attn + Lumen Norm | CRASH | — | — |

**Key findings**: FP8 Attention (dpa) is the only add-on that reduces memory
below BF16 baseline (-11%). Activation Store has no effect on HF models. Param
Gather and Lumen Norm crash on FSDP1 (need FSDP2 or AITER fixes).

### VERL Integration Status

VERL 0.7.1 integration scaffolding is in `lumen/rl/verl/` but cannot run on
this ROCm container because VERL removed co-located HF rollout in favor of
async server-based rollout (vllm/sglang). ROCm-compatible vllm or sglang is
required.

See [outputs/benchmark/BENCHMARK_RESULTS.md](../../outputs/benchmark/BENCHMARK_RESULTS.md)
for full analysis including Demo A, B, C, and D-1.

## Running the Benchmarks

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `hf-internal-testing/tiny-random-LlamaForCausalLM` | Model path or HF ID |
| `LINEAR_FP8` | `0` | `1` to enable FP8 linear quantization |
| `LORA_RANK` | `0` | LoRA rank (`0` = full fine-tune) |
| `NUM_PROCESSES` | `2` | Number of GPUs |
| `MAX_STEPS` | `30` | Training steps |
| `TRAIN_DATA_PATH` | `trl-lib/Capybara` | Dataset name or local path |
| `OUTPUT_DIR` | auto-generated | Output directory |
| `FSDP_VERSION` | `1` | `1` for FSDP1, `2` for FSDP2 |
| `LUMEN_FP8_ATTN` | `none` | FP8 attention mode (`none`, `dpa`, `mha`) |
| `LUMEN_NORM` | `0` | `1` to enable Lumen norm replacement |
| `LUMEN_FP8_ACTIVATION_STORE` | `0` | `1` to enable FP8 activation storage |
| `LUMEN_FP8_PARAM_GATHER` | `0` | `1` to enable FP8 param all-gather |

### BF16 Full Fine-Tune (FSDP1)

```bash
LINEAR_FP8=0 LORA_RANK=0 NUM_PROCESSES=8 MAX_STEPS=10 \
  MODEL_NAME=/dev/shm/model/llama-3.1-8b \
  TRAIN_DATA_PATH=trl-lib/DeepMath-103K \
  OUTPUT_DIR=./outputs/demo-b/bf16-full \
  bash examples/rl/trl/run_grpo_fsdp.sh 1
```

### BF16 + LoRA (DDP)

LoRA configs require DDP due to FSDP1+PEFT mixed-precision incompatibility:

```bash
accelerate launch \
  --config_file examples/rl/trl/accelerate/ddp.yaml \
  --num_processes 8 \
  examples/rl/trl/run_grpo_fsdp.py \
  --model-name-or-path /dev/shm/model/llama-3.1-8b \
  --dataset-name trl-lib/DeepMath-103K \
  --output-dir ./outputs/demo-b/bf16-lora \
  --max-steps 10 --micro-batch-size 1 --gradient-accumulation-steps 4 \
  --num-generations 4 --max-completion-length 256 --lr 5e-6 \
  --log-interval 1 --seed 1234 --lora-rank 16
```

### FP8 Full Fine-Tune (FSDP1)

```bash
LINEAR_FP8=1 LORA_RANK=0 NUM_PROCESSES=8 MAX_STEPS=10 \
  MODEL_NAME=/dev/shm/model/llama-3.1-8b \
  TRAIN_DATA_PATH=trl-lib/DeepMath-103K \
  OUTPUT_DIR=./outputs/demo-b/fp8-full \
  bash examples/rl/trl/run_grpo_fsdp.sh 1
```

### FP8 + LoRA (DDP)

```bash
LINEAR_FP8=1 accelerate launch \
  --config_file examples/rl/trl/accelerate/ddp.yaml \
  --num_processes 8 \
  examples/rl/trl/run_grpo_fsdp.py \
  --model-name-or-path /dev/shm/model/llama-3.1-8b \
  --dataset-name trl-lib/DeepMath-103K \
  --output-dir ./outputs/demo-b/fp8-lora \
  --max-steps 10 --micro-batch-size 1 --gradient-accumulation-steps 4 \
  --num-generations 4 --max-completion-length 256 --lr 5e-6 \
  --log-interval 1 --seed 1234 --lora-rank 16 --linear-fp8
```

### Advanced Benchmark (R1-R5 Configs)

The `benchmark/` folder contains a more comprehensive benchmark runner
supporting 5 configurations:

```bash
# Individual runs
bash examples/rl/trl/benchmark/llama-3.1-8b/run.sh R1  # BF16 baseline
bash examples/rl/trl/benchmark/llama-3.1-8b/run.sh R2  # Lumen BF16
bash examples/rl/trl/benchmark/llama-3.1-8b/run.sh R3  # FP8 Linear only
bash examples/rl/trl/benchmark/llama-3.1-8b/run.sh R4  # FP8 Full suite
bash examples/rl/trl/benchmark/llama-3.1-8b/run.sh R5  # FP8 Full + LoRA

# All core runs
bash examples/rl/trl/benchmark/llama-3.1-8b/run.sh ALL

# Analyze results
python examples/rl/trl/benchmark/analyze_benchmark.py outputs/benchmark/llama-3.1-8b
```

## Reward Function

The example `reward_fn()` in `run_grpo_fsdp.py` rewards concise, substantive
completions (sweet spot around 30-60 words). This is intentional: GRPO requires
reward variance across grouped completions to compute non-zero advantages.
Replace it with your task-specific reward for production training.

## Test Commands

```bash
# Fast unit + wiring suite
python -m pytest \
  tests/rl/trl/test_args.py \
  tests/rl/trl/test_modeling.py \
  tests/rl/trl/test_warmup.py \
  tests/rl/trl/test_runner_smoke.py \
  -v

# Opt-in distributed smoke
LUMEN_RUN_SLOW_RL_TESTS=1 python -m pytest tests/rl/trl/test_runner_smoke.py -v

# Docker-based unit tests
docker run --rm lumen-trl-test
```

## File Reference

### Example Scripts

| File | Description |
|---|---|
| `examples/rl/trl/run_grpo_fsdp.py` | Accelerate entrypoint for Lumen-integrated GRPO |
| `examples/rl/trl/run_grpo_fsdp.sh` | Shell launcher for smoke runs and bring-up |
| `examples/rl/trl/run_grpo_baseline.py` | Standalone TRL baseline (zero Lumen imports) |
| `examples/rl/trl/compare_runs.py` | Side-by-side comparison of two training runs |
| `examples/rl/trl/Dockerfile` | ROCm Docker image for reproducible runs |
| `examples/rl/trl/requirements.txt` | Python dependencies |

### Lumen Modules

| File | Description |
|---|---|
| `lumen/rl/trl/runner.py` | High-level `run_grpo()` function |
| `lumen/rl/trl/args.py` | Argument translation to `GRPOConfig` |
| `lumen/rl/trl/modeling.py` | `build_actor_model` (FP8, LoRA, gradient checkpointing) |
| `lumen/rl/trl/warmup.py` | Synthetic warmup for FSDP |
| `lumen/rl/trl/eval_callback.py` | `GRPOEvalCallback` for per-step JSONL logging |
| `lumen/rl/trl/plot_curves.py` | Matplotlib plotting for eval logs |

### VERL Integration (Scaffolding Only)

| File | Description |
|---|---|
| `lumen/rl/verl/config.py` | `VerlLumenArgs` — VERL-Lumen configuration contract |
| `lumen/rl/verl/fsdp_backend.py` | Lumen-aware FSDP model builders for VERL roles |
| `lumen/rl/verl/verl_entry.py` | Monkey-patching entrypoint for Lumen+VERL |
| `examples/rl/verl/run_grpo_fsdp2.sh` | VERL GRPO launcher with Hydra CLI overrides |

## Notes

- The Accelerate YAML files wrap `LlamaDecoderLayer`, so this path is intended for LLaMA-class models.
- `WARMUP_STEPS=0` is the default until the target `TRL + Accelerate + FSDP` stack is validated on the machine you care about.
- Rollout is pinned to Transformers' default `model.generate()` via `use_vllm=False`. No vLLM or paged-attention backends are used.
- For 70B training, `--shm-size=128g` and `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` are recommended to avoid OOM.
- The `fsdp1.yaml` config sets `num_processes: 2` by default; override with `--num_processes 8` for 8-GPU runs.
