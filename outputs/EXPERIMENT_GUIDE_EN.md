# Lumen TRL GRPO Experiment Guide — Reproduction Manual

## Hardware Environment

| Item | Specification |
|---|---|
| GPU | 8× AMD Instinct MI300X (192 GB HBM3, CDNA3 architecture) |
| Interconnect | xGMI fully connected, 64 GB/s theoretical per link |
| CPU | AMD EPYC |
| Memory | ≥ 512 GB DDR5 |
| Storage | NVMe SSD (loading models from `/dev/shm` is recommended for speed) |

## Software Environment

| Component | Version Requirement |
|---|---|
| ROCm | ≥ 6.x |
| PyTorch | ≥ 2.4 (required for FSDP2) |
| TRL | ≥ 0.16 |
| Accelerate | ≥ 1.6.0 |
| Transformers | ≥ 4.50 |
| Lumen | Current repository HEAD |

> **Note**: Exact version information at test time is automatically recorded to
> `env_info.json` by the benchmark runner (see `_collect_env_info()` in `run_grpo_benchmark.py`).

## Experiment Overview

This directory contains 4 completed experiments, all run on the hardware described above:

| Directory | Experiment | FSDP | Actor Build | Compared Against |
|---|---|---|---|---|
| `trl-grpo-70b/` | **Lumen + FSDP1** — primary experiment | v1 | Lumen `build_actor_model()` | — |
| `trl-grpo-70b-baseline/` | **Pure TRL + FSDP1** — BF16 baseline | v1 | TRL default | vs `trl-grpo-70b/` |
| `trl-grpo-70b-fsdp2/` | **Lumen + FSDP2** — FSDP version comparison | v2 | Lumen `build_actor_model()` | vs `trl-grpo-70b/` |
| `trl-grpo-70b-baseline-fsdp2/` | **Pure TRL + FSDP2** — FSDP2 baseline | v2 | TRL default | vs `trl-grpo-70b-fsdp2/` |

---

## Shared Training Parameters

All 4 experiments use identical training hyperparameters:

| Parameter | Value |
|---|---|
| Model | `NousResearch/Llama-2-70b-hf` (70B) |
| Training Steps | 30 |
| Random Seed | 1234 |
| GPU Count | 8 |
| Micro Batch Size | 1 |
| Gradient Accumulation | 1 |
| Num Generations | 8 |
| Max Completion Length | 256 tokens |
| Max Prompt Length | 512 tokens |
| Learning Rate | 5e-6 (linear decay) |
| Beta (KL penalty) | 0.0 |
| Gradient Checkpointing | ON |
| Dataset | `trl-lib/Capybara` |
| Reward Function | Word-count conciseness reward (see below) |

**Reward Function Definition** (located in `examples/rl/trl/run_grpo_fsdp.py`):

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

## Script Reference

### Training Scripts

| File | Purpose |
|---|---|
| `examples/rl/trl/run_grpo_fsdp.sh` | Shell launcher; accepts FSDP version argument (1 or 2); all hyperparameters configurable via environment variables |
| `examples/rl/trl/run_grpo_fsdp.py` | Python training script; builds actor model + TRL GRPOTrainer + training loop |
| `examples/rl/trl/accelerate/fsdp1.yaml` | Accelerate FSDP1 config (FULL_SHARD, BACKWARD_PRE) |
| `examples/rl/trl/accelerate/fsdp2.yaml` | Accelerate FSDP2 config (fully_shard API, reshard_after_forward) |

### Benchmark Scripts (for systematic FP8 performance comparison)

| File | Purpose |
|---|---|
| `examples/rl/trl/benchmark/run_grpo_benchmark.sh` | Benchmark launcher; supports R1-R5 configurations |
| `examples/rl/trl/benchmark/run_grpo_benchmark.py` | Benchmark runner; automatically records environment info and performance metrics |

Benchmark Run ID definitions:

| Run ID | Configuration | FP8 | LoRA | Purpose |
|---|---|---|---|---|
| R1 | Pure TRL baseline | OFF | OFF | BF16 performance baseline |
| R2 | Lumen BF16 | OFF | OFF | Measure Lumen framework overhead |
| R3 | Lumen FP8 Linear | Linear only | OFF | FP8 GEMM acceleration |
| R4 | Lumen FP8 Full | act+wgrad+reduce | OFF | Full FP8 optimization suite |
| R5 | Lumen FP8 + LoRA | act+wgrad+reduce | r=32 | FP8 + LoRA combination |

### Visualization Scripts

| File | Purpose | Output |
|---|---|---|
| `lumen/rl/trl/plot_curves.py` | Plot 6-panel training curves for a single run | `grpo_curves.png` |
| `examples/rl/trl/compare_runs.py` | Plot comparison curves for two runs + generate COMPARISON.md | `compare_curves.png` + `COMPARISON.md` |

### Callbacks and Utilities

| File | Purpose |
|---|---|
| `lumen/rl/trl/eval_callback.py` | Records per-step metrics to `grpo_eval_log.jsonl` during training |
| `lumen/rl/trl/perf_callback.py` | Performance callback for benchmarks; records step time, memory, throughput |
| `lumen/rl/trl/patched_trainer.py` | Patched GRPOTrainer with Lumen FP8 lifecycle support |

---

## Reproduction Steps

### Step 1: Environment Setup

```bash
# Verify ROCm and PyTorch are available
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"

# Verify TRL and Accelerate versions
python -c "import trl; print(trl.__version__)"
python -c "import accelerate; print(accelerate.__version__)"
```

### Step 2: Model Download

```bash
# Download model to /dev/shm for faster loading (recommended)
huggingface-cli download NousResearch/Llama-2-70b-hf --local-dir /dev/shm/model/llama-2-70b
```

### Step 3: Run Training

All experiments are executed from the Lumen repository root.

#### Experiment A: Lumen + FSDP1 (corresponds to `trl-grpo-70b/`)

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

`run_grpo_fsdp.sh` internally invokes `examples/rl/trl/run_grpo_fsdp.py`,
which builds the actor via Lumen's `build_actor_model()` + `run_grpo()`.

#### Experiment B: Pure TRL + FSDP1 Baseline (corresponds to `trl-grpo-70b-baseline/`)

The baseline experiment uses the benchmark runner's **R1** mode (no Lumen imports; passes model name directly to TRL GRPOTrainer):

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

> **R1 vs Lumen difference**: R1 does not import any Lumen code. It passes the model name
> string directly to `GRPOTrainer`, letting TRL handle model loading and FSDP wrapping.

#### Experiment C: Lumen + FSDP2 (corresponds to `trl-grpo-70b-fsdp2/`)

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

#### Experiment D: Pure TRL + FSDP2 Baseline (corresponds to `trl-grpo-70b-baseline-fsdp2/`)

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

### Step 4: Generate Visualizations

After training completes, each output directory will contain `grpo_eval_log.jsonl`. Run these commands to generate plots:

```bash
# Generate per-run training curves
python -m lumen.rl.trl.plot_curves outputs/trl-grpo-70b
python -m lumen.rl.trl.plot_curves outputs/trl-grpo-70b-baseline
python -m lumen.rl.trl.plot_curves outputs/trl-grpo-70b-fsdp2
python -m lumen.rl.trl.plot_curves outputs/trl-grpo-70b-baseline-fsdp2
```

```bash
# Comparison 1: Lumen + FSDP1 vs Pure TRL + FSDP1 (baseline)
# Output written to dir_b = outputs/trl-grpo-70b-baseline/
python examples/rl/trl/compare_runs.py \
    outputs/trl-grpo-70b \
    outputs/trl-grpo-70b-baseline \
    --label-a "Lumen" --label-b "Baseline (TRL)"

# Comparison 2: Lumen + FSDP1 vs Lumen + FSDP2
# Output written to dir_b = outputs/trl-grpo-70b-fsdp2/
python examples/rl/trl/compare_runs.py \
    outputs/trl-grpo-70b \
    outputs/trl-grpo-70b-fsdp2 \
    --label-a "Lumen + FSDP1" --label-b "Lumen + FSDP2"

# Comparison 3: Lumen + FSDP2 vs Pure TRL + FSDP2 (baseline)
# Output written to dir_b = outputs/trl-grpo-70b-baseline-fsdp2/
python examples/rl/trl/compare_runs.py \
    outputs/trl-grpo-70b-fsdp2 \
    outputs/trl-grpo-70b-baseline-fsdp2 \
    --label-a "Lumen + FSDP2" --label-b "Baseline + FSDP2"
```

> **Note**: `compare_runs.py` writes `compare_curves.png` and `COMPARISON.md` to the **second directory** (`dir_b`).

### Step 5: Run Benchmarks (Optional — more systematic performance testing)

```bash
# R1: BF16 baseline
bash examples/rl/trl/benchmark/run_grpo_benchmark.sh R1

# R4: Lumen FP8 Full
bash examples/rl/trl/benchmark/run_grpo_benchmark.sh R4

# Or run all R1-R4 at once
bash examples/rl/trl/benchmark/run_grpo_benchmark.sh ALL
```

---

## Output File Reference

Each experiment directory contains the following files:

| File | Contents | Generated By |
|---|---|---|
| `grpo_eval_log.jsonl` | Per-step training metrics (reward, entropy, loss, grad_norm, length, win_rate) | Automatically written by `GRPOEvalCallback` during training |
| `grpo_curves.png` | 6-panel training curves plot | `python -m lumen.rl.trl.plot_curves <dir>` |
| `compare_curves.png` | Comparison curves for two runs (only in comparison directories) | `python examples/rl/trl/compare_runs.py <dir_a> <dir_b>` |
| `COMPARISON.md` | Comparison statistics report (Pearson r, mean, std) | Same as above, auto-generated |
| `ANALYSIS.md` | Detailed experiment analysis (only in `trl-grpo-70b/`) | Manually written |

Additional outputs from benchmark experiments:

| File | Contents |
|---|---|
| `env_info.json` | Hardware/software environment snapshot (GPU name, ROCm version, PyTorch version, Lumen commit, etc.) |
| `grpo_perf_log.jsonl` | Per-step performance metrics (step_time, peak_memory, tokens) |
| `perf_summary.json` | Performance summary statistics (mean +/- std, excluding warmup steps) |
| `run.log` | Complete training stdout/stderr log |

---

## Environment Variable Reference

All environment variables supported by `run_grpo_fsdp.sh`:

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `hf-internal-testing/tiny-random-LlamaForCausalLM` | Model path or HuggingFace ID |
| `OUTPUT_DIR` | `outputs/trl-grpo-smoke` | Output directory |
| `NUM_PROCESSES` | 2 | Number of GPUs |
| `MICRO_BATCH_SIZE` | 1 | Micro batch size per GPU per step |
| `GRAD_ACCUM` | 1 | Gradient accumulation steps |
| `MAX_STEPS` | 2 | Total training steps |
| `LR` | 5e-6 | Learning rate |
| `LR_WARMUP_STEPS` | 0 | Learning rate warmup steps |
| `MAX_PROMPT_LENGTH` | 1024 | Maximum prompt length |
| `MAX_COMPLETION_LENGTH` | 512 | Maximum generation length |
| `NUM_GENERATIONS` | 4 | Generations per prompt |
| `SEED` | 1234 | Random seed |
| `LINEAR_FP8` | 0 | 1=enable FP8 Linear |
| `LORA_RANK` | 0 | LoRA rank (0=disabled) |
| `LORA_ALPHA` | 32 | LoRA alpha |
| `DATASET_NAME` | `trl-lib/Capybara` | HuggingFace dataset name |
| `TRAIN_DATA_PATH` | (empty) | Local JSONL data path (takes priority over DATASET_NAME) |
| `TOKENIZER_NAME_OR_PATH` | (empty) | Custom tokenizer path (defaults to model's built-in) |
| `WARMUP_STEPS` | 0 | Lumen synthetic FP8 warmup steps |
| `LOG_INTERVAL` | 1 | Logging interval (steps) |
| `SAVE_INTERVAL` | 0 | Checkpoint save interval (0=no saving) |
| `SEQ_LENGTH` | (empty) | Sequence length (default = MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH) |
| `LORA_DROPOUT` | 0.1 | LoRA dropout |

---

## Test Flow Summary

```
Step 1: Environment Setup
  ├── Verify ROCm / PyTorch / TRL / Accelerate
  └── Confirm GPU availability (8× MI300X)

Step 2: Model Download
  └── huggingface-cli download → /dev/shm/model/

Step 3: Training (4 experiments, identical hyperparameters)
  ├── A: Lumen + FSDP1  →  outputs/trl-grpo-70b/
  ├── B: Baseline + FSDP1  →  outputs/trl-grpo-70b-baseline/
  ├── C: Lumen + FSDP2  →  outputs/trl-grpo-70b-fsdp2/
  └── D: Baseline + FSDP2  →  outputs/trl-grpo-70b-baseline-fsdp2/
      ~2 hours each (~240s/step × 30 steps)

Step 4: Visualization
  ├── plot_curves.py  →  generates grpo_curves.png per directory
  └── compare_runs.py →  generates compare_curves.png + COMPARISON.md

Step 5 (Optional): Benchmark (R1-R5)
  └── More systematic FP8 performance comparison
```

## `grpo_eval_log.jsonl` Field Reference

One JSON object per line, one entry per training step. Key fields:

| Field | Type | Description |
|---|---|---|
| `step` | int | Training step number |
| `reward` / `rewards/reward_fn/mean` | float | Mean reward across 8 completions for the current step |
| `reward_std` / `rewards/reward_fn/std` | float | Reward standard deviation |
| `loss` | float | GRPO policy loss |
| `grad_norm` | float | Gradient norm |
| `entropy` | float | Token distribution entropy |
| `completions/mean_length` | float | Mean generation length (tokens) |
| `completions/clipped_ratio` | float | Fraction of completions truncated at max_length |
| `step_time` | float | Time for current step (seconds) |
| `num_tokens` | float | Cumulative tokens processed |
| `learning_rate` | float | Current learning rate |
| `win_rate` | float | Rolling 5-step window win rate vs first-3-steps baseline |

---

## Known Issues and Notes

1. **Memory configuration**: The 70B model requires `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` to avoid OOM from memory fragmentation.

2. **FSDP2 entropy overflow**: TRL's entropy computation occasionally produces BF16 numerical overflow under FSDP2, manifesting as extreme outlier values (~1e23). This is a TRL/BF16 issue, not caused by Lumen. `plot_curves.py` and `compare_runs.py` have built-in MAD-based outlier filtering.

3. **Step 18 collapse**: During the 30-step training, step 18 may exhibit a collapse where the model generates extremely short completions (<5 tokens). This is a stochastic property of GRPO with length-based reward and occurs under both FSDP versions. The model recovers by the next step.

4. **Baseline actor build difference**: Lumen experiments build the actor via `build_actor_model()` (load + bf16 + sdpa + gradient checkpointing), while baseline experiments pass the model name string directly to TRL. The two approaches produce statistically equivalent training dynamics (Pearson r > 0.77; see each COMPARISON.md).
