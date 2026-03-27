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
- actor-side optional LoRA and linear FP8
- `beta=0.0` (no standalone reference-model path in v1)
- no `DeepSpeed`, `vLLM`, or Megatron backend in this folder

## Install

```bash
python -m pip install -r examples/rl/trl/requirements.txt
```

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
can load with the `json` builder. The training dataset must expose a `prompt`
column because `GRPOTrainer` expects it.

```bash
MODEL_NAME=hf-internal-testing/tiny-random-LlamaForCausalLM \
TRAIN_DATA_PATH=/data/rl_prompts.jsonl \
OUTPUT_DIR=./outputs/trl-grpo-local \
MAX_STEPS=2 \
bash examples/rl/trl/run_grpo_fsdp.sh 1
```

## LLaMA2-70B Bring-Up

Start with `FSDP1` first, then try `FSDP2` after the stack is stable on your
target machine.

```bash
MODEL_NAME=meta-llama/Llama-2-70b-hf \
TOKENIZER_NAME_OR_PATH=meta-llama/Llama-2-70b-hf \
TRAIN_DATA_PATH=/data/rl_prompts.jsonl \
OUTPUT_DIR=/results/trl-grpo-70b \
NUM_PROCESSES=8 \
MICRO_BATCH_SIZE=1 \
GRAD_ACCUM=8 \
MAX_STEPS=100 \
MAX_PROMPT_LENGTH=1024 \
MAX_COMPLETION_LENGTH=512 \
SEQ_LENGTH=1536 \
WARMUP_STEPS=0 \
bash examples/rl/trl/run_grpo_fsdp.sh 1
```

Optional actor-side knobs:

```bash
LINEAR_FP8=1
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.1
```

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
```

## Notes

- `examples/rl/trl/run_grpo_fsdp.py` is the direct Accelerate entrypoint.
- `examples/rl/trl/run_grpo_fsdp.sh` is the canonical launcher for smoke runs and bring-up.
- `WARMUP_STEPS=0` is the default until the target `TRL + Accelerate + FSDP` stack is validated on the machine you care about.
- The Accelerate YAML files wrap `LlamaDecoderLayer`, so this path is intended for LLaMA-class models.
- The example `reward_fn()` in `run_grpo_fsdp.py` is only a placeholder constant reward. Replace it before real training.
