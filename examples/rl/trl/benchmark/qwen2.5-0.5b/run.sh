#!/usr/bin/env bash
###############################################################################
# Qwen2.5-0.5B-Instruct — model-specific config for Lumen FP8 GRPO benchmark
#
# Aligned with TRL GRPO Trainer defaults (TRL v0.29+):
#   Model:   Qwen/Qwen2.5-0.5B-Instruct
#   Dataset: trl-lib/DeepMath-103K
#   Reward:  accuracy_reward (requires \boxed{} in completions)
#
# Key tuning vs TRL defaults:
#   max_completion_length=1024 (TRL default 256 is too short for DeepMath —
#     model needs ~500-1000 tokens to complete reasoning + \boxed{answer})
#   Single GPU (0.5B model doesn't benefit from FSDP sharding)
#
# Usage:
#   bash examples/rl/trl/benchmark/qwen2.5-0.5b/run.sh R1
#   bash examples/rl/trl/benchmark/qwen2.5-0.5b/run.sh ALL
###############################################################################

export MODEL_DIR="${MODEL_DIR:-/dev/shm/model/qwen2.5-0.5b-instruct}"
export DATASET_NAME="${DATASET_NAME:-trl-lib/DeepMath-103K}"
export DATASET_SPLIT="${DATASET_SPLIT:-train}"
export OUTPUT_BASE="${OUTPUT_BASE:-$(cd "$(dirname "$0")/../../../../.." && pwd)/outputs/benchmark/qwen2.5-0.5b}"

export MAX_STEPS="${MAX_STEPS:-2000}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export LR="${LR:-1e-6}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
export BETA="${BETA:-0.0}"

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"

export EARLY_STOP="${EARLY_STOP:-0}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-100}"
export EVAL_AFTER_TRAINING="${EVAL_AFTER_TRAINING:-1}"
export EVAL_SAMPLES="${EVAL_SAMPLES:-200}"
export PERIODIC_EVAL="${PERIODIC_EVAL:-1}"
export PERIODIC_EVAL_EVERY="${PERIODIC_EVAL_EVERY:-100}"
export PERIODIC_EVAL_SAMPLES="${PERIODIC_EVAL_SAMPLES:-50}"

MODEL_CONFIG_DIR="$(cd "$(dirname "$0")" && pwd)"
export ACCEL_CONFIG="${ACCEL_CONFIG:-${MODEL_CONFIG_DIR}/single_gpu.yaml}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "${SCRIPT_DIR}/run_grpo_benchmark.sh" "${@}"
