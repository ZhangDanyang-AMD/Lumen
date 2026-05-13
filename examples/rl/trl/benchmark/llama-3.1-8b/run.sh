#!/usr/bin/env bash
###############################################################################
# LLaMA 3.1 8B — model-specific config for Lumen FP8 GRPO benchmark
#
# Usage:
#   bash examples/rl/trl/benchmark/llama-3.1-8b/run.sh R1
#   bash examples/rl/trl/benchmark/llama-3.1-8b/run.sh ALL
###############################################################################

export MODEL_DIR="${MODEL_DIR:-/dev/shm/model/llama-3.1-8b}"
export DATASET_NAME="${DATASET_NAME:-nvidia/OpenMathInstruct-2}"
export DATASET_SPLIT="${DATASET_SPLIT:-train_1M}"
export OUTPUT_BASE="${OUTPUT_BASE:-$(cd "$(dirname "$0")/../../../../.." && pwd)/outputs/benchmark/llama-3.1-8b}"

export MAX_STEPS="${MAX_STEPS:-200}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export LR="${LR:-5e-7}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-2}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant_with_warmup}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
export BETA="${BETA:-0.01}"

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"

export EARLY_STOP="${EARLY_STOP:-0}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-50}"
export EVAL_AFTER_TRAINING="${EVAL_AFTER_TRAINING:-0}"
export PERIODIC_EVAL="${PERIODIC_EVAL:-0}"
export PERIODIC_EVAL_EVERY="${PERIODIC_EVAL_EVERY:-50}"
export PERIODIC_EVAL_SAMPLES="${PERIODIC_EVAL_SAMPLES:-50}"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "${SCRIPT_DIR}/run_grpo_benchmark.sh" "${@}"
