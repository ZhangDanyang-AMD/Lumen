#!/usr/bin/env bash
###############################################################################
# Qwen2.5-0.5B-Instruct — 8-GPU DDP alignment test
#
# Purpose: Verify that 8-GPU distributed training produces the same
# learning curve as the 1-GPU baseline (R1). Uses DDP (MULTI_GPU),
# NOT FSDP, matching the TRL reference setup.
#
# Effective batch size: 8 GPUs * 1 prompt/GPU * 8 gens = 64 completions/step
# (same as 1 GPU * 8 prompts/GPU * 8 gens = 64 completions/step)
#
# Usage:
#   bash examples/rl/trl/benchmark/qwen2.5-0.5b/run_8gpu.sh R1
###############################################################################

export MODEL_DIR="${MODEL_DIR:-/dev/shm/model/qwen2.5-0.5b-instruct}"
export DATASET_NAME="${DATASET_NAME:-trl-lib/DeepMath-103K}"
export DATASET_SPLIT="${DATASET_SPLIT:-train}"
export OUTPUT_BASE="${OUTPUT_BASE:-$(cd "$(dirname "$0")/../../../../.." && pwd)/outputs/benchmark/qwen2.5-0.5b-8gpu}"

export MAX_STEPS="${MAX_STEPS:-500}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
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
export EVAL_AFTER_TRAINING="${EVAL_AFTER_TRAINING:-0}"
export PERIODIC_EVAL="${PERIODIC_EVAL:-0}"

MODEL_CONFIG_DIR="$(cd "$(dirname "$0")" && pwd)"
export ACCEL_CONFIG="${ACCEL_CONFIG:-${MODEL_CONFIG_DIR}/multi_gpu.yaml}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "${SCRIPT_DIR}/run_grpo_benchmark.sh" "${@}"
