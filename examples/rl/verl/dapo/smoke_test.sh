#!/usr/bin/env bash
# Quick 2-step smoke test for Qwen3-8B-Base to verify the DAPO pipeline works.
# Run this BEFORE launching full experiments.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROJECT_NAME="FP8-ALIGN-SMOKE"
export EXP_NAME="smoke-qwen3-8b-bf16"
export MODEL_PATH="${MODEL_PATH:-/dev/shm/model/qwen3-8b-base}"

# Minimal config for fast iteration
export TOTAL_STEPS="2"
export TEST_FREQ="1"
export SAVE_FREQ="999"
export TRAIN_BSZ="16"
export GEN_BSZ="16"
export MINI_BSZ="16"
export N_RESP="2"
export MAX_RESPONSE_LENGTH="512"
export GPU_MEM_UTIL="0.5"
export ROLLOUT_QUANTIZATION="null"
export ROLLOUT_IS="null"

source "${SCRIPT_DIR}/common.sh"
launch_training
