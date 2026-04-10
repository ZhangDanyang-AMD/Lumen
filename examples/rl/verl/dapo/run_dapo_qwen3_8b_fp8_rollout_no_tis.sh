#!/usr/bin/env bash
# Experiment 1C: Qwen3-8B-Base — BF16 training + FP8 rollout (no TIS ablation)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROJECT_NAME="FP8-ALIGN"
export EXP_NAME="1C-qwen3-8b-bf16-rollout-fp8-no-tis"
export MODEL_PATH="${MODEL_PATH:-/dev/shm/model/qwen3-8b-base}"
export GPU_MEM_UTIL="0.9"
export ROLLOUT_QUANTIZATION="fp8"
export ROLLOUT_IS="null"

source "${SCRIPT_DIR}/common.sh"
launch_training
