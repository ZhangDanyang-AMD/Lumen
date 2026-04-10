#!/usr/bin/env bash
# Experiment 2C: Qwen3-30B-A3B-Base MoE — Lumen FP8 E2E (FP8PM training + FP8 rollout + TIS)
#
# Uses Lumen's FP8ParamManager for training (on-the-fly FP8 quantization in autograd)
# and vLLM FP8 for rollout, with token-level TIS rollout correction.
# Same model as 2A/2B for direct comparison on one chart.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROJECT_NAME="FP8-ALIGN"
export EXP_NAME="2C-qwen3-30b-moe-fp8-e2e-lumen"
export MODEL_PATH="${MODEL_PATH:-/dev/shm/model/qwen3-30b-a3b-base}"

# Adapted for 8 GPUs
export TRAIN_BSZ="16"
export GEN_BSZ="48"
export MINI_BSZ="16"
export GPU_MEM_UTIL="0.5"
export SP_SIZE="4"
export OFFLOAD="true"
export ROLLOUT_QUANTIZATION="fp8"
export ROLLOUT_IS="token"
export ROLLOUT_IS_THRESHOLD="2.0"

# Lumen FP8 training
export FP8_PARAM_MANAGER="1"

source "${SCRIPT_DIR}/common.sh"
launch_training
