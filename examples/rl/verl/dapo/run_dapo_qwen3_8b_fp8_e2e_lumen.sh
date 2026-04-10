#!/usr/bin/env bash
# Experiment 1D: Qwen3-8B-Base — Lumen FP8 E2E (FP8PM training + FP8 rollout + TIS)
#
# Uses Lumen's FP8ParamManager for training (on-the-fly FP8 quantization in autograd)
# and vLLM FP8 for rollout, with token-level TIS rollout correction.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROJECT_NAME="FP8-ALIGN"
export EXP_NAME="1D-qwen3-8b-fp8-e2e-lumen"
export MODEL_PATH="${MODEL_PATH:-/dev/shm/model/qwen3-8b-base}"
export GPU_MEM_UTIL="0.9"
export ROLLOUT_QUANTIZATION="fp8"
export ROLLOUT_IS="token"
export ROLLOUT_IS_THRESHOLD="2.0"

# Lumen FP8 training
export FP8_PARAM_MANAGER="1"

source "${SCRIPT_DIR}/common.sh"
launch_training
