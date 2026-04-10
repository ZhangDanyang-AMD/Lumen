#!/usr/bin/env bash
# Experiment 2A: Qwen3-30B-A3B-Base MoE — BF16 training + BF16 rollout + TIS (baseline)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROJECT_NAME="FP8-ALIGN"
export EXP_NAME="2A-qwen3-30b-moe-bf16-rollout-bf16-tis"
export MODEL_PATH="${MODEL_PATH:-/dev/shm/model/qwen3-30b-a3b-base}"

# Adapted for 8 GPUs (reference used 16)
export TRAIN_BSZ="16"
export GEN_BSZ="48"
export MINI_BSZ="16"
export GPU_MEM_UTIL="0.5"
export SP_SIZE="4"
export OFFLOAD="true"
export ROLLOUT_QUANTIZATION="null"
export ROLLOUT_IS="token"
export ROLLOUT_IS_THRESHOLD="2.0"

source "${SCRIPT_DIR}/common.sh"
launch_training
