#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# MI300X 1-node 8-GPU configuration for LLaMA2-70B LoRA fine-tuning.
#
# Ported from Docker-src-mxfp8/llama2-lora-70b/config_MI300X_1x8x1.sh
# with NVTE/NeMo variables mapped to Transformer Light equivalents.
#
# Usage:
#   CONFIG=config_MI300X_1x8x1.sh bash run_finetune.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config_MI355X_1x8x1.sh"

# ---- Hardware ----------------------------------------------------------------
export NGPU=8
export NNODES=1

# ---- MI300X GPU performance tuning -------------------------------------------
export VBOOST_VALUE=1
export CUBLAS_FORCE_XMMA_KERNEL_INIT=DEVICE
export CK_FUSED_ATTN_LOG_CONFIG=0
export POSSIBLE_USER_WARNINGS=0
export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0

# ---- Training hyperparameters (MLPerf) ---------------------------------------
export BACKEND=megatron
export MODEL_SIZE="llama2-70B"
export TRAIN_STEPS=1024
export MBS=1
export GBS=8
export LR=0.0004
export EVAL_INTERVAL=384

# ---- Parallelism (single node, no model parallelism) -------------------------
export TP=1
export PP=1
export CP=1
export SP=0

# ---- LoRA (all-to-all for multi-GPU efficiency) -----------------------------
export LORA_RANK=16
export LORA_A2A=1

# ---- FP8 training (MLPerf MI300X defaults) -----------------------------------
export FP8_TRAINING=1
export FP8_FORMAT="fp8_e4m3"
export FP8_AMAX_ALGO="most_recent"
export FP8_REDUCE_AMAX=0
export FP8_AMAX_HISTORY=4
export FP8_ACTIVATION=1

# ---- Warmup ------------------------------------------------------------------
export WARMUP_STEPS=5

# ---- MLPerf submission -------------------------------------------------------
export MLPERF_SUBMISSION_ORG="AMD"
export MLPERF_SUBMISSION_PLATFORM="MI300X"
