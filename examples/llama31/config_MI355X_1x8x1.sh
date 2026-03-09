#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# MI355X 1-node 8-GPU configuration for LLaMA 3.1 pretraining.
#
# Usage:
#   bash run_pretrain.sh                                       # uses this config
#   CONFIG=config_MI300X_1x8x1.sh bash run_pretrain.sh         # uses MI300X config

# ---- Backend selection -------------------------------------------------------
export BACKEND="megatron"

# ---- Hardware ----------------------------------------------------------------
export NGPU=8
export NNODES=1
export GPUS_PER_NODE=${NGPU}

# ---- Model -------------------------------------------------------------------
export SIZE="8b"

# ---- Training hyperparameters ------------------------------------------------
export MBS=2
export GBS=32
export SEQ_LEN=8192
export MAX_LR="8e-4"
export MIN_LR="8e-5"
export TRAIN_STEPS=1200000
export LR_WARMUP_STEPS=128
export LOG_INTERVAL=1
export SAVE_INTERVAL=0
export EVAL_INTERVAL=0
export SAVE_DIR="/results/checkpoints"
export SEED=1234

# ---- Data / tokenizer --------------------------------------------------------
export TRAIN_DATA="/data/train.npy"
export VALID_DATA="/data/validation.npy"
export TOKENIZER="/model"

# ---- LoRA --------------------------------------------------------------------
export LORA_RANK=0
export LORA_ALPHA=32
export LORA_DROPOUT=0.1

# ---- FP8 training (MLPerf defaults) -----------------------------------------
export FP8_TRAINING=1
export FP8_FORMAT="fp8_e4m3"
export FP8_SCALING="delayed"
export FP8_BLOCK_SIZE=128
export FP8_AMAX_ALGO="most_recent"
export FP8_REDUCE_AMAX=0
export FP8_AMAX_HISTORY=4
export FP8_ACTIVATION=1
export GRAD_QUANT_TYPE=""

# ---- Warmup / early stopping -------------------------------------------------
export WARMUP_STEPS=0
export VAL_LOSS_TARGET=""

# ---- MXFP8 attention block sizes --------------------------------------------
export MXFP8_BLOCK_M_FWD=128
export MXFP8_BLOCK_N_FWD=128
export MXFP8_BLOCK_M_DQ_BWD=128
export MXFP8_BLOCK_N_DQ_BWD=128
export MXFP8_BLOCK_M_DKV_BWD=128
export MXFP8_BLOCK_N_DKV_BWD=128
export MXFP8_QUANT_BLOCK_SIZE=128

# ---- Checkpoint management (Docker run_and_time.sh compat) -------------------
export USE_CKPT=0
export FROM_HF=1
export SAVE_CKPT=0
export CONTINUAL_CKPT="/results/saved_ckpts"
export CKPT_START_STEP=0
export FP8_PARAMS=1

# ---- MLPerf / experiment management -----------------------------------------
export TAG=""
export TARGET_LOG_PPL="3.3"
export STEP_TIME_ATOL=18000

# ---- Evaluation (in training sequences, Docker convention) -------------------
export EVAL_EVERY=12288
export START_EVAL_AT=0

# ---- Primus Turbo attention --------------------------------------------------
export PRIMUS_FP8_ATTN=0
export PRIMUS_MXFP8_ATTN=1
export DBG_ATTN_OUTPUT=0

# ---- Megatron backend --------------------------------------------------------
export TP=1
export PP=1
export CP=1
export VP=0
export SP=0
export WEIGHT_DECAY=0.1
export GRADIENT_CLIP=1.0
export EVAL_ITERS=10
export PRECISION="bf16"
export CKPT_DIR="/model"
export LORA_A2A=0

# ---- FSDP backend ------------------------------------------------------------
export MODEL="/model"
export GRAD_ACCUM=8
export MAX_GRAD_NORM=1.0
export NUM_WORKERS=4
export TRAIN_SAMPLES=""
export VAL_SAMPLES=""
export SHARDING="full_shard"

# ---- MI355X platform ---------------------------------------------------------
export MLPERF_SUBMISSION_ORG="AMD"
export MLPERF_SUBMISSION_PLATFORM="MI355X"
