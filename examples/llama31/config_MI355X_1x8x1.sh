#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# MI355X 1-node 8-GPU configuration for LLaMA 3.1 pretraining.
#
# Usage:
#   bash run_pretrain.sh                                       # uses this config
#   CONFIG=config_MI300X_1x8x1.sh bash run_pretrain.sh         # uses MI300X config
#   GBS=64 bash run_pretrain.sh                                # override individual vars
#
# All variables use ${VAR:-default} so they can be overridden by exporting
# them before sourcing this file.

# ---- Backend selection -------------------------------------------------------
export BACKEND=${BACKEND:-"megatron"}

# ---- Hardware ----------------------------------------------------------------
export NGPU=${NGPU:-8}
export NNODES=${NNODES:-1}
export GPUS_PER_NODE=${GPUS_PER_NODE:-${NGPU}}

# ---- Model -------------------------------------------------------------------
export SIZE=${SIZE:-"8b"}

# ---- Training hyperparameters ------------------------------------------------
export MBS=${MBS:-2}
export GBS=${GBS:-32}
export SEQ_LEN=${SEQ_LEN:-8192}
export MAX_LR=${MAX_LR:-"8e-4"}
export MIN_LR=${MIN_LR:-"8e-5"}
export TRAIN_STEPS=${TRAIN_STEPS:-1200000}
export LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-128}
export LOG_INTERVAL=${LOG_INTERVAL:-1}
export SAVE_INTERVAL=${SAVE_INTERVAL:-0}
export EVAL_INTERVAL=${EVAL_INTERVAL:-0}
export SAVE_DIR=${SAVE_DIR:-"/results/checkpoints"}
export SEED=${SEED:-1234}

# ---- Data / tokenizer --------------------------------------------------------
export TRAIN_DATA=${TRAIN_DATA:-"/data/train.jsonl"}
export VALID_DATA=${VALID_DATA:-""}
export TOKENIZER=${TOKENIZER:-"meta-llama/Llama-3.1-8B"}

# ---- LoRA --------------------------------------------------------------------
export LORA_RANK=${LORA_RANK:-0}
export LORA_ALPHA=${LORA_ALPHA:-32}
export LORA_DROPOUT=${LORA_DROPOUT:-0.1}

# ---- FP8 training (MLPerf defaults) -----------------------------------------
export FP8_TRAINING=${FP8_TRAINING:-1}
export FP8_FORMAT=${FP8_FORMAT:-"fp8_e4m3"}
export FP8_SCALING=${FP8_SCALING:-"delayed"}
export FP8_BLOCK_SIZE=${FP8_BLOCK_SIZE:-128}
export FP8_AMAX_ALGO=${FP8_AMAX_ALGO:-"most_recent"}
export FP8_REDUCE_AMAX=${FP8_REDUCE_AMAX:-0}
export FP8_AMAX_HISTORY=${FP8_AMAX_HISTORY:-4}
export FP8_ACTIVATION=${FP8_ACTIVATION:-1}

# ---- Warmup / early stopping -------------------------------------------------
export WARMUP_STEPS=${WARMUP_STEPS:-0}
export VAL_LOSS_TARGET=${VAL_LOSS_TARGET:-""}

# ---- MXFP8 attention block sizes --------------------------------------------
export MXFP8_BLOCK_M_FWD=${MXFP8_BLOCK_M_FWD:-128}
export MXFP8_BLOCK_N_FWD=${MXFP8_BLOCK_N_FWD:-128}
export MXFP8_BLOCK_M_DQ_BWD=${MXFP8_BLOCK_M_DQ_BWD:-128}
export MXFP8_BLOCK_N_DQ_BWD=${MXFP8_BLOCK_N_DQ_BWD:-128}
export MXFP8_BLOCK_M_DKV_BWD=${MXFP8_BLOCK_M_DKV_BWD:-128}
export MXFP8_BLOCK_N_DKV_BWD=${MXFP8_BLOCK_N_DKV_BWD:-128}
export MXFP8_QUANT_BLOCK_SIZE=${MXFP8_QUANT_BLOCK_SIZE:-128}

# ---- Checkpoint management (Docker run_and_time.sh compat) -------------------
export USE_CKPT=${USE_CKPT:-0}
export FROM_HF=${FROM_HF:-1}
export SAVE_CKPT=${SAVE_CKPT:-0}
export CONTINUAL_CKPT=${CONTINUAL_CKPT:-"/data/model/saved_ckpts"}
export CKPT_START_STEP=${CKPT_START_STEP:-0}
export FP8_PARAMS=${FP8_PARAMS:-1}

# ---- MLPerf / experiment management -----------------------------------------
export TAG=${TAG:-""}
export TARGET_LOG_PPL=${TARGET_LOG_PPL:-"3.3"}
export STEP_TIME_ATOL=${STEP_TIME_ATOL:-18000}

# ---- Evaluation (in training sequences, Docker convention) -------------------
export EVAL_EVERY=${EVAL_EVERY:-12288}
export START_EVAL_AT=${START_EVAL_AT:-0}

# ---- Primus Turbo attention --------------------------------------------------
export PRIMUS_FP8_ATTN=${PRIMUS_FP8_ATTN:-0}
export PRIMUS_MXFP8_ATTN=${PRIMUS_MXFP8_ATTN:-1}
export DBG_ATTN_OUTPUT=${DBG_ATTN_OUTPUT:-0}

# ---- Megatron backend --------------------------------------------------------
export TP=${TP:-1}
export PP=${PP:-1}
export CP=${CP:-1}
export VP=${VP:-0}
export SP=${SP:-0}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
export GRADIENT_CLIP=${GRADIENT_CLIP:-1.0}
export EVAL_ITERS=${EVAL_ITERS:-10}
export PRECISION=${PRECISION:-"bf16"}
export CKPT_DIR=${CKPT_DIR:-"/ckpt"}
export LORA_A2A=${LORA_A2A:-0}

# ---- FSDP backend ------------------------------------------------------------
export MODEL=${MODEL:-"meta-llama/Llama-3.1-8B"}
export GRAD_ACCUM=${GRAD_ACCUM:-8}
export MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
export NUM_WORKERS=${NUM_WORKERS:-4}
export TRAIN_SAMPLES=${TRAIN_SAMPLES:-""}
export VAL_SAMPLES=${VAL_SAMPLES:-""}
export SHARDING=${SHARDING:-"full_shard"}

# ---- MI355X platform ---------------------------------------------------------
export MLPERF_SUBMISSION_ORG=${MLPERF_SUBMISSION_ORG:-"AMD"}
export MLPERF_SUBMISSION_PLATFORM=${MLPERF_SUBMISSION_PLATFORM:-"MI355X"}
