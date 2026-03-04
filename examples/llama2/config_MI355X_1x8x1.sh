#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# MI355X 1-node 8-GPU configuration for LLaMA2 SFT fine-tuning.
#
# Usage:
#   bash run_finetune.sh                                      # uses this config
#   CONFIG=config_MI300X_1x8x1.sh bash run_finetune.sh        # uses MI300X config
#   TRAIN_STEPS=500 bash run_finetune.sh                      # override individual vars
#
# All variables use ${VAR:-default} so they can be overridden by exporting
# them before sourcing this file.

# ---- Backend selection -------------------------------------------------------
export BACKEND=${BACKEND:-"megatron"}

# ---- Hardware ----------------------------------------------------------------
export NGPU=${NGPU:-8}
export NNODES=${NNODES:-1}

# ---- Training hyperparameters ------------------------------------------------
export MBS=${MBS:-1}
export SEQ_LEN=${SEQ_LEN:-8192}
export LR=${LR:-4e-4}
export MIN_LR=${MIN_LR:-0}
export TRAIN_STEPS=${TRAIN_STEPS:-800}
export LOG_INTERVAL=${LOG_INTERVAL:-1}
export SAVE_INTERVAL=${SAVE_INTERVAL:-200}
export SAVE_DIR=${SAVE_DIR:-"/results/checkpoints"}

# ---- Data / tokenizer --------------------------------------------------------
export TRAIN_DATA=${TRAIN_DATA:-"/data/train.jsonl"}
export VALID_DATA=${VALID_DATA:-"/data/validation.jsonl"}
export TOKENIZER=${TOKENIZER:-"meta-llama/Llama-2-70b-hf"}

# ---- LoRA --------------------------------------------------------------------
export LORA_RANK=${LORA_RANK:-0}
export LORA_ALPHA=${LORA_ALPHA:-32}
export LORA_DROPOUT=${LORA_DROPOUT:-0.1}

# ---- FP8 training ------------------------------------------------------------
export FP8_TRAINING=${FP8_TRAINING:-0}
export FP8_FORMAT=${FP8_FORMAT:-"fp8_e4m3"}
export FP8_SCALING=${FP8_SCALING:-"delayed"}
export FP8_BLOCK_SIZE=${FP8_BLOCK_SIZE:-128}
export FP8_AMAX_ALGO=${FP8_AMAX_ALGO:-"max"}
export FP8_REDUCE_AMAX=${FP8_REDUCE_AMAX:-0}
export FP8_AMAX_HISTORY=${FP8_AMAX_HISTORY:-16}
export FP8_ACTIVATION=${FP8_ACTIVATION:-1}

# ---- Warmup / early stopping -------------------------------------------------
export WARMUP_STEPS=${WARMUP_STEPS:-0}
export VAL_LOSS_TARGET=${VAL_LOSS_TARGET:-""}

# ---- Megatron backend --------------------------------------------------------
export MODEL_SIZE=${MODEL_SIZE:-"llama2-70B"}
export TP=${TP:-8}
export PP=${PP:-1}
export CP=${CP:-1}
export VP=${VP:-0}
export SP=${SP:-0}
export GBS=${GBS:-8}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
export GRADIENT_CLIP=${GRADIENT_CLIP:-0.3}
export EVAL_INTERVAL=${EVAL_INTERVAL:-50}
export PRECISION=${PRECISION:-"bf16"}
export CKPT_DIR=${CKPT_DIR:-"/ckpt"}
export LORA_A2A=${LORA_A2A:-0}

# Transformer Light attention
export TL_ATTN_BACKEND=${TL_ATTN_BACKEND:-"aiter"}
export TL_FP8_QUANT=${TL_FP8_QUANT:-"fp8_blockwise"}

# MXFP8 attention block sizes
export MXFP8_BLOCK_M_FWD=${MXFP8_BLOCK_M_FWD:-128}
export MXFP8_BLOCK_N_FWD=${MXFP8_BLOCK_N_FWD:-128}
export MXFP8_BLOCK_M_DQ_BWD=${MXFP8_BLOCK_M_DQ_BWD:-128}
export MXFP8_BLOCK_N_DQ_BWD=${MXFP8_BLOCK_N_DQ_BWD:-128}
export MXFP8_BLOCK_M_DKV_BWD=${MXFP8_BLOCK_M_DKV_BWD:-128}
export MXFP8_BLOCK_N_DKV_BWD=${MXFP8_BLOCK_N_DKV_BWD:-128}
export MXFP8_QUANT_BLOCK_SIZE=${MXFP8_QUANT_BLOCK_SIZE:-128}

# ---- FSDP backend ------------------------------------------------------------
export MODEL=${MODEL:-"meta-llama/Llama-2-7b-hf"}
export GRAD_ACCUM=${GRAD_ACCUM:-8}
export MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
export NUM_WORKERS=${NUM_WORKERS:-4}
export TRAIN_SAMPLES=${TRAIN_SAMPLES:-10000}
export VAL_SAMPLES=${VAL_SAMPLES:-500}
export SHARDING=${SHARDING:-"full_shard"}

# ---- MI355X platform ---------------------------------------------------------
export MLPERF_SUBMISSION_ORG=${MLPERF_SUBMISSION_ORG:-"AMD"}
export MLPERF_SUBMISSION_PLATFORM=${MLPERF_SUBMISSION_PLATFORM:-"MI355X"}
