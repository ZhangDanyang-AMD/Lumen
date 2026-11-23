#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Qwen3-30B-A3B continued pretraining on 4 x MI350X/MI355X (gfx950), Megatron
# backend, EP4/TP1/PP1/DP4. Every value can be overridden from the environment.
#
# Recipes:
#   MXFP4 (default)  EXPERT_RECIPE=a4w4 LUMEN_ATTN_LINEAR_RECIPE=a4w4 LUMEN_ROPE_IMPL=apex
#   BF16 reference   EXPERT_RECIPE=bf16 LUMEN_ATTN_LINEAR_RECIPE=bf16 LUMEN_ROPE_IMPL=aiter
#
# Usage:
#   bash run_pretrain_qwen3_30b_a3b_megatron.sh
#   EXPERT_RECIPE=bf16 LUMEN_ATTN_LINEAR_RECIPE=bf16 LUMEN_ROPE_IMPL=aiter \
#       bash run_pretrain_qwen3_30b_a3b_megatron.sh

# ---- Hardware / parallelism --------------------------------------------------
export NGPU=${NGPU:-4}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29571}
export TP=${TP:-1}
export PP=${PP:-1}
export CP=${CP:-1}
export EP=${EP:-4}
export ETP=${ETP:-1}

# ---- Batch / schedule --------------------------------------------------------
export SEQ_LEN=${SEQ_LEN:-4096}
export MBS=${MBS:-1}
export GBS=${GBS:-8}
export TRAIN_ITERS=${TRAIN_ITERS:-3000}
# Stop after this many iterations while the data schedule still covers
# TRAIN_ITERS (0 = run to TRAIN_ITERS). The training samples built, and so the
# batch each step sees, depend on TRAIN_ITERS only.
export EXIT_INTERVAL=${EXIT_INTERVAL:-0}
export LR=${LR:-1e-5}
export MIN_LR=${MIN_LR:-1e-5}
export LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-0}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
export CLIP_GRAD=${CLIP_GRAD:-1.0}
export ADAM_BETA1=${ADAM_BETA1:-0.9}
export ADAM_BETA2=${ADAM_BETA2:-0.95}
export ADAM_EPS=${ADAM_EPS:-1e-8}
export AUX_LOSS_COEFF=${AUX_LOSS_COEFF:-1e-3}
export SEED=${SEED:-1234}

# ---- Evaluation / checkpointing ----------------------------------------------
# Megatron evaluates once after the last iteration; EVAL_INTERVAL adds periodic
# evaluations. SAVE_INTERVAL=0 disables checkpointing.
export EVAL_INTERVAL=${EVAL_INTERVAL:-1000000}
export EVAL_ITERS=${EVAL_ITERS:-8}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
export LOG_INTERVAL=${LOG_INTERVAL:-1}

# ---- Data / tokenizer / initial checkpoint -----------------------------------
# JSONL files with one {"text": ...} document per line (lumen.models.llama31.dataset).
export TRAIN_DATA=${TRAIN_DATA:-/data/train.jsonl}
export VALID_DATA=${VALID_DATA:-/data/validation.jsonl}
export TOKENIZER=${TOKENIZER:-/model}
# Megatron torch-format checkpoint converted from the HF weights for EP4.
export LOAD_DIR=${LOAD_DIR:-/checkpoint}
export SAVE_DIR=${SAVE_DIR:-/results/checkpoints}
export RESULTS_DIR=${RESULTS_DIR:-/results}

# ---- W&B (optional) ----------------------------------------------------------
# Logging goes through Megatron's TensorBoard writer, so `tensorboard` must be
# importable. The API key is read from ~/.netrc or WANDB_API_KEY.
export WANDB_PROJECT=${WANDB_PROJECT:-}
export WANDB_ENTITY=${WANDB_ENTITY:-}
export WANDB_EXP_NAME=${WANDB_EXP_NAME:-qwen3-30b-a3b-ep4}

# ---- Recipe ------------------------------------------------------------------
# SonicMoE expert GEMMs: bf16 | a4w4 (MXFP4 activations and weights) | w4a16 | a8w4 | a8w4f
export EXPERT_RECIPE=${EXPERT_RECIPE:-a4w4}
# Attention QKV / output projections: bf16 | a4w4 (BF16 Wgrad)
export LUMEN_ATTN_LINEAR_RECIPE=${LUMEN_ATTN_LINEAR_RECIPE:-a4w4}
# Fused RoPE kernel: apex (native apex kernel) | aiter (AITER rope_cached)
export LUMEN_ROPE_IMPL=${LUMEN_ROPE_IMPL:-apex}
# Attention kernel. The csrc backend calls aiter.ops.mha.flash_attn_func.
# LUMEN_ATTN_AITER_TRITON_MHA=1 serves it from AITER's Triton MHA, which needs no
# CK build (set 0 when AITER was built with its composable_kernel submodule);
# LUMEN_ATTN_AITER_FUSED_BWD=1 selects AITER's fused dq/dk/dv backward for it.
export LUMEN_ATTN_BACKEND=${LUMEN_ATTN_BACKEND:-csrc}
export LUMEN_ATTN_AITER_TRITON_MHA=${LUMEN_ATTN_AITER_TRITON_MHA:-1}
export LUMEN_ATTN_AITER_FUSED_BWD=${LUMEN_ATTN_AITER_FUSED_BWD:-1}

# ---- SonicMoE / MXFP4 runtime ------------------------------------------------
# Expert Wgrad stays BF16 and accumulates straight into main_grad.
export SONIC_MOE_MXFP4_WGRAD=${SONIC_MOE_MXFP4_WGRAD:-0}
export SONIC_MOE_BF16_WGRAD_MAIN_GRAD=${SONIC_MOE_BF16_WGRAD_MAIN_GRAD:-1}
# Packed MXFP4 expert weights are cached per optimizer step (generation-stamped).
export SONIC_MOE_PACK_CACHE=${SONIC_MOE_PACK_CACHE:-1}
export R80_CACHE_GENERATION=${R80_CACHE_GENERATION:-1}
export R80_DUAL_WEIGHT_PACK=${R80_DUAL_WEIGHT_PACK:-1}
export AITER_GROUPED_MXFP4_DUAL_V2=${AITER_GROUPED_MXFP4_DUAL_V2:-1}
# The MORI-count-sync weight pre-pack is on by default; LUMEN_MORI_PRESYNC_PACK=0 disables it.
export LUMEN_MORI_PRESYNC_PACK=${LUMEN_MORI_PRESYNC_PACK:-1}

# ---- ROCm runtime ------------------------------------------------------------
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-32}
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-4}
export GPU_ARCHS=${GPU_ARCHS:-gfx950}
export CU_NUM=${CU_NUM:-256}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
