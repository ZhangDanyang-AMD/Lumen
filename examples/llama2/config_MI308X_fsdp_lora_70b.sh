#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# MI308X 1-node 8-GPU configuration for LLaMA2-70B LoRA SFT via PyTorch FSDP.
#
# BF16 LoRA fine-tuning aligned with the MLPerf v5.1 Llama2-70B reference
# hyperparameters (see examples/llama2/README.md), but using the FSDP backend
# instead of Megatron and BF16 instead of FP8 (FSDP+FP8+LoRA is less validated).
#
# Usage (inside container, via run_fsdp_lora_mi308.sh):
#   CONFIG=config_MI308X_fsdp_lora_70b.sh BACKEND=fsdp bash run_finetune.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Inherit all FSDP launcher defaults (NCCL/ROCm perf env, arg plumbing).
source "${SCRIPT_DIR}/config_MI355X_1x8x1.sh"

# ---- Backend / hardware ------------------------------------------------------
export BACKEND="fsdp"
export NGPU=8
export NNODES=1

# ---- Model (HuggingFace Llama-2-70B, mounted at /model-hf) -------------------
export MODEL="/model-hf"
export TOKENIZER="${SCRIPT_DIR}/tokenizer"

# ---- Data (preprocessed .npy, mounted at /data) -----------------------------
export TRAIN_DATA="/data/train.npy"
export VALID_DATA="/data/validation.npy"
export TRAIN_SAMPLES=10000
export VAL_SAMPLES=500

# ---- Training hyperparameters (MLPerf-aligned, see README) ------------------
export SEQ_LEN=8192
export MBS=1
export GRAD_ACCUM=1          # GBS = MBS(1) x DP(8) x accum(1) = 8  (matches README GBS=8)
export TRAIN_STEPS=1024      # README: cosine decay over full 1024 steps
export LR=4e-4
export MIN_LR=0
export LR_WARMUP_STEPS=0     # README: 0 warmup steps
export WEIGHT_DECAY=1e-4
export MAX_GRAD_NORM=0.3     # README: gradient clip 0.3
export SEED=1234

# ---- LoRA (rank 16, alpha 32, dropout 0.1) ----------------------------------
export LORA_RANK=16
export LORA_ALPHA=32
export LORA_DROPOUT=0.1

# ---- Precision: BF16 (FP8 disabled) -----------------------------------------
export FP8_TRAINING=0
export LUMEN_NORM=0          # keep plain HF RMSNorm for stability

# ---- Warmup / early stopping -------------------------------------------------
export WARMUP_STEPS=0        # no synthetic FP8 calibration needed in BF16
export VAL_LOSS_TARGET=""    # run the full 1024 steps (no early stop)

# ---- Logging / checkpoint / eval --------------------------------------------
export LOG_INTERVAL=1
export SAVE_DIR="/results/checkpoints"
export SAVE_INTERVAL=256     # LoRA adapters are small
export EVAL_EVERY=384        # eval cadence in sequences -> ~48 steps at GBS=8
export EVAL_INTERVAL=0       # let run_finetune.sh derive from EVAL_EVERY
export START_EVAL_AT=0
export NUM_WORKERS=4
export SHARDING="full_shard"

# ---- Misc --------------------------------------------------------------------
export USE_SDMA=0
export USE_CKPT=0
export SAVE_CKPT=0

# Disable online GEMM autotuning: at 70B/8192 the first-step tuning sweep takes
# 20+ min and only buys per-step speed, which is irrelevant for a convergence
# reproduction. Default hipBLASLt heuristics keep step 1 fast.
export PYTORCH_TUNABLEOP_ENABLED=0

# Diagnostics for the 80-layer hang: per-rank RCCL INFO to separate files so the
# main training log stays readable; if a collective stalls, the tail of these
# files points at the stuck ring/op.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE="/results/nccl_rank.%h.%p.log"
