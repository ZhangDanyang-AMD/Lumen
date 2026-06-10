#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# MI308X 1-node 8-GPU configuration for LLaMA2-7B LoRA SFT via PyTorch FSDP.
#
# Smaller sibling of config_MI308X_fsdp_lora_70b.sh. Intended as an
# end-to-end test of the FSDP + LoRA + dataset + optimizer + backward path
# on a real Llama2-7B checkpoint, using the same MLPerf-style hyperparameters.
#
# Precision/scaling is selected at launch via the MODE env var:
#   MODE=bf16           (default) BF16, FP8 disabled
#   MODE=fp8_delayed    FP8 e4m3, delayed scaling, full FP8 backward
#   MODE=fp8_blockwise  FP8 e4m3, blockwise scaling, BF16 wgrad (see CLAUDE.md)
#
# Architecture (hidden/layers/heads/...) is read from the HF checkpoint's
# config.json at load time, so no --num-layers/--hidden-size args are needed.
#
# Usage (the launcher forwards MODE into the container):
#   MODE=fp8_delayed CONFIG=config_MI308X_fsdp_lora_7b.sh \
#       bash run_fsdp_lora_mi308.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Inherit all FSDP launcher defaults (NCCL/ROCm perf env, arg plumbing).
source "${SCRIPT_DIR}/config_MI355X_1x8x1.sh"

# ---- Backend / hardware ------------------------------------------------------
export BACKEND="fsdp"
export NGPU=8
export NNODES=1

# ---- Model (HuggingFace Llama-2-7B, mounted at /model-hf) --------------------
export MODEL="/model-hf"
export TOKENIZER="${SCRIPT_DIR}/tokenizer"

# ---- Data (preprocessed .npy, mounted at /data) -----------------------------
export TRAIN_DATA="/data/train.npy"
export VALID_DATA="/data/validation.npy"
export TRAIN_SAMPLES=10000
export VAL_SAMPLES=500

# ---- Training hyperparameters (MLPerf-aligned, mode-agnostic) ---------------
export SEQ_LEN=8192
export MBS=1
export GRAD_ACCUM=1          # GBS = MBS(1) x DP(8) x accum(1) = 8
export TRAIN_STEPS=200       # full bring-up run
export LR=4e-4
export MIN_LR=0
export LR_WARMUP_STEPS=0
export WEIGHT_DECAY=1e-4
export MAX_GRAD_NORM=0.3
export SEED=1234

# ---- LoRA (rank 16, alpha 32, dropout 0.1) ----------------------------------
export LORA_RANK=16
export LORA_ALPHA=32
export LORA_DROPOUT=0.1

# ---- Precision / scaling mode (switchable via MODE) -------------------------
export MODE="${MODE:-bf16}"
export LUMEN_NORM=0          # keep plain HF RMSNorm for stability

# Shared FP8 knobs (consumed only when FP8_TRAINING=1; overridden per-mode).
export FP8_FORMAT="fp8_e4m3"
export FP8_BLOCK_SIZE=128
export FP8_AMAX_ALGO="max"
export FP8_AMAX_HISTORY=16
export FP8_REDUCE_AMAX=0
export FP8_ACTIVATION=1
export GRAD_QUANT_TYPE=""
export FIRST_LAST_BF16=0

case "${MODE}" in
  bf16)
    export FP8_TRAINING=0
    ;;
  fp8_delayed)
    export FP8_TRAINING=1
    export FP8_SCALING="delayed"
    export FP8_WGRAD=1          # delayed scales survive weight.t() -> full FP8 bwd
    ;;
  fp8_blockwise)
    export FP8_TRAINING=1
    export FP8_SCALING="blockwise"
    # Blockwise per-block scales become misaligned after weight.t(), so full
    # FP8 backward is unsupported (CLAUDE.md). Compute weight gradient in BF16.
    export FP8_WGRAD=0
    ;;
  *)
    echo "ERROR: unknown MODE='${MODE}' (use: bf16 | fp8_delayed | fp8_blockwise)" >&2
    exit 1
    ;;
esac
echo "[config_MI308X_fsdp_lora_7b] MODE=${MODE} FP8_TRAINING=${FP8_TRAINING} scaling=${FP8_SCALING:-n/a} wgrad=${FP8_WGRAD:-n/a}"

# ---- Warmup / early stopping -------------------------------------------------
export WARMUP_STEPS=0        # match across modes; amax calibrates over real steps
export VAL_LOSS_TARGET=""    # no val-loss early-stop target
# Inherited MI355 defaults would otherwise abort the run: TARGET_LOG_PPL=3.3
# early-stops once val_loss < 3.3 (happened at step 10), and STEP_TIME_ATOL=18000
# asserts on slow steps. Disable both so the full TRAIN_STEPS run completes.
export TARGET_LOG_PPL=0.0    # unreachable target -> never early-stop
export STEP_TIME_ATOL=0      # no step-time assertion

# ---- Logging / checkpoint / eval --------------------------------------------
export LOG_INTERVAL=1
export SAVE_DIR="/results/checkpoints"
export SAVE_INTERVAL=0       # no checkpoint saves during the test
export EVAL_EVERY=80         # eval cadence in sequences -> 10 steps at GBS=8
export EVAL_INTERVAL=0       # let run_finetune.sh derive from EVAL_EVERY
export START_EVAL_AT=0
export NUM_WORKERS=4
export SHARDING="full_shard"

# ---- Misc --------------------------------------------------------------------
export USE_SDMA=0
export USE_CKPT=0
export SAVE_CKPT=0

# Disable online GEMM autotuning: the first-step tuning sweep adds minutes of
# startup that is wasted for a short pipeline test. Default hipBLASLt heuristics
# keep step 1 fast. (perf_env.sh honors this pre-set value.)
export PYTORCH_TUNABLEOP_ENABLED=0
