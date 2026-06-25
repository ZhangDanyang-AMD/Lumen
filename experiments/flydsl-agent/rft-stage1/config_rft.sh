#!/bin/bash
# RFT (Rejection Fine-Tuning) configuration.
# Lower LR than SFT (5e-6 vs 1e-5) to protect existing capabilities.
# 1 epoch only — RFT data is small but high-quality (sandbox-verified).

# ---- Hardware ----------------------------------------------------------------
export NGPU="${NGPU:-8}"

# ---- Model (v5e merged, not base) -------------------------------------------
export MODEL="${MODEL:-/sft-model}"

# ---- Data (RFT merged: SFT + verified candidates) ---------------------------
export TRAIN_DATA="${TRAIN_DATA:-/rft-data/rft_train.jsonl}"
export VAL_DATA="${VAL_DATA:-/data/data/sft/validation-00000-of-00001.jsonl}"

# ---- Training ----------------------------------------------------------------
# MAX_STEPS computed dynamically in run script based on dataset size
export SEQ_LEN="${SEQ_LEN:-16384}"
export MBS="${MBS:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export MAX_STEPS="${MAX_STEPS:-600}"
export LR="${LR:-5e-6}"
export MIN_LR="${MIN_LR:-0}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-30}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# ---- LoRA (same as SFT) -----------------------------------------------------
export LORA_RANK="${LORA_RANK:-64}"
export LORA_ALPHA="${LORA_ALPHA:-128}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# ---- Logging / checkpoints ---------------------------------------------------
export LOG_INTERVAL="${LOG_INTERVAL:-5}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
export SAVE_DIR="${SAVE_DIR:-/devshm/rft-checkpoints}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
export NUM_WORKERS="${NUM_WORKERS:-4}"

# ---- NCCL / ROCm performance ------------------------------------------------
export NCCL_MIN_P2P_NCHANNELS=32
export NCCL_MIN_CTAS=32
export NCCL_NCHANNELS_PER_NET_PEER=32
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export HSA_ENABLE_SDMA=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN

export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1
