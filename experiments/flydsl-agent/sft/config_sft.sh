#!/bin/bash
# Qwen2.5-Coder-32B SFT configuration for 8x MI355X.
#
# Hyperparameters from plan.md §7.3:
#   - LoRA r=32, alpha=64 (SFT task more focused than CPT)
#   - LR=1e-5 (lower than CPT to protect base knowledge)
#   - 3 epochs, dropout=0.1 for regularization
#   - Skip CPT, train directly on base model

# ---- Hardware ----------------------------------------------------------------
export NGPU="${NGPU:-8}"

# ---- Model (directly from base, no CPT) -------------------------------------
export MODEL="${MODEL:-/dev/shm/qwen2.5-coder-32b}"

# ---- Data (SFT split: 2808 train + 264 val) ---------------------------------
export TRAIN_DATA="${TRAIN_DATA:-/data/data/sft/train-00000-of-00001.jsonl}"
export VAL_DATA="${VAL_DATA:-/data/data/sft/validation-00000-of-00001.jsonl}"

# ---- Training ----------------------------------------------------------------
# 2916 samples / GBS=8 ≈ 365 steps/epoch × 3 epochs = 1094 steps
# seq_len=16384 to avoid 37.6% truncation at 8192 (kernel code is long)
# MBS=1 + grad_accum=1 → GBS=8 (smaller batch for longer sequences)
export SEQ_LEN="${SEQ_LEN:-16384}"
export MBS="${MBS:-1}"                    # micro batch per GPU
export GRAD_ACCUM="${GRAD_ACCUM:-1}"      # GBS = 1 × 8 × 1 = 8
export MAX_STEPS="${MAX_STEPS:-1459}"       # 3889 / GBS=8 × 3 epochs (v5e: boosted correct kernels)
export LR="${LR:-1e-5}"
export MIN_LR="${MIN_LR:-0}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-73}"  # ~5% of 1459
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# ---- LoRA (r=64 for more capacity, validated in CPT sweep) -------------------
export LORA_RANK="${LORA_RANK:-64}"
export LORA_ALPHA="${LORA_ALPHA:-128}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.1}"   # regularization for small dataset

# ---- Logging / checkpoints ---------------------------------------------------
export LOG_INTERVAL="${LOG_INTERVAL:-5}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-0}"     # only save final
export SAVE_DIR="${SAVE_DIR:-/devshm/sft-checkpoints}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-50}"    # validate every 50 steps
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
