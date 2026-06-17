#!/bin/bash
# Qwen2.5-Coder-32B CPT configuration for 8x MI355X (gfx950).
#
# Hyperparameters from plan.md §7.2:
#   - LoRA r=64, alpha=128 (CPT needs more capacity to learn new domain)
#   - LR=2e-5 (higher than SFT, learning entirely new domain)
#   - 3 epochs over ~11M tokens -> ~125 steps
#   - BF16 training (no FP8 — simpler and CPT is short)

# ---- Hardware ----------------------------------------------------------------
export NGPU="${NGPU:-8}"

# ---- Model -------------------------------------------------------------------
export MODEL="${MODEL:-/dev/shm/qwen2.5-coder-32b}"
export TOKENIZER="${TOKENIZER:-}"   # defaults to MODEL path in train_cpt.py

# ---- Data --------------------------------------------------------------------
export TRAIN_DATA="${TRAIN_DATA:-/data/cpt/train-00000-of-00001.jsonl}"

# ---- Training hyperparameters (plan.md §7.2) ---------------------------------
export SEQ_LEN="${SEQ_LEN:-8192}"
export MBS="${MBS:-2}"                    # micro batch per GPU
export GRAD_ACCUM="${GRAD_ACCUM:-2}"      # GBS = MBS(2) x GPU(8) x accum(2) = 32
export MAX_STEPS="${MAX_STEPS:-232}"       # 10 epochs: 743 chunks / GBS=32 ≈ 23 steps/epoch × 10
export EPOCHS="${EPOCHS:-10}"
export LR="${LR:-2e-5}"
export MIN_LR="${MIN_LR:-0}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-11}"  # ~5% of 232 steps
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# ---- LoRA (plan.md §3: CPT uses r=64 for more capacity) ---------------------
export LORA_RANK="${LORA_RANK:-64}"
export LORA_ALPHA="${LORA_ALPHA:-128}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.0}"   # no dropout for CPT

# ---- Logging / checkpoints ---------------------------------------------------
export LOG_INTERVAL="${LOG_INTERVAL:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"    # save every 25 steps to /dev/shm (fast)
export SAVE_DIR="${SAVE_DIR:-/devshm/cpt-checkpoints}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export TRAIN_SAMPLES="${TRAIN_SAMPLES:-0}"     # 0 = use all

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
