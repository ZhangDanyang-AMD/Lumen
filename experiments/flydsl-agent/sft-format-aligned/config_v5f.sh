#!/bin/bash
# v5f SFT config — same hyperparams as v5e but with format-aligned data.
# This is a full retrain from base, NOT a patch on v5e.

NGPU=8
MODEL="/dev/shm/qwen2.5-coder-32b"

TRAIN_DATA="/data/data/format_aligned/train.jsonl"
VAL_DATA="/data/data/format_aligned/validation.jsonl"

SEQ_LEN=32768  # Qwen2.5-Coder-32B native max_position_embeddings; data built with --max-kernel-chars 120000 to match
MBS=1
GRAD_ACCUM=1

# Same as v5e
LR="1e-5"
MIN_LR="0"
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0

# Same as v5e
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.1

LOG_INTERVAL=5
SAVE_INTERVAL=0  # final only
EVAL_INTERVAL=50
NUM_WORKERS=4

# ROCm / NCCL
NCCL_MIN_P2P_NCHANNELS=32
NCCL_MIN_CTAS=32
NCCL_NCHANNELS_PER_NET_PEER=2
NCCL_NVLS_ENABLE=0
TORCH_NCCL_AVOID_RECORD_STREAMS=1
HSA_ENABLE_SDMA=0
NCCL_IB_DISABLE=0
NCCL_SOCKET_IFNAME=ens12f0np0
NCCL_DEBUG=WARN
USE_HIPBLASLT=1
TORCH_BLAS_PREFER_HIPBLASLT=1
CUDA_DEVICE_MAX_CONNECTIONS=1
OMP_NUM_THREADS=8
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
TORCHDYNAMO_DISABLE=1
