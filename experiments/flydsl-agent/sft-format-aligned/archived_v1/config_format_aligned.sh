#!/bin/bash
# Format-Aligned SFT — plan.md §Stage 0 training config.
# Intentionally "light": LoRA r=32, 1 epoch, low lr to minimize perturbation.

NGPU=8
MODEL="/dev/shm/qwen2.5-coder-32b"

TRAIN_DATA="/data/data/format_aligned/train.jsonl"
VAL_DATA="/data/data/format_aligned/validation.jsonl"

SEQ_LEN=32768  # Qwen2.5-Coder-32B native max_position_embeddings; covers ~84% of cat1 kernels fully
MBS=1
GRAD_ACCUM=1
# MAX_STEPS computed dynamically by run script based on data size

LR="5e-6"
MIN_LR="0"
LR_WARMUP_STEPS=30  # ~10% of expected ~300 steps
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0

# Low-rank LoRA to minimize perturbation to v5e weights
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0.05

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
