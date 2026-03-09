#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# MI355X 1-node 8-GPU configuration for LLaMA2 SFT fine-tuning.
#
# Usage:
#   bash run_finetune.sh                                      # uses this config
#   CONFIG=config_MI300X_1x8x1.sh bash run_finetune.sh        # uses MI300X config
#   TRAIN_STEPS=500 bash run_finetune.sh                      # override by editing this file

# ---- Backend selection -------------------------------------------------------
export BACKEND="fsdp"

# ---- Hardware ----------------------------------------------------------------
export NGPU=8
export NNODES=1

# ---- Training hyperparameters ------------------------------------------------
export MBS=1
export SEQ_LEN=8192
export LR=4e-4
export MIN_LR=0
export TRAIN_STEPS=800
export LOG_INTERVAL=1
export SAVE_INTERVAL=200
export SAVE_DIR="/results/checkpoints"

# ---- Data / tokenizer --------------------------------------------------------
export TRAIN_DATA="/data/train.npy"
export VALID_DATA="/data/validation.npy"
_TL_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TOKENIZER="${_TL_CONFIG_DIR}/tokenizer"

# ---- LoRA --------------------------------------------------------------------
export LORA_RANK=0
export LORA_ALPHA=32
export LORA_DROPOUT=0.1

# ---- FP8 training ------------------------------------------------------------
export FP8_TRAINING=1
export FP8_FORMAT="fp8_e4m3"
export FP8_SCALING="delayed"
export FP8_BLOCK_SIZE=128
export FP8_AMAX_ALGO="max"
export FP8_REDUCE_AMAX=0
export FP8_AMAX_HISTORY=16
export FP8_ACTIVATION=1
export GRAD_QUANT_TYPE=""

# ---- Warmup / early stopping -------------------------------------------------
export WARMUP_STEPS=0
export VAL_LOSS_TARGET=""

# ---- Megatron backend --------------------------------------------------------
export MODEL_SIZE="llama2-70B"
export TP=8
export PP=1
export CP=1
export VP=0
export SP=1
export GBS=8
export WEIGHT_DECAY=1e-4
export GRADIENT_CLIP=0.3
export EVAL_INTERVAL=50
export PRECISION="bf16"
export CKPT_DIR="/model"
export LORA_A2A=0

# Transformer Light
# AITER's asm-v3 flash-attention kernel (fmha_v3_fwd) does not currently
# contain compiled variants for every GQA configuration on gfx950 (MI355X).
# For LLaMA2-70B with TP=8 the per-partition head ratio is 8Q:1KV, which
# triggers "invalid argument for fmha_fwd" from aiter::mha_fwd returning -1.
# Use the Triton FP8-blockwise backend; it supports all GQA ratios on gfx950.
# TL_ATTN_BACKEND options:
#   triton        – plain BF16/FP16 Triton attention (no attention quantization)
#   triton_fp8    – Triton FP8-quantized attention; quant type set by TL_FP8_QUANT:
#                     fp8_blockwise  – per-block FP8 scaling (works on all MI-series)
#                     mxfp8          – microscaling FP8 (gfx950 / MI355X only)
export TL_ATTN_BACKEND="triton_fp8"
# TRANSFORMER_LIGHT_ATTN_BACKEND controls attention_impl.py's module-load-time
# csrc probe (_probe_aiter_csrc). Keep as "triton" whenever TL_ATTN_BACKEND is
# any triton-family value to prevent aiter csrc kernels from being probed.
export TRANSFORMER_LIGHT_ATTN_BACKEND="triton"
export TL_FP8_QUANT="mxfp8"
export TL_RMSNORM=0

# MXFP8 attention block sizes
export MXFP8_BLOCK_M_FWD=128
export MXFP8_BLOCK_N_FWD=128
export MXFP8_BLOCK_M_DQ_BWD=128
export MXFP8_BLOCK_N_DQ_BWD=128
export MXFP8_BLOCK_M_DKV_BWD=128
export MXFP8_BLOCK_N_DKV_BWD=128
export MXFP8_QUANT_BLOCK_SIZE=128

# ---- FSDP backend ------------------------------------------------------------
export MODEL="/model-hf"
export GRAD_ACCUM=8
export MAX_GRAD_NORM=1.0
export NUM_WORKERS=4
export TRAIN_SAMPLES=10000
export VAL_SAMPLES=500
export SHARDING="full_shard"

# ---- MI355X platform ---------------------------------------------------------
export MLPERF_SUBMISSION_ORG="AMD"
export MLPERF_SUBMISSION_PLATFORM="MI355X"

# ---- NCCL / ROCm performance (MI355X tuned) ----------------------------------
export NCCL_MIN_P2P_NCHANNELS=32
export NCCL_MIN_CTAS=32
export NCCL_NCHANNELS_PER_NET_PEER=32
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
# Disable SDMA engines — known to cause RCCL P2P hangs on MI-series GPUs.
export HSA_ENABLE_SDMA=0
# Disable InfiniBand probing — avoids "No device found" errors.
export NCCL_IB_DISABLE=1
# Force RCCL bootstrap sockets onto loopback for single-node runs.
# Without this, RCCL scans all interfaces and may select a physical NIC
# (eno1, eno0, …) that is unreachable inside Kubernetes / container
# environments, causing "socketPollConnect: No route to host" at init time.
# Actual GPU data transfers use PCIe/xGMI and are unaffected by this setting.
# For multi-node runs override with the inter-node interface: NCCL_SOCKET_IFNAME=ens3
export NCCL_SOCKET_IFNAME=lo
# Set to INFO to debug RCCL errors; WARN for normal runs.
export NCCL_DEBUG=WARN

# ---- hipBLASLt ---------------------------------------------------------------
export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1

# ---- Misc ROCm perf ----------------------------------------------------------
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_FILENAME="tunableop_results.csv"
export OMP_NUM_THREADS=1

# ---- Torch Dynamo / Inductor ------------------------------------------------
# Megatron's _warmup_jit_function calls torch.compile on bias_swiglu, which
# triggers the Inductor backend.  Inductor imports triton_key from Triton at
# graph-cache time; if the installed Triton build doesn't export that symbol
# the warmup crashes.  Disabling Dynamo skips the torch.compile path entirely
# — Megatron's own fused kernels are used instead, with no perf impact.
export TORCHDYNAMO_DISABLE=1
