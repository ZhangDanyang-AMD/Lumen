#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# MI450X single-node, single-GPU configuration for LLaMA 3.1 pretraining.
#
# Usage:
#   CONFIG=config_MI450X_1x1x1.sh bash run_pretrain.sh
#   NGPU=2 CONFIG=config_MI450X_1x1x1.sh bash run_pretrain.sh

_MI450_NGPU="${NGPU:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config_MI355X_1x8x1.sh"

# ---- Hardware ----------------------------------------------------------------
export NGPU="${_MI450_NGPU}"
export NNODES=1
export GPUS_PER_NODE=${NGPU}
unset _MI450_NGPU

# ---- gfx1250 / AITER ----------------------------------------------------------
# The MI450 image and launcher mount the local AITER checkout at /opt/aiter.
# AITER dispatches these operators to its gfx1250 Gluon/WMMA/TDM kernels.
export GPU_ARCHS="gfx1250"
export ENABLE_CK=1
export PYTHONPATH="/opt/aiter:${PYTHONPATH:-}"
export AITER_LOG_LEVEL="${AITER_LOG_LEVEL:-WARNING}"

# Lumen parallel linears are required for --linear-fp8 to reach AITER GEMMs.
# Blockwise FP8 selects AITER's gfx1250 gemm_a8w8_blockscale implementation;
# BF16 linears use the LLaMA-shape-tuned gfx1250 FlyDSL A16W16 kernel.
export LUMEN_LINEAR=1
export LUMEN_RMSNORM=1
export LUMEN_NORM=1
export LUMEN_GEMM_BF16=1
export LUMEN_GEMM_BF16_BACKEND="flydsl"
export AITER_CONFIG_GEMM_BF16="${SCRIPT_DIR}/configs/llama31_8b_bf16_gfx1250_flydsl_tuned.csv"
export LUMEN_FUSED_SWIGLU=1
export LUMEN_FUSED_RESIDUAL_NORM=1
export LUMEN_CROSS_ENTROPY=1
export LUMEN_FUSED_ROPE=1
export FP8_SCALING="blockwise"
export FP8_BLOCK_SIZE=128

# AITER csrc dispatch selects the gfx1250 FMHA v3 assembly code objects for
# both BF16 forward and backward (head dim 128, causal, no dropout).
export LUMEN_ATTN_BACKEND="aiter_csrc"
export LUMEN_FP8_QUANT="blockwise"
export LUMEN_FP8_BLOCKSCALE_GFX1250_TUNED=1

# AITER owns GEMM selection on gfx1250; a full PyTorch TunableOp sweep is both
# redundant and very expensive during a short smoke run.
export PYTORCH_TUNABLEOP_ENABLED=0
export CUDA_DEVICE_MAX_CONNECTIONS=8

# Single-GPU runs do not need RCCL P2P tuning.
export NCCL_MIN_P2P_NCHANNELS=1
export NCCL_MIN_CTAS=1
export NCCL_NCHANNELS_PER_NET_PEER=1
export HSA_ENABLE_SDMA=1

# ---- MLPerf / experiment metadata --------------------------------------------
export MLPERF_SUBMISSION_ORG="AMD"
export MLPERF_SUBMISSION_PLATFORM="MI450X"
