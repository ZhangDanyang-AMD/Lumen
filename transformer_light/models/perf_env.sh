#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Model-agnostic performance tuning environment variables for AMD MI GPUs.
#
# Source this file in your training launch script:
#   source "$(dirname "$0")/../../transformer_light/models/perf_env.sh"
#
# All variables respect existing values — only set defaults when unset.

# ---- NCCL tuning -------------------------------------------------------------
export NCCL_MIN_P2P_NCHANNELS=${NCCL_MIN_P2P_NCHANNELS:-32}
export NCCL_MIN_CTAS=${NCCL_MIN_CTAS:-32}
export NCCL_NCHANNELS_PER_NET_PEER=${NCCL_NCHANNELS_PER_NET_PEER:-32}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}

# ---- hipBLASLt / BLAS --------------------------------------------------------
export USE_HIPBLASLT=${USE_HIPBLASLT:-1}
export TORCH_BLAS_PREFER_HIPBLASLT=${TORCH_BLAS_PREFER_HIPBLASLT:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# ---- TunableOp ---------------------------------------------------------------
export PYTORCH_TUNABLEOP_ENABLED=${PYTORCH_TUNABLEOP_ENABLED:-1}
export PYTORCH_TUNABLEOP_FILENAME=${PYTORCH_TUNABLEOP_FILENAME:-tunableop_results.csv}
