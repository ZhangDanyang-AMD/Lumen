#!/usr/bin/env bash
# Experiment 1A-v5e: Qwen3-8B-Base — BF16 training + BF16 rollout + Dynamic Sampling
#
# Based on V5d (gpu_mem=0.6, max_num_seqs=128) + dynamic sampling from V4.
# V5d crashed at step 27 during compute_log_prob (GPU OOM, seqlen_max=593K).
# Dynamic sampling filters uninformative prompt groups, reducing effective
# sequence load during training and mitigating the OOM risk.
#
# V4 (gpu_mem=0.85, max_num_seqs=256, dynamic sampling) ran 34+ steps stably.
# V5e uses more conservative gpu_mem=0.6 with dynamic sampling for extra safety.
#
# Config aligned to VERL FP8 docs reference (Qwen3-8B-Base, 8×H100 GPU):
#   train_bsz=32, gen_bsz=96 (32×3), mini_bsz=32, n=16, max_resp=20K
#   rollout_tp=1, sp_size=1
#
# Container: lumen_verl_test (rocm/sgl-dev, vLLM 0.9.2rc2, ROCm 7.0)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROJECT_NAME="FP8-ALIGN"
export EXP_NAME="1A-v5e-qwen3-8b-bf16-gpumem06-dynsamp"
export MODEL_PATH="${MODEL_PATH:-/dev/shm/model/qwen3-8b-base}"
export GPU_MEM_UTIL="0.6"
export GEN_BSZ="96"
export TRAIN_BSZ="32"
export ROLLOUT_TP="1"
export SP_SIZE="1"
export MAX_NUM_SEQS="128"
export OFFLOAD="true"
export ROLLOUT_QUANTIZATION="null"
export ROLLOUT_IS="null"
export FREE_CACHE_ENGINE="true"
export PPO_MAX_TOKEN_LEN="21504"
export LOG_PROB_MAX_TOKEN_LEN="21504"

export EXTRA_OVERRIDES="actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2560"

# Dynamic sampling (filter uninformative prompt groups to reduce effective batch load)
export VERL_FILTER_GROUPS_ENABLE=1
export VERL_FILTER_GROUPS_METRIC=acc
export VERL_FILTER_GROUPS_MAX_GEN=10

source "${SCRIPT_DIR}/common.sh"
launch_training
