#!/usr/bin/env bash
# Experiment 1A-v5b: Qwen3-8B-Base — BF16 training + BF16 rollout
#
# Config aligned to VERL FP8 docs reference (Qwen3-8B-Base, 8×H100 GPU):
#   https://github.com/verl-project/verl/blob/main/docs/advance/fp8.md#qwen3-8b-base-dense-model
#   train_bsz=32, gen_bsz=96 (32×3), mini_bsz=32, n=16, max_resp=20K
#   rollout_tp=1, sp_size=1
#
# Fallback to vLLM 0.9.2 (V4 container) after vLLM 0.16 CuMemAllocator bug:
#   hipMemMap produces read-only pages on ROCm, causing "Memory access fault"
#   on second generation step after weight update.
#
# ROCm adaptations:
#   gpu_memory_utilization: tunable (start at 0.3, sweep up for speed)
#   free_cache_engine=True (vLLM sleep/wake is ROCm-patched in V4 container)
#   max_num_seqs=64 (prevents KV cache OOM with long sequences on ROCm)
#
# Container: lumen_verl_test (rocm/sgl-dev, vLLM 0.9.2rc2, ROCm 7.0)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROJECT_NAME="FP8-ALIGN"
export EXP_NAME="1A-v5b-qwen3-8b-bf16-ref-aligned"
export MODEL_PATH="${MODEL_PATH:-/dev/shm/model/qwen3-8b-base}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.3}"
export GEN_BSZ="96"
export TRAIN_BSZ="32"
export ROLLOUT_TP="1"
export SP_SIZE="1"
export MAX_NUM_SEQS="64"
export OFFLOAD="true"
export ROLLOUT_QUANTIZATION="null"
export ROLLOUT_IS="null"
export FREE_CACHE_ENGINE="true"
export PPO_MAX_TOKEN_LEN="21504"
export LOG_PROB_MAX_TOKEN_LEN="21504"

# Safety: increase bucket for unsharded embed_tokens with TP=1 (may be needed if weights are fp32)
export EXTRA_OVERRIDES="actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2560"

source "${SCRIPT_DIR}/common.sh"
launch_training
