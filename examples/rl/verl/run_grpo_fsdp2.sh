#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# VERL + Lumen GRPO launcher (FSDP2 actor + sglang rollout, 8x MI300X)
#
# Usage:
#   bash examples/rl/verl/run_grpo_fsdp2.sh
#
# Environment variables:
#   MODEL_NAME     - Model path (default: /dev/shm/model/llama-3.1-8b)
#   TRAIN_DATA     - Training parquet (default: /workspace/data/deepmath_500.parquet)
#   VAL_DATA       - Validation parquet (default: /workspace/data/deepmath_val_20.parquet)
#   NUM_GPUS       - Number of GPUs (default: 8)
#   MAX_STEPS      - Training steps (default: 10)
#   TRAIN_BSZ      - Training batch size (default: 64)
#   ROLLOUT_TP     - Rollout tensor parallelism (default: 1)
#   ROLLOUT_GPU_UTIL - Rollout GPU memory utilization for KV cache (default: 0.4)
#   MICRO_BSZ      - Per-GPU micro batch size (default: 2)
#   LOG_PROB_MICRO_BSZ - Log prob micro batch size per GPU (default: 4)
#   NUM_GENERATIONS - Number of rollout generations (default: 4)
#   EXPERIMENT_NAME - Experiment name (default: auto from model)
#   LUMEN_FP8      - Enable FP8 linear (default: 0)
#   LUMEN_FP8_ATTN - FP8 attention mode (default: none)
#   LUMEN_NORM     - Enable Lumen norm replacement (default: 0)

set -euo pipefail

# Prevent Ray from stripping GPU visibility for CPU-only workers.
# sglang queries device properties at import time and needs GPU access.
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Disable torch.compile/dynamo — inductor has ROCm Triton incompatibilities
# (missing triton_key, RLock pickling errors in async kernel compilation).
export TORCHDYNAMO_DISABLE=1

export MODEL_NAME="${MODEL_NAME:-/dev/shm/model/llama-3.1-8b}"
TRAIN_DATA="${TRAIN_DATA:-/workspace/data/deepmath_500.parquet}"
VAL_DATA="${VAL_DATA:-/workspace/data/deepmath_val_20.parquet}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_STEPS="${MAX_STEPS:-10}"
TRAIN_BSZ="${TRAIN_BSZ:-64}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL:-0.4}"
MICRO_BSZ="${MICRO_BSZ:-2}"
LOG_PROB_MICRO_BSZ="${LOG_PROB_MICRO_BSZ:-4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo-fsdp2}"
export LUMEN_FP8="${LUMEN_FP8:-0}"
export LUMEN_FP8_ATTN="${LUMEN_FP8_ATTN:-none}"
export LUMEN_NORM="${LUMEN_NORM:-0}"
export LUMEN_FP8_WEIGHT_CACHE="${LUMEN_FP8_WEIGHT_CACHE:-0}"
export LUMEN_FP8_ACTIVATION_STORE="${LUMEN_FP8_ACTIVATION_STORE:-0}"
export LUMEN_FP8_PARAM_GATHER="${LUMEN_FP8_PARAM_GATHER:-0}"

echo "=== VERL + Lumen GRPO — FSDP2 + sglang rollout ==="
echo "Model:        ${MODEL_NAME}"
echo "Dataset:      ${TRAIN_DATA}"
echo "GPUs:         ${NUM_GPUS}"
echo "Steps:        ${MAX_STEPS}"
echo "Batch:        ${TRAIN_BSZ}"
echo "Rollout TP:   ${ROLLOUT_TP}"
echo "Rollout Util: ${ROLLOUT_GPU_UTIL}"
echo "Micro BSZ:    ${MICRO_BSZ}"
echo "LogProb uBSZ: ${LOG_PROB_MICRO_BSZ}"
echo "Generations:  ${NUM_GENERATIONS}"
echo "Experiment:   ${EXPERIMENT_NAME}"
echo "FP8:          ${LUMEN_FP8}"
echo "FP8 Attn:     ${LUMEN_FP8_ATTN}"
echo "Lumen Norm:   ${LUMEN_NORM}"
echo "FP8 Wt Cache: ${LUMEN_FP8_WEIGHT_CACHE}"
echo "FP8 Act Store:${LUMEN_FP8_ACTIVATION_STORE}"
echo "FP8 Param Gth:${LUMEN_FP8_PARAM_GATHER}"
echo ""

VERL_ENTRY="verl.trainer.main_ppo"
if [ "${LUMEN_FP8}" = "1" ] || [ "${LUMEN_FP8_ATTN}" != "none" ] || [ "${LUMEN_NORM}" = "1" ] \
   || [ "${LUMEN_FP8_WEIGHT_CACHE}" = "1" ] || [ "${LUMEN_FP8_ACTIVATION_STORE}" = "1" ] \
   || [ "${LUMEN_FP8_PARAM_GATHER}" = "1" ]; then
    VERL_ENTRY="lumen.rl.verl.verl_entry"
fi

python3 -m "${VERL_ENTRY}" \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="${MODEL_NAME}" \
    actor_rollout_ref.model.trust_remote_code=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BSZ}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BSZ}" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.prompt_length=512 \
    actor_rollout_ref.rollout.response_length=256 \
    actor_rollout_ref.rollout.n="${NUM_GENERATIONS}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_UTIL}" \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.load_format=dummy \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BSZ}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BSZ}" \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size="${TRAIN_BSZ}" \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    algorithm.use_kl_in_reward=false \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.total_training_steps="${MAX_STEPS}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.project_name=lumen-verl-grpo \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=false \
    trainer.logger='[console]'
