#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# VERL + Lumen GRPO launcher (Megatron actor + vLLM rollout, 8x MI300X)
#
# Usage:
#   bash examples/rl/verl/run_grpo_megatron_vllm.sh
#
# Requires: megatron-core installed. Uses Megatron tensor parallelism for the
# actor/reference and vLLM for rollout generation.

set -euo pipefail

export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export TORCHDYNAMO_DISABLE=1

export MODEL_NAME="${MODEL_NAME:-/dev/shm/model/llama-3.1-8b}"
TRAIN_DATA="${TRAIN_DATA:-/workspace/data/deepmath_500.parquet}"
VAL_DATA="${VAL_DATA:-/workspace/data/deepmath_val_20.parquet}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_STEPS="${MAX_STEPS:-10}"
TRAIN_BSZ="${TRAIN_BSZ:-64}"
ACTOR_TP="${ACTOR_TP:-4}"
ROLLOUT_TP="${ROLLOUT_TP:-4}"
ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL:-0.4}"
MICRO_BSZ="${MICRO_BSZ:-2}"
LOG_PROB_MICRO_BSZ="${LOG_PROB_MICRO_BSZ:-4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo-megatron-vllm}"
export LUMEN_FP8="${LUMEN_FP8:-0}"
export LUMEN_FP8_ATTN="${LUMEN_FP8_ATTN:-none}"
export LUMEN_NORM="${LUMEN_NORM:-0}"
export LUMEN_FP8_WEIGHT_CACHE="${LUMEN_FP8_WEIGHT_CACHE:-0}"
export LUMEN_FP8_ACTIVATION_STORE="${LUMEN_FP8_ACTIVATION_STORE:-0}"
export LUMEN_FP8_PARAM_GATHER="${LUMEN_FP8_PARAM_GATHER:-0}"
export FP8_PARAM_MANAGER="${FP8_PARAM_MANAGER:-0}"
export USE_8BIT_ADAM="${USE_8BIT_ADAM:-0}"

echo "=== VERL + Lumen GRPO — Megatron + vLLM rollout ==="
echo "Model:        ${MODEL_NAME}"
echo "Dataset:      ${TRAIN_DATA}"
echo "GPUs:         ${NUM_GPUS}"
echo "Steps:        ${MAX_STEPS}"
echo "Actor TP:     ${ACTOR_TP}"
echo "Rollout TP:   ${ROLLOUT_TP}"
echo "FP8:          ${LUMEN_FP8}"
echo "FP8 PM:       ${FP8_PARAM_MANAGER}"
echo ""

VERL_ENTRY="verl.trainer.main_ppo"
if [ "${LUMEN_FP8}" = "1" ] || [ "${LUMEN_FP8_ATTN}" != "none" ] || [ "${LUMEN_NORM}" = "1" ] \
   || [ "${LUMEN_FP8_WEIGHT_CACHE}" = "1" ] || [ "${LUMEN_FP8_ACTIVATION_STORE}" = "1" ] \
   || [ "${LUMEN_FP8_PARAM_GATHER}" = "1" ] || [ "${FP8_PARAM_MANAGER}" = "1" ]; then
    VERL_ENTRY="lumen.rl.verl.verl_entry"
fi

python3 -m "${VERL_ENTRY}" \
    --config-name=ppo_megatron_trainer \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="${MODEL_NAME}" \
    actor_rollout_ref.model.trust_remote_code=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BSZ}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BSZ}" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${ACTOR_TP}" \
    actor_rollout_ref.actor.megatron.sequence_parallel=false \
    +actor_rollout_ref.actor.megatron.override_transformer_config.transformer_impl=local \
    +actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=false \
    actor_rollout_ref.rollout.name=vllm \
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
