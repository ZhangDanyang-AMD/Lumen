#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Shell launcher for TRL + Lumen GRPO runs with FSDP1 or FSDP2.
#
# Usage:
#   bash examples/rl/trl/run_grpo_fsdp.sh 1   # FSDP1
#   bash examples/rl/trl/run_grpo_fsdp.sh 2   # FSDP2
#
# All tunables are environment variables with sensible defaults for smoke
# runs on a tiny model.  Override any of them on the command line:
#
#   MODEL_NAME=meta-llama/Llama-2-70b-hf NUM_PROCESSES=8 MAX_STEPS=30 \
#     bash examples/rl/trl/run_grpo_fsdp.sh 1

set -euo pipefail

FSDP_VERSION="${1:?Usage: $0 <1|2>}"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

ACCEL_CONFIG="${ACCEL_CONFIG:-${REPO_ROOT}/examples/rl/trl/accelerate/fsdp${FSDP_VERSION}.yaml}"

if [[ ! -f "${ACCEL_CONFIG}" ]]; then
    echo "ERROR: accelerate config not found: ${ACCEL_CONFIG}" >&2
    exit 1
fi

MODEL_NAME="${MODEL_NAME:-hf-internal-testing/tiny-random-LlamaForCausalLM}"
TOKENIZER_NAME="${TOKENIZER_NAME_OR_PATH:-}"
DATASET_NAME="${DATASET_NAME:-trl-lib/Capybara}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/trl-grpo-smoke}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-2}"
LR="${LR:-5e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
SEQ_LENGTH="${SEQ_LENGTH:-}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
SEED="${SEED:-1234}"

LINEAR_FP8="${LINEAR_FP8:-0}"
LORA_RANK="${LORA_RANK:-0}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"

LUMEN_NORM="${LUMEN_NORM:-0}"
LUMEN_FP8_ATTN="${LUMEN_FP8_ATTN:-none}"
LUMEN_FP8_ACTIVATION_STORE="${LUMEN_FP8_ACTIVATION_STORE:-0}"
LUMEN_FP8_PARAM_GATHER="${LUMEN_FP8_PARAM_GATHER:-0}"
LUMEN_FUSED_MLP="${LUMEN_FUSED_MLP:-0}"
LUMEN_CPU_OFFLOAD="${LUMEN_CPU_OFFLOAD:-0}"
LUMEN_FP8_CHECKPOINT="${LUMEN_FP8_CHECKPOINT:-0}"
FP8_PARAM_MANAGER="${FP8_PARAM_MANAGER:-0}"
USE_8BIT_ADAM="${USE_8BIT_ADAM:-0}"

CMD=(
    python -m accelerate.commands.launch
    --config_file "${ACCEL_CONFIG}"
    --num_processes "${NUM_PROCESSES}"
    "${REPO_ROOT}/examples/rl/trl/run_grpo_fsdp.py"
    --model-name-or-path "${MODEL_NAME}"
    --output-dir "${OUTPUT_DIR}"
    --fsdp-version "${FSDP_VERSION}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --gradient-accumulation-steps "${GRAD_ACCUM}"
    --max-steps "${MAX_STEPS}"
    --lr "${LR}"
    --lr-warmup-steps "${LR_WARMUP_STEPS}"
    --log-interval "${LOG_INTERVAL}"
    --save-interval "${SAVE_INTERVAL}"
    --max-prompt-length "${MAX_PROMPT_LENGTH}"
    --max-completion-length "${MAX_COMPLETION_LENGTH}"
    --num-generations "${NUM_GENERATIONS}"
    --warmup-steps "${WARMUP_STEPS}"
    --lora-rank "${LORA_RANK}"
    --lora-alpha "${LORA_ALPHA}"
    --lora-dropout "${LORA_DROPOUT}"
)

if [[ -n "${TRAIN_DATA_PATH}" ]]; then
    CMD+=(--train-data-path "${TRAIN_DATA_PATH}")
else
    CMD+=(--dataset-name "${DATASET_NAME}")
fi

if [[ -n "${TOKENIZER_NAME}" ]]; then
    CMD+=(--tokenizer-name-or-path "${TOKENIZER_NAME}")
fi

if [[ -n "${SEQ_LENGTH}" ]]; then
    CMD+=(--seq-length "${SEQ_LENGTH}")
fi

if [[ "${LINEAR_FP8}" == "1" ]]; then
    CMD+=(--linear-fp8)
fi

if [[ "${LUMEN_NORM}" == "1" ]]; then
    CMD+=(--lumen-norm)
fi

if [[ "${LUMEN_FP8_ATTN}" != "none" ]]; then
    CMD+=(--lumen-fp8-attn "${LUMEN_FP8_ATTN}")
fi

if [[ "${LUMEN_FP8_ACTIVATION_STORE}" == "1" ]]; then
    CMD+=(--lumen-fp8-activation-store)
fi

if [[ "${LUMEN_FP8_PARAM_GATHER}" == "1" ]]; then
    CMD+=(--lumen-fp8-param-gather)
fi

if [[ "${LUMEN_FUSED_MLP}" == "1" ]]; then
    CMD+=(--lumen-fused-mlp)
fi

if [[ "${LUMEN_CPU_OFFLOAD}" == "1" ]]; then
    CMD+=(--lumen-cpu-offload)
fi

if [[ "${LUMEN_FP8_CHECKPOINT}" == "1" ]]; then
    CMD+=(--lumen-fp8-checkpoint)
fi

if [[ "${FP8_PARAM_MANAGER}" == "1" ]]; then
    CMD+=(--fp8-param-manager)
fi

if [[ "${USE_8BIT_ADAM}" == "1" ]]; then
    CMD+=(--use-8bit-adam)
fi

echo "=== TRL + Lumen GRPO — FSDP${FSDP_VERSION} ==="
echo "Model:     ${MODEL_NAME}"
echo "Output:    ${OUTPUT_DIR}"
echo "Processes: ${NUM_PROCESSES}"
echo "Steps:     ${MAX_STEPS}"
echo "Config:    ${ACCEL_CONFIG}"
echo ""

exec "${CMD[@]}"
