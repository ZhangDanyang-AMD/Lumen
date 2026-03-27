#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   # FSDP1 smoke
#   MODEL_NAME=hf-internal-testing/tiny-random-LlamaForCausalLM \
#   MAX_STEPS=2 \
#   bash examples/rl/trl/run_grpo_fsdp.sh 1
#
#   # FSDP2 smoke
#   MODEL_NAME=hf-internal-testing/tiny-random-LlamaForCausalLM \
#   MAX_STEPS=2 \
#   bash examples/rl/trl/run_grpo_fsdp.sh 2
#
#   # LLaMA2-70B GRPO bring-up with a local JSONL prompt dataset
#   MODEL_NAME=meta-llama/Llama-2-70b-hf \
#   TOKENIZER_NAME_OR_PATH=meta-llama/Llama-2-70b-hf \
#   TRAIN_DATA_PATH=/data/rl_prompts.jsonl \
#   OUTPUT_DIR=/results/trl-grpo-70b \
#   NUM_PROCESSES=8 \
#   MAX_STEPS=100 \
#   GRAD_ACCUM=8 \
#   bash examples/rl/trl/run_grpo_fsdp.sh 1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
cd "${ROOT_DIR}"

FSDP_VERSION="${1:-1}"
ACCELERATE_CFG="examples/rl/trl/accelerate/fsdp1.yaml"
if [[ "${FSDP_VERSION}" == "2" ]]; then
  ACCELERATE_CFG="examples/rl/trl/accelerate/fsdp2.yaml"
fi

MODEL_NAME="${MODEL_NAME:-hf-internal-testing/tiny-random-LlamaForCausalLM}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-${MODEL_NAME}}"
DATASET_NAME="${DATASET_NAME:-trl-lib/Capybara}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/trl-grpo-smoke}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MAX_STEPS="${MAX_STEPS:-2}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-1e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
SEQ_LENGTH="${SEQ_LENGTH:-$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
# Keep warmup opt-in until the target TRL + Accelerate + FSDP stack is validated.
LINEAR_FP8="${LINEAR_FP8:-0}"
LORA_RANK="${LORA_RANK:-0}"
LORA_ALPHA="${LORA_ALPHA:-32.0}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"

CLI_ARGS=(
  --model-name-or-path "${MODEL_NAME}"
  --tokenizer-name-or-path "${TOKENIZER_NAME_OR_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --fsdp-version "${FSDP_VERSION}"
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACCUM}"
  --max-steps "${MAX_STEPS}"
  --lr "${LR}"
  --lr-warmup-steps "${LR_WARMUP_STEPS}"
  --log-interval "${LOG_INTERVAL}"
  --save-interval "${SAVE_INTERVAL}"
  --seq-length "${SEQ_LENGTH}"
  --max-prompt-length "${MAX_PROMPT_LENGTH}"
  --max-completion-length "${MAX_COMPLETION_LENGTH}"
  --num-generations "${NUM_GENERATIONS}"
  --warmup-steps "${WARMUP_STEPS}"
  --lora-rank "${LORA_RANK}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
)

if [[ -n "${TRAIN_DATA_PATH}" ]]; then
  CLI_ARGS+=(--train-data-path "${TRAIN_DATA_PATH}")
else
  CLI_ARGS+=(--dataset-name "${DATASET_NAME}")
  if [[ -n "${DATASET_CONFIG_NAME}" ]]; then
    CLI_ARGS+=(--dataset-config-name "${DATASET_CONFIG_NAME}")
  fi
  CLI_ARGS+=(--dataset-split "${DATASET_SPLIT}")
fi

if [[ "${LINEAR_FP8}" == "1" ]]; then
  CLI_ARGS+=(--linear-fp8)
fi

accelerate launch \
  --config_file "${ACCELERATE_CFG}" \
  --num_processes "${NUM_PROCESSES}" \
  examples/rl/trl/run_grpo_fsdp.py \
  "${CLI_ARGS[@]}"
