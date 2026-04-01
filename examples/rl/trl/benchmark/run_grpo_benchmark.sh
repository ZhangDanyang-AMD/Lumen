#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Shared launcher for Lumen FP8 GRPO benchmark.
# All tunables are read from env vars, which model-specific scripts export.
#
# Direct usage (fallback defaults):
#   bash examples/rl/trl/benchmark/run_grpo_benchmark.sh R1
#
# Preferred usage via model configs:
#   bash examples/rl/trl/benchmark/qwen2.5-0.5b/run.sh R1
#   bash examples/rl/trl/benchmark/llama-3.1-8b/run.sh R4

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/examples/rl/trl/benchmark"
ACCEL_DIR="${REPO_ROOT}/examples/rl/trl/accelerate"

# ─── Tunables ────────────────────────────────────────────────────────────
RUN_ID="${1:?Usage: $0 <R1|R2|R3|R4|R5|ALL>}"
MODEL_DIR="${MODEL_DIR:-/dev/shm/model/qwen2.5-0.5b-instruct}"  # override via model-specific run.sh
DATASET_NAME="${DATASET_NAME:-trl-lib/DeepMath-103K}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/outputs/benchmark}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
FSDP_VERSION="${FSDP_VERSION:-2}"

MAX_STEPS="${MAX_STEPS:-2000}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-1e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
BETA="${BETA:-0.0}"
EARLY_STOP="${EARLY_STOP:-0}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-100}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
SEED="${SEED:-42}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
PERF_WARMUP_STEPS="${PERF_WARMUP_STEPS:-10}"
EVAL_AFTER_TRAINING="${EVAL_AFTER_TRAINING:-0}"
EVAL_SAMPLES="${EVAL_SAMPLES:-200}"
PERIODIC_EVAL="${PERIODIC_EVAL:-1}"
PERIODIC_EVAL_EVERY="${PERIODIC_EVAL_EVERY:-100}"
PERIODIC_EVAL_SAMPLES="${PERIODIC_EVAL_SAMPLES:-50}"

# ─── Functions ───────────────────────────────────────────────────────────

run_single() {
    local rid="$1"
    local output_dir="${OUTPUT_BASE}/${rid}"
    local accel_config="${ACCEL_CONFIG:-${ACCEL_DIR}/fsdp${FSDP_VERSION}.yaml}"

    if [[ ! -f "${accel_config}" ]]; then
        echo "ERROR: accelerate config not found: ${accel_config}" >&2
        exit 1
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  Lumen FP8 GRPO Benchmark — Run ${rid}                    ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║  Model:      ${MODEL_DIR}"
    echo "║  Output:     ${output_dir}"
    echo "║  Processes:  ${NUM_PROCESSES}"
    echo "║  Steps:      ${MAX_STEPS} (warmup: ${PERF_WARMUP_STEPS})"
    echo "║  LR:         ${LR} (${LR_SCHEDULER_TYPE}, warmup: ${LR_WARMUP_STEPS})"
    echo "║  Beta/KL:    ${BETA} | WD: ${WEIGHT_DECAY}"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    mkdir -p "${output_dir}"

    python -m accelerate.commands.launch \
        --config_file "${accel_config}" \
        --num_processes "${NUM_PROCESSES}" \
        "${SCRIPT_DIR}/run_grpo_benchmark.py" \
        --run-id "${rid}" \
        --model-name-or-path "${MODEL_DIR}" \
        --dataset-name "${DATASET_NAME}" \
        --dataset-split "${DATASET_SPLIT}" \
        --output-dir "${output_dir}" \
        --micro-batch-size "${MICRO_BATCH_SIZE}" \
        --gradient-accumulation-steps "${GRAD_ACCUM}" \
        --max-steps "${MAX_STEPS}" \
        --lr "${LR}" \
        --lr-warmup-steps "${LR_WARMUP_STEPS}" \
        --lr-scheduler-type "${LR_SCHEDULER_TYPE}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --beta "${BETA}" \
        --log-interval "${LOG_INTERVAL}" \
        --max-prompt-length "${MAX_PROMPT_LENGTH}" \
        --max-completion-length "${MAX_COMPLETION_LENGTH}" \
        --num-generations "${NUM_GENERATIONS}" \
        --seed "${SEED}" \
        --fsdp-version "${FSDP_VERSION}" \
        --perf-warmup-steps "${PERF_WARMUP_STEPS}" \
        $([ "${EVAL_AFTER_TRAINING}" = "1" ] && echo "--eval-after-training --eval-samples ${EVAL_SAMPLES}") \
        $([ "${EARLY_STOP}" = "1" ] && echo "--early-stop --early-stop-patience ${EARLY_STOP_PATIENCE}") \
        $([ "${PERIODIC_EVAL}" = "1" ] && echo "--periodic-eval --periodic-eval-every ${PERIODIC_EVAL_EVERY} --periodic-eval-samples ${PERIODIC_EVAL_SAMPLES}") \
        2>&1 | tee "${output_dir}/run.log"

    echo ""
    echo "✓ Run ${rid} complete → ${output_dir}"
    echo ""
}

# ─── Main ────────────────────────────────────────────────────────────────

if [[ "${RUN_ID}" == "ALL" ]]; then
    for rid in R1 R2 R3 R4; do
        run_single "${rid}"
    done
    echo ""
    echo "Core runs (R1-R4) complete. R5 (LoRA stretch) must be run separately: $0 R5"
    echo "Run the analysis: python ${SCRIPT_DIR}/analyze_benchmark.py ${OUTPUT_BASE}"
else
    run_single "${RUN_ID}"
fi
