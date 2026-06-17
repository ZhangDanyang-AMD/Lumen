#!/bin/bash
# Sweep LoRA rank to find optimal CPT configuration.
# Runs rank=64, 128, 256 sequentially (8 GPUs each, can't parallel on same node).
#
# Usage:
#   bash sweep_lora_rank.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RANKS=(64 128 256)

for RANK in "${RANKS[@]}"; do
    ALPHA=$((RANK * 2))
    RESULT_DIR="/home/danyzhan/cpt-results/rank_${RANK}"
    mkdir -p "${RESULT_DIR}"

    echo ""
    echo "================================================================"
    echo "  SWEEP: LoRA rank=${RANK}, alpha=${ALPHA}, 10 epochs (232 steps)"
    echo "  Results: ${RESULT_DIR}"
    echo "================================================================"

    LORA_RANK="${RANK}" \
    LORA_ALPHA="${ALPHA}" \
    HOST_RESULTS="${RESULT_DIR}" \
    CONTAINER_NAME="lumen_cpt_r${RANK}" \
    SAVE_INTERVAL=0 \
    bash "${SCRIPT_DIR}/run_cpt.sh"

    echo ""
    echo "--- rank=${RANK} training log ---"
    grep "step.*loss" "${RESULT_DIR}/train.log" | head -5
    echo "..."
    grep "step.*loss" "${RESULT_DIR}/train.log" | tail -5
    echo ""
done

echo ""
echo "================================================================"
echo "  SWEEP COMPLETE — Summary"
echo "================================================================"
for RANK in "${RANKS[@]}"; do
    RESULT_DIR="/home/danyzhan/cpt-results/rank_${RANK}"
    FINAL_LOSS=$(grep "step.*loss" "${RESULT_DIR}/train.log" | tail -1 | grep -oP 'loss \K[0-9.]+')
    FIRST_LOSS=$(grep "step.*loss" "${RESULT_DIR}/train.log" | head -1 | grep -oP 'loss \K[0-9.]+')
    echo "  rank=${RANK}: loss ${FIRST_LOSS} → ${FINAL_LOSS}"
done
