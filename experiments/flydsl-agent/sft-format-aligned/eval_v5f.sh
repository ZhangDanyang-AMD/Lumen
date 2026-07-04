#!/bin/bash
# Evaluate v5f model: API Score + format compliance + sandbox compilation.
# Compares against v5e baseline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_LUMEN="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

V5F_MODEL="${V5F_MODEL:-/home/danyzhan/sft-results/Qwen2.5-Coder-SFT-v5f}"
V5E_MODEL="${V5E_MODEL:-/home/danyzhan/sft-results/Qwen2.5-Coder-SFT-v5e}"
HOST_RESULTS="${HOST_RESULTS:-/home/danyzhan/sft-results}"
IMAGE="${IMAGE:-lumen/flydsl-cpt:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-lumen_v5f_eval}"

if [ ! -d "${V5F_MODEL}" ]; then
    echo "ERROR: v5f model not found at ${V5F_MODEL}"
    echo "Run export_v5f.sh first."
    exit 1
fi

echo "================================================================"
echo " Evaluate v5f Model"
echo "  v5f Model:  ${V5F_MODEL}"
echo "  v5e Base:   ${V5E_MODEL}"
echo "  Output:     ${HOST_RESULTS}/benchmark_v5f.json"
echo "================================================================"

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

docker run --rm --init \
    --name "${CONTAINER_NAME}" \
    --device /dev/dri --device /dev/kfd \
    --group-add video --group-add render \
    --ipc=host --network=host \
    --security-opt=seccomp=unconfined \
    -v "${HOST_LUMEN}:/workspace/Lumen" \
    -v "${HOST_RESULTS}:/sft-results" \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    "${IMAGE}" \
    python /workspace/Lumen/experiments/flydsl-agent/sft-format-aligned/eval_format_aligned.py \
        --model /sft-results/Qwen2.5-Coder-SFT-v5f \
        --base-model /sft-results/Qwen2.5-Coder-SFT-v5e \
        --sandbox \
        --output /sft-results/benchmark_v5f.json \
    2>&1 | tee "${HOST_RESULTS}/eval_v5f.log"

echo "Evaluation done. Results at ${HOST_RESULTS}/benchmark_v5f.json"
