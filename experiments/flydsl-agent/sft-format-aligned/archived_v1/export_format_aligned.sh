#!/bin/bash
# Export Format-Aligned DCP checkpoint to HF format with LoRA merged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_LUMEN="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

HOST_MODEL="${HOST_MODEL:-/home/danyzhan/sft-results/Qwen2.5-Coder-SFT-v5e}"
HOST_RESULTS="${HOST_RESULTS:-/home/danyzhan/sft-results}"
DCP_PATH="${HOST_RESULTS}/format_aligned_final/final"
OUTPUT="${HOST_RESULTS}/Qwen2.5-Coder-SFT-Format-Aligned"
IMAGE="${IMAGE:-lumen/flydsl-cpt:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-lumen_format_aligned_export}"

LORA_RANK=32
LORA_ALPHA=64

if [ ! -d "${DCP_PATH}" ]; then
    echo "ERROR: DCP checkpoint not found at ${DCP_PATH}"
    echo "Run run_format_aligned.sh first."
    exit 1
fi

echo "================================================================"
echo " Export Format-Aligned Model"
echo "  Base:    ${HOST_MODEL}"
echo "  DCP:     ${DCP_PATH}"
echo "  Output:  ${OUTPUT}"
echo "  LoRA:    r=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "================================================================"

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

docker run --rm --init \
    --name "${CONTAINER_NAME}" \
    --device /dev/dri --device /dev/kfd \
    --group-add video --group-add render \
    --ipc=host --network=host \
    --security-opt=seccomp=unconfined \
    -v "${HOST_LUMEN}:/workspace/Lumen" \
    -v "${HOST_MODEL}:/model:ro" \
    -v "${HOST_RESULTS}:/sft-results" \
    -v /dev/shm:/devshm \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    "${IMAGE}" \
    bash -c "
set -euo pipefail
cd /workspace/Lumen/experiments/flydsl-agent/cpt
python export_hf.py \
    --base-model /model \
    --dcp-path /sft-results/format_aligned_final/final \
    --output /sft-results/Qwen2.5-Coder-SFT-Format-Aligned \
    --lora-rank ${LORA_RANK} \
    --lora-alpha ${LORA_ALPHA}
"

echo "Export done. Model at ${OUTPUT}"
