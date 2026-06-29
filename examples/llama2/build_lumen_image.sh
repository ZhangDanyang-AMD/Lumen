#!/bin/bash
# Build a Docker image with the latest Lumen code baked in on top of the
# base lumen/llama2:latest image. Run from the Lumen repo root or from anywhere.
set -euo pipefail

LUMEN_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TAG="${1:-dev}"
IMAGE="lumen/llama2:${TAG}"

echo "[build] Context: ${LUMEN_DIR}"
echo "[build] Target:  ${IMAGE}"

docker build \
    --file "${LUMEN_DIR}/Dockerfile.lumen" \
    --tag  "${IMAGE}" \
    "${LUMEN_DIR}"

echo "[build] Done: ${IMAGE}"
echo "Run training with:"
echo "  bash /mnt/raid0/leiwu/mlperf/run_mlperf_70b_latest_blockwise2d.sh"
