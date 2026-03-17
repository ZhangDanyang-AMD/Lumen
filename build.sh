#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_NAME="${IMAGE_NAME:-lumen/tests:latest}"

echo "==> Building test image: ${IMAGE_NAME}"
docker build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t "${IMAGE_NAME}" \
    "${SCRIPT_DIR}"

echo "==> Running tests"
docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --shm-size=16g \
    -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    "${IMAGE_NAME}" \
    pytest tests/ops/ -v --tb=short "$@"
