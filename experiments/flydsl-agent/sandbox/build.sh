#!/bin/bash
# Build FlyDSL-Gym sandbox Docker image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-flydsl-gym:latest}"
BASE_IMAGE="${BASE_IMAGE:-rocm/vllm-dev:nightly_main_20260603}"

echo "Building FlyDSL-Gym sandbox: ${IMAGE_NAME}"
echo "  Base: ${BASE_IMAGE}"

docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${IMAGE_NAME}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${SCRIPT_DIR}"

echo "Done. Test with:"
echo "  docker run --rm --device /dev/dri --device /dev/kfd --group-add video --ipc=host ${IMAGE_NAME} python3 -c 'import flydsl; print(\"OK\")'"
