#!/bin/bash
# Build the gfx1250 Lumen/LLaMA image with the local AITER checkout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-lumen/llama31:mi450}"
BASE_IMAGE="${BASE_IMAGE:-local/qwen38-gfx1250-aiter:0.1.26.dev21-g9fd025efd-gemmtuned-prefill}"
AITER_DIR="${AITER_DIR:-${REPO_ROOT}/../aiter}"

if [[ ! -f "${AITER_DIR}/setup.py" ]]; then
    echo "ERROR: AITER_DIR does not contain setup.py: ${AITER_DIR}" >&2
    exit 1
fi

docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-context "aiter_source=${AITER_DIR}" \
    -t "${IMAGE_NAME}" \
    -f "${SCRIPT_DIR}/Dockerfile.mi450" \
    "${REPO_ROOT}"
