#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# LLaMA2 LoRA SFT — 8x MI308X, PyTorch FSDP backend (generic launcher).
#
# Model size and precision are chosen by the config (CONFIG=) and, for the 7B
# config, the MODE env var (bf16 / fp8_delayed / fp8_blockwise). Launches the
# lumen/llama2 container, overlays the host Lumen package and examples (keeping
# the image's compiled third_party AITER/mori), mounts the HF model + dataset,
# and runs run_finetune.sh.
#
# Overridable env: CONFIG, MODE, HOST_MODEL, HOST_DATA, HOST_RESULTS, IMAGE,
# CONTAINER_NAME.
#
# Examples:
#   bash run_fsdp_lora_mi308.sh                          # 70B bf16 (defaults)
#   HOST_MODEL=/path/to/7b HOST_RESULTS=/path/out \
#     CONFIG=config_MI308X_fsdp_lora_7b.sh MODE=fp8_delayed \
#     bash run_fsdp_lora_mi308.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_LUMEN="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---- Host paths (model / data / results) — overridable via env --------------
HOST_MODEL="${HOST_MODEL:-/mnt/raid0/leiwu/mlperf/llama2_70b_hf}"
HOST_DATA="${HOST_DATA:-/mnt/raid0/leiwu/mlperf/data_mlperf}"   # answer-only loss-mask (MLPerf-aligned)
HOST_RESULTS="${HOST_RESULTS:-/mnt/raid0/leiwu/mlperf/results/fsdp_lora_bf16}"
CONFIG="${CONFIG:-config_MI308X_fsdp_lora_70b.sh}"
IMAGE="${IMAGE:-lumen/llama2:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-lumen_fsdp_lora_mi308}"

mkdir -p "${HOST_RESULTS}"

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

docker run --rm --init \
    --name "${CONTAINER_NAME}" \
    --device /dev/dri --device /dev/kfd \
    --group-add video --group-add render \
    --ipc=host --network=host \
    --security-opt=seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    -v "${HOST_LUMEN}/lumen:/workspace/Lumen/lumen" \
    -v "${HOST_LUMEN}/examples:/workspace/Lumen/examples" \
    -v "${HOST_MODEL}:/model-hf:ro" \
    -v "${HOST_DATA}:/data:ro" \
    -v "${HOST_RESULTS}:/results" \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e TOKENIZERS_PARALLELISM=false \
    -e PYTORCH_TUNABLEOP_ENABLED="${PYTORCH_TUNABLEOP_ENABLED:-0}" \
    -e RUN_CONFIG="${CONFIG}" \
    -e MODE="${MODE:-}" \
    -e TRAIN_STEPS="${TRAIN_STEPS:-}" \
    -e SAVE_INTERVAL="${SAVE_INTERVAL:-}" \
    -e SAVE_CKPT="${SAVE_CKPT:-}" \
    "${IMAGE}" \
    bash -c '
set -euo pipefail
cd /workspace/Lumen/examples/llama2
CONFIG="${RUN_CONFIG}" BACKEND=fsdp bash run_finetune.sh 2>&1 \
    | tee /results/train.log
'
