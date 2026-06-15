#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# LLaMA2-7B LoRA SFT — 8x MI308X, PyTorch FSDP backend.
#
# Thin convenience wrapper around run_fsdp_lora_mi308.sh that pins the 7B config
# and sensible 7B defaults. Precision/scaling is chosen by MODE:
#   bash run_fsdp_lora_7b_mi308.sh                    # bf16 (default)
#   MODE=fp8_delayed   bash run_fsdp_lora_7b_mi308.sh
#   MODE=fp8_blockwise bash run_fsdp_lora_7b_mi308.sh
#
# The 7B HF checkpoint is not bundled. Override HOST_MODEL if it lives elsewhere:
#   HOST_MODEL=/path/to/llama2_7b_hf bash run_fsdp_lora_7b_mi308.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${MODE:-bf16}"
HOST_MODEL="${HOST_MODEL:-/mnt/raid0/leiwu/mlperf/llama2_7b_hf}"

if [ ! -d "${HOST_MODEL}" ]; then
    echo "ERROR: 7B model dir not found: ${HOST_MODEL}"
    echo "  Set HOST_MODEL=/path/to/llama2_7b_hf (a HuggingFace LlamaForCausalLM dir)."
    exit 1
fi

# Delegate to the generic launcher; results dir defaults to a per-mode path.
HOST_MODEL="${HOST_MODEL}" \
HOST_RESULTS="${HOST_RESULTS:-/mnt/raid0/leiwu/mlperf/results/fsdp_lora_7b_${MODE}}" \
CONFIG="config_MI308X_fsdp_lora_7b.sh" \
CONTAINER_NAME="${CONTAINER_NAME:-lumen_fsdp_7b_${MODE}}" \
MODE="${MODE}" \
bash "${SCRIPT_DIR}/run_fsdp_lora_mi308.sh"
