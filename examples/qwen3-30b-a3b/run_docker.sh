#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Run run_pretrain_qwen3_30b_a3b_megatron.sh in a Lumen container.
#
# The image provides PyTorch/ROCm, apex and Megatron-LM; this checkout is
# mounted over /workspace/Lumen (third_party/aiter must be checked out).
# Python packages the image lacks (MORI, a newer Triton, tensorboard for W&B)
# can be mounted as prebuilt site directories through EXTRA_SITE.
#
# Usage:
#   DATA_DIR=... TOKENIZER_DIR=... CKPT_DIR=... RESULTS_DIR=... bash run_docker.sh
#   EXPERT_RECIPE=bf16 LUMEN_ATTN_LINEAR_RECIPE=bf16 LUMEN_ROPE_IMPL=aiter bash run_docker.sh
#
# Variables:
#   IMAGE          container image (default below)
#   LUMEN_DIR      Lumen checkout (default: this repository)
#   MEGATRON_DIR   optional Megatron-LM checkout mounted over /workspace/Megatron-LM
#   DATA_DIR       directory with train.jsonl / validation.jsonl      -> /data
#   TOKENIZER_DIR  HF Qwen3-30B-A3B directory (tokenizer files)        -> /model
#   CKPT_DIR       Megatron EP4 torch checkpoint (latest_checkpointed_iteration.txt) -> /checkpoint
#   RESULTS_DIR    logs, W&B / TensorBoard output, checkpoints          -> /results
#   CACHE_DIR      AITER JIT / Triton / torch extension caches          -> /cache
#   EXTRA_SITE     colon-separated host site-packages dirs prepended to PYTHONPATH
#   RENDER_NODES   space-separated /dev/dri/renderD<N> numbers to expose (default: all GPUs)
#   NETRC          host .netrc with the W&B key (mounted read-only when set)
#   NETWORK        docker network mode (default host; W&B needs outbound access)
#   CONTAINER_NAME container name
# Any variable of config_MI350X_ep4_mxfp4.sh set in the environment is forwarded.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMEN_DIR=${LUMEN_DIR:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
IMAGE=${IMAGE:-docker.io/zhangdanyangamd/lumen@sha256:d6f91f8ba5ee967055dc8d1eba45b457afcacd9b251a2253c09d02d1560a757d}
DATA_DIR=${DATA_DIR:?set DATA_DIR}
TOKENIZER_DIR=${TOKENIZER_DIR:?set TOKENIZER_DIR}
CKPT_DIR=${CKPT_DIR:?set CKPT_DIR}
RESULTS_DIR=${RESULTS_DIR:?set RESULTS_DIR}
CACHE_DIR=${CACHE_DIR:-${RESULTS_DIR}/cache}
EXTRA_SITE=${EXTRA_SITE:-}
RENDER_NODES=${RENDER_NODES:-}
NETRC=${NETRC:-}
NETWORK=${NETWORK:-host}
CONTAINER_NAME=${CONTAINER_NAME:-lumen_qwen3_30b_a3b_${EXPERT_RECIPE:-a4w4}}

[ -n "$(ls -A "${LUMEN_DIR}/third_party/aiter" 2>/dev/null)" ] || {
    echo "ERROR: ${LUMEN_DIR}/third_party/aiter is empty"
    echo "  git submodule update --init third_party/aiter"
    exit 1
}
mkdir -p "${RESULTS_DIR}" "${CACHE_DIR}/aiter" "${CACHE_DIR}/triton" "${CACHE_DIR}/torch" "${CACHE_DIR}/home"

DEVICES=(--device=/dev/kfd)
for group in video render; do
    gid=$(getent group "${group}" | cut -d: -f3 || true)
    [ -n "${gid}" ] && DEVICES+=(--group-add "${gid}")
done
if [ -n "${RENDER_NODES}" ]; then
    for n in ${RENDER_NODES}; do DEVICES+=(--device="/dev/dri/renderD${n}"); done
else
    DEVICES+=(--device=/dev/dri)
fi

MOUNTS=(
    --mount "type=bind,src=${LUMEN_DIR},dst=/workspace/Lumen,readonly"
    --mount "type=bind,src=${DATA_DIR},dst=/data,readonly"
    --mount "type=bind,src=${TOKENIZER_DIR},dst=/model,readonly"
    --mount "type=bind,src=${CKPT_DIR},dst=/checkpoint,readonly"
    --mount "type=bind,src=${RESULTS_DIR},dst=/results"
    --mount "type=bind,src=${CACHE_DIR},dst=/cache"
)
[ -n "${MEGATRON_DIR:-}" ] && MOUNTS+=(--mount "type=bind,src=${MEGATRON_DIR},dst=/workspace/Megatron-LM,readonly")

PYPATH=""
i=0
IFS=':' read -r -a SITES <<< "${EXTRA_SITE}"
for site in "${SITES[@]}"; do
    [ -n "${site}" ] || continue
    MOUNTS+=(--mount "type=bind,src=${site},dst=/extra-site/${i},readonly")
    PYPATH="${PYPATH}/extra-site/${i}:"
    i=$((i + 1))
done
PYPATH="${PYPATH}/workspace/Lumen:/workspace/Lumen/third_party/aiter:/workspace/Megatron-LM"

ENV_ARGS=(
    --env "PYTHONPATH=${PYPATH}"
    --env HOME=/cache/home
    --env AITER_JIT_DIR=/cache/aiter
    --env TRITON_CACHE_DIR=/cache/triton
    --env TORCH_EXTENSIONS_DIR=/cache/torch
    --env TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor
    --env TRAIN_DATA=/data/train.jsonl
    --env VALID_DATA=/data/validation.jsonl
    --env TOKENIZER=/model
    --env LOAD_DIR=/checkpoint
    --env RESULTS_DIR=/results
    --env SAVE_DIR=/results/checkpoints
)
if [ -n "${NETRC}" ]; then
    MOUNTS+=(--mount "type=bind,src=${NETRC},dst=/cache/home/.netrc,readonly")
fi
FORWARD=(NGPU NNODES NODE_RANK MASTER_ADDR MASTER_PORT TP PP CP EP ETP SEQ_LEN MBS GBS
         TRAIN_ITERS EXIT_INTERVAL LR MIN_LR LR_WARMUP_ITERS WEIGHT_DECAY CLIP_GRAD
         ADAM_BETA1 ADAM_BETA2 ADAM_EPS AUX_LOSS_COEFF SEED EVAL_INTERVAL EVAL_ITERS
         SAVE_INTERVAL LOG_INTERVAL WANDB_PROJECT WANDB_ENTITY WANDB_EXP_NAME
         EXPERT_RECIPE LUMEN_ATTN_LINEAR_RECIPE LUMEN_ROPE_IMPL LUMEN_ATTN_BACKEND LUMEN_ATTN_AITER_TRITON_MHA LUMEN_ATTN_AITER_FUSED_BWD
         SONIC_MOE_MXFP4_WGRAD SONIC_MOE_BF16_WGRAD_MAIN_GRAD SONIC_MOE_PACK_CACHE
         R80_CACHE_GENERATION R80_DUAL_WEIGHT_PACK AITER_GROUPED_MXFP4_DUAL_V2
         LUMEN_MORI_PRESYNC_PACK CUDA_DEVICE_MAX_CONNECTIONS GPU_MAX_HW_QUEUES
         HIP_VISIBLE_DEVICES)
for name in "${FORWARD[@]}"; do
    [ -n "${!name+x}" ] && ENV_ARGS+=(--env "${name}=${!name}")
done
# apex picks its RoPE backend when it is first imported, before the launcher runs.
if [ "${LUMEN_ROPE_IMPL:-apex}" = "apex" ]; then
    ENV_ARGS+=(--env USE_ROCM_AITER_ROPE_BACKEND=0)
fi

docker run --rm --init --name "${CONTAINER_NAME}" \
    --network="${NETWORK}" \
    --ulimit memlock=-1:-1 --shm-size=16g --tmpfs /tmp:rw,size=64g \
    "${DEVICES[@]}" \
    "${MOUNTS[@]}" "${ENV_ARGS[@]}" \
    --entrypoint bash "${IMAGE}" \
    /workspace/Lumen/examples/qwen3-30b-a3b/run_pretrain_qwen3_30b_a3b_megatron.sh
