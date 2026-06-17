#!/bin/bash
# Launch Qwen2.5-Coder-32B CPT training in Docker with Lumen FSDP2.
#
# Mid-train checkpoints go to /dev/shm (fast, tmpfs).
# Only the final checkpoint is also saved to HOST_RESULTS (persistent).
#
# Usage:
#   bash run_cpt.sh                                   # defaults
#   MAX_STEPS=5 bash run_cpt.sh                       # smoke test
#   MODEL=/path/to/model TRAIN_DATA=/path bash run_cpt.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_LUMEN="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---- Host paths (overridable via env) ----------------------------------------
HOST_MODEL="${HOST_MODEL:-/dev/shm/qwen2.5-coder-32b}"
HOST_DATA="${HOST_DATA:-/home/danyzhan/flydsl-agent-dataset}"
HOST_RESULTS="${HOST_RESULTS:-/home/danyzhan/cpt-results}"
IMAGE="${IMAGE:-lumen/flydsl-cpt:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-lumen_flydsl_cpt}"

mkdir -p "${HOST_RESULTS}"

# ---- Load config -------------------------------------------------------------
source "${SCRIPT_DIR}/config_cpt.sh"

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# ---- Build torchrun command --------------------------------------------------
CMD="torchrun --nproc_per_node=${NGPU} train_cpt.py"
CMD+=" --backend fsdp"
CMD+=" --model-name-or-path /model"
CMD+=" --train-data-path /data/data/cpt/train-00000-of-00001.jsonl"
CMD+=" --seq-length ${SEQ_LEN}"
CMD+=" --micro-batch-size ${MBS}"
CMD+=" --gradient-accumulation-steps ${GRAD_ACCUM}"
CMD+=" --max-steps ${MAX_STEPS}"
CMD+=" --epochs ${EPOCHS}"
CMD+=" --lr ${LR}"
CMD+=" --min-lr ${MIN_LR}"
CMD+=" --lr-warmup-steps ${LR_WARMUP_STEPS}"
CMD+=" --weight-decay ${WEIGHT_DECAY}"
CMD+=" --max-grad-norm ${MAX_GRAD_NORM}"
CMD+=" --log-interval ${LOG_INTERVAL}"
CMD+=" --save-interval ${SAVE_INTERVAL}"
CMD+=" --save-dir ${SAVE_DIR}"
CMD+=" --results-dir /results/final"
CMD+=" --num-workers ${NUM_WORKERS}"
CMD+=" --sharding-strategy full_shard"
CMD+=" --fsdp-version 2"

# LoRA
CMD+=" --lora-rank ${LORA_RANK}"
CMD+=" --lora-alpha ${LORA_ALPHA}"
CMD+=" --lora-dropout ${LORA_DROPOUT}"

# Lumen optimizations (AITER operator patching)
[ "${LUMEN_NORM:-0}" -eq 1 ] && CMD+=" --lumen-norm"

echo "================================================================"
echo " FlyDSL Agent — CPT Training (Lumen FSDP2)"
echo "  Image:      ${IMAGE}"
echo "  Model:      ${HOST_MODEL}"
echo "  Dataset:    ${HOST_DATA}"
echo "  Results:    ${HOST_RESULTS}"
echo "  Ckpt dir:   /dev/shm/cpt-checkpoints (fast mid-train saves)"
echo "  GPUs:       ${NGPU}"
echo "  Batch:      MBS=${MBS} x accum=${GRAD_ACCUM} x GPU=${NGPU} = $((MBS * GRAD_ACCUM * NGPU))"
echo "  LoRA:       rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "  Steps:      ${MAX_STEPS} (${EPOCHS} epochs)"
echo "================================================================"

docker run --rm --init \
    --name "${CONTAINER_NAME}" \
    --device /dev/dri --device /dev/kfd \
    --group-add video --group-add render \
    --ipc=host --network=host \
    --security-opt=seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    -v "${HOST_LUMEN}/lumen:/workspace/Lumen/lumen" \
    -v "${HOST_LUMEN}/experiments:/workspace/Lumen/experiments" \
    -v "${HOST_MODEL}:/model:ro" \
    -v "${HOST_DATA}:/data:ro" \
    -v "${HOST_RESULTS}:/results" \
    -v /dev/shm:/devshm \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e TOKENIZERS_PARALLELISM=false \
    -e NCCL_MIN_P2P_NCHANNELS="${NCCL_MIN_P2P_NCHANNELS}" \
    -e NCCL_MIN_CTAS="${NCCL_MIN_CTAS}" \
    -e NCCL_NCHANNELS_PER_NET_PEER="${NCCL_NCHANNELS_PER_NET_PEER}" \
    -e NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE}" \
    -e TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS}" \
    -e HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA}" \
    -e NCCL_IB_DISABLE="${NCCL_IB_DISABLE}" \
    -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
    -e NCCL_DEBUG="${NCCL_DEBUG}" \
    -e USE_HIPBLASLT="${USE_HIPBLASLT}" \
    -e TORCH_BLAS_PREFER_HIPBLASLT="${TORCH_BLAS_PREFER_HIPBLASLT}" \
    -e CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS}" \
    -e OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    -e TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE}" \
    -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 \
    -e NCCL_TIMEOUT=7200000 \
    "${IMAGE}" \
    bash -c "
set -euo pipefail
cd /workspace/Lumen/experiments/flydsl-agent/cpt
${CMD} 2>&1 | tee /results/train.log
"
