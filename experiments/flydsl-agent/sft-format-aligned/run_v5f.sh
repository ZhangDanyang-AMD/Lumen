#!/bin/bash
# Launch v5f SFT training — full retrain from base model (NOT from v5e).
# Same hyperparams as v5e (LoRA r=64, 3 epochs, lr 1e-5) but with
# format-aligned data (kernel samples have <plan>+<code> tags).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_LUMEN="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SFT_DIR="${SCRIPT_DIR}/../sft"

# v5f trains from BASE model, not from v5e
HOST_MODEL="${HOST_MODEL:-/dev/shm/qwen2.5-coder-32b}"
HOST_DATA="${HOST_DATA:-/home/danyzhan/flydsl-agent-dataset}"
HOST_RESULTS="${HOST_RESULTS:-/home/danyzhan/sft-results}"
IMAGE="${IMAGE:-lumen/flydsl-cpt:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-lumen_v5f_sft}"
LOG_NAME="auto_pipeline-v5f.log"

mkdir -p "${HOST_RESULTS}"
source "${SCRIPT_DIR}/config_v5f.sh"

# Compute steps: 3 epochs
TRAIN_FILE="${HOST_DATA}/data/format_aligned/train.jsonl"
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "ERROR: Training data not found at ${TRAIN_FILE}"
    echo "Run generate_v5f_data.py first."
    exit 1
fi
NUM_SAMPLES=$(wc -l < "${TRAIN_FILE}")
GBS=$((MBS * GRAD_ACCUM * NGPU))
STEPS_PER_EPOCH=$(( (NUM_SAMPLES + GBS - 1) / GBS ))
EPOCHS=3
MAX_STEPS=$(( STEPS_PER_EPOCH * EPOCHS ))
LR_WARMUP_STEPS=$(( MAX_STEPS * 5 / 100 ))  # 5% warmup

echo "================================================================"
echo " FlyDSL Agent — v5f SFT Training (v5e data + format alignment)"
echo "  Base Model:  ${HOST_MODEL}"
echo "  Dataset:     ${TRAIN_FILE}"
echo "  Samples:     ${NUM_SAMPLES}"
echo "  Results:     ${HOST_RESULTS}"
echo "  GPUs:        ${NGPU}"
echo "  GBS:         ${GBS}"
echo "  LoRA:        r=${LORA_RANK}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
echo "  Steps:       ${MAX_STEPS} (${EPOCHS} epochs x ${STEPS_PER_EPOCH})"
echo "  Warmup:      ${LR_WARMUP_STEPS}"
echo "  LR:          ${LR}"
echo "================================================================"

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

CMD="torchrun --nproc_per_node=${NGPU} train_sft.py"
CMD+=" --backend fsdp --fsdp-version 2"
CMD+=" --model-name-or-path /model"
CMD+=" --train-data-path ${TRAIN_DATA}"
CMD+=" --val-data-path ${VAL_DATA}"
CMD+=" --seq-length ${SEQ_LEN}"
CMD+=" --micro-batch-size ${MBS}"
CMD+=" --gradient-accumulation-steps ${GRAD_ACCUM}"
CMD+=" --max-steps ${MAX_STEPS}"
CMD+=" --lr ${LR} --min-lr ${MIN_LR}"
CMD+=" --lr-warmup-steps ${LR_WARMUP_STEPS}"
CMD+=" --weight-decay ${WEIGHT_DECAY}"
CMD+=" --max-grad-norm ${MAX_GRAD_NORM}"
CMD+=" --log-interval ${LOG_INTERVAL}"
CMD+=" --save-interval ${SAVE_INTERVAL}"
CMD+=" --save-dir /results/v5f_ckpt"
CMD+=" --results-dir /results/v5f_final"
CMD+=" --eval-interval ${EVAL_INTERVAL}"
CMD+=" --num-workers ${NUM_WORKERS}"
CMD+=" --sharding-strategy full_shard"
CMD+=" --lora-rank ${LORA_RANK}"
CMD+=" --lora-alpha ${LORA_ALPHA}"
CMD+=" --lora-dropout ${LORA_DROPOUT}"

docker run --rm --init \
    --name "${CONTAINER_NAME}" \
    --device /dev/dri --device /dev/kfd \
    --group-add video --group-add render \
    --ipc=host --network=host \
    --security-opt=seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    -v "${HOST_LUMEN}/lumen:/workspace/Lumen/lumen" \
    -v "${HOST_LUMEN}/experiments:/workspace/Lumen/experiments" \
    -v "${SFT_DIR}:/workspace/Lumen/experiments/flydsl-agent/sft" \
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
    "${IMAGE}" \
    bash -c "
set -euo pipefail
cd /workspace/Lumen/experiments/flydsl-agent/sft
${CMD} 2>&1 | tee /results/${LOG_NAME}
"
