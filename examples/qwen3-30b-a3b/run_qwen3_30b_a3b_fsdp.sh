#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Qwen3-30B-A3B MoE Training — 2×MI308X nodes (16 GPUs), FSDP2 + EP=8.
#
# Launches the lumen/llama2 container, overlays the host Lumen package and
# examples, mounts the model + dataset, and runs FSDP2 with overlapping
# dense-DP and EP process groups across 2 nodes.
#
# This script must be run on EACH node. Set NODE_RANK=0 on the master node,
# NODE_RANK=1 on the worker node.
#
# Example (master node):
#   NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
#   HOST_MODEL=/mnt/raid0/models/Qwen3-30B-A3B \
#   HOST_DATA=/mnt/raid0/danyzhan/datasets/alpaca \
#   TRAIN_FILE=train.jsonl VAL_FILE=test.jsonl \
#     bash run_qwen3_30b_a3b_fsdp.sh
#
# Example (worker node):
#   NODE_RANK=1 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
#   HOST_MODEL=/mnt/raid0/models/Qwen3-30B-A3B \
#   HOST_DATA=/mnt/raid0/danyzhan/datasets/alpaca \
#   TRAIN_FILE=train.jsonl VAL_FILE=test.jsonl \
#     bash run_qwen3_30b_a3b_fsdp.sh
#
# Single-node dense-DP=8 EP=8:
#   NNODES=1 DP_SIZE=8 EP_SIZE=8 \
#   HOST_MODEL=/mnt/raid0/models/Qwen3-30B-A3B \
#   HOST_DATA=/mnt/raid0/danyzhan/datasets/alpaca \
#   TRAIN_FILE=train.jsonl VAL_FILE=test.jsonl \
#     bash run_qwen3_30b_a3b_fsdp.sh
#
# Throughput-oriented config (memory permitting). TE grouped experts are
# not covered by Lumen blockwise2d FP8; use sequential or sonic:
#   MODE=fp8_blockwise2d EXPERT_BACKEND=sonic SHARDING=shard_grad_op GRAD_CKPT=0 \
#   AITER_ATTN=1 LUMEN_NORM=1 FUSE_ROPE=1 \
#     bash run_qwen3_30b_a3b_fsdp.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_LUMEN="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---- Host paths ----
HOST_MODEL="${HOST_MODEL:-/mnt/raid0/models/Qwen3-30B-A3B}"
HOST_DATA="${HOST_DATA:-/mnt/raid0/danyzhan/datasets/alpaca}"
HOST_RESULTS="${HOST_RESULTS:-${SCRIPT_DIR}/results}"
TRAIN_FILE="${TRAIN_FILE:-train.jsonl}"
VAL_FILE="${VAL_FILE:-test.jsonl}"

# ---- Multi-node config ----
NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ---- Training config ----
MODE="${MODE:-bf16}"                    # bf16 | fp8_blockwise2d
EXPERT_BACKEND="${EXPERT_BACKEND:-te_grouped}"  # sequential | te_grouped | sonic
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
MAX_STEPS="${MAX_STEPS:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
EP_SIZE="${EP_SIZE:-8}"
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
DP_SIZE="${DP_SIZE:-${WORLD_SIZE}}"
if [[ "${DP_SIZE}" -ne "${WORLD_SIZE}" ]]; then
    echo "ERROR: DP_SIZE=${DP_SIZE} must equal WORLD_SIZE=${WORLD_SIZE}; dense DP overlaps EP" >&2
    exit 2
fi
if (( WORLD_SIZE % EP_SIZE != 0 )); then
    echo "ERROR: WORLD_SIZE=${WORLD_SIZE} must be divisible by EP_SIZE=${EP_SIZE}" >&2
    exit 2
fi
case "${EXPERT_BACKEND}" in
    sequential|te_grouped|sonic) ;;
    *)
        echo "ERROR: EXPERT_BACKEND must be sequential, te_grouped, or sonic" >&2
        exit 2
        ;;
esac
if [[ "${MODE}" == "fp8_blockwise2d" && "${EXPERT_BACKEND}" == "te_grouped" ]]; then
    echo "ERROR: MODE=fp8_blockwise2d does not cover TE grouped experts; use EXPERT_BACKEND=sequential or sonic" >&2
    exit 2
fi
SHARDING="${SHARDING:-full_shard}"      # full_shard | shard_grad_op
GRAD_CKPT="${GRAD_CKPT:-1}"
FP8_SCALING="${FP8_SCALING:-blockwise2d}"

# ---- Optimization flags ----
AITER_ATTN="${AITER_ATTN:-}"
LUMEN_NORM="${LUMEN_NORM:-}"
FUSE_ROPE="${FUSE_ROPE:-}"
FUSED_ROUTER="${FUSED_ROUTER:-}"
MOE_DISPATCH_OVERLAP="${MOE_DISPATCH_OVERLAP:-}"
MOE_GLOBAL_EXPERT_LAYOUT="${MOE_GLOBAL_EXPERT_LAYOUT:-}"

IMAGE="${IMAGE:-lumen/llama2:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-lumen_qwen3_30b_a3b_dp${DP_SIZE}_ep${EP_SIZE}_node${NODE_RANK}}"

EXTRA_DOCKER_ARGS=()
# RCCL tuning
for _v in NCCL_MIN_P2P_NCHANNELS NCCL_MIN_CTAS NCCL_NCHANNELS_PER_NET_PEER NCCL_NVLS_ENABLE NCCL_DEBUG; do
    [[ -n "${!_v:-}" ]] && EXTRA_DOCKER_ARGS+=( -e "${_v}=${!_v}" )
done

mkdir -p "${HOST_RESULTS}"
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

echo "=== Qwen3-30B-A3B dense-DP=${DP_SIZE}, EP=${EP_SIZE} Training ==="
echo "  Model:       ${HOST_MODEL}"
echo "  Data:        ${HOST_DATA}"
echo "  Mode:        ${MODE}"
echo "  Experts:     ${EXPERT_BACKEND}"
echo "  Sharding:    ${SHARDING} (FSDP2 fully_shard)"
echo "  Steps:       ${MAX_STEPS}"
echo "  SeqLen:      ${SEQ_LENGTH}"
echo "  Training:    full-param ${MODE}"
echo "  Nodes:       ${NNODES} (node_rank=${NODE_RANK})"
echo "  GPUs/node:   ${NPROC_PER_NODE}"
echo "  Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Image:       ${IMAGE}"
echo ""

docker run --rm --init \
    ${EXTRA_DOCKER_ARGS[@]+"${EXTRA_DOCKER_ARGS[@]}"} \
    --name "${CONTAINER_NAME}" \
    --device /dev/dri --device /dev/kfd \
    --group-add video --group-add render \
    --ipc=host --network=host \
    --security-opt=seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    -v "${HOST_LUMEN}/lumen:/workspace/Lumen/lumen" \
    -v "${HOST_LUMEN}/examples:/workspace/Lumen/examples" \
    -v "${HOST_LUMEN}/third_party/aiter:/workspace/Lumen/third_party/aiter" \
    -v "${HOST_MODEL}:/model-qwen3-moe:ro" \
    -v "${HOST_DATA}:/data:ro" \
    -v "${HOST_RESULTS}:/results" \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e TOKENIZERS_PARALLELISM=false \
    -e PYTORCH_TUNABLEOP_ENABLED="${PYTORCH_TUNABLEOP_ENABLED:-0}" \
    "${IMAGE}" \
    bash -c '
set -euo pipefail
cd /workspace/Lumen/examples/qwen3-30b-a3b

EXTRA=""
# Optimization flags
[[ -n "'"${AITER_ATTN}"'" ]]          && EXTRA="${EXTRA} --aiter-attn"
[[ -n "'"${LUMEN_NORM}"'" ]]          && EXTRA="${EXTRA} --lumen-norm"
[[ -n "'"${FUSE_ROPE}"'" ]]           && EXTRA="${EXTRA} --fuse-rope"
[[ -n "'"${FUSED_ROUTER}"'" ]]        && EXTRA="${EXTRA} --fused-router"
[[ -n "'"${MOE_DISPATCH_OVERLAP}"'" ]] && EXTRA="${EXTRA} --lumen-moe-dispatch-overlap"
[[ -n "'"${MOE_GLOBAL_EXPERT_LAYOUT}"'" ]] && EXTRA="${EXTRA} --lumen-moe-global-expert-layout"
[[ "'"${GRAD_CKPT}"'" == "0" ]]       && EXTRA="${EXTRA} --no-gradient-checkpointing"

torchrun \
    --nnodes='"${NNODES}"' \
    --nproc_per_node='"${NPROC_PER_NODE}"' \
    --node_rank='"${NODE_RANK}"' \
    --master_addr='"${MASTER_ADDR}"' \
    --master_port='"${MASTER_PORT}"' \
    pretrain_qwen3_30b_a3b_fsdp.py \
    --model-name-or-path /model-qwen3-moe \
    --data-format alpaca \
    --train-data-path "/data/'"${TRAIN_FILE}"'" \
    --val-data-path "/data/'"${VAL_FILE}"'" \
    --mode '"${MODE}"' \
    --expert-backend '"${EXPERT_BACKEND}"' \
    --ep-size '"${EP_SIZE}"' \
    --dp-size '"${DP_SIZE}"' \
    --sharding '"${SHARDING}"' \
    --fp8-scaling '"${FP8_SCALING}"' \
    --seq-length '"${SEQ_LENGTH}"' \
    --max-steps '"${MAX_STEPS}"' \
    --eval-interval '"${EVAL_INTERVAL}"' \
    --seed 1234 ${EXTRA} 2>&1 \
    | tee "/results/train_'"${MODE}"'_dp'"${DP_SIZE}"'_ep'"${EP_SIZE}"'_node'"${NODE_RANK}"'.log"
'
