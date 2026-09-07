#!/usr/bin/env bash
# Qwen3-30B-A3B FSDP2 accuracy run aligned with the Megatron launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=${CONFIG:-"${SCRIPT_DIR}/config_MI350X_1x8x1.sh"}
source "${CONFIG}"
EXPERT_BACKEND=${EXPERT_BACKEND:-te_grouped}
SHARDING=${SHARDING:-full_shard}
LUMEN_NORM=${LUMEN_NORM:-1}
FUSED_ROUTER=${FUSED_ROUTER:-0}
MOE_DISPATCH_OVERLAP=${MOE_DISPATCH_OVERLAP:-0}
MOE_GLOBAL_EXPERT_LAYOUT=${MOE_GLOBAL_EXPERT_LAYOUT:-0}
MODEL_ARGS=()
if [ -n "${MODEL_PATH:-}" ]; then
    MODEL_ARGS=(--model-name-or-path "${MODEL_PATH}")
fi
FEATURE_ARGS=()
[ "${LUMEN_NORM}" = "1" ] && FEATURE_ARGS+=(--lumen-norm)
[ "${FUSED_ROUTER}" = "1" ] && FEATURE_ARGS+=(--fused-router)
[ "${MOE_DISPATCH_OVERLAP}" = "1" ] && FEATURE_ARGS+=(--lumen-moe-dispatch-overlap)
[ "${MOE_GLOBAL_EXPERT_LAYOUT}" = "1" ] && FEATURE_ARGS+=(--lumen-moe-global-expert-layout)

case "${EXPERT_BACKEND}" in
    sequential|te_grouped|sonic) ;;
    *)
        echo "ERROR: EXPERT_BACKEND must be sequential, te_grouped, or sonic" >&2
        exit 2
        ;;
esac

mkdir -p "$(dirname "${DATA_PATH}")" "${RESULTS_DIR}"
if [ ! -s "${DATA_PATH}" ]; then
    python3 - "${DATA_PATH}" "${SEQ_LEN}" "${GBS}" "${TRAIN_STEPS}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
seq_len, global_batch, steps = map(int, sys.argv[2:])
documents = max(128, global_batch * (steps + 4))
text = " ".join(["training"] * max(1024, seq_len))
with path.open("w") as output:
    for index in range(documents):
        output.write(json.dumps({"text": f"{index} {text}"}) + "\n")
PY
fi

FP8_MODE=${FP8_MODE:-${MODE:-bf16}}
case "${FP8_MODE}" in
    bf16)
        TRAIN_MODE=bf16
        ;;
    blockwise2d|fp8_blockwise2d)
        if [ "${EXPERT_BACKEND}" = "te_grouped" ]; then
            echo "ERROR: FP8_MODE=blockwise2d does not cover TE grouped experts; use EXPERT_BACKEND=sequential or sonic" >&2
            exit 2
        fi
        TRAIN_MODE=fp8_blockwise2d
        ;;
    *)
        echo "ERROR: FP8_MODE must be bf16 or blockwise2d" >&2
        exit 2
        ;;
esac

RUN_SUFFIX=${RUN_SUFFIX:-}
if [ "${LUMEN_FP8_EXPERTS_ONLY:-0}" = "1" ] && [[ "${RUN_SUFFIX}" != *experts-only* ]]; then
    RUN_SUFFIX="${RUN_SUFFIX:+${RUN_SUFFIX}-}experts-only"
fi
LOG_FILE="${RESULTS_DIR}/qwen3-30b-a3b-fsdp-${EXPERT_BACKEND}${RUN_SUFFIX:+-${RUN_SUFFIX}}-${TRAIN_MODE}-seq${SEQ_LEN}-mbs${MBS}-gbs${GBS}.log"
echo "FSDP ${TRAIN_MODE}: EXPERT_BACKEND=${EXPERT_BACKEND} LUMEN_FP8_EXPERTS_ONLY=${LUMEN_FP8_EXPERTS_ONLY:-0} log=${LOG_FILE}"
WORLD_SIZE=$((NGPU * NNODES))
DP=${DP:-${WORLD_SIZE}}
if [ $((WORLD_SIZE % EP)) -ne 0 ]; then
    echo "ERROR: WORLD_SIZE=${WORLD_SIZE} must be divisible by EP=${EP}" >&2
    exit 2
fi
if [ "${DP}" -ne "${WORLD_SIZE}" ]; then
    echo "ERROR: DP=${DP} must equal WORLD_SIZE=${WORLD_SIZE}; dense DP overlaps EP" >&2
    exit 2
fi
GRAD_ACCUM=$((GBS / (MBS * DP)))
if [ $((GBS % (MBS * DP))) -ne 0 ]; then
    echo "ERROR: GBS=${GBS} must be divisible by MBS*DP=$((MBS * DP))" >&2
    exit 2
fi

torchrun \
    --nproc_per_node="${NGPU}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/pretrain_qwen3_30b_a3b_fsdp.py" \
    "${MODEL_ARGS[@]}" \
    "${FEATURE_ARGS[@]}" \
    --tokenizer-name-or-path "${TOKENIZER_PATH}" \
    --train-data-path "${DATA_PATH}" \
    --val-data-path "${DATA_PATH}" \
    --data-format pretrain \
    --mode "${TRAIN_MODE}" \
    --seq-length "${SEQ_LEN}" \
    --max-position-embeddings 4096 \
    --micro-batch-size "${MBS}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --max-steps "${TRAIN_STEPS}" \
    --lr "${LR}" \
    --min-lr "${MIN_LR}" \
    --lr-decay-style cosine \
    --lr-warmup-steps "${LR_WARMUP_ITERS}" \
    --weight-decay 0.1 \
    --max-grad-norm 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --aux-loss-coeff "${AUX_LOSS_COEFF:-1e-3}" \
    --ep-size "${EP}" \
    --dp-size "${DP}" \
    --expert-backend "${EXPERT_BACKEND}" \
    --sharding "${SHARDING}" \
    --no-gradient-checkpointing \
    --num-workers 2 \
    --shuffle-data \
    --data-seed 0 \
    --log-interval 1 \
    --eval-interval 0 \
    --seed 1234 \
    2>&1 | tee "${LOG_FILE}"

echo "Log: ${LOG_FILE}"
