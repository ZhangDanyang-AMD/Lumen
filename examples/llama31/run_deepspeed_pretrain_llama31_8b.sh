#!/usr/bin/env bash
###############################################################################
# Lumen — Llama3.1-8B pretraining with DeepSpeed ZeRO-1/2.
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMEN_DIR="${LUMEN_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
ZERO_STAGE="${ZERO_STAGE:-2}"
PRECISION="${PRECISION:-bf16}"
MBS="${MBS:-1}"
GBS="${GBS:-32}"
SEQ_LEN="${SEQ_LEN:-8192}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"
SEED="${SEED:-1234}"
TARGET_LOSS="${TARGET_LOSS:-3.3}"
STEP_TIME_ATOL="${STEP_TIME_ATOL:-30000}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${SCRIPT_DIR}/tokenizer}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results/deepspeed}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${SCRIPT_DIR}/results/mock_data/deepspeed_llama31_8b_train.jsonl}"

DENOM=$((MBS * NPROC_PER_NODE))
if (( GBS % DENOM != 0 )); then
    echo "GBS=${GBS} must be divisible by MBS*NPROC_PER_NODE=${DENOM}" >&2
    exit 2
fi
GRAD_ACCUM=$((GBS / DENOM))
LOCAL_TRAIN_SAMPLES=$((MBS * GRAD_ACCUM * (TRAIN_STEPS + 2)))

mkdir -p "${RESULTS_DIR}" "$(dirname "${TRAIN_DATA_PATH}")"
if [[ ! -f "${TRAIN_DATA_PATH}" ]]; then
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH}" \
    SEQ_LEN="${SEQ_LEN}" GBS="${GBS}" TRAIN_STEPS="${TRAIN_STEPS}" \
    SEED="${SEED}" NPROC_PER_NODE="${NPROC_PER_NODE}" \
    python3 - <<'PY'
import json
import os
import random

path = os.environ["TRAIN_DATA_PATH"]
seq_len = int(os.environ["SEQ_LEN"])
samples = int(os.environ["GBS"]) * (int(os.environ["TRAIN_STEPS"]) + 2)
random.seed(int(os.environ["SEED"]))
words_per_doc = 8192
estimated_tokens = samples * (seq_len + 1)
documents = max(
    int(os.environ["NPROC_PER_NODE"]),
    estimated_tokens // words_per_doc + 1,
)
with open(path, "w", encoding="utf-8") as output:
    for _ in range(documents):
        words = [str(random.randint(1, 128255)) for _ in range(words_per_doc)]
        output.write(json.dumps({"text": " ".join(words)}) + "\n")
print(f"Wrote {documents} mock documents to {path}")
PY
fi

FP8_ARGS=()
if [[ "${PRECISION}" == "fp8" ]]; then
    FP8_ARGS=(
        --linear-fp8
        --linear-fp8-format fp8_e4m3
        --linear-fp8-scaling dynamic
        --deepspeed-fp8-weight-cache
    )
fi

export PYTHONPATH="${LUMEN_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCHDYNAMO_DISABLE=1
export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
export LUMEN_PREFER_HIPBLASLT=1
export LUMEN_SKIP_BACKEND_SYNC=1

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/pretrain_llama31.py" \
    --backend deepspeed \
    --tokenizer-name-or-path "${TOKENIZER_DIR}" \
    --train-data-path "${TRAIN_DATA_PATH}" \
    --val-data-path "${TRAIN_DATA_PATH}" \
    --train-samples "${LOCAL_TRAIN_SAMPLES}" \
    --val-samples 1 \
    --eval-interval "${TRAIN_STEPS}" \
    --eval-iters 1 \
    --target-loss "${TARGET_LOSS}" \
    --step-time-atol "${STEP_TIME_ATOL}" \
    --hidden-size 4096 \
    --intermediate-size 14336 \
    --num-layers 32 \
    --num-attention-heads 32 \
    --num-key-value-heads 8 \
    --vocab-size 128256 \
    --max-position-embeddings 131072 \
    --rope-theta 500000 \
    --seq-length "${SEQ_LEN}" \
    --micro-batch-size "${MBS}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --max-steps "${TRAIN_STEPS}" \
    --zero-stage "${ZERO_STAGE}" \
    --linear-backend aiter \
    --hf-attn-patch \
    --lumen-fused-rope \
    --lumen-fused-mlp \
    --no-gradient-checkpointing \
    --lr 8e-4 \
    --min-lr 8e-5 \
    --lr-warmup-steps 128 \
    --weight-decay 0.1 \
    --max-grad-norm 1.0 \
    --seed "${SEED}" \
    --log-interval 1 \
    "${FP8_ARGS[@]}" \
    "$@" \
    2>&1 | tee "${RESULTS_DIR}/llama31_8b_zero${ZERO_STAGE}_${PRECISION}.log"
