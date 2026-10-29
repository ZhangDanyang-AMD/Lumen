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
FP8_FORMAT="${FP8_FORMAT:-fp8_e4m3}"
FP8_SCALING="${FP8_SCALING:-dynamic}"
FP8_AMAX_ALGO="${FP8_AMAX_ALGO:-max}"
FP8_AMAX_HISTORY="${FP8_AMAX_HISTORY:-1024}"
FP8_WEIGHT_CACHE="${FP8_WEIGHT_CACHE:-1}"
FP8_FUSED_KERNELS="${FP8_FUSED_KERNELS:-}"
BLOCKWISE_TUNING_CONFIG="${BLOCKWISE_TUNING_CONFIG:-${SCRIPT_DIR}/configs/deepspeed_llama31_8b_blockwise2d_triton_gfx950.json}"
HF_ATTN_PATCH="${HF_ATTN_PATCH:-0}"
FUSED_ROPE="${FUSED_ROPE:-0}"
FUSED_MLP="${FUSED_MLP:-}"
LUMEN_LINEAR="${LUMEN_LINEAR:-0}"
LUMEN_NORM="${LUMEN_NORM:-0}"
SHARED_INPUT_QUANT="${SHARED_INPUT_QUANT:-1}"
MBS="${MBS:-2}"
GBS="${GBS:-128}"
SEQ_LEN="${SEQ_LEN:-8192}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"
SEED="${SEED:-1234}"
TARGET_LOSS="${TARGET_LOSS:-}"
STEP_TIME_ATOL="${STEP_TIME_ATOL:-30000}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${SCRIPT_DIR}/tokenizer}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results/deepspeed}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${SCRIPT_DIR}/results/mock_data/deepspeed_llama31_8b_train.jsonl}"
LOG_PATH="${LOG_PATH:-${RESULTS_DIR}/llama31_8b_zero${ZERO_STAGE}_${PRECISION}.log}"

DENOM=$((MBS * NPROC_PER_NODE))
if (( GBS % DENOM != 0 )); then
    echo "GBS=${GBS} must be divisible by MBS*NPROC_PER_NODE=${DENOM}" >&2
    exit 2
fi
GRAD_ACCUM=$((GBS / DENOM))
LOCAL_TRAIN_SAMPLES=$((MBS * GRAD_ACCUM * (TRAIN_STEPS + 2)))

if [[ "${ZERO_STAGE}" != "1" && "${ZERO_STAGE}" != "2" && "${ZERO_STAGE}" != "3" ]]; then
    echo "Lumen DeepSpeed supports ZERO_STAGE=1, 2, or 3" >&2
    exit 2
fi
if [[ "${PRECISION}" != "bf16" && "${PRECISION}" != "fp8" ]]; then
    echo "PRECISION must be bf16 or fp8" >&2
    exit 2
fi
if [[ "${ZERO_STAGE}" == "3" && "${PRECISION}" == "fp8" ]]; then
    echo "ZeRO-3 is currently validated for BF16 only" >&2
    exit 2
fi
if [[ -z "${TARGET_LOSS}" ]]; then
    # zero.Init changes the deterministic synthetic-corpus initialization.
    [[ "${ZERO_STAGE}" == "3" ]] && TARGET_LOSS=6.3 || TARGET_LOSS=5.8
fi
if [[ -z "${FUSED_MLP}" ]]; then
    [[ "${PRECISION}" == "fp8" ]] && FUSED_MLP=1 || FUSED_MLP=0
fi
if [[ -z "${FP8_FUSED_KERNELS}" ]]; then
    FP8_FUSED_KERNELS=1
fi

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
samples = int(os.environ["GBS"]) * (int(os.environ["TRAIN_STEPS"]) + 5)
random.seed(int(os.environ["SEED"]))
words_per_doc = 4000
estimated_tokens = int(samples * (seq_len + 1) * 1.2)
documents = max(
    int(os.environ["NPROC_PER_NODE"]),
    estimated_tokens // words_per_doc + 1,
)
with open(path, "w", encoding="utf-8") as output:
    for _ in range(documents):
        words = [str(random.randint(1, 127999)) for _ in range(words_per_doc)]
        output.write(json.dumps({"text": " ".join(words)}) + "\n")
print(f"Wrote {documents} mock documents to {path}")
PY
fi

FP8_ARGS=()
if [[ "${PRECISION}" == "fp8" ]]; then
    FP8_ARGS=(
        --linear-fp8
        --linear-fp8-format "${FP8_FORMAT}"
        --linear-fp8-scaling "${FP8_SCALING}"
        --linear-fp8-block-size 128
        --linear-fp8-amax-algo "${FP8_AMAX_ALGO}"
        --linear-fp8-amax-history "${FP8_AMAX_HISTORY}"
    )
    if [[ "${FP8_WEIGHT_CACHE}" == "1" ]]; then
        FP8_ARGS+=(--deepspeed-fp8-weight-cache)
    fi
    if [[ "${FP8_SCALING}" == "blockwise2d" ]]; then
        FP8_ARGS+=(
            --deepspeed-blockwise-tuning-config
            "${BLOCKWISE_TUNING_CONFIG}"
        )
    fi
fi

PERF_ARGS=()
if [[ "${HF_ATTN_PATCH}" == "1" ]]; then
    PERF_ARGS+=(--hf-attn-patch)
else
    PERF_ARGS+=(--no-hf-attn-patch)
fi
if [[ "${FUSED_ROPE}" == "1" ]]; then
    PERF_ARGS+=(--lumen-fused-rope)
else
    PERF_ARGS+=(--no-lumen-fused-rope)
fi
if [[ "${FUSED_MLP}" == "1" ]]; then
    PERF_ARGS+=(--lumen-fused-mlp)
else
    PERF_ARGS+=(--no-lumen-fused-mlp)
fi
if [[ "${LUMEN_LINEAR}" == "1" && "${PRECISION}" == "bf16" ]]; then
    PERF_ARGS+=(--lumen-linear)
fi
if [[ "${LUMEN_NORM}" == "1" ]]; then
    PERF_ARGS+=(--lumen-norm)
else
    PERF_ARGS+=(--no-lumen-norm)
fi

export PYTHONPATH="${LUMEN_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export HSA_NO_SCRATCH_RECLAIM=1
export HIP_FORCE_DEV_KERNARG=1
export GPU_MAX_HW_QUEUES=8
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_DEVICE_MAX_CONNECTIONS=8
export TORCHDYNAMO_DISABLE=1
export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
export LUMEN_PREFER_HIPBLASLT=1
export LUMEN_SKIP_BACKEND_SYNC=1
if [[ "${PRECISION}" == "fp8" ]]; then
    python3 - <<'PY'
import torch
from aiter.ops.enum import QuantType
from aiter.ops.quant import get_hip_quant

torch.cuda.set_device(0)
x = torch.zeros(128, 128, device="cuda", dtype=torch.bfloat16)
get_hip_quant(QuantType.per_1x128)(
    x, quant_dtype=torch.float8_e4m3fn
)
torch.cuda.synchronize()
PY
    export LUMEN_FUSED_QUANT_TRANSPOSE_CPP=1
    export LUMEN_FUSED_QUANT_SCALE=1
    export LUMEN_FUSED_CAST_TRANSPOSE=1
    export LUMEN_FUSED_CAST_TRANSPOSE_V2=1
    export LUMEN_TRANSPOSE_CACHE=1
    export LUMEN_FAST_QUANT_DISPATCH=1
    if [[ "${SHARED_INPUT_QUANT}" == "1" ]]; then
        export LUMEN_SHARED_INPUT_QUANT=1
    fi
    if [[ "${FP8_FUSED_KERNELS}" == "1" ]]; then
        export LUMEN_FUSED_QUANT_AMAX=1
    fi
    if [[ "${FP8_SCALING}" == "delayed" ]]; then
        export LUMEN_STEP_BATCH_SCALES=1
    fi
    if [[ "${FUSED_MLP}" == "1" ]]; then
        export LUMEN_FUSED_SWIGLU_QUANT=1
        export LUMEN_HF_FUSED_SWIGLU_QUANT=1
    fi
fi

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
    --performance-warmup-steps 2 \
    --hidden-size 4096 \
    --intermediate-size 14336 \
    --num-layers 32 \
    --num-attention-heads 32 \
    --num-key-value-heads 8 \
    --vocab-size 128256 \
    --max-position-embeddings "${SEQ_LEN}" \
    --rope-theta 500000 \
    --seq-length "${SEQ_LEN}" \
    --micro-batch-size "${MBS}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --max-steps "${TRAIN_STEPS}" \
    --zero-stage "${ZERO_STAGE}" \
    --linear-backend aiter \
    --no-gradient-checkpointing \
    --lr 1.0e-5 \
    --min-lr 0.0 \
    --lr-warmup-steps 2 \
    --weight-decay 0.1 \
    --max-grad-norm 1.0 \
    --seed "${SEED}" \
    --log-interval 1 \
    "${FP8_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "$@" \
    2>&1 | tee "${LOG_PATH}"
