#!/bin/bash
###############################################################################
# Lumen — Llama3.1-8B pretrain (BF16 or FP8 delayed/hybrid)
#
# Usage:
#   PRECISION=bf16 bash run_pretrain_llama31_8b.sh
#   PRECISION=fp8  bash run_pretrain_llama31_8b.sh   # default
#
# All paths are relative to this script; no absolute/user-specific paths.
# Override IMAGE / NPROC / MBS / GBS / SEQ_LEN / TRAIN_STEPS / SEED via env.
###############################################################################
set -euo pipefail

# Repo root = two levels up from examples/llama31/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMEN_DIR="${LUMEN_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

PRECISION="${PRECISION:-fp8}"          # bf16 | fp8
IMAGE="${IMAGE:-lumen:dev}"
NPROC="${NPROC:-8}"
AITER_DIR="${AITER_DIR:-}"
AITER_GFX1250_CONFIG_DIR="${AITER_GFX1250_CONFIG_DIR:-${SCRIPT_DIR}/aiter_configs/gfx1250}"
ATTN_BACKEND="${ATTN_BACKEND:-csrc}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${LUMEN_DIR}/examples/llama31/tokenizer}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
CONTAINER_NAME="${CONTAINER_NAME:-lumen_llama31_8b_${PRECISION}}"

MBS="${MBS:-2}"
GBS="${GBS:-128}"
SEQ_LEN="${SEQ_LEN:-8192}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"
NUM_LAYERS="${NUM_LAYERS:-32}"
SEED="${SEED:-1234}"
FP8_SCALING="${FP8_SCALING:-delayed}"
LUMEN_CROSS_ENTROPY="${LUMEN_CROSS_ENTROPY:-1}"
LUMEN_FUSED_ROPE="${LUMEN_FUSED_ROPE:-1}"
LUMEN_LINEAR="${LUMEN_LINEAR:-1}"
LUMEN_GEMM_BF16_BACKEND="${LUMEN_GEMM_BF16_BACKEND:-flydsl}"

mkdir -p "${RESULTS_DIR}"
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

EXTRA_MOUNTS=()
if [[ -n "${AITER_DIR}" ]]; then
    if [[ ! -f "${AITER_DIR}/setup.py" ]]; then
        echo "ERROR: AITER_DIR does not contain setup.py: ${AITER_DIR}" >&2
        exit 1
    fi
    EXTRA_MOUNTS+=(-v "${AITER_DIR}:/opt/aiter")
fi
SWIGLU_BWD_CONFIG_DIR="${AITER_GFX1250_CONFIG_DIR}/triton/fusions/silu_and_mul_backward"
if [[ -d "${SWIGLU_BWD_CONFIG_DIR}" ]]; then
    EXTRA_MOUNTS+=(
        -v "${SWIGLU_BWD_CONFIG_DIR}:/opt/aiter/aiter/ops/triton/configs/gfx1250/triton/fusions/silu_and_mul_backward:ro"
    )
fi

GPU_GID="$(stat -c '%g' /dev/kfd)"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$(seq -s, 0 $((NPROC - 1)))}"

# ---- FP8-only env switches (validated forward optimizations) ----------------
FP8_ENV=()
if [ "${PRECISION}" = "fp8" ]; then
    FP8_ENV=(
        -e LUMEN_FUSED_QUANT_TRANSPOSE_CPP=1
        -e LUMEN_FUSED_QUANT_AMAX=1
        -e LUMEN_FUSED_QUANT_SCALE=1
        -e LUMEN_FUSED_CAST_TRANSPOSE=1
        -e LUMEN_FUSED_CAST_TRANSPOSE_V2=1
        -e LUMEN_FUSED_SWIGLU_QUANT=1
        -e LUMEN_FUSED_NORM_QUANT=1
        -e LUMEN_FUSED_NORM_QUANT_V2=1
        -e LUMEN_TRANSPOSE_CACHE=1
        -e LUMEN_FAST_QUANT_DISPATCH=1
        -e LUMEN_WEIGHT_QUANT_ONCE=1
        -e LUMEN_SYNC_FP8_CACHE_INVALIDATE=1
        -e LUMEN_FP8_BLOCKSCALE_GFX1250_TUNED=1
    )
fi

# ---- FP8-only training flags -----------------------------------------------
FP8_ARGS=()
if [ "${PRECISION}" = "fp8" ]; then
    FP8_ARGS=(
        --linear-fp8
        --fp8-format hybrid
        --linear-fp8-scaling "${FP8_SCALING}"
        --linear-fp8-amax-algo max
        --linear-fp8-amax-history 1024
    )
fi

# Single-rank gfx1250 uses no distributed communication and serializes kernel
# submission to avoid the ROCm stream-dependency race seen with TP/DP size 1.
if [ "${NPROC}" -eq 1 ]; then
    USE_DISTRIBUTED_OPTIMIZER="${USE_DISTRIBUTED_OPTIMIZER:-0}"
    OVERLAP_GRAD_REDUCE="${OVERLAP_GRAD_REDUCE:-0}"
    if [ "${PRECISION}" = "fp8" ]; then
        AMD_SERIALIZE_KERNEL="${AMD_SERIALIZE_KERNEL:-0}"
        GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-1}"
        CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
    else
        AMD_SERIALIZE_KERNEL="${AMD_SERIALIZE_KERNEL:-3}"
        GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-8}"
        CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
    fi
    LUMEN_SYNC_BEFORE_GRAD_NORM="${LUMEN_SYNC_BEFORE_GRAD_NORM:-1}"
    LUMEN_SYNC_BF16_LINEAR_BWD="${LUMEN_SYNC_BF16_LINEAR_BWD:-1}"
    LUMEN_SYNC_LINEAR_BWD="${LUMEN_SYNC_LINEAR_BWD:-1}"
else
    USE_DISTRIBUTED_OPTIMIZER="${USE_DISTRIBUTED_OPTIMIZER:-1}"
    OVERLAP_GRAD_REDUCE="${OVERLAP_GRAD_REDUCE:-1}"
    AMD_SERIALIZE_KERNEL="${AMD_SERIALIZE_KERNEL:-0}"
    GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-8}"
    CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
    LUMEN_SYNC_BEFORE_GRAD_NORM="${LUMEN_SYNC_BEFORE_GRAD_NORM:-0}"
    LUMEN_SYNC_BF16_LINEAR_BWD="${LUMEN_SYNC_BF16_LINEAR_BWD:-0}"
    LUMEN_SYNC_LINEAR_BWD="${LUMEN_SYNC_LINEAR_BWD:-0}"
fi

DIST_ARGS=()
if [ "${USE_DISTRIBUTED_OPTIMIZER}" = "1" ]; then
    DIST_ARGS+=(--use-distributed-optimizer)
fi
if [ "${OVERLAP_GRAD_REDUCE}" = "1" ]; then
    DIST_ARGS+=(--overlap-grad-reduce)
fi

CE_ARGS=()
if [ "${LUMEN_CROSS_ENTROPY}" = "1" ]; then
    CE_ARGS=(--lumen-cross-entropy)
fi

ROPE_ARGS=()
if [ "${LUMEN_FUSED_ROPE}" = "1" ]; then
    ROPE_ARGS=(--lumen-fused-rope)
fi

LINEAR_ARGS=()
if [ "${LUMEN_LINEAR}" = "1" ]; then
    LINEAR_ARGS=(--lumen-linear)
fi

docker run --rm --init \
    --name "${CONTAINER_NAME}" \
    --device /dev/dri --device /dev/kfd \
    --group-add "${GPU_GID}" \
    --ipc=host --network=host \
    --ulimit nofile=1048576:1048576 \
    --ulimit memlock=-1:-1 \
    --security-opt=seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    --shm-size 16G \
    -v "${LUMEN_DIR}:/workspace/Lumen" \
    -v "${TOKENIZER_DIR}:/tokenizer:ro" \
    -v "${RESULTS_DIR}:/results" \
    "${EXTRA_MOUNTS[@]}" \
    -e GPU_ARCHS=gfx1250 \
    -e ENABLE_CK=1 \
    -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}" \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e PYTHONPATH="/workspace/Lumen/examples/llama31/aiter_configs/python:/workspace/Lumen:/workspace/megatron_lm:/opt/aiter" \
    -e TOKENIZERS_PARALLELISM=false \
    -e HSA_NO_SCRATCH_RECLAIM=1 \
    -e HIP_FORCE_DEV_KERNARG=1 \
    -e GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-8}" \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_SOCKET_IFNAME=lo \
    -e NCCL_DEBUG=WARN \
    -e CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}" \
    -e OMP_NUM_THREADS=1 \
    -e TORCHDYNAMO_DISABLE=1 \
    -e USE_HIPBLASLT=1 \
    -e TORCH_BLAS_PREFER_HIPBLASLT=1 \
    -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    -e LUMEN_PREFER_HIPBLASLT=1 \
    -e LUMEN_GEMM_BF16="${LUMEN_GEMM_BF16:-1}" \
    -e LUMEN_GEMM_BF16_BACKEND="${LUMEN_GEMM_BF16_BACKEND}" \
    -e AITER_CONFIG_GEMM_BF16="/workspace/Lumen/examples/llama31/configs/llama31_8b_bf16_gfx1250_flydsl_tuned.csv" \
    -e LUMEN_FUSE_NORM_LINEAR="${LUMEN_FUSE_NORM_LINEAR:-1}" \
    -e LUMEN_FUSED_SWIGLU="${LUMEN_FUSED_SWIGLU:-1}" \
    -e LUMEN_FUSED_RESIDUAL_NORM="${LUMEN_FUSED_RESIDUAL_NORM:-1}" \
    -e LUMEN_FUSED_RES_BWD="${LUMEN_FUSED_RES_BWD:-1}" \
    -e LUMEN_DISABLE_OPTIMIZER_PATCHES=1 \
    -e LUMEN_SYNC_BEFORE_GRAD_NORM="${LUMEN_SYNC_BEFORE_GRAD_NORM}" \
    -e LUMEN_SYNC_BF16_LINEAR_BWD="${LUMEN_SYNC_BF16_LINEAR_BWD}" \
    -e LUMEN_SYNC_LINEAR_BWD="${LUMEN_SYNC_LINEAR_BWD}" \
    -e LUMEN_SKIP_BACKEND_SYNC="${LUMEN_SKIP_BACKEND_SYNC:-1}" \
    -e AMD_SERIALIZE_KERNEL="${AMD_SERIALIZE_KERNEL}" \
    "${FP8_ENV[@]}" \
    -e NPROC="${NPROC}" -e MBS="${MBS}" -e GBS="${GBS}" -e SEQ_LEN="${SEQ_LEN}" \
    -e NUM_LAYERS="${NUM_LAYERS}" \
    -e ATTN_BACKEND="${ATTN_BACKEND}" \
    -e TRAIN_STEPS="${TRAIN_STEPS}" -e SEED="${SEED}" \
    "${IMAGE}" \
    bash -c '
set -euo pipefail

LUMEN_ROOT="/workspace/Lumen"
PRETRAIN_DIR="${LUMEN_ROOT}/examples/llama31"
PATCH_SCRIPT="${LUMEN_ROOT}/examples/dsv4/patch_megatron_source.py"
DATA_DIR="/results/mock_data"
TRAIN_JSONL="${DATA_DIR}/mock_train.jsonl"
mkdir -p "${DATA_DIR}"

python -c "import megatron" 2>/dev/null || { echo "ERROR: megatron not found in image"; exit 1; }

MEGATRON_ROOT="${MEGATRON_ROOT:-/workspace/megatron_lm}"
PYTHONPATH="${LUMEN_ROOT}" python3 "${PATCH_SCRIPT}" "${MEGATRON_ROOT}" --tag llama

python - <<PYEOF
import os, json, random
seq = int(os.environ["SEQ_LEN"])
gbs = int(os.environ["GBS"])
steps = int(os.environ["TRAIN_STEPS"])
need_chunks = gbs * (steps + 5)
need_tokens = int(need_chunks * (seq + 1) * 1.2)
random.seed(int(os.environ["SEED"]))
path = "${TRAIN_JSONL}"
words_per_doc = 4000
docs = need_tokens // words_per_doc + 1
with open(path, "w") as f:
    for _ in range(docs):
        toks = [str(random.randint(1, 127999)) for _ in range(words_per_doc)]
        f.write(json.dumps({"text": " ".join(toks)}) + "\n")
print(f"[mock-data] wrote {docs} docs to {path}")
PYEOF

cd "${PRETRAIN_DIR}"
set -x
torchrun --nproc_per_node="${NPROC}" --nnodes=1 pretrain_llama31.py \
    --backend megatron \
    --num-layers "${NUM_LAYERS}" \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length "${SEQ_LEN}" \
    --max-position-embeddings "${SEQ_LEN}" \
    --use-rotary-position-embeddings \
    --rotary-base 500000 \
    --no-position-embedding \
    --normalization RMSNorm \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --micro-batch-size "${MBS}" \
    --global-batch-size "${GBS}" \
    --train-iters "${TRAIN_STEPS}" \
    --lr 1.0e-5 --min-lr 0.0 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
    --bf16 \
    --no-gradient-accumulation-fusion \
    '"${DIST_ARGS[*]}"' \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /tokenizer \
    --train-data-path "${TRAIN_JSONL}" \
    --valid-data-path "${TRAIN_JSONL}" \
    --test-data-path "${TRAIN_JSONL}" \
    --split 98,1,1 \
    --seed "${SEED}" \
    --eval-iters 1 \
    --eval-interval "${TRAIN_STEPS}" \
    --save-interval 1000000 \
    --log-interval 1 \
    --lumen-attn-backend "${ATTN_BACKEND}" \
    '"${LINEAR_ARGS[*]}"' \
    '"${CE_ARGS[*]}"' \
    '"${ROPE_ARGS[*]}"' \
    '"${FP8_ARGS[*]}"' \
    2>&1 | tee "/results/lumen_llama31_8b_'"${PRECISION}"'.log"
'

echo ""
echo "[DONE] log: ${RESULTS_DIR}/lumen_llama31_8b_${PRECISION}.log"
