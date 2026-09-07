#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME=${IMAGE_NAME:-zhangdanyangamd/lumen:qwen3-30b-a3b-350x-pretrain260829-multistream}
COMMAND=${COMMAND:-"bash run_qwen3_30b_a3b_megatron.sh"}
HOST_ASSET_ROOT=${HOST_ASSET_ROOT:-/dev/shm/qwen3-30b-a3b}

NCCL_ENV_ARGS=()
for _v in \
    NCCL_MIN_P2P_NCHANNELS NCCL_MIN_CTAS NCCL_NCHANNELS_PER_NET_PEER \
    NCCL_NVLS_ENABLE NCCL_DEBUG NCCL_PROTO NCCL_ALGO \
    NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_P2P_NET_CHUNKSIZE \
    NCCL_BUFFSIZE NCCL_IB_DISABLE NCCL_SOCKET_IFNAME \
    RCCL_MSCCL_ENABLE TORCH_NCCL_AVOID_RECORD_STREAMS
do
    if [ -n "${!_v:-}" ]; then
        NCCL_ENV_ARGS+=(--env "${_v}=${!_v}")
    fi
done

docker run --rm --init \
    "${NCCL_ENV_ARGS[@]}" \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --ipc=host \
    --network=host \
    --security-opt=seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    --shm-size=64G \
    --volume "${REPO_ROOT}/lumen/models/megatron.py:/workspace/Lumen/lumen/models/megatron.py" \
    --volume "${REPO_ROOT}/lumen/models/fsdp.py:/workspace/Lumen/lumen/models/fsdp.py" \
    --volume "${REPO_ROOT}/lumen/models/llama31:/workspace/Lumen/lumen/models/llama31" \
    --volume "${REPO_ROOT}/lumen/models/qwen3_30b_a3b/fsdp:/workspace/Lumen/lumen/models/qwen3_30b_a3b/fsdp" \
    --volume "${REPO_ROOT}/lumen/config.py:/workspace/Lumen/lumen/config.py" \
    --volume "${REPO_ROOT}/lumen/modules/sonic_moe.py:/workspace/Lumen/lumen/modules/sonic_moe.py" \
    --volume "${REPO_ROOT}/lumen/quantize:/workspace/Lumen/lumen/quantize" \
    --volume "${REPO_ROOT}/lumen/ops/quantize:/workspace/Lumen/lumen/ops/quantize" \
    --volume "${REPO_ROOT}/lumen/ops/moe/__init__.py:/workspace/Lumen/lumen/ops/moe/__init__.py" \
    --volume "${REPO_ROOT}/lumen/ops/moe/dispatch_layout.py:/workspace/Lumen/lumen/ops/moe/dispatch_layout.py" \
    --volume "${REPO_ROOT}/lumen/ops/moe/dispatch_overlap.py:/workspace/Lumen/lumen/ops/moe/dispatch_overlap.py" \
    --volume "${REPO_ROOT}/lumen/ops/moe/fused_router.py:/workspace/Lumen/lumen/ops/moe/fused_router.py" \
    --volume "${REPO_ROOT}/lumen/ops/moe/fused_routing.py:/workspace/Lumen/lumen/ops/moe/fused_routing.py" \
    --volume "${REPO_ROOT}/tests:/workspace/Lumen/tests" \
    --volume "${REPO_ROOT}/third_party/aiter:/workspace/Lumen/third_party/aiter" \
    --volume "${REPO_ROOT}/examples:/workspace/Lumen/examples" \
    --volume "${HOST_ASSET_ROOT}:/nobackup" \
    --volume lumen-qwen3-triton-cache:/root/.triton \
    --workdir /workspace/Lumen/examples/qwen3-30b-a3b \
    --env PYTHONPATH=/workspace/Lumen:/workspace/Lumen/third_party/aiter:/workspace/Megatron-LM \
    --env MOE_IMPL="${MOE_IMPL:-sequential}" \
    --env TRAIN_STEPS="${TRAIN_STEPS:-20}" \
    --env SEQ_LEN="${SEQ_LEN:-4096}" \
    --env MBS="${MBS:-1}" \
    --env GBS="${GBS:-8}" \
    --env AUX_LOSS_COEFF="${AUX_LOSS_COEFF:-1e-3}" \
    --env LR="${LR:-1e-5}" \
    --env MIN_LR="${MIN_LR:-0}" \
    --env MODEL_PATH="${MODEL_PATH:-}" \
    --env DATA_PATH="${DATA_PATH:-}" \
    --env MEGATRON_LOAD_PATH="${MEGATRON_LOAD_PATH:-}" \
    --env LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-2}" \
    --env MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" \
    --env MASTER_PORT="${MASTER_PORT:-29500}" \
    --env RUN_SUFFIX="${RUN_SUFFIX:-}" \
    --env EXPERT_BACKEND="${EXPERT_BACKEND:-te_grouped}" \
    --env LUMEN_NORM="${LUMEN_NORM:-1}" \
    --env FUSED_ROUTER="${FUSED_ROUTER:-0}" \
    --env MOE_DISPATCH_OVERLAP="${MOE_DISPATCH_OVERLAP:-0}" \
    --env MOE_GLOBAL_EXPERT_LAYOUT="${MOE_GLOBAL_EXPERT_LAYOUT:-0}" \
    --env PYTORCH_TUNABLEOP_ENABLED="${PYTORCH_TUNABLEOP_ENABLED:-0}" \
    --env SONIC_MOE_GEMM_BACKEND="${SONIC_MOE_GEMM_BACKEND:-triton}" \
    --env SONIC_MOE_GROUPED_GEMM_BACKEND="${SONIC_MOE_GROUPED_GEMM_BACKEND:-triton}" \
    --env SONIC_MOE_USE_QWEN3_TUNED_GEMM="${SONIC_MOE_USE_QWEN3_TUNED_GEMM:-1}" \
    --env SONIC_MOE_MULTISTREAM_PRIORITY="${SONIC_MOE_MULTISTREAM_PRIORITY:-0}" \
    --env SONIC_MOE_LOG_BACKEND="${SONIC_MOE_LOG_BACKEND:-0}" \
    --env SONIC_MOE_TRACE_GEMM="${SONIC_MOE_TRACE_GEMM:-0}" \
    --env SONIC_MOE_TRACE_ALL_RANKS="${SONIC_MOE_TRACE_ALL_RANKS:-0}" \
    --env NVTE_USE_CUTLASS_GROUPED_GEMM="${NVTE_USE_CUTLASS_GROUPED_GEMM:-0}" \
    --env NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK="${NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK:-0}" \
    --env NVTE_USE_HIPBLASLT="${NVTE_USE_HIPBLASLT:-}" \
    --env NVTE_USE_HIPKITTENS_GEMM="${NVTE_USE_HIPKITTENS_GEMM:-}" \
    --env NVTE_USE_HIPKITTENS_GROUPED_GEMM="${NVTE_USE_HIPKITTENS_GROUPED_GEMM:-}" \
    --env LUMEN_USE_MEGATRON_ATTENTION="${LUMEN_USE_MEGATRON_ATTENTION:-0}" \
    --env LUMEN_ATTN_BACKEND="${LUMEN_ATTN_BACKEND:-csrc}" \
    --env FP8_MODE="${FP8_MODE:-bf16}" \
    --env USE_ROCM_AITER_ROPE_BACKEND="${USE_ROCM_AITER_ROPE_BACKEND:-1}" \
    --env MEGATRON_OVERLAP="${MEGATRON_OVERLAP:-1}" \
    --env OVERLAP_MOE_EP_COMM="${OVERLAP_MOE_EP_COMM:-1}" \
    --env CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}" \
    --env MOE_PAD_TO_CAPACITY="${MOE_PAD_TO_CAPACITY:-0}" \
    --env MOE_EXPERT_CAPACITY_FACTOR="${MOE_EXPERT_CAPACITY_FACTOR:-1.0}" \
    --env GRAD_ACC_FUSION="${GRAD_ACC_FUSION:-0}" \
    --env CUDA_GRAPH_IMPL="${CUDA_GRAPH_IMPL:-transformer_engine}" \
    --env CUDA_GRAPH_SCOPE="${CUDA_GRAPH_SCOPE:-attn}" \
    --env CUDA_GRAPH_WARMUP_STEPS="${CUDA_GRAPH_WARMUP_STEPS:-}" \
    --env QWEN_PARITY_DUMP_DIR="${QWEN_PARITY_DUMP_DIR:-}" \
    --env QWEN_PARITY_LOG_LOCAL_LOSS="${QWEN_PARITY_LOG_LOCAL_LOSS:-0}" \
    --env LUMEN_PROFILE_OUTPUT="${LUMEN_PROFILE_OUTPUT:-}" \
    --env LUMEN_PROFILE_STEP="${LUMEN_PROFILE_STEP:-1}" \
    --env TE_HIPBLASLT_ALGO_LOAD="${TE_HIPBLASLT_ALGO_LOAD:-}" \
    --env TE_HIPBLASLT_ALGO_SAVE="${TE_HIPBLASLT_ALGO_SAVE:-}" \
    --env TE_HIPBLASLT_TUNING_RUN_COUNT="${TE_HIPBLASLT_TUNING_RUN_COUNT:-}" \
    --env TE_HIPBLASLT_TUNING_ALGO_COUNT="${TE_HIPBLASLT_TUNING_ALGO_COUNT:-}" \
    "${IMAGE_NAME}" \
    bash -lc "${COMMAND}"
