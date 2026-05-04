#!/bin/bash
set -euo pipefail

# MLPerf-aligned Llama2-70B LoRA SFT — 8x MI300X (v47)
#
# All speed and convergence optimizations enabled:
#   - hipBLASLt for all GEMMs (fwd+bwd) with .t() view (no weight transpose copy)
#   - Wgrad .t() view (eliminates grad_fp8.t().contiguous())
#   - Fused quant+amax, quant+scale, norm+quant, cast+transpose
#   - Fused SwiGLU fwd/bwd with fused amax (fused_amax_abs in FP8 path)
#   - Post-eval allocator fixes (recompute, warmup, GC, cache clear)
#   - hipThreadExchangeStreamCaptureMode stub (LD_PRELOAD, ~10k fewer HIP calls/3 steps)
#   - Backend caching + sync elimination
#   - FP8 weight gradients (hipBLASLt)
#   - ACL=21 activation checkpointing
#   - Epoch-level data shuffling
#
# AITER: lumen/triton_kernels branch (cfaeaad3b) from ZhangDanyang-AMD/aiter.git
#   - Mixed-dtype hipBLASLt GEMM (E5M2 grad x E4M3 weight)
#   - Triton fused SwiGLU fwd/bwd kernels
#   - Triton fused cast+transpose+amax kernel
#   - CK FMHA v3 attention (fwd+bwd)
#
# Expected results (v47):
#   Pre-eval step time:  ~4,730 ms  (power/thermal dependent)
#   Post-eval step time: ~5,550 ms  (+17% allocator fragmentation)
#   Memory utilization:  98.7%
#   Best val_loss:       ~0.922     (passes MLPerf target 0.925)
#   Convergence step:    ~576
#
# Local MLPerf reference (same machine, SEED=1234):
#   Step time: 3,967 ms | Speed ratio: 1.19x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/runtime_tunables.sh" 2>/dev/null || true

CONTAINER_NAME="lumen_tp1_dp8"
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

sudo mkdir -p /data1/lumen/results/tp1_fp8
sudo chmod 777 /data1/lumen/results/tp1_fp8

docker run --rm --init \
    --name "$CONTAINER_NAME" \
    --device /dev/dri --device /dev/kfd \
    --ipc=host --network=host \
    --security-opt=seccomp=unconfined \
    -v /data1:/data1 \
    -v /home/danyzhan:/home/danyzhan \
    -v /home/danyzhan/Lumen:/workspace/Lumen \
    -e HSA_ENABLE_SDMA=0 \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_SOCKET_IFNAME=lo \
    -e NCCL_DEBUG=WARN \
    -e NCCL_MIN_P2P_NCHANNELS=32 \
    -e NCCL_MIN_CTAS=32 \
    -e NCCL_NCHANNELS_PER_NET_PEER=32 \
    -e NCCL_NVLS_ENABLE=0 \
    -e TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
    -e TORCHDYNAMO_DISABLE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8 \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e LUMEN_DEBUG_FORWARD=0 \
    -e TORCH_SHOW_CPP_STACKTRACES=1 \
    -e CUBLAS_FORCE_XMMA_KERNEL_INIT=DEVICE \
    -e OMP_NUM_THREADS=1 \
    -e USE_HIPBLASLT=1 \
    -e TORCH_BLAS_PREFER_HIPBLASLT=1 \
    -e LUMEN_PREFER_HIPBLASLT=1 \
    -e LUMEN_TUNED_GEMM=${LUMEN_TUNED_GEMM:-} \
    -e LUMEN_TUNED_GEMM_VALIDATE=${LUMEN_TUNED_GEMM_VALIDATE:-1} \
    -e USE_ROCM_AITER_ROPE_BACKEND=0 \
    -e LUMEN_FUSED_ROPE=1 \
    -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 \
    -e NCCL_TIMEOUT=7200 \
    -e LUMEN_SHUFFLE_TRAIN=1 \
    -e LUMEN_EVAL_ALIGNED=1 \
    -e LUMEN_SKIP_BACKEND_SYNC=1 \
    -e LUMEN_FUSED_MLP=1 \
    -e LUMEN_FUSED_NORM_QUANT=1 \
    -e LUMEN_FUSED_NORM_QUANT_V2=1 \
    -e LUMEN_FUSED_QUANT_SCALE=1 \
    -e LUMEN_FUSED_QUANT_AMAX=1 \
    -e LUMEN_FAST_QUANT_DISPATCH=${LUMEN_FAST_QUANT_DISPATCH:-1} \
    -e LUMEN_FUSED_CAST_TRANSPOSE=1 \
    -e LUMEN_FUSED_CAST_TRANSPOSE_V2=1 \
    -e LUMEN_TRANSPOSE_CACHE=1 \
    -e LUMEN_FUSED_QUANT_TRANSPOSE_CPP=0 \
    -e LUMEN_FUSED_SWIGLU_QUANT=1 \
    -e LUMEN_EVAL_RECOMPUTE=1 \
    -e LUMEN_WARMUP_EVAL_STEPS=2 \
    -e LUMEN_MANUAL_GC=1 \
    -e LUMEN_POST_EVAL_CACHE_CLEAR=1 \
    -e LUMEN_POST_EVAL_REWARM=1 \
    -e LUMEN_POST_EVAL_STRATEGY=gc_only \
    -e LUMEN_FUSED_SWIGLU=1 \
    -e LUMEN_MLP_FP8_STORE=1 \
    -e LUMEN_FUSED_RESIDUAL_NORM=1 \
    -e LUMEN_FUSED_RESIDUAL_KERNEL=1 \
    -e LUMEN_FUSED_LORA=1 \
    -e LUMEN_FUSED_NORM_QUANT_GEMM=1 \
    -e LUMEN_FP8_ATTN_BWD=0 \
    -e LUMEN_HIP_GRAPHS=${LUMEN_HIP_GRAPHS:-0} \
    -e LUMEN_HIP_GRAPHS_MAX_LAYERS=${LUMEN_HIP_GRAPHS_MAX_LAYERS:-3} \
    -e RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-21} \
    -e LUMEN_LOG_INTERVAL=${LUMEN_LOG_INTERVAL:-10} \
    -e LUMEN_MLP_RECOMPUTE=0 \
    -e LUMEN_BATCH_PRECOMPUTE_SCALES=${LUMEN_BATCH_PRECOMPUTE_SCALES:-0} \
    -e TRAIN_STEPS=${TRAIN_STEPS:-1024} \
    -e LUMEN_PROF_START=${LUMEN_PROF_START:-} \
    -e LUMEN_PROF_END=${LUMEN_PROF_END:-} \
    -e LUMEN_PROF_STOP_AFTER=${LUMEN_PROF_STOP_AFTER:-0} \
    -e LUMEN_PROF_STACK=${LUMEN_PROF_STACK:-0} \
    -e LUMEN_PROF_OUTPUT=${LUMEN_PROF_OUTPUT:-/home/danyzhan/profile_summary_current.txt} \
    -e LUMEN_PROF_TRACE=${LUMEN_PROF_TRACE:-} \
    -e LUMEN_COPY_TRACE=${LUMEN_COPY_TRACE:-0} \
    -e LUMEN_TRACE_AMAX=${LUMEN_TRACE_AMAX:-0} \
    -e LUMEN_CPP_QUANT_DISPATCH=${LUMEN_CPP_QUANT_DISPATCH:-0} \
    -e LUMEN_LOG_PATH=${LUMEN_LOG_PATH:-/home/danyzhan/mlperf_llama2_70b.log} \
    -e LD_PRELOAD=${LUMEN_LD_PRELOAD:-/workspace/Lumen/lumen/csrc/hip_no_stream_capture.so} \
    lumen_unit_test:latest \
    bash -c '
set -euo pipefail

MEGATRON_ROOT="/workspace/megatron_lm"
LUMEN_DIR="/workspace/Lumen/examples/llama2"

pip install -q huggingface-hub==0.30.0 pandas pyarrow sentencepiece "transformers>=4.43.0" peft safetensors 2>&1 | tail -1

python -c "import numpy; numpy.product = numpy.prod" 2>/dev/null || true
sed -i "s/np\\.product(/np.prod(/g" "${MEGATRON_ROOT}/megatron/core/dist_checkpointing/exchange_utils.py" 2>/dev/null || true

python "${LUMEN_DIR}/scripts/patch_gpt_layer_specs.py" "${MEGATRON_ROOT}"
python "${LUMEN_DIR}/scripts/patch_checkpointing.py"   "${MEGATRON_ROOT}"
python "${LUMEN_DIR}/scripts/patch_requires_grad.py"    "${MEGATRON_ROOT}"
python "${LUMEN_DIR}/scripts/patch_lora_scaling.py"     "${MEGATRON_ROOT}"
python "${LUMEN_DIR}/scripts/patch_sft_loss_norm.py"    "${MEGATRON_ROOT}"

cd "${LUMEN_DIR}"
CONFIG="${LUMEN_DIR}/config_MI300X_tp1_dp8.sh" bash run_finetune.sh 2>&1 | tee "${LUMEN_LOG_PATH:-/home/danyzhan/mlperf_llama2_70b.log}"
'
