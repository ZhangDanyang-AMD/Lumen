#!/bin/bash
set -euo pipefail

CONTAINER_NAME="lumen_tp1_dp8"
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

sudo mkdir -p /data1/lumen/results/tp1_fp8
sudo chmod 777 /data1/lumen/results/tp1_fp8
sudo mkdir -p /data1/lumen/results/tp1_fp8/saved_ckpts
sudo chmod 777 /data1/lumen/results/tp1_fp8/saved_ckpts

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
    -e TORCHDYNAMO_DISABLE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e USE_ROCM_AITER_ROPE_BACKEND=0 \
    -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 \
    -e NCCL_TIMEOUT=7200 \
    -e LUMEN_MLP_RECOMPUTE=1 \
    lumen_unit_test:latest \
    bash -c '
set -euo pipefail

MEGATRON_ROOT="/workspace/megatron_lm"
LUMEN_DIR="/workspace/Lumen/examples/llama2"

pip install -q huggingface-hub==0.30.0 pandas pyarrow sentencepiece "transformers>=4.43.0" peft safetensors 2>&1 | tail -1

python -c "import numpy; numpy.product = numpy.prod" 2>/dev/null || true
sed -i "s/np\.product(/np.prod(/g" "${MEGATRON_ROOT}/megatron/core/dist_checkpointing/exchange_utils.py" 2>/dev/null || true

python "${LUMEN_DIR}/scripts/patch_gpt_layer_specs.py" "${MEGATRON_ROOT}"
python "${LUMEN_DIR}/scripts/patch_checkpointing.py" "${MEGATRON_ROOT}"
python "${LUMEN_DIR}/scripts/patch_requires_grad.py" "${MEGATRON_ROOT}"
python "${LUMEN_DIR}/scripts/patch_swiglu_fp8_dtype.py" "${MEGATRON_ROOT}"
python "${LUMEN_DIR}/scripts/patch_swiglu_chunked_bwd.py" "${MEGATRON_ROOT}"
python "${LUMEN_DIR}/scripts/patch_fused_mlp.py" "${MEGATRON_ROOT}"

cd "${LUMEN_DIR}"

echo "============================================================"
echo "TP=1 DP=8 FP8 HYBRID + LoRA(attention-only), 1024 steps (v9)"
echo "  FULLY ALIGNED WITH MLPerf MI300X REFERENCE:"
echo "  - LoRA: attention-only (matches MLPerf)"
echo "  - FP8 format: hybrid E4M3/E5M2 (matches MLPerf)"
echo "  - Optimizer: fused_adam (matches MLPerf)"
echo "  - Vocab padding: 128 (matches MLPerf)"
echo "  - LR warmup: 0 (matches MLPerf warmup_ratio=0.0)"
echo "  - ACL=21 (matches MLPerf exactly!)"
echo "  - FIX: deterministic=False → CK v3 tiled bwd (no 16GB alloc)"
echo "  - FIX: FP8 GEMM for mixed-dtype bwd (no BF16 weight dequant)"
echo "  - FIX: MLP-only recompute (free intermediate BF16 tensors)"
echo "  - activation_func_fp8_input_store=True (SwiGLU saves in FP8)"
echo "  - FP8-aware activation checkpointing enabled"
echo "  - eval_iters=22, lr=4e-4, weight_decay=1e-4, clip=0.3"
echo "============================================================"

CONFIG="${LUMEN_DIR}/config_MI300X_tp1_dp8.sh" bash run_finetune.sh 2>&1 | tee /home/danyzhan/tp1_dp8_v13.log
'
