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

cd "${LUMEN_DIR}"

echo "============================================================"
echo "Phase 1: FP8 Diagnostic — verify quantization quality"
echo "============================================================"
python "${LUMEN_DIR}/scripts/verify_fp8_quality.py" --ckpt-dir /data1/lumen/megatron_ckpt_nous_tp1 2>&1 | tee /home/danyzhan/fp8_diagnostic.log || echo "[WARN] Diagnostic script failed; continuing with training..."

echo "============================================================"
echo "TP=1 DP=8 FP8+LoRA+FP8ParamStorage, 1024 steps"
echo "seed=1234, MLPerf params (wd/beta2/clip) + FP8-stable lr=2e-4"
echo "  lr=2e-4, min_lr=0, lr_warmup=100, clip_grad=0.3"
echo "  weight_decay=1e-4, beta2=0.999"
echo "  warmup=5 synthetic steps (zero loss_mask)"
echo "============================================================"

CONFIG="${LUMEN_DIR}/config_MI300X_tp1_dp8.sh" bash run_finetune.sh 2>&1 | tee /home/danyzhan/tp1_dp8_v3.log
'
