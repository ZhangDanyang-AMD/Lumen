#!/bin/bash
# RFT Stage A: Generate → Verify → Build dataset → Train → Export
# Full pipeline from plan.md Stage A (Diversity-Preserving RFT)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMEN_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SFT_MODEL="${SFT_MODEL:-/home/danyzhan/sft-results/Qwen2.5-Coder-SFT-v5f}"
SFT_DATA="${SFT_DATA:-/home/danyzhan/flydsl-agent-dataset/data/format_aligned/train.jsonl}"
RL_SPECS="${RL_SPECS:-/home/danyzhan/flydsl-agent-dataset/data/rl/train-00000-of-00001.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-/home/danyzhan/rft-results}"
N_CANDIDATES="${N_CANDIDATES:-16}"
MAX_SPECS="${MAX_SPECS:-213}"
HARDWARE="${HARDWARE:-gfx950}"
IMAGE="${IMAGE:-lumen/flydsl-cpt:latest}"

mkdir -p "${RESULTS_DIR}"

echo "================================================================"
echo "[$(date)] RFT Stage A Pipeline (Diversity-Preserving)"
echo "  SFT Model:     ${SFT_MODEL}"
echo "  SFT Data:      ${SFT_DATA}"
echo "  RL Specs:      ${RL_SPECS}"
echo "  Hardware:      ${HARDWARE}"
echo "  Max Specs:     ${MAX_SPECS}"
echo "  N Candidates:  ${N_CANDIDATES}"
echo "  Results:       ${RESULTS_DIR}"
echo "================================================================"

# ── Step 1: Generate candidates ──
CAND_FILE="${RESULTS_DIR}/candidates_v5f_${HARDWARE}.jsonl"
echo "[$(date)] Step 1: Generating ${N_CANDIDATES} candidates per spec ..."
docker rm -f lumen_rft_gen 2>/dev/null || true
docker run --rm --init \
    --name lumen_rft_gen \
    --device /dev/dri --device /dev/kfd \
    --group-add video --group-add render \
    --ipc=host --network=host \
    --security-opt=seccomp=unconfined \
    -v "${LUMEN_DIR}/experiments:/workspace/experiments" \
    -v "${SFT_MODEL}:/model:ro" \
    -v "$(dirname "${RL_SPECS}"):/specs:ro" \
    -v "${RESULTS_DIR}:/results" \
    -e TRANSFORMERS_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false \
    -e TORCHDYNAMO_DISABLE=1 -e HSA_ENABLE_SDMA=0 \
    "${IMAGE}" \
    python3 /workspace/experiments/flydsl-agent/rft-stage1/generate_candidates.py \
        --model /model \
        --specs "/specs/$(basename "${RL_SPECS}")" \
        --output "/results/candidates_v5f_${HARDWARE}.jsonl" \
        --n-candidates "${N_CANDIDATES}" \
        --max-specs "${MAX_SPECS}" \
        --hardware "${HARDWARE}" \
        --device cuda:0

echo "[$(date)] Step 1 done. Candidates: ${CAND_FILE}"

# ── Step 2+3: Verify in sandbox + diversity filter ──
VERIFIED_FILE="${RESULTS_DIR}/verified_v5f_${HARDWARE}.jsonl"
STATS_FILE="${RESULTS_DIR}/verify_stats_v5f_${HARDWARE}.json"
echo "[$(date)] Step 2-3: Verifying candidates (static + sandbox + runtime) ..."
python3 "${SCRIPT_DIR}/verify_candidates.py" \
    --input "${CAND_FILE}" \
    --output "${VERIFIED_FILE}" \
    --metadata "${STATS_FILE}" \
    --use-sandbox

echo "[$(date)] Step 2-3 done."
echo "=== Verification Stats ==="
cat "${STATS_FILE}"
echo ""

# ── Step 4: Build RFT dataset ──
RFT_TRAIN="${RESULTS_DIR}/rft_v5f_train.jsonl"
echo "[$(date)] Step 4: Building RFT dataset ..."
python3 "${SCRIPT_DIR}/build_rft_dataset.py" \
    --verified "${VERIFIED_FILE}" \
    --sft-data "${SFT_DATA}" \
    --output "${RFT_TRAIN}" \
    --rft-repeat 2

echo "[$(date)] Step 4 done. Dataset: ${RFT_TRAIN}"

# ── Step 5: RFT Training ──
source "${SCRIPT_DIR}/config_rft.sh"
NUM_SAMPLES=$(wc -l < "${RFT_TRAIN}")
GBS=$((MBS * GRAD_ACCUM * NGPU))
MAX_STEPS=$(( (NUM_SAMPLES + GBS - 1) / GBS ))
WARMUP=$(( MAX_STEPS / 20 ))

echo "[$(date)] Step 5: RFT Training (${MAX_STEPS} steps, lr=${LR}) ..."
docker rm -f lumen_rft_train 2>/dev/null || true

CMD="torchrun --nproc_per_node=${NGPU} train_sft.py"
CMD+=" --backend fsdp --fsdp-version 2"
CMD+=" --model-name-or-path /model"
CMD+=" --train-data-path /rft-data/rft_v5f_train.jsonl"
CMD+=" --val-data-path /data/data/sft/validation-00000-of-00001.jsonl"
CMD+=" --seq-length ${SEQ_LEN}"
CMD+=" --micro-batch-size ${MBS}"
CMD+=" --gradient-accumulation-steps ${GRAD_ACCUM}"
CMD+=" --max-steps ${MAX_STEPS}"
CMD+=" --lr ${LR} --min-lr ${MIN_LR}"
CMD+=" --lr-warmup-steps ${WARMUP}"
CMD+=" --weight-decay ${WEIGHT_DECAY}"
CMD+=" --max-grad-norm ${MAX_GRAD_NORM}"
CMD+=" --log-interval ${LOG_INTERVAL}"
CMD+=" --save-interval ${SAVE_INTERVAL}"
CMD+=" --save-dir /results/v5f_rft_ckpt"
CMD+=" --results-dir /results/v5f_rft_final"
CMD+=" --eval-interval ${EVAL_INTERVAL}"
CMD+=" --num-workers ${NUM_WORKERS}"
CMD+=" --sharding-strategy full_shard"
CMD+=" --lora-rank ${LORA_RANK}"
CMD+=" --lora-alpha ${LORA_ALPHA}"
CMD+=" --lora-dropout ${LORA_DROPOUT}"

docker run --rm --init \
    --name lumen_rft_train \
    --device /dev/dri --device /dev/kfd \
    --group-add video --group-add render \
    --ipc=host --network=host \
    --security-opt=seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    -v "${LUMEN_DIR}/lumen:/workspace/Lumen/lumen" \
    -v "${LUMEN_DIR}/experiments:/workspace/Lumen/experiments" \
    -v "${SFT_MODEL}:/model:ro" \
    -v /home/danyzhan/flydsl-agent-dataset:/data:ro \
    -v "${RESULTS_DIR}:/rft-data:ro" \
    -v "${RESULTS_DIR}:/results" \
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
${CMD} 2>&1 | tee /results/rft_v5f_train.log
"
echo "[$(date)] Step 5 done."

# ── Step 6: Export to HF format ──
echo "[$(date)] Step 6: Exporting RFT model to HF format ..."
docker rm -f lumen_rft_export 2>/dev/null || true
docker run --rm --init \
    --name lumen_rft_export \
    --device /dev/dri --device /dev/kfd \
    --group-add video --group-add render \
    --ipc=host --network=host \
    --security-opt=seccomp=unconfined \
    -v "${LUMEN_DIR}:/workspace/Lumen" \
    -v "${SFT_MODEL}:/model:ro" \
    -v "${RESULTS_DIR}:/results" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    "${IMAGE}" \
    python /workspace/Lumen/experiments/flydsl-agent/cpt/export_hf.py \
        --base-model /model \
        --dcp-path /results/v5f_rft_final/final \
        --output /results/Qwen2.5-Coder-RFT-v5f \
        --lora-rank "${LORA_RANK}" --lora-alpha "${LORA_ALPHA}"

sudo chown -R "$(id -un):$(id -gn)" "${RESULTS_DIR}/Qwen2.5-Coder-RFT-v5f" 2>/dev/null || true
echo "[$(date)] Step 6 done. Model: ${RESULTS_DIR}/Qwen2.5-Coder-RFT-v5f"

# ── Step 7: Benchmark ──
echo "[$(date)] Step 7: Running benchmark (API Score + format + sandbox) ..."
docker rm -f lumen_rft_eval 2>/dev/null || true
docker run --rm --init \
    --name lumen_rft_eval \
    --device /dev/dri --device /dev/kfd \
    --group-add video --group-add render \
    --ipc=host --network=host \
    --security-opt=seccomp=unconfined \
    -v "${LUMEN_DIR}:/workspace/Lumen" \
    -v "${SFT_MODEL}:/sft-model:ro" \
    -v "${RESULTS_DIR}:/results" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e HSA_ENABLE_SDMA=0 -e TORCHDYNAMO_DISABLE=1 \
    "${IMAGE}" \
    python /workspace/Lumen/experiments/flydsl-agent/sft-format-aligned/eval_format_aligned.py \
        --model /results/Qwen2.5-Coder-RFT-v5f \
        --base-model /sft-model \
        --sandbox \
        --output /results/benchmark_rft_v5f.json \
        --device cuda:0

echo "[$(date)] Step 7 done."

echo ""
echo "================================================================"
echo "[$(date)] === RFT Pipeline COMPLETE ==="
echo "  Model:      ${RESULTS_DIR}/Qwen2.5-Coder-RFT-v5f"
echo "  Benchmark:  ${RESULTS_DIR}/benchmark_rft_v5f.json"
echo "  Stats:      ${STATS_FILE}"
echo "================================================================"
