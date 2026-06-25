#!/bin/bash
# RFT: Generate → Verify → Build dataset → Train
# Full pipeline from plan.md Stage A
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMEN_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SFT_MODEL="${SFT_MODEL:-/home/danyzhan/sft-results/Qwen2.5-Coder-SFT-v2}"
RL_SPECS="${RL_SPECS:-/home/danyzhan/flydsl-agent-dataset/data/rl/train-00000-of-00001.jsonl}"
SFT_DATA="${SFT_DATA:-/home/danyzhan/flydsl-agent-dataset/data/sft/train-00000-of-00001.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-/home/danyzhan/rft-results}"
DATASET_DIR="${DATASET_DIR:-/home/danyzhan/flydsl-agent-dataset/data/rft}"
N_CANDIDATES="${N_CANDIDATES:-16}"
MAX_SPECS="${MAX_SPECS:-200}"

mkdir -p "${RESULTS_DIR}" "${DATASET_DIR}"

echo "================================================================"
echo " FlyDSL Agent — RFT Pipeline"
echo "  SFT Model:     ${SFT_MODEL}"
echo "  RL Specs:       ${RL_SPECS}"
echo "  Max Specs:      ${MAX_SPECS}"
echo "  N Candidates:   ${N_CANDIDATES}"
echo "  Results:        ${RESULTS_DIR}"
echo "================================================================"

# Step 1: Generate candidates (runs on GPU with SFT model)
echo "[Step 1/4] Generating ${N_CANDIDATES} candidates per spec ..."
docker run --rm --init \
    --name lumen_rft_gen \
    --device /dev/dri --device /dev/kfd \
    --group-add video --group-add render \
    --ipc=host --network=host \
    --security-opt=seccomp=unconfined \
    -v "${LUMEN_DIR}/experiments:/workspace/experiments" \
    -v "${SFT_MODEL}:/model:ro" \
    -v "$(dirname ${RL_SPECS}):/specs:ro" \
    -v "${RESULTS_DIR}:/results" \
    -e TRANSFORMERS_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false \
    -e TORCHDYNAMO_DISABLE=1 -e HSA_ENABLE_SDMA=0 \
    lumen/flydsl-cpt:latest \
    python3 /workspace/experiments/flydsl-agent/rft/generate_candidates.py \
        --model /model \
        --specs "/specs/$(basename ${RL_SPECS})" \
        --output /results/candidates.jsonl \
        --n-candidates "${N_CANDIDATES}" \
        --max-specs "${MAX_SPECS}" \
        --device cuda:0

echo "[Step 1/4] Done. Candidates: ${RESULTS_DIR}/candidates.jsonl"

# Step 2+3: Verify in sandbox + diversity filter
echo "[Step 2-3/4] Verifying candidates (static analysis + sandbox) ..."
python3 "${SCRIPT_DIR}/verify_candidates.py" \
    --input "${RESULTS_DIR}/candidates.jsonl" \
    --output "${RESULTS_DIR}/verified.jsonl" \
    --metadata "${RESULTS_DIR}/verify_stats.json" \
    --use-sandbox

echo "[Step 2-3/4] Done. Verified: ${RESULTS_DIR}/verified.jsonl"

# Step 4: Build RFT dataset
echo "[Step 4/4] Building RFT training dataset ..."
python3 "${SCRIPT_DIR}/build_rft_dataset.py" \
    --verified "${RESULTS_DIR}/verified.jsonl" \
    --output "${DATASET_DIR}/train-00000-of-00001.jsonl"

echo "[Step 4/4] Done. RFT dataset: ${DATASET_DIR}/train-00000-of-00001.jsonl"

# Summary
echo ""
echo "================================================================"
echo " RFT Pipeline Complete"
echo "  Candidates: ${RESULTS_DIR}/candidates.jsonl"
echo "  Verified:   ${RESULTS_DIR}/verified.jsonl"
echo "  Stats:      ${RESULTS_DIR}/verify_stats.json"
echo "  Dataset:    ${DATASET_DIR}/train-00000-of-00001.jsonl"
echo ""
echo " Next: Train RFT model with:"
echo "   TRAIN_DATA=/data/data/rft/train-00000-of-00001.jsonl \\"
echo "   MAX_STEPS=<auto> LR=5e-6 LORA_RANK=64 \\"
echo "   bash experiments/flydsl-agent/sft/run_sft.sh"
echo "================================================================"
