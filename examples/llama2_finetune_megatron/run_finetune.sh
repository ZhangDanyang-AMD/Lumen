#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# LLaMA2 SFT fine-tuning using Megatron-LM-AMD + Transformer Light attention.
# This script mirrors the ../llama2_finetune example (NeMo-based) but uses
# Megatron-LM-AMD for the training loop and Transformer Light for attention.
#
# Prerequisites:
#   pip install megatron-lm   (or: pip install git+https://github.com/ROCm/Megatron-LM.git)
#   pip install transformer_light
#
# Usage:
#   bash run_finetune.sh
#
# Environment variables (override defaults):
#   MODEL_SIZE       - llama2-7B | llama2-13B | llama2-70B (default: llama2-70B)
#   TP               - tensor parallel size (default: 8)
#   PP               - pipeline parallel size (default: 1)
#   CP               - context parallel size (default: 1)
#   VP               - virtual pipeline stages (default: 0 = disabled)
#   SP               - 1 to enable sequence parallelism (default: 0)
#   MBS              - micro batch size (default: 1)
#   GBS              - global batch size (default: 8)
#   SEQ_LEN          - sequence length (default: 8192)
#   LR               - learning rate (default: 4e-4)
#   MIN_LR           - min learning rate (default: 0)
#   TRAIN_STEPS      - max training steps (default: 800)
#   NGPU             - number of GPUs per node (default: 8)
#   NNODES           - number of nodes (default: 1)
#   CKPT_DIR         - path to Megatron-format checkpoint (default: /ckpt)
#   TRAIN_DATA       - path to training jsonl (default: /data/train.jsonl)
#   VALID_DATA       - path to validation jsonl (default: /data/validation.jsonl)
#   SAVE_DIR         - path to save checkpoints (default: /results/checkpoints)
#   TOKENIZER        - HF tokenizer name or path (default: meta-llama/Llama-2-70b-hf)
#   TL_ATTN_BACKEND  - aiter | triton | triton_fp8 (default: aiter)
#   TL_FP8_QUANT     - fp8_blockwise | mxfp8 (default: fp8_blockwise, for triton_fp8)
#   LORA_RANK        - LoRA rank, 0 = disabled / full finetuning (default: 0)
#   LORA_ALPHA       - LoRA alpha (default: 32)
#   LORA_DROPOUT     - LoRA dropout (default: 0.1)
#   LORA_A2A         - 1 to enable LoRA all-to-all comm optimisation (default: 0)
#   FP8_TRAINING     - 1 to enable FP8 quantised training (default: 0)
#   FP8_FORMAT       - fp8_e4m3 | fp8_e5m2 | mxfp8 (default: fp8_e4m3)
#   FP8_SCALING      - dynamic | delayed | blockwise (default: delayed)
#   FP8_BLOCK_SIZE   - block size for blockwise FP8 (default: 128)
#   WARMUP_STEPS     - synthetic warmup steps before real training (default: 0)
#   VAL_LOSS_TARGET  - early stop when loss EMA < this value (default: unset)
#   MXFP8_BLOCK_M_FWD     - MXFP8 query block fwd (default: 128)
#   MXFP8_BLOCK_N_FWD     - MXFP8 key block fwd (default: 128)
#   MXFP8_BLOCK_M_DQ_BWD  - MXFP8 dQ block bwd (default: 128)
#   MXFP8_BLOCK_N_DQ_BWD  - MXFP8 dQ key block bwd (default: 128)
#   MXFP8_BLOCK_M_DKV_BWD - MXFP8 dKV block bwd (default: 128)
#   MXFP8_BLOCK_N_DKV_BWD - MXFP8 dKV key block bwd (default: 128)
#   MXFP8_QUANT_BLOCK_SIZE - MXFP8 quantisation block (default: 128)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Verify Megatron is installed -------------------------------------------

python -c "import megatron" 2>/dev/null || {
    echo "ERROR: Megatron-LM is not installed."
    echo "  pip install git+https://github.com/ROCm/Megatron-LM.git"
    exit 1
}

# ---- NCCL / runtime / hipBLASLt performance tuning --------------------------

export NCCL_MIN_P2P_NCHANNELS=${NCCL_MIN_P2P_NCHANNELS:-32}
export NCCL_MIN_CTAS=${NCCL_MIN_CTAS:-32}
export NCCL_NCHANNELS_PER_NET_PEER=${NCCL_NCHANNELS_PER_NET_PEER:-32}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}

export USE_HIPBLASLT=${USE_HIPBLASLT:-1}
export TORCH_BLAS_PREFER_HIPBLASLT=${TORCH_BLAS_PREFER_HIPBLASLT:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

export PYTORCH_TUNABLEOP_ENABLED=${PYTORCH_TUNABLEOP_ENABLED:-1}
export PYTORCH_TUNABLEOP_FILENAME=${PYTORCH_TUNABLEOP_FILENAME:-tunableop_results.csv}

# ---- Configurable parameters ------------------------------------------------

MODEL_SIZE=${MODEL_SIZE:-"llama2-70B"}
TP=${TP:-8}
PP=${PP:-1}
CP=${CP:-1}
VP=${VP:-0}
SP=${SP:-0}
MBS=${MBS:-1}
GBS=${GBS:-8}
SEQ_LEN=${SEQ_LEN:-8192}
LR=${LR:-4e-4}
MIN_LR=${MIN_LR:-0}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
GRADIENT_CLIP=${GRADIENT_CLIP:-0.3}
TRAIN_STEPS=${TRAIN_STEPS:-800}
EVAL_INTERVAL=${EVAL_INTERVAL:-50}
SAVE_INTERVAL=${SAVE_INTERVAL:-200}
LOG_INTERVAL=${LOG_INTERVAL:-1}

NGPU=${NGPU:-8}
NNODES=${NNODES:-1}

CKPT_DIR=${CKPT_DIR:-"/ckpt"}
TRAIN_DATA=${TRAIN_DATA:-"/data/train.jsonl"}
VALID_DATA=${VALID_DATA:-"/data/validation.jsonl"}
SAVE_DIR=${SAVE_DIR:-"/results/checkpoints"}
TOKENIZER=${TOKENIZER:-"meta-llama/Llama-2-70b-hf"}

PRECISION=${PRECISION:-"bf16"}

TL_ATTN_BACKEND=${TL_ATTN_BACKEND:-"aiter"}
TL_FP8_QUANT=${TL_FP8_QUANT:-"fp8_blockwise"}

LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.1}
LORA_A2A=${LORA_A2A:-0}

FP8_TRAINING=${FP8_TRAINING:-0}
FP8_FORMAT=${FP8_FORMAT:-"fp8_e4m3"}
FP8_SCALING=${FP8_SCALING:-"delayed"}
FP8_BLOCK_SIZE=${FP8_BLOCK_SIZE:-128}

WARMUP_STEPS=${WARMUP_STEPS:-0}
VAL_LOSS_TARGET=${VAL_LOSS_TARGET:-""}

# Fine-grained MXFP8 attention block sizes (for triton_fp8 + mxfp8)
MXFP8_BLOCK_M_FWD=${MXFP8_BLOCK_M_FWD:-128}
MXFP8_BLOCK_N_FWD=${MXFP8_BLOCK_N_FWD:-128}
MXFP8_BLOCK_M_DQ_BWD=${MXFP8_BLOCK_M_DQ_BWD:-128}
MXFP8_BLOCK_N_DQ_BWD=${MXFP8_BLOCK_N_DQ_BWD:-128}
MXFP8_BLOCK_M_DKV_BWD=${MXFP8_BLOCK_M_DKV_BWD:-128}
MXFP8_BLOCK_N_DKV_BWD=${MXFP8_BLOCK_N_DKV_BWD:-128}
MXFP8_QUANT_BLOCK_SIZE=${MXFP8_QUANT_BLOCK_SIZE:-128}

# ---- Model architecture (LLaMA2 variants) -----------------------------------

case "${MODEL_SIZE}" in
    llama2-7B|llama2-7b)
        NUM_LAYERS=32; HIDDEN=4096; FFN_HIDDEN=11008; HEADS=32; KV_HEADS=32
        ;;
    llama2-13B|llama2-13b)
        NUM_LAYERS=40; HIDDEN=5120; FFN_HIDDEN=13824; HEADS=40; KV_HEADS=40
        ;;
    llama2-70B|llama2-70b)
        NUM_LAYERS=80; HIDDEN=8192; FFN_HIDDEN=28672; HEADS=64; KV_HEADS=8
        ;;
    *)
        echo "ERROR: Unknown MODEL_SIZE=${MODEL_SIZE}"
        echo "Supported: llama2-7B, llama2-13B, llama2-70B"
        exit 1
        ;;
esac

# ---- Build GQA argument (only needed when KV_HEADS != HEADS) ----------------

GQA_ARGS=""
if [ "${KV_HEADS}" -ne "${HEADS}" ]; then
    GQA_ARGS="--group-query-attention --num-query-groups ${KV_HEADS}"
fi

# ---- Launch ------------------------------------------------------------------

DISTRIBUTED_ARGS="
    --nproc_per_node=${NGPU}
    --nnodes=${NNODES}
"

LLAMA2_ARGS="
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN}
    --ffn-hidden-size ${FFN_HIDDEN}
    --num-attention-heads ${HEADS}
    ${GQA_ARGS}
    --seq-length ${SEQ_LEN}
    --max-position-embeddings ${SEQ_LEN}
    --use-rotary-position-embeddings
    --no-position-embedding
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
"

VP_ARGS=""
if [ "${VP}" -gt 0 ]; then
    VP_ARGS="--num-layers-per-virtual-pipeline-stage ${VP}"
fi

SP_ARGS=""
if [ "${SP}" -eq 1 ]; then
    SP_ARGS="--sequence-parallel"
fi

TRAINING_ARGS="
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --context-parallel-size ${CP}
    ${VP_ARGS}
    ${SP_ARGS}
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --train-iters ${TRAIN_STEPS}
    --lr ${LR}
    --min-lr ${MIN_LR}
    --lr-decay-style cosine
    --lr-warmup-fraction 0.0
    --weight-decay ${WEIGHT_DECAY}
    --clip-grad ${GRADIENT_CLIP}
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-8
    --${PRECISION}
    --no-gradient-accumulation-fusion
    --reset-position-ids
    --reset-attention-mask
    --eod-mask-loss
"

DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER}
    --train-data-path ${TRAIN_DATA}
    --valid-data-path ${VALID_DATA}
    --split 100,0,0
"

CHECKPOINT_ARGS="
    --load ${CKPT_DIR}
    --save ${SAVE_DIR}
    --finetune
    --no-load-optim
    --no-load-rng
    --auto-detect-ckpt-format
"

EVAL_ARGS="
    --eval-iters 10
    --eval-interval ${EVAL_INTERVAL}
    --save-interval ${SAVE_INTERVAL}
    --log-interval ${LOG_INTERVAL}
"

TL_ARGS="
    --tl-attn-backend ${TL_ATTN_BACKEND}
    --tl-fp8-quant-type ${TL_FP8_QUANT}
"

LORA_ARGS=""
if [ "${LORA_RANK}" -gt 0 ]; then
    LORA_ARGS="
        --lora-rank ${LORA_RANK}
        --lora-alpha ${LORA_ALPHA}
        --lora-dropout ${LORA_DROPOUT}
    "
    if [ "${LORA_A2A}" -eq 1 ]; then
        LORA_ARGS="${LORA_ARGS} --lora-a2a"
    fi
fi

FP8_ARGS=""
if [ "${FP8_TRAINING}" -eq 1 ]; then
    FP8_ARGS="
        --fp8-training
        --fp8-format ${FP8_FORMAT}
        --fp8-scaling ${FP8_SCALING}
        --fp8-block-size ${FP8_BLOCK_SIZE}
    "
fi

MXFP8_ARGS="
    --mxfp8-block-m-fwd ${MXFP8_BLOCK_M_FWD}
    --mxfp8-block-n-fwd ${MXFP8_BLOCK_N_FWD}
    --mxfp8-block-m-dq-bwd ${MXFP8_BLOCK_M_DQ_BWD}
    --mxfp8-block-n-dq-bwd ${MXFP8_BLOCK_N_DQ_BWD}
    --mxfp8-block-m-dkv-bwd ${MXFP8_BLOCK_M_DKV_BWD}
    --mxfp8-block-n-dkv-bwd ${MXFP8_BLOCK_N_DKV_BWD}
    --mxfp8-quant-block-size ${MXFP8_QUANT_BLOCK_SIZE}
"

WARMUP_ARGS=""
if [ "${WARMUP_STEPS}" -gt 0 ]; then
    WARMUP_ARGS="--warmup-steps ${WARMUP_STEPS}"
fi

EARLY_STOP_ARGS=""
if [ -n "${VAL_LOSS_TARGET}" ]; then
    EARLY_STOP_ARGS="--val-loss-target ${VAL_LOSS_TARGET}"
fi

# ---- Start timing ------------------------------------------------------------

start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING LLAMA2 FINETUNE AT ${start_fmt}"
echo "Model: ${MODEL_SIZE} | TP=${TP} PP=${PP} CP=${CP} VP=${VP} SP=${SP} | GPUs=${NGPU}x${NNODES}"
echo "Transformer Light: backend=${TL_ATTN_BACKEND} fp8_quant=${TL_FP8_QUANT}"
if [ "${LORA_RANK}" -gt 0 ]; then
    echo "LoRA: rank=${LORA_RANK} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT} a2a=${LORA_A2A}"
fi
if [ "${FP8_TRAINING}" -eq 1 ]; then
    echo "FP8 Training: format=${FP8_FORMAT} scaling=${FP8_SCALING} block_size=${FP8_BLOCK_SIZE}"
fi
if [ "${TL_ATTN_BACKEND}" = "triton_fp8" ] && [ "${TL_FP8_QUANT}" = "mxfp8" ]; then
    echo "MXFP8 blocks: fwd=${MXFP8_BLOCK_M_FWD}x${MXFP8_BLOCK_N_FWD} dq_bwd=${MXFP8_BLOCK_M_DQ_BWD}x${MXFP8_BLOCK_N_DQ_BWD} dkv_bwd=${MXFP8_BLOCK_M_DKV_BWD}x${MXFP8_BLOCK_N_DKV_BWD} quant=${MXFP8_QUANT_BLOCK_SIZE}"
fi
if [ "${WARMUP_STEPS}" -gt 0 ]; then
    echo "Warmup: ${WARMUP_STEPS} synthetic steps"
fi
if [ -n "${VAL_LOSS_TARGET}" ]; then
    echo "Early Stop: val_loss_target=${VAL_LOSS_TARGET}"
fi
echo "NCCL: P2P_NCHANNELS=${NCCL_MIN_P2P_NCHANNELS} CTAS=${NCCL_MIN_CTAS} | hipBLASLt=${USE_HIPBLASLT} TunableOp=${PYTORCH_TUNABLEOP_ENABLED}"

torchrun ${DISTRIBUTED_ARGS} \
    "${SCRIPT_DIR}/finetune_llama2.py" \
    ${LLAMA2_ARGS} \
    ${TRAINING_ARGS} \
    ${DATA_ARGS} \
    ${CHECKPOINT_ARGS} \
    ${EVAL_ARGS} \
    ${TL_ARGS} \
    ${MXFP8_ARGS} \
    ${LORA_ARGS} \
    ${FP8_ARGS} \
    ${WARMUP_ARGS} \
    ${EARLY_STOP_ARGS}

ret_code=$?

# ---- End timing --------------------------------------------------------------

end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING LLAMA2 FINETUNE AT ${end_fmt}"
result=$(( end - start ))
echo "RESULT,LLM_FINETUNING,,${result},AMD,${start_fmt}"

exit ${ret_code}
