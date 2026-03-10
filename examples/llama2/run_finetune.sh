#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# LLaMA2 SFT fine-tuning — unified launcher.
#
# Supports two backends via the BACKEND environment variable:
#   BACKEND=megatron  — Megatron-LM-AMD (TP/PP/CP/VP/SP)  [default]
#   BACKEND=fsdp      — PyTorch FSDP + HuggingFace
#
# Usage:
#   bash run_finetune.sh                                       # default config
#   CONFIG=config_MI300X_1x8x1.sh bash run_finetune.sh         # MI300X config
#   TP=2 TRAIN_STEPS=500 bash run_finetune.sh                  # override vars

set -euo pipefail

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---- Load configuration -----------------------------------------------------
CONFIG=${CONFIG:-"${SCRIPT_DIR}/config_MI355X_1x8x1.sh"}
source "${CONFIG}"

# ---- Data path fallback: .jsonl -> .npy when only .npy exists -----------------
for _var in TRAIN_DATA VALID_DATA; do
    _path="${!_var}"
    [ -z "${_path}" ] && continue
    if [ ! -f "${_path}" ] && [[ "${_path}" == *.jsonl ]]; then
        _npy="${_path%.jsonl}.npy"
        if [ -f "${_npy}" ]; then
            echo "WARNING: ${_var}='${_path}' not found; using ${_npy}"
            export "${_var}=${_npy}"
        fi
    fi
done

# ---- Validate tokenizer path (fall back to script-relative path) -------------
if [ ! -d "${TOKENIZER:-}" ]; then
    _FALLBACK="${SCRIPT_DIR}/tokenizer"
    if [ -d "${_FALLBACK}" ]; then
        echo "WARNING: TOKENIZER='${TOKENIZER:-}' not found; using ${_FALLBACK}"
        export TOKENIZER="${_FALLBACK}"
    else
        echo "ERROR: Tokenizer directory not found: ${TOKENIZER:-<unset>}"
        echo "  Set TOKENIZER=/path/to/tokenizer before running, or ensure the"
        echo "  'tokenizer/' directory exists next to the config/script file."
        exit 1
    fi
fi

# ---- Performance tuning (model-agnostic, from common module) -----------------
source "${REPO_ROOT}/lumen/models/perf_env.sh"


###############################################################################
# MEGATRON BACKEND
###############################################################################

run_megatron() {
    python -c "import megatron" 2>/dev/null || {
        echo "ERROR: Megatron-LM is not installed."
        echo "  pip install git+https://github.com/ROCm/Megatron-LM.git"
        exit 1
    }

    case "${MODEL_SIZE}" in
        llama2-7B|llama2-7b)
            NUM_LAYERS=32; HIDDEN=4096; FFN_HIDDEN=11008; HEADS=32; KV_HEADS=32 ;;
        llama2-13B|llama2-13b)
            NUM_LAYERS=40; HIDDEN=5120; FFN_HIDDEN=13824; HEADS=40; KV_HEADS=40 ;;
        llama2-70B|llama2-70b)
            NUM_LAYERS=80; HIDDEN=8192; FFN_HIDDEN=28672; HEADS=64; KV_HEADS=8 ;;
        *)
            echo "ERROR: Unknown MODEL_SIZE=${MODEL_SIZE} (use llama2-7B/13B/70B)"
            exit 1 ;;
    esac

    GQA_ARGS=""
    if [ "${KV_HEADS}" -ne "${HEADS}" ]; then
        GQA_ARGS="--group-query-attention --num-query-groups ${KV_HEADS}"
    fi

    VP_ARGS=""; [ "${VP}" -gt 0 ] && VP_ARGS="--num-layers-per-virtual-pipeline-stage ${VP}"
    # Megatron-LM-AMD requires sequence parallelism whenever tensor parallelism is
    # used (validate_args enforces this unconditionally for TP > 1).
    if [ "${TP}" -gt 1 ] && [ "${SP}" -ne 1 ]; then
        echo "WARNING: TP=${TP} > 1 requires SP=1 (Megatron-LM-AMD enforce). Enabling automatically."
        SP=1
    fi
    SP_ARGS=""; [ "${SP}" -eq 1 ] && SP_ARGS="--sequence-parallel"

    LORA_ARGS=""
    if [ "${LORA_RANK}" -gt 0 ]; then
        LORA_ARGS="--lora-rank ${LORA_RANK} --lora-alpha ${LORA_ALPHA} --lora-dropout ${LORA_DROPOUT}"
        [ "${LORA_A2A}" -eq 1 ] && LORA_ARGS="${LORA_ARGS} --lora-a2a"
    fi

    FP8_ARGS=""
    if [ "${FP8_TRAINING}" -eq 1 ]; then
        FP8_ARGS="--linear-fp8 --fp8-format ${FP8_FORMAT} --linear-fp8-scaling ${FP8_SCALING} --linear-fp8-block-size ${FP8_BLOCK_SIZE}"
        FP8_ARGS+=" --linear-fp8-amax-algo ${FP8_AMAX_ALGO} --linear-fp8-amax-history ${FP8_AMAX_HISTORY}"
        [ "${FP8_REDUCE_AMAX}" = "1" ] && FP8_ARGS+=" --linear-fp8-reduce-amax"
        [ "${FP8_ACTIVATION}" = "0" ] && FP8_ARGS+=" --no-linear-fp8-activation"
        [ -n "${GRAD_QUANT_TYPE}" ] && FP8_ARGS+=" --grad-quant-type ${GRAD_QUANT_TYPE}"
    fi

    TL_ATTN_ARGS="--tl-attn-backend ${TL_ATTN_BACKEND}"
    if [ "${TL_ATTN_BACKEND}" = "aiter_triton_fp8" ] || [ "${TL_ATTN_BACKEND}" = "aiter_csrc_fp8" ]; then
        TL_ATTN_ARGS+=" --tl-fp8-quant-type ${TL_FP8_QUANT}"
        if [ "${TL_FP8_QUANT}" = "mxfp8" ]; then
            TL_ATTN_ARGS+=" --mxfp8-block-m-fwd ${MXFP8_BLOCK_M_FWD}"
            TL_ATTN_ARGS+=" --mxfp8-block-n-fwd ${MXFP8_BLOCK_N_FWD}"
            TL_ATTN_ARGS+=" --mxfp8-block-m-dq-bwd ${MXFP8_BLOCK_M_DQ_BWD}"
            TL_ATTN_ARGS+=" --mxfp8-block-n-dq-bwd ${MXFP8_BLOCK_N_DQ_BWD}"
            TL_ATTN_ARGS+=" --mxfp8-block-m-dkv-bwd ${MXFP8_BLOCK_M_DKV_BWD}"
            TL_ATTN_ARGS+=" --mxfp8-block-n-dkv-bwd ${MXFP8_BLOCK_N_DKV_BWD}"
            TL_ATTN_ARGS+=" --mxfp8-quant-block-size ${MXFP8_QUANT_BLOCK_SIZE}"
        fi
    fi

    TL_RMSNORM_ARGS=""
    [ "${TL_RMSNORM}" -eq 1 ] && TL_RMSNORM_ARGS="--tl-rmsnorm"

    WARMUP_ARGS=""; [ "${WARMUP_STEPS}" -gt 0 ] && WARMUP_ARGS="--warmup-steps ${WARMUP_STEPS}"
    EARLY_STOP_ARGS=""; [ -n "${VAL_LOSS_TARGET}" ] && EARLY_STOP_ARGS="--val-loss-target ${VAL_LOSS_TARGET}"

    echo "================================================================"
    echo "LLaMA2 SFT — MEGATRON backend"
    echo "  Config:   ${CONFIG}"
    echo "  Model:    ${MODEL_SIZE} | TP=${TP} PP=${PP} CP=${CP} VP=${VP} SP=${SP}"
    echo "  GPUs:     ${NGPU}x${NNODES}"
    echo "  Batch:    MBS=${MBS} GBS=${GBS} | seq_len=${SEQ_LEN}"
    echo "  TL attn:  ${TL_ATTN_BACKEND}$([ "${TL_ATTN_BACKEND}" = "aiter_triton_fp8" ] || [ "${TL_ATTN_BACKEND}" = "aiter_csrc_fp8" ] && echo " (fp8_quant=${TL_FP8_QUANT})") rmsnorm=${TL_RMSNORM}"
    echo "  LoRA:     rank=${LORA_RANK} a2a=${LORA_A2A}"
    echo "  FP8:      training=${FP8_TRAINING} format=${FP8_FORMAT} algo=${FP8_AMAX_ALGO} hist=${FP8_AMAX_HISTORY}"
    echo "================================================================"

    torchrun --nproc_per_node=${NGPU} --nnodes=${NNODES} \
        "${SCRIPT_DIR}/finetune_llama2.py" \
        --backend megatron \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN} \
        --ffn-hidden-size ${FFN_HIDDEN} \
        --num-attention-heads ${HEADS} \
        ${GQA_ARGS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --use-rotary-position-embeddings \
        --no-position-embedding \
        --normalization RMSNorm \
        --swiglu \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --no-masked-softmax-fusion \
        --attention-softmax-in-fp32 \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        ${VP_ARGS} ${SP_ARGS} \
        --micro-batch-size ${MBS} \
        --global-batch-size ${GBS} \
        --train-iters ${TRAIN_STEPS} \
        --lr ${LR} --min-lr ${MIN_LR} \
        --lr-decay-style cosine --lr-warmup-fraction 0.0 \
        --weight-decay ${WEIGHT_DECAY} \
        --clip-grad ${GRADIENT_CLIP} \
        --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 \
        --${PRECISION} \
        --no-gradient-accumulation-fusion \
        --reset-position-ids --reset-attention-mask --eod-mask-loss \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model ${TOKENIZER} \
        --train-data-path ${TRAIN_DATA} \
        --valid-data-path ${VALID_DATA} \
        --split 100,0,0 \
        --load ${CKPT_DIR} \
        --save ${SAVE_DIR} \
        --finetune --no-load-optim --no-load-rng --auto-detect-ckpt-format \
        --eval-iters 10 --eval-interval ${EVAL_INTERVAL} \
        --save-interval ${SAVE_INTERVAL} --log-interval ${LOG_INTERVAL} \
        ${TL_ATTN_ARGS} ${TL_RMSNORM_ARGS} \
        ${LORA_ARGS} ${FP8_ARGS} ${WARMUP_ARGS} ${EARLY_STOP_ARGS}
}


###############################################################################
# FSDP BACKEND
###############################################################################

run_fsdp() {
    python -c "import transformers" 2>/dev/null || {
        echo "ERROR: HuggingFace Transformers is not installed."
        echo "  pip install transformers peft"
        exit 1
    }

    CMD="torchrun --nproc_per_node=${NGPU}"
    CMD+=" ${SCRIPT_DIR}/finetune_llama2.py"
    CMD+=" --backend fsdp"
    CMD+=" --model-name-or-path ${MODEL}"
    CMD+=" --tokenizer-name-or-path ${TOKENIZER}"
    CMD+=" --seq-length ${SEQ_LEN}"
    CMD+=" --micro-batch-size ${MBS}"
    CMD+=" --gradient-accumulation-steps ${GRAD_ACCUM}"
    CMD+=" --max-steps ${TRAIN_STEPS}"
    CMD+=" --lr ${LR}"
    CMD+=" --min-lr ${MIN_LR}"
    CMD+=" --weight-decay ${WEIGHT_DECAY}"
    CMD+=" --max-grad-norm ${MAX_GRAD_NORM}"
    CMD+=" --log-interval ${LOG_INTERVAL}"
    CMD+=" --save-interval ${SAVE_INTERVAL}"
    CMD+=" --save-dir ${SAVE_DIR}"
    CMD+=" --num-workers ${NUM_WORKERS}"
    CMD+=" --train-data-path ${TRAIN_DATA}"
    CMD+=" --train-samples ${TRAIN_SAMPLES}"
    CMD+=" --val-samples ${VAL_SAMPLES}"
    CMD+=" --sharding-strategy ${SHARDING}"

    [ -n "${VALID_DATA}" ] && CMD+=" --val-data-path ${VALID_DATA}"

    if [ "${LORA_RANK}" -gt 0 ]; then
        CMD+=" --lora-rank ${LORA_RANK} --lora-alpha ${LORA_ALPHA} --lora-dropout ${LORA_DROPOUT}"
    fi

    if [ "${FP8_TRAINING}" = "1" ]; then
        CMD+=" --linear-fp8"
        CMD+=" --linear-fp8-format ${FP8_FORMAT} --linear-fp8-scaling ${FP8_SCALING}"
        CMD+=" --linear-fp8-block-size ${FP8_BLOCK_SIZE} --linear-fp8-amax-algo ${FP8_AMAX_ALGO}"
        CMD+=" --linear-fp8-amax-history ${FP8_AMAX_HISTORY}"
        [ "${FP8_REDUCE_AMAX}" = "1" ] && CMD+=" --linear-fp8-reduce-amax"
        [ "${FP8_ACTIVATION}" = "0" ] && CMD+=" --no-linear-fp8-activation"
        [ -n "${GRAD_QUANT_TYPE}" ] && CMD+=" --grad-quant-type ${GRAD_QUANT_TYPE}"
    fi

    [ "${WARMUP_STEPS}" -gt 0 ] && CMD+=" --warmup-steps ${WARMUP_STEPS}"
    [ -n "${VAL_LOSS_TARGET}" ] && CMD+=" --val-loss-target ${VAL_LOSS_TARGET}"

    echo "================================================================"
    echo "LLaMA2 SFT — FSDP backend"
    echo "  Config:     ${CONFIG}"
    echo "  Model:      ${MODEL}"
    echo "  GPUs:       ${NGPU}"
    echo "  Batch:      MBS=${MBS} x accum=${GRAD_ACCUM} | seq_len=${SEQ_LEN}"
    echo "  Sharding:   ${SHARDING}"
    echo "  LoRA:       rank=${LORA_RANK}"
    echo "  FP8:        training=${FP8_TRAINING} format=${FP8_FORMAT}"
    echo "================================================================"

    eval ${CMD}
}


###############################################################################
# DISPATCH
###############################################################################

start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING LLAMA2 FINETUNE AT ${start_fmt} (backend=${BACKEND})"

set -x

ret_code=0
case "${BACKEND}" in
    megatron) run_megatron || ret_code=$? ;;
    fsdp)     run_fsdp || ret_code=$? ;;
    *)
        echo "ERROR: Unknown BACKEND=${BACKEND}. Use 'megatron' or 'fsdp'."
        exit 1 ;;
esac

if [[ ${ret_code} != 0 ]]; then exit ${ret_code}; fi

end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING LLAMA2 FINETUNE AT ${end_fmt}"
result=$(( end - start ))
echo "RESULT,LLM_FINETUNING,,${result},AMD,${start_fmt}"

exit 0
