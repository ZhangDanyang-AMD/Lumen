#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Qwen3-30B-A3B MXFP4 MoE pretraining — Megatron backend launcher.
#
# Runs inside an environment that has Lumen (this checkout, with third_party/aiter),
# Megatron-LM, MORI and Triton importable; see README.md and run_docker.sh.
#
# Usage:
#   bash run_pretrain_qwen3_30b_a3b_megatron.sh
#   CONFIG=/path/to/config.sh bash run_pretrain_qwen3_30b_a3b_megatron.sh
#   TRAIN_ITERS=3000 EXIT_INTERVAL=10 SAVE_INTERVAL=0 bash run_pretrain_qwen3_30b_a3b_megatron.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=${CONFIG:-"${SCRIPT_DIR}/config_MI350X_ep4_mxfp4.sh"}
# shellcheck source=config_MI350X_ep4_mxfp4.sh
source "${CONFIG}"

for f in "${TRAIN_DATA}" "${VALID_DATA}" "${LOAD_DIR}/latest_checkpointed_iteration.txt"; do
    [ -s "${f}" ] || { echo "ERROR: missing ${f}"; exit 1; }
done
python -c "import mori" 2>/dev/null || {
    echo "ERROR: MORI is not importable (pip install third_party/mori, see README.md)"
    exit 1
}
mkdir -p "${RESULTS_DIR}"

SAVE_ARGS=(--save-interval 1000000)
if [ "${SAVE_INTERVAL}" != "0" ]; then
    SAVE_ARGS=(--save "${SAVE_DIR}" --save-interval "${SAVE_INTERVAL}" --no-save-optim --no-save-rng)
fi

EXIT_ARGS=()
[ "${EXIT_INTERVAL}" != "0" ] && EXIT_ARGS=(--exit-interval "${EXIT_INTERVAL}")

WANDB_ARGS=()
if [ -n "${WANDB_PROJECT}" ]; then
    WANDB_ARGS=(--wandb-project "${WANDB_PROJECT}" --wandb-exp-name "${WANDB_EXP_NAME}"
                --wandb-save-dir "${RESULTS_DIR}/wandb"
                --tensorboard-dir "${RESULTS_DIR}/tensorboard" --tensorboard-log-interval 1
                --log-timers-to-tensorboard)
    [ -n "${WANDB_ENTITY}" ] && WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

echo "================================================================"
echo "Qwen3-30B-A3B Pretrain — MEGATRON backend (SonicMoE + MORI EP)"
echo "  Config:   ${CONFIG}"
echo "  Parallel: GPUs=${NGPU}x${NNODES} TP=${TP} PP=${PP} CP=${CP} EP=${EP} ETP=${ETP}"
echo "  Batch:    MBS=${MBS} GBS=${GBS} seq=${SEQ_LEN} iters=${TRAIN_ITERS} exit=${EXIT_INTERVAL}"
echo "  Recipe:   experts=${EXPERT_RECIPE} attn_linear=${LUMEN_ATTN_LINEAR_RECIPE} rope=${LUMEN_ROPE_IMPL} attn=${LUMEN_ATTN_BACKEND}"
echo "================================================================"

torchrun \
    --nproc_per_node="${NGPU}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/pretrain_qwen3_30b_a3b_megatron.py" \
    --backend megatron \
    --lumen-attn-backend "${LUMEN_ATTN_BACKEND}" \
    --num-layers 48 \
    --hidden-size 2048 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 4 \
    --kv-channels 128 \
    --make-vocab-size-divisible-by 1187 \
    --seq-length "${SEQ_LEN}" \
    --max-position-embeddings 40960 \
    --use-rotary-position-embeddings \
    --rotary-base 1000000 \
    --no-position-embedding \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --qk-layernorm \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-bias-swiglu-fusion \
    --num-experts 128 \
    --moe-ffn-hidden-size 768 \
    --moe-router-topk 8 \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff "${AUX_LOSS_COEFF}" \
    --moe-token-dispatcher-type flex \
    --moe-flex-dispatcher-backend mori \
    --moe-mori-kernel-type IntraNode \
    --moe-mori-max-tokens-per-rank "$((SEQ_LEN * MBS))" \
    --moe-permute-fusion \
    --lumen-sonic-moe \
    --lumen-delay-wgrad \
    --lumen-fused-router \
    --enable-experimental \
    --expert-model-parallel-size "${EP}" \
    --expert-tensor-parallel-size "${ETP}" \
    --tensor-model-parallel-size "${TP}" \
    --pipeline-model-parallel-size "${PP}" \
    --context-parallel-size "${CP}" \
    --micro-batch-size "${MBS}" \
    --global-batch-size "${GBS}" \
    --train-iters "${TRAIN_ITERS}" \
    --lr "${LR}" \
    --min-lr "${MIN_LR}" \
    --lr-decay-style cosine \
    --lr-warmup-iters "${LR_WARMUP_ITERS}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --clip-grad "${CLIP_GRAD}" \
    --adam-beta1 "${ADAM_BETA1}" \
    --adam-beta2 "${ADAM_BETA2}" \
    --adam-eps "${ADAM_EPS}" \
    --seed "${SEED}" \
    --bf16 \
    --no-gradient-accumulation-fusion \
    --use-distributed-optimizer \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --manual-gc \
    --manual-gc-interval 0 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "${TOKENIZER}" \
    --train-data-path "${TRAIN_DATA}" \
    --valid-data-path "${VALID_DATA}" \
    --test-data-path "${VALID_DATA}" \
    --dataloader-type cyclic \
    --no-data-sharding \
    --split 98,1,1 \
    --num-workers 2 \
    --eval-interval "${EVAL_INTERVAL}" \
    --eval-iters "${EVAL_ITERS}" \
    "${SAVE_ARGS[@]}" \
    "${EXIT_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    --log-interval "${LOG_INTERVAL}" \
    --log-throughput \
    --load "${LOAD_DIR}" \
    --ckpt-format torch \
    --finetune \
    --no-load-optim \
    --no-load-rng \
    --exit-on-missing-checkpoint \
    2>&1 | tee "${RESULTS_DIR}/qwen3-30b-a3b-ep4-${EXPERT_RECIPE}.log"
