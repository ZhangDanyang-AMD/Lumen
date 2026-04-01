#!/bin/bash
# TP=1 DP=8 config matching MLPerf reference parallelism.
# NousResearch/Llama-2-70b-hf, FP8 + LoRA, seed=1234.
# Self-contained (does NOT source base config).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Backend -----------------------------------------------------------------
export BACKEND=megatron
export MODEL_SIZE="llama2-70B"

# ---- Hardware ----------------------------------------------------------------
export NGPU=8
export NNODES=1

# ---- Parallelism: TP=1 DP=8 (matching MLPerf) -------------------------------
export TP=1
export PP=1
export VP=0
export CP=1
export SP=0

# ---- Training hyperparameters (aligned with MLPerf MI300X actual submission) -
export MBS=1
export GBS=8
export SEQ_LEN=8192
export LR=2e-4
export MIN_LR=0
export TRAIN_STEPS=1024
export WARMUP_STEPS=5
export LR_WARMUP_STEPS=100
export LOG_INTERVAL=1
export SAVE_INTERVAL=999999
export SEED=1234
export WEIGHT_DECAY=1e-4
export GRADIENT_CLIP=0.3
export ADAM_BETA1=0.9
export ADAM_BETA2=0.999
export ADAM_EPS=1e-8
export PRECISION="bf16"

# ---- LoRA (matching MLPerf: rank=16, alpha=32, dropout=0.1) ------------------
export LORA_RANK=16
export LORA_ALPHA=32
export LORA_DROPOUT=0.1
export LORA_A2A=1

# ---- Activation checkpointing (matching MLPerf) -----------------------------
export RECOMPUTE_GRANULARITY="full"
export RECOMPUTE_METHOD="block"
export RECOMPUTE_NUM_LAYERS=80

# ---- FP8 training (matching MLPerf) -----------------------------------------
export FP8_TRAINING=1
export FP8_FORMAT="e4m3"
export FP8_SCALING="delayed"
export FP8_BLOCK_SIZE=128
export FP8_AMAX_ALGO="most_recent"
export FP8_REDUCE_AMAX=0
export FP8_AMAX_HISTORY=4
export FP8_ACTIVATION=1
export FP8_WGRAD=1
export FIRST_LAST_BF16=0
export GRAD_QUANT_TYPE=""

# ---- Lumen attention ---------------------------------------------------------
export LUMEN_ATTN_BACKEND="csrc"
export LUMEN_ATTN_KERNEL_BACKEND="triton"
export LUMEN_FP8_QUANT="blockwise"
export LUMEN_RMSNORM=0
export LUMEN_NORM=0

# ---- Evaluation (matching MLPerf: VAL_CHECK_INTERVAL=384 samples) -----------
export EVAL_EVERY=384
export EVAL_INTERVAL=0
export EVAL_ITERS=10
export START_EVAL_AT=0

# ---- Warmup / early stopping ------------------------------------------------
export VAL_LOSS_TARGET=""

# ---- Checkpoint management ---------------------------------------------------
export USE_CKPT=0
export FROM_HF=1
export SAVE_CKPT=0
export CONTINUAL_CKPT="/data1/lumen/results/tp1_fp8/saved_ckpts"
export CKPT_START_STEP=0
export FP8_PARAMS=0
export FP8_PARAM_STORAGE=1

# ---- Distributed optimizer (shard optimizer states across DP ranks) ----------
export USE_DIST_OPTIMIZER=1

# ---- Experiment management ---------------------------------------------------
export TAG=""
export TARGET_LOG_PPL="3.3"
export STEP_TIME_ATOL=18000

# ---- Launcher compatibility --------------------------------------------------
export USE_SDMA=0
export PRIMUS_FP8_ATTN=0
export PRIMUS_MXFP8_ATTN=0
export DBG_ATTN_OUTPUT=0
export LOG_MEMORY=1

# ---- Paths (TP=1 checkpoint) ------------------------------------------------
export CKPT_DIR="/data1/lumen/megatron_ckpt_nous_tp1"
export TRAIN_DATA="/data1/lumen/data/train.npy"
export VALID_DATA="/data1/lumen/data/validation.npy"
export TOKENIZER="${SCRIPT_DIR}/tokenizer"
export SAVE_DIR="/data1/lumen/results/tp1_fp8"

# ---- MLPerf-aligned flags ----------------------------------------------------
export MAKE_VOCAB_DIVISIBLE_BY=1
export DISABLE_RESET_FLAGS=1

# ---- MI300X tuning -----------------------------------------------------------
export VBOOST_VALUE=1
export CUBLAS_FORCE_XMMA_KERNEL_INIT=DEVICE
export CK_FUSED_ATTN_LOG_CONFIG=0
export POSSIBLE_USER_WARNINGS=0
export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0

# ---- hipBLASLt ---------------------------------------------------------------
export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1

# ---- Misc ROCm perf ---------------------------------------------------------
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_FILENAME="tunableop_results.csv"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1

# ---- NCCL / ROCm -----------------------------------------------------------
export NCCL_MIN_P2P_NCHANNELS=32
export NCCL_MIN_CTAS=32
export NCCL_NCHANNELS_PER_NET_PEER=32
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export HSA_ENABLE_SDMA=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN

# ---- MLPerf submission -------------------------------------------------------
export MLPERF_SUBMISSION_ORG="AMD"
export MLPERF_SUBMISSION_PLATFORM="MI300X"

# ---- MXFP8 attention block sizes (unused but expected by scripts) ------------
export MXFP8_BLOCK_M_FWD=128
export MXFP8_BLOCK_N_FWD=128
export MXFP8_BLOCK_M_DQ_BWD=128
export MXFP8_BLOCK_N_DQ_BWD=128
export MXFP8_BLOCK_M_DKV_BWD=128
export MXFP8_BLOCK_N_DKV_BWD=128
export MXFP8_QUANT_BLOCK_SIZE=128
