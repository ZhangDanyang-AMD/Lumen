# Qwen3-30B-A3B: reproducing the four best runs

中文：[qwen3-30b-a3b-reproduce.md](qwen3-30b-a3b-reproduce.md)

Optimization write-ups:

- [Megatron BF16](qwen3-30b-a3b-bf16-megatron-optimization.md)
- [FSDP BF16](qwen3-30b-a3b-bf16-fsdp-optimization.md)
- [Megatron FP8](qwen3-30b-a3b-fp8-megatron-optimization.md)
- [FSDP FP8](qwen3-30b-a3b-fp8-fsdp-optimization.md)

These are the four best 20-step runs remeasured on 2026-10-12 on `dev/qwen3-30b-a3b-opt-fsdp-fp8`. Every run is 8×MI350X, TP=PP=CP=1, EP=8, seq=4096, MBS=2, GBS=256. The median is the average of the 5th and 6th values after sorting steps 11–20. samples/s = 256 / median seconds. The first three steps include load and autotune and are left out. Step-time noise is about 50–100 ms.

| Path | Median step | samples/s | step 1 lm / grad | step 20 lm / grad | Log |
|---|---:|---:|---|---|---|
| Megatron BF16 | **8.519 s** | **30.05** | 2.562231 / 6.714 | 2.410159 / 0.574 | `sonic-latest-branch-bf16-correct-tokenizer` |
| Megatron FP8 | **8.435 s** | **30.35** | 2.561891 / 6.516 | 2.412969 / 0.567 | `sonic-latest-branch-fp8-correct-tokenizer` |
| FSDP BF16 | **10.209 s** | **25.08** | 2.561523 / 6.727 | 2.384460 / 0.420 | `fsdp-sonic-latest-branch-bf16` |
| FSDP FP8 | **9.545 s** | **26.82** | 2.562195 / 6.563 | 2.412476 / 0.556 | `fsdp-sonic-latest-branch-fp8_blockwise2d` |

Logs live in `examples/qwen3-30b-a3b/results/`, with the suffix `-seq4096-mbs2-gbs256.log`. A 3-step FSDP FP8 run and a 20-step run do not share the step-1 batch. Check numbers against the 20-step log.

Step 20 is a different number on each numerical path. Megatron BF16 with FlyDSL lands at 2.410. Both FP8 runs land near 2.413. FSDP BF16 lands at 2.384.

## 1. What one reproduction needs

Image: `zhangdanyangamd/lumen:qwen3-30b-a3b-350x-pretrain260829-multistream`. `run_docker.sh` mounts this repo's training code and `third_party/aiter` into the container, so aiter stays at the commit recorded on this branch, `79c1a41`. FlyDSL lives on the host at `/home/leiwu/FlyDSL` and is mounted only when `SONIC_MOE_GEMM_BACKEND=flydsl`.

The host directory `/home/leiwu/qwen3-30b-a3b` is mounted at `/nobackup` inside the container:

- `model/Qwen3-30B-A3B`
- `checkpoints/Qwen3-30B-A3B-tp1-pp1-ep8`
- `data/fineweb-sample-10BT-26624.jsonl`

Point `HOST_ASSET_ROOT` at that host directory. Run all four commands from `examples/qwen3-30b-a3b`. Pick a free `MASTER_PORT` each time; the ports below are the ones used for the remeasure.

`run_docker.sh` forwards only the environment variables it lists. `TOKENIZER_PATH` is not on that list, so it has to be inside `COMMAND` and point at `/nobackup/model/Qwen3-30B-A3B`. If it is missing, Megatron uses the example tokenizer baked into the image and step 1 leaves the table above.

## 2. Megatron BF16

FlyDSL expert GEMM, OPUS forward, dQ atomic16, Q RMSNorm, bf16 router backward, native apex RoPE, manual GC, expert remap, attention CUDA graph, and EP overlap. `LUMEN_FUSED_ROPE` and `MANUAL_GC` default on in the Megatron script. Leave `FP8_MODE` at `bf16`. FlyDSL consumes BF16.

```bash
HOST_ASSET_ROOT=/home/leiwu/qwen3-30b-a3b \
MODEL_PATH=/nobackup/model/Qwen3-30B-A3B \
MEGATRON_LOAD_PATH=/nobackup/checkpoints/Qwen3-30B-A3B-tp1-pp1-ep8 \
DATA_PATH=/nobackup/data/fineweb-sample-10BT-26624.jsonl \
FP8_MODE=bf16 MOE_IMPL=sonic SONIC_MOE_GEMM_BACKEND=flydsl \
LUMEN_ATTN_BACKEND=opus LUMEN_ATTN_BWD_ATOMIC_FP32=0 \
LUMEN_QK_RMSNORM=1 LUMEN_ROUTER_BWD_BF16=1 LUMEN_EXPERT_REMAP=1 \
MASTER_PORT=30218 TRAIN_STEPS=20 SEQ_LEN=4096 MBS=2 GBS=256 \
RUN_SUFFIX=latest-branch-bf16-correct-tokenizer \
COMMAND='export TOKENIZER_PATH=/nobackup/model/Qwen3-30B-A3B; bash run_qwen3_30b_a3b_megatron.sh' \
bash run_docker.sh
```

The log should contain `Lumen fused RoPE enabled` and `Lumen expert remap enabled`. Step 1 grad norm is 6.714.

## 3. Megatron FP8

The same Megatron stack, with compute in blockwise2d. Expert GEMM uses Triton (`SONIC_MOE_GEMM_BACKEND=triton`). Gluon QKV, the tiled FP8 transpose, fp32 `main_grad` accumulation, and w1 N=128 are on this default path.

```bash
HOST_ASSET_ROOT=/home/leiwu/qwen3-30b-a3b \
MODEL_PATH=/nobackup/model/Qwen3-30B-A3B \
MEGATRON_LOAD_PATH=/nobackup/checkpoints/Qwen3-30B-A3B-tp1-pp1-ep8 \
DATA_PATH=/nobackup/data/fineweb-sample-10BT-26624.jsonl \
FP8_MODE=blockwise2d MOE_IMPL=sonic SONIC_MOE_GEMM_BACKEND=triton \
LUMEN_ATTN_BACKEND=opus LUMEN_ATTN_BWD_ATOMIC_FP32=0 \
LUMEN_QK_RMSNORM=1 LUMEN_ROUTER_BWD_BF16=1 LUMEN_EXPERT_REMAP=1 \
MASTER_PORT=30219 TRAIN_STEPS=20 SEQ_LEN=4096 MBS=2 GBS=256 \
RUN_SUFFIX=latest-branch-fp8-correct-tokenizer \
COMMAND='export TOKENIZER_PATH=/nobackup/model/Qwen3-30B-A3B; bash run_qwen3_30b_a3b_megatron.sh' \
bash run_docker.sh
```

Step 1 is lm loss 2.561891 and grad norm 6.516.

## 4. FSDP BF16

`run_fsdp.sh` reads `AITER_ATTN`, `SHARDING`, and `LUMEN_APEX_ROPE`. Docker does not forward them, so they go inside `COMMAND`. Docker defaults `USE_ROCM_AITER_ROPE_BACKEND` to 1; set it to 0 here, or apex selects the AITER RoPE backend. `LUMEN_MANUAL_GC` defaults on in `pretrain.py`.

```bash
HOST_ASSET_ROOT=/home/leiwu/qwen3-30b-a3b \
MODEL_PATH=/nobackup/model/Qwen3-30B-A3B \
DATA_PATH=/nobackup/data/fineweb-sample-10BT-26624.jsonl \
FP8_MODE=bf16 EXPERT_BACKEND=sonic MOE_GLOBAL_EXPERT_LAYOUT=1 \
LUMEN_ATTN_BACKEND=opus LUMEN_ATTN_BWD_ATOMIC_FP32=0 \
MASTER_PORT=30213 TRAIN_STEPS=20 SEQ_LEN=4096 MBS=2 GBS=256 \
RUN_SUFFIX=latest-branch \
COMMAND='export TOKENIZER_PATH=/nobackup/model/Qwen3-30B-A3B \
LUMEN_EXPERT_REMAP=1 AITER_ATTN=1 SHARDING=shard_grad_op \
LUMEN_APEX_ROPE=1 USE_ROCM_AITER_ROPE_BACKEND=0; bash run_fsdp.sh' \
bash run_docker.sh
```

The log should contain `Enabled apex native fused rotary embedding` and `Manual GC`. Step 1 is lm loss 2.561523 and grad norm 6.727.

## 5. FSDP FP8

Start from the FSDP BF16 recipe, switch on blockwise2d, and quantize the dense QKV/proj FP8 weights once per optimizer step. Expert GEMM stays on Triton. `LUMEN_ASYNC_AUX_COUNTS` defaults to 1.

```bash
HOST_ASSET_ROOT=/home/leiwu/qwen3-30b-a3b \
MODEL_PATH=/nobackup/model/Qwen3-30B-A3B \
DATA_PATH=/nobackup/data/fineweb-sample-10BT-26624.jsonl \
FP8_MODE=blockwise2d EXPERT_BACKEND=sonic MOE_GLOBAL_EXPERT_LAYOUT=1 \
LUMEN_ATTN_BACKEND=opus LUMEN_ATTN_BWD_ATOMIC_FP32=0 \
LUMEN_NORM=1 LUMEN_EXPERT_FUSED_SWIGLU=1 SONIC_MOE_GEMM_BACKEND=triton \
MASTER_PORT=30215 TRAIN_STEPS=20 SEQ_LEN=4096 MBS=2 GBS=256 \
RUN_SUFFIX=latest-branch \
COMMAND='export TOKENIZER_PATH=/nobackup/model/Qwen3-30B-A3B \
LUMEN_EXPERT_REMAP=1 AITER_ATTN=1 SHARDING=shard_grad_op \
LUMEN_APEX_ROPE=1 USE_ROCM_AITER_ROPE_BACKEND=0 \
LUMEN_DENSE_WEIGHT_QUANT_ONCE=1; bash run_fsdp.sh' \
bash run_docker.sh
```

Step 1 is lm loss 2.562195 and grad norm 6.563.
