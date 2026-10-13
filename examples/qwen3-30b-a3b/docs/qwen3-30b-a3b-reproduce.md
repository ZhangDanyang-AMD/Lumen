# Qwen3-30B-A3B 四组最佳结果复现

English: [qwen3-30b-a3b-reproduce.en.md](qwen3-30b-a3b-reproduce.en.md)

下面是 2026-10-13 在 `dev/qwen3-30b-a3b-opt-fsdp-fp8` 上复测的四组 20 步。专家座位不写进源码：开训前用第一批 token 的原始专家 id 现算一张表，在优化器状态出现之前写入专家权重。每个 token 仍进原来的专家，变的只是它落在哪个 rank。形状都是 8×MI350X、TP=PP=CP=1、EP=8、seq=4096、MBS=2、GBS=256。中位数取 steps 11–20 排序后第 5、第 6 个的平均，samples/s = 256 / 中位秒。前 3 步含加载和 autotune，不进统计。步时噪声大约 50–100 ms。

| 路径 | 中位 step | samples/s | step 1 lm / grad | step 20 lm / grad | 日志 |
|---|---:|---:|---|---|---|
| Megatron BF16 | **8.562 s** | **29.90** | 2.594373 / 6.609 | 2.387932 / 0.566 | `sonic-place20-bf16` |
| Megatron FP8 | **8.292 s** | **30.87** | 2.594404 / 6.491 | 2.390064 / 0.553 | `sonic-place20-fp8` |
| FSDP BF16 | **10.251 s** | **24.97** | 2.561523 / 6.727 | 2.384644 / 0.420 | `fsdp-sonic-place20-bf16-bf16` |
| FSDP FP8 | **9.561 s** | **26.77** | 2.562195 / 6.563 | 2.412720 / 0.554 | `fsdp-sonic-place20-fp8-fp8_blockwise2d` |

日志都在 `examples/qwen3-30b-a3b/results/`，文件名后缀是 `-seq4096-mbs2-gbs256.log`。FSDP 的 3 步和 20 步 step 1 不是同一批样本，数值对账用 20 步日志。FSDP 两条的 step 1 与 10 月 12 日相同。Megatron 的 step 1 跟着这张现算的表变，不再是 2.562 / 6.7。四条日志都是 128 个专家换了座位，最重 rank 是平均负载的 1.00 倍。

BF16 和 FP8 的 step 20 不互相比较。Megatron BF16 是 2.388，Megatron FP8 是 2.390，FSDP BF16 是 2.385，FSDP FP8 是 2.413。和 10 月 12 日比，前三组步时落在 50–100 ms 噪声里；Megatron FP8 快了 143 ms。

## 1. 一次复现需要什么

镜像 `zhangdanyangamd/lumen:qwen3-30b-a3b-350x-pretrain260829-multistream`。`run_docker.sh` 会把本仓库的训练代码和 `third_party/aiter` 挂进容器，所以 aiter 要停在本分支记录的 `79c1a41`。FlyDSL 在宿主机 `/home/leiwu/FlyDSL`，只有 `SONIC_MOE_GEMM_BACKEND=flydsl` 时才会挂进去。

宿主机目录 `/home/leiwu/qwen3-30b-a3b` 挂到容器里的 `/nobackup`：

- `model/Qwen3-30B-A3B`
- `checkpoints/Qwen3-30B-A3B-tp1-pp1-ep8`
- `data/fineweb-sample-10BT-26624.jsonl`

不要把 `HOST_ASSET_ROOT` 指到 `/dev/shm`。四条命令都在 `examples/qwen3-30b-a3b` 下执行。`MASTER_PORT` 每次换一个空闲端口；下面沿用复测时的端口。

`run_docker.sh` 只转发它自己列出的环境变量。`TOKENIZER_PATH` 不在这份名单里，必须写进 `COMMAND`，指向 `/nobackup/model/Qwen3-30B-A3B`。漏掉时 Megatron 会用镜像里的示例 tokenizer，step 1 会离开上表。

## 2. Megatron BF16

FlyDSL 专家 GEMM、OPUS 前向、dQ atomic16、Q RMSNorm、router 反向 bf16、apex 原生 RoPE、手动 GC、专家置换、attention CUDA graph、EP overlap。`LUMEN_FUSED_ROPE` 和 `MANUAL_GC` 在 Megatron 脚本里默认开。不要设 `FP8_MODE=blockwise2d`，FlyDSL 只吃 BF16。

```bash
HOST_ASSET_ROOT=/home/leiwu/qwen3-30b-a3b \
MODEL_PATH=/nobackup/model/Qwen3-30B-A3B \
MEGATRON_LOAD_PATH=/nobackup/checkpoints/Qwen3-30B-A3B-tp1-pp1-ep8 \
DATA_PATH=/nobackup/data/fineweb-sample-10BT-26624.jsonl \
FP8_MODE=bf16 MOE_IMPL=sonic SONIC_MOE_GEMM_BACKEND=flydsl \
LUMEN_ATTN_BACKEND=opus LUMEN_ATTN_BWD_ATOMIC_FP32=0 \
LUMEN_QK_RMSNORM=1 LUMEN_ROUTER_BWD_BF16=1 LUMEN_EXPERT_REMAP=1 \
MASTER_PORT=30240 TRAIN_STEPS=20 SEQ_LEN=4096 MBS=2 GBS=256 \
RUN_SUFFIX=place20-bf16 \
COMMAND='export TOKENIZER_PATH=/nobackup/model/Qwen3-30B-A3B; bash run_qwen3_30b_a3b_megatron.sh' \
bash run_docker.sh
```

日志里应有 `Lumen fused RoPE enabled` 和 `packed a placement from the first batch`。step 1 是 lm loss 2.594373、grad norm 6.609。

## 3. Megatron FP8

同一条 Megatron 栈，计算改成 blockwise2d。专家 GEMM 用 Triton，不要设 `flydsl`。Gluon QKV、tiled FP8 transpose、fp32 `main_grad` 累加和 w1 N=128 都在这条默认路径上。

```bash
HOST_ASSET_ROOT=/home/leiwu/qwen3-30b-a3b \
MODEL_PATH=/nobackup/model/Qwen3-30B-A3B \
MEGATRON_LOAD_PATH=/nobackup/checkpoints/Qwen3-30B-A3B-tp1-pp1-ep8 \
DATA_PATH=/nobackup/data/fineweb-sample-10BT-26624.jsonl \
FP8_MODE=blockwise2d MOE_IMPL=sonic SONIC_MOE_GEMM_BACKEND=triton \
LUMEN_ATTN_BACKEND=opus LUMEN_ATTN_BWD_ATOMIC_FP32=0 \
LUMEN_QK_RMSNORM=1 LUMEN_ROUTER_BWD_BF16=1 LUMEN_EXPERT_REMAP=1 \
MASTER_PORT=30241 TRAIN_STEPS=20 SEQ_LEN=4096 MBS=2 GBS=256 \
RUN_SUFFIX=place20-fp8 \
COMMAND='export TOKENIZER_PATH=/nobackup/model/Qwen3-30B-A3B; bash run_qwen3_30b_a3b_megatron.sh' \
bash run_docker.sh
```

step 1 是 lm loss 2.594404、grad norm 6.491。

## 4. FSDP BF16

`AITER_ATTN`、`SHARDING`、`LUMEN_APEX_ROPE` 由 `run_fsdp.sh` 读取，docker 不转发，所以放在 `COMMAND` 里。`USE_ROCM_AITER_ROPE_BACKEND` docker 默认是 1，这里要写成 0，否则 apex 会选 AITER RoPE。`LUMEN_MANUAL_GC` 在 `pretrain.py` 里默认开。

```bash
HOST_ASSET_ROOT=/home/leiwu/qwen3-30b-a3b \
MODEL_PATH=/nobackup/model/Qwen3-30B-A3B \
DATA_PATH=/nobackup/data/fineweb-sample-10BT-26624.jsonl \
FP8_MODE=bf16 EXPERT_BACKEND=sonic MOE_GLOBAL_EXPERT_LAYOUT=1 \
LUMEN_ATTN_BACKEND=opus LUMEN_ATTN_BWD_ATOMIC_FP32=0 \
MASTER_PORT=30242 TRAIN_STEPS=20 SEQ_LEN=4096 MBS=2 GBS=256 \
RUN_SUFFIX=place20-bf16 \
COMMAND='export TOKENIZER_PATH=/nobackup/model/Qwen3-30B-A3B \
LUMEN_EXPERT_REMAP=1 AITER_ATTN=1 SHARDING=shard_grad_op \
LUMEN_APEX_ROPE=1 USE_ROCM_AITER_ROPE_BACKEND=0; bash run_fsdp.sh' \
bash run_docker.sh
```

日志里应有 `Enabled apex native fused rotary embedding`、`Manual GC` 和 `packed a placement from the first batch`。step 1 是 lm loss 2.561523、grad norm 6.727。

## 5. FSDP FP8

在 FSDP BF16 上打开 blockwise2d，并让 dense QKV/proj 的 FP8 权重量化每步只做一次。专家 GEMM 保持 Triton。`LUMEN_ASYNC_AUX_COUNTS` 默认是 1。

```bash
HOST_ASSET_ROOT=/home/leiwu/qwen3-30b-a3b \
MODEL_PATH=/nobackup/model/Qwen3-30B-A3B \
DATA_PATH=/nobackup/data/fineweb-sample-10BT-26624.jsonl \
FP8_MODE=blockwise2d EXPERT_BACKEND=sonic MOE_GLOBAL_EXPERT_LAYOUT=1 \
LUMEN_ATTN_BACKEND=opus LUMEN_ATTN_BWD_ATOMIC_FP32=0 \
LUMEN_NORM=1 LUMEN_EXPERT_FUSED_SWIGLU=1 SONIC_MOE_GEMM_BACKEND=triton \
MASTER_PORT=30243 TRAIN_STEPS=20 SEQ_LEN=4096 MBS=2 GBS=256 \
RUN_SUFFIX=place20-fp8 \
COMMAND='export TOKENIZER_PATH=/nobackup/model/Qwen3-30B-A3B \
LUMEN_EXPERT_REMAP=1 AITER_ATTN=1 SHARDING=shard_grad_op \
LUMEN_APEX_ROPE=1 USE_ROCM_AITER_ROPE_BACKEND=0 \
LUMEN_DENSE_WEIGHT_QUANT_ONCE=1; bash run_fsdp.sh' \
bash run_docker.sh
```

step 1 是 lm loss 2.562195、grad norm 6.563。
