# Qwen3-30B-A3B Megatron + SonicMoE

This example runs Qwen3-30B-A3B on one 8×MI350X node (TP=1, PP=1, CP=1, EP=8).
Expert implementations:

- `sequential`: Megatron `SequentialMLP`
- `te_grouped`: Transformer Engine `TEGroupedMLP`
- `sonic`: AITER SonicMoE (replaces `MoELayer.experts` only)

## Fastest Megatron SonicMoE recipe (default)

These knobs are the script defaults. Set `MOE_IMPL=sonic` and, for the
reported numbers, `MBS=2 GBS=256` with a real checkpoint and FineWeb.

| Knob | Default | What it does |
|---|---|---|
| `LUMEN_ATTN_BACKEND` | `csrc` | AITER CK `fmha_v3` |
| `SONIC_MOE_GROUPED_GEMM_BACKEND` | `triton` | Triton grouped GEMM |
| `SONIC_MOE_USE_QWEN3_TUNED_GEMM` | `1` | Qwen3-tuned Triton configs |
| `OVERLAP_MOE_EP_COMM` | `1` | Combined 1F1B EP all-to-all overlap |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `8` | Extra HIP queues for overlap |
| `CUDA_GRAPH_IMPL` | `transformer_engine` | Megatron TE `make_graphed_callables` |
| `CUDA_GRAPH_SCOPE` | `attn` | Graph `_forward_attention` only (not MoE/A2A) |
| `MOE_PAD_TO_CAPACITY` | `0` | Leave off; drop-and-pad changes loss |
| `GRAD_ACC_FUSION` | `0` | Leave off; small-batch gain was ~2.6% |

Disable graphs with `CUDA_GRAPH_SCOPE=none`. Restore Triton attention with
`LUMEN_ATTN_BACKEND=triton`.

`run_docker.sh` does **not** forward arbitrary host env into the training
process. Anything the Python job must see (`LUMEN_ATTN_BACKEND`,
`SONIC_MOE_*`, `CUDA_GRAPH_*`, `QWEN_E2E_PROFILE_*`, …) has to be listed
in `run_docker.sh --env` (already true for the table above) **or**
`export`ed again inside `COMMAND`.

### Production e2e (real weights + FineWeb)

Image: `zhangdanyangamd/lumen:qwen3-30b-a3b-350x-pretrain260829-multistream`.
BF16, seq=4096, 20 steps, median of steps 11–20. Loss matches the
pre-graph recipe (step 20 lm loss 2.3846 vs 2.3847).

| Config | Step (median) | Throughput | TFLOP/s/GPU | mem usages |
|---|---|---|---|---|
| Triton attn + multistream GEMM (no overlap extras) | 24.71 s | 10.36 samples/s | 122.0 | 0.609 |
| 4 opts (CK + Triton GEMM + EP overlap + CONN=8) | 17.47 s | 14.66 samples/s | 172.7 | 0.710 |
| **4 opts + attn CUDA Graph (current default)** | **12.72 s** | **20.13 samples/s** | **237.2** | 0.692 |

```bash
cd /home/leiwu/Lumen   # or your clone

HOST_ASSET_ROOT=/dev/shm/qwen3-30b-a3b \
MODEL_PATH=/nobackup/model/Qwen3-30B-A3B \
DATA_PATH=/nobackup/data/fineweb-sample-10BT-26624.jsonl \
MEGATRON_LOAD_PATH=/nobackup/checkpoints/Qwen3-30B-A3B-tp1-pp1-ep8 \
MOE_IMPL=sonic \
RUN_SUFFIX=mbs2-gbs256-best \
TRAIN_STEPS=20 SEQ_LEN=4096 MBS=2 GBS=256 \
COMMAND='export TOKENIZER_PATH=/nobackup/model/Qwen3-30B-A3B
bash run_qwen3_30b_a3b_megatron.sh' \
bash examples/qwen3-30b-a3b/run_docker.sh
```

Host paths under `HOST_ASSET_ROOT` are mounted at `/nobackup` in the
container. Attribution, profiles, and A/B notes:
`docs/qwen3-30b-a3b-perf-optimization.md`.

### Smoke (mock data, default MBS=1 GBS=8)

```bash
MOE_IMPL=sonic TRAIN_STEPS=5 SEQ_LEN=1024 \
  bash examples/qwen3-30b-a3b/run_docker.sh
```

### Blockwise FP8 training and offline export

The default FP8 recipe (`FP8_MODE=blockwise2d`) matches official
[`Qwen/Qwen3-30B-A3B-FP8`](https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8)
fine-grained FP8 (`fmt=e4m3`, `weight_block_size=[128,128]`,
`activation_scheme=dynamic`):

| | Official `-FP8` | This training path |
|---|---|---|
| Weight | E4M3, 128×128 tiles | same (`blockwise2d`) |
| Activation | dynamic 1×128 | same (per-token × 128 channels) |
| FP8 GEMMs | `q/k/v/o_proj` + expert `gate/up/down` | fused `linear_qkv` / `linear_proj` + Sonic fused `w1` (gate+up) / `w2` |
| Forward + backward | inference dump is forward-only | those GEMMs use FP8 dgrad/wgrad too |
| Kept in BF16 | embed, `lm_head`, norms, router (`mlp.gate`) | same; attention SDPA stays BF16 |
| Checkpoint on disk | HF safetensors (FP8) | BF16 Megatron master (FP8 is compute-only) |

Qwen3 dims (QKV out=5120, expert `2×768=1536`) are 128-aligned, so fused QKV/`w1`
tiles match split official Linear tiles. ROCm uses `float8_e4m3fnuz` where
CUDA official weights use `float8_e4m3fn`. Do **not** set
`LUMEN_FP8_EXPERTS_ONLY=1` if you want official layer coverage (that flag
leaves QKV/proj in BF16).

```bash
FP8_MODE=blockwise2d MOE_IMPL=sonic \
TRAIN_STEPS=5 SEQ_LEN=1024 MBS=1 GBS=8 \
  bash examples/qwen3-30b-a3b/run_docker.sh
```

`MOE_IMPL=sequential` also works. TE grouped experts are not covered.

The resumable training checkpoint remains BF16. Export the last **Megatron**
iteration with `checkpoint/export_megatron_blockwise_fp8.py` (E4M3 + sibling
`weight_scale_inv`, same dequant as official). Megatron key names are kept;
a Hugging Face Transformers tree is still a separate conversion. FSDP
training does not have an equivalent exporter.

## Build

```bash
git submodule update --init third_party/aiter
bash examples/qwen3-30b-a3b/build.sh
```

The image is built from the official ROCm 7.2 PyTorch base. Transformer Engine
`v2.10_rocm`, ROCm Megatron-LM, the SonicMoE AITER revision, and Lumen are all
built from source; it does not inherit Miles or Primus.

To reproduce without rebuilding:

```bash
docker pull zhangdanyangamd/lumen:qwen3-30b-a3b-350x-pretrain260829-multistream

IMAGE_NAME=zhangdanyangamd/lumen:qwen3-30b-a3b-350x-pretrain260829-multistream \
  MOE_IMPL=sonic bash examples/qwen3-30b-a3b/run_docker.sh
```

## SequentialMLP versus TEGroupedMLP

```bash
COMMAND="bash benchmark_mlp.sh" \
  bash examples/qwen3-30b-a3b/run_docker.sh
```

The default benchmark runs 20 iterations at sequence length 4096 and writes
per-run logs plus `results/mlp_summary.csv`. Override `TRAIN_STEPS`, `SEQ_LEN`,
`MBS`, and `GBS` through the environment.

## Kernel-level Qwen shape

With EP=8, each GPU owns 16 experts. Top-8 routing produces roughly eight
dispatched rows per source token after all-to-all, so the representative local
SonicMoE shape at sequence length 4096 is:

```bash
cd /workspace/Lumen/third_party/aiter
python op_tests/test_sonicmoe.py \
  --activation swiglu --benchmark \
  --T 32768 --H 2048 --I 768 --E 16 --K 1
```

SonicMoE tuning data is stored under
`third_party/aiter/aiter/ops/triton/configs/moe/`. Tune against the kernel
benchmark first, then verify gains with the e2e `sonic` run because routing,
all-to-all, and weight-gradient costs are not represented by GEMM-only timing.

## Transformers + FSDP2

The FSDP case uses the HuggingFace `Qwen3MoeForCausalLM` implementation with a
Megatron-compatible overlapping dense-DP and EP layout:

- Experts are partitioned across each EP row and tokens are dispatched with
  differentiable all-to-all collectives.
- Every global rank consumes a different microbatch. Shared parameters are
  sharded over the full dense-DP world, while corresponding local experts are
  sharded only over expert-DP replicas (`world_size / EP_SIZE`).
- `--expert-backend` selects `sequential`, `te_grouped`, or `sonic` (Python
  default is `te_grouped`). The Sonic path supports both AITER general-routing
  and pre-routed APIs and keeps its gate/up weights in the interleaved layout
  required for correct gradients.
- BF16 full-parameter training is supported for all three backends.
- Lumen FP8 `blockwise2d` is supported for `sequential` (HF `nn.Linear` hooks)
  and `sonic` (`grouped_fp8_expert_mlp` on `w1`/`w2`). The router, embed,
  `lm_head`, and norms stay BF16, matching official `Qwen3-30B-A3B-FP8`.
  **TE grouped experts are not covered** (`MODE`/`FP8_MODE=blockwise2d` with
  `EXPERT_BACKEND=te_grouped` exits). There is no FSDP FP8 export yet; only
  Megatron `checkpoint/export_megatron_blockwise_fp8.py`.
- Sonic expert FP8 needs the AITER grouped GEMM that accepts `A_scale` /
  `B_scale` (host `third_party/aiter` branch `lumen/moe`). The Lumen gitlink
  pin (`ccd9200`) does not include that kernel; without it the GPU tests skip
  and training cannot run official expert FP8. `run_docker.sh` bind-mounts
  the host aiter tree; `run_qwen3_30b_a3b_fsdp.sh` now mounts the same.
- `expert_dp > 1` (multi-node expert replicas) plus Sonic FP8 is not
  smoke-tested. The one-node recipe is `DP=8 EP=8` so `expert_dp=1`.
- `LUMEN_FP8_EXPERTS_ONLY=1` skips HF `q/k/v/o_proj` as well as Megatron
  `linear_qkv` / `linear_proj`. Do not set it for official layer coverage.

Run dense-DP=8, EP=8 on one eight-GPU node (Alpaca, default TE grouped, BF16):

```bash
NNODES=1 DP_SIZE=8 EP_SIZE=8 \
HOST_MODEL=/path/to/Qwen3-30B-A3B \
HOST_DATA=/path/to/alpaca \
TRAIN_FILE=train.jsonl VAL_FILE=test.jsonl \
bash examples/qwen3-30b-a3b/run_qwen3_30b_a3b_fsdp.sh
```

For two eight-GPU nodes, run the same command on both nodes with `NNODES=2`,
`DP_SIZE=16`, a shared `MASTER_ADDR`, and `NODE_RANK=0`/`1`. For FP8 on that
launcher:

```bash
MODE=fp8_blockwise2d EXPERT_BACKEND=sonic \
NNODES=1 DP_SIZE=8 EP_SIZE=8 \
HOST_MODEL=/path/to/Qwen3-30B-A3B \
HOST_DATA=/path/to/alpaca \
TRAIN_FILE=train.jsonl VAL_FILE=test.jsonl \
bash examples/qwen3-30b-a3b/run_qwen3_30b_a3b_fsdp.sh
```

Docker accuracy runs (`run_fsdp.sh`) use the same `FP8_MODE` as Megatron.
Default `EXPERT_BACKEND` in docker is `te_grouped`; set `sonic` or
`sequential` for FP8. Prefer a unique `MASTER_PORT`.

```bash
HOST_ASSET_ROOT=/dev/shm/qwen3-30b-a3b \
MODEL_PATH=/nobackup/model/Qwen3-30B-A3B \
DATA_PATH=/nobackup/data/fineweb-sample-10BT-26624.jsonl \
FP8_MODE=blockwise2d EXPERT_BACKEND=sonic MASTER_PORT=29512 \
TRAIN_STEPS=20 SEQ_LEN=4096 GBS=256 MBS=2 \
COMMAND='export TOKENIZER_PATH=/nobackup/model/Qwen3-30B-A3B
bash run_fsdp.sh' \
bash examples/qwen3-30b-a3b/run_docker.sh
```

The Python entry point can also be launched directly. Pass `--mode` and
`--expert-backend` explicitly; omitting them is BF16 + TE grouped:

```bash
torchrun --nproc_per_node=8 \
  pretrain_qwen3_30b_a3b_fsdp.py \
  --model-name-or-path /path/to/Qwen3-30B-A3B \
  --train-data-path /path/to/train.jsonl \
  --ep-size 8 --dp-size 8 \
  --mode fp8_blockwise2d --expert-backend sonic
```

For a from-scratch BF16 accuracy run aligned with the Megatron TE flow
(same architecture, sequence/global batch sizes, optimizer, LR schedule,
router normalization, and normalized auxiliary-loss coefficient), run:

```bash
TRAIN_STEPS=100 COMMAND="bash run_fsdp.sh" \
  bash examples/qwen3-30b-a3b/run_docker.sh
```

To run the GBS=256 comparison at the largest validated microbatch on one
8×MI350X node:

```bash
GBS=256 MBS=4 SEQ_LEN=4096 EXPERT_BACKEND=sonic \
TRAIN_STEPS=20 COMMAND="bash run_fsdp.sh" \
  bash examples/qwen3-30b-a3b/run_docker.sh
```

At sequence length 4096, MBS 32, 16, and 8 exceed the 288 GiB device-memory
limit with activation checkpointing disabled. MBS 4 is the largest divisor of
GBS 256 that completed the BF16 one-step memory probe. FSDP FP8 at the same
GBS OOMs at MBS=4 on the logits; use MBS=2.

`run_docker.sh` defaults to
`zhangdanyangamd/lumen:qwen3-30b-a3b-350x-pretrain260829-multistream` and
overlays the local FSDP implementation plus the host `third_party/aiter`
tree (needed for Sonic FP8 grouped GEMM).

## Benchmark

The optimized BF16 benchmark uses 8× MI350X GPUs, sequence length 4096,
micro-batch size 2, global batch size 256, TP=1, PP=1, and EP=8. Runs load
the real Qwen3-30B-A3B checkpoint and FineWeb data; the 4,513 physical
sequences are cycled to provide the 5,120 samples required by 20 steps.
Performance is the median of steps 11–20.

![Qwen3-30B-A3B MoE benchmark](results/qwen3-30b-a3b-real-optimized20-speed.png)

Best validated configurations:

- **FSDP Sequential:** 32.413 s/step, 7.898 samples/s, 161.2 GiB peak memory.
- **FSDP SonicMoE:** 22.356 s/step, 11.451 samples/s, 128.0 GiB peak memory.
  Uses global expert layout, Triton forward GEMM, multistream grouped GEMM,
  and normal-priority (`0`) HIP streams.
- **FSDP TE Grouped:** 22.375 s/step, 11.441 samples/s, 153.0 GiB peak memory.
  Uses the TE CK/CUTLASS grouped GEMM path.
- **Megatron Sequential:** 29.364 s/step, 8.718 samples/s, 142.5 GiB peak memory.
- **Megatron SonicMoE:** 24.498 s/step, 10.450 samples/s, 123.9 GiB peak memory.
  Uses multistream grouped GEMM with normal-priority (`0`) HIP streams.
- **Megatron TE Grouped:** 28.587 s/step, 8.956 samples/s, 142.5 GiB peak memory.
  Uses the TE CK/CUTLASS grouped GEMM path.

All six runs completed 20 steps with final LM loss in the 2.385–2.386 range.
High-priority (`-1`) Sonic streams starved Megatron RCCL traffic and were
slower; normal priority also edged out high priority for FSDP. TE hipBLASLt
autotuning was not selected because loading its generated cache failed in the
tested Transformer Engine build (`Invalid scale name: float3`).

Reproduce the optimized matrices:

```bash
cd examples/qwen3-30b-a3b
TRAIN_STEPS=20 MBS=2 GBS=256 SEQ_LEN=4096 \
  python run_qwen3_benchmark.py fsdp
TRAIN_STEPS=20 MBS=2 GBS=256 SEQ_LEN=4096 \
  python run_qwen3_benchmark.py megatron
python run_qwen3_benchmark.py summarize --warmup-steps 10
```
