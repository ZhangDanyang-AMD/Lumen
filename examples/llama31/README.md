# LLaMA 3.1 Pretraining

Pretraining LLaMA 3.1 (8B) with FP8 hybrid training and MXFP8 attention, aligned with MLPerf LLM pretraining config.

## Quick Start

```bash
# 1. Prepare data and model checkpoint
bash examples/llama31/scripts/prepare_data_and_model.sh

# 2. Run training — Megatron backend (default)
BACKEND=megatron bash examples/llama31/run_pretrain.sh

# 2. Or: FSDP backend (no Megatron dependency)
BACKEND=fsdp bash examples/llama31/run_pretrain.sh
```

The entry point (`pretrain_llama31.py`) selects the backend via `--backend megatron|fsdp`.

## MI450X (gfx1250)

All MI450 training commands use the published image
`zhangdanyangamd/lumen:dev-gfx1250`. Pull it once before launching:

```bash
docker pull zhangdanyangamd/lumen:dev-gfx1250
```

Published digest for this revision:
`sha256:8d6ad43e9b98f03e4983ceb55612c5498a6c74ade3186c5e6b818774246d54d3`.

The launcher mounts the current Lumen checkout, tokenizer, and results directory
into the container. It passes `/dev/kfd` and `/dev/dri` through automatically.
The host therefore needs Docker access to those devices and a tokenizer under
`examples/llama31/tokenizer`.

### Reproduce the single-MI450 results

Run commands from the repository root. These commands use one GPU, MBS=2,
GBS=128, sequence length 8192, and 50 optimizer steps.

```bash
# BF16
GPU_ARCHS=gfx1250 NPROC=1 HIP_VISIBLE_DEVICES=0 \
PRECISION=bf16 MBS=2 GBS=128 SEQ_LEN=8192 TRAIN_STEPS=50 \
RESULTS_DIR=examples/llama31/results/repro_mi450_bf16 \
bash examples/llama31/run_pretrain_llama31_8b.sh

# FP8 blockwise
GPU_ARCHS=gfx1250 NPROC=1 HIP_VISIBLE_DEVICES=0 \
PRECISION=fp8 FP8_SCALING=blockwise \
MBS=2 GBS=128 SEQ_LEN=8192 TRAIN_STEPS=50 \
RESULTS_DIR=examples/llama31/results/repro_mi450_fp8_blockwise \
bash examples/llama31/run_pretrain_llama31_8b.sh

# FP8 delayed
GPU_ARCHS=gfx1250 NPROC=1 HIP_VISIBLE_DEVICES=0 \
PRECISION=fp8 FP8_SCALING=delayed \
MBS=2 GBS=128 SEQ_LEN=8192 TRAIN_STEPS=50 \
RESULTS_DIR=examples/llama31/results/repro_mi450_fp8_delayed \
bash examples/llama31/run_pretrain_llama31_8b.sh
```

### FSDP backend

The same container launcher supports the HuggingFace/PyTorch FSDP flow. It
derives gradient accumulation from `GBS / (MBS * NPROC)` and uses `no_shard`
for one GPU or `full_shard` for multiple GPUs.

```bash
# FSDP BF16
BACKEND=fsdp GPU_ARCHS=gfx1250 NPROC=1 HIP_VISIBLE_DEVICES=0 \
PRECISION=bf16 MBS=2 GBS=128 SEQ_LEN=8192 TRAIN_STEPS=50 \
RESULTS_DIR=examples/llama31/results/repro_mi450_fsdp_bf16 \
bash examples/llama31/run_pretrain_llama31_8b.sh

# FSDP FP8 delayed; use FP8_SCALING=blockwise for blockwise FP8
BACKEND=fsdp GPU_ARCHS=gfx1250 NPROC=1 HIP_VISIBLE_DEVICES=0 \
PRECISION=fp8 FP8_SCALING=delayed \
MBS=2 GBS=128 SEQ_LEN=8192 TRAIN_STEPS=50 \
RESULTS_DIR=examples/llama31/results/repro_mi450_fsdp_fp8 \
bash examples/llama31/run_pretrain_llama31_8b.sh
```

FSDP evaluation is disabled by default because the first evaluation compiles
additional inference attention kernels. Set `FSDP_EVAL_INTERVAL=50` and
`FSDP_VAL_SAMPLES=128` to evaluate after step 50. FSDP disables the
pre-wrapping FP8 weight cache because FSDP replaces parameter storage while
flattening/sharding.

`IMAGE` defaults to `zhangdanyangamd/lumen:dev-gfx1250`. Each run writes
`lumen_llama31_8b_<precision>.log` and generated mock data under its
`RESULTS_DIR`. Published reference logs are:

- BF16: `results/mi450_llama31_8b_bf16.log`
- FP8 blockwise: `results/mi450_llama31_8b_fp8.log`
- FP8 delayed: `results/mi450_llama31_8b_fp8_delayed.log`

On gfx1250, BF16 linears use tuned FlyDSL A16W16. FP8 blockwise uses tuned
Gluon blockscale GEMMs. FP8 delayed dispatches forward, DGrad, and WGrad to
gfx1250 FlyDSL A8W8 kernels. Attention forward/backward uses AITER FMHA v3.

## CLI Flags

| Feature | CLI Flag | Default |
|---------|----------|---------|
| Model size | `SIZE=8b` (env var) | 8b |
| MXFP8 attention | `--lumen-fp8-quant-type mxfp8` | mxfp8 |
| FP8 training | `--linear-fp8` | enabled |
| Amax algorithm | `--linear-fp8-amax-algo most_recent` | most_recent |
| Amax history | `--linear-fp8-amax-history 4` | 4 |
| Learning rate | `MAX_LR=8e-4` (env var) | 8e-4 |
| Cosine LR warmup | `LR_WARMUP_STEPS=128` | 128 |
| GQA (8 KV heads) | auto from SIZE | 32 heads / 8 KV groups |

See `run_pretrain.sh` for all environment variables.

## Standalone Docker Launcher

`run_pretrain_llama31_8b.sh` is a self-contained launcher that runs Llama3.1-8B
in either BF16 or FP8 delayed/hybrid with all validated Lumen fusion
optimizations, using mock data (no dataset download needed).

It applies SOURCE patches with ``--tag llama`` via
``examples/dsv4/patch_megatron_source.py`` before ``torchrun``.
The non-Docker ``run_pretrain.sh`` Megatron path does the same when ``MEGATRON_ROOT`` is set.

### Launch

```bash
# Pull the default image
docker pull zhangdanyangamd/lumen:dev-gfx1250

# FP8 delayed/hybrid (default)
bash examples/llama31/run_pretrain_llama31_8b.sh

# BF16
PRECISION=bf16 bash examples/llama31/run_pretrain_llama31_8b.sh

# Override common knobs (defaults shown)
PRECISION=fp8 MBS=2 GBS=128 SEQ_LEN=8192 TRAIN_STEPS=50 \
    bash examples/llama31/run_pretrain_llama31_8b.sh
```

### Environment overrides

| Variable | Default | Notes |
|----------|---------|-------|
| `BACKEND` | `megatron` | `megatron` or `fsdp` |
| `PRECISION` | `fp8` | `fp8` (delayed/hybrid) or `bf16` |
| `IMAGE` | `zhangdanyangamd/lumen:dev-gfx1250` | Published gfx1250 Docker image |
| `GPU_ARCHS` | `gfx1250` | Kernel architecture selector |
| `NPROC` | `1` | Number of visible training GPUs |
| `GRAD_ACCUM` | derived | FSDP gradient accumulation; defaults to `GBS/(MBS*NPROC)` |
| `FSDP_SHARDING` | `no_shard` on one GPU | FSDP sharding strategy |
| `FSDP_EVAL_INTERVAL` | `0` | Set above zero to enable FSDP validation |
| `MBS` / `GBS` | `2` / `128` | micro / global batch size |
| `SEQ_LEN` | `8192` | sequence length |
| `TRAIN_STEPS` | `50` | training iterations |
| `TOKENIZER_DIR` | `examples/llama31/tokenizer` | HuggingFace tokenizer dir |
| `RESULTS_DIR` | `examples/llama31/results` | logs + mock data output |

FP8 mode enables fused quantization, fused SwiGLU/norm paths, transpose and
weight caches, and architecture-specific GEMMs. Delayed mode uses hybrid FP8
configuration with the gfx1250 FlyDSL A8W8 fast path; blockwise mode uses the
gfx1250 Gluon kernel.
BF16 keeps only the precision-agnostic fusions (SwiGLU, residual-norm).

### Model config

32 layers · hidden 4096 · FFN 14336 · 32 heads · GQA 8 KV groups · RoPE base 5e5 · RMSNorm · SwiGLU.

### Reference results (50 steps, steady-state step time)

| Precision | Log | step time |
|-----------|-----|-----------|
| BF16 | [`results/llama31_8b_pretrain_bf16.log`](results/llama31_8b_pretrain_bf16.log) | ~13.7 s |
| FP8 delayed | [`results/llama31_8b_pretrain_fp8_delayed.log`](results/llama31_8b_pretrain_fp8_delayed.log) | ~9.8 s |

### Lumen vs TransformerEngine (8×MI325X)

Same config (MBS=2, GBS=128, seq=8192, TP=1, PP=1), steady-state step time and
peak allocated memory:

| Precision | Framework | step time | peak memory |
|-----------|-----------|-----------|-------------|
| BF16 | TransformerEngine | ~13.94 s | 122.1 GB |
| BF16 | Lumen | ~13.66 s | 133.1 GB |
| BF16 | Δ | −2.1% | +9.0% |
| FP8 delayed | TransformerEngine | ~10.73 s | 126.1 GB |
| FP8 delayed | Lumen | ~10.00 s | 133.4 GB |
| FP8 delayed | Δ | −6.8% | +5.8% |
