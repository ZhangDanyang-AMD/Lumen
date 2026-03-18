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
