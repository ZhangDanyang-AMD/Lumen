# LLaMA2 SFT

Full fine-tuning or LoRA on LLaMA2 (7B / 13B / 70B) with FP8 attention, packed sequences, and early stopping.

## Quick Start

```bash
# 1. Prepare data and model checkpoint
bash examples/llama2/scripts/prepare_data_and_model.sh

# 2. Run training — Megatron backend (default)
BACKEND=megatron bash examples/llama2/run_finetune.sh

# 2. Or: FSDP backend (no Megatron dependency)
BACKEND=fsdp bash examples/llama2/run_finetune.sh
```

The training script (`finetune_llama2.py`) selects the backend via `--backend megatron|fsdp`.

## CLI Flags

| Feature | CLI Flag |
|---------|----------|
| Attention backend | `--lumen-attn-backend {aiter_csrc,aiter_triton,aiter_triton_fp8,aiter_csrc_fp8}` |
| FP8 quantised training | `--linear-fp8 --fp8-format e4m3` |
| MXFP8 block sizes | `--mxfp8-block-m-fwd 128 ...` (6 independent dims) |
| LoRA | `--lora-rank 16 --lora-alpha 32` |
| LoRA A2A comm opt | `--lora-a2a` |
| Synthetic warmup | `--warmup-steps 5` |
| Early stopping | `--val-loss-target 1.5` |
| Context Parallelism | `--context-parallel-size 2` |

See `run_finetune.sh` for the full list of environment variables and defaults.

## Reference Logs

See [`results/`](results/) for full training logs from LLaMA2-70B SFT runs on 8x MI355X GPUs across different quantization configurations (BF16, FP8 blockwise, MXFP8, FSDP).
