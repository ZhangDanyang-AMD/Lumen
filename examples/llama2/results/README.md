# Training Reference Logs

This folder contains full training logs from LLaMA2-70B SFT runs on **8x MI355X (gfx950)** GPUs. They serve as reference outputs for verifying correctness and comparing performance across different quantization configurations.

## Log files

| Log | Backend | Precision | Description |
|-----|---------|-----------|-------------|
| `llama2_megatron_train_bf16.log` | Megatron | BF16 | Baseline (no quantization) |
| `llama2_megatron_train_fp8_blockwise.log` | Megatron | FP8 blockwise | FP8 attention with per-block scaling |
| `llama2_megatron_train_mxfp8.log` | Megatron | MXFP8 | MXFP8 attention (microscaling FP8) |
| `llama2_megatron_train_mxfp8_fp8linear.log` | Megatron | MXFP8 + FP8 linear | MXFP8 attention + FP8 quantised linear layers |
| `llama2_fsdp_train_mxfp8.log` | FSDP | MXFP8 | MXFP8 attention with PyTorch FSDP backend |

## Log format

Each log captures the full training run from launch to final checkpoint:

1. **Launch config** — backend, model size, parallelism (TP/PP/CP/SP), batch size, learning rate
2. **Megatron/FSDP initialization** — argument dump, model parameter counts
3. **Lumen quantization** — `quant.enable()` summary (format, scaling, backend, layer count)
4. **Training iterations** — per-step loss, learning rate, throughput (tokens/sec/GPU)
5. **Validation** — periodic eval loss and perplexity
6. **Checkpoint save** — final checkpoint path and format
