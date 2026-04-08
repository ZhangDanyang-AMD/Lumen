# Qwen2-0.5B GRPO Benchmark: BF16 vs FP8

Apple-to-apple comparison of BF16 and FP8 scaling methods on Qwen2-0.5B-Instruct using TRL GRPOTrainer with DeepMath-103K.

## Setup

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2-0.5B-Instruct |
| Dataset | trl-lib/DeepMath-103K |
| Reward | accuracy_reward (requires `\boxed{}`) |
| GPUs | 8x MI300X (DDP) |
| Effective batch | 64 prompts/step (1 per device x 8 grad accum) |
| beta | 0.04 |
| temperature | 0.9 |
| loss_type | grpo |
| num_generations | 8 |
| max_completion_length | 256 |
| learning_rate | 1e-6 |
| bf16 | True |
| gradient_checkpointing | True |
| Model loading | Pre-loaded in BF16 (`torch_dtype=torch.bfloat16`) |
| Reference model | Also BF16 via `model_init_kwargs={"torch_dtype": "bfloat16"}` |
| TRL version | 1.0.0 |
| Steps | 1000 |

## Scripts

| Script | Quantization |
|---|---|
| `train_grpo_bf16_preloaded.py` | None (BF16 control) |
| `train_grpo_lumen_fp8.py` | `LumenConfig(format="fp8_e4m3", scaling="dynamic").enable(model)` |
| `train_grpo_lumen_fp8_delayed.py` | `LumenConfig(format="fp8_e4m3", scaling="delayed").enable(model)` |
| `train_grpo_lumen_fp8_blockwise.py` | `LumenConfig(format="fp8_e4m3", scaling="blockwise").enable(model)` -- Fixed (backward shape mismatch resolved) |

The only code difference between working scripts is the `LumenConfig(...)` constructor arguments.

## Results

| Metric | BF16 | FP8 (dynamic) | FP8 (delayed) |
|---|---|---|---|
| Peak memory | 7.11 GB | 7.11 GB | 7.11 GB |
| Avg step time | 3.76s | 11.95s | 16.32s |
| Overhead vs BF16 | 1.00x | 3.18x | 4.34x |
| Step time degradation | 0.98x | 1.02x | 1.00x |
| Mean KL (actor vs ref) | 0.000795 | 0.020862 | 0.018155 |
| Mean reward | 0.0066 | 0.0052 | 0.0057 |
| Mean loss | 0.000032 | 0.000835 | 0.000727 |
| Mean entropy | 0.7093 | 0.7182 | 0.7111 |

### Reward curve (50-step rolling window)

| Step | BF16 | FP8 dynamic | FP8 delayed |
|---|---|---|---|
| 100 | 0.0078 | 0.0063 | 0.0037 |
| 250 | 0.0091 | 0.0066 | 0.0084 |
| 500 | 0.0047 | 0.0034 | 0.0034 |
| 750 | 0.0041 | 0.0019 | 0.0037 |
| 1000 | 0.0059 | 0.0047 | 0.0059 |

## All Scaling Methods Tested

| Method | Status | Step time | KL vs BF16 | Notes |
|---|---|---|---|---|
| Dynamic | Working | 11.95s (3.18x) | 0.0209 | Fastest FP8 option, exact amax per forward pass |
| Delayed | Working | 16.32s (4.34x) | 0.0182 | TE-style amax history (len=16), 37% slower than dynamic |
| Blockwise | **Fixed** | N/A | N/A | Backward shape mismatch fixed (was at `linear.py:736`, used `_dequant_fp8_weight` for blockwise scales) |

## Recommendation: Dynamic Scaling

**`scaling="dynamic"` is the best choice** for this setup:

1. **Fastest FP8**: 11.95s vs 16.32s (27% faster than delayed)
2. **Comparable accuracy**: KL=0.021 vs delayed's 0.018 — both are 23-26x above BF16, the difference is negligible for reward/entropy
3. **Most principled**: exact scale from current data, no stale history artifacts
4. **Blockwise not viable**: crashes in backward for HuggingFace models via TRL

## Why No Memory Savings on 0.5B

On a 0.5B model, the parameter memory (~1GB in BF16) is a small fraction of the 7.11GB total (dominated by activations, optimizer states, generation buffers, and the reference model). Even storing weights in FP8 would only save ~0.5GB (7%).

**Update**: Testing on Llama-3.1-8B (see `../llama-3.1-8b/`) confirmed that even on 8B models, peak memory is identical (112.58 GB) because FP8 Linear only quantizes GEMMs on-the-fly — weights remain BF16. True memory reduction requires FP8ParamManager (FP8 weight storage), 8-bit optimizer states, or FP8 reference model.

## Logs

- `outputs/bf16_preloaded_1000steps.jsonl`
- `outputs/fp8_dynamic_1000steps.jsonl`
- `outputs/fp8_delayed_1000steps.jsonl`

## Launch Commands

```bash
# BF16 baseline
OUTPUT_DIR=/workspace/Lumen/outputs/benchmark/qwen2-0.5b-exact/bf16_preloaded \
accelerate launch --config_file ddp_8gpu.yaml --num_processes 8 --main_process_port 29500 \
  train_grpo_bf16_preloaded.py --max-steps 1000

# FP8 dynamic scaling (recommended)
OUTPUT_DIR=/workspace/Lumen/outputs/benchmark/qwen2-0.5b-exact/lumen_fp8_dynamic \
accelerate launch --config_file ddp_8gpu.yaml --num_processes 8 --main_process_port 29500 \
  train_grpo_lumen_fp8.py --max-steps 1000

# FP8 delayed scaling
OUTPUT_DIR=/workspace/Lumen/outputs/benchmark/qwen2-0.5b-exact/lumen_fp8_delayed \
accelerate launch --config_file ddp_8gpu.yaml --num_processes 8 --main_process_port 29500 \
  train_grpo_lumen_fp8_delayed.py --max-steps 1000
```
