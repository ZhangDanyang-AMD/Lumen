# Llama-3.1-8B GRPO Benchmark: BF16 vs FP8

Apple-to-apple comparison of BF16, FP8 dynamic scaling, and FP8 cached weights on Llama-3.1-8B using TRL GRPOTrainer with DeepMath-103K.

## Setup

| Parameter | Value |
|---|---|
| Model | meta-llama/Llama-3.1-8B (LlamaForCausalLM, 32 layers, 4096 hidden) |
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
| `train_grpo_lumen_fp8.py` | `quant.enable(model, ..., scaling="dynamic")` |
| `train_grpo_lumen_fp8_stored.py` | `quant.enable` + `quant.store_weights_fp8` (FP8 weight cache) |
| `train_grpo_lumen_fp8_blockwise.py` | `quant.enable(model, ..., scaling="blockwise")` |

## Results (1000 steps each, 8x MI300X DDP)

| Metric | BF16 | FP8 dynamic | FP8 cached |
|---|---|---|---|
| Peak memory | 112.58 GB | 112.58 GB | 124.35 GB |
| Avg step time | 5.11s | 19.79s | 12.69s |
| Overhead vs BF16 | 1.00x | 3.87x | 2.48x |
| Mean KL | 0.000701 | 0.011545 | 0.007713 |
| Mean reward | 0.004469 | 0.003719 | 0.003722 |
| Mean loss | 2.80e-5 | 4.62e-4 | 3.09e-4 |
| Mean entropy | 0.5824 | 0.6169 | 0.6102 |

## Analysis

### FP8 Weight Cache (`store_weights_fp8`)

The FP8 cached approach pre-quantizes all `nn.Linear` weights to FP8 and stores them as non-parameter buffers. During forward, the cached FP8 data is fed directly to the GEMM kernel, **skipping the per-forward weight quantization** that `quant.enable()` normally performs.

- **35% faster than FP8 dynamic** (12.78s vs 19.79s) — eliminates per-forward `quantize_input` on the weight tensor (amax + scale computation + quantization kernel)
- **33% lower KL than FP8 dynamic** (0.0077 vs 0.0115) — the cached weight is quantized once at initialization quality, rather than re-quantized on every forward with different numerical states
- **+10.4% memory** over BF16 baseline (124.35 vs 112.58 GB) — the FP8 buffer adds ~7.5 GB on top of the existing BF16 master weight

After each `optimizer.step()`, a post-hook re-quantizes the updated BF16 master weight into the FP8 cache.

### Why True FP8 Weight Storage Is Not Feasible in DDP

Storing `nn.Parameter.data` directly in FP8 (to halve the 16 GB weight memory) was attempted but fails for three reasons:

1. **Gradient dtype constraint**: PyTorch requires `param.grad.dtype == param.dtype`. FP8 gradients result.
2. **No FP8 arithmetic**: `ufunc_add_CUDA` is not implemented for `Float8_e4m3fn`, so gradient accumulation fails.
3. **DDP incompatibility**: `torch.distributed.all_reduce` cannot handle FP8 tensors.
4. **Autograd type promotion**: The C++ autograd engine rejects mixed FP8+BF16 backward graphs ("Promotion for Float8 Types is not supported"), blocking even custom autograd workarounds.

True FP8 parameter storage requires framework-level support (FSDP with FP8 communication, or a custom all-reduce).

### Speed Overhead

FP8 dynamic introduces a **3.87x overhead** due to per-layer amax/scale computation on every forward. FP8 cached reduces this to **2.50x** by amortizing the weight quantization cost. The remaining overhead comes from:
- Input (activation) quantization on every forward (still needed)
- FP8 GEMM kernel dispatch overhead vs native BF16 GEMM

### Accuracy

- FP8 cached KL (0.0077) is 11x higher than BF16 but 33% lower than FP8 dynamic, suggesting quantization noise is reduced when weights are quantized fewer times.
- Reward curves are comparable across all three methods.

## Logs

- `../../outputs/benchmark/llama-3.1-8b/bf16_preloaded_1000steps.jsonl`
- `../../outputs/benchmark/llama-3.1-8b/fp8_dynamic_1000steps.jsonl`
- `../../outputs/benchmark/llama-3.1-8b/lumen_fp8_stored/grpo_perf_log.jsonl`

## Launch Commands

```bash
# BF16 baseline
accelerate launch --config_file ddp_8gpu.yaml --num_processes 8 --main_process_port 29500 \
  train_grpo_bf16_preloaded.py --max-steps 1000

# FP8 dynamic scaling
accelerate launch --config_file ddp_8gpu.yaml --num_processes 8 --main_process_port 29500 \
  train_grpo_lumen_fp8.py --max-steps 1000

# FP8 cached weights (recommended)
accelerate launch --config_file ddp_8gpu.yaml --num_processes 8 --main_process_port 29502 \
  train_grpo_lumen_fp8_stored.py --max-steps 1000
```
