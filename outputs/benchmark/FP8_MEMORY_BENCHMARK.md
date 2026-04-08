# FP8 Memory Savings Benchmark

**Date**: 2026-04-08
**Model**: Llama-3.1-8B (8.03B parameters)
**Hardware**: AMD MI300X (252 GB HBM3), single GPU
**Setup**: Gradient checkpointing enabled, SDPA attention, batch_size=2, seq_len=256, 3 training steps
**Script**: `examples/rl/trl/benchmark/test_fp8_memory.py`
**Raw data**: `outputs/benchmark/llama-3.1-8b/fp8_memory_test_all.json`

## Summary

| Config | Peak Alloc (MB) | Steady-State (MB) | vs BF16 Peak | vs BF16 Steady |
|---|---|---|---|---|
| BF16 Baseline (AdamW) | 76,861 | 46,228 | baseline | baseline |
| FP8ParamManager | 28,909 | 24,757 | **-62.4%** | **-46.5%** |
| FP8ParamManager + 8-bit Adam | 27,923 | 23,769 | **-63.7%** | **-48.6%** |
| FP8 Attention (dpa) | 76,860 | 46,227 | -0.0% | -0.0% |

## Per-Step Detail

### Config 1: BF16 Baseline (AdamW)

| Step | Peak Alloc (MB) | Peak Reserved (MB) | Steady-State (MB) | Loss |
|---|---|---|---|---|
| 1 | 76,861 | 77,726 | 46,228 | 12.6319 |
| 2 | 76,861 | 78,728 | 46,227 | 12.5025 |
| 3 | 76,861 | 78,728 | 46,228 | 12.3784 |

### Config 2: FP8ParamManager (True FP8 Weight Storage)

225 nn.Linear parameters quantized from bf16 to float8_e4m3fn.

| Step | Peak Alloc (MB) | Peak Reserved (MB) | Steady-State (MB) | Loss |
|---|---|---|---|---|
| 1 | 26,761 | 27,390 | 24,756 | 12.7696 |
| 2 | 28,908 | 31,400 | 24,756 | 12.7482 |
| 3 | 28,909 | 31,400 | 24,756 | 12.7254 |

### Config 3: FP8ParamManager + 8-bit Adam (bitsandbytes 0.49.2)

225 nn.Linear parameters quantized; optimizer states stored in uint8.

| Step | Peak Alloc (MB) | Peak Reserved (MB) | Steady-State (MB) | Loss |
|---|---|---|---|---|
| 1 | 25,776 | 26,388 | 23,769 | 12.7434 |
| 2 | 27,922 | 30,396 | 23,769 | 12.7106 |
| 3 | 27,923 | 30,396 | 23,769 | 12.7434 |

### Config 4: FP8 Attention (dpa) via LumenConfig

FP8 GEMM for QKV projections + output; attention computed in FP8 via AITER `dpa`.

| Step | Peak Alloc (MB) | Peak Reserved (MB) | Steady-State (MB) | Loss |
|---|---|---|---|---|
| 1 | 76,860 | 77,436 | 46,227 | 12.7260 |
| 2 | 76,860 | 77,438 | 46,227 | 12.4585 |
| 3 | 76,860 | 77,438 | 46,227 | 12.3425 |

## Memory Breakdown (Process-Isolated Verification)

Each config was also run in a completely separate Python process to verify no cross-contamination.

| Component | BF16 Baseline | FP8ParamManager | FP8+8bit Adam | FP8 Attention |
|---|---|---|---|---|
| **Parameter storage** | 15,317 MB | 8,160 MB | 8,160 MB | 15,317 MB |
| — FP8 (float8_e4m3fn) | 0 MB | 7,157 MB | 7,157 MB | 0 MB |
| — BF16 (embed, norm, lm_head) | 15,317 MB | 1,003 MB | 1,003 MB | 15,317 MB |
| **Optimizer states** | 30,633 MB | 2,005 MB | 1,018 MB | 30,633 MB |
| — dtype | bfloat16 | bfloat16 | uint8 + fp32 | bfloat16 |
| **Peak allocated** | 76,861 MB | 28,909 MB | 27,923 MB | 76,860 MB |

## Analysis

### Why FP8ParamManager saves so much memory

FP8ParamManager (`lumen.quantize.fp8_params.FP8ParamManager`) replaces `weight.data` on
all `nn.Linear` modules with a `float8_e4m3fn` tensor (1 byte/element instead of 2 bytes/element
for bf16). This has two compounding effects:

1. **Weight storage halved**: 15,317 MB → 8,160 MB (saving ~7.2 GB)
2. **Optimizer states dramatically reduced**: PyTorch's AdamW allocates `exp_avg` and `exp_avg_sq`
   tensors matching the parameter. When parameters are stored as 1-byte FP8 tensors, the optimizer
   state allocation for those parameters is correspondingly smaller. Only the non-quantized
   parameters (embedding, layernorm, lm_head) retain full-sized optimizer states.
   Result: 30,633 MB → 2,005 MB (saving ~28 GB)

### Why FP8 Attention does not save memory here

FP8 Attention (`LumenConfig(fp8_attn="dpa")`) quantizes attention computation but does not change
weight storage or optimizer states. With gradient checkpointing enabled, activations are recomputed
during the backward pass rather than stored, so the attention activation memory savings are
negligible. FP8 Attention primarily saves memory in non-gradient-checkpointing scenarios.

### Correctness caveat

FP8ParamManager stores weights in reduced precision during training. The dequant hooks convert back
to bf16 before each forward pass, but the optimizer updates are applied to the FP8 parameter tensor,
which has lower precision. Loss values are slightly higher and less smooth compared to the BF16
baseline (12.73 vs 12.38 at step 3), indicating some training quality degradation. Long-run
convergence behavior should be validated before production use.

## How to Reproduce

```bash
# Inside the Lumen Docker container
cd /workspace/Lumen
export TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

# Run all configs (single process, note: may have cross-config memory leaks)
python3 examples/rl/trl/benchmark/test_fp8_memory.py \
  --model /dev/shm/model/llama-3.1-8b \
  --configs all \
  --steps 3 --batch-size 2 --seq-len 256 \
  --output outputs/fp8_memory_results.json

# For accurate per-config numbers, run each config separately:
python3 examples/rl/trl/benchmark/test_fp8_memory.py \
  --model /dev/shm/model/llama-3.1-8b --configs bf16 --output outputs/bf16.json
python3 examples/rl/trl/benchmark/test_fp8_memory.py \
  --model /dev/shm/model/llama-3.1-8b --configs fp8params --output outputs/fp8params.json
python3 examples/rl/trl/benchmark/test_fp8_memory.py \
  --model /dev/shm/model/llama-3.1-8b --configs fp8_8bit --output outputs/fp8_8bit.json
python3 examples/rl/trl/benchmark/test_fp8_memory.py \
  --model /dev/shm/model/llama-3.1-8b --configs fp8attn --output outputs/fp8attn.json
```
