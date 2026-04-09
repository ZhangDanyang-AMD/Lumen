# FP8 Memory Savings — Comprehensive Benchmark Report

**Date**: 2026-04-08
**Hardware**: 8x AMD MI300X (256 GB HBM3 each), Llama-3.1-8B / Llama-2-70B
**Framework**: Lumen + TRL (GRPO) + VERL (GRPO) on ROCm
**API**: Unified `LumenConfig.enable()` — one interface for all FP8, LoRA, and distributed features

---

## Executive Summary

FP8ParamManager + LoRA delivers **up to 82% peak memory reduction** on multi-GPU
training by combining FP8 weight storage (1 byte vs 2 bytes BF16), LoRA
(parameter-efficient training), and 8-bit Adam (halved optimizer precision).

**Headline results** (Llama-3.1-8B, 8x MI300X):
- **VERL actor (DDP)**: 90.08 GB → **16.46 GB** per GPU (**-81.7%**)
- **VERL actor (FSDP2)**: 12.67 GB → **9.17 GB** per GPU (**-27.6%** with LoRA)
- **TRL 70B (DDP)**: 142.73 GB → **73.48 GB** per GPU (**-49%**)

A custom autograd function (`_FP8LinearFunc`) ensures that only the compact FP8
weight + scalar scale are saved for backward, not full BF16 copies — critical
for keeping peak memory proportional to FP8 storage rather than BF16.

The alternative approach (FP8 Linear via `LumenConfig(format="fp8_e4m3", ...)`)
**increases** memory by +3% to +34% and slows training by 10-30%, because it
keeps BF16 weights and adds FP8 GEMM overhead from untuned AITER kernels.

---

## Unified API: `LumenConfig.enable()`

All FP8 management, LoRA, and distributed features are consolidated into a single
interface. Instead of calling `quant.enable()`, `apply_lora()`, `apply_fp8_training()`,
`FP8ParamManager` separately, all downstream consumers now use:

```python
from lumen.config import LumenConfig

cfg = LumenConfig(
    format="fp8_e4m3",       # FP8 Linear format
    scaling="dynamic",        # FP8 scaling strategy
    fp8_param_manager=True,   # FP8 weight storage (memory savings)
    lora_rank=16,             # LoRA rank (0 = disabled)
    use_8bit_adam=True,       # 8-bit Adam optimizer hint
    fp8_attn="dpa",           # FP8 attention mode
)
manager, model = cfg.enable(model)
```

Or from CLI args:
```python
cfg = LumenConfig.from_args(args)
manager, model = cfg.enable(model)
```

### Orchestration order inside `enable()`:

1. **FP8ParamManager** — quantize `nn.Linear` weights to FP8 storage (if `fp8_param_manager=True`)
2. **LoRA (PEFT)** — wrap linears with trainable adapters (if `lora_rank > 0`)
3. **Norm patching** — replace norms with Lumen fused norms (if `lumen_norm=True`)
4. **Pre-quant flags** — delay_wgrad, grad-accum fusion, etc.
5. **FP8 Linear** — patch `nn.Linear.forward` with FP8 GEMM kernels (if `format` is set)
6. **Post-quant features** — fp8_checkpoint, fp8_param_gather, etc.

FP8ParamManager runs **before** LoRA so that adapter weights stay BF16/trainable.

---

## Terminology

| Term | Mechanism | Weight Storage | Memory Impact |
|------|-----------|---------------|---------------|
| **FP8 Linear** (`format="fp8_e4m3"`) | Patches `nn.Linear.forward` with FP8 GEMM | BF16 (unchanged) | **+3% to +34%** |
| **FP8 Weight Cache** (`fp8_weight_cache=True`) | Adds FP8 buffer alongside BF16 | BF16 + FP8 copy | **+2%** |
| **FP8ParamManager** (`fp8_param_manager=True`) | **Replaces** `weight.data` with FP8 | **FP8 (1 byte/elem)** | **-61% to -62%** |

---

## Key Findings

### 1. FP8ParamManager is the only mechanism that saves significant memory

FP8ParamManager (`lumen.quantize.fp8_params.FP8ParamManager`) replaces all
`nn.Linear` weights with `float8_e4m3fn` (1 byte/element instead of 2 bytes for BF16).
This has two compounding effects:
- **Weight storage halved**: 15,317 MB -> 8,160 MB (saving ~7.2 GB)
- **Optimizer states 15x smaller**: AdamW only tracks non-FP8 params (embedding, norms, lm_head). 30,633 MB -> 2,005 MB (saving ~28 GB)

### 2. FP8ParamManager is incompatible with FSDP

- **FSDP1**: `ValueError: Must flatten tensors with uniform dtype` (FP8 + BF16 in same wrap unit)
- **FSDP2**: `RuntimeError: Unconvertible NCCL type Float8_e4m3fn` (NCCL cannot communicate FP8)
- **Solution**: Use DDP (`MULTI_GPU` distributed type) which handles mixed dtypes

### 3. FP8ParamManager freezes linear weights

FP8ParamManager sets `requires_grad=False` on all quantized `nn.Linear` weights
because PyTorch autograd cannot compute gradients for FP8 tensors. Only
non-linear parameters (LayerNorm, Embedding, lm_head) remain trainable (~1.3%).

---

## Demo A: 8B Model — BF16 vs FP8ParamManager (8-GPU, 20 steps)

All configs use DDP (`MULTI_GPU`) for fair comparison on Llama-3.1-8B.

| Config | Peak Mem/GPU | Elapsed | vs BF16 DDP |
|--------|-------------|---------|-------------|
| BF16 DDP | **80.5 GB** | 150s | baseline |
| FP8ParamManager DDP | **31.4 GB** | 793s | **-61% memory** |
| FP8PM + 8-bit Adam DDP | **30.4 GB** | 793s | **-62% memory** |
| BF16 FSDP (8-way shard) | 22.0 GB | 422s | -73% (sharding) |

**Memory savings**: ~50 GB per GPU with FP8ParamManager vs BF16 DDP.

**Note**: FSDP shards the model across 8 GPUs, giving lower per-GPU memory than
any single-GPU technique. FP8ParamManager + DDP is best compared against BF16 DDP.

---

## Demo B: Memory Optimization Matrix — 8B + 70B

### Llama-3.1-8B (single GPU, 3 training steps)

| Config | Peak Mem | vs BF16+LoRA |
|--------|---------|-------------|
| BF16 + LoRA r=16 (AdamW) | 17.07 GB | baseline |
| FP8PM + LoRA r=16 + 8-bit Adam | **13.31 GB** | **-22%** |

### Llama-3.1-8B (8-GPU DDP, 20 steps)

| Config | Peak Mem/GPU | vs BF16 full |
|--------|-------------|-------------|
| BF16 full | 80.5 GB | baseline |
| FP8ParamManager | 31.4 GB | **-61%** |
| BF16 + LoRA r=16 | 18.2 GB | **-77%** |

### Llama-2-70B (8-GPU DDP, 3 steps)

| Config | Peak Mem/GPU | vs BF16+LoRA | Step Time |
|--------|-------------|-------------|-----------|
| BF16 DDP (no LoRA) | OOM (~420 GB) | — | — |
| BF16 + LoRA r=16 DDP | **142.73 GB** | baseline | ~80s |
| FP8PM + 8-bit Adam DDP | **73.31 GB** | **-49%** | ~314s |
| FP8PM + 8-bit Adam + LoRA r=16 DDP | **73.48 GB** | **-49%** | ~324s |

**Key results**:
- FP8PM cuts 70B per-GPU memory **in half** vs BF16+LoRA (73 GB vs 143 GB)
- Both FP8PM configs fit comfortably on MI300X (256 GB) with 190 GB free
- LoRA does not significantly change FP8PM peak memory (73.31 vs 73.48 GB)
- Step time ~4x slower than BF16+LoRA due to serial dequantization overhead

---

## Demo C: Full Lumen Stack — 8B (8-GPU DDP, 20 steps)

Combining FP8ParamManager with other Lumen optimizations:

| Config | Peak Mem/GPU | Notes |
|--------|-------------|-------|
| FP8PM alone | 31.4 GB | baseline for FP8PM |
| FP8PM + FP8 Linear + FP8 Attn + Act Store | **31.4 GB** | No additional savings |

The full stack doesn't add memory savings beyond FP8PM alone because:
- FP8 Linear patches forward but FP8PM already has FP8 weights
- FP8 Attention has negligible effect with gradient checkpointing
- Activation Store doesn't reduce peak memory in this configuration

---

## Demo D: VERL Actor Training — Memory Savings (8-GPU, 3 steps, Llama-3.1-8B)

Benchmarks of VERL's actor training component with different optimization configs.
Environment: `rocm/sgl-dev:v0.5.9-rocm700` container, PyTorch 2.9.0+rocm7.0,
8x MI300X (270 GB HBM each).

### FSDP2 Results (model sharded across 8 GPUs)

| Config | Peak Mem/GPU | vs BF16 FSDP2 | Trainable Params |
|--------|-------------|---------------|-----------------|
| BF16 full finetune (FSDP2) | **12.67 GB** | baseline | 8.03B (100%) |
| BF16 + LoRA r=16 (FSDP2) | **9.17 GB** | **-27.6%** | 41.9M (0.52%) |

### DDP Results (model replicated on each GPU)

| Config | Peak Mem/GPU | vs BF16 DDP | Trainable Params |
|--------|-------------|-------------|-----------------|
| BF16 full finetune (DDP) | **90.08 GB** | baseline | 8.03B (100%) |
| BF16 + LoRA r=16 (DDP) | **21.89 GB** | **-75.7%** | 41.9M (0.52%) |
| FP8PM + LoRA r=16 + 8-bit Adam (DDP) | **16.46 GB** | **-81.7%** | 41.9M (0.52%) |

### Analysis

- **LoRA on FSDP2** saves 27.6% — reduces optimizer states (only LoRA adapter
  params tracked) while FSDP2 shards the frozen base model.
- **FP8PM + LoRA on DDP** saves 81.7% vs BF16 DDP (90.08 → 16.46 GB) by
  combining FP8 weight storage (halved), LoRA (minimal optimizer states),
  and 8-bit Adam (halved optimizer precision).
- **FP8PM + LoRA saves 25% more** than LoRA alone on DDP (16.46 vs 21.89 GB)
  because FP8ParamManager reduces the base model footprint from 16 GB (BF16)
  to ~8 GB (FP8).

### Previous VERL + SGLang Full Pipeline Results

VERL with SGLang rollout in async hybrid mode (actor + rollout share GPU):

| Config | Peak Alloc/GPU | Peak Res/GPU | Step Time | Throughput |
|--------|---------------|-------------|-----------|------------|
| BF16 baseline | **11.77 GB** | 18.91 GB | 14.2s | 790 tok/s |
| FP8 Linear | **12.22 GB** | 18.91 GB | 19.0s | 592 tok/s |

FP8 Linear on VERL+FSDP2: +4% allocated memory, -25% throughput (no savings).

---

## Compatibility Matrix

| Feature | DDP | FSDP1 | FSDP2 |
|---------|-----|-------|-------|
| FP8ParamManager | yes | no (mixed dtype) | no (NCCL FP8) |
| FP8 Linear | yes | yes | yes |
| LoRA | yes | caution (dtype) | caution (DTensor) |
| FP8PM + LoRA | yes (fixed) | no | no |
| 8-bit Adam | yes | yes | yes |

---

## VERL Backend Combinations

Lumen supports VERL training via `lumen.rl.verl.verl_entry` (monkey-patching FSDP/Megatron
workers) and via `benchmark_verl_actor_memory.py` (direct torchrun benchmark):

| Training Backend | Rollout Backend | Launch Script | Status |
|-----------------|----------------|---------------|--------|
| FSDP2 | SGLang | `run_grpo_fsdp2.sh` | **Tested** (BF16, FP8 Linear) |
| FSDP2 | (actor only) | `benchmark_verl_actor_memory.py` | **Tested** (BF16, LoRA) |
| DDP | (actor only) | `benchmark_verl_actor_memory.py` | **Tested** (BF16, LoRA, FP8PM+LoRA) |
| FSDP2 | vLLM | `run_grpo_fsdp2_vllm.sh` | Blocked (vLLM-ROCm) |
| Megatron | SGLang | `run_grpo_megatron_sglang.sh` | Not tested |

All scripts are in `examples/rl/verl/` and support:
- `LUMEN_FP8=1` to enable FP8 Linear
- `FP8_PARAM_MANAGER=1` to enable FP8ParamManager (DDP only)
- `USE_8BIT_ADAM=1` to enable 8-bit Adam

### VERL Actor Memory Benchmark

The standalone actor benchmark (`benchmark_verl_actor_memory.py`) isolates the
training memory footprint without SGLang/vLLM rollout overhead:

```bash
torchrun --nproc_per_node=8 examples/rl/verl/benchmark_verl_actor_memory.py \
    --config bf16        # BF16 full finetune (FSDP2)
    --config lora        # BF16 + LoRA r=16 (FSDP2)
    --config fp8pm_lora  # FP8PM + LoRA + 8-bit Adam (DDP)
    --config bf16_ddp    # BF16 full finetune (DDP)
    --config lora_ddp    # BF16 + LoRA r=16 (DDP)
```

---

## Limitations and Future Work

1. **FP8ParamManager freezes linear weights**: Only ~1.3% of parameters (norms,
   embeddings, lm_head) are actually trained. Full fine-tuning requires FSDP
   integration via `FP8CommTensor`.

2. **FSDP compatibility**: Requires implementing FP8-aware `fsdp_pre_all_gather` /
   `fsdp_post_all_gather` hooks to quantize before communication and dequantize
   after, enabling FSDP to shard FP8 parameters.

3. **LoRA + FP8PM**: Fully supported on DDP with two fixes:
   - PEFT adapter dtype: re-cast LoRA params to BF16 after `get_peft_model`
   - Dequant memory leak: replaced `F.linear` with `_FP8LinearFunc` custom
     autograd that saves FP8 weight + scale (not full BF16 copy) for backward,
     reducing 70B peak from 210 GB to 73 GB

4. **FP8 Linear overhead**: AITER FP8 GEMM kernels on MI300X are untuned,
   causing 10-30% slowdown. Kernel optimization would make compute-path FP8
   beneficial.

---

## How to Reproduce

### TRL GRPO (Demo A/B/C)

```bash
# BF16 baseline (DDP)
ACCEL_CONFIG=examples/rl/trl/accelerate/ddp.yaml \
MODEL_NAME=/path/to/llama-3.1-8b NUM_PROCESSES=8 MAX_STEPS=20 \
bash examples/rl/trl/run_grpo_fsdp.sh 1

# FP8ParamManager (DDP)
ACCEL_CONFIG=examples/rl/trl/accelerate/ddp.yaml \
FP8_PARAM_MANAGER=1 MODEL_NAME=/path/to/llama-3.1-8b NUM_PROCESSES=8 MAX_STEPS=20 \
bash examples/rl/trl/run_grpo_fsdp.sh 1

# FP8ParamManager + 8-bit Adam (DDP)
ACCEL_CONFIG=examples/rl/trl/accelerate/ddp.yaml \
FP8_PARAM_MANAGER=1 USE_8BIT_ADAM=1 MODEL_NAME=/path/to/llama-3.1-8b NUM_PROCESSES=8 \
bash examples/rl/trl/run_grpo_fsdp.sh 1

# 70B FP8PM + 8-bit Adam + LoRA (DDP, 8-GPU)
ACCEL_CONFIG=examples/rl/trl/accelerate/ddp.yaml \
FP8_PARAM_MANAGER=1 USE_8BIT_ADAM=1 LORA_RANK=16 LORA_DROPOUT=0.0 \
MODEL_NAME=/path/to/llama-2-70b NUM_PROCESSES=8 MAX_STEPS=5 \
bash examples/rl/trl/run_grpo_fsdp.sh 1
```

### VERL Actor Memory Benchmark (Demo D)

```bash
# BF16 baseline (FSDP2)
torchrun --nproc_per_node=8 examples/rl/verl/benchmark_verl_actor_memory.py \
    --config bf16 --steps 3 --micro_bs 2 --seq_len 256

# LoRA (FSDP2) — 27.6% memory savings
torchrun --nproc_per_node=8 examples/rl/verl/benchmark_verl_actor_memory.py \
    --config lora --steps 3 --micro_bs 2 --seq_len 256

# FP8PM + LoRA + 8-bit Adam (DDP) — 81.7% memory savings
torchrun --nproc_per_node=8 examples/rl/verl/benchmark_verl_actor_memory.py \
    --config fp8pm_lora --steps 3 --micro_bs 2 --seq_len 256

# BF16 DDP baseline (for FP8PM comparison)
torchrun --nproc_per_node=8 examples/rl/verl/benchmark_verl_actor_memory.py \
    --config bf16_ddp --steps 3 --micro_bs 2 --seq_len 256
```

### VERL Full Pipeline (FSDP2 + SGLang rollout)

```bash
# FSDP2 + SGLang — BF16 baseline
MODEL_NAME=/path/to/llama-3.1-8b MAX_STEPS=10 \
bash examples/rl/verl/run_grpo_fsdp2.sh

# FSDP2 + SGLang — FP8 Linear
LUMEN_FP8=1 MODEL_NAME=/path/to/llama-3.1-8b MAX_STEPS=10 \
bash examples/rl/verl/run_grpo_fsdp2.sh
```

### Model-Specific Benchmarks

```bash
# Llama-2-70B FSDP (see benchmark/llama-2-70b-chat/)
accelerate launch --config_file fsdp_8gpu.yaml --num_processes 8 \
  examples/rl/trl/benchmark/llama-2-70b-chat/train_grpo_bf16.py --max-steps 20

# Qwen2.5-32B FSDP (see benchmark/qwen2.5-32b-instruct/)
accelerate launch --config_file fsdp_8gpu.yaml --num_processes 8 \
  examples/rl/trl/benchmark/qwen2.5-32b-instruct/train_grpo_bf16.py --max-steps 10

# FP8 Memory benchmark (single GPU, process-isolated)
cd examples/rl/trl/benchmark
python3 test_fp8_memory.py --model /path/to/llama-3.1-8b --configs all --steps 3
```
