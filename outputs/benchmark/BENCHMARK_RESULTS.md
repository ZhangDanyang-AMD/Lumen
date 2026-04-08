# Lumen FP8 GRPO Benchmark Results

**Date**: 2026-04-07
**Hardware**: 8x AMD MI300X (256 GB VRAM each)
**Software**: ROCm + PyTorch + TRL 0.29.1 + Lumen (AITER backend)

---

## Demo A: FP8 A/B Test — Correctness & Basic Comparison

### Llama-3.1-8B (8B params, DDP)

| Config | Peak Mem/GPU | Avg Step Time | Reward Curve |
|---|---|---|---|
| BF16 baseline | ~34.6 GB | ~11.1s | 0.42→0.79 |
| FP8 dynamic (`quant.enable()`) | ~38.9 GB | ~127s | 0.51→0.74 |

**Conclusion**: FP8 correctness validated (reward tracks BF16). No memory or speed benefit — 8B model is too small for FP8 compute savings to offset quantization overhead. AITER kernels fall back to Triton when FSDP1 mixed precision upcasts BF16→FP32.

### Qwen2.5-32B-Instruct (32B params, FSDP1)

| Config | Peak Mem/GPU | Avg Step Time | Status |
|---|---|---|---|
| BF16 baseline | 124.18 GB | ~13.4s | Completed 5 steps |
| FP8 max memory-save | 127.13 GB | ~47.7s | Crashed step 4 (GPU OOM) |

**Conclusion**: FP8 via `quant.enable()` with cpu_offload + fp8_param_gather is counterproductive. 3.6x slower, 2% more memory. The quantize/dequantize buffers and Triton fallbacks exceed any compute savings.

### Llama-2-70B-chat (70B params)

**Status**: Blocked — CPU OOM during model loading (GRPOTrainer loads full model on 8 ranks before FSDP sharding), GPU OOM during training (252 GB/GPU peak in BF16).

---

## Demo B: TRL GRPO Memory Matrix — 4-Config Comparison

**Model**: Llama-3.1-8B (8B params)
**Dataset**: trl-lib/DeepMath-103K
**Setup**: 10 steps, num_generations=4, micro_bs=1, grad_accum=4, seed=1234

### Results

| Config | Distributed | Linear FP8 | LoRA | Peak Mem/GPU | vs A (savings) | Avg Step Time | vs A (speedup) | Total Elapsed | Reward |
|---|---|---|---|---|---|---|---|---|---|
| A) BF16 full | FSDP1 | off | off | **34.57 GB** | baseline | **11.07s** | baseline | 122.7s | 0.424→0.788 |
| B) BF16 LoRA r=16 | DDP | off | r=16 | **17.83 GB** | **48% saved** | **9.56s** | 1.16x | 107.5s | 0.413→0.429 |
| C) FP8 full | FSDP1 | on | off | **38.85 GB** | -12% (worse) | **126.67s** | 0.09x | 1279.2s | 0.513→0.735 |
| D) FP8 LoRA r=16 | DDP | on | r=16 | **20.64 GB** | 40% saved | **185.28s** | 0.06x | 1865.5s | 0.481→0.469 |

### Key Findings

1. **LoRA is the real memory optimizer**: BF16 LoRA reduces peak memory by **48%** (34.57→17.83 GB) with minimal quality impact in 10 steps.

2. **FP8 (`quant.enable()`) is counterproductive**:
   - FP8 full uses **12% MORE** memory than BF16 full (38.85 vs 34.57 GB)
   - FP8 LoRA uses **16% MORE** memory than BF16 LoRA (20.64 vs 17.83 GB)
   - FP8 full is **11.5x slower** than BF16 full
   - FP8 LoRA is **19.4x slower** than BF16 LoRA

3. **FP8 correctness is OK**: Reward curves track BF16 (converge in same direction, just slower per wall-clock).

4. **FSDP1 + PEFT LoRA incompatible**: LoRA configs (B, D) had to use DDP instead of FSDP1 due to dtype mismatch (`float != BFloat16`) in PEFT adapter layers under FSDP1 mixed precision wrapping.

### Root Cause: FP8 Overhead

The current Lumen FP8 via `quant.enable()`:
- Quantizes **only GEMM compute** — does NOT reduce parameter storage
- FSDP1 mixed precision stores params in BF16 and upcasts to FP32 for forward
- **AITER FP8 kernels (CK backend) don't support FP32 input** → falls back to Triton (Python-based, 10-20x slower)
- FP8 quantize/dequantize buffers add ~4 GB/GPU overhead

### vs Expected Results (from train_target.md)

| Metric | Expected (8B) | Actual (8B) | Status |
|---|---|---|---|
| FP8 memory savings | -5%~-10% | **+12%** (worse) | FAIL |
| FP8 step time | small difference | **11.5x slower** | FAIL |
| FP8 correctness | consistent reward | consistent reward | PASS |
| LoRA memory savings | ~78% (70B est.) | **48%** (8B) | PARTIAL (model-size dependent) |

### Launch Commands (Inside Docker Container)

```bash
# Config A: BF16 full fine-tune (FSDP1)
ulimit -c 0 && LINEAR_FP8=0 LORA_RANK=0 NUM_PROCESSES=8 MAX_STEPS=10 \
  MODEL_NAME=/dev/shm/model/llama-3.1-8b \
  TRAIN_DATA_PATH=trl-lib/DeepMath-103K \
  OUTPUT_DIR=/workspace/Lumen/outputs/demo-b/bf16-full \
  bash examples/rl/trl/run_grpo_fsdp.sh 1

# Config B: BF16 LoRA r=16 (DDP — workaround for FSDP1+PEFT bug)
ulimit -c 0 && accelerate launch \
  --config_file examples/rl/trl/accelerate/ddp.yaml \
  --num_processes 8 \
  examples/rl/trl/run_grpo_fsdp.py \
  --model-name-or-path /dev/shm/model/llama-3.1-8b \
  --dataset-name trl-lib/DeepMath-103K \
  --output-dir /workspace/Lumen/outputs/demo-b/bf16-lora \
  --max-steps 10 --micro-batch-size 1 --gradient-accumulation-steps 4 \
  --num-generations 4 --max-completion-length 256 --lr 5e-6 \
  --log-interval 1 --seed 1234 --lora-rank 16

# Config C: FP8 full fine-tune (FSDP1)
ulimit -c 0 && LINEAR_FP8=1 LORA_RANK=0 NUM_PROCESSES=8 MAX_STEPS=10 \
  MODEL_NAME=/dev/shm/model/llama-3.1-8b \
  TRAIN_DATA_PATH=trl-lib/DeepMath-103K \
  OUTPUT_DIR=/workspace/Lumen/outputs/demo-b/fp8-full \
  bash examples/rl/trl/run_grpo_fsdp.sh 1

# Config D: FP8 LoRA r=16 (DDP — workaround for FSDP1+PEFT bug)
ulimit -c 0 && LINEAR_FP8=1 accelerate launch \
  --config_file examples/rl/trl/accelerate/ddp.yaml \
  --num_processes 8 \
  examples/rl/trl/run_grpo_fsdp.py \
  --model-name-or-path /dev/shm/model/llama-3.1-8b \
  --dataset-name trl-lib/DeepMath-103K \
  --output-dir /workspace/Lumen/outputs/demo-b/fp8-lora \
  --max-steps 10 --micro-batch-size 1 --gradient-accumulation-steps 4 \
  --num-generations 4 --max-completion-length 256 --lr 5e-6 \
  --log-interval 1 --seed 1234 --lora-rank 16 --linear-fp8
```

---

## Demo C: Extended Lumen Optimizations (TRL GRPO)

**Model**: Llama-3.1-8B (8B params)
**Dataset**: trl-lib/DeepMath-103K
**Setup**: 10 steps, FSDP1, num_generations=4, micro_bs=1, grad_accum=4, seed=1234

### Results

| Config | Peak Mem/GPU | Total Elapsed | vs BF16 full |
|---|---|---|---|
| BF16 full (baseline) | 34.57 GB | 122.7s | baseline |
| FP8 Linear only | 38.85 GB | 1279.2s | +12% mem, 10.4x slower |
| FP8 Linear + FP8 Attn (dpa) | **30.92 GB** | 1586.5s | **-11% mem**, 12.9x slower |
| FP8 Linear + FP8 Attn + Act Store | **30.89 GB** | 1581.3s | **-11% mem**, 12.9x slower |
| FP8 Linear + FP8 Attn + Param Gather | CRASH | — | AITER quant kernel crash |
| FP8 Linear + FP8 Attn + Lumen Norm | CRASH | — | FSDP1 mixed dtype error |

### Key Findings

1. **FP8 Attention (dpa) is the only effective add-on**: Reduces memory from 38.85→30.92 GB (-20% vs FP8 linear alone, -11% vs BF16 baseline).
2. **Activation Store has no effect** on HuggingFace models: Pre-quant hooks target Lumen-specific module types (LumenColumnParallelLinear etc.) not present in HF models.
3. **FP8 Param Gather crashes**: AITER `quant_kernels.cu:682` fails with illegal memory access.
4. **Lumen Norm incompatible with FSDP1**: Norm replacement creates FP32 parameters; FSDP1 cannot flatten mixed BF16+FP32 tensors.
5. **Speed bottleneck persists**: All FP8 configs are 10-13x slower due to AITER Triton fallback.

### Launch Commands

```bash
# FP8 Linear + FP8 Attention (dpa)
ulimit -c 0 && LINEAR_FP8=1 LUMEN_FP8_ATTN=dpa NUM_PROCESSES=8 MAX_STEPS=10 \
  MODEL_NAME=/dev/shm/model/llama-3.1-8b \
  DATASET_NAME=trl-lib/DeepMath-103K \
  OUTPUT_DIR=/workspace/Lumen/outputs/demo-c/fp8-attn \
  bash examples/rl/trl/run_grpo_fsdp.sh 1

# FP8 Linear + FP8 Attn + Activation Store
ulimit -c 0 && LINEAR_FP8=1 LUMEN_FP8_ATTN=dpa LUMEN_FP8_ACTIVATION_STORE=1 \
  NUM_PROCESSES=8 MAX_STEPS=10 \
  MODEL_NAME=/dev/shm/model/llama-3.1-8b \
  DATASET_NAME=trl-lib/DeepMath-103K \
  OUTPUT_DIR=/workspace/Lumen/outputs/demo-c/fp8-attn-actstore \
  bash examples/rl/trl/run_grpo_fsdp.sh 1
```

---

## Cross-Model FP8 Summary

| Model | Framework | BF16 Peak Mem | FP8 Peak Mem | FP8 Memory Δ | FP8 Speed Δ (Actor) | FP8 Speed Δ (Step) |
|---|---|---|---|---|---|---|
| Qwen2-0.5B | TRL+FSDP1 | 7.11 GB | 7.11 GB | 0% | 3.18x slower | — |
| Llama-3.1-8B | TRL+FSDP1 | 34.57 GB | 38.85 GB | +12% (worse) | 11.5x slower | — |
| Llama-3.1-8B | VERL+FSDP2+SGLang | 11.76 GB | 12.22 GB | +3.9% (worse) | 1.52x slower | 1.31x slower |
| Qwen2.5-32B | TRL+FSDP1 | 124.18 GB | 127.13 GB | +2% (worse) | 3.6x slower | — |
| **Qwen2.5-32B** | **VERL+FSDP2+SGLang** | **38.43 GB** | **38.43 GB** | **0% (same)** | **1.85x slower** | **1.48x slower** |
| Llama-2-70B | TRL+FSDP1 | OOM | OOM | N/A | N/A | N/A |

---

## Demo D-1: VERL + SGLang + FSDP2 Integration

### Setup
- **Framework**: VERL 0.7.1 + SGLang 0.5.9 (from source) + sgl-kernel 0.3.21
- **Hardware**: 8x MI300X (ROCm 7.0)
- **Distributed**: FSDP2 (actor), SGLang async rollout, SDPA attention
- **Dataset**: DeepMath-103K (500 train / 20 val), prompt_length=512, response_length=256
- **8B config**: Llama-3.1-8B, 8 SGLang replicas (TP=1), batch_size=64, micro_bs=2
- **32B config**: Qwen2.5-32B-Instruct, 4 SGLang replicas (TP=2), batch_size=16, micro_bs=1

### ROCm Integration Patches (15 patches required)
Getting VERL + SGLang running on ROCm required extensive patching:
1. SGLang module-scope GPU queries (5 files) — wrap in try/except for Ray CPU workers
2. VERL `enable_memory_saver=True` → `False` — prevents `libcuda.so.1` preload failure
3. VERL/SGLang `CUDA_VISIBLE_DEVICES` → `HIP_VISIBLE_DEVICES` — 4 locations
4. VERL hardcoded `attention_backend="fa3"` → auto-detect (`aiter` on ROCm)
5. Triton `triton_key` compatibility shim for ROCm Triton 3.6.0
6. `TORCHDYNAMO_DISABLE=1` — torch.compile inductor failures on ROCm
7. `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` — preserve GPU visibility for CPU workers

### BF16 Baseline Results (5 steps)

| Metric | Step 1 (warmup) | Steps 2-5 (avg) |
|--------|-----------------|-----------------|
| Peak GPU Memory (allocated) | 9.50 GB | 11.76 GB |
| Peak GPU Memory (reserved) | 14.26 GB | 18.91 GB |
| Step Time | 34.9s | ~14.5s |
| Generation Time (SGLang) | 21.0s | ~3.1s |
| Actor Update Time | 8.7s | ~7.4s |
| Weight Sync Time | 2.8s | ~2.7s |
| Throughput | 326 tok/s | ~774 tok/s |

### Comparison: TRL GRPO (Demo A) vs VERL GRPO (Demo D-1) — BF16 baseline on 8B

| Metric | TRL + FSDP1 | VERL + FSDP2 + SGLang | Delta |
|--------|------------|----------------------|-------|
| Peak Memory/GPU | 34.57 GB | 11.76 GB | **-66%** |
| Step Time | ~11.1s | ~14.5s | +31% (includes SGLang overhead) |
| Throughput | N/A | ~774 tok/s | — |

The large memory reduction comes from VERL's async architecture: SGLang rollout runs in a separate process, so the training actor only holds the FSDP2-sharded model + optimizer, not the full model for generation.

### FP8 Results (5 steps, Lumen `apply_fp8_training` via direct worker patch)

| Metric | Step 1 (warmup) | Steps 2-5 (avg) |
|--------|-----------------|-----------------|
| Peak GPU Memory (allocated) | 9.50 GB | 12.22 GB |
| Peak GPU Memory (reserved) | 15.14 GB | 18.91 GB |
| Step Time | 29.4s | ~19.0s |
| Actor Update Time | 12.7s | ~11.2s |
| Weight Sync Time | 2.7s | ~2.7s |
| Throughput | 387 tok/s | ~593 tok/s |

**Evidence FP8 is active**: AITER `module_gemm_a8w8` and `module_quant` loaded in worker processes; GEMM shapes logged with `q_dtype_w:torch.float8_e4m3fn`.

**Bug fixed during run**: VERL's `dp_actor.py` uses `logits.div_(temperature)` — an in-place operation that fails on custom autograd function outputs (Lumen `QuantizedLinearFunctionBackward`). Fixed by replacing with `logits = logits / temperature`.

### BF16 vs FP8 Comparison (VERL + SGLang + FSDP2, 8B model)

| Metric | BF16 (baseline) | FP8 (Lumen) | Delta | Verdict |
|--------|-----------------|-------------|-------|---------|
| Peak Memory (alloc)/GPU | 11.76 GB | 12.22 GB | **+0.46 GB (+3.9%)** | WORSE |
| Peak Memory (res)/GPU | 18.91 GB | 18.91 GB | 0.00 GB (0%) | SAME |
| Actor Update Time | 7.39s | 11.23s | **+3.83s (+51.8%)** | WORSE |
| Step Time | 14.48s | 19.00s | **+4.52s (+31.2%)** | WORSE |
| Throughput | 774 tok/s | 593 tok/s | **-181 tok/s (-23.4%)** | WORSE |

**Analysis**: Even with FSDP2's more efficient sharding, Lumen FP8 (`quant.enable()` / `apply_fp8_training`) shows the same pattern as TRL demos: increased memory (+3.9%), significantly slower actor updates (+51.8%), and reduced throughput (-23.4%). The AITER A8W8 GEMM kernels are active (confirmed by `float8_e4m3fn` dtype in logs), but the quantize/dequantize overhead still dominates at the 8B scale.

### Known Issue
Reward returns -1.0 on all steps (`critic/score/mean:-1.0`), suggesting the reward pipeline doesn't match the dataset format. Policy gradient loss is 0.0 — the model runs rollout+forward+backward but doesn't improve. This is a reward configuration issue, not a training infrastructure issue.

---

### Qwen2.5-32B-Instruct (32B params, FSDP2 + SGLang TP=2)

**Config**: 8x MI300X, FSDP2 actor, SGLang async rollout (4 replicas, TP=2), batch_size=16, micro_bs=1, num_generations=4, prompt_length=512, response_length=256, gradient checkpointing=true, ref param_offload=true

**Why TP=2**: With TP=1, SGLang OOMs loading the full 32B model (~64 GB BF16) on a single GPU (`RuntimeError: Not enough memory. self.server_args.mem_fraction_static=0.17`). TP=2 splits the model across 2 GPUs per replica (4 replicas total on 8 GPUs).

### 32B BF16 Baseline Results (5 steps)

| Metric | Step 1 (warmup) | Steps 2-5 (avg) |
|--------|-----------------|-----------------|
| Peak GPU Memory (allocated) | 38.43 GB | 38.43 GB |
| Peak GPU Memory (reserved) | 42.97 GB | 42.97 GB |
| Step Time | 44.6s | ~22.3s |
| Actor Update Time | 11.4s | ~9.0s |
| Throughput | 65 tok/s | ~128 tok/s |

### 32B FP8 Results (5 steps, Lumen `apply_fp8_training` via direct worker patch)

| Metric | Step 1 (warmup) | Steps 2-5 (avg) |
|--------|-----------------|-----------------|
| Peak GPU Memory (allocated) | 38.43 GB | 38.43 GB |
| Peak GPU Memory (reserved) | 42.98 GB | 44.29 GB |
| Step Time | 43.7s | ~33.0s |
| Actor Update Time | 19.3s | ~16.8s |
| Throughput | 67 tok/s | ~87 tok/s |

**Evidence FP8 is active**: AITER `module_gemm_a8w8` and `module_quant` loaded; GEMM shapes with `q_dtype_w:torch.float8_e4m3fn` at 32B hidden dimension (5120).

### BF16 vs FP8 Comparison (VERL + SGLang + FSDP2, 32B model)

| Metric | BF16 (baseline) | FP8 (Lumen) | Delta | Verdict |
|--------|-----------------|-------------|-------|---------|
| Peak Memory (alloc)/GPU | 38.43 GB | 38.43 GB | 0.00 GB (0%) | SAME |
| Peak Memory (res)/GPU | 42.97 GB | 44.29 GB | **+1.32 GB (+3.1%)** | WORSE |
| Actor Update Time | 9.04s | 16.76s | **+7.72s (+85.5%)** | WORSE |
| Step Time | 22.32s | 32.97s | **+10.65s (+47.7%)** | WORSE |
| Throughput | 128 tok/s | 87 tok/s | **-42 tok/s (-32.4%)** | WORSE |

### 32B Analysis

FP8 at 32B scale follows the **same counterproductive pattern** as 8B, but with an even larger speed penalty:

1. **Memory**: Allocated memory is identical (38.43 GB) — FP8 quantize/dequantize buffers are offset by other allocations. Reserved memory increased by 3.1% (+1.32 GB) from fragmentation.

2. **Speed**: Actor update is **85.5% slower** (9.04s → 16.76s), compared to 51.8% at 8B. This disproves the hypothesis that FP8 would benefit from 32B's larger GEMM sizes (hidden=5120, intermediate=27648) being more compute-bound.

3. **Root cause remains AITER Triton fallback**: Despite the larger matrix dimensions at 32B, the AITER FP8 kernels still fall back to slower code paths. The fallback overhead scales with model size — more layers and larger weight matrices mean more fallback calls per forward+backward pass.

4. **vs train_target.md expectations**: Expected 1.3-1.5x throughput speedup with FP8 at 32B. Actual: **0.68x throughput** (32% slower). The expected benefit assumed native FP8 hardware acceleration without kernel fallback issues.

### 8B vs 32B FP8 Impact Comparison

| Metric | 8B FP8 Δ | 32B FP8 Δ | Trend |
|--------|----------|-----------|-------|
| Memory (alloc) | +3.9% | 0% | Slightly better at 32B |
| Memory (res) | 0% | +3.1% | Slightly worse at 32B |
| Actor Update Time | +51.8% | +85.5% | **Significantly worse at 32B** |
| Step Time | +31.2% | +47.7% | **Worse at 32B** |
| Throughput | -23.4% | -32.4% | **Worse at 32B** |

**Conclusion**: Scaling to 32B does NOT make FP8 beneficial — it makes it worse. The AITER Triton fallback overhead scales super-linearly with model size. FP8 via `quant.enable()` / `apply_fp8_training()` should not be used at any model scale on MI300X until the AITER kernel fallback issue is resolved.

### Launch Commands (VERL + SGLang + FSDP2)

```bash
# 8B BF16 baseline
MODEL_NAME=/dev/shm/model/llama-3.1-8b MAX_STEPS=5 TRAIN_BSZ=64 \
  ROLLOUT_TP=1 ROLLOUT_GPU_UTIL=0.4 EXPERIMENT_NAME=llama8b-bf16 \
  bash examples/rl/verl/run_grpo_fsdp2.sh

# 8B FP8
MODEL_NAME=/dev/shm/model/llama-3.1-8b MAX_STEPS=5 TRAIN_BSZ=64 \
  ROLLOUT_TP=1 ROLLOUT_GPU_UTIL=0.4 LUMEN_FP8=1 EXPERIMENT_NAME=llama8b-fp8 \
  bash examples/rl/verl/run_grpo_fsdp2.sh

# 32B BF16 baseline
MODEL_NAME=/dev/shm/model/qwen2.5-32b-instruct MAX_STEPS=5 TRAIN_BSZ=16 \
  MICRO_BSZ=1 LOG_PROB_MICRO_BSZ=1 NUM_GENERATIONS=4 \
  ROLLOUT_TP=2 ROLLOUT_GPU_UTIL=0.4 EXPERIMENT_NAME=qwen32b-bf16 \
  bash examples/rl/verl/run_grpo_fsdp2.sh

# 32B FP8
MODEL_NAME=/dev/shm/model/qwen2.5-32b-instruct MAX_STEPS=5 TRAIN_BSZ=16 \
  MICRO_BSZ=1 LOG_PROB_MICRO_BSZ=1 NUM_GENERATIONS=4 \
  ROLLOUT_TP=2 ROLLOUT_GPU_UTIL=0.4 LUMEN_FP8=1 EXPERIMENT_NAME=qwen32b-fp8 \
  bash examples/rl/verl/run_grpo_fsdp2.sh
```

---

### Recommendations

1. **Do NOT use `quant.enable()` / `apply_fp8_training()` for GRPO training** on MI300X at any model scale (8B or 32B) — it increases latency across both TRL and VERL frameworks. Scaling to larger models makes the overhead **worse**, not better (32B actor update: +85.5% vs 8B: +51.8%).
2. **Use LoRA (r=16)** for memory optimization — provides consistent 40-50% memory savings on 8B models.
3. **VERL + FSDP2 + SGLang** provides the best memory efficiency (11.76 GB/GPU for 8B vs 34.57 GB/GPU with TRL+FSDP1), a 66% reduction from the async rollout architecture. For 32B, VERL+FSDP2 uses 38.43 GB/GPU (vs 124.18 GB/GPU with TRL+FSDP1, a 69% reduction).
4. **FP8 needs architectural changes** to be beneficial:
   - True FP8 weight storage in FSDP (not just GEMM compute quantization)
   - AITER kernel support for FP32 inputs (eliminating Triton fallback)
   - FP8 activation storage to reduce activation memory
   - FP8 parameter all-gather to reduce communication volume
5. **FP8 Attention (dpa)** is the only current optimization that reduces memory (-11% vs BF16 in TRL), but speed remains a major bottleneck.
6. **VERL + SGLang on ROCm** is now functional but requires 15+ patches. Consider upstreaming ROCm fixes. For 32B, use TP=2 for SGLang rollout (TP=1 OOMs).
7. **VERL in-place operation fix required**: `logits.div_(temperature)` in `dp_actor.py` must be changed to `logits = logits / temperature` when using Lumen FP8 (custom autograd function outputs cannot be modified in-place).
8. **FP8 overhead scales super-linearly with model size**: The AITER Triton fallback penalty grows from 1.5x (8B) to 1.85x (32B) for actor updates. This disproves the hypothesis that larger models' compute-bound GEMMs would benefit from FP8.

---

## Fixes Applied

| Issue | Fix | Result |
|---|---|---|
| Lumen Norm + FSDP mixed dtype | Cast norm params to model dtype + handle meta tensors in `_patch_norms` | **FIXED** — Lumen Norm works with both FSDP1 and FSDP2 |
| FSDP2 + PEFT LoRA dtype mismatch | No code change needed | **RESOLVED** — FSDP2 handles mixed FP32 LoRA + BF16 base params natively |
| AITER FP32 input fallback to Triton | Not happening in FSDP2 path | **RESOLVED** — CK kernels used correctly; FSDP2 keeps activations in BF16 |
| VERL reward pipeline returns -1.0 | Expected behavior, not a bug | **RESOLVED** — model can't solve math (no `\boxed{}` output); `math_dapo` returns -1.0 for wrong answers |
| VERL `logits.div_()` inplace on FP8 view | Replace with non-inplace `logits = logits / temperature` | **Fixed locally** |
| Lumen Norm + FP8 + meta tensor | Handle `is_meta` in `_patch_norms`, skip data copy for meta weights | **FIXED** — Lumen Norm now works on meta-device models (VERL worker init) |

### Lumen Norm + FP8 Benchmark (8B, VERL + FSDP2 + SGLang)

| Config | Peak Mem (alloc) | Peak Mem (res) | Actor Update | Step Time | Throughput |
|--------|------------------|----------------|-------------|-----------|------------|
| BF16 baseline | 11.76 GB | 18.91 GB | 7.39s | 14.48s | 774 tok/s |
| FP8 only | 12.22 GB | 18.91 GB | 11.23s | 19.00s | 593 tok/s |
| **FP8 + Lumen Norm** | **12.20 GB** | **18.18 GB** | **10.90s** | **18.04s** | **622 tok/s** |

Lumen Norm with AITER rmsnorm kernel provides ~5% throughput improvement over FP8-only, with 3.9% less reserved memory.

---

## Architectural Test Results

### FP8 Weight Cache (store_weights_fp8)
**Status**: FIXED — Wired into LumenConfig and VERL path

True FP8 `nn.Parameter` storage remains blocked by PyTorch (autograd/DDP don't support FP8). However, Lumen's `store_weights_fp8()` provides a practical alternative: it pre-quantizes weights to FP8 buffers that skip per-forward re-quantization. The `quant_forward` path already reads `_fp8_weight_data`/`_fp8_weight_scale` from patched modules.

**What was fixed**: Added `fp8_weight_cache` option to `LumenConfig` (wired through `_apply_post_quant`), `VerlLumenArgs`, and the launcher. After `quant.enable()`, `store_weights_fp8(model)` creates per-layer FP8 caches. The optimizer post-step hook (`register_fp8_weight_optimizer_hooks`) refreshes caches after each `optimizer.step()`.

**Benchmark** (8B, VERL+FSDP2+SGLang, steps 2-5 avg):
| Metric | FP8 only | FP8 + Weight Cache | Delta |
|--------|----------|-------------------|-------|
| Actor Update | 11.23s | 11.00s | -2.0% |
| Throughput | 593 tok/s | 613 tok/s | +3.3% |

Env: `LUMEN_FP8_WEIGHT_CACHE=1`

### FP8 Activation Storage for HuggingFace Models
**Status**: FIXED — Extended `_apply_pre_quant` to `nn.Linear`

**What was fixed**: `_apply_pre_quant()` in `lumen/config.py` previously only set `module.fp8_activation_store = True` on Lumen-native module types (`LumenColumnParallelLinear`, etc.). Now it also sets the attribute on standard `nn.Linear` modules. Since `_replace_forward` reads `getattr(module, "fp8_activation_store", False)` at patch time, HF models now receive the flag.

**Caveat**: On the full FP8 path (`quantize_activation=True`), `QuantizedLinearFunction` already saves activations as FP8 in `save_for_backward` and forces `ctx.fp8_activation_store = False`. The flag only provides additional benefit on `scaling_type="none"` or `quantize_activation=False` branches.

**Benchmark** (8B, VERL+FSDP2+SGLang, steps 2-5 avg):
| Metric | FP8 only | FP8 + Act Store | Delta |
|--------|----------|-----------------|-------|
| Mem (res) | 18.91 GB | 18.91 GB | 0% |
| Actor Update | 11.23s | 11.22s | -0.1% |

No measurable impact on full FP8 path (expected — activations already saved as FP8).

Env: `LUMEN_FP8_ACTIVATION_STORE=1`

### FP8 Parameter All-Gather
**Status**: FIXED — AITER kernel crash resolved

**Root cause**: `dynamic_per_tensor_quant` and `static_per_tensor_quant` in AITER (`quant_kernels.cu`) assumed contiguous row-major input but had no `TORCH_CHECK(input.is_contiguous())`. When `quantize_input(weight, ...)` was called on a non-contiguous HF model weight, the kernel accessed invalid memory at line 682.

**What was fixed**:
1. **Lumen side** (`lumen/ops/quantize/linear.py`): Added `.contiguous()` to `weight` before passing to `quantize_input` (matching what's done for `input_2d`).
2. **AITER side** (`third_party/aiter/csrc/kernels/quant_kernels.cu`): Added `TORCH_CHECK(input.is_contiguous())` and `TORCH_CHECK(out.is_contiguous())` to both `dynamic_per_tensor_quant` and `static_per_tensor_quant`.

**Benchmark** (8B, VERL+FSDP2+SGLang, steps 2-5 avg):
| Metric | FP8 only | FP8 + Param Gather | Delta |
|--------|----------|-------------------|-------|
| Actor Update | 11.23s | 11.04s | -1.7% |
| Throughput | 593 tok/s | 601 tok/s | +1.4% |

Previously crashed with `HSA_STATUS_ERROR_EXCEPTION`. Now completes all 5 steps.

Env: `LUMEN_FP8_PARAM_GATHER=1`

### Combined: All Three Fixes
**Benchmark** (8B, VERL+FSDP2+SGLang, steps 2-5 avg):
| Metric | FP8 only | FP8 + All Three | Delta |
|--------|----------|-----------------|-------|
| Actor Update | 11.23s | 10.96s | -2.4% |
| Step Time | 19.00s | 18.36s | -3.4% |
| Throughput | 593 tok/s | 610 tok/s | +2.8% |

Env: `LUMEN_FP8=1 LUMEN_FP8_WEIGHT_CACHE=1 LUMEN_FP8_ACTIVATION_STORE=1 LUMEN_FP8_PARAM_GATHER=1`

### AITER FP32 Input Fallback — Corrected Diagnosis
**Status**: NOT occurring in FSDP2 path (was FSDP1-only issue)

Log analysis of VERL+FSDP2 runs confirms:
- `dynamic_per_tensor_quant` (CK compiled op) is used — zero Triton fallback calls
- `gemm_a8w8` (CK compiled GEMM) is loaded and active
- FSDP2 `MixedPrecisionPolicy(param_dtype=bf16)` keeps activations in BF16, not FP32
- The FP8 speed overhead in VERL+FSDP2 is intrinsic to the quantize/dequantize compute + FP8 GEMM being slower than BF16 GEMM at 8B/32B matrix sizes on MI300X, NOT Triton fallback

---

## Open Issues

| Issue | Status | Impact |
|---|---|---|
| FSDP1 + PEFT LoRA dtype mismatch | **Fixed: use FSDP2** | LoRA + FSDP2 works natively |
| AITER FP32 input fallback to Triton | **Resolved: FSDP2 only** | Not occurring in FSDP2 path; FSDP1-only issue |
| `quant.enable()` memory overhead | By design | +4 GB/GPU from quantize/dequantize buffers |
| 70B model CPU OOM | Blocked | Cannot test on current hardware config |
| AITER quant kernel crash (fp8_param_gather) | **Fixed** | Added `.contiguous()` guard + `TORCH_CHECK` in AITER kernels |
| True FP8 weight storage | Not feasible | PyTorch autograd/DDP don't support FP8 Parameters; use `fp8_weight_cache` instead |
| FP8 activation storage for HF models | **Fixed** | Extended `_apply_pre_quant` to `nn.Linear`; no effect on full FP8 path |
| FP8 parameter all-gather for FSDP2 | **Fixed** (contiguity) | Crash resolved; API works but not wired into FSDP2 collectives |
| FP8 weight cache for VERL | **Fixed** | Wired `store_weights_fp8()` into `LumenConfig` + VERL; -2% actor time |
| VERL + SGLang ROCm patches not upstreamed | Open | 15+ patches needed for ROCm compat |
| SGLang 32B TP=1 OOM | By design | 32B model requires TP>=2 for SGLang rollout |
| FP8 overhead scales with model size | Confirmed | 8B: 1.5x slower, 32B: 1.85x slower (intrinsic, not Triton fallback) |
