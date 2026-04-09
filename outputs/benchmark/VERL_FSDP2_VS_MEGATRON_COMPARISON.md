# VERL FSDP2 vs Megatron Comparison on MI300X (ROCm 7.0)

## Test Environment

- **Container**: `rocm/sgl-dev:v0.5.9-rocm700` with VERL 0.8.0.dev
- **GPUs**: 4x MI300X (252 GB usable)
- **Model**: Qwen2.5-0.5B-Instruct
- **Rollout**: SGLang (v0.5.9-dev) or vLLM (v0.9.2rc2-dev), TP=1 or TP=2
- **Training**: GRPO, 2 steps, micro_batch=2, seq_len=512+256
- **Memory**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- **Date**: 2026-04-09

## Results Summary

### SGLang Rollout

| Configuration | Offload | Batch | Peak VRAM | Throughput | vs BF16 (same offload) |
|---|---|---|---|---|---|
| FSDP2+SGLang BF16 | Yes | 16 | **48.06 GB** | 797 tok/s | baseline |
| **FSDP2+SGLang FP8PM** | **Yes** | **16** | **45.50 GB** | **942 tok/s** | **-5% SAVINGS** |
| FSDP2+SGLang BF16 | No | 16 | 73.49 GB | ~900 tok/s | baseline |
| **FSDP2+SGLang FP8PM** | **No** | **16** | **54.87 GB** | **913 tok/s** | **-25% SAVINGS** |
| Megatron+SGLang BF16 | Yes | 16 | 70.52 GB | 704 tok/s | N/A |
| **Megatron+SGLang FP8PM** | **Yes** | **16** | **50.06 GB** | **369 tok/s** | **-29% vs Megatron BF16** |

### vLLM Rollout

| Configuration | Offload | Batch | gpu_util | Peak VRAM | Throughput (step 2) | Weight Update |
|---|---|---|---|---|---|---|
| FSDP2+vLLM BF16 (1 GPU) | Yes | 8 | 0.4 | — | 1,537 tok/s | 0.66s |
| FSDP2+vLLM BF16 (4 GPU) | Yes | 64 | 0.4 | 125.75 GB | 1,766 tok/s | 0.47–0.58s |
| FSDP2+vLLM FP8PM (4 GPU) | Yes | 64 | 0.4 | 125.78 GB | 1,871 tok/s | 0.55–0.70s |
| FSDP2+vLLM FP8PM (4 GPU) | No | 64 | 0.4 | 128.51 GB | 1,906 tok/s | 0.46–0.97s |
| **FSDP2+vLLM BF16 (4 GPU)** | **No** | **16** | **0.1** | **34.26 GB** | **445 tok/s** | **1.28s** |
| **FSDP2+vLLM FP8PM (4 GPU)** | **No** | **16** | **0.1** | **29.80 GB** | **477 tok/s** | **1.04s** |
| Megatron+vLLM BF16 (4 GPU, actor TP=2) | Yes | 16 | 0.3 | 85.01 GB | 338 tok/s | 2.03s |
| **Megatron+vLLM FP8PM (4 GPU, actor TP=2)** | **Yes** | **16** | **0.3** | **86.89 GB** | **291 tok/s** | **2.57s** |

**Notes on vLLM gpu_memory_utilization:**
- With `gpu_util=0.4`: vLLM reserves ~101 GB per GPU for KV cache, masking actor FP8PM savings in peak VRAM. Throughput gains (+6%) come from reduced memory bandwidth.
- With `gpu_util=0.1`: vLLM reservation is minimal, making FP8PM savings visible in peak VRAM (**29.80 vs 34.26 GB** = -13%) and throughput (**477 vs 445 tok/s** = +7%).

FSDP2+vLLM tests use rollout TP=2, `load_format=dummy`, `enforce_eager=true`. Megatron+vLLM tests use rollout TP=1 (TP>=2 hangs on ROCm).

### GPU-level Detail — SGLang (GPUs 4–7, with offloading)

| Config | GPU 4 | GPU 5 | GPU 6 | GPU 7 |
|---|---|---|---|---|
| FSDP2 BF16 (offload) | 48.06 GB | 11.88 GB | 11.32 GB | 48.03 GB |
| **FSDP2 FP8PM (offload)** | **45.50 GB** | **8.52 GB** | **8.57 GB** | **45.49 GB** |
| FSDP2 BF16 (no offload) | 73.49 GB | 18.16 GB | 17.01 GB | 72.75 GB |
| FSDP2 FP8PM (no offload) | 54.87 GB | 17.81 GB | 16.99 GB | 53.78 GB |
| Megatron BF16 (offload) | 70.52 GB | 11.62 GB | 11.58 GB | 70.49 GB |
| **Megatron FP8PM (offload)** | **50.06 GB** | **10.57 GB** | **10.58 GB** | **50.01 GB** |

GPUs 4,7 host both actor training and SGLang rollout. GPUs 5,6 host only rollout workers.

### GPU-level Detail — vLLM (GPUs 0–3, no offloading, gpu_util=0.1)

| Config | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---|---|---|---|---|
| FSDP2 BF16 | 34.26 GB | 33.92 GB | 33.97 GB | 34.07 GB |
| **FSDP2 FP8PM** | **29.80 GB** | **28.60 GB** | **28.91 GB** | **28.84 GB** |

All 4 GPUs host both FSDP2 training and vLLM rollout (TP=2, 2 replicas).

## Key Findings

### 1. FP8PM saves memory with both offload modes (after clone fix)

`_FP8LinearFunc.forward` previously called `ctx.save_for_backward(fp8_weight, scale)`, which pinned FSDP2's allgathered weight tensors on GPU and prevented offload reclamation. This caused a +44% memory regression with offloading (48→69 GB).

**Fix**: `ctx.save_for_backward(fp8_weight.clone(), scale)` in `lumen/quantize/fp8_params.py`. The clone creates an independent FP8 copy (~0.5 GB for Qwen 0.5B), allowing FSDP2 to free the allgathered buffer.

| Mode | BF16 | FP8PM (after fix) | Savings |
|---|---|---|---|---|
| With offload | 48.06 GB | **45.50 GB** | **-5%** |
| Without offload | 73.49 GB | **54.87 GB** | **-25%** |

FP8PM now works correctly with offloading. Without offloading still gives the largest savings because FP8 params stay on GPU at half the size of BF16.

### 2. FSDP2 is more memory-efficient than Megatron for BF16

With offloading, FSDP2 achieves **32% lower peak memory** (48 vs 71 GB) and **13% higher throughput** (797 vs 704 tok/s) compared to Megatron for the same model.

### 3. Megatron FP8PM now works (on-the-fly quantization)

`FP8ParamManager` was extended to target Megatron's `ColumnParallelLinear`/`RowParallelLinear`. Unlike FSDP2 (in-place FP8 quantization), Megatron uses **on-the-fly** quantization: parameters stay BF16 for optimizer/DDP compatibility, while `_FP8MegatronLinearFunc` quantizes to FP8 during forward and saves compact FP8 data in the autograd graph.

In-place quantization was tried first but crashed Megatron's distributed optimizer (`Failed to unpickle serialized exception`). The on-the-fly approach halves the weight portion of autograd-graph memory without modifying parameter storage.

Injection point: `verl/workers/engine/megatron/transformer_impl.py:_build_megatron_module()`, gated by `FP8_PARAM_MANAGER=1` env var. Only applied to the actor model (not the reference model).

**LoRA with Megatron** is now supported via Lumen's `MegatronLoraAdapter`, which wraps `ColumnParallelLinear`/`RowParallelLinear` with TP-aware low-rank adapters injected before DDP wrapping. Set `LORA_RANK=32` to enable.

### 4. vLLM V1 works on ROCm after `get_device_uuid` fix

vLLM V1 was previously broken on ROCm — both FSDP2+vLLM and Megatron+vLLM hung indefinitely during weight updates. The root cause was a **missing `@with_amdsmi_context` decorator** on `RocmPlatform.get_device_uuid()` in `vllm/platforms/rocm.py`. Without it, `amdsmi` was uninitialized in VERL worker processes, producing a fallback UUID (`rocm-gpu-0`) that didn't match the EngineCore's real hardware UUID (`0xDE72E6A9A0230550`). This created mismatched ZMQ IPC socket paths for the weight transfer, causing a deadlock.

**Fix**: Add `@with_amdsmi_context` to `get_device_uuid()` — a one-line change. After the fix, FSDP2+vLLM (1 and 4 GPU) and Megatron+vLLM (4 GPU) all complete training successfully.

**Note**: Rollout TP>=2 (multiple EngineCore processes per vLLM server) may still hang during initialization due to a separate `spawn` multiprocessing issue on ROCm. Use TP=1 for rollout until upstream vLLM addresses this.

### 5. FSDP2 outperforms Megatron across both rollout engines

| Config | Rollout | Throughput (step 2) |
|---|---|---|
| FSDP2 BF16 | SGLang | ~900 tok/s |
| Megatron BF16 | SGLang | 704 tok/s |
| FSDP2 BF16 | vLLM (gpu_util=0.4, batch=64) | 1,813 tok/s |
| FSDP2 BF16 | vLLM (gpu_util=0.1, batch=16) | 445 tok/s |
| Megatron BF16 | vLLM (gpu_util=0.3, batch=16) | 338 tok/s |

FSDP2 consistently achieves **1.3x higher throughput** than Megatron for the same model and comparable batch size.

### 6. vLLM FP8PM: savings visible at low gpu_memory_utilization

With `gpu_memory_utilization=0.4`, vLLM reserves ~101 GB per GPU, masking FP8PM savings in peak VRAM. Throughput improves +6% (1,766→1,871 tok/s) from reduced memory bandwidth.

With `gpu_memory_utilization=0.1` (minimal vLLM reservation), FP8PM savings become visible:
- Peak VRAM: **29.80 GB** (FP8PM) vs **34.26 GB** (BF16) — **13% less**
- FSDP memory after init: **0.12 GB** (FP8PM) vs **0.46 GB** (BF16) — 74% less
- Throughput: **477 tok/s** (FP8PM) vs **445 tok/s** (BF16) — **7% faster**

For production with larger models, lower `gpu_memory_utilization` makes FP8PM savings proportionally more significant since actor memory dominates.

### 8. Megatron FP8PM: memory savings visible, throughput regression with vLLM

With SGLang rollout, Megatron FP8PM shows clear memory savings: **50.06 GB** vs **70.52 GB** (BF16) — **29% less peak VRAM**. Throughput drops from 704 to 369 tok/s, partly due to on-the-fly BF16→FP8 quantization overhead in `_FP8MegatronLinearFunc`.

With vLLM rollout (gpu_util=0.3), Megatron FP8PM shows **86.89 GB** vs BF16's **85.01 GB** peak — negligible difference since vLLM KV cache dominates. Throughput is **291 tok/s** vs BF16's **338 tok/s** (**-14%** regression) from the on-the-fly quantization overhead. Weight update slows from 2.03s to 2.57s. For throughput-sensitive workloads, FSDP2+FP8PM remains the better choice.

### 7. `USE_8BIT_ADAM` env var does not change the optimizer

The `USE_8BIT_ADAM=1` flag in `LumenConfig` is a hint — VERL's `build_optimizer` uses its own `FSDPOptimizerConfig` (default: `torch.optim.AdamW`). To use 8-bit Adam, set `actor_rollout_ref.actor.optim.optimizer=AdamW8bit` and `optimizer_impl=bitsandbytes.optim` in VERL config. Note: bitsandbytes `AdamW8bit` is **incompatible with FSDP2 DTensor** — use `torchao.optim` if DTensor-compatible 8-bit Adam is needed.

## Recommended Configuration

For VERL GRPO on MI300X (ROCm) with FP8 memory savings:

```
strategy=fsdp2
rollout=sglang            # or vllm (with get_device_uuid fix)
FP8_PARAM_MANAGER=1
# Offloading now works with FP8PM (after clone fix):
#   param_offload=true  → -5% VRAM savings vs BF16 offload
#   param_offload=false → -25% VRAM savings vs BF16 no-offload (best)
```

Expected savings: **-25% peak VRAM** vs BF16 (no offload), **-5%** with offload.

For vLLM rollout, additionally set:
```
rollout.tensor_model_parallel_size=1   # TP>=2 may hang on ROCm
rollout.free_cache_engine=false
rollout.enforce_eager=true
```

## Reproduction

```bash
# FSDP2+SGLang FP8PM (recommended: no offloading)
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FP8_PARAM_MANAGER=1 USE_8BIT_ADAM=1 \
NUM_GPUS=4 ROLLOUT_TP=2 ROLLOUT_GPU_UTIL=0.1 \
TRAIN_BSZ=16 MAX_STEPS=2 \
MODEL_NAME=/dev/shm/model/qwen2.5-0.5b-instruct \
bash examples/rl/verl/run_grpo_fsdp2.sh
# NOTE: Ensure run_grpo_fsdp2.sh sets param_offload=false, optimizer_offload=false

# FSDP2+SGLang BF16 baseline (no offloading, for fair comparison)
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NUM_GPUS=4 ROLLOUT_TP=2 ROLLOUT_GPU_UTIL=0.1 \
TRAIN_BSZ=16 MAX_STEPS=2 \
MODEL_NAME=/dev/shm/model/qwen2.5-0.5b-instruct \
bash examples/rl/verl/run_grpo_fsdp2.sh

# Megatron+SGLang BF16 (offloading works for BF16)
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NUM_GPUS=4 ACTOR_TP=2 ROLLOUT_TP=2 ROLLOUT_GPU_UTIL=0.1 \
TRAIN_BSZ=16 MAX_STEPS=2 \
PARAM_OFFLOAD=true OPTIMIZER_OFFLOAD=true \
MODEL_NAME=/dev/shm/model/qwen2.5-0.5b-instruct \
bash examples/rl/verl/run_grpo_megatron_sglang.sh

# FSDP2+vLLM BF16 (4 GPU, requires get_device_uuid fix)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NUM_GPUS=4 ROLLOUT_TP=1 ROLLOUT_GPU_UTIL=0.4 \
TRAIN_BSZ=64 MAX_STEPS=2 \
MODEL_NAME=/dev/shm/model/qwen2.5-0.5b-instruct \
bash examples/rl/verl/run_grpo_fsdp2_vllm.sh

# FSDP2+vLLM FP8PM (4 GPU, with offload)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FP8_PARAM_MANAGER=1 \
NUM_GPUS=4 ROLLOUT_TP=1 ROLLOUT_GPU_UTIL=0.4 \
TRAIN_BSZ=64 MAX_STEPS=2 \
PARAM_OFFLOAD=true OPTIMIZER_OFFLOAD=true \
MODEL_NAME=/dev/shm/model/qwen2.5-0.5b-instruct \
bash examples/rl/verl/run_grpo_fsdp2_vllm.sh

# FSDP2+vLLM FP8PM (4 GPU, no offload, best throughput)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FP8_PARAM_MANAGER=1 \
NUM_GPUS=4 ROLLOUT_TP=1 ROLLOUT_GPU_UTIL=0.4 \
TRAIN_BSZ=64 MAX_STEPS=2 \
MODEL_NAME=/dev/shm/model/qwen2.5-0.5b-instruct \
bash examples/rl/verl/run_grpo_fsdp2_vllm.sh

# Megatron+SGLang FP8PM (4 GPU, actor TP=2, on-the-fly FP8)
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FP8_PARAM_MANAGER=1 \
NUM_GPUS=4 ACTOR_TP=2 ROLLOUT_TP=2 ROLLOUT_GPU_UTIL=0.1 \
TRAIN_BSZ=16 MAX_STEPS=2 \
PARAM_OFFLOAD=true OPTIMIZER_OFFLOAD=true \
MODEL_NAME=/dev/shm/model/qwen2.5-0.5b-instruct \
bash examples/rl/verl/run_grpo_megatron_sglang.sh
# NOTE: Requires patched transformer_impl.py for FP8PM injection

# Megatron+vLLM BF16 (4 GPU, actor TP=2, rollout TP=1)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NUM_GPUS=4 ACTOR_TP=2 ROLLOUT_TP=1 ROLLOUT_GPU_UTIL=0.3 \
TRAIN_BSZ=16 MAX_STEPS=2 \
MODEL_NAME=/dev/shm/model/qwen2.5-0.5b-instruct \
bash examples/rl/verl/run_grpo_megatron_vllm.sh

# Megatron+vLLM FP8PM (4 GPU, actor TP=2, rollout TP=1)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FP8_PARAM_MANAGER=1 \
NUM_GPUS=4 ACTOR_TP=2 ROLLOUT_TP=1 ROLLOUT_GPU_UTIL=0.3 \
TRAIN_BSZ=16 MAX_STEPS=2 \
PARAM_OFFLOAD=true OPTIMIZER_OFFLOAD=true \
MODEL_NAME=/dev/shm/model/qwen2.5-0.5b-instruct \
bash examples/rl/verl/run_grpo_megatron_vllm.sh
# NOTE: Requires patched transformer_impl.py; rollout TP must be 1 on ROCm
```

## vLLM ROCm Fix

The vLLM hang on ROCm is caused by a missing `@with_amdsmi_context` decorator in `vllm/platforms/rocm.py`. Apply this one-line fix before using vLLM rollout:

```python
# In vllm/platforms/rocm.py, add @with_amdsmi_context to get_device_uuid:
@classmethod
@with_amdsmi_context        # <-- ADD THIS LINE
def get_device_uuid(cls, device_id: int = 0) -> str:
    ...
```

Without this fix, the VERL training worker and the vLLM EngineCore subprocess resolve different device UUIDs, creating mismatched ZMQ socket paths that deadlock the weight transfer.

## FP8PM FSDP2 Offload Fix

FP8ParamManager previously regressed memory by +44% when used with FSDP2 offloading. The fix is a one-line change in `lumen/quantize/fp8_params.py`:

```python
# In _FP8LinearFunc.forward, change:
ctx.save_for_backward(fp8_weight, scale)
# to:
ctx.save_for_backward(fp8_weight.clone(), scale)
```

Without the clone, `save_for_backward` pins FSDP2's allgathered weight buffer, preventing FSDP2 from freeing it after each layer's forward pass. The clone creates an independent FP8 copy that doesn't block FSDP2's memory management.

## Megatron FP8PM Patch

FP8ParamManager for Megatron requires a patch to `verl/workers/engine/megatron/transformer_impl.py` to inject `LumenConfig.enable()` into `_build_megatron_module()`, after weight loading but before the module is returned. The patch is gated by `FP8_PARAM_MANAGER=1` env var and only applies to the actor model (not the reference model).

Key design difference from FSDP2: Megatron params stay in BF16 (on-the-fly quantization) to preserve distributed optimizer and DDP compatibility. In-place FP8 quantization crashes Megatron's optimizer.
