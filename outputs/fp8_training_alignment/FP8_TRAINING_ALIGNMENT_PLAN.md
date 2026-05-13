# FP8 Training Alignment — Lumen Reproduction Plan

**Date**: 2026-04-10 (updated 2026-04-13)
**Reference**: [VERL FP8 RL documentation](https://github.com/verl-project/verl/blob/main/docs/advance/fp8.md) | [FP8-RL paper (arXiv:2601.18150)](https://arxiv.org/abs/2601.18150)
**Goal**: Reproduce VERL's FP8 RL benchmarks using Lumen on MI350X, demonstrating that FP8 training aligns with BF16 training when both use FP8 rollout. Test three FP8 linear scaling methods (blockwise, MXFP8 with block-size sweep, per-tensor delayed), FP8 attention quantization (DPA/MHA modes), low-precision optimizer states (BNB AdamW8bit, TorchAO _AdamW), and identify gaps vs the FP8-RL paper (KV-cache FP8, QKV recalibration).

---

## Hardware & Environment

| Item | Spec |
|------|------|
| GPUs | 8x AMD Instinct MI350X |
| Container | `rocm/sgl-dev:v0.5.9-rocm700-mi35x-20260224` (`lumen_verl_test`) |
| VERL | 0.8.0.dev |
| vLLM | 0.9.2rc2.dev (ROCm) |
| HF Transformers | 4.57.1 |
| PyTorch | 2.9.0+rocm7.0 |

**vs Reference**: VERL docs used 8–16x H100 with CUDA 12.6/12.9, Transformer Engine, vLLM 0.10–0.11.

---

## Models & Data

| Asset | Location | Size |
|-------|----------|------|
| Qwen3-8B-Base | `/dev/shm/model/qwen3-8b-base` | 16 GB |
| Qwen3-30B-A3B-Base (MoE) | `/dev/shm/model/qwen3-30b-a3b-base` | 57 GB |
| Qwen3-30B-A3B (MoE) | `/dev/shm/model/qwen3-30b-a3b` | 57 GB |
| DAPO-Math-17k (train) | `/dev/shm/data/dapo-math-17k.parquet` | 286 MB |
| AIME-2024 (val) | `/dev/shm/data/aime-2024.parquet` | 29 KB |

---

## VERL Capabilities Verified

| Feature | Status | How |
|---------|--------|-----|
| DAPO reward manager | ✅ | `verl.workers.reward_manager.dapo.DAPORewardManager` |
| Decoupled clipping | ✅ | `cliprange_low`, `cliprange_high`, `clip_ratio_c` in `compute_policy_loss` |
| Token-level loss | ✅ | `loss_agg_mode="token-mean"` |
| Rollout correction (TIS) | ✅ | `rollout_is=token`, `rollout_is_threshold=2.0` in `rollout_corr_helper` |
| Overlong reward buffer | ✅ | Built into `DAPORewardManager` |
| vLLM FP8 rollout | ✅ | `rollout.quantization=fp8` → `apply_vllm_fp8_patches()` |
| Qwen3ForCausalLM (vLLM) | ✅ | Registered in vLLM model registry |
| Qwen3MoeForCausalLM (vLLM) | ✅ | Registered in vLLM model registry |
| Dynamic sampling (filter_groups) | ⚠️ Config exists, not in RayPPOTrainer loop | Both BF16/FP8 runs skip it, comparison is still fair |

---

## Quantization Method

This section describes exactly what "FP8 training" means in each system: the VERL reference (Transformer Engine on H100) and the Lumen reproduction (AITER on MI350X).

### VERL Reference: Transformer Engine Blockwise FP8

The VERL E2E reference uses NVIDIA Transformer Engine (TE) with **blockwise FP8 scaling** on H100 GPUs:

| Property | VERL Reference (TE) |
|----------|-------------------|
| FP8 format (forward) | E4M3 (`float8_e4m3fn`, max 448) |
| FP8 format (backward) | E5M2 (`float8_e5m2`, max 57344) — "hybrid" recipe |
| Weight quantization | Per-block (128-element blocks) |
| Activation quantization | Per-block (1×128 for activations, 128×128 for weights) |
| Scaling strategy | Blockwise — each block has its own FP8 scale factor |
| Optimizer states | FP8 (`fp8_recipe: "blockwise"` for optimizer) |
| GEMM backend | cuBLAS / cuDNN FP8 tile ops |
| MoE router | BF16 (excluded from FP8) |
| Env requirement | `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`, CUDA 12.9+ |

### Lumen Reproduction: AITER Per-Tensor FP8

Lumen's FP8 stack is built for AMD MI350X using AITER (CK, Triton, hipBLASLt) backends. The quantization method differs from TE in scaling granularity but targets the same goal: FP8 GEMMs with minimal accuracy loss.

#### FP8 Data Types

MI350X is gfx950+ (arch version >= 9.5), so Lumen auto-selects the **OCP** (Open Compute Project) FP8 variants — the same format family as H100/TE:

| Tensor | FP8 dtype | Max representable | Notes |
|--------|-----------|-------------------|-------|
| Weights (forward) | `float8_e4m3fn` | 448 | OCP E4M3 — same format as TE on H100 |
| Activations (forward) | `float8_e4m3fn` | 448 | Same as weights |
| Gradients (backward) | `float8_e5m2` | 57344 | HYBRID recipe: E4M3 forward, E5M2 backward |

On older gfx94x GPUs (MI300X), the FNUZ variants are used instead (`float8_e4m3fnuz` max 240, `float8_e5m2fnuz` max 57344). The selection is in `_is_fp8_fnuz()`: returns True when `(major, minor) < (9, 5)`.

#### Scaling Strategies

Lumen supports multiple FP8 scaling modes. These experiments test three of them to compare against the VERL reference's blockwise TE approach:

| Scaling mode | Description | Experiments |
|--------------|-------------|-------------|
| **Blockwise** | One FP32 scale per 128-element block along the K dimension. For input `(M, K)`: scale shape is `(M, ceil(K/128))`. Each block independently computes `scale = fp8_max / block_amax`. Matches the FP8-RL paper's approach (1x128 for activations). | **1D, 2C, 3C** (priority 1) |
| **MXFP8 (Microscaling)** | One E8M0 (exponent-only uint8) scale per B-element block (B=32/64/128 sweep). Power-of-two scales: stored as `uint8`, dequanted via `(uint8 << 23).view(float32)`. OCP MX standard. Finer granularity at smaller B. | **1E/1E-64/1E-128, 2D/2D-64/2D-128, 3D/3D-64/3D-128** (priority 2) |
| **Per-tensor (delayed)** | One FP32 scale per entire matrix, derived from amax history (configurable window). Next step's scale = `fp8_max / amax_history_max / 2^margin`. Fused quant+amax kernel computes quant + next amax in one pass. | **1F, 2E, 3E** (priority 3) |
| Per-tensor (dynamic) | One scale per matrix from current tensor's global amax. No history. | Fallback when no ScalingManager |
| Per-token | Per-row scales `(M, 1)`. | Not used in E2E experiments |

##### Per-Tensor Delayed Scaling (runs 1F, 2E, 3E)

```
amax = max(amax_history)           # or MOST_RECENT from history deque
scale = (fp8_max / amax) / 2^margin
fp8_tensor = clamp(tensor * scale, -fp8_max, fp8_max).to(fp8_dtype)
```

The `_safe_fp8_desc()` guard handles the edge case where amax=0 (all-zero input) produces scale=0 and NaN artifacts: it replaces the tensor with clean zeros and scale with 1.0.

##### Blockwise FP8 Scaling (runs 1D, 2C, 3C)

Following the FP8-RL paper (arXiv:2601.18150), blockwise scaling assigns one FP32 scale factor per block of elements, providing finer-grained dynamic range than per-tensor:

```
For input (M, K) with block_size=128:
  Reshape to blocks of size 128 along K axis
  For each block:
    block_amax = max(abs(block_elements))
    block_scale = fp8_max / max(block_amax, 1e-4)
    fp8_block = clamp(block * block_scale, -fp8_max, fp8_max).to(fp8_dtype)
    stored_scale = 1 / block_scale   # dequant: fp8 * stored_scale
  Scale tensor shape: (M, ceil(K/128))
```

Implementation: Triton `quant_fp8_blockwise_kernel` (`lumen/ops/quantize/ops.py`), GEMM via `gemm_blockscale` (CK `gemm_a8w8_blockscale` preferred, Triton fallback).

The FP8-RL paper uses blockwise for both rollout (vLLM FP8 W8A8) and E2E training (TE blockwise). Our blockwise runs reproduce the same **scaling granularity** as the VERL reference but via Lumen's AITER/CK/Triton backend instead of TE/cuBLAS.

##### MXFP8 Microscaling (runs 1E/1E-64/1E-128, 2D/2D-64/2D-128, 3D/3D-64/3D-128)

MXFP8 (Microscaling FP8) uses the OCP MX standard with exponent-only (E8M0) block scales:

```
For input (M, K) with mxfp8_block=B:
  Reshape to blocks of size B along K axis
  For each block:
    Compute E8M0 exponent-only scale (power-of-two)
    Quantize block elements using the E8M0 scale
  Scale tensor: uint8 E8M0 values, shape (M, ceil(K/B))
  Dequant: fp8 * (e8m0_scale << 23).view(float32)  # power-of-two multiply
```

Implementation: Triton `_convert_to_mxfp8_kernel` with optional ASM path on gfx950 (CDNA4), GEMM via `gemm_mxfp8` which converts E8M0→FP32 scales then delegates to `gemm_a8w8_blockscale`.

MXFP8 uses power-of-two scales (zero rounding error in scale application), providing lower quantization noise than FP32-scaled methods at the same block size. This comes at the cost of more scale storage and slightly different GEMM dispatch.

**MXFP8 block size sweep**: We sweep three block sizes (32, 64, 128) to measure the accuracy/throughput trade-off:

| Config block_size | Effective MXFP8 block | Scales per row (K=4096) | Notes |
|-------------------|----------------------|------------------------|-------|
| 32 | 32 | 128 | Finest granularity (OCP MX default) |
| 64 | 64 | 64 | Medium granularity |
| 128 | 128 | 32 | Coarsest MXFP8 granularity, matches blockwise FP32 scale count |

**Code change required**: The current `quantize_input` in `lumen/ops/quantize/linear.py` clamps `mxfp8_block = 32 if block_size > 64 else block_size`, so `block_size=128` is silently reduced to 32. To enable the full sweep, this clamp must be removed or raised to allow 128-element MXFP8 blocks. The `convert_to_mxfp8` kernel and `gemm_mxfp8` already accept arbitrary block sizes — only the `quantize_input` routing logic needs updating.

##### Scaling Comparison

| Property | Blockwise (128) | MXFP8-32 | MXFP8-64 | MXFP8-128 | Per-tensor delayed | VERL Reference (TE blockwise) |
|----------|-----------------|----------|----------|-----------|-------------------|------------------------------|
| Scale granularity | 1 per 128 elements | 1 per 32 elements | 1 per 64 elements | 1 per 128 elements | 1 per matrix | 1 per 128 elements |
| Scale dtype | FP32 | E8M0 (uint8) | E8M0 (uint8) | E8M0 (uint8) | FP32 | FP32 |
| Scale storage | `(M, K/128)` | `(M, K/32)` | `(M, K/64)` | `(M, K/128)` | negligible | `(M, K/128)` |
| Quantization noise | medium | lowest | low | medium | highest | medium |
| GEMM backend | CK/Triton blockscale | Triton MXFP8→blockscale | same | same | hipBLASLt/CK/Triton | cuBLAS/cuDNN |
| Backward grad scaling | per-tensor dynamic | MXFP8 (dgrad FP8) | same | same | per-tensor delayed | blockwise |

#### Forward Pass (per `nn.Linear` layer)

When both FP8ParamManager (`FP8_PARAM_MANAGER=1`) and FP8 Linear (`LUMEN_FP8=1`) are enabled, the forward pass for each linear layer works as follows:

```
1. Weight: already stored as FP8 by FP8ParamManager (via FP8Descriptor)
   - FP8PM quantized the weight at initialization: amax → scale → fp8_data
   - FP8 Linear detects the pre-quantized weight via weight._fp8_desc
   - No re-quantization of weights (no double quant)

2. Input activation: quantized to FP8 on the fly
   - Per-tensor delayed scaling via ScalingManager (when configured)
   - Or per-tensor dynamic scaling as fallback
   - Result: FP8Descriptor(data=fp8_input, scale=input_scale)

3. GEMM: Y = fp8_input @ fp8_weight^T
   - Dispatched to hipBLASLt, CK (gemm_a8w8_CK), or Triton (gemm_a8w8)
   - Backend selection: hipBLASLt preferred when LUMEN_PREFER_HIPBLASLT=1
   - Output in BF16 (dequantized by the GEMM kernel using input_scale * weight_scale)

4. Saved for backward: FP8 weight descriptor + FP8 input descriptor
   - NOT the full BF16 tensors — memory savings from FP8 storage
```

#### Backward Pass

```
1. dgrad (gradient w.r.t. input):
   - Quantize grad_output to FP8 (E5M2 in HYBRID mode)
   - GEMM: grad_input = fp8_grad_output @ fp8_weight
   - Weight's cached transpose (FP8Descriptor._transpose) avoids recomputing .t().contiguous()
   - If grad and weight FP8 dtypes differ (E5M2 vs E4M3): gemm_per_tensor_mixed via hipBLASLt

2. wgrad (gradient w.r.t. weight):
   - Prefer gemm_wgrad_fp8 (hipBLASLt) when K >= 64 and delayed/dynamic scaling
   - Otherwise: dequant to BF16 and BF16 GEMM
   - Optional ScalingManager.quantize_grad: quant→dequant round-trip on BF16 wgrad

3. All gradients flow in BF16 to the optimizer
   - FP8 is used only for GEMM operands, not for optimizer state updates
```

#### FP8ParamManager (Weight Storage)

FP8ParamManager (`lumen/quantize/fp8_params.py`) is a **memory optimization layer** independent of the GEMM path:

- Replaces `nn.Linear` weight storage from BF16 (2 bytes/element) to FP8 (1 byte/element)
- Quantization: `amax = param.abs().amax()`, `scale = fp8_max / amax`, `fp8 = (param.float() * scale).to(fp8_dtype)` — pure per-tensor, no history
- Wraps forward with `_FP8LinearFunc` autograd function that dequants FP8→BF16 for matmul
- When FP8 Linear is also enabled, `_FP8LinearFunc` is bypassed: the patched forward reads `weight._fp8_desc` directly for FP8 GEMM
- `save_for_backward` stores `fp8_weight.clone() + scale` (1 byte/elem clone), not the full BF16 weight — FSDP2 can reclaim the all-gathered buffer after each layer's forward
- Frozen weights: FP8PM-managed params have `requires_grad=False`; only non-linear params (embeddings, norms, LM head) retain gradients

#### FP8 Activation Store

When `LUMEN_FP8_ACTIVATION_STORE=1` (used in 2C/3C for MoE memory savings):

- MLP activations saved for backward are stored as FP8 instead of BF16
- `_fp8_store_activation`: quantizes activation to FP8 via dynamic per-tensor scaling, saves `fp8_data + scale`
- `_fp8_restore_activation`: dequantizes back to BF16 for backward GEMM: `bf16 = fp8.to(float32) * scale`
- Fused MLP path: saves `x` and post-activation `hidden` as uint8 views of FP8 + scales; backward dequantizes to BF16 for all MLP GEMMs
- Trade-off: ~50% memory reduction for saved activations at the cost of quantization noise in backward

#### Fused Kernel Optimizations

These are performance optimizations that reduce kernel launch overhead and memory bandwidth. They do not change numerical behavior (same quant math, fewer passes over data):

| Env Var | Kernel | What it fuses |
|---------|--------|---------------|
| `LUMEN_FUSED_QUANT_AMAX=1` | Triton `static_quant_with_amax` | Quantize tensor to FP8 **and** compute global amax for next delayed-scaling step in one pass (halves memory bandwidth) |
| `LUMEN_FUSED_CAST_TRANSPOSE=1` | Triton `cast_transpose_fp8` | Quantize to FP8 **and** write both normal + transposed layout in one pass (fills `FP8Descriptor._transpose` for hipBLASLt) |
| `LUMEN_FUSED_QUANT_SCALE=1` | AITER `static_per_tensor_quant_fp8_i8` | FP8 quantization given a precomputed scale (skips amax computation) |
| `LUMEN_TRANSPOSE_CACHE=1` | Lazy `.t().contiguous()` in `FP8Descriptor` | Caches the transposed FP8 tensor; avoids recomputation on every backward GEMM |

#### FP8Descriptor Dataclass

The `FP8Descriptor` (`lumen/quantize/descriptor.py`) is the central data structure that flows through the FP8 pipeline:

```
FP8Descriptor:
    data: Tensor          # FP8 tensor (uint8 storage)
    scale: Tensor         # float32 scale factor (1/scale to dequantize)
    fp8_dtype: torch.dtype # e4m3fn/fnuz or e5m2/fnuz
    _transpose: Tensor?   # lazy-cached transposed FP8 (for hipBLASLt backward)
```

- `transpose_cached` property: returns cached transpose or computes it (optionally via Triton fast transpose)
- `invalidate_transpose()`: clears cache when the underlying FP8 buffer is replaced (e.g., after FP8PM re-quantization)

### Lumen vs Transformer Engine: Key Differences

| Property | VERL Reference (TE on H100) | Lumen blockwise (1D/2C/3C) | Lumen MXFP8-32 (1E/2D/3D) | Lumen MXFP8-64 (*-64) | Lumen MXFP8-128 (*-128) | Lumen per-tensor (1F/2E/3E) |
|----------|---------------------------|----------------------------|----------------------------|------------------------|--------------------------|------------------------|
| Scaling granularity | Blockwise (128) | **Blockwise (128)** | MXFP8 (32, E8M0) | MXFP8 (64, E8M0) | MXFP8 (128, E8M0) | Per-tensor (1 per matrix) |
| FP8 format | `e4m3fn` / `e5m2` | Same | `mxfp8` / `e5m2` | same | same | Same |
| FP8 max (E4M3) | 448 | 448 | 448 | 448 | 448 | 448 |
| Optimizer states | FP8 blockwise | BF16 AdamW (Exp 1–4); **BNB 8bit (5A), TorchAO BF16 (5B)** | same | same | same | BF16 AdamW |
| GEMM backend | cuBLAS / cuDNN | CK/Triton blockscale | Triton MXFP8→blockscale | same | same | hipBLASLt/CK/Triton |
| Weight storage | TE internal | FP8PM + FP8 Linear | FP8PM + FP8 Linear | same | same | FP8PM + FP8 Linear |
| Backward dgrad | blockwise FP8 | BF16 (dequant path) | FP8 MXFP8 | same | same | FP8 per-tensor |
| Activation store | TE internal | Explicit flag | Explicit flag | same | same | Explicit flag |
| MoE router | BF16 | BF16 | BF16 | same | same | BF16 |
| **Attention FP8** | **TE FP8 attention (blockwise)** | **none (Exp 1–3); Exp 4 sweeps: dpa+blockwise (4A), dpa+mxfp8 (4B), dpa+dynamic (4C), mha+blockwise2d (4D)** | same | same | same | **none (Exp 1–3)** |
| Rollout FP8 | vLLM monkey-patch | Same | Same | same | same | Same |
| Rollout correction | Token-level TIS, C=2 | Same | Same | same | same | Same |
| Training backend | Megatron | FSDP2 | FSDP2 | same | same | FSDP2 |

**Design rationale for testing all scaling methods + MXFP8 sweep**: The FP8-RL paper ([arXiv:2601.18150](https://arxiv.org/abs/2601.18150)) uses TE blockwise scaling for E2E training. By testing blockwise first, then sweeping MXFP8 block sizes (32/64/128), then per-tensor, we can:
1. **Match the reference first**: Blockwise (1D/2C/3C) uses the same scaling granularity as TE, providing the closest apples-to-apples comparison — run first
2. **Sweep MXFP8 block sizes next**: MXFP8-32/64/128 tests how E8M0 scale granularity affects alignment and throughput — finer blocks (32) should give lowest quantization noise, while 128-element MXFP8 isolates the E8M0-vs-FP32 scale dtype difference vs blockwise
3. **Validate the lower bound last**: Per-tensor (1F/2E/3E) tests whether the coarsest scaling is sufficient for RL alignment
4. **RL noise tolerance**: If all methods align with BF16 across the block-size sweep, it confirms RL training is robust to both scaling granularity and scale dtype
5. **Attention FP8 sweep** (Experiment 4): Linear and attention are **independent axes** — Experiment 4 fixes linear to blockwise and sweeps attention FP8 modes (`dpa`/`mha`) with different quant types (blockwise/MXFP8/dynamic/blockwise2d), validating the full Lumen FP8 stack end-to-end
6. **Low-precision optimizer sweep** (Experiment 5): Optimizer is a **third independent axis** — Experiment 5 fixes linear to blockwise and attention to BF16, then tests BNB `AdamW8bit` and TorchAO `_AdamW` (bf16 stochastic round) for memory savings without alignment regression

### Lumen FP8 E2E Stack Summary

All E2E experiments use the **full Lumen FP8 stack**, controlled by these environment variables. Run order: blockwise (1D/2C/3C) first, MXFP8 (1E/2D/3D) second, per-tensor (1F/2E/3E) third:

| Feature | Env Var | Effect |
|---------|---------|--------|
| FP8 Param Manager | `FP8_PARAM_MANAGER=1` | Stores weights as FP8 in autograd graph (memory savings) |
| FP8 Linear GEMM | `LUMEN_FP8=1` | Replaces `nn.Linear` forward with FP8 GEMM via AITER |
| **FP8 Scaling Method** | **`LUMEN_FP8_SCALING=delayed\|blockwise`** | **Selects per-tensor delayed (default) or blockwise scaling** |
| **FP8 Format** | **`LUMEN_FP8_FORMAT=fp8_e4m3\|mxfp8`** | **Selects standard FP8 (default) or MXFP8 microscaling** |
| **FP8 Block Size** | **`LUMEN_FP8_BLOCK_SIZE=128`** | **Block size for blockwise/MXFP8 scaling (default 128; MXFP8 internally uses 32)** |
| **FP8 Attention** | **`LUMEN_FP8_ATTN=none\|dpa\|mha`** | **Quantizes Q/K/V to FP8 for attention compute (see FP8 Attention section)** |
| **Attention Quant Type** | **`LUMEN_FP8_QUANT_TYPE=blockwise`** | **Scaling method for attention Q/K/V tensors (blockwise, mxfp8, dynamic, etc.)** |
| **Attention Kernel** | **`LUMEN_ATTN_KERNEL_BACKEND=auto`** | **Attention kernel backend: auto, triton, or csrc** |
| Lumen Norm | `LUMEN_NORM=1` | Optimized RMSNorm/LayerNorm kernels (BF16, not quantized) |
| FP8 Activation Store | `LUMEN_FP8_ACTIVATION_STORE=1` | MLP activations stored in FP8 for backward (MoE runs only) |
| Fused Quant+Amax | `LUMEN_FUSED_QUANT_AMAX=1` | Single-kernel quant + amax for delayed scaling (auto-enabled) |
| Fused Cast+Transpose | `LUMEN_FUSED_CAST_TRANSPOSE=1` | Single-kernel FP8 cast + matrix transpose (auto-enabled) |
| Fused Quant Scale | `LUMEN_FUSED_QUANT_SCALE=1` | Static per-tensor quant via AITER Triton (auto-enabled) |
| Transpose Cache | `LUMEN_TRANSPOSE_CACHE=1` | `FP8Descriptor` lazy transpose caching (auto-enabled) |
| Entry point | `lumen.rl.verl.verl_entry` | Patches FSDP2 worker with `LumenConfig.enable()` before FSDP2 wrap |

The integration path: `verl_entry.py` reads env vars → builds `VerlLumenArgs` → calls `patch_verl_fsdp_workers` → wraps `apply_fsdp2` so that `LumenConfig.from_args(lumen_args).enable(model)` patches all `nn.Linear` modules **before** FSDP2 sharding.

### FP8 Attention Quantization

Lumen's FP8 stack has two **independent quantization axes**:

1. **Linear FP8** (`LUMEN_FP8` + `LUMEN_FP8_SCALING` + `LUMEN_FP8_FORMAT`): quantizes weight/activation GEMMs in `nn.Linear` layers
2. **Attention FP8** (`LUMEN_FP8_ATTN` + `LUMEN_FP8_QUANT_TYPE`): quantizes Q, K, V tensors before computing attention via fused Triton or CK kernels

These axes use **separate env vars, separate scaling methods, and separate kernel paths**. You can enable one without the other, or use different scaling granularities for each. Experiments 1–3 sweep the linear axis with BF16 attention; Experiment 4 fixes linear to blockwise and sweeps attention.

#### Attention FP8 Modes

| Mode | Env `LUMEN_FP8_ATTN` | What it does |
|------|----------------------|--------------|
| **none** (default) | `none` | BF16 attention — Q/K/V stay in BF16/FP16 through attention compute |
| **dpa** (dot-product attention) | `dpa` | After QKV linear projections (in BF16), Q/K/V are quantized to FP8 and attention is computed using FP8 fused kernels. Output projection remains in BF16 |
| **mha** (multi-head attention) | `mha` | Full MHA FP8: QKV projections + attention + output projection all in FP8. Attaches a `Blockwise2DScaleManager` per attention module for shared scale context across the attention layers |

#### Attention Quant Types

The attention FP8 path supports its own quantization type, independent of the linear FP8 scaling method:

| Quant type | Env `LUMEN_FP8_QUANT_TYPE` | Description |
|------------|---------------------------|-------------|
| **blockwise** (default) | `blockwise` | Blockwise FP8 scaling for Q/K/V tensors. Default when `attn_quant_type` is not specified |
| **blockwise2d** | `blockwise2d` | 2D blockwise scaling with `Blockwise2DScaleManager`. Used with `mha` mode for shared scale context |
| **mxfp8** | `mxfp8` | MXFP8 microscaling for attention tensors. Uses E8M0 scales and dedicated `mxfp8_attention_kernel` |
| **dynamic** | `dynamic` | Dynamic per-tensor scaling (single scale per Q/K/V matrix) |
| **none** | `none` | Disables FP8 quant within attention (falls back to BF16 attention path) |

#### Attention Kernel Backends

| Backend | Env `LUMEN_ATTN_KERNEL_BACKEND` | Description |
|---------|--------------------------------|-------------|
| **auto** (default) | `auto` | Automatically selects best available backend |
| **triton** | `triton` | Triton FP8 attention kernel (`fp8_attention_kernel` or `mxfp8_attention_kernel`) |
| **csrc** | `csrc` | CK (Composable Kernel) FP8 flash attention via AITER (`flash_attn_fp8_pertensor_fwd`) — inference-style per-tensor FP8 with Q/K/V descales |

#### Attention FP8 Internals

When `LUMEN_FP8_ATTN=dpa`:
1. QKV linear projections produce BF16 tensors (via the FP8 linear path if `LUMEN_FP8=1`)
2. `attention_fp8_quant()` quantizes Q, K, V to FP8 using the `ScalingManager`:
   - `quantize_block_fp8` for Q and K (blockwise scales)
   - `quantize_v_fp8` for V (produces V scales + `p_scale` for softmax scaling inside the kernel)
   - For MXFP8: `quantize_block_mxfp8` + `compute_p_scale_mxfp8`
3. Fused FP8 attention kernel computes `softmax(Q_fp8 @ K_fp8^T * scale) @ V_fp8` in FP8 arithmetic
4. Output is dequantized back to BF16

When `LUMEN_FP8_ATTN=mha`:
- Same as `dpa` plus `enable_fp8_for_parallel_linear(..., fp8_mha=True)` which configures the QKV and output projection linears to share a `Blockwise2DScaleManager`

**Experimental**: `LUMEN_FP8_ATTN_BWD=1` enables a hybrid path — CK BF16 forward + Triton FP8 backward (known issues with varlen/LSE; treat as experimental).

#### Attention FP8 Env Vars Summary

| Feature | Env Var | Default | Effect |
|---------|---------|---------|--------|
| FP8 Attention mode | `LUMEN_FP8_ATTN` | `none` | `none` / `dpa` / `mha` |
| Attention quant type | `LUMEN_FP8_QUANT_TYPE` | `blockwise` | Scaling method for Q/K/V quantization |
| Attention kernel backend | `LUMEN_ATTN_KERNEL_BACKEND` | `auto` | `auto` / `triton` / `csrc` |
| FP8 attention backward | `LUMEN_FP8_ATTN_BWD` | `0` | `1` = CK fwd + Triton FP8 bwd (experimental) |

**VERL wiring gap**: `VerlLumenArgs` currently has `lumen_fp8_attn` but **not** `lumen_fp8_quant_type` or `lumen_attn_backend`. These need to be added to `VerlLumenArgs` and wired as env vars in `verl_entry.py` before attention FP8 experiments. See [Code Changes Required](#code-changes-required).

---

## FP8-RL Paper Alignment: Gap Analysis

The FP8-RL paper ([arXiv:2601.18150](https://arxiv.org/abs/2601.18150)) describes three core techniques for FP8 in LLM RL. This section maps each technique to the current Lumen+VERL+vLLM stack, identifies what is implemented, what is missing, and what to do about each gap.

### Paper Method 1: FP8 W8A8 Blockwise Linear-Layer Rollout

**Paper**: FP8 W8A8 rollout using blockwise quantization (1x128 for activations, 128x128 for weights, following DeepSeek). Weights are quantized on-the-fly during weight sync from trainer to inference engine. Implemented via monkey-patching vLLM's `process_weights_after_loading` and a custom `quant_weights` generator that yields `(name, fp8_tensor)` + `(name + "_scale_inv", scale)` pairs.

**Current status: IMPLEMENTED (in VERL, used by Lumen)**

| Component | Status | Where |
|-----------|--------|-------|
| `quant_weights()` — on-the-fly BF16→FP8 blockwise quant during weight sync | Present | `verl/utils/vllm/vllm_fp8_utils.py:quant_weights()` |
| `scaled_fp8_blockwise()` — blockwise quant kernel | Present | `verl/utils/kernel/fp8_kernel.py` |
| `apply_vllm_fp8_patches()` — monkey-patch `process_weights_after_loading` for Linear + MoE | Present | `verl/utils/vllm/vllm_fp8_utils.py:apply_vllm_fp8_patches()` |
| `load_quanted_weights()` — loads quantized weights into vLLM model runner | Present | `verl/utils/vllm/vllm_fp8_utils.py:load_quanted_weights()` |
| vLLM version support | v0.10, v0.11, v0.12, v0.14+ | Version-gated patches in `apply_vllm_fp8_patches()` |
| Block shape: 128×128 weights, 1×128 activations | Present | `weight_block_size` from `quant_config`; activation is dynamic per-token in vLLM |
| Config knob | `actor_rollout_ref.rollout.quantization=fp8` | Hydra override via `common.sh` |

**Lumen integration**: Lumen's DAPO scripts set `ROLLOUT_QUANTIZATION="fp8"` which passes through to VERL's Hydra config. The actual vLLM FP8 patches live in **VERL**, not Lumen. Lumen does not modify the rollout weight-loading path.

**ROCm note**: The vLLM patches reference `rocm_aiter_moe_enabled` and `is_rocm_aiter_moe_enabled()`, confirming basic ROCm awareness. However, the `scaled_fp8_blockwise` kernel in VERL uses cuBLAS-oriented block shapes — on ROCm/MI350X, the actual FP8 GEMM dispatch goes through vLLM's ROCm AITER backend.

**Gap**: None for the core method. ROCm-specific testing is needed (covered in Experiments 1–3).

---

### Paper Method 2: FP8 KV-Cache with Per-Step QKV Scale Recalibration

**Paper**: Extends FP8 to the KV-cache to remove long-context memory bottlenecks. After each training step, a percentile-based calibration pass computes per-layer Q/K/V scales, which are reduced across ranks and injected into the inference engine before the next rollout. This allows the KV-cache to be stored in FP8 rather than BF16/FP16, reducing memory by ~50%.

**Current status: NOT IMPLEMENTED**

| Component | Status | Where |
|-----------|--------|-------|
| vLLM `kv_cache_dtype="fp8"` support | Not wired | VERL's rollout config has no `kv_cache_dtype` knob |
| Per-step QKV scale calibration | Not present | No `calibrate`, `kv_scale`, or `recalibrate` in `lumen/rl/verl/` or VERL rollout code |
| Scale reduction across ranks | Not present | No distributed scale sync for KV-cache |
| VERL config knob for KV-cache FP8 | Not present | No `rollout.kv_cache_dtype` or similar in VERL's YAML configs |

**Why it matters**: For long-context RL (max_response_length=20K as in our experiments), the KV-cache can dominate GPU memory during rollout. FP8 KV-cache halves this cost, enabling larger batch sizes or longer sequences without OOM.

**What to do**:

1. **vLLM side**: vLLM (CUDA) already supports `kv_cache_dtype="fp8_e4m3"` natively — it stores K/V in FP8 with per-tensor scales. On ROCm, check whether vLLM's FP8 KV-cache path works (it may require ROCm-compatible attention kernels that support FP8 K/V inputs). Test with:
   ```python
   # In vLLM engine config:
   kv_cache_dtype="fp8_e4m3"
   ```

2. **VERL side**: Add `rollout.kv_cache_dtype` Hydra config and pass it to `EngineArgs` during vLLM engine construction. This is a small config-plumbing change in `verl/workers/rollout/vllm_rollout/`.

3. **Per-step QKV scale recalibration**: This is the paper's novel contribution. It runs a short calibration pass after each training step to compute per-layer Q/K/V scales, then injects them into vLLM before the next rollout. Implementation requires:
   - A calibration hook in the training-to-rollout transition
   - Scale computation: `percentile(abs(tensor), p=99.9)` → `fp8_max / percentile_value`
   - All-reduce across FSDP ranks
   - Scale injection into vLLM's attention layers
   
   **This is the most complex gap and should be treated as a separate work item.**

4. **Simpler alternative**: Use vLLM's native FP8 KV-cache with **static scales** (no per-step recalibration). This loses the paper's dynamic calibration but still gets the memory savings. Test first, add recalibration later if needed.

---

### Paper Method 3: Rollout Correction via Token-Level Importance Sampling (TIS/MIS)

**Paper**: Mitigates train-inference mismatch (FP8 rollout vs BF16/FP8 training) via importance-sampling-based correction. Two variants:
- **Token-level TIS**: Clips per-token importance weights `w_t = π_train(a_t|s_t) / π_rollout(a_t|s_t)` to `[1/C, C]` (C=2 in experiments), applied to the policy gradient loss.
- **MIS (Multilevel IS)**: Sequence-level correction using the product of per-token ratios.

**Current status: IMPLEMENTED (in VERL, used by Lumen)**

| Component | Status | Where |
|-----------|--------|-------|
| Token-level TIS | Present | `verl/trainer/ppo/core_algos.py`, `verl/trainer/ppo/rollout_corr_helper.py` |
| MIS (sequence-level) | Present | Same files; controlled via `algorithm.rollout_correction.rollout_rs` |
| VERL config knobs | Present | `algorithm.rollout_correction.rollout_is`, `rollout_is_threshold` |
| Lumen script integration | Present | `common.sh` sets `ROLLOUT_IS="token"`, `ROLLOUT_IS_THRESHOLD=2` for FP8 runs |
| Loss integration | Present | `verl/workers/actor/dp_actor.py`, `verl/workers/utils/losses.py` |

**Gap**: None. TIS is fully implemented in VERL and correctly wired from Lumen's DAPO scripts.

---

### Paper Method 4 (Lumen Extension): FP8 E2E Training

**Paper**: The FP8-RL paper uses Transformer Engine (TE) blockwise FP8 for E2E training on NVIDIA H100. Lumen **extends this to AMD MI350X** using its own FP8 stack (AITER/CK/Triton) as a TE replacement.

**Current status: IMPLEMENTED (Lumen's main contribution)**

| Component | Status | Where |
|-----------|--------|-------|
| FP8ParamManager (weight storage) | Present | `lumen/quantize/fp8_params.py` |
| FP8 Linear GEMM (blockwise/MXFP8/per-tensor) | Present | `lumen/ops/quantize/linear.py` |
| Blockwise quant kernel | Present | `lumen/ops/quantize/ops.py:quant_fp8_blockwise_impl` |
| MXFP8 quant kernel | Present | `lumen/ops/quantize/ops.py:convert_to_mxfp8` |
| FP8 Activation Store | Present | `lumen/ops/quantize/linear.py` |
| FP8 Attention (DPA/MHA) | Present | `lumen/ops/attention/attention.py` |
| VERL integration | Present | `lumen/rl/verl/verl_entry.py` patches FSDP2 |

**Gap**: Env var wiring for blockwise/MXFP8/attention (documented in Code Changes Required section).

---

### Paper Method 5: FP8 / Low-Precision Optimizer States

**Paper**: The FP8-RL E2E configuration uses Transformer Engine's FP8 blockwise recipe for optimizer states, storing Adam first/second moments in FP8 rather than FP32. This halves optimizer memory (which dominates in FSDP2 with offloading) and is especially impactful for large MoE models.

**Current status: PARTIAL — plumbing exists, optimizer itself needs implementation/wiring**

#### What exists today

| Component | Status | Where |
|-----------|--------|-------|
| `VerlLumenArgs.use_8bit_adam` field | Present | `lumen/rl/verl/config.py` |
| `LumenConfig.use_8bit_adam` hint | Present | `lumen/config.py` — documented as "read by trainer to select 8-bit Adam from bitsandbytes" |
| `USE_8BIT_ADAM` env var read | Present | `lumen/rl/verl/verl_entry.py:main()` |
| BitsAndBytes `Adam8bit` (TRL path) | Present | `lumen/rl/trl/runner.py` — `_LumenGRPOTrainer` swaps optimizer when `use_8bit_adam=True` |
| `ScalingManager.register_fp8_optimizer_hook` | Present | `lumen/quantize/scaling_manager.py` — post-step hook to mark FP8 weights stale |
| `FP32MasterWeightOptimizer` wrapper | Present | `lumen/quantize/optimizer_manager.py` — FP32 master copy for BF16 model params |
| VERL `build_optimizer` with `optimizer_impl` | Present | `verl/workers/config/optimizer.py` — supports `torch.optim`, `torchao.optim`, `bitsandbytes.optim` |

#### What is missing

| Component | Gap | Impact |
|-----------|-----|--------|
| **VERL FSDP path**: `use_8bit_adam` is read but **not applied** — `verl_entry.py` passes it to `VerlLumenArgs` but never swaps VERL's optimizer | `USE_8BIT_ADAM=1` silently does nothing in VERL FSDP runs | Must patch VERL's optimizer construction or use `override_optimizer_config` |
| **`any_lumen` gate**: `use_8bit_adam` is only read inside the `if any_lumen:` block, but it doesn't contribute to `any_lumen` itself | Setting `USE_8BIT_ADAM=1` alone (without other Lumen flags) won't trigger Lumen patching | Fix `any_lumen` logic or allow standalone 8-bit Adam |
| **BitsAndBytes + FSDP2 DTensor**: BNB `Adam8bit` has known compatibility issues with FSDP2's DTensor parameter wrapping | May crash or silently fall back to FP32 | Needs testing; TorchAO `_AdamW` with `bf16_stochastic_round` may be more FSDP2-friendly |
| **FP8 blockwise optimizer** (TE-style): No implementation of FP8 optimizer states in Lumen | Cannot match the paper's E2E FP8 optimizer configuration | Implement via TorchAO's quantized optimizer or custom FP8 state quantization |

#### Implementation options (ordered by effort)

**Option A: VERL `override_optimizer_config` (low effort)**
Use VERL's built-in `override_optimizer_config` to swap the optimizer without modifying Lumen:
```yaml
actor_rollout_ref.actor.optim.optimizer_impl: "bitsandbytes.optim"
actor_rollout_ref.actor.optim.optimizer: "AdamW8bit"
```
Or for TorchAO (better FSDP2 compatibility):
```yaml
actor_rollout_ref.actor.optim.optimizer_impl: "torchao.optim"
actor_rollout_ref.actor.optim.optimizer: "_AdamW"
actor_rollout_ref.actor.optim.override_optimizer_config:
  bf16_stochastic_round: true
```
Pro: Zero Lumen code changes. Con: 8-bit (INT8), not FP8; no blockwise FP8 states.

**Option B: Wire `USE_8BIT_ADAM` through VERL's optimizer config (medium effort)**
Modify `verl_entry.py` to inject `optimizer_impl` and `optimizer` overrides into the Hydra config when `USE_8BIT_ADAM=1`:
```python
if use_8bit_adam:
    overrides.append("actor_rollout_ref.actor.optim.optimizer_impl=bitsandbytes.optim")
    overrides.append("actor_rollout_ref.actor.optim.optimizer=AdamW8bit")
```
Pro: Env var–driven, consistent with other Lumen flags. Con: Still 8-bit INT8, not FP8.

**Option C: FP8 blockwise optimizer states via TorchAO (higher effort)**
Implement a custom `FP8AdamW` that stores first/second moments in FP8 with blockwise scales, matching TE's approach. TorchAO provides building blocks (`Float8Tensor`, `ScaledGroupedMMTensor`) that could be adapted for optimizer states.
Pro: True FP8 optimizer matching the paper. Con: Significant development effort; unclear if the memory savings justify the precision risk for RL.

**Option D: Quantized optimizer states via `torch.optim` hooks (higher effort)**
Use PyTorch's optimizer state hooks (`register_load_state_dict_pre_hook`, etc.) to quantize states to FP8 after each step and dequantize before the next step. Similar to what TE does internally.
Pro: Flexible, works with any base optimizer. Con: Custom implementation needed; performance overhead from quant/dequant.

---

**Gap**: Env var wiring for blockwise/MXFP8/attention (documented in Code Changes Required section).

---

### Summary: Gap Analysis Matrix

| FP8-RL Paper Technique | Lumen | VERL | vLLM | Status | Action Required |
|------------------------|-------|------|------|--------|-----------------|
| **W8A8 blockwise rollout** | Config only | Implemented | Used via patches | **Ready** | ROCm validation only |
| **FP8 KV-cache** | Not present | Not present | Supported (CUDA) | **GAP** | Wire `kv_cache_dtype`; test on ROCm; add recalibration later |
| **TIS/MIS rollout correction** | Script config | Implemented | N/A | **Ready** | None |
| **E2E FP8 training (blockwise)** | Implemented | N/A (uses TE on NVIDIA) | N/A | **Ready** | Wire env vars (Step 3.5) |
| **E2E FP8 training (MXFP8)** | Implemented | N/A | N/A | **Ready** | Remove block_size clamp + wire env vars |
| **E2E FP8 attention** | Implemented | N/A | N/A | **Ready** | Wire quant_type + backend env vars (Step 3.6) |
| **Low-precision optimizer** | Plumbing only | `build_optimizer` supports BNB/TorchAO | N/A | **DEV** | Wire `USE_8BIT_ADAM` or `override_optimizer_config`; test BNB/TorchAO with FSDP2 |

### Priority Action Items

1. **Immediate (before experiments)**: Wire `LUMEN_FP8_SCALING`, `LUMEN_FP8_FORMAT`, `LUMEN_FP8_BLOCK_SIZE` env vars (Step 3.5) and attention env vars (Step 3.6)
2. **Immediate**: Remove MXFP8 block_size clamp in `quantize_input`
3. **Short-term**: Wire `kv_cache_dtype` from VERL config into vLLM `EngineArgs` — small config change, big memory win
4. **Short-term**: Test vLLM FP8 KV-cache on ROCm (`kv_cache_dtype="fp8_e4m3"`) — may work out of the box
5. **Short-term**: Wire `USE_8BIT_ADAM` through VERL's `override_optimizer_config` (Option B) — enables BNB `AdamW8bit` or TorchAO `_AdamW` with `bf16_stochastic_round` for FSDP2 runs
6. **Medium-term**: Test BNB `AdamW8bit` vs TorchAO `_AdamW` with FSDP2 DTensor on MI350X (compatibility + memory savings)
7. **Medium-term**: Implement per-step QKV scale recalibration for KV-cache (paper's novel contribution)
8. **Long-term**: FP8 blockwise optimizer states (Option C/D) — true paper parity, but high effort

---

## Experiment Matrix

### Design Rationale

All variants within an experiment use the **same model** so results are directly comparable on a single chart. Experiments 2 and 3 both test the 30B MoE architecture but on different model variants (Base vs Instruct), giving independent confirmation of FP8 alignment.

### Experiment 1: Qwen3-8B-Base Dense — FP8 Rollout + FP8 E2E

**Reference**: [Qwen3-8B-Base Dense Model](https://github.com/verl-project/verl/blob/main/docs/advance/fp8.md#qwen3-8b-base-dense-model)

| Run ID | Training | Rollout | TIS | Script |
|--------|----------|---------|-----|--------|
| 1A | BF16 (FSDP2) | BF16 | — | `run_dapo_qwen3_8b_bf16.sh` |
| 1B | BF16 (FSDP2) | FP8 + TIS | token, C=2 | `run_dapo_qwen3_8b_fp8_rollout_tis.sh` |
| 1C | BF16 (FSDP2) | FP8 | — | `run_dapo_qwen3_8b_fp8_rollout_no_tis.sh` |
| **1D** | **FP8 E2E (blockwise 128)** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_8b_fp8_e2e_blockwise.sh`** |
| **1E** | **FP8 E2E (MXFP8 block=32)** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_8b_fp8_e2e_mxfp8_b32.sh`** |
| **1E-64** | **FP8 E2E (MXFP8 block=64)** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_8b_fp8_e2e_mxfp8_b64.sh`** |
| **1E-128** | **FP8 E2E (MXFP8 block=128)** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_8b_fp8_e2e_mxfp8_b128.sh`** |
| **1F** | **FP8 E2E (per-tensor delayed)** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_8b_fp8_e2e_lumen.sh`** |
| **4A** | **FP8 E2E (blockwise 128) + DPA blockwise attn** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_8b_fp8_e2e_blockwise_attn_dpa.sh`** |
| **4B** | **FP8 E2E (blockwise 128) + DPA MXFP8 attn** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_8b_fp8_e2e_blockwise_attn_dpa_mxfp8.sh`** |
| **4C** | **FP8 E2E (blockwise 128) + DPA dynamic attn** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_8b_fp8_e2e_blockwise_attn_dpa_dynamic.sh`** |
| **4D** | **FP8 E2E (blockwise 128) + MHA blockwise2d attn** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_8b_fp8_e2e_blockwise_attn_mha.sh`** |
| **5A** | **FP8 E2E (blockwise 128) + BNB AdamW8bit** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_8b_fp8_e2e_blockwise_optim_bnb8bit.sh`** |
| **5B** | **FP8 E2E (blockwise 128) + TorchAO _AdamW bf16** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_8b_fp8_e2e_blockwise_optim_torchao.sh`** |

**Comparison chart**: 1A vs 1B vs 1C vs 1D vs 1E vs 1E-64 vs 1E-128 vs 1F vs 4A vs 4B vs 4C vs 4D vs 5A vs 5B on the same axes.
- 1A = pure BF16 baseline
- 1B = FP8 rollout with TIS (should align with 1A)
- 1C = FP8 rollout without TIS (ablation: expected accuracy drop)
- 1D = Lumen FP8 E2E blockwise: **blockwise scaling** (128-element blocks, matching FP8-RL paper/TE) — run first
- 1E = Lumen FP8 E2E MXFP8 block=32: finest MXFP8 granularity (OCP MX default) — run second
- 1E-64 = Lumen FP8 E2E MXFP8 block=64: medium MXFP8 granularity
- 1E-128 = Lumen FP8 E2E MXFP8 block=128: coarsest MXFP8 (matches blockwise scale count)
- 1F = Lumen FP8 E2E per-tensor: per-tensor delayed scaling + Lumen Norm + fused kernels — run third
- 4A = Lumen FP8 E2E blockwise + **DPA blockwise** attention (default attn FP8, matching linear granularity)
- 4B = Lumen FP8 E2E blockwise + **DPA MXFP8** attention (finest-grained E8M0 attn scaling)
- 4C = Lumen FP8 E2E blockwise + **DPA dynamic** attention (coarsest: single per-tensor scale on Q/K/V)
- 4D = Lumen FP8 E2E blockwise + **MHA blockwise2d** attention (most aggressive: QKV proj + attn + output proj all FP8)
- 5A = Lumen FP8 E2E blockwise + **BNB AdamW8bit** (INT8 optimizer states) — same as 1D except optimizer
- 5B = Lumen FP8 E2E blockwise + **TorchAO _AdamW** (BF16 stochastic round optimizer) — same as 1D except optimizer

**Config** (adapted for 8x MI350):

| Parameter | Value | Ref (H100) | Notes |
|-----------|-------|------------|-------|
| Model | Qwen/Qwen3-8B-Base | Same | Dense, 8B params |
| GPUs | 8 | 8 | Same |
| Prompt batch size | 32 | 32 | Same |
| Responses per prompt (n) | 16 | 16 | Same |
| Train batch size | 32 | 32 | Same |
| Gen batch size | 32 | 32×3=96 | Reduced for memory (MI350 ROCm) |
| PPO mini batch size | 32 | 32 | Same |
| PPO micro batch size | 1 | 1 | Same |
| Log prob micro batch size | 1 | 2 | Reduced for MI350 memory |
| Max prompt length | 1024 | 1024 | Same |
| Max response length | 20480 (20K) | 20K | Same |
| LR | 1e-6 | 1e-6 | Same |
| Clip (low/high) | 0.2 / 0.28 | Same | DAPO decoupled clip |
| Loss aggregation | token-mean | Same | DAPO token-level loss |
| Reward manager | dapo | Same | Math accuracy scoring |
| Overlong buffer | enable=True, len=512, penalty=1.0 | Same | Same as reference |
| Total steps | **275** | 500 | Reduced for speed (~5 min/step avg on MI350) |
| Val frequency | 5 steps | 5 | Same (yields 55 validation points) |
| Save frequency | 20 steps | 5 | Reduced I/O frequency |
| FSDP2 offload | param + optimizer | Same | Memory savings |
| vLLM gpu_memory_util | **0.3** | 0.9 | MI350 ROCm: vLLM sleep() is no-op, must use low reservation + free_cache_engine |
| free_cache_engine | **True** | N/A | Essential on ROCm: explicitly frees vLLM KV cache between generation and training |
| Rollout TP | 1 | 1 | ROCm vLLM TP=1 |
| enforce_eager | True | False | Required for ROCm vLLM |

**MI350 adaptations summary**: `gpu_memory_utilization=0.3` + `free_cache_engine=True` + `log_prob_micro_batch_size=1` + `gen_batch_size=train_batch_size` to avoid OOM (vLLM `sleep()` is a no-op on ROCm, so KV cache must be explicitly freed).

**Expected outcome**: 1A ≈ 1B ≈ 1D ≈ 1E ≈ 1E-64 ≈ 1E-128 ≈ 1F ≈ 4A ≈ 4B ≈ 4C ≈ 5A ≈ 5B (aligned curves). 1C shows accuracy drop (no TIS). 4D (full MHA FP8) may diverge — it is the most aggressive configuration. Expected quantization noise ordering: per-tensor (1F) > blockwise (1D) ≈ MXFP8-128 (1E-128) > MXFP8-64 (1E-64) > MXFP8-32 (1E). 4A/4B/4C should align with 1D (same linear FP8, only attention changes). 5A/5B should align with 1D (same linear FP8, different optimizer only). The MXFP8 sweep reveals how block granularity affects alignment — smaller blocks should align better with BF16. The attention sweep reveals whether FP8 attention on top of FP8 linear causes additional alignment regression. The optimizer sweep reveals whether low-precision optimizer states cause any alignment regression when combined with FP8 E2E training.

---

### Experiment 2: Qwen3-30B-A3B-Base MoE — FP8 Rollout + FP8 E2E (Unified)

**Reference**: [Qwen3-30B-A3B-Base MoE](https://github.com/verl-project/verl/blob/main/docs/advance/fp8.md#qwen3-30b-a3b-base-moe-model) + [Qwen3-30B-A3B E2E](https://github.com/verl-project/verl/blob/main/docs/advance/fp8.md#qwen3-30b-a3b-moe-model)

| Run ID | Training | Rollout | TIS | Script |
|--------|----------|---------|-----|--------|
| 2A | BF16 (FSDP2) | BF16 + TIS | token, C=2 | `run_dapo_qwen3_30b_moe_bf16_tis.sh` |
| 2B | BF16 (FSDP2) | FP8 + TIS | token, C=2 | `run_dapo_qwen3_30b_moe_fp8_rollout_tis.sh` |
| **2C** | **FP8 E2E (blockwise 128)** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_30b_moe_fp8_e2e_blockwise.sh`** |
| **2D** | **FP8 E2E (MXFP8 block=32)** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_30b_moe_fp8_e2e_mxfp8_b32.sh`** |
| **2D-64** | **FP8 E2E (MXFP8 block=64)** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_30b_moe_fp8_e2e_mxfp8_b64.sh`** |
| **2D-128** | **FP8 E2E (MXFP8 block=128)** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_30b_moe_fp8_e2e_mxfp8_b128.sh`** |
| **2E** | **FP8 E2E (per-tensor delayed)** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_30b_moe_fp8_e2e_lumen.sh`** |

**Comparison chart**: 2A vs 2B vs 2C vs 2D vs 2D-64 vs 2D-128 vs 2E on the same axes.
- 2A = BF16 baseline (with TIS, needed for MoE rollout correction even in BF16)
- 2B = FP8 rollout only (should align with 2A)
- 2C = Lumen FP8 E2E blockwise: **blockwise scaling** (128-element blocks, matching FP8-RL paper/TE) — run first
- 2D = Lumen FP8 E2E MXFP8 block=32: finest MXFP8 granularity — run second
- 2D-64 = Lumen FP8 E2E MXFP8 block=64: medium MXFP8 granularity
- 2D-128 = Lumen FP8 E2E MXFP8 block=128: coarsest MXFP8 (matches blockwise scale count)
- 2E = Lumen FP8 E2E per-tensor: per-tensor delayed scaling + Lumen Norm + FP8 Activation Store + fused kernels — run third

**Config adaptation for 8 GPUs** (reference used 2×8=16):

| Parameter | Value | Ref (H100) | Notes |
|-----------|-------|------------|-------|
| Model | Qwen/Qwen3-30B-A3B-Base | Same | MoE: 128 experts, 8 active |
| GPUs | 8 | 16 (2×8) | Half of reference |
| Prompt batch size | 16 | 32 | Halved for 8 GPUs |
| Responses per prompt (n) | 16 | 16 | Same |
| Train batch size | 16 | 32 | Halved |
| Gen batch size | 16 | 32×3=96 | Reduced for memory |
| PPO mini batch size | 16 | 32 | Halved |
| PPO micro batch size | 1 | 1 | Same |
| Log prob micro batch size | 1 | 2 | Reduced for MI350 memory |
| Max response length | 20480 | 20K | Same |
| FSDP2 offload | param + optimizer | Same | Essential for memory |
| SP size | 4 | 4 | Ulysses sequence parallelism |
| vLLM gpu_memory_util | **0.3** | 0.9 | MI350 ROCm adaptation |
| free_cache_engine | **True** | N/A | Essential on ROCm |
| Rollout TP | 1 | 1 | ROCm constraint |
| enforce_eager | True | False | Required for ROCm vLLM |
| Total steps | **275** | 500 | Reduced for speed |

**Lumen FP8 E2E config** (shared across all E2E runs):

| Parameter | Blockwise (1D/2C/3C) | MXFP8-32 (1E/2D/3D) | MXFP8-64 (1E-64/2D-64/3D-64) | MXFP8-128 (1E-128/2D-128/3D-128) | Per-tensor (1F/2E/3E) |
|-----------|----------------------|----------------------|-------------------------------|-----------------------------------|----------------------|
| `FP8_PARAM_MANAGER` | 1 | 1 | 1 | 1 | 1 |
| `LUMEN_FP8` | 1 | 1 | 1 | 1 | 1 |
| `LUMEN_FP8_SCALING` | `blockwise` | `blockwise` | `blockwise` | `blockwise` | `delayed` (default) |
| `LUMEN_FP8_FORMAT` | `fp8_e4m3` (default) | `mxfp8` | `mxfp8` | `mxfp8` | `fp8_e4m3` (default) |
| `LUMEN_FP8_BLOCK_SIZE` | `128` | `32` | `64` | `128` | — (not used) |
| `LUMEN_NORM` | 1 | 1 | 1 | 1 | 1 |
| `LUMEN_FP8_ACTIVATION_STORE` | 1 (MoE only) | 1 (MoE only) | 1 (MoE only) | 1 (MoE only) | 1 (MoE only) |
| Fused kernels (auto) | quant+amax, cast+transpose, quant_scale | same | same | same | same |
| Entry point | `lumen.rl.verl.verl_entry` | same | same | same | same |

**Note**: `LUMEN_FP8_SCALING`, `LUMEN_FP8_FORMAT`, and `LUMEN_FP8_BLOCK_SIZE` are new env vars that need to be wired into `verl_entry.py` (currently it hardcodes `linear_fp8_scaling="delayed"`). The wiring maps to `VerlLumenArgs.linear_fp8_scaling`, `VerlLumenArgs.linear_fp8_format`, and `LumenConfig.block_size` respectively. For the MXFP8 sweep, `LUMEN_FP8_BLOCK_SIZE` is set to 32, 64, or 128 directly — the code clamp in `quantize_input` must be removed first (see [Code Changes Required](#code-changes-required)).

**Expected outcome**: 2A ≈ 2B ≈ 2C ≈ 2D ≈ 2D-64 ≈ 2D-128 ≈ 2E (aligned curves). Expected mismatch KL ordering: 2E > 2D-128 ≈ 2C > 2D-64 > 2D > 2B > 2A. The MXFP8 sweep reveals the block-size/accuracy trade-off on MoE — smaller blocks should produce lower mismatch KL. Higher overall mismatch KL for MoE vs dense (matching reference observation).

---

### Experiment 3: Qwen3-30B-A3B MoE (Instruct) — FP8 Rollout + FP8 E2E

**Reference**: [Qwen3-30B-A3B MoE Model (FP8 End-to-End)](https://github.com/verl-project/verl/blob/main/docs/advance/fp8.md#qwen3-30b-a3b-moe-model)

This experiment uses the **instruct** (non-Base) variant of the 30B MoE model, matching the VERL reference's third benchmark. It provides independent confirmation that FP8 alignment holds on an instruction-tuned starting point, not just a base model.

| Run ID | Training | Rollout | TIS | Script |
|--------|----------|---------|-----|--------|
| 3A | BF16 (FSDP2) | BF16 + TIS | token, C=2 | `run_dapo_qwen3_30b_moe_instruct_bf16_tis.sh` |
| 3B | BF16 (FSDP2) | FP8 + TIS | token, C=2 | `run_dapo_qwen3_30b_moe_instruct_fp8_rollout_tis.sh` |
| **3C** | **FP8 E2E (blockwise 128)** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_30b_moe_instruct_fp8_e2e_blockwise.sh`** |
| **3D** | **FP8 E2E (MXFP8 block=32)** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_30b_moe_instruct_fp8_e2e_mxfp8_b32.sh`** |
| **3D-64** | **FP8 E2E (MXFP8 block=64)** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_30b_moe_instruct_fp8_e2e_mxfp8_b64.sh`** |
| **3D-128** | **FP8 E2E (MXFP8 block=128)** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_30b_moe_instruct_fp8_e2e_mxfp8_b128.sh`** |
| **3E** | **FP8 E2E (per-tensor delayed)** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_30b_moe_instruct_fp8_e2e_lumen.sh`** |

**Comparison chart**: 3A vs 3B vs 3C vs 3D vs 3D-64 vs 3D-128 vs 3E on the same axes.
- 3A = BF16 baseline (instruct model, with TIS)
- 3B = FP8 rollout only (should align with 3A)
- 3C = Lumen FP8 E2E blockwise: **blockwise scaling** (128-element blocks) — run first
- 3D = Lumen FP8 E2E MXFP8 block=32: finest MXFP8 granularity — run second
- 3D-64 = Lumen FP8 E2E MXFP8 block=64: medium MXFP8 granularity
- 3D-128 = Lumen FP8 E2E MXFP8 block=128: coarsest MXFP8 (matches blockwise scale count)
- 3E = Lumen FP8 E2E per-tensor: per-tensor delayed scaling — run third

**Config**: Same as Experiment 2, except:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen/Qwen3-30B-A3B | MoE instruct (non-Base) |

All other parameters (batch sizes, offload, SP, gpu_memory_util=0.3, free_cache_engine=True, 275 steps, etc.) are identical to Experiment 2.

**Cross-experiment comparison**: After all runs complete, overlay 2A–2E vs 3A–3E to compare Base vs Instruct model behavior under DAPO RL + FP8 across all scaling methods including the MXFP8 block-size sweep.

**Expected outcome**: 3A ≈ 3B ≈ 3C ≈ 3D ≈ 3D-64 ≈ 3D-128 ≈ 3E (aligned curves). Instruct model may show different absolute reward/val_score levels compared to Base, but the relative FP8-vs-BF16 alignment should hold across all scaling methods. Expected quantization noise ordering: per-tensor (3E) > blockwise (3C) ≈ MXFP8-128 (3D-128) > MXFP8-64 (3D-64) > MXFP8-32 (3D).

---

### Experiment 4: FP8 Attention Sweep — Qwen3-8B-Base Dense

Linear FP8 and attention FP8 are **independent quantization axes** in Lumen — they use separate scaling methods, separate env vars, and separate kernel paths. Experiments 1–3 sweep the **linear** axis (blockwise / MXFP8 / per-tensor) with attention kept in BF16. Experiment 4 **fixes linear to blockwise** (the closest match to the VERL reference) and **sweeps the attention axis** across all supported modes and quant types.

**Design**: Linear is locked to `LUMEN_FP8_SCALING=blockwise` + `LUMEN_FP8_BLOCK_SIZE=128` for every run. Only the attention-side env vars change.

**Baseline**: Run 1D (blockwise FP8 linear, BF16 attention) from Experiment 1.

| Run ID | Linear FP8 (fixed) | Attention Mode | Attn Quant Type | Rollout | TIS | Script |
|--------|-------------------|----------------|-----------------|---------|-----|--------|
| 1D | blockwise 128 | **none** (BF16 attn) | — | FP8 + TIS | token, C=2 | `run_dapo_qwen3_8b_fp8_e2e_blockwise.sh` (from Exp 1) |
| **4A** | blockwise 128 | **dpa** | **blockwise** | FP8 + TIS | token, C=2 | `run_dapo_qwen3_8b_fp8_e2e_blockwise_attn_dpa.sh` |
| **4B** | blockwise 128 | **dpa** | **mxfp8** | FP8 + TIS | token, C=2 | `run_dapo_qwen3_8b_fp8_e2e_blockwise_attn_dpa_mxfp8.sh` |
| **4C** | blockwise 128 | **dpa** | **dynamic** | FP8 + TIS | token, C=2 | `run_dapo_qwen3_8b_fp8_e2e_blockwise_attn_dpa_dynamic.sh` |
| **4D** | blockwise 128 | **mha** | **blockwise2d** | FP8 + TIS | token, C=2 | `run_dapo_qwen3_8b_fp8_e2e_blockwise_attn_mha.sh` |

**Comparison chart**: 1D vs 4A vs 4B vs 4C vs 4D on the same axes.
- 1D = blockwise FP8 linear, **BF16 attention** (baseline — no attention FP8)
- 4A = blockwise FP8 linear + **DPA blockwise** attention (default attn FP8, matching linear granularity)
- 4B = blockwise FP8 linear + **DPA MXFP8** attention (finest-grained E8M0 attn scaling)
- 4C = blockwise FP8 linear + **DPA dynamic** attention (coarsest: single per-tensor scale on Q/K/V)
- 4D = blockwise FP8 linear + **full MHA blockwise2d** attention (most aggressive: QKV proj + attn + output proj all FP8, shared `Blockwise2DScaleManager`)

**Config**: Same as Experiment 1 (8B dense, 8x MI350). Linear env vars are **identical** across all runs (same as 1D). Only the attention env vars differ:

| Env var | 1D (baseline) | 4A | 4B | 4C | 4D |
|---------|--------------|-----|-----|-----|-----|
| `LUMEN_FP8` | 1 | 1 | 1 | 1 | 1 |
| `LUMEN_FP8_SCALING` | blockwise | blockwise | blockwise | blockwise | blockwise |
| `LUMEN_FP8_BLOCK_SIZE` | 128 | 128 | 128 | 128 | 128 |
| `LUMEN_FP8_ATTN` | **none** | **dpa** | **dpa** | **dpa** | **mha** |
| `LUMEN_FP8_QUANT_TYPE` | — | **blockwise** | **mxfp8** | **dynamic** | **blockwise2d** |
| `LUMEN_ATTN_KERNEL_BACKEND` | — | auto | auto | auto | auto |

**Expected outcome**: 1D ≈ 4A ≈ 4B ≈ 4C ≈ 4D (aligned curves). Adding FP8 attention should not degrade alignment when linear FP8 is already stable. Expected attention quantization noise ordering: dynamic (4C) > blockwise (4A) > MXFP8 (4B). MHA mode (4D) is the most aggressive — if it aligns, the full Lumen FP8 stack (linear + attention + norms) is validated end-to-end.

**Key questions this experiment answers**:
1. Does adding FP8 attention on top of blockwise FP8 linear training maintain BF16 alignment?
2. Which attention quant type (blockwise vs MXFP8 vs dynamic) gives the best accuracy/throughput trade-off?
3. Is full MHA FP8 (4D) stable for RL training, or does it cause divergence?
4. What is the throughput improvement from FP8 attention (tokens/sec, `timing_s/update_actor`)?
5. Can linear and attention use different scaling granularities without interaction effects?

---

### Experiment 5: Low-Precision Optimizer — Qwen3-8B-Base Dense

This experiment tests **low-precision optimizer states** as an independent axis from linear/attention FP8. The optimizer is the third major memory consumer (alongside weights and activations); quantizing optimizer states reduces memory by ~50%, enabling larger batch sizes or reduced CPU offloading.

**Design**: Linear is fixed to blockwise, attention is BF16 (same as 1D). Only the optimizer implementation changes. This isolates the optimizer's impact on alignment.

**Baseline**: Run 1D (blockwise FP8 E2E with standard PyTorch AdamW) from Experiment 1.

| Run ID | Linear FP8 (fixed) | Attention | Optimizer | Rollout | TIS | Script |
|--------|-------------------|-----------|-----------|---------|-----|--------|
| 1D | blockwise 128 | none (BF16) | **PyTorch AdamW (FP32 states)** | FP8 + TIS | token, C=2 | `run_dapo_qwen3_8b_fp8_e2e_blockwise.sh` (from Exp 1) |
| **5A** | blockwise 128 | none (BF16) | **BNB AdamW8bit** | FP8 + TIS | token, C=2 | `run_dapo_qwen3_8b_fp8_e2e_blockwise_optim_bnb8bit.sh` |
| **5B** | blockwise 128 | none (BF16) | **TorchAO _AdamW (bf16 stochastic round)** | FP8 + TIS | token, C=2 | `run_dapo_qwen3_8b_fp8_e2e_blockwise_optim_torchao.sh` |

**Comparison chart**: 1D vs 5A vs 5B on the same axes.
- 1D = blockwise FP8 E2E with standard PyTorch AdamW (FP32 optimizer states) — baseline
- 5A = + BitsAndBytes `AdamW8bit` (INT8 quantized first/second moments)
- 5B = + TorchAO `_AdamW` with `bf16_stochastic_round=True` (BF16 optimizer states with stochastic rounding)

**Config**: Same as Experiment 1. Linear and attention env vars are identical to 1D. Only the optimizer is different, controlled via VERL's `override_optimizer_config`:

| VERL config | 1D (baseline) | 5A | 5B |
|-------------|--------------|-----|-----|
| `actor_rollout_ref.actor.optim.optimizer_impl` | `torch.optim` (default) | `bitsandbytes.optim` | `torchao.optim` |
| `actor_rollout_ref.actor.optim.optimizer` | `AdamW` (default) | `AdamW8bit` | `_AdamW` |
| `actor_rollout_ref.actor.optim.override_optimizer_config` | — | — | `{bf16_stochastic_round: true}` |

**Expected outcome**: 1D ≈ 5A ≈ 5B (aligned curves). Low-precision optimizer states should not significantly degrade RL alignment since the gradient signal (computed in FP8/BF16) dominates training dynamics. Expected memory reduction: ~40–50% on optimizer states.

**Key questions this experiment answers**:
1. Does quantizing optimizer states to 8-bit/BF16 maintain BF16 alignment when combined with FP8 E2E training?
2. Is BNB `AdamW8bit` compatible with FSDP2 DTensor on MI350X, or does TorchAO work better?
3. What is the actual memory savings from low-precision optimizer states (peak GPU memory)?
4. How does optimizer quantization interact with FP8 weight quantization (`FP8ParamManager`)?

---

## Metrics to Track

### Accuracy Metrics (alignment comparison — main result)

These determine whether FP8 aligns with BF16. Plot on shared axes per experiment.

| Metric | Log key | Description |
|--------|---------|-------------|
| **Validation accuracy** | `val-core/math_dapo/acc/mean@1` | AIME-2024 accuracy (primary metric) |
| **Reward** | `critic/rewards/mean` | DAPO reward signal |
| **Score** | `critic/score/mean` | DAPO math accuracy score |
| **Mismatch KL** | `rollout_corr/kl` | Rollout/training distribution KL divergence |
| **Response length** | `response_length/mean` | Average generation length |

### Performance Metrics (throughput comparison)

These measure FP8 speedup. Record per-run averages.

| Metric | Log key | Description |
|--------|---------|-------------|
| **Throughput** | `perf/throughput` | Tokens/second |
| **Step time** | `timing_s/step` | Seconds per training step |
| **Rollout time** | `timing_s/gen` | Seconds for generation phase |
| **Update time** | `timing_s/update_actor` | Seconds for actor gradient update |
| **Log prob time** | `timing_s/old_log_prob` | Seconds for reference log prob |

### Expected Results Format (matching [VERL FP8 docs](https://github.com/verl-project/verl/blob/main/docs/advance/fp8.md))

For each experiment, produce:
1. **Accuracy chart**: `val-core/acc@1` vs training step (all variants overlaid)
2. **Performance chart**: `timing_s/gen` (rollout time) vs step (FP8 vs BF16 speedup)
3. **Summary table**: avg throughput, rollout speedup %, mismatch KL range
4. **Scaling method comparison**: per-tensor vs blockwise vs MXFP8-32 vs MXFP8-64 vs MXFP8-128 for each model — alignment quality, mismatch KL, and training throughput
5. **MXFP8 block-size sweep**: overlay MXFP8-32 vs MXFP8-64 vs MXFP8-128 per experiment to isolate the block-size/accuracy trade-off
6. **Attention FP8 sweep** (Experiment 4): 1D (no attn FP8) vs 4A (DPA blockwise) vs 4B (DPA MXFP8) vs 4C (DPA dynamic) vs 4D (MHA blockwise2d) — alignment, throughput, and `timing_s/update_actor` improvement. Linear is fixed to blockwise 128 across all runs.
7. **Low-precision optimizer sweep** (Experiment 5): 1D (AdamW FP32 states) vs 5A (BNB AdamW8bit) vs 5B (TorchAO _AdamW bf16 stochastic round) — alignment, **peak GPU memory** (key metric), and throughput. Linear is fixed to blockwise 128, attention is BF16.

---

## Execution Order

Following the user's roadmap:

### Step 1: Understand the training method ✅
- Read VERL FP8 docs, DAPO recipe, rollout correction
- Identify Lumen's FP8PM as the substitution for Transformer Engine

### Step 2: Write test plan and scripts (this document + launch scripts)
- Write this plan → `Lumen/outputs/fp8_training_alignment/FP8_TRAINING_ALIGNMENT_PLAN.md`
- Write launch scripts → `Lumen/examples/rl/verl/dapo/`

### Step 3: BF16 baseline tests
- Run 1A (8B BF16 baseline), 2A (30B MoE Base BF16 + TIS), 3A (30B MoE Instruct BF16 + TIS)
- Verify training converges, metrics are logged correctly
- Confirm curves match expected DAPO behavior (improving val_score, increasing response_length)

### Step 3.5: Wire blockwise/MXFP8 env vars into verl_entry.py
- Add `LUMEN_FP8_SCALING`, `LUMEN_FP8_FORMAT`, `LUMEN_FP8_BLOCK_SIZE` env var reads to `verl_entry.py:main()`
- Map to `VerlLumenArgs.linear_fp8_scaling`, `VerlLumenArgs.linear_fp8_format`, and a new `linear_fp8_block_size` field
- **Remove the `mxfp8_block = 32 if block_size > 64 else block_size` clamp** in `quantize_input` (`lumen/ops/quantize/linear.py:195`) to allow block_size=128 for MXFP8
- Smoke test blockwise and MXFP8 (all 3 block sizes: 32, 64, 128) with 2-step runs before full experiments

### Step 3.6: Wire attention FP8 env vars into verl_entry.py
- Add `LUMEN_FP8_QUANT_TYPE`, `LUMEN_ATTN_KERNEL_BACKEND` env var reads to `verl_entry.py:main()`
- Add `lumen_fp8_quant_type` and `lumen_attn_backend` fields to `VerlLumenArgs`
- Smoke test `LUMEN_FP8_ATTN=dpa` with a 2-step run on 8B before full Experiment 4

### Step 4: FP8 tests

Priority order: **blockwise first** (closest match to VERL reference TE blockwise), **then MXFP8** (finest granularity), **then per-tensor** (coarsest, fallback).

**4a. FP8 rollout-only (BF16 training)**
- Run 1B, 1C (FP8 rollout ± TIS for 8B)
- Run 2B (FP8 rollout + TIS for 30B MoE Base)
- Run 3B (FP8 rollout + TIS for 30B MoE Instruct)

**4b. FP8 E2E — blockwise (priority 1: matches FP8-RL paper)**
- Run 1D (8B blockwise)
- Run 2C (30B MoE Base blockwise)
- Run 3C (30B MoE Instruct blockwise)

**4c. FP8 E2E — MXFP8 block-size sweep (priority 2: finest-grained scaling)**
- Run 1E, 1E-64, 1E-128 (8B MXFP8 block=32/64/128)
- Run 2D, 2D-64, 2D-128 (30B MoE Base MXFP8 block=32/64/128)
- Run 3D, 3D-64, 3D-128 (30B MoE Instruct MXFP8 block=32/64/128)

**4d. FP8 E2E — per-tensor delayed (priority 3: coarsest scaling)**
- Run 1F (8B per-tensor)
- Run 2E (30B MoE Base per-tensor)
- Run 3E (30B MoE Instruct per-tensor)

**4e. FP8 Attention sweep (priority 4: after linear FP8 is validated)**
- Linear is **fixed** to blockwise 128 for all runs (same as 1D)
- Run 4A (DPA blockwise attn)
- Run 4B (DPA MXFP8 attn)
- Run 4C (DPA dynamic attn)
- Run 4D (full MHA blockwise2d attn — most aggressive)

**4f. Low-precision optimizer sweep (priority 5: after FP8 linear + attention are validated)**
- Linear is **fixed** to blockwise 128, attention is BF16 (same as 1D)
- Run 5A (BNB `AdamW8bit`)
- Run 5B (TorchAO `_AdamW` with `bf16_stochastic_round`)
- **Pre-requisite**: Validate BNB/TorchAO compatibility with FSDP2 DTensor in a 2-step smoke test

### Step 5: Compare
- Experiment 1 chart: overlay 1A vs 1B vs 1C vs 1D vs 1E vs 1E-64 vs 1E-128 vs 1F vs 4A vs 4B vs 4C vs 4D vs 5A vs 5B
- Experiment 2 chart: overlay 2A vs 2B vs 2C vs 2D vs 2D-64 vs 2D-128 vs 2E
- Experiment 3 chart: overlay 3A vs 3B vs 3C vs 3D vs 3D-64 vs 3D-128 vs 3E
- **Experiment 4 chart**: overlay 1D vs 4A vs 4B vs 4C vs 4D (attention FP8 sweep, linear fixed to blockwise)
- **Experiment 5 chart**: overlay 1D vs 5A vs 5B (optimizer sweep — alignment + peak memory)
- Cross-experiment chart: overlay Exp 2 vs Exp 3 (Base vs Instruct, all scaling methods + MXFP8 sweep)
- Scaling method comparison chart: per-tensor vs blockwise vs MXFP8-32 vs MXFP8-64 vs MXFP8-128 across all models
- **MXFP8 block-size sweep chart**: overlay MXFP8-32 vs MXFP8-64 vs MXFP8-128 per experiment to isolate block-size effect
- **Attention FP8 chart**: DPA-blockwise (4A) vs DPA-MXFP8 (4B) vs DPA-dynamic (4C) vs MHA-blockwise2d (4D) — alignment quality and throughput (linear fixed to blockwise 128)
- **Optimizer chart**: BNB 8bit (5A) vs TorchAO BF16 (5B) vs baseline AdamW (1D) — alignment quality + peak GPU memory
- Generate comparison plots and report
- Write → `Lumen/outputs/fp8_training_alignment/FP8_TRAINING_ALIGNMENT_RESULTS.md`

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| vLLM FP8 rollout not working on ROCm | Blocks all FP8 rollout experiments | Test with 2-step smoke run first |
| Qwen3 MoE OOM on 8 GPUs | Blocks experiment 2 | Reduce batch sizes, increase offloading, lower gpu_memory_util |
| vLLM TP>1 hang on ROCm | Limits rollout throughput | Use TP=1 (proven to work) |
| No dynamic sampling | Training curves differ from reference | Both BF16/FP8 skip it — comparison still fair |
| FP8PM + FSDP2 MoE untested | Unknown interaction with expert routing | Smoke test first; fall back to dense-only if broken |
| AITER kernels cause training mismatch | FP8 curves diverge from BF16 | Disable AITER (`USE_ROCM_AITER_ROPE_BACKEND=0`), compare |
| Blockwise CK GEMM untested on MI350X | `gemm_a8w8_blockscale` (CK) may not work | Falls back to Triton blockscale GEMM; smoke test first |
| MXFP8 requires gfx950 (CDNA4) | MI350X is gfx950+ so should work; MI300X would fail | `is_cdna4()` gate in `convert_to_mxfp8`; verify ASM path in smoke test |
| `verl_entry.py` missing blockwise/MXFP8 env vars | Cannot select scaling without code change | Wire `LUMEN_FP8_SCALING` + `LUMEN_FP8_FORMAT` + `LUMEN_FP8_BLOCK_SIZE` before blockwise/MXFP8 runs |
| Blockwise/MXFP8 backward path falls back to BF16 dgrad | Blockwise dgrad uses dequant→BF16 GEMM (not FP8 dgrad); may reduce speedup | Expected behavior per code; accuracy should be fine, throughput measured separately |
| MXFP8 E8M0 scale storage overhead for MoE | 32-element blocks produce more scales than 128-element blocks | Monitor GPU memory in smoke tests; may need to reduce batch size for MoE MXFP8 runs |
| MXFP8 block_size=128 clamped to 32 in code | `quantize_input` has `mxfp8_block = 32 if block_size > 64` — silently defeats the 128-element sweep point | Must patch `lumen/ops/quantize/linear.py:195` to remove/raise the clamp before running 1E-128/2D-128/3D-128 |
| MXFP8 block=64/128 kernel correctness | `convert_to_mxfp8` Triton kernel and `gemm_mxfp8` dispatch accept arbitrary block sizes but have only been tested with block=32 | Validate with unit tests on random matrices (abs error vs BF16 reference) before sweep runs |
| MXFP8 sweep increases total run count by 6 | 9 MXFP8 runs instead of 3, extending wall time | MXFP8 sweep runs can be parallelized across GPUs; consider running the 3 block sizes concurrently per experiment if GPU availability permits |
| FP8 attention Triton kernel NaN on varlen | `fp8_attention_kernel` has known issues with variable-length sequences (LSE precision) | Use `LUMEN_ATTN_KERNEL_BACKEND=auto` and let the dispatcher choose; if NaN occurs, fall back to `csrc` (CK BF16 forward) |
| `LUMEN_FP8_ATTN_BWD=1` hybrid path unstable | CK fwd + Triton FP8 bwd has known LSE issues | Do not enable `LUMEN_FP8_ATTN_BWD` in Experiment 4; keep standard attention backward path |
| `VerlLumenArgs` missing attention quant/backend fields | Cannot set `LUMEN_FP8_QUANT_TYPE` or `LUMEN_ATTN_KERNEL_BACKEND` via env vars | Wire two new fields + env var reads into `verl_entry.py` before Experiment 4 (Step 3.6) |
| Full MHA FP8 (4D) causes divergence | Quantizing QKV projections + attention + output projection all in FP8 may be too aggressive for RL | 4D is the most aggressive test — if it diverges, `dpa` mode (4A/4B/4C) is the recommended fallback |
| FP8 attention + FSDP2 untested | Interaction between attention FP8 scaling managers and FSDP2 sharding is not validated | Smoke test 4A with a 2-step run first; check for memory errors or NaN in attention output |
| BNB `AdamW8bit` + FSDP2 DTensor crash | BitsAndBytes may not handle FSDP2's DTensor parameter wrapping correctly | Smoke test 5A with 2-step run; if it crashes, skip 5A and rely on 5B (TorchAO) |
| TorchAO `_AdamW` unavailable on ROCm | TorchAO may not have ROCm-specific backends compiled | Check `import torchao` before running 5B; fall back to 5A or skip Experiment 5 if unavailable |
| `USE_8BIT_ADAM` silently ignored in VERL | Current `verl_entry.py` reads `USE_8BIT_ADAM` but never applies it to VERL's optimizer — must wire it (see Code Changes Required) | Apply the `verl_entry.py` patch before running Experiment 5 |
| `any_lumen` gate blocks standalone optimizer | `use_8bit_adam` alone doesn't trigger `any_lumen`, so no Lumen patching occurs | Fix `any_lumen` to include `use_8bit_adam` (see Code Changes Required) |
| Low-precision optimizer states + FP8 weight sync | Quantized optimizer states may interact with `FP8ParamManager`'s master weight → FP8 refresh cycle | `register_fp8_optimizer_hook` fires after optimizer step regardless of optimizer type; should be fine, but verify in smoke test |
| Stochastic rounding non-determinism | TorchAO's `bf16_stochastic_round` introduces random rounding; RL reward noise may mask its effect | Run 5B twice to check reproducibility; compare variance vs 1D baseline |

---

## Code Changes Required

### Wire blockwise/MXFP8 env vars into `verl_entry.py`

`lumen/rl/verl/verl_entry.py:main()` currently reads `LUMEN_FP8=1` and constructs `VerlLumenArgs` with default `linear_fp8_scaling="delayed"`. To support blockwise and MXFP8 scaling, add:

```python
linear_fp8_scaling = os.environ.get("LUMEN_FP8_SCALING", "delayed")
linear_fp8_format = os.environ.get("LUMEN_FP8_FORMAT", "fp8_e4m3")
linear_fp8_block_size = int(os.environ.get("LUMEN_FP8_BLOCK_SIZE", "128"))
```

And pass them to `VerlLumenArgs`:

```python
lumen_args = VerlLumenArgs(
    ...
    linear_fp8_scaling=linear_fp8_scaling,
    linear_fp8_format=linear_fp8_format,
    # block_size needs to be added to VerlLumenArgs and wired through LumenConfig
)
```

The `VerlLumenArgs` dataclass already has `linear_fp8_scaling` and `linear_fp8_format` fields (default `"delayed"` and `"fp8_e4m3"`). The `from_verl_config()` function already reads `lumen.linear_fp8_scaling` from YAML. The env var override is what's missing.

For block size, `VerlLumenArgs` needs a new field `linear_fp8_block_size: int = 128`, and `LumenConfig.from_args` needs to map it via `_ARG_MAP` to `LumenConfig.block_size`.

### Remove MXFP8 block_size clamp in `quantize_input`

`lumen/ops/quantize/linear.py:195` currently has:

```python
mxfp8_block = 32 if block_size > 64 else block_size
```

This silently reduces block_size=128 to 32, defeating the MXFP8 sweep. Change to:

```python
mxfp8_block = block_size
```

The downstream `convert_to_mxfp8` kernel and `gemm_mxfp8` already accept arbitrary block sizes. After this change, `LUMEN_FP8_BLOCK_SIZE=128` will produce 128-element MXFP8 blocks as intended.

### Wire attention FP8 env vars into `verl_entry.py`

`verl_entry.py:main()` already reads `LUMEN_FP8_ATTN` and passes it to `VerlLumenArgs.lumen_fp8_attn`. However, `VerlLumenArgs` is **missing** `lumen_fp8_quant_type` and `lumen_attn_backend`, and `verl_entry.py` does not read the corresponding env vars. To support Experiment 4, add:

```python
lumen_fp8_quant_type = os.environ.get("LUMEN_FP8_QUANT_TYPE", "blockwise")
lumen_attn_backend = os.environ.get("LUMEN_ATTN_KERNEL_BACKEND", "auto")
```

And add two new fields to `VerlLumenArgs` (`lumen/rl/verl/config.py`):

```python
lumen_fp8_quant_type: str = "blockwise"
lumen_attn_backend: str = "auto"
```

Then wire them through `_ARG_MAP` in `LumenConfig` (`lumen/config.py`):
- `"attn_quant_type"` → `("lumen_fp8_quant_type",)` — already exists in `_ARG_MAP`
- `"attn_backend"` → `("lumen_attn_backend",)` — already exists in `_ARG_MAP`

Both `_ARG_MAP` entries already exist, so only the `VerlLumenArgs` fields and `verl_entry.py` env var reads are needed.

### Script env var patterns

Blockwise scripts set:
```bash
export LUMEN_FP8_SCALING=blockwise
export LUMEN_FP8_BLOCK_SIZE=128
```

MXFP8 scripts set (one script per block size):
```bash
# MXFP8 block=32 (e.g., run_dapo_qwen3_8b_fp8_e2e_mxfp8_b32.sh)
export LUMEN_FP8_SCALING=blockwise   # ScalingType; format determines MXFP8
export LUMEN_FP8_FORMAT=mxfp8
export LUMEN_FP8_BLOCK_SIZE=32

# MXFP8 block=64 (e.g., run_dapo_qwen3_8b_fp8_e2e_mxfp8_b64.sh)
export LUMEN_FP8_SCALING=blockwise
export LUMEN_FP8_FORMAT=mxfp8
export LUMEN_FP8_BLOCK_SIZE=64

# MXFP8 block=128 (e.g., run_dapo_qwen3_8b_fp8_e2e_mxfp8_b128.sh)
export LUMEN_FP8_SCALING=blockwise
export LUMEN_FP8_FORMAT=mxfp8
export LUMEN_FP8_BLOCK_SIZE=128
```

Attention FP8 scripts (Experiment 4) add these **on top of** the blockwise linear FP8 env vars (linear is fixed, only attention changes):
```bash
# All Exp 4 runs share these linear env vars (same as 1D):
export LUMEN_FP8=1
export LUMEN_FP8_SCALING=blockwise
export LUMEN_FP8_BLOCK_SIZE=128

# 4A: DPA + blockwise attention quant
export LUMEN_FP8_ATTN=dpa
export LUMEN_FP8_QUANT_TYPE=blockwise

# 4B: DPA + MXFP8 attention quant
export LUMEN_FP8_ATTN=dpa
export LUMEN_FP8_QUANT_TYPE=mxfp8

# 4C: DPA + dynamic per-tensor attention quant
export LUMEN_FP8_ATTN=dpa
export LUMEN_FP8_QUANT_TYPE=dynamic

# 4D: Full MHA + blockwise2d attention quant (most aggressive)
export LUMEN_FP8_ATTN=mha
export LUMEN_FP8_QUANT_TYPE=blockwise2d
```

### Wire low-precision optimizer into VERL config (Experiment 5)

The simplest approach is **Option B**: inject optimizer overrides in `verl_entry.py` when `USE_8BIT_ADAM=1`, using VERL's existing `override_optimizer_config` mechanism.

**Changes to `lumen/rl/verl/verl_entry.py`**:

```python
use_8bit_adam = bool(int(os.environ.get("USE_8BIT_ADAM", "0")))
optimizer_impl = os.environ.get("LUMEN_OPTIMIZER_IMPL", "")
optimizer_name = os.environ.get("LUMEN_OPTIMIZER", "")
optimizer_extra = os.environ.get("LUMEN_OPTIMIZER_CONFIG", "")
```

Then in the Hydra override injection:
```python
if use_8bit_adam:
    overrides.append("actor_rollout_ref.actor.optim.optimizer_impl=bitsandbytes.optim")
    overrides.append("actor_rollout_ref.actor.optim.optimizer=AdamW8bit")
elif optimizer_impl:
    overrides.append(f"actor_rollout_ref.actor.optim.optimizer_impl={optimizer_impl}")
    overrides.append(f"actor_rollout_ref.actor.optim.optimizer={optimizer_name}")
    if optimizer_extra:
        overrides.append(
            f"actor_rollout_ref.actor.optim.override_optimizer_config={optimizer_extra}"
        )
```

**Fix `any_lumen` gate**: Add `use_8bit_adam` to the `any_lumen` condition so that setting `USE_8BIT_ADAM=1` alone triggers Lumen patching:
```python
any_lumen = lumen_fp8 or lumen_norm or lumen_lora or use_8bit_adam
```

**Experiment 5 scripts** set:
```bash
# 5A: BNB AdamW8bit (on top of blockwise linear FP8 env vars from 1D)
export USE_8BIT_ADAM=1

# 5B: TorchAO _AdamW with bf16 stochastic rounding
export LUMEN_OPTIMIZER_IMPL=torchao.optim
export LUMEN_OPTIMIZER=_AdamW
export LUMEN_OPTIMIZER_CONFIG='{"bf16_stochastic_round": true}'
```

---

## File Layout

```
/dev/shm/
├── data/
│   ├── dapo-math-17k.parquet        # Training data (ramdisk for I/O speed)
│   └── aime-2024.parquet            # Validation data
├── model/
│   ├── qwen3-8b-base/               # Dense 8B
│   ├── qwen3-30b-a3b-base/          # MoE 30B (base)
│   └── qwen3-30b-a3b/               # MoE 30B (instruct)
└── ckpts/                            # Checkpoints (per-experiment subdirs)

Lumen/
├── examples/rl/verl/dapo/
│   ├── common.sh                                        # Shared DAPO config
│   ├── smoke_test.sh                                    # 2-step validation
│   ├── run_dapo_qwen3_8b_bf16.sh                       # Exp 1A
│   ├── run_dapo_qwen3_8b_fp8_rollout_tis.sh            # Exp 1B
│   ├── run_dapo_qwen3_8b_fp8_rollout_no_tis.sh         # Exp 1C
│   ├── run_dapo_qwen3_8b_fp8_e2e_blockwise.sh          # Exp 1D (blockwise)
│   ├── run_dapo_qwen3_8b_fp8_e2e_mxfp8_b32.sh          # Exp 1E (MXFP8 block=32)
│   ├── run_dapo_qwen3_8b_fp8_e2e_mxfp8_b64.sh          # Exp 1E-64 (MXFP8 block=64)
│   ├── run_dapo_qwen3_8b_fp8_e2e_mxfp8_b128.sh         # Exp 1E-128 (MXFP8 block=128)
│   ├── run_dapo_qwen3_8b_fp8_e2e_lumen.sh              # Exp 1F (per-tensor)
│   ├── run_dapo_qwen3_30b_moe_bf16_tis.sh              # Exp 2A (Base)
│   ├── run_dapo_qwen3_30b_moe_fp8_rollout_tis.sh       # Exp 2B (Base)
│   ├── run_dapo_qwen3_30b_moe_fp8_e2e_blockwise.sh     # Exp 2C (Base, blockwise)
│   ├── run_dapo_qwen3_30b_moe_fp8_e2e_mxfp8_b32.sh     # Exp 2D (Base, MXFP8 block=32)
│   ├── run_dapo_qwen3_30b_moe_fp8_e2e_mxfp8_b64.sh     # Exp 2D-64 (Base, MXFP8 block=64)
│   ├── run_dapo_qwen3_30b_moe_fp8_e2e_mxfp8_b128.sh    # Exp 2D-128 (Base, MXFP8 block=128)
│   ├── run_dapo_qwen3_30b_moe_fp8_e2e_lumen.sh         # Exp 2E (Base, per-tensor)
│   ├── run_dapo_qwen3_30b_moe_instruct_bf16_tis.sh     # Exp 3A (Instruct)
│   ├── run_dapo_qwen3_30b_moe_instruct_fp8_rollout_tis.sh  # Exp 3B (Instruct)
│   ├── run_dapo_qwen3_30b_moe_instruct_fp8_e2e_blockwise.sh # Exp 3C (Instruct, blockwise)
│   ├── run_dapo_qwen3_30b_moe_instruct_fp8_e2e_mxfp8_b32.sh  # Exp 3D (Instruct, MXFP8 block=32)
│   ├── run_dapo_qwen3_30b_moe_instruct_fp8_e2e_mxfp8_b64.sh  # Exp 3D-64 (Instruct, MXFP8 block=64)
│   ├── run_dapo_qwen3_30b_moe_instruct_fp8_e2e_mxfp8_b128.sh # Exp 3D-128 (Instruct, MXFP8 block=128)
│   ├── run_dapo_qwen3_30b_moe_instruct_fp8_e2e_lumen.sh    # Exp 3E (Instruct, per-tensor)
│   ├── run_dapo_qwen3_8b_fp8_e2e_blockwise_attn_dpa.sh          # Exp 4A (linear=blockwise, attn=DPA blockwise)
│   ├── run_dapo_qwen3_8b_fp8_e2e_blockwise_attn_dpa_mxfp8.sh    # Exp 4B (linear=blockwise, attn=DPA MXFP8)
│   ├── run_dapo_qwen3_8b_fp8_e2e_blockwise_attn_dpa_dynamic.sh  # Exp 4C (linear=blockwise, attn=DPA dynamic)
│   ├── run_dapo_qwen3_8b_fp8_e2e_blockwise_attn_mha.sh          # Exp 4D (linear=blockwise, attn=MHA blockwise2d)
│   ├── run_dapo_qwen3_8b_fp8_e2e_blockwise_optim_bnb8bit.sh    # Exp 5A (blockwise + BNB AdamW8bit)
│   └── run_dapo_qwen3_8b_fp8_e2e_blockwise_optim_torchao.sh    # Exp 5B (blockwise + TorchAO _AdamW bf16 stoch round)
└── outputs/fp8_training_alignment/
    ├── FP8_TRAINING_ALIGNMENT_PLAN.md                   # This document
    └── FP8_TRAINING_ALIGNMENT_RESULTS.md                # Final comparison (after runs)
```
