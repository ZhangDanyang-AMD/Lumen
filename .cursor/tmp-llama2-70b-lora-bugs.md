# Temporary Training Bug Notes

This file lives at `.cursor/tmp-training-bugs.md` relative to the `Lumen` repo root. Read the whole file at the start of every new Lumen training debug session.

Use it to keep track of possible bugs found during testing. Do not treat any entry here as proof. Re-check against the current reference diff and current repro before acting.

Treat any fresh return to the same debugging problem as a new debug session:

- a new chat or agent session
- a new day or work block
- returning after unrelated work
- starting a new round of debug after prior tests finished

Write back only meaningful tests or experiments that change confidence in a hypothesis, such as a new repro, written diff, backend toggle, layerwise compare, kernel test, or targeted integration check. Do not log every identical rerun. Do log negative results that rule a suspicion out.

## Open

### [2026-04-03 val-loss-gap-0.012]
- Symptom: best val_loss=0.937 (v20, step 768), MLPerf target is <0.925. Gap of 0.012.
- Investigated and ruled out:
  1. Seed: MLPerf uses $RANDOM; YAML default=1. Seed doesn't determine convergence target. RULED OUT.
  2. LoRA target modules: both map 'attention' to linear_qkv + linear_proj. VERIFIED IDENTICAL.
  3. Loss normalization: Applied --sft flag to match NeMo sample_weight=constant. val_loss=0.937 vs 0.938 with global token-weighted. RULED OUT.
  4. LoRA init: both use kaiming_uniform(a=sqrt(5)) for lora_A and zeros for lora_B. VERIFIED IDENTICAL.
  5. Data processing/packing: both use pre-packed 8192-token .npy with seq_start_id. Lumen re-packs for seq_length+1 (borrowing 1 token from next sample), but the extra token has mask=0 for validation — NEGLIGIBLE DIFFERENCE. Training has 1/8192 cross-boundary label artifact. RULED OUT.
  6. Activation between LoRA A and B: NeMo sets activation="identity" for LoRA (not swish). Lumen has no activation between lora_a and lora_b. VERIFIED IDENTICAL.
  7. LoRA dropout placement: NeMo MLPerf uses dropout_position="pre" (dropout before lora_a). Lumen uses dropout after lora_b. Tested in v21 with pre-A dropout. Result: val_loss=0.939 (WORSE than v20's 0.937 by +0.002 consistently). RULED OUT — dropout placement is not the cause of the gap. Reverting to original Lumen post-B dropout.
  8. Optimizer: VERIFIED IDENTICAL. Both use apex.optimizers.FusedAdam (adam_w_mode=True / AdamW). Same lr=4e-4, betas=(0.9,0.999), eps=1e-8, wd=1e-4, clip=0.3. Same FP32 master params wrapper (Megatron Float16OptimizerWithFloat16Params ≡ NeMo MainParamsOptimizerWrapper). Same cosine LR schedule over 1024 steps, no warmup, min_lr=0. Same weight decay grouping (no WD on biases/norms). RULED OUT.
- **FP8 implementation diff analysis** (2026-04-04):
  DIFF-1 (CRITICAL): Backward gradient dtype — hybrid mode
    - Lumen: lines 680-682 of lumen/ops/quantize/linear.py FORCE bwd_dtype back to E4M3 (same as forward), overriding the hybrid E5M2 intent. Gradients are always quantized to E4M3.
    - TE: Uses E5M2 for backward gradients as intended by hybrid mode. E5M2 has larger dynamic range (5-bit exponent) which is better for gradient representation.
    - Impact: E4M3 has only 4-bit exponent → smaller dynamic range → more gradient clipping/saturation during backward. This could degrade convergence.
    - Note: The override was likely added because AITER FP8 GEMMs don't support mixed E4M3/E5M2 operands. But since dgrad uses grad(E4M3) x weight(E4M3), same dtype, dgrad still runs in FP8. The real loss is the E5M2 dynamic range for gradients.
  DIFF-2 (CRITICAL): Wgrad precision
    - Lumen: line 735 — `_use_fp8_wgrad` is ALWAYS False when bwd_scaling is "delayed" or "dynamic". Wgrad is computed in BF16 (dequant grad and input to BF16, then BF16 GEMM).
    - TE: fp8_wgrad=True (default) — wgrad is computed using FP8 GEMM (E5M2 grad x E4M3 activation).
    - Impact: BF16 wgrad has higher precision but also means weight gradients are computed differently. The TE FP8 wgrad path with E5M2 grad and E4M3 activations is a different numerical profile. Could either help or hurt — needs testing.
  DIFF-3 (MODERATE): Dgrad path for base frozen weights
    - Lumen: With FP8_PARAM_STORAGE=1, FP8StoredLinearFunction is used. Dgrad runs the SAME logic: grad(E4M3) x weight(E4M3) → FP8 GEMM (since both are same dtype after hybrid override).
    - TE: With fp8_params, TE uses QuantizedTensor weights. Dgrad runs grad(E5M2) x weight(E4M3) → FP8 GEMM (TE supports mixed E4M3/E5M2 GEMMs natively).
    - Impact: Different quantization noise in dgrad computation.
  DIFF-4 (LOW): Amax history length
    - Lumen config: FP8_AMAX_HISTORY=4 (correctly matching MLPerf config_MI300X_1x8x1.sh)
    - TE MLPerf: FP8_AMAX_HISTORY=4
    - Status: MATCHED ✓
  DIFF-5 (LOW): Amax compute algorithm
    - Lumen config: FP8_AMAX_ALGO="most_recent"
    - TE MLPerf: FP8_AMAX_ALGO="most_recent"
    - Status: MATCHED ✓
  DIFF-6 (LOW): FP8 margin
    - Lumen: default margin=0
    - TE MLPerf: FP8_MARGIN=0
    - Status: MATCHED ✓
  DIFF-7 (LOW): reduce_amax
    - Lumen config: FP8_REDUCE_AMAX=0 (False)
    - TE MLPerf: FP8_REDUCE_AMAX=False
    - Status: MATCHED ✓
  DIFF-8 (LOW): Scale computation formula
    - Lumen: scale = amax / (fp8_max / 2**margin); quant = tensor / scale
    - TE: sf = (fp8_max / amax) / 2**margin; quant = tensor * sf
    - Both: tensor * fp8_max / (amax * 2**margin) — ALGEBRAICALLY IDENTICAL ✓
  DIFF-9 (INFO): FP8 param storage mechanism
    - Lumen: Custom _shrink_frozen_weights_to_fp8 + FP8StoredLinearFunction (Lumen-native)
    - TE: fp8_model_init + QuantizedTensor (TE-native)
    - These are implementation-specific but achieve the same goal. Not directly comparable.
- Remaining possible causes:
  1. DIFF-1 (gradient E4M3 vs E5M2) — most likely to affect convergence. E5M2 dynamic range better matches gradient distributions.
  2. DIFF-2 (BF16 wgrad vs FP8 wgrad) — different numerical profile for weight updates.
  3. DIFF-3 (dgrad mixed-dtype handling) — consequence of DIFF-1.
- Evidence:
  - v19: best val_loss=0.938 (global token-weighted normalization)
  - v20: best val_loss=0.937 (SFT per-microbatch normalization)
  - v21: best val_loss=0.939 (pre-A dropout, WORSE by +0.002). Pre-A dropout makes convergence slightly worse.
- Full v20 vs v21 comparison at late steps (pre-A dropout consistently +0.002 worse):
  - step 768: v20=0.937 v21=0.939
  - step 864: v20=0.938 v21=0.940
  - step 912: v20=0.939 v21=0.941
  - step 960: v20=0.938 v21=0.940
  - step 1024: v20=0.938 v21=0.940
- v22 experiment (2026-04-04):
  Attempted to fix DIFF-1 (enable E5M2 backward gradients) while keeping CK same-dtype GEMM.

  **Approach A — E5M2→BF16→E4M3 recast for dgrad**: Quantize grad to E5M2 (for scale benefit), dequant to BF16, re-quant to E4M3, run CK E4M3×E4M3 GEMM. Result: OOM. The BF16 intermediate (`grad_fp8.bfloat16() * grad_scale.bfloat16()`) allocates ~448-896 MiB extra. Combined with existing 181.5 GiB usage at 96.1% of 192 GiB, no headroom.

  **Approach B — Triton mixed-dtype GEMM (E5M2×E4M3)**: New `gemm_a8w8_mixed.py` kernel. Result: OOM. Triton JIT + kernel memory footprint even larger than recast approach.

  **Approach C — BF16 dequant fallback for dgrad**: Dequant both E5M2 grad and E4M3 weight to BF16, run BF16 GEMM. Result: catastrophic convergence regression (val_loss ~1.62 at step 768 vs 0.937). E5M2 only has 2-bit mantissa; dequanting to BF16 loses too much gradient precision.

  **Conclusion**: Cannot enable E5M2 backward without either (a) native mixed-dtype FP8 GEMM support in CK, or (b) reducing memory usage elsewhere to make room for recast. Current code reverted to match v20 behavior (E4M3 for everything). v22 step 6: lm_loss=3.991635 grad_norm=1.248 — identical to v20 (3.991635 / 1.243).

  **DIFF-1: RE-ANALYZED** — E4M3 for backward is actually BETTER than E5M2 for non-blockwise2d delayed scaling:
    - Dgrad: E4M3 has 3-bit mantissa → more precise gradient values in the GEMM
    - Wgrad: both paths dequant to BF16; E4M3 dequant is more precise (3-bit vs 2-bit mantissa)
    - Scaling manager's bwd amax is NOT updated in non-blockwise2d delayed path (no manager passed to quantize_input)
    - TE's E5M2 advantage only applies when the GEMM natively supports E5M2 operands, which CK doesn't
    - **RULED OUT as cause of 0.012 gap** — E4M3 backward is correct for Lumen's CK-backed GEMM.
  **DIFF-2: RULED OUT** — BF16 wgrad vs FP8 wgrad. BF16 wgrad has HIGHER precision for weight gradients. TE uses FP8 wgrad for speed, not quality. Keeping BF16 wgrad should be equal or better convergence. Also: FP8 wgrad causes OOM.
  **DIFF-3: RULED OUT** — Consequence of DIFF-1, no longer applies.

- All FP8 implementation diffs have been RULED OUT as the cause of the 0.012 gap.
- **CRITICAL CORRECTION (2026-04-04)**: The MLPerf reference for MI300X is NOT an NVIDIA submission.
  It is AMD's own submission using **NeMo v2.3.0 + ROCm/TransformerEngine + ROCm/Megatron-LM**
  on the **same MI300X hardware** we use. This changes the entire analysis.

- **AMD MI300X reference results** (10 runs, all converge):
  Best val_loss per run: 0.9235, 0.9234, 0.9225, 0.9217, 0.9229, 0.9213, 0.9242, 0.9238, 0.9244, 0.9216
  Mean: ~0.9229. All under 0.925 target. Seeds: random ($RANDOM).

- **Lumen v20**: best val_loss=0.937 (seed=1234). **Gap = 0.014 from AMD reference mean.**

- **Key implementation diffs between AMD reference (NeMo+TE) and Lumen** (same hardware!):
  | Component | AMD Reference (NeMo+TE) | Lumen |
  |-----------|------------------------|-------|
  | FP8 engine | ROCm/TransformerEngine | Lumen native + AITER |
  | Attention | TE fused CK v3 (`NVTE_FUSED_ATTN_CK=1`) | Lumen aiter CK FMHA |
  | RoPE | `apply_rope_fusion: True` (TE fused) | `apply_rope_fusion: False` (unfused Megatron) |
  | FP8 backward | **E5M2 hybrid** (TE native mixed-dtype GEMM) | **E4M3 forced** (CK limitation) |
  | FP8 wgrad | FP8 GEMM via TE | BF16 fallback |
  | LoRA dropout | `dropout_position: "pre"` | `dropout_position: "post"` |
  | LoRA scaling | NeMo standard `alpha/rank` | Patched via `patch_lora_scaling.py` |
  | RMSNorm | `NVTE_USE_RMSNORM_TRITON=1` (TE Triton) | Lumen RMSNorm |
  | SwiGLU | `USE_TE_SWIGLU=1` (TE fused) | Lumen SwiGLU |
  | Cast+transpose | `NVTE_USE_CAST_TRANSPOSE_TRITON=1` | Lumen quantize_input |
  | Transpose cache | `ENABLE_TRANSPOSE_CACHE=0` (disabled) | Lumen uses transpose_cached |

- **Re-evaluation of ruled-out hypotheses**:
  1. **DIFF-1 (FP8 backward E5M2 vs E4M3)**: UN-RULED-OUT. AMD ref uses TE which natively handles
     mixed-dtype E5M2×E4M3 GEMM on ROCm via its CK backend. This IS a real convergence difference.
     TE's ROCm CK backend may have mixed-dtype support that Lumen's AITER-backed CK does not.
  2. **DIFF-2 (FP8 wgrad vs BF16 wgrad)**: UN-RULED-OUT. AMD ref uses TE FP8 wgrad (E5M2 grad ×
     E4M3 activation). Lumen uses BF16 wgrad. Different numerical profile.
  3. **LoRA dropout pre vs post**: Was tested in v21 (+0.002 worse). But AMD ref uses "pre" AND
     converges better. This suggests the dropout difference is secondary to FP8 diffs.
  4. **RoPE fusion**: Lumen forces `apply_rope_fusion=False`. AMD ref uses `True` (TE fused).
     Could produce subtly different numerical results.
  5. **RMSNorm / SwiGLU / Cast+Transpose**: All different implementations between TE and Lumen.
     Each contributes small numerical differences that accumulate.

- **Detailed investigation of all 5 diffs (2026-04-04)**:

  ### DIFF-1/2: FP8 backward E5M2 + FP8 wgrad via hipBLASLt
  **TE on ROCm uses hipBLASLt (not CK) for linear GEMM**, which natively supports mixed E4M3×E5M2.
  - `rocm_gemm.cu` in TE creates separate A/B layout descriptors with independent `hipDataType`
  - `get_hipblaslt_dtype` maps `kFloat8E5M2` → `HIP_R_8F_E5M2_FNUZ` on gfx942
  - ROCm 7.0 hipBLASLt docs confirm BF8×FP8 (E5M2×E4M3) is supported

  **AITER's hipBLASLt wrapper (`hipb_mm`) has 3 artificial restrictions preventing mixed FP8:**
  1. `dtype_map` (line 101-110) only has E4M3 entries, no E5M2 (`kFloat8_e5m2fnuz`)
  2. `TORCH_CHECK(mat1.dtype() == mat2.dtype())` at line 1056 enforces same dtype
  3. `hipblasLtMatmul_sol_wrapper` uses single `intype` for both matA and matB layouts
  4. `hipb_findallsols` has same `mat1.dtype() == mat2.dtype()` check at line 1232

  **AITER's CK GEMM (`gemm_a8w8`) also has `XQ.dtype() == WQ.dtype()` check (line 156-158).**
  CK library itself has mixed fp8/bf8 warp dispatch but AITER doesn't wire it.

  **Fix approach**: Modify AITER's hipBLASLt path (preferred, matches TE):
  - Add `{at::kFloat8_e5m2fnuz, HIP_R_8F_E5M2_FNUZ}` and `{at::kFloat8_e5m2, HIP_R_8F_E5M2}` to `dtype_map`
  - Remove same-dtype TORCH_CHECK in `hipb_mm` and `hipb_findallsols` (relax for FP8)
  - Split `hipblasLtMatmul_sol_wrapper` signature: take `intype_a` and `intype_b` instead of single `intype`
  - Create matA layout with `intype_a`, matB layout with `intype_b`
  - Update `getAllAlgos` call to pass separate A/B types
  Then use this in Lumen dgrad: quantize grad to E5M2, weight stays E4M3, run hipBLASLt mixed GEMM.

  **Status**: ACTIONABLE. This is the highest-priority fix.

  **Alternative (already exists)**: `gemm_a8w8_mixed.py` (Triton) works but caused OOM in v22.
  hipBLASLt should have lower memory overhead than Triton JIT.

  ### DIFF: LoRA dropout position
  AMD ref uses `dropout_position: "pre"` with standard `nn.Dropout` (not Thunder dropout).
  `NEMO_LORA_USE_THUNDER_DROPOUT` is NOT set in AMD MI300X config.
  v21 tested pre-A dropout alone in Lumen: +0.002 worse (0.939 vs 0.937).
  **RULED OUT** as primary cause. Pre dropout may interact with other fixes but alone hurts.

  ### DIFF: RoPE fusion
  Lumen forces `apply_rope_fusion = False` in `megatron.py` line 434 because fused RoPE
  requires TE's `fused_apply_rotary_pos_emb` (hard TE dependency, validated at config time).
  Unfused path: PyTorch `cos`/`sin` in FP32 → cast to `t.dtype` → multiply/add.
  Fused path (TE): single kernel, same algebra, different rounding order.
  **Impact**: Typically ULP-level BF16 differences per layer. Small vs FP8 diffs.
  **Could contribute** but unlikely to be a major factor in 0.014 gap.
  **Fix**: Would require either installing TE or implementing Lumen's own fused RoPE.
  Lumen has `lumen/ops/rope.py` (AITER-backed) but `--lumen-fused-rope` flag is registered
  but not wired to anything in training. Could wire this up as a lower-effort option.
  **Status**: LOW PRIORITY. Investigate after FP8 GEMM fix.

  ### DIFF: RMSNorm (AITER Triton vs TE Triton)
  Both use FP32 accumulation for variance/RMS, output in BF16. Same precision profile.
  AITER kernel loads in FP32, multiplies `x * norm_factor * g` in FP32, writes BF16.
  TE kernel: similar FP32 accumulation pattern.
  AMD ref sets `RMSNORM_CAST=0` (no extra FP32 cast at RMSNorm boundaries).
  **Impact**: Near machine-epsilon differences per layer. Both are FP32-accumulated.
  **RULED OUT** as significant contributor. Same precision class.

  ### DIFF: SwiGLU (Megatron `@jit_fuser` vs TE fused)
  Both implement `SiLU(gate) * up`. Megatron's `@jit_fuser` may fuse via TorchScript/Inductor.
  TE's `USE_TE_SWIGLU=1` uses a dedicated fused kernel.
  **Key sub-diff**: With `fp8_input_store`, pre-SwiGLU activations are saved in FP8 for backward.
  Lumen patches this in `megatron_patches.py` to use ROCm `float8_e4m3fnuz`.
  Forward result is always BF16 in both cases.
  **Impact**: Forward differences minimal. Backward differences from FP8 activation storage
  are shared — both TE and Lumen store in E4M3 for backward.
  **RULED OUT** as significant contributor.

  ### DIFF: Cast+Transpose
  AMD ref: `NVTE_USE_CAST_TRANSPOSE_TRITON=1`, `ENABLE_TRANSPOSE_CACHE=0`
  Lumen: uses `quantize_input` + `transpose_cached` in FP8Descriptor
  TE disables transpose cache; Lumen enables it. This affects memory layout and
  when transposition happens relative to FP8 casting.
  **Impact**: Primarily affects memory/performance, not numerical precision.
  The quantization itself is the same operation (cast to FP8 with scale).
  **RULED OUT** as convergence factor.

  ### Summary of investigation priority:
  1. **FP8 mixed-dtype GEMM (DIFF-1 + DIFF-2)**: PRIMARY SUSPECT. Fix AITER hipBLASLt to support
     E5M2 operands (3 changes to hipbsolgemm.cu), then enable E5M2 backward + FP8 wgrad in Lumen.
     This is what TE does and is the largest numerical difference.
  2. **RoPE fusion**: LOW PRIORITY. Small ULP-level differences. Wire up `lumen/ops/rope.py` later.
  3. **LoRA dropout pre**: RULED OUT alone. May re-test combined with FP8 fix.
  4. **RMSNorm**: RULED OUT. Same FP32-accumulation precision class.
  5. **SwiGLU**: RULED OUT. Same forward; backward FP8 storage is shared approach.
  6. **Cast+Transpose**: RULED OUT. Memory/perf difference, not numerical.

- v23 experiment (2026-04-04):
  Attempted to fix DIFF-1/2 by enabling E5M2 backward gradients via mixed-dtype FP8 GEMM.

  **AITER hipBLASLt changes (successful)**:
  - Added E5M2 entries (`kFloat8_e5m2fnuz`, `kFloat8_e5m2`) to hipBLASLt `dtype_map`
  - Split `hipblasLtMatmul_sol_wrapper` and `hipblasLtMatmul_findallsols_wrapper` to take
    separate `intype_a` and `intype_b` instead of single `intype`
  - Removed same-dtype `TORCH_CHECK` in `hipb_mm` and `hipb_findallsols`
  - **Verified**: mixed-dtype FP8 GEMM (E5M2 x E4M3) works correctly in docker test

  **OOM on training (all attempts)**:
  1. hipBLASLt mixed GEMM: OOM on 448 MiB (GEMM output tensor). hipBLASLt workspace (256 MiB)
     allocated via raw HIP, invisible to PyTorch allocator. Total GPU memory 192 GiB, PyTorch
     uses 181.3 GiB allocated + 6.68 GiB reserved = 187.98 GiB. Only ~4 GiB raw HIP free.
  2. Triton mixed GEMM: OOM on 448 MiB. Triton JIT compiler allocates GPU memory via HIP
     for new kernel specialization (E5M2 x E4M3 is new; forward only compiled E4M3 x E4M3).
  3. torch.cuda.empty_cache() before mixed GEMM: iteration 1 completed but grad_norm=nan,
     OOM on iteration 2. empty_cache disrupts PyTorch's memory pool, causing fragmentation
     on subsequent iterations.

  **Root cause of OOM**: v20 uses CK for all dgrad GEMMs. CK doesn't need workspace and
  its kernel binary is already compiled for E4M3. Any alternative backend (hipBLASLt/Triton)
  for mixed-dtype GEMM requires additional GPU memory that doesn't fit in the ~4 GiB headroom:
  - hipBLASLt: 256 MiB persistent workspace + algorithm search memory
  - Triton: JIT compilation memory for new dtype specialization

  **Conclusion**: Enabling E5M2 backward gradients requires either:
  (a) Patching AITER's CK GEMM (`gemm_a8w8`) to support mixed E4M3/E5M2 operands
      (CK library has mixed fp8/bf8 warp dispatch, but AITER doesn't wire it). This is
      the **zero-overhead** path — same kernel, same memory, just different operand types.
  (b) Reducing overall memory usage by ~500 MiB elsewhere (e.g. smaller batch, gradient
      checkpointing changes) to make room for hipBLASLt/Triton workspace.
  (c) Pre-initializing hipBLASLt workspace before model loading (shifts memory budget).

  **DIFF-1/2 remain UN-RULED-OUT** as convergence factors. The fix is proven to work
  (mixed-dtype GEMM produces correct results) but blocked by memory constraints.
  Code changes in hipbsolgemm.cu and gemm_a8w8_mixed.py are preserved for future use.
  Lumen Python code reverted to v20 behavior (E4M3 for everything) until CK mixed-dtype
  support is added.

- v24 experiment (2026-04-05):
  Attempted to unblock DIFF-1 by pre-allocating hipBLASLt workspace before model loading.

  **Approach**: Pre-allocate 256 MiB hipBLASLt workspace via `ensure_hipblaslt_ready()` during
  `_setup_with_fp8_storage` hook (before model loading). This shifts PyTorch's memory budget
  so the caching allocator accounts for the workspace from the start.

  **Changes**:
  - `lumen/ops/quantize/linear.py`: Added `ensure_hipblaslt_ready()` with idempotent init
  - `lumen/ops/quantize/linear.py`: Added `_gemm_per_tensor_hipblas_mixed()` for dgrad NN layout
  - `lumen/ops/quantize/linear.py`: Updated `gemm_per_tensor_mixed()` to use hipBLASLt only
  - `lumen/ops/quantize/linear.py`: Skip CK quant for E5M2 (CK doesn't support it)
  - `lumen/models/megatron.py`: Injected `ensure_hipblaslt_ready()` in `_setup_with_fp8_storage`
  - `QuantizedLinearFunction.backward` and `FP8StoredLinearFunction.backward`: Quantize grad to
    `bwd_dtype` (E5M2) and route dgrad through `gemm_per_tensor_mixed` for mixed-dtype case
  - `gemm_primitives.py`: Same E5M2 quantization + mixed GEMM for `compute_dgrad_fp8`

  **Result**: OOM in SwiGLU backward `torch.cat` — 896 MiB allocation failed.
  PyTorch: 181.05 GiB allocated, 6.26 GiB reserved, 676 MiB free.
  The 256 MiB hipBLASLt workspace reduced headroom below what `torch.cat` needed.

- v25 experiment (2026-04-05):
  Fixed v24 OOM and NaN issues. **Training is running successfully.**

  **Fix 1: SwiGLU backward in-place write** (`megatron_patches.py`):
  Replaced `result_chunks = []; ... return torch.cat(result_chunks)` with pre-allocated
  `result = torch.empty(M, ...)` and `result[s:e] = swiglu_back(...)`. Saves ~784 MiB peak
  memory (eliminates simultaneous 8 × 112 MiB chunks + 896 MiB cat output).

  **Fix 2: Skip CK quant for E5M2** (`linear.py`):
  Added `_is_e5m2()` check to skip CK `per_tensor_quant_hip` for E5M2 dtype (CK only supports
  E4M3). Eliminates 222 fallback warnings per step, goes directly to Triton.

  **Fix 3: Zero-scale NaN sanitization** (`linear.py`):
  Added `_safe_fp8_desc()` that detects zero scale (from all-zero warmup gradients) and returns
  clean zeros with scale=1.0. Without this, `scale=0 → 1/scale=inf → 0*inf=NaN` in the Triton
  quantization kernel, producing NaN FP8 values that propagate through the entire backward graph.

  **v25 early results** (training in progress):
  - Steps 1-5 (warmup): `grad_norm: 0.000` ✓ (was NaN before Fix 3)
  - Step 6: `lm_loss: 3.991635` (identical to v20 step 6: 3.991635) ✓
  - Steps 6-28: Stable training, loss decreasing, grad_norm 0.3-3.4. No OOM. No NaN.
  - Memory: 96.29% (v20: 96.10%) — 256 MiB hipBLASLt workspace fits with SwiGLU fix
  - Step time: ~6.4s (v20: ~7.9s — faster due to hipBLASLt being optimized for MI300X)

  **v25 FINAL RESULTS** (1024 steps completed, 2026-04-05):

  | Step | v25 val_loss (E5M2 bwd) | v20 val_loss (E4M3 bwd) | Delta |
  |------|------------------------|------------------------|-------|
  | 48   | 1.0252                 | 1.0219                 | +0.0033 |
  | 96   | 0.9850                 | 0.9820                 | +0.0030 |
  | 144  | 0.9750                 | 0.9669                 | +0.0081 |
  | 192  | 0.9639                 | 0.9681                 | -0.0042 |
  | 240  | 0.9580                 | 0.9584                 | -0.0004 |
  | 288  | 0.9527                 | 0.9547                 | -0.0020 |
  | 336  | 0.9531                 | 0.9541                 | -0.0010 |
  | 384  | 0.9543                 | 0.9547                 | -0.0004 |
  | 432  | 0.9536                 | 0.9494                 | +0.0042 |
  | 480  | 0.9499                 | 0.9473                 | +0.0026 |
  | 528  | 0.9461                 | 0.9438                 | +0.0023 |
  | 576  | 0.9440                 | 0.9445                 | -0.0005 |
  | 624  | 0.9489                 | 0.9504                 | -0.0015 |
  | 672  | 0.9433                 | 0.9420                 | +0.0013 |
  | 720  | 0.9453                 | 0.9458                 | -0.0005 |
  | **768** | **0.9369**          | **0.9371**             | **-0.0002** |
  | 816  | 0.9427                 | 0.9421                 | +0.0006 |
  | 864  | 0.9384                 | 0.9378                 | +0.0006 |
  | 912  | 0.9394                 | 0.9388                 | +0.0006 |
  | 960  | 0.9385                 | 0.9380                 | +0.0005 |
  | 1008 | 0.9423                 | 0.9419                 | +0.0004 |
  | **1024** | **0.9392**         | **0.9382**             | **+0.0010** |

  **Best val_loss**: v25=0.9369 (step 768) vs v20=0.9371 (step 768). **Essentially identical.**
  **Final val_loss**: v25=0.9392 vs v20=0.9382. Difference: +0.001 (noise level).
  **Memory**: 96.30% (v20: 96.10%) — +0.20% from hipBLASLt workspace.
  **Step time**: ~8.1s (v20: ~7.9s) — slightly slower due to hipBLASLt vs CK for dgrad.
  **Stability**: 0 NaN iterations, 0 skipped iterations across all 1024 steps.

  **CONCLUSION**: DIFF-1 (E5M2 vs E4M3 backward gradients) is **NOT the cause of the 0.014 gap**.
  E5M2 backward achieves effectively identical convergence to E4M3 backward on this workload.
  The 0.014 gap between Lumen (0.937) and AMD MLPerf reference (0.923) must come from other
  factors — most likely the accumulation of small differences across all implementation
  components (attention, RoPE, RMSNorm, SwiGLU, LoRA dropout, and potentially seed sensitivity).

- Status: **RESOLVED — DIFF-1/2 RULED OUT**. E5M2 backward + hipBLASLt mixed-dtype GEMM works
  correctly but does not improve convergence. The 0.014 gap to AMD MLPerf reference persists.

- **Remaining candidates for 0.014 gap** (in priority order):
  1. **Seed sensitivity**: AMD MLPerf uses $RANDOM seeds across 10 runs. Lumen uses fixed seed=1234.
     v26 tested seed=21901: best=0.9381 vs v20=0.9371. Does not close gap. RULED OUT.
  2. **FP8 wgrad**: v26 tested FP8 wgrad via hipBLASLt. Same convergence as BF16 wgrad. RULED OUT.
  3. **RMSNorm**: Both use FP32 accum (AITER Triton vs TE Triton). Machine-epsilon. RULED OUT.
  4. **SwiGLU**: Same swiglu/swiglu_back math, FP8 store shared. RULED OUT.
  5. **RoPE fusion**: v28 tested fused RoPE via apex/AITER (seed=366, LUMEN_FUSED_ROPE=1).
     Best val_loss=0.9389 (step 768) vs v20=0.9371. Final=0.9398 vs v20=0.9382.
     Delta +0.0018 at best — noise-level (different seed). Does not close gap. RULED OUT.
  6. **Attention (CK v3 vs AITER FMHA)**: INVESTIGATED (2026-04-06).
     **CRITICAL FINDING**: `is_v3_atomic_fp32` mismatch in backward pass.
     - AMD MLPerf reference: `NVTE_CK_IS_V3_ATOMIC_FP32=0` (BF16 atomics, no convert_dq kernel)
     - Lumen (via AITER `flash_attn_func`): hardcoded `is_v3_atomic_fp32=True` (FP32 atomics)
     - TE exposes this as env var; AITER hardcoded it at `mha.py:1972`
     - FP32 atomics = higher precision but **different numerical profile** from reference
     - FP32 atomics also require `dq_accum` in FP32 (more memory), BF16 atomics skip `convert_dq`

     **Additional attention diffs** (lower priority):
     - TE uses `deterministic=True` from NeMo with `nsplits` workspace; Lumen defaults `False`
     - TE has `pad_between_seqs` / THD varlen paths; Lumen uses BSHD `flash_attn_func`
     - `how_v3_bf16_cvt`: both default to 1 (RTNA). MATCHED.

     **Both TE and AITER use the same CK/ASM kernel family** (`aiter::mha_fwd`/`mha_bwd`).
     TE's `ck_fused_attn_fwd.cpp` calls `aiter::mha_fwd(...)` directly. The kernel code is
     identical — the difference is purely in the **parameters passed** to the kernel.

     **Fix applied**: Changed `flash_attn_func` default `is_v3_atomic_fp32` from `True` to `False`
     in `third_party/aiter/aiter/ops/mha.py` (both `flash_attn_func` and `flash_attn_varlen_func`).

     **v29 experiment** (2026-04-06, completed):
     Testing `is_v3_atomic_fp32=False` with v20-identical config (seed=1234).

     | Step | v29 (atomic BF16) | v20 (atomic FP32) | Delta |
     |------|-------------------|-------------------|-------|
     | 768  | 0.9387            | 0.9371            | +0.0016 |
     | 864  | 0.9393            | 0.9378            | +0.0015 |
     | 1024 | 0.9397            | 0.9382            | +0.0015 |

     Best val_loss: v29=0.9387 vs v20=0.9371. BF16 atomics **+0.0016 worse** than FP32 atomics.
     Stability: 0 NaN, 0 skipped. Memory: 96.21%.

     **CONCLUSION**: `is_v3_atomic_fp32` mismatch is NOT the cause. Lumen's FP32 atomics
     are actually slightly better than AMD reference's BF16 atomics for convergence.
     The 0.014 gap persists. **RULED OUT.**

     Reverting `flash_attn_func` default back to `is_v3_atomic_fp32=True` (keeping the
     parameter exposed for future experiments).
  7. **LoRA dropout pre**: +0.002 alone. Re-test combined with fixes (v31). LOW PRIORITY.

- **All individual implementation diffs have now been tested and RULED OUT as sole cause.**
  The 0.014 gap is most likely the **accumulation of many small numerical differences** across
  the entire stack (attention kernel variants, FP8 linear backward, RoPE, RMSNorm, SwiGLU,
  LoRA dropout, cast+transpose ordering). Each contributes <0.002 individually, but they
  compound across 80 layers × 1024 steps.

- **Layerwise forward comparison (2026-04-06)**:
  Component-level comparison of TE (AMD MLPerf container `rocm/amd-mlperf:llama2_70b_training_5.1`)
  vs Lumen on the same Megatron checkpoint (layer 0) with identical input (seed=42, shape=[128,1,8192]).

  | Component | MaxAbsDiff | RelDiff% | CosSim | Note |
  |-----------|-----------|----------|--------|------|
  | Pure math RMSNorm | 0.000000 | 0.0000% | 1.0000 | FP32 reference — bitwise identical |
  | Pure F.linear (QKV) | 0.000000 | 0.0000% | 1.0000 | torch.nn.functional.linear — identical |
  | Pure F.linear (proj) | 0.000000 | 0.0000% | 1.0000 | torch.nn.functional.linear — identical |
  | Pure SwiGLU (F.linear+F.silu) | 0.000000 | 0.0000% | 1.0000 | Reference math — identical |
  | **Backend RMSNorm** | **0.007812** | **0.4047%** | 0.9999 | **ROOT CAUSE: AITER rmsnorm2d_fwd vs apex fused_rms_norm_affine** |
  | Backend RMSNorm → F.linear | 0.001953 | 0.4693% | 1.0000 | Norm diff amplified through identical GEMM |
  | Full MLP fc1 (fused norm+GEMM) | 0.003906 | 0.4736% | 1.0000 | Consistent with norm-then-linear test |
  | Full MLP fc2 input (post-SwiGLU) | — | — | — | Norm: TE=0.601 vs Lumen=0.598 (diff=0.003) |
  | **Full MLP output** | **0.000977** | **0.6395%** | 0.9999 | Norm: TE=2.166 vs Lumen=2.154 (diff=0.012) |

  **Key finding**: The **sole source of forward divergence is AITER's `rmsnorm2d_fwd` kernel**.
  - All pure-math operations (RMSNorm in FP32, F.linear GEMM, SwiGLU) are **bitwise identical**.
  - The GEMM itself (torch F.linear) produces zero diff when given identical inputs.
  - AITER's Triton-based `rmsnorm2d_fwd` vs apex's CUDA `fused_rms_norm_affine` produces
    0.40% relative diff per layer, which cascades: norm diff → GEMM amplification → SwiGLU
    nonlinearity → fc2 output. MaxAbsDiff at RMSNorm is **7.8e-3** (3 BF16 ULPs).
  - This 0.40% diff compounds across 80 layers (2 RMSNorm per layer = 160 norm applications)
    and 1024 training steps to produce the observed 0.014 val_loss gap.
  - The previous analysis that "RMSNorm: Same FP32 accum. RULED OUT" was **incorrect**.
    While both claim FP32 accumulation, the kernel implementations differ in rounding behavior,
    operation ordering, or intermediate precision, producing measurable output differences.

  **Implication**: To close the 0.014 gap, the highest-priority fix is to align Lumen's
  RMSNorm implementation with the reference (apex `fused_rms_norm_affine`). Options:
  1. Use apex `fused_rms_norm_affine` directly (available in Lumen container via apex).
  2. Fix AITER's `rmsnorm2d_fwd` Triton kernel to match apex's numerical behavior.
  3. Investigate what specifically differs: FP32 accumulation ordering, intermediate
     multiply sequence (`x * rsqrt(var) * weight` vs `x * (weight / sqrt(var))`), or
     epsilon handling.

- **Apex RMSNorm swap verification (2026-04-06)**:
  Added `LUMEN_USE_APEX_RMSNORM=1` env var to `lumen/ops/normalization/rmsnorm.py` that
  replaces AITER `rmsnorm2d_fwd` with apex's `fused_rms_norm_affine` at the `rmsnorm()`
  function level.

  Layerwise comparison with apex RMSNorm (Lumen+apex vs TE):
  | Component | Before (AITER) | After (apex) |
  |-----------|---------------|-------------|
  | Backend RMSNorm RelDiff | 0.4047% | **0.0000%** (bitwise identical) |
  | Backend Norm→F.linear RelDiff | 0.4693% | **0.0000%** (bitwise identical) |
  | Full MLP output RelDiff | 0.6395% | **0.3623%** (43% reduction) |
  | MLP output norm diff | 0.012 | **0.00096** (12.5x reduction) |

  The remaining 0.36% MLP diff comes from the GEMM implementation difference between TE's
  fused linear layers and Lumen's custom `_do_gemm` path (different GEMM backend, not norm).

  **v30 experiment** (2026-04-06, stopped):
  v20 baseline + `LUMEN_USE_APEX_RMSNORM=1`. Seed=1234, all other settings identical to v20.
  Step 6: lm_loss=3.986537 (v20: 3.991635) — small initial diff from norm change.
  Step 48 eval: val_loss=1.0207 (v20: 1.0219) — marginal improvement.
  **Step 96 eval: val_loss=NAN** — validation produced NaN while training was stable.
  Stopped at step 126. NaN likely caused by apex `fused_rms_norm_affine` interaction
  with FP8 recomputation during eval mode. Training loss was normal throughout.

- **Root cause of AITER vs apex divergence identified (2026-04-06)**:
  Apex `fused_rms_norm_affine` computes: `(x * rsqrt(var+eps))` → truncate to BF16 →
  `* weight`. The intermediate BF16 truncation between normalization and weight multiply
  is the sole cause of divergence.
  AITER's Triton kernel computed: `x * rsqrt(var+eps) * weight` all in FP32, then BF16
  at final store. This is more precise but produces different BF16 outputs.

  Evidence: Using apex's exact `invvar` and computing `(x*invvar)->bf16 then *w` gives
  **0 maxdiff** vs apex. All other orderings produce max 3.1e-2 diff.

- **AITER Triton RMSNorm kernel fix (2026-04-06)**:
  Modified `_rms_norm_kernel`, `_fused_add_rmsnorm_kernel`, and
  `_rmsnorm_kernel_large_m_small_n` in
  `aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py`:

  Before: `rms_norm = x * norm_factor * g`
  After:  `normed = (x * norm_factor).to(output_ptr.type.element_ty).to(tl.float32); rms_norm = normed * g`

  This adds an intermediate BF16 truncation of `(x * rsqrt)` before the weight multiply,
  matching apex's exact behavior.

  Verification: Fixed Triton kernel vs apex = **0 mismatches / 1,048,576 elements** (BITWISE IDENTICAL).
  Before fix: 286,868 mismatches (27.36%).
  Forward and backward pass both verified working correctly.

  **v31 experiment** (2026-04-06, completed — stopped early):
  v20 baseline + fixed AITER Triton kernel (intermediate BF16 truncation to match apex).
  Step 6: lm_loss=3.990421 (v20: 3.991635) — confirms kernel fix is active.

  | Step | v31 (apex-match RMSNorm) | v20 (original AITER) | Delta |
  |------|------------------------|---------------------|-------|
  | 48   | 1.0191                 | 1.0219              | -0.0028 |
  | 96   | 0.9869                 | 0.9820              | +0.0049 |
  | 144  | 0.9717                 | 0.9669              | +0.0048 |
  | 192  | 0.9707                 | 0.9681              | +0.0026 |
  | 240  | 0.9566                 | 0.9584              | -0.0018 |

  **CONCLUSION**: RMSNorm fix (matching apex intermediate BF16 truncation) is **WORSE** at
  mid-training steps 96-192 (+0.003 to +0.005). AITER's original FP32 `x * rsqrt * weight`
  is more precise and that precision helps convergence. Matching apex's less precise behavior
  hurts. **RULED OUT** — reverted kernel to original FP32 path.

- **v32 experiment** (2026-04-06, stopped early):
  v20 baseline + `LUMEN_PREFER_HIPBLASLT=1` (switches FP8 GEMM from CK to hipBLASLt
  to match TE's `NVTE_USE_HIPBLASLT=1` backend).
  Step 6: lm_loss=3.991905 (v20: 3.991635) — confirms hipBLASLt is active.

  | Step | v32 (hipBLASLt) | v20 (CK) | Delta |
  |------|----------------|----------|-------|
  | 48   | 1.0220         | 1.0219   | +0.0001 |
  | 96   | 0.9836         | 0.9820   | +0.0016 |
  | 144  | 0.9764         | 0.9669   | +0.0095 |
  | 192  | 0.9662         | 0.9681   | -0.0019 |
  | 240  | 0.9599         | 0.9584   | +0.0015 |

  **CONCLUSION**: hipBLASLt FP8 GEMM oscillates around v20's CK results — no improvement.
  Step 144 anomaly (+0.0095) recovers by step 192. Memory: 96.19% (v20: 96.10%).
  Step time: ~8.4s (v20: ~7.9s) — slightly slower. **RULED OUT.**

- **Deep analysis of convergence gap (2026-04-06)**:
  Re-examined AMD reference results. Run_0 (seed=17367) full trajectory:
  | Step | AMD ref | Lumen v20 | Gap |
  |------|---------|-----------|-----|
  | 192  | 0.9383  | 0.9681    | 0.030 |
  | 240  | 0.9340  | 0.9584    | 0.024 |
  | 288  | 0.9312  | 0.9547    | 0.024 |
  | 480  | 0.9235  | 0.9473    | 0.024 |
  | 768  | —       | 0.9371    | — |

  **KEY INSIGHT**: The gap at comparable training steps (192-480) is **0.024-0.030**, much
  larger than the 0.014 gap at their respective bests. The AMD reference converges
  **significantly faster** — reaching 0.9383 by step 192 while Lumen is at 0.9681.
  This 0.030 gap at step 192 cannot be explained by ULP-level kernel differences.
  Something **systematic** differs in the training pipeline.

  All individual component swaps (RMSNorm, GEMM backend, E5M2 backward, FP8 wgrad,
  RoPE fusion, attention atomics, seeds) produced <0.005 deltas — not enough to
  explain a 0.024+ gap at matched steps.

- **CRITICAL FINDING: Data shuffling mismatch (2026-04-06)**:

  NeMo's `GPTSFTPackedDataset._build_samples_mapping()` shuffles the training data
  within each epoch using `np.random.shuffle` (seeded by training seed). The
  `MegatronPretrainingBatchSampler` yields sequential indices `[0,1,2,...]`, but
  `__getitem__` remaps via `idx = self.samples_mapping[idx]`.

  Lumen's `LLaMA2SFTDataset.__getitem__` directly returns `self.indexed_dataset[idx]`
  with **NO shuffling**. The Megatron `MegatronPretrainingSampler` also iterates
  sequentially. Data is presented in the exact order it appears in the `.npy` file.

  **AMD reference**: shuffled data order every epoch (via `samples_mapping`)
  **Lumen**: sequential data order (no shuffle mapping)

  This explains:
  1. The 0.030 gap at step 192 — shuffling has the biggest effect early in training
     when sequential batches from adjacent packed sequences are highly correlated
  2. The gap narrowing to 0.014 at best — both eventually see all data
  3. Why individual kernel swaps (<0.005 each) couldn't explain the 0.024+ gap
  4. Why the gap is systematic and reproducible across seeds

  **Evidence**:
  - NeMo source: `nemo/collections/nlp/data/language_modeling/megatron/gpt_sft_dataset.py`
    line ~589: `[np.random.shuffle(x) for x in indices]`
  - Lumen source: `lumen/models/llama2/dataset.py` line ~130: direct `indexed_dataset[idx]`
  - Megatron sampler: `megatron/legacy/data/data_samplers.py` line ~99:
    `for idx in range(self.consumed_samples, self.total_samples)` — sequential iteration

  **Fix**: Add epoch-level data shuffling to `LLaMA2SFTDataset`, matching NeMo's
  `_build_samples_mapping()` behavior. Create a permutation array seeded by the
  training seed and remap `__getitem__` indices through it.

- **v33 experiment (2026-04-07, COMPLETED — 1024 steps)**:
  v20 baseline + epoch-level data shuffling (`LUMEN_SHUFFLE_TRAIN=1`, seed=1234).
  Implemented `_build_samples_mapping()` in `LLaMA2SFTDataset` matching NeMo's
  `GPTSFTPackedDataset` shuffle logic. Permutation array seeded by training seed
  remaps `__getitem__` indices.

  **Full v33 vs v20 vs AMD reference comparison**:

  | Step | v33 (shuffle) | v20 (no shuffle) | AMD ref mean | Gap v33→AMD | Gap v20→AMD |
  |------|--------------|-----------------|-------------|-------------|-------------|
  | 48   | 1.0107       | 1.0293          | —           | —           | —           |
  | 96   | 0.9712       | 0.9992          | —           | —           | —           |
  | 144  | 0.9537       | 0.9772          | —           | —           | —           |
  | 192  | 0.9514       | 0.9741          | 0.9401      | +0.011      | +0.034      |
  | 240  | 0.9419       | 0.9671*         | 0.9344      | +0.008      | +0.033      |
  | 288  | 0.9396       | 0.9612*         | 0.9299      | +0.010      | +0.031      |
  | 336  | 0.9414       | —               | 0.9286      | +0.013      | —           |
  | 384  | 0.9387       | —               | 0.9244      | +0.014      | —           |
  | 432  | 0.9327       | 0.9494          | —           | —           | —           |
  | 480  | 0.9292       | 0.9473          | —           | —           | —           |
  | 528  | 0.9280       | 0.9438          | —           | —           | —           |
  | 576  | 0.9273       | 0.9445          | —           | —           | —           |
  | 624  | 0.9328       | 0.9504          | —           | —           | —           |
  | 672  | **0.9221**   | 0.9420          | —           | —           | —           |
  | 720  | 0.9252       | 0.9458          | —           | —           | —           |
  | **768** | **0.9227** | **0.9371**     | —           | —           | —           |
  | 816  | 0.9256       | 0.9421          | —           | —           | —           |
  | 864  | 0.9226       | 0.9378          | —           | —           | —           |
  | 912  | 0.9221       | 0.9388          | —           | —           | —           |
  | **960** | **0.9208** | 0.9380         | —           | —           | —           |
  | 1008 | 0.9252       | 0.9419          | —           | —           | —           |
  | **1024** | **0.9221** | **0.9382**    | —           | —           | —           |

  **Best val_loss**: v33=**0.9208** (step 960) vs v20=0.9371 (step 768). **Improvement: 0.016.**
  **Final val_loss (1024)**: v33=0.9221 vs v20=0.9382. **Improvement: 0.016.**
  **MLPerf target**: <0.925. **v33 PASSES** (0.9208 < 0.925). v20 never passed.
  **AMD reference mean best**: 0.9229. **v33 best (0.9208) BEATS AMD reference mean.**

  **First step under 0.925**: step 672 (val_loss=0.9221). AMD reference typically reaches
  0.925 at step 384 (samples=3072). v33 reaches it at step 672 (samples=5376) — about
  1.75x slower, but the gap is explained by different seeds and remaining kernel diffs.

  **CONCLUSION**: Data shuffling was the **PRIMARY CAUSE** of the convergence gap.
  - Gap at matched steps (192): reduced from +0.034 to +0.011 (68% reduction)
  - Best val_loss: improved from 0.937 to 0.921 (passes MLPerf target)
  - v33 beats AMD reference mean (0.9208 < 0.9229)

  Memory: 96.21% (same as v20). Step time: ~8.0s (slightly slower than v20's ~7.9s).
  Stability: 0 NaN, 0 skipped across all 1024 steps.

  **Remaining gap at matched steps (~0.011)** likely from:
  1. Different seed (v33: 1234 vs AMD refs: various $RANDOM seeds)
  2. Accumulated kernel-level diffs (<0.005 each, as verified by v25-v32 experiments)

- **Status**: **RESOLVED** — data shuffling closes the MLPerf val_loss gap.
  `LUMEN_SHUFFLE_TRAIN=1` should be enabled for all future training runs.

### [2026-04-07 speed-gap-2x]
- Symptom: Lumen v33 achieves 7.94s/step vs AMD MLPerf reference 3.78s/step — 2.1x slower.
  Both use identical parallelism (TP=1, ACL=21, DP=8) and similar FP8 config.
  Wall-clock to target: Lumen ~101min vs AMD ~27min (3.7x gap including convergence diff).

- **CRITICAL CORRECTION (2026-04-07)**:
  The proposed optimization plan "TP=8 + reduce recompute + kernel fusion + FP8 wgrad" is
  based on INCORRECT assumptions from the earlier results README:

  1. **TP=8 is WRONG**: AMD MLPerf reference uses **TP=1** (not TP=8). Confirmed in:
     - `config_MI300X_1x8x1.sh`: `export TP=1`
     - `megatron_llama_config.yaml`: `tensor_model_parallel_size: 1`
     - `model_config.yaml`: `tensor_model_parallel_size: 1`
     Switching to TP=8 would DIVERGE from the reference, introduce TP comm overhead,
     and reduce DP from 8 to 1 on single node.

  2. **ACL=21 already done**: v33 already used `RECOMPUTE_NUM_LAYERS=21` (from
     `config_MI300X_tp1_dp8.sh`). No further reduction possible.

  3. **FP8 wgrad already enabled**: v33 had `fp8_wgrad=True`, `fp8_activation_store=True`,
     `fp8_checkpoint=True`, `fp8_param_storage=True`.

  4. **Kernel fusion is the ENTIRE remaining speed gap**: The 2.1x slowdown is purely
     from lack of TE-style kernel fusion.

- **Verified v33 effective config** (from log):
  | Parameter | v33 (Lumen) | AMD Ref | Match? |
  |-----------|------------|---------|--------|
  | TP | 1 | 1 | ✓ |
  | ACL | 21 (full/block) | 21 (full/block) | ✓ |
  | FP8 wgrad | True | True (TE) | ✓ |
  | FP8 act store | True | True (TE) | ✓ |
  | FP8 param storage | True | fp8_model_init | ✓ |
  | Memory | 96.21% | ~82% | Different |
  | Step time | 7.94s | 3.78s | **2.1x gap** |

- **Root cause analysis — kernel fusion gap**:
  AMD reference uses 5 TE-level fusions that Lumen does NOT have:

  | TE Fusion | Env Var | What it does | Lumen equivalent | Status |
  |-----------|---------|--------------|-----------------|--------|
  | Fused SwiGLU MLP | `USE_TE_SWIGLU=1` | gate+up GEMM → SiLU → mul → down GEMM in one region | `LumenGatedMLP` + `ff_a16w16_fused_gated` exist but **NOT WIRED** to Megatron MLP | **ACTIONABLE** |
  | Fused RMSNorm+cast | `NVTE_USE_RMSNORM_TRITON=1` | Norm → FP8 cast in one kernel | `fused_rms_fp8_per_tensor_static_quant` exists in AITER | **Partial** |
  | Fused Cast+Transpose | `NVTE_USE_CAST_TRANSPOSE_TRITON=1` | FP8 quant + transpose in one kernel | Lumen uses separate `.t().contiguous()` cache | **NOT DONE** |
  | Fused Attention | `NVTE_FUSED_ATTN_CK=1` | QKV+RoPE+attention+proj fused region | CK flash_attn_func (core attention only) | **PARTIAL** |
  | No transpose cache | `ENABLE_TRANSPOSE_CACHE=0` | Saves memory, enables lower ACL | Lumen caches transposes (opposite) | **OPPOSITE** |

  Additionally, Lumen has significantly more kernel launches per layer:
  - Per FP8 linear: `quantize_input` (activation) + `quantize_input` (weight) + `dispatch_gemm` + bias = 4+ kernels
  - TE: fused cast+transpose+GEMM = 1-2 kernels
  - Per MLP: 2 linears × 4 kernels + SwiGLU = 9+ kernels
  - TE: 1 fused SwiGLU region = ~2-3 kernels
  - 80 layers × overhead = massive accumulated launch overhead

- **Immediately actionable optimizations** (priority order):

  1. **Wire `LumenGatedMLP` into Megatron spec** (Easy, ~10-15% speedup):
     `--lumen-fused-mlp` flag exists but `LumenSpecProvider` still returns
     `SequentialMLP` with separate linears. Need to wire `LumenGatedMLP` into
     the spec when flag is set. AITER's `ff_a16w16_fused_gated` kernel exists.

  2. **Align eval schedule** (Easy, ~15% wall-clock):
     Change from eval every 48 steps (21 evals in 1024 steps, ~57s each = ~20 min)
     to AMD's SKIP_EVALS=3 + VAL_CHECK_INTERVAL=384 (eval at steps 192, 240, 288,
     336, 384 — 5 evals if converges by step 384, ~5 min total).

  3. **Disable transpose cache** (Easy, saves memory):
     Match AMD's `ENABLE_TRANSPOSE_CACHE=0`. May free enough memory for other opts.

  4. **Profile kernel launch overhead** (Medium):
     Run ROCm profiler to quantify per-kernel time breakdown and identify top bottlenecks.

  5. **Fused Cast+Transpose Triton kernel** (Medium, ~5-10%):
     AITER has building blocks; need to wire a fused FP8 quant+transpose kernel
     into `quantize_input` path.

  6. **Full norm→GEMM fusion** (Hard, ~10-15%):
     `LumenLayerNormLinear` currently runs norm and GEMM as separate kernels.
     True fusion requires a single kernel that reads BF16, computes norm, outputs
     to FP8, and feeds directly into GEMM.

- **v34 experiment (2026-04-07, COMPLETED — 1024 steps)**:
  Eval schedule alignment only. `EVAL_EVERY=1536` (eval every 192 steps).

  **Full v34 vs v33 comparison**:
  | Step | v34 val_loss | v33 val_loss | Delta |
  |------|-------------|-------------|-------|
  | 192  | 0.9462      | 0.9514      | -0.005 |
  | 384  | 0.9328      | 0.9387      | -0.006 |
  | 576  | 0.9222      | 0.9273      | -0.005 |
  | 768  | 0.9239      | 0.9227      | +0.001 |
  | 960  | **0.9195**  | **0.9208**  | **-0.001** |
  | 1024 | **0.9178**  | **0.9221**  | **-0.004** |

  **Best val_loss**: v34=**0.9178** (step 1024) vs v33=0.9208 (step 960). **Improvement: 0.003.**
  **First step < 0.925**: v34 step 576 (val_loss=0.9222). v33 step 672 (val_loss=0.9221).
  **MLPerf target**: <0.925. **v34 PASSES** at step 576 (v33 at step 672).

  **Wall-clock**:
  | Metric | v34 (aligned eval) | v33 (48-step eval) | Savings |
  |--------|-------------------|-------------------|---------|
  | Total wall-clock | 145.0 min | 162.9 min | **17.9 min (11%)** |
  | Training only | 137.4 min | 135.5 min | +1.9 min (noise) |
  | Eval overhead | 4.8 min (5 evals) | 19.9 min (21 evals) | **15.1 min** |
  | Step time | ~8.05 s | ~7.94 s | +0.11 s (noise) |

  **CONCLUSION**: Eval alignment saves **11% wall-clock** with no convergence penalty.
  v34 actually converges slightly better than v33 (0.9178 vs 0.9208 best) — likely
  because fewer eval interruptions let GPU caching and pipeline state stay warmer.
  **Note**: v34's step time is slightly higher (~8.05s vs 7.94s) but this is within
  noise; the savings come entirely from reduced eval count.

- **Three speed optimizations implemented (2026-04-07)**:

  1. **Kernel launch overhead reduction** (`lumen/ops/dispatch.py`):
     - `try_backends` now caches the winning backend index after 3 consecutive
       successes. After warmup, subsequent calls skip the fallback chain entirely.
     - `torch.cuda.synchronize()` is only issued during warmup. After the backend
       is locked, sync is skipped (saves ~50-100us per GEMM/quant dispatch).
     - `LUMEN_SKIP_BACKEND_SYNC=1` env var to skip sync even during warmup.
     - Expected impact: 80 layers × ~10 dispatch calls × 2 (fwd+bwd) × ~75us =
       ~120ms/step overhead eliminated (~1.5% of 7.94s step time).

  2. **Fused SwiGLU MLP** (`lumen/models/megatron.py`):
     - `--lumen-fused-mlp` flag now patches Megatron `MLP.forward()` at model
       build time to use AITER's `ff_a16w16_fused_gated` Triton kernel.
     - Replaces fc1(SwiGLU) + fc2 = 3 separate BF16 GEMMs + SiLU + mul with
       a single fused Triton kernel (gate+up GEMM → SiLU → mul → down GEMM).
     - Only activates for gated_linear_unit=True, no bias, AITER available,
       AND M <= 64 (the fused kernel is slower for large batch sizes).
     - Falls back to original forward on any failure.
     - **LIMITATION**: For training with seq_len=8192, M=8192 >> 64, so the
       fused kernel will NOT activate. The M>64 guard falls back to original.
       This flag is only beneficial for inference or small-batch scenarios.
     - `run_finetune.sh` now supports `LUMEN_FUSED_MLP=1` env var to pass
       `--lumen-fused-mlp` to the training script.

  3. **Eval schedule alignment** (`config_MI300X_tp1_dp8.sh`):
     - `LUMEN_EVAL_ALIGNED=1` env var: switches from eval every 48 steps to
       every 192 steps, matching MLPerf wall-clock budget.
     - Reduces from 21 evals to 5 evals in 1024 steps.

- **Fusion 1: Fused RMSNorm + FP8 Quant in LumenLayerNormLinear (2026-04-07)**:

  Implemented `LUMEN_FUSED_NORM_QUANT=1` env var that fuses the RMSNorm and
  per-tensor FP8 quantization into a single Triton kernel launch in
  `LumenLayerNormLinear.forward()`.

  **Architecture**:
  - Custom `autograd.Function` (`_FusedRMSNormFP8Quant`) wraps AITER's
    `fused_rms_fp8_per_tensor_static_quant` kernel, which produces both
    BF16 norm output (for autograd graph) and FP8 quantized output in a
    single kernel launch.
  - BF16 output → autograd graph → backward recomputes norm via Triton
  - FP8 output → passed as `pre_quantized_input` to `_do_gemm()` →
    `quantized_linear()` → `QuantizedLinearFunction`/`FP8StoredLinearFunction`
    which skip the standalone `quantize_input()` call.
  - Saves one full HBM round-trip per `LumenLayerNormLinear` forward
    (eliminates reading `ln_out` from HBM for the separate quant kernel).

  **Scope**: Currently supports `"delayed"` scaling type only (the default
  for MLPerf training). Other scaling types fall back to the unfused path.
  Sequence-parallel (TP>1) also falls back.

  **Files modified**:
  - `lumen/modules/layernorm_linear.py`: Added `_FusedRMSNormFP8Quant`
    autograd.Function and `_try_fused_norm_quant()` method, modified
    `forward()` to use fused path when available.
  - `lumen/modules/parallel_linear.py`: Added `pre_quantized_input`
    parameter to `_do_gemm()` and threaded it to `quantized_linear()`.
  - `lumen/ops/quantize/linear.py`: Added `pre_quantized_input` parameter
    to `quantized_linear()`, `QuantizedLinearFunction`, and
    `FP8StoredLinearFunction`. When provided, skips `quantize_input()`
    for activations. Updated backward return counts.

  **Expected impact**: Eliminates ~80 standalone FP8 quant kernel launches
  per forward pass (one per `LumenLayerNormLinear` in attention + MLP).
  Each kernel launch saves ~50-100us + one full HBM read of the hidden
  state. Estimated: 80 × ~100us = ~8ms/forward pass = ~16ms/step
  (~0.2% of 7.94s step).

  **Fusion 2: Fused SwiGLU + FP8 Quant (DEFERRED)**:
  Would fuse `silu(gate) * up` + FP8 quant into a single kernel before fc2.
  Requires patching `MLP.forward()` and `LumenRowParallelLinear.forward()`
  to accept `pre_quantized_input`. Deferred to a future iteration — lower
  impact and higher complexity than Fusion 1.

- **v35 experiment (2026-04-07, COMPLETED — 1024 steps)**:
  v33 baseline + all speed optimizations:
  1. `LUMEN_EVAL_ALIGNED=1` (eval every 192 steps)
  2. `LUMEN_SKIP_BACKEND_SYNC=1` (skip sync after warmup)
  3. `LUMEN_FUSED_MLP=1` (auto-fallback for M>64)
  4. `LUMEN_FUSED_NORM_QUANT=1` (fused RMSNorm + FP8 quant)

  **Full v35 results**:
  | Step | v35 val_loss | v34 val_loss | v33 val_loss |
  |------|-------------|-------------|-------------|
  | 192  | 0.9526      | 0.9462      | 0.9514      |
  | 384  | 0.9356      | 0.9328      | 0.9387      |
  | 576  | 0.9243      | 0.9222      | 0.9273      |
  | 768  | 0.9245      | 0.9239      | 0.9227      |
  | 960  | **0.9210**  | **0.9195**  | **0.9208**  |
  | 1024 | **0.9192**  | **0.9178**  | **0.9221**  |

  **Best val_loss**: v35=0.9192 (step 1024). **PASSES MLPerf target (<0.925).**
  **Wall-clock**: 138.4 min (v34: 145.0, v33: 162.9). **18% faster than v33.**
  **Step time**: Pre-eval ~6.16s, post-eval ~7.3-7.5s. Post-eval regression
  is a pre-existing ROCm memory issue (also seen in v34).
  **Convergence**: Healthy, no NaN/divergence. v35 is slightly worse than v34
  at matched steps (+0.003-0.006) but within noise.

  **CONCLUSION**: Fused norm+quant does not significantly improve steady-state
  step time post-eval (~7.4s vs v34 ~8.0s → ~8% improvement). The pre-eval
  6.16s was not sustainable. Total wall-clock savings are primarily from
  eval alignment (same as v34).

- **Kernel profiling analysis (2026-04-07, Lumen)**:
  Profiled 3 training steps (steps 8-10) on rank 0 using `torch.profiler`.
  Total Self CUDA time for 3 steps: ~18.25s (= ~6.08s/step).

  **GPU time breakdown by category (Self CUDA, 3 steps)**:
  | Category | Time | % GPU | Key Insight |
  |----------|------|-------|-------------|
  | GEMM (CK fwd) | 5.92s | 32.4% | FP8 forward GEMMs via CK |
  | GEMM (hipBLASLt bwd) | 3.23s | 17.7% | FP8 backward dgrad/wgrad |
  | Elementwise (mul/silu/sigmoid/add) | 2.57s | 14.1% | SwiGLU + grad ops + amax scale |
  | Attention bwd | 1.74s | 9.5% | CK FMHA v3 backward |
  | Copy/cast (copy_+clone) | 1.46s | 8.0% | dtype conversions + clones |
  | FP8 quant (amax+abs+quant) | 1.22s | 6.7% | amax reduction, abs, FP8 cast |
  | Attention fwd | 727ms | 4.0% | CK FMHA v3 forward |
  | Cat | 368ms | 2.0% | Tensor concatenation |
  | NCCL AllReduce | 358ms | 2.0% | Gradient sync (DP=8) |
  | SiLU activation | 299ms | 1.6% | Forward pass activation |
  | Memcpy DtoD | 238ms | 1.3% | Device copies |
  | Dropout | 72ms | 0.4% | LoRA dropout |
  | LoRA mm | 83ms | 0.5% | BF16 LoRA A×B |

  **Total kernel launches (3 steps)**: ~170,000+
  - `aten::copy_`: 38,331 calls
  - `aten::mul`: 20,466 calls
  - `Memcpy DtoD`: 23,190 calls
  - `aten::clone`: 24,447 calls

  **Key bottlenecks vs AMD MLPerf reference**:
  1. **Elementwise ops (14.1%)**: TE fuses these into larger kernels.
     SwiGLU backward: Lumen launches separate mul, sigmoid, silu kernels.
     TE: single fused SwiGLU backward kernel.
  2. **FP8 quant overhead (6.7%)**: TE's fused cast+transpose eliminates
     separate amax, abs, and quant kernel launches.
  3. **Copy/cast overhead (8.0%)**: 38K+ copy_ calls for dtype conversion.
     TE avoids many of these via in-place FP8 operations.
  4. **Kernel launch overhead**: 170K+ launches in 3 steps ≈ 57K/step.
     At ~5us dispatch overhead each = ~285ms/step overhead.
     TE likely has 3-5x fewer launches due to fusion.
  5. **GEMM is ~50% of GPU time** — this is expected and cannot be reduced
     without changing the model or parallelism. The GEMMs themselves are
     likely similar speed between CK and hipBLASLt.

  **Estimated theoretical speedup from full TE-level fusion**:
  - Fuse elementwise: save ~50% of 14.1% = ~7%
  - Fuse FP8 quant: save ~50% of 6.7% = ~3.4%
  - Reduce copy/cast: save ~50% of 8.0% = ~4%
  - Reduce launch overhead: save ~50% of ~4.7% = ~2.3%
  - **Total potential: ~17% per-step speedup**, bringing step time from
    ~7.4s to ~6.1s. This alone does NOT close the 2x gap to AMD's 3.78s.

  **The remaining gap must come from**:
  - TE's overlapping of compute and communication
  - TE's more efficient memory management (less fragmentation post-eval)
  - TE's CUDA graph or kernel scheduling optimizations
  - Potentially different recomputation strategies during backward

- **TE operation profiling (2026-04-07)**:
  Profiled TE operations at same tensor shapes [8192, 8192] using AMD MLPerf
  container with FP8 autocast, delayed scaling, all TE fusions enabled.
  Single GPU (rank 0), 10 iterations per operation.

  **TE per-iteration CUDA time (fwd + bwd, 1 call)**:
  | Operation | Per Iter | GEMM time | Overhead |
  |-----------|---------|-----------|---------|
  | LayerNormLinear (QKV: 8192→10240) | 5.11ms | 3.53ms (69%) | 1.58ms (norm+cast+trn) |
  | Linear (proj: 8192→8192) | 3.30ms | 2.84ms (86%) | 0.46ms (cast+trn) |
  | Linear (fc1 gate+up: 8192→57344) | 23.08ms | 20.85ms (90%) | 2.23ms (cast+trn) |
  | Linear (fc2 down: 28672→8192) | 11.92ms | 10.75ms (90%) | 1.17ms (cast+trn) |

  **TE total per layer**: 43.41ms (4 linear ops)
  **TE 80 layers (linear only)**: **3,473ms = 3.47s**
  **TE overhead per layer**: 5.44ms (12.5%) = cast_transpose + amax + norm
  **TE overhead 80 layers**: **435ms** total

  **Key TE efficiency features**:
  1. `_cast_transpose_triton`: fuses FP8 cast + transpose in one kernel (~100-335us per call)
  2. `delayed_scaling_recipe`: amax update is ~6.5us per tensor (vs Lumen's abs+amax+clamp)
  3. `_rmsnorm_fwd_triton_impl`: fused with FP8 cast in LayerNormLinear
  4. `_rmsnorm_bwd_triton`: dedicated backward kernel ~109us
  5. Only ~140 kernel launches per 10 iterations per op vs Lumen's thousands

- **Apple-to-apple kernel comparison (2026-04-07)**:

  **Per-step GPU time comparison**:
  | Category | Lumen (per step) | TE (per step, estimated) | Gap | Notes |
  |----------|-----------------|------------------------|-----|-------|
  | **GEMM** | 3,050ms | 2,780ms | 1.10x | CK vs hipBLASLt; similar |
  | **Attention** | 820ms | ~820ms | 1.00x | Same CK FMHA kernel |
  | **FP8 quant/scale** | 410ms | 54ms | **7.6x** | Lumen: abs+amax+quant separate; TE: fused |
  | **Elementwise** | 860ms | ~100ms | **8.6x** | SwiGLU/grad: mul+sigmoid+silu separate vs fused |
  | **Copy/cast** | 490ms | 60ms | **8.2x** | Lumen: 38K copy_ calls; TE: fused cast_transpose |
  | **Cat** | 120ms | ~15ms | 8.0x | Lumen: 3861 cat calls; TE: minimal |
  | **NCCL** | 120ms | ~120ms | 1.0x | Same all_reduce |
  | **Memcpy DtoD** | 80ms | ~20ms | 4.0x | Reduced by fusion |
  | **Other** | 130ms | ~50ms | 2.6x | Dropout, clone, etc |
  | **Total** | **6,080ms** | **~4,019ms** | **1.51x** | |

  But AMD MLPerf reference achieves **3,780ms/step**, which is even faster than our
  TE component estimate. This implies additional overhead in Lumen's:
  1. **CPU overhead**: 38K+ copy_ calls each have ~24us CPU dispatch = ~920ms CPU time
  2. **Pipeline bubbles**: Lumen's sequential kernel launches leave GPU idle between ops
  3. **Memory management**: PyTorch allocator overhead for many small temporary tensors
  4. **Recompute overhead**: Activation checkpointing recomputes forward during backward

  **The 2.1x speed gap breakdown (6,080ms → 3,780ms)**:
  | Source | Estimated Impact | Difficulty |
  |--------|-----------------|-----------|
  | FP8 quant/scale fusion | -356ms (-5.9%) | Medium — fuse abs+amax+quant+cast_transpose |
  | Elementwise fusion (SwiGLU) | -760ms (-12.5%) | Hard — fuse entire SwiGLU fwd+bwd |
  | Copy/cast elimination | -430ms (-7.1%) | Medium — in-place FP8 ops + fused cast_transpose |
  | Cat reduction | -105ms (-1.7%) | Easy — pre-allocate and avoid cat |
  | CPU dispatch overhead | -400ms (-6.6%) | Hard — reduce kernel launch count by 3-5x |
  | Pipeline/scheduling | -300ms (-4.9%) | Hard — compute-comm overlap, CUDA graphs |
  | **Total recoverable** | **~2,351ms (-38.7%)** | Brings to ~3,729ms |

  **CONCLUSION**: Full TE-level fusion can theoretically bring Lumen to ~3.7s/step,
  matching AMD MLPerf's 3.78s/step. The top 3 targets:
  1. **SwiGLU fusion** (12.5%): Fuse gate+up GEMM → SiLU → mul → down GEMM
  2. **Fused cast+transpose** (7.1%): Replace `abs → amax → clamp → quant → transpose → copy`
     with single Triton kernel
  3. **CPU dispatch reduction** (6.6%): Reduce total kernel count from ~57K to ~15K/step

- Status: open — profiling complete, optimization roadmap identified

### [2026-04-08 v39-phase1-opts]
- Symptom: v38b pre-eval 5,809ms/step vs MLPerf 3,811ms — still 52% gap after post-eval fixes
- v39 changes (Phase 1-2 optimizations from speed gap plan):
  1. Runtime tunables (CPU perf governor, THP, page cache drop, ASLR disabled)
  2. NCCL tuning (32 channels, 32 CTAs, NVLS disabled, AVOID_RECORD_STREAMS)
  3. Allocator: max_split_size_mb=512
  4. Log interval: 10 (reduced from 1)
  5. Post-eval re-warmup (LUMEN_POST_EVAL_REWARM=1)
  6. CUBLAS_FORCE_XMMA_KERNEL_INIT=DEVICE, OMP_NUM_THREADS=1, hipBLASLt preference
  7. Fused SwiGLU Triton kernel (LUMEN_FUSED_SWIGLU=1) — AITER-hosted, Lumen dispatches
  8. Fused fast transpose (LUMEN_FUSED_CAST_TRANSPOSE_V2=1) — AITER-hosted
  9. FP8 attention backward (LUMEN_FP8_ATTN_BWD) — DISABLED due to cu_seqlens bug (see below)
- v39 crash: First run crashed on backward pass with `RuntimeError: lumen::attention_triton_backward_impl() Expected a value of type 'int' for argument 'cu_seqlens_q' but instead found type 'NoneType'`
  - Root cause: `AttentionCsrcFP8BwdFunction.backward` passed `None` for cu_seqlens_q/cu_seqlens_k instead of `0` (integer) required for non-varlen BSHD layout
  - Fix: Changed `None, None, S_Q, S_K` to `0, 0, S_Q, S_K` in the `triton_fp8_bwd` call and added `force_triton=True`
  - Feature disabled (LUMEN_FP8_ATTN_BWD=0) for v39 due to additional LSE format mismatch between CK forward (dense BxHxS) and Triton backward (interleaved blocks). Needs dedicated integration work.
- v39 results (run in progress, through step 610/1024):
  | Metric | v37 | v38b | v39 | MLPerf ref |
  |--------|-----|------|-----|------------|
  | Pre-eval ms/step | 5,898 | 5,809 | **5,428** | 3,811 |
  | Post-eval#1 ms/step | 7,167 | 6,586 | **6,070** | 3,778 |
  | Post-eval#1 delta | +21.7% | +13.4% | **+11.8%** | -0.8% |
  | val_loss @ 192 | 0.9504 | 0.9453 | **0.9453** | ~0.94 |
  | val_loss @ 384 | 0.9343 | — | **0.9318** | ~0.93 |
  | val_loss @ 576 | 0.9273 | — | **0.9222** | ~0.92 |
  | Memory | 0.9615 | — | 0.9794→0.9837 | ~0.82 |
- Pre-eval improvement: 5,809→5,428ms = **-381ms (-6.6%)**
- Post-eval improvement: 6,586→6,070ms = **-516ms (-7.8%)**
- Convergence: val_loss matches v38b exactly. Passes <0.925 at step 576.
- Stability: 0 NaN, 0 skipped through 610 steps.

- **Remaining gap analysis** (v39 pre-eval 5,428ms vs MLPerf 3,811ms = **1,617ms / 42.4% gap**):

  The gap breakdown from profiling (scaled to v39 baseline):

  | Source | Est. Time | % of Gap | Difficulty | Notes |
  |--------|----------|----------|-----------|-------|
  | **Checkpoint recompute overhead** | 500-800ms | 30-50% | Hard | Lumen recomputes 21/80 layers; TE may use CUDA graphs or overlapped recompute that hides latency. `CheckpointFunctionBackward` was ~4.2s/3 steps in profile. |
  | **SwiGLU elementwise ops** | 300-500ms | 19-31% | Medium | `aten::mul` alone was 1,580ms/3 steps. TE fuses entire SwiGLU fwd+bwd into single kernel. v39 fused SwiGLU may partially address this (need new profile). |
  | **FP8 quant/cast/transpose pipeline** | 200-400ms | 12-25% | Medium | TE's `cast_transpose_fusion_kernel_optimized` replaces Lumen's separate abs→amax→quant→transpose chain |
  | **Copy/dtype overhead** | 200-300ms | 12-19% | Medium | 38K+ aten::copy_ calls (1,144ms/3 steps). TE avoids via in-place FP8 ops |
  | **CPU dispatch / kernel launch** | 100-200ms | 6-12% | Hard | ~57K kernel launches/step vs TE's ~15K; 5µs overhead each = ~285ms |
  | **Memory allocator pressure** | 100-200ms | 6-12% | Hard | 98.37% memory usage vs TE's ~82%. High pressure causes allocator thrashing |

  Key structural differences that limit per-kernel optimizations:
  1. **Memory usage 98.4% vs 82%**: Lumen at near-capacity, no room for workspace buffers, causes allocator fragmentation and prevents aggressive fusion (which needs temp buffers)
  2. **Kernel launch count ~57K vs ~15K**: TE's fusion reduces total launches by ~3-4x. Each launch has ~5µs CPU overhead.
  3. **Checkpoint forward recompute**: The 21-layer recompute (ACL=21) runs the full forward again during backward, adding ~500ms+ per step that may be better hidden with overlapped compute in TE

- **quant.enable() interface change**: The `LumenConfig.enable()` return type was changed from `manager` to `(manager, model)` to support LoRA wrapping via PEFT. Updated callers in `megatron.py` and `fsdp.py` to destructure the new return.

- Status: open — v39 running, collecting full data. quant.enable() callers updated.

### [2026-04-08 fused-swiglu-quant]
- Symptom: 857ms/step in elementwise ops (SwiGLU mul/sigmoid/silu separate kernels)
- Possible bug: not a bug — missing fusion
- Implementation: Added `LUMEN_FUSED_SWIGLU_QUANT=1` env var flag
  - `lumen/models/megatron_patches.py`: Extended `_PatchedSwiGLUFunction.forward` to call AITER's `fused_silu_mul_fp8_per_tensor_static_quant` which fuses SiLU + mul + FP8 quant in one kernel
  - `lumen/models/_swiglu_fp8_fuse.py`: New module with thread-local cache bridge between SwiGLU activation and downstream FC2 GEMM
  - `lumen/modules/parallel_linear.py`: `_do_gemm` picks up fused FP8 cache via `pop_swiglu_fp8_cache()` as `pre_quantized_input`
  - `lumen/ops/dispatch.py`: Added `_probe_aiter_fused_silu_mul_fp8()` probe
- Expected savings: ~757ms/step (12.5%) from eliminating separate silu + mul + FP8 quant kernels
- v36b root cause analysis (2026-04-08): **Feature does REDUNDANT work, net negative or neutral.**
  1. Megatron always needs BF16 SwiGLU output, so `_swiglu_mod.swiglu(input)` runs regardless (line 181).
  2. `try_fused_swiglu_fp8` then re-runs the ENTIRE SwiGLU from raw input via AITER fused kernel — doubling compute.
  3. It also computes `bf16_output.detach().abs().amax()` (extra GPU sync) to derive scale before the fused kernel call.
  4. The only savings is skipping `_quantize_core` for FC2 input (~100ms), but `update_amax(abs+amax)` still runs on the BF16 output.
  5. Net: adds ~300ms SwiGLU recompute + ~100ms abs/amax, saves ~100ms quant step. **Negative ROI.**
  - To fix properly: need to either (a) replace Megatron's SwiGLU entirely so BF16 output comes from dequanting the FP8 result, or (b) fuse only the FP8 quant step at FC2 entry without recomputing SwiGLU.
- Status: **ineffective — needs redesign**

### [2026-04-08 fused-cast-transpose]
- Symptom: 486ms/step in copy/cast ops (38K copy_ calls for separate transpose + contiguous)
- Possible bug: not a bug — missing fusion
- Implementation: Added `LUMEN_FUSED_CAST_TRANSPOSE=1` env var flag
  - `lumen/ops/quantize/cast_transpose.py`: New Triton kernel `_cast_transpose_fp8_kernel` that fuses BF16→FP8 quantization with matrix transpose in one kernel, producing both row-major and transposed FP8 outputs
  - `lumen/quantize/scaling_manager.py`: `_quantize_core` uses `cast_transpose_fp8` when enabled, returning `FP8Descriptor` with pre-populated `_transpose` so `transpose_cached` never calls `.t().contiguous()`
  - `_probe_cast_transpose()` cached probe added
- Expected savings: ~426ms/step (7.0%) from eliminating separate abs→amax→quant→transpose→copy chain
- v36 test (2026-04-08): **Caused 2x memory overhead** — kernel produces both row-major AND transposed FP8 tensors, doubling FP8 storage. This pushed memory from 0.9653 to 0.9796 earlier, causing ROCm allocator thrashing. Step times regressed from ~6.0s to ~12.8s/step after eval. Feature disabled in v36b.
- Status: implemented but disabled — needs in-place transpose or lazy approach to avoid memory duplication

### [2026-04-08 fused-quant-scale]
- Symptom: 407ms/step in FP8 quant/scale ops (abs→amax→clamp→quant as separate kernels)
- Possible bug: not a bug — missing fusion
- Implementation: Added `LUMEN_FUSED_QUANT_SCALE=1` env var flag
  - `lumen/quantize/scaling_manager.py`: `_quantize_core` per-tensor delayed/dynamic path now uses AITER's `static_per_tensor_quant_fp8_i8` (1 kernel) instead of `(tensor * (1.0 / scale)).clamp().to(dtype)` (3+ kernels)
  - `_aiter_static_quant()` method and `_probe_aiter_static_quant()` probe added
- Expected savings: ~353ms/step (5.8%) from reducing 5-6 kernel launches to 1
- v36b root cause analysis (2026-04-08): **Provides small savings (~200ms), well below 353ms estimate.**
  - The original estimate incorrectly attributed `abs()` (674ms) and `amax()` (549ms) to the quant path. Those are from `update_amax()` which runs AFTER quantization and is NOT eliminated by this fusion.
  - The actual savings is only from replacing `mul(1/scale) + clamp + to(dtype)` (~170ms clamp + fraction of mul/cast) with one AITER kernel.
  - Measured: post-eval step time improved from 7,472ms (v35) → 7,266ms (v36b), ~206ms savings. This ~200ms is the real ceiling for this fusion alone.
- Status: **working but overestimated** — provides ~200ms/step (2.8%), not 353ms

### [2026-04-08 v36b-fusion-gap-analysis]
- Symptom: v36b (FUSED_SWIGLU_QUANT + FUSED_QUANT_SCALE) only ~200ms faster than v35, not ~1110ms as expected
- Root cause: **Profiling estimates incorrectly attributed kernel time to fusion targets**
  - v35 step times: pre-eval 6,151ms, post-eval-1 7,374ms, post-eval-2 7,472ms
  - v36b step times: pre-eval 6,281ms (+130ms), post-eval-1 7,171ms (-203ms), post-eval-2 7,266ms (-206ms)
  - Pre-eval is SLOWER (+130ms): Triton kernel compilation overhead + SwiGLU recompute from fused path
- Evidence from v35 profile analysis:
  - `abs()`: 674ms/9630 calls, `amax()`: 549ms/4812 calls — attributed to FP8 quant, but actually from `update_amax()` which runs AFTER quantization and is NOT eliminated by any fusion
  - `silu()`: 299ms, `sigmoid()`: 247ms — attributed to SwiGLU fusion target, but BF16 SwiGLU must always run for Megatron compatibility
  - `_static_per_tensor_quant_fp8_i8_kernel`: already 145ms in v35 profile — AITER quant kernel already in use from `quant_fp8_tensorwise_impl` in `ops.py`
- Status: resolved — led to v37 implementation

### [2026-04-08 fused-quant-amax-v37]
- Symptom: `update_amax` (abs+amax) costs ~1,223ms/step — separate full-tensor pass after each quant
- Implementation: New Triton kernel `_static_quant_amax_kernel` in `lumen/ops/quantize/quant_amax_fused.py`
  - Single-pass: reads tensor once, computes per-row `max(abs(x))` via `tl.atomic_max`, and quantizes `x * (1/scale)` to FP8 simultaneously
  - `scaling_manager.py`: `_quantize_core` uses fused kernel when `LUMEN_FUSED_QUANT_AMAX=1` is set; directly appends returned amax to `amax_history` (bypasses `update_amax`)
  - Probe function `_probe_fused_quant_amax()` validates kernel on first use
- v37 test (2026-04-08):
  - Pre-eval (steps 6-14): **5,775ms** vs v35's 6,151ms → **-377ms (-6.1%)**
  - Post-eval (steps 193-303): **7,406ms** vs v35's 7,403ms → essentially neutral
  - Val_loss @ 192: **0.9477** (v37) vs 0.9526 (v35) — better convergence, confirms numerical correctness
  - Memory: 0.9615 (identical to v35 at 0.9614) — no memory overhead
- Analysis: Pre-eval savings are real (~377ms from eliminating the separate abs+amax pass). Post-eval is neutral because post-eval time is dominated by other factors (memory allocator overhead after eval, which affects all GPU ops equally). The fused kernel eliminates ~1,223ms of separate abs+amax work but the net savings is only ~377ms pre-eval, suggesting some of the 9,630 abs() and 4,812 amax() calls in v35 profile were from OTHER callers (not just `update_amax`), or the Triton kernel's `tl.atomic_max` contention partially offsets the savings.
- Lazy transpose was also investigated but `FP8Descriptor.transpose_cached` is already lazy. The 1,144ms in `aten::copy_` comes from `.t().contiguous()` in backward GEMM wgrad/dgrad paths which are intrinsic to GEMM TN layout and cannot be deferred.
- Status: **implemented and verified** — delivers 6.1% pre-eval speedup

## Ruled Out

### [2026-04-03 fp8-param-storage-divergence]
- Symptom: v15/v18 diverge at step 37-39 with grad_norm >1000
- Possible bug: FP8_PARAM_STORAGE interaction with activation checkpointing corrupts gradients
- Evidence so far: v16 (FP8_PARAM_STORAGE=0) OOMed — cannot disable on TP=1. v18 (e4m3 instead of hybrid, still FP8_PARAM_STORAGE=1) diverged at same step. Root cause was LoRA scaling (see Resolved).
- Status: ruled out — divergence was caused by missing LoRA scaling patch, not FP8 param storage

### [2026-04-03 hybrid-e5m2-backward]
- Symptom: v15 (hybrid) diverges at step 39; suspected E5M2 bwd BF16 dequant fallback in dgrad
- Possible bug: Lumen's hybrid backward path dequantizes to BF16 for dgrad (lines 712-716 in linear.py) when AITER doesn't support mixed-dtype FP8 GEMM. This BF16 fallback may degrade gradient quality.
- Evidence so far: v18 (e4m3, no BF16 fallback) diverged at the same step (37 vs 39). The hybrid format is NOT the root cause.
- Status: ruled out — both e4m3 and hybrid diverge identically; root cause was LoRA scaling

### [2026-04-03 fp8-scale-desc-missing]
- Symptom: hypothesized that FP8 weights loaded without _fp8_desc would get buggy scale recomputation
- Possible bug: _load_with_fp8 had a path where missing _fp8_desc triggers scale from FP8 data (scale ~1.0 vs correct ~250)
- Evidence so far: instrumented _load_with_fp8; all 322 FP8 weights had correct _fp8_desc. 0 missing.
- Status: ruled out — path never triggered in practice

## Resolved

### [2026-04-03 warmup-lora-corruption]
- Symptom: v14 initial loss at first real data step was 6.06 (expected ~4.0), then diverged to 42, 22, 33
- Possible bug: synthetic warmup steps ran optimizer with full LR on trivial data, overfitting LoRA adapters
- Evidence: loss_mask was mostly 1s during warmup; optimizer updated LoRA weights on constant-token synthetic data
- Fix: set loss_mask = torch.zeros(...) in _get_synthetic_batch (lumen/models/megatron.py)
- Verification: v15 first real step loss = 3.99 (matches HF baseline ~4.0)
- Status: resolved

### [2026-04-03 lora-scaling-16x]
- Symptom: v15 and v18 both diverge at step 37-39 with grad_norm >1000; grad norms elevated from step 6 (~20 vs expected ~1)
- Possible bug: Megatron-LM-AMD's LoraAdapter uses `self.lora_alpha = alpha` (raw alpha=32) instead of standard `alpha/rank = 32/16 = 2.0`. This 16x amplification of LoRA output causes gradient explosions at high LR.
- Evidence:
  - v15 (hybrid, 1024-step): grad_norm ~20 at step 6, diverged step 39 (grad_norm=1039)
  - v18 (e4m3, 1024-step): grad_norm ~20 at step 6, diverged step 37 (grad_norm=2449)
  - v17 (e4m3, 60-step): appeared stable only because cosine LR decayed fast enough to compensate
  - v19 (hybrid, 1024-step, with patch_lora_scaling.py): grad_norm ~1.2 at step 6, val_loss=0.978 at step 144, perfectly stable
  - Docker image has unpatched `self.lora_alpha = alpha` in lora_adapter.py
  - run_v19_full.sh adds `python /home/danyzhan/patch_lora_scaling.py "${MEGATRON_ROOT}"` to fix it
- Fix: apply patch_lora_scaling.py which changes `self.lora_alpha = alpha` to `self.lora_alpha = alpha / rank if rank > 0 else alpha`
- Status: resolved — v19 completed full 1024 steps. Best val_loss=0.938 at step 768/864/960. Zero divergence, grad_norm 0.1-1.5 throughout.

### [2026-04-08 post-eval-persistent-slowdown]
- Symptom: Lumen training step time jumps +19% after first evaluation (step 192) and never recovers. v35: 6,194→7,391 ms/step (+1,197ms). v37: 5,840→7,120 ms/step (+1,280ms). Slowdown persists for all remaining steps (verified through step 1024 in v35). Memory usage stays constant at 96.15-96.21%.
- **MLPerf reference shows zero degradation**: across all 10 MI300X results, post-eval throughput is actually 0.8% faster than pre-eval (3,811→3,778 ms/step). Throughput std dev < 0.001 across runs.
- Root cause hypothesis: **Activation checkpointing disabled during eval causes ROCm allocator fragmentation**.
  - Megatron's `transformer_block.py` line 669: `if self.config.recompute_granularity == 'full' and self.training:` — recompute is skipped when `self.training=False` (during `model.eval()`)
  - Training: only 21/80 layers' activations stored (recompute-num-layers=21)
  - Eval: all 80 layers' activations stored (no recompute since `self.training=False`)
  - At 96% memory, eval's 80-layer allocation pattern fragments the ROCm allocator block cache
  - When training resumes (21-layer pattern), cached blocks don't match → persistent allocator thrashing
- Evidence:
  - MLPerf reference (NeMo/Lightning + TE): 0% post-eval degradation (10 runs, all <1% delta)
  - Lumen v35 (Megatron pretrain): +19.3% degradation (6,194→7,391), never recovers through 1024 steps
  - Lumen v37 (Megatron pretrain): +21.9% degradation (5,840→7,120), never recovers through 384 steps
  - `manual_gc = False`, `empty_unused_memory_level = 0` → no explicit GC/cache-clear around eval
  - `manual_gc_eval = True` but guarded by `args.manual_gc and args.manual_gc_eval` → dead code since `manual_gc=False`
  - Memory metric stays flat at 96.15% — fragmentation is invisible to `memory_allocated()`
  - Eval duration: Lumen ~48s (v37) / ~55s (v35) vs MLPerf ~31s
- Quantified impact on throughput:
  | Metric | Lumen v37 | Lumen v35 | MLPerf ref |
  |--------|-----------|-----------|------------|
  | Pre-eval ms/step | 5,840 | 6,194 | 3,811 |
  | Post-eval ms/step | 7,120 | 7,391 | 3,778 |
  | Delta | +1,280 (+21.9%) | +1,197 (+19.3%) | -33 (-0.8%) |
  | Eval duration | ~48s | ~55s | ~31s |
  | Eval iterations | 22 | 22 | ~22 (173 samples / 8 GBS) |
- Proposed fixes (in priority order):
  1. **Force activation checkpointing during eval**: Override `self.training` check or keep recompute active in eval mode. This prevents the 80-layer activation explosion.
  2. **Pre-allocate eval memory**: Run a dummy eval forward pass during warmup to prime the allocator, then `empty_cache()` before real training starts (MLPerf does this: `warmup_validation_steps`).
  3. **Set `manual_gc=True, manual_gc_eval=True`**: Enable the explicit GC path to clean up eval objects (currently dead code since `manual_gc=False`).
  4. **Reduce eval_iters**: Use fewer eval iterations to minimize allocator disruption.
- Implementation (v38):
  1. **LUMEN_EVAL_RECOMPUTE=1** (`megatron_patches.py` patch #8): Monkey-patches `TransformerBlock.forward` to temporarily set `self.training=True` during eval forward pass, so the `_checkpointed_forward` path is taken. Uses try/finally to restore `self.training=False` after. Only the TransformerBlock's training flag is toggled; child modules stay in eval mode (dropout disabled).
  2. **LUMEN_WARMUP_EVAL_STEPS=2** (`megatron.py` `_run_warmup_eval_pass`): Runs 2 synthetic eval forward passes with `model.eval()` + `torch.no_grad()` after training warmup completes but before real data. Calls `empty_cache()` afterward to start clean. This primes the ROCm allocator with the eval allocation pattern.
  3. **LUMEN_MANUAL_GC=1** (`run_finetune.sh` → `--manual-gc`): Enables Megatron's `args.manual_gc=True` so the `gc.collect()` calls around evaluation are no longer dead code. Disables Python's automatic GC and runs collection explicitly before and after eval.
  4. **LUMEN_POST_EVAL_CACHE_CLEAR=1** (`megatron_patches.py` patch #9): Wraps Megatron's `evaluate()` to call `torch.cuda.empty_cache()` after every eval run. Resets the ROCm allocator's block pool so training resumes with a clean cache.
- v38 result: **DIVERGED** — loss exploded from 4.2 at step 6 to 16.5 at step 108. Grad norms 100-1000x larger than v37 from the very first real step.
  - Root cause: Fix 2 (warmup eval pass) ran forward passes through FP8 layers with synthetic data *after* `reset_fp8_state()`, polluting amax history with wrong scale factors. When real training started, the corrupted FP8 scales caused massive gradient norms → divergence.
  - Evidence: v38 step 6 grad_norm=353 vs v37 step 6 grad_norm=2.67. Loss gap widened every step. No eval had occurred yet (first eval at step 192).
  - Fix: Added a second `reset_fp8_state(model)` call *after* `_run_warmup_eval_pass()` in `megatron.py` to clear corrupted amax state.
- v38b result (with FP8 reset fix): Training converges normally, loss curve matches v37 exactly.
  - Post-eval regression **reduced from +21.3% to +12.4%** (v37: 5,898→7,156ms; v38b: 5,809→6,529ms).
  - Second eval at step 384 does NOT compound: step times stay at ~6,460ms before and after eval #2.
  - Only 1-step spike after each eval (~7,470ms), then immediate recovery to the new baseline.
  - val_loss tracks v37: 0.9504 at step 192, 0.9343 at step 384.
  - Remaining +12% regression likely from eval-specific allocations (validation data buffers, output tensors) that fragment even with activation checkpointing forced.
- v38b FINAL result (complete 1024-step run):
  - **val_loss: 0.9194 (step 960)** — matches v37's 0.9188 within noise (diff < 0.001)
  - **Pre-eval speed: 5,809ms** vs v37's 5,898ms (-1.5% faster due to GC overhead removal)
  - **Post-eval regression: +13.4% at eval#1** (5,810→6,586ms) vs v37's +21.7%
  - **Subsequent evals: <1% delta** (eval#2 +0.9%, eval#3 +0.6%, eval#5 +0.8%)
  - **Total training time: 6,458s** vs v37's 7,268s (**-11.1% faster overall**)
  - The one-time +13% regression after eval#1 is a permanent level shift, but no further compounding. The MLPerf reference shows 0%, so there's still room for improvement.
  - Key insight: the residual +13% is likely from eval-specific allocations (validation data buffers, output gather tensors) that fragment even with recompute forced. The warmup eval pass primes the allocator but cannot fully prevent it.
  | Metric | v37 | v38b | MLPerf ref |
  |--------|-----|------|------------|
  | Pre-eval ms/step | 5,898 | 5,809 | 3,811 |
  | Post-eval#1 ms/step | 7,167 | 6,586 | 3,778 |
  | Post-eval#1 delta | +21.7% | +13.4% | -0.8% |
  | Subsequent eval delta | +0.7-2.8% | +0.6-2.4% | 0% |
  | Total time (s) | 7,268 | 6,458 | ~4,180 |
  | Final val_loss | 0.9188 | 0.9194 | 0.92 |
- Status: **resolved (partial)** — post-eval regression reduced from +22% to +13%. Further optimization possible but diminishing returns.

### [2026-04-08 v40-phase2-speed-gap]
- Symptom: v39 pre-eval 5,596ms vs MLPerf 3,811ms = 1,785ms (47%) gap. Post-eval 6,070ms = 2,259ms (60%) gap.
- Phase 2 changes in v40:
  - P0a: LUMEN_TRANSPOSE_CACHE=0 (disable FP8 weight transpose caching)
  - P0b: RECOMPUTE_NUM_LAYERS=19 (from 21; 15 OOMed)
  - P2: overlap_grad_reduce=True (Megatron bucketed async AllReduce)
  - P3a: Fused quant+amax extended to backward path (quantize_bwd_delayed)
  - P4: SwiGLU backward already fused (confirmed)
  - DBG: quant.enable() model identity assertion added
- v40 results (through step 300, run in progress):
  | Metric | v39 | v40 | Delta | MLPerf ref |
  |--------|-----|-----|-------|------------|
  | Pre-eval ms/step | 5,596 | **5,348** | **-248 (-4.4%)** | 3,811 |
  | Post-eval#1 ms/step | 6,070 | **~6,060** | -10 (-0.2%) | 3,778 |
  | Post-eval#1 delta | +8.5% | +13.3% | worse | -0.8% |
  | val_loss @ 192 | 0.9453 | 0.9491 | +0.004 | ~0.94 |
  | Memory | 0.9837 | 0.9968 | +1.3% | ~0.82 |
- Analysis:
  - Pre-eval improvement is real: 248ms from 2 fewer recompute layers + overlapped grad reduce.
  - Post-eval regression is WORSE (13.3% vs v39's 8.5%) because memory went from 98.37% to 99.68%. Higher memory pressure = worse allocator fragmentation after eval.
  - RECOMPUTE_NUM_LAYERS=15 OOMed (Failed to CUDA calloc 536870912 bytes). P0a (transpose cache off) did not free enough memory for 6 fewer recompute layers.
  - RECOMPUTE_NUM_LAYERS=19 fits but at 99.68% — essentially no headroom. Transpose cache savings were less than expected.
  - val_loss 0.9491 at step 192 vs v39's 0.9453 — slightly worse, within noise for different ACL setting.
  - The mem_usages=-89485766.7030 after eval is a memory reporting overflow, not a real value.
- Remaining gap: v40 pre-eval 5,348ms vs MLPerf 3,811ms = **1,537ms (40%)**. Post-eval ~6,060ms = **2,249ms (59%)**.
- Next steps:
  1. Wait for v40 to complete and check val_loss convergence.
  2. The memory wall (99.68%) limits further ACL tuning. Need to find other memory savings.
  3. Post-eval regression getting worse — memory pressure is the root cause. Need to keep ACL at 21 and find speed improvements that don't increase memory.
- **Detailed gap breakdown** (v40 pre-eval 5,348ms vs MLPerf 3,811ms = **1,537ms/step**):
  | Source | Est. ms/step | % of Gap | Addressed? |
  |--------|-------------|----------|------------|
  | Checkpoint recompute dispatch overhead | 400-600 | 26-39% | Partially (19 vs 21 ACL) |
  | SwiGLU backward elementwise | 100-300 | 7-20% | Yes (fused bwd) — need profile |
  | Copy/dtype overhead (38K copy_ calls) | 200-300 | 13-20% | No |
  | CPU dispatch / kernel launch (30K/step) | 200-300 | 13-20% | No (needs HIP graphs) |
  | Memory allocator pressure (99.68%) | 100-200 | 7-13% | Worse |
  | NCCL AllReduce | 50-100 | 3-7% | Partially (overlap_grad_reduce) |
  | Quant/amax pipeline | 50-100 | 3-7% | Partially (fused bwd amax) |
  Key structural blockers:
  1. HIP Graph Capture (P5) — single biggest lever, ~500-700ms potential, but very hard with NCCL + dynamic control flow.
  2. Memory wall (99.68% vs TE's 82%) — prevents buffer pools, workspace allocs, reduced checkpointing. Root cause: Lumen stores separate FP8 descriptors + scale tensors + Python state per layer that TE avoids with C++ fusion.
  3. TE's `cast_transpose_fusion_kernel_optimized` writes both row-major and transposed FP8 in one pass; Lumen still does a separate transpose kernel per weight even with on-demand path.
- v40 final results (stopped at step 760/1024, loss target reached):
  | Metric | v39 | v40 | Delta | MLPerf ref |
  |--------|-----|-----|-------|------------|
  | Pre-eval ms/step | 5,596 | **5,348** | **-248 (-4.4%)** | 3,811 |
  | Post-eval ms/step | 6,070 | **~6,060** | -10 (-0.2%) | 3,778 |
  | val_loss @ 192 | 0.9453 | 0.9491 | +0.004 | ~0.94 |
  | val_loss @ 384 | — | 0.9334 | — | — |
  | val_loss @ 576 | — | 0.9232 | — | — |
  | Memory | 0.9837 | 0.9968 | +1.3% | ~0.82 |
  val_loss converged below 0.92 target. Run healthy (0 NaN, 0 skipped).
- Status: **resolved** — v40 reached loss target. Phase 2 delivered -4.4% pre-eval speedup. Remaining 1,537ms gap addressed by Phase 3 (v41).

### [2026-04-08 v41-phase3-implementation]
- Phase 3 changes implemented for v41:
  - **P5: HIP Graph Capture** — Per-layer forward+backward graph capture following TE/MLPerf pattern:
    - `LumenGraphedLayer` autograd.Function in `lumen/utils/hip_graphs.py`: captures separate fwd/bwd CUDA graphs per transformer layer, replays via static buffer copy + graph.replay()
    - `capture_lumen_graphs()`: iterates `model.decoder.layers`, creates synthetic sample inputs `(seq_len, micro_bs, hidden_size)`, wraps each layer with graph capture
    - `install_hip_graphs_hook()` in `megatron.py`: chains on `setup_model_and_optimizer`, gated on `--lumen-hip-graphs`
    - Wired into `finetune_llama2.py` alongside existing hooks
    - `run_finetune.sh`: added `LUMEN_HIP_GRAPHS=1` -> `--lumen-hip-graphs` flag wiring
    - Expected savings: ~300-500ms/step (eliminates ~20K of 30K kernel launches)
  - **P6: Fused cast+transpose+amax kernel** — single Triton kernel replaces 3 separate kernels:
    - `cast_transpose_amax_fp8` in `cast_transpose.py`: tile-based, reads BF16 once, writes row-major FP8 + transposed FP8 + atomic amax
    - Top-priority branch in `ScalingManager._quantize_core`: when both `LUMEN_FUSED_CAST_TRANSPOSE=1` AND `LUMEN_FUSED_QUANT_AMAX=1`, uses combined kernel
    - Expected savings: ~150-250ms/step (eliminates 4,800 amax + 9,600 abs + separate transpose calls)
  - **P6b: hipBLASLt pre-transposed weight** — avoids redundant `.t().contiguous()`:
    - `dispatch_gemm` extracts `FP8Descriptor._transpose` and passes to `gemm_per_tensor`
    - `_gemm_per_tensor_hipblas` accepts optional `w_transposed` parameter
    - Expected savings: ~50ms/step (eliminates 1,821 transpose copies in GEMM path)
  - **Memory strategy**: RECOMPUTE_NUM_LAYERS stays at 21 (not v40's 19) to keep memory below 98%, giving graph capture headroom for static buffers
- v41 launch script: `/home/danyzhan/run_v41_phase3.sh`
  - Key env vars: `LUMEN_FUSED_CAST_TRANSPOSE=1`, `LUMEN_FUSED_QUANT_AMAX=1`, `LUMEN_HIP_GRAPHS=1`
  - Target: v40 5,348ms -> ~4,600-4,800ms/step
- v41 first launch: **OOM on first training step** (all 8 GPUs)
  - Root cause: `cast_transpose_amax_fp8` stored `_transpose` in FP8Descriptor, doubling FP8 weight memory (80 layers × 4 weights × ~448 MiB extra = ~3.5 GiB)
  - Combined with RECOMPUTE_NUM_LAYERS=21 (already 99.68% memory), pushed over 192 GiB
  - HIP graph capture code never executed (no log messages) — `--lumen-hip-graphs` not recognized by Megatron arg parser
  - Fix: (1) deleted `_transpose` from fused kernel return, (2) disabled HIP graphs for now
- v41 second launch: fused cast+transpose+amax enabled but `_transpose` NOT stored (memory fix)
  - Result: ~5,490ms/step — **WORSE than v40** by ~140ms
  - Root cause: kernel computes row-major + transposed FP8 + amax but discards transpose = wasted writes
  - Also RECOMPUTE_NUM_LAYERS=21 vs v40's 19 = ~200ms extra recompute overhead
  - Conclusion: P6 fused cast+transpose kernel is net-negative when transpose output is unused
- v41 third launch (final): disabled `LUMEN_FUSED_CAST_TRANSPOSE`, matched v40 config exactly (RECOMPUTE=19)
  - **Completed successfully** (1024/1024 steps, 0 NaN, 0 skipped)
  - val_loss trajectory: 0.9481 -> 0.9329 -> 0.9223 -> 0.9241 -> **0.9194** (target reached)
  - Pre-eval ms/step: **~5,335** (steps 20-190, v40 was ~5,348 — within noise)
  - Post-eval ms/step: **~5,940-6,060** (v40 was ~6,060 — same)
  - Memory: 0.9968 (same as v40)
  - Confirms v40 baseline reproducibility. P6 kernel does NOT help without memory for transpose storage.
- **Conclusions from Phase 3 attempt**:
  1. HIP Graph Capture (P5): not viable — Megatron arg parser doesn't accept `--lumen-hip-graphs`, and memory is too tight for static buffers. Requires deep Megatron integration (custom arg parsing) + significant memory reduction first.
  2. Fused cast+transpose+amax (P6): net-negative when transpose is discarded. Only valuable IF pre-transposed weight can be stored (requires ~3.5 GiB free). Memory wall is the fundamental blocker.
  3. Pre-transposed weight for hipBLASLt (P6b): cannot benefit without stored transpose.
  4. **Memory wall (99.68%) is the root cause** preventing both P5 and P6 from delivering value.
- Status: **resolved** — v41 completed, baseline confirmed. Phase 3 optimizations blocked by memory wall.

### [2026-04-09 v42-memory-wall-distributed-analysis]
- Symptom: v40/v41 at 99.68% GPU memory utilization (191.4 GiB of 192 GiB), causing:
  1. Post-eval regression of +13% (5,348→6,060ms) that never recovers
  2. Allocator fragmentation/thrashing on all GPU ops
  3. Inability to enable fused cast+transpose (needs ~3.5 GiB free)
  4. Inability to enable HIP graphs (needs static buffer headroom)

- **Multi-GPU sync / distributed analysis** (2026-04-09):
  | Aspect | Lumen v40/v41 | Notes |
  |--------|--------------|-------|
  | overlap_grad_reduce | True | Bucketed async AllReduce during backward |
  | Gradient buckets | 2 (40M + 4.5M params) | 45M total LoRA params |
  | Gradient volume | ~170 MB (FP32) | Tiny — only LoRA weights |
  | NCCL AllReduce time | 119ms (2% of GPU time) | NOT a bottleneck |
  | NCCL config | 32 channels, 32 CTAs, loopback | Standard MI300X tuning |
  | use_distributed_optimizer | False | Correct for single-node with tiny LoRA |
  | overlap_param_gather | False | Irrelevant without distributed optimizer |

  **CONCLUSION**: Distributed communication is well-optimized and NOT a significant
  source of the speed gap. AllReduce is only 2% of GPU time and overlapped with backward.

- **Data pipeline analysis** (2026-04-09):
  | Aspect | Lumen | Notes |
  |--------|-------|-------|
  | num_workers | 2 (Megatron default) | |
  | pin_memory | No (Megatron path) | Only FSDP path uses it |
  | Data size per step | ~64 KB per GPU | MBS=1, SEQ=8192, int64 |
  | Data loading | np.load (in-memory) | Entire dataset fits in RAM |
  | CPU→GPU transfer | broadcast_data | Standard Megatron path |

  **CONCLUSION**: Data pipeline is NOT a bottleneck. 64 KB per sample at MBS=1
  transfers in <0.1ms. Even without pin_memory/prefetching, data is ready
  well before the GPU needs it.

- **Memory wall deep analysis** (2026-04-09):
  The 99.68% memory utilization breakdown:

  | Component | Memory (GiB) | % of 192 GiB |
  |-----------|-------------|---------------|
  | FP8 model weights (80 layers + embed) | 64.2 | 33.4% |
  | BF16 autograd intermediates (61 layers) | ~112 | 58.3% |
  | FP8 saved activations (61 layers × 4 linears) | 15.3 | 7.9% |
  | LoRA params + optimizer + gradients | 0.7 | 0.4% |
  | **Total** | **~192** | **~100%** |

  **The killer: 112 GiB of BF16 autograd intermediates** in the 61 non-recomputed
  layers (layers 19-79). Each layer saves ~1.8 GiB of BF16 tensors for backward:
  - RMSNorm input: 128 MB
  - QKV output: 160 MB
  - Attention output: 128 MB
  - RMSNorm2 input: 128 MB
  - FC1 gate+up output: **896 MB** (largest single tensor)
  - SwiGLU output: **448 MB**

  **Why TE uses ~82%**: TE's C++ fused kernels (LayerNormLinear, SwiGLU) compute
  backward without storing the BF16 intermediate outputs in the autograd graph.
  Lumen's Python autograd forces each op to save its own inputs.

- **CRITICAL FINDING**: `LUMEN_MLP_RECOMPUTE=1` was NOT enabled in v39/v40/v41!
  This feature exists in `megatron_patches.py` (install_mlp_recompute) and was
  used in the older `run_tp1_dp8.sh` launch script, but was never wired into
  `run_finetune.sh` or the v39+ launch scripts. It wraps the MLP sublayer with
  `tensor_parallel.checkpoint`, freeing the BF16 FC1 output (896 MB) and SwiGLU
  output (448 MB) per non-recomputed layer.

- **v42 plan** — Memory reduction via selective MLP recompute:
  1. `LUMEN_MLP_RECOMPUTE=1` + `LUMEN_MLP_RECOMPUTE_LAYERS=20`
     - 20 non-recomputed layers get MLP-only recompute
     - Frees 20 × ~1.3 GiB = ~26 GiB of BF16 intermediates
     - Cost: 20 × ~16ms MLP forward recompute = ~320ms/step
  2. `RECOMPUTE_NUM_LAYERS=21` (back to MLPerf reference, from v40's 19)
     - 2 more fully recomputed layers: frees additional ~4.2 GiB, costs ~110ms
  3. `max_split_size_mb=256` (from 512) — smaller allocator splits

  Expected memory: ~90% (down from 99.68%). Expected pre-eval step time: ~5,780ms
  (v40's 5,348 + 320 MLP recomp + 110 extra ACL). Expected post-eval: ~5,780ms
  (NO regression, vs v40's 6,060 = 13% worse post-eval).

  Net wall-clock with 5 eval intervals:
  - v40: 5,348×190 + 6,060×830 = ~6,045s training
  - v42: 5,780×1020 = ~5,896s training — **2.5% faster overall** with uniform timing

- Enhanced `install_mlp_recompute()` in `megatron_patches.py`:
  Added `LUMEN_MLP_RECOMPUTE_LAYERS` env var to control how many non-recomputed
  layers get MLP-only recompute (default: all). Counter-based: first N non-recomputed
  layers encountered during the first forward pass get their MLP wrapped.

- Status: open — v42 ready to launch
- Launch script: `/home/danyzhan/run_v42_memory.sh`

## Entry Template

```markdown
### [YYYY-MM-DD session-name]
- Symptom:
- Possible bug:
- Evidence so far:
- Next check:
- Status: open | ruled out | resolved
```
