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

(none)

## Ruled Out

Move disproved suspicions here instead of deleting them.

## Resolved

### [2026-04-08 fp8pm-multi-gpu-integration]
- Symptom: FP8ParamManager works on single GPU but needs multi-GPU support.
- Fixes applied:
  1. FSDP1/FSDP2 incompatible — switched to DDP (MULTI_GPU)
  2. Device mismatch in dequant hook: scale tensor stayed on CPU after model.cuda(); fixed by using `weight.data` and `scale.to(device)`
  3. DDP unused-parameter error: set `requires_grad=False` on FP8 params
- Results (8-GPU DDP, Llama-3.1-8B, 20 steps):
  | Config | Peak Mem/GPU | vs BF16 DDP |
  |--------|-------------|-------------|
  | BF16 DDP | 80.5 GB | baseline |
  | FP8ParamManager DDP | 31.4 GB | **-61%** |
  | FP8PM + 8-bit Adam | 30.4 GB | **-62%** |
- 70B result: FP8PM + 8-bit Adam = 210.8 GB (fits MI300X 256 GB), BF16 DDP = OOM
- Limitation: FP8ParamManager freezes nn.Linear weights (only ~1.3% params trainable)
- Status: resolved

### [2026-04-08 fp8-memory-savings-experiment]
- Symptom: Need to prove FP8 training saves memory. Previous runs on wrong branch showed no savings.
- Reference: BF16 baseline single-GPU Llama-3.1-8B, AdamW, gradient checkpointing, SDPA, bs=2, seq=256.
- Fix: Synced dev/RL to Docker, fixed FP8ParamManager to skip nn.Embedding (was crashing on bias attr), synced descriptor.py.
- Results (Llama-3.1-8B, 3 steps, single MI300X GPU, **process-isolated** — each config in its own python process starting from 0 MB):
  | Config                 | Peak Alloc (MB) | Peak Res (MB) | Steady-State (MB) | vs BF16   |
  |------------------------|-----------------|---------------|-------------------|-----------|
  | BF16 baseline (AdamW)  | 76,861          | 78,728        | 46,228            | baseline  |
  | FP8ParamManager        | 28,909          | 32,178        | 24,757            | **-62.4%** peak, **-46.5%** steady |
  | FP8Param + 8-bit Adam  | 27,923          | 30,172        | 23,769            | **-63.7%** peak, **-48.6%** steady |
  | FP8 Attention (dpa)    | 76,860          | 77,414        | 46,227            | -0.0% peak |
- Detailed memory breakdown (process-isolated):
  | Component              | BF16 Baseline | FP8ParamManager | FP8+8bit Adam | FP8 Attn  |
  |------------------------|---------------|-----------------|---------------|-----------|
  | Param storage          | 15,317 MB     | 8,160 MB        | 8,160 MB      | 15,317 MB |
  |   FP8 params           | 0 MB          | 7,157 MB        | 7,157 MB      | 0 MB      |
  |   BF16 params          | 15,317 MB     | 1,003 MB        | 1,003 MB      | 15,317 MB |
  | Optimizer states       | 30,633 MB     | 2,005 MB        | 1,018 MB      | 30,633 MB |
  |   state dtype          | bf16          | bf16            | uint8+fp32    | bf16      |
- Key findings:
  1. FP8ParamManager saves ~48 GB peak (62.4%). Savings come from BOTH weights AND optimizer states:
     - Weights: bf16->fp8 saves ~7.2 GB (15,317 -> 8,160 MB)
     - Optimizer states: AdamW creates exp_avg/exp_avg_sq matching param dtype. FP8 params get bf16 states sized to numel() but the numel() is the same — the saving is from dtype: the optimizer internally matches the param dtype (bf16 for the few remaining bf16 params only). The FP8 params have 1-byte elements so AdamW allocates bf16 states with the same numel but for FP8 params the optimizer states are much smaller because the parameter tensor's element_size is 1 byte.
     - **Correction**: AdamW allocates states based on param shape, but uses bf16 (not fp32) for all states in this PyTorch version. For FP8 params, AdamW only tracks non-FP8 params (lm_head, embedding, norms). The 225 quantized Linear params are FP8 and their gradients still flow, but optimizer states are allocated as bf16 tensors matching param numel — total only 2,005 MB vs 30,633 MB.
  2. Adding 8-bit Adam saves another ~1 GB (optimizer states 2,005->1,018 MB as uint8).
  3. FP8 Attention alone saves negligible peak memory with gradient checkpointing enabled.
- Status: resolved

| Config | Peak Mem/GPU | Elapsed | vs BF16 full |
|--------|-------------|---------|-------------|
| BF16 full (baseline) | 34.57 GB | 122.7s | baseline |
| FP8 Linear only | 38.85 GB | 1279.2s | +12% mem, 10.4x slower |
| FP8 Linear + FP8 Attn (dpa) | **30.92 GB** | 1586.5s | **-11% mem**, 12.9x slower |
| FP8 Linear + FP8 Attn + Act Store | **30.89 GB** | 1581.3s | **-11% mem**, 12.9x slower |
| FP8 Linear + FP8 Attn + Param Gather | CRASH | — | AITER quant kernel crash |
| FP8 Linear + FP8 Attn + Lumen Norm | CRASH | — | FSDP1 mixed dtype flatten error |

### [2026-04-08 fp8-architectural-fixes]
- Symptom: Three FP8 features marked "NOT FEASIBLE" in BENCHMARK_RESULTS.md
- Fix 1 (FP8 Weight Cache): Wired store_weights_fp8() into LumenConfig + VERL. Result: actor -2%, throughput +3.3% vs FP8-only.
- Fix 2 (FP8 Activation Store): Extended _apply_pre_quant to nn.Linear. Result: no measurable effect on full FP8 path.
- Fix 3 (AITER Kernel Crash): Added weight.contiguous() + TORCH_CHECK. Result: crash resolved, actor -1.7%, throughput +1.4%.
- Combined: actor -2.4%, throughput +2.8% vs FP8-only.
- Status: resolved

## Entry Template

```markdown
### [YYYY-MM-DD session-name]
- Symptom:
- Possible bug:
- Evidence so far:
- Next check:
- Status: open | ruled out | resolved
```
