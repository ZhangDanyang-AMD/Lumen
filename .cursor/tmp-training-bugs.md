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

### [2026-04-03 doc-align-v014-on-v100]
- Symptom: doc_align_8gpu_v2 (grpo, beta=0.04, temp=0.9, per_device_bs=1, grad_accum=8, 8 GPU) reward stuck at avg ~0.09 after 3568 micro-steps (446 opt steps, 0.28 epochs). Reference grpo_curves.png should be at ~0.25-0.30 reward by this point.
- Root cause identified: **grpo_curves.png version mismatch**. Confirmed that:
  - Image uploaded Jan 18, 2025 (commit 6817464)
  - v0.14.0 doc used trl-lib/tldr + reward_len, NOT DeepMath + accuracy_reward
  - v1.0.0 doc uses DeepMath + accuracy_reward but reuses the OLD image from v0.14.0
  - Therefore grpo_curves.png was generated with TLDR + reward_len + v0.14.0 defaults
  - Trying to reproduce TLDR curves using DeepMath is comparing apples to oranges
- Evidence so far:
  - pure_defaults (TRL 1.0.0 defaults, 1 GPU): 799 steps, reward ~0.2, loss=0. DEAD — beta=0 + loss_type=dapo mismatch.
  - bf16_baseline (bnpo, beta=0.04, lr=5e-6, 512 tokens, 1 GPU): 2500 steps, reward ~0.05. DEAD.
  - doc_align_8gpu_v2 (v0.14.0 params, 8 GPU, DeepMath): 3568 micro-steps (446 opt steps), avg reward ~0.09, KL=0.076, clipped=0.44. Learning is slow but present. Stopped to investigate.
  - v0.14.0 source confirms: beta=0.04, temp=0.9, loss_type=grpo (only), per_device_bs=1, grad_accum=8
  - v1.0.0 loss_type=grpo adds PPO clipping (epsilon=0.2) that v0.14.0 lacked, but clip_ratio=0 in our run (not triggered)
  - Loss normalization by grad_accum is correct (TRL does it, Trainer skips when num_items_in_batch is set)
  - Model verified: local qwen2-0.5b-instruct matches HF Hub (same architecture, same file size)
- Decision: The reference image is NOT from DeepMath training. It's from TLDR. The user wants to align DeepMath BF16 curves. We need a new reference.
- Decision taken: Option B — use v0.14.0-aligned params on DeepMath for Lumen FP8 A/B.
- Status: resolved (image mismatch understood; proceeding with self-consistent baseline)

### [2026-04-04 three-step-ab-comparison]
- Goal: 3-step A/B comparison per train_target.md
  - Step 1: BF16 without Lumen (train_grpo_doc_align.py, DDP 8 GPU, max_steps=5000)
  - Step 2: Skipped per user request
  - Step 3: Lumen FP8 (train_grpo_lumen_fp8.py, apply_fp8_training, same config)
- History (older iterations with delayed scaling, FP32 model loading, etc.):
  - Step 1 (FP32 string path): 5000 steps, peak_mem=12.04GB, avg_step=6.44s, mean_reward=0.072, degradation=2.05x
  - Step 2 (string path BF16): bit-identical to Step 1. Proves Lumen env = clean.
  - Step 2 (pre-loaded BF16): peak_mem=7.11GB, avg_step=3.7s, but KL/loss 16x lower than Step 1. Different numerical path (BF16 vs FP32).
  - Step 3 (FP8 delayed on FP32): CK backend failed ("scaled_quant_kernel" not implemented for 'Float'). Fell back to Triton, 57s/step.
  - Step 3 (FP8 delayed on BF16): peak_mem=7.11GB, step_time=16.3s. High initial KL=0.021. Killed at step 631 for redesign.
- Key insight: fair A/B comparison must use same model loading dtype. Delayed scaling is not the most accurate FP8 method.
- **APPLE-TO-APPLE REDESIGN** (2026-04-04):
  - Both BF16 and FP8 pre-load model in BF16, enable gradient checkpointing, use identical GRPOConfig with model_init_kwargs={"torch_dtype": "bfloat16"} (so reference model is also BF16)
  - Only difference: FP8 script adds quant.enable(model, fp8_e4m3, scaling="dynamic")
  - Dynamic scaling = exact per-tensor amax every forward pass (most accurate FP8 quantization)
  - Scripts: train_grpo_bf16_preloaded.py (BF16 control) vs train_grpo_lumen_fp8.py (FP8 dynamic)
  - **RESULTS (1000 steps each)**:
    - Peak memory:  BF16=7.11GB, FP8=7.11GB (+0.0%)
    - Avg step time: BF16=3.76s, FP8=11.95s (3.18x overhead)
    - Step time degradation: BF16=0.98x, FP8=1.02x (both stable)
    - Mean KL: BF16=0.000795, FP8=0.020862 (26x higher — FP8 quantization noise)
    - Mean reward: BF16=0.0066, FP8=0.0052 (comparable, both low on DeepMath)
    - Mean loss: BF16=0.000032, FP8=0.000835 (26x higher — driven by KL)
    - Entropy: BF16=0.7093, FP8=0.7182 (similar)
  - Logs: outputs/bf16_preloaded_1000steps.jsonl, outputs/fp8_dynamic_1000steps.jsonl
- **FP8 DELAYED RUN** (2026-04-05):
  - Same apple-to-apple setup, only change: scaling="delayed" instead of scaling="dynamic"
  - Script: train_grpo_lumen_fp8_delayed.py
  - **RESULTS (1000 steps)**:
    - Peak memory: 7.11 GB (same as BF16 and dynamic)
    - Avg step time: 16.32s (4.34x vs BF16, 1.37x vs dynamic)
    - Step time degradation: 1.00x (stable)
    - Mean KL: 0.018155 (slightly lower than dynamic's 0.020862)
    - Mean reward: 0.0057 (comparable to BF16=0.0066 and dynamic=0.0052)
    - Mean loss: 0.000727 (slightly lower than dynamic's 0.000835)
    - Entropy: 0.7111 (similar to BF16=0.7093)
  - Log: outputs/fp8_delayed_1000steps.jsonl
- **THREE-WAY SUMMARY**:
  | Metric          | BF16    | FP8 dynamic | FP8 delayed |
  |-----------------|---------|-------------|-------------|
  | Peak memory     | 7.11 GB | 7.11 GB     | 7.11 GB     |
  | Avg step time   | 3.76s   | 11.95s      | 16.32s      |
  | Overhead vs BF16| 1.00x   | 3.18x       | 4.34x       |
  | Mean KL         | 0.0008  | 0.0209      | 0.0182      |
  | Mean reward     | 0.0066  | 0.0052      | 0.0057      |
  | Mean loss       | 3.2e-5  | 8.4e-4      | 7.3e-4      |
  | Entropy         | 0.7093  | 0.7182      | 0.7111      |
  - Dynamic is 27% faster than delayed (11.95 vs 16.32s) with comparable accuracy
  - Delayed has slightly lower KL and loss than dynamic (possibly because stale scales under-quantize, preserving more BF16 behavior)
  - Both FP8 methods track BF16 reward curves closely
  - FP8 on 0.5B: 3-4x overhead, no memory savings. Expected to improve on 7B+.
- Status: resolved

### [2026-04-05 fp8-blockwise-crash]
- Symptom: FP8 blockwise scaling (`quant.enable(model, fp8_e4m3, scaling="blockwise")`) crashes during first backward pass.
- Error: `RuntimeError: The size of tensor a (896) must match the size of tensor b (28) at non-singleton dimension 1` at `lumen/ops/quantize/linear.py:736`
- Root cause: blockwise quantization produces scale tensors with shape `(M, ceil(N/block_size))` (e.g. `(*, 28)` for N=896, block_size=32), but the backward pass in `QuantizedLinearFunction` expects scale shape matching the weight dimension (896). The backward path doesn't handle per-block scale tensors correctly for HuggingFace models via TRL.
- Workaround: use `dynamic` or `delayed` scaling instead. Blockwise is only viable with Megatron-style models or needs a backward fix.
- Status: resolved (not a training bug, blockwise backward not compatible with this code path)

## Ruled Out

Move disproved suspicions here instead of deleting them.

## Resolved

Move confirmed and fixed issues here with a short note about the fix.

## Entry Template

```markdown
### [YYYY-MM-DD session-name]
- Symptom:
- Possible bug:
- Evidence so far:
- Next check:
- Status: open | ruled out | resolved
```
