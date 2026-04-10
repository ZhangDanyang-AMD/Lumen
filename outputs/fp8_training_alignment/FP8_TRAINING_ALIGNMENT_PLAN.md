# FP8 Training Alignment — Lumen Reproduction Plan

**Date**: 2026-04-10
**Reference**: [VERL FP8 RL documentation](https://github.com/verl-project/verl/blob/main/docs/advance/fp8.md)
**Goal**: Reproduce VERL's FP8 RL benchmarks using Lumen on MI350X, demonstrating that FP8 training aligns with BF16 training when both use FP8 rollout.

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
| DAPO-Math-17k (train) | `/home/danyzhan/Lumen/data/dapo-math-17k.parquet` | 286 MB |
| AIME-2024 (val) | `/home/danyzhan/Lumen/data/aime-2024.parquet` | 29 KB |

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

## Lumen FP8 Method vs VERL Reference

| Aspect | VERL Reference | Lumen Reproduction |
|--------|---------------|-------------------|
| FP8 Rollout | vLLM FP8 monkey-patch (`quantization=fp8`) | Same — vLLM FP8 |
| FP8 Training (E2E) | Transformer Engine blockwise FP8 GEMM | Lumen `FP8ParamManager` on-the-fly quantization |
| Rollout Correction | Token-level TIS, C=2 | Same — `rollout_is=token`, threshold=2.0 |
| Training Backend | FSDP / Megatron | FSDP2 (primary), Megatron (experiment 3 option) |

**Key difference**: Lumen's FP8ParamManager quantizes weights to FP8 in the autograd graph (halving saved tensor memory) but keeps the actual GEMM in BF16 (dequant on-the-fly). TE's approach does the GEMM in FP8 directly. Lumen saves memory; TE saves both memory and compute.

---

## Experiment Matrix

### Design Rationale

All variants within an experiment use the **same model** so results are directly comparable on a single chart. Experiment 3 (separate 30B baseline) is eliminated — 2A serves as the shared BF16 baseline for both FP8 rollout and FP8 E2E on the 30B MoE model.

### Experiment 1: Qwen3-8B-Base Dense — FP8 Rollout + FP8 E2E

**Reference**: [Qwen3-8B-Base Dense Model](https://github.com/verl-project/verl/blob/main/docs/advance/fp8.md#qwen3-8b-base-dense-model)

| Run ID | Training | Rollout | TIS | Script |
|--------|----------|---------|-----|--------|
| 1A | BF16 (FSDP2) | BF16 | — | `run_dapo_qwen3_8b_bf16.sh` |
| 1B | BF16 (FSDP2) | FP8 + TIS | token, C=2 | `run_dapo_qwen3_8b_fp8_rollout_tis.sh` |
| 1C | BF16 (FSDP2) | FP8 | — | `run_dapo_qwen3_8b_fp8_rollout_no_tis.sh` |
| **1D** | **FP8 (Lumen FP8PM)** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_8b_fp8_e2e_lumen.sh`** |

**Comparison chart**: 1A vs 1B vs 1C vs 1D on the same axes.
- 1A = pure BF16 baseline
- 1B = FP8 rollout with TIS (should align with 1A)
- 1C = FP8 rollout without TIS (ablation: expected accuracy drop)
- 1D = Lumen FP8 E2E (should align with 1A/1B if FP8PM doesn't degrade training)

**Config** (adapted for 8x MI350):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen/Qwen3-8B-Base | Dense, 8B params |
| GPUs | 8 | Same as reference |
| Prompt batch size | 32 | Same as reference |
| Responses per prompt (n) | 16 | Same as reference |
| Train batch size | 32 | Same as reference |
| PPO mini batch size | 32 | Same as reference |
| Max prompt length | 1024 | Same as reference |
| Max response length | 20480 (20K) | Same as reference |
| LR | 1e-6 | Same as reference |
| Clip (low/high) | 0.2 / 0.28 | DAPO decoupled clip |
| Loss aggregation | token-mean | DAPO token-level loss |
| Reward manager | dapo | Math accuracy scoring |
| Overlong buffer | enable=True, len=512, penalty=1.0 | Same as reference |
| Total steps | 500 | Same as reference |
| Val frequency | 5 steps | Same as reference |
| FSDP2 offload | param + optimizer | Memory savings |
| vLLM gpu_memory_util | 0.9 | Same as reference |
| Rollout TP | 1 | ROCm vLLM TP=1 required |

**Expected outcome**: 1A ≈ 1B ≈ 1D (aligned curves). 1C shows accuracy drop (no TIS).

---

### Experiment 2: Qwen3-30B-A3B-Base MoE — FP8 Rollout + FP8 E2E (Unified)

**Reference**: [Qwen3-30B-A3B-Base MoE](https://github.com/verl-project/verl/blob/main/docs/advance/fp8.md#qwen3-30b-a3b-base-moe-model) + [Qwen3-30B-A3B E2E](https://github.com/verl-project/verl/blob/main/docs/advance/fp8.md#qwen3-30b-a3b-moe-model)

| Run ID | Training | Rollout | TIS | Script |
|--------|----------|---------|-----|--------|
| 2A | BF16 (FSDP2) | BF16 + TIS | token, C=2 | `run_dapo_qwen3_30b_moe_bf16_tis.sh` |
| 2B | BF16 (FSDP2) | FP8 + TIS | token, C=2 | `run_dapo_qwen3_30b_moe_fp8_rollout_tis.sh` |
| **2C** | **FP8 (Lumen FP8PM)** | **FP8 + TIS** | **token, C=2** | **`run_dapo_qwen3_30b_moe_fp8_e2e_lumen.sh`** |

**Comparison chart**: 2A vs 2B vs 2C on the same axes.
- 2A = BF16 baseline (with TIS, needed for MoE rollout correction even in BF16)
- 2B = FP8 rollout only (should align with 2A)
- 2C = Lumen FP8 E2E (should align with 2A/2B)

**Config adaptation for 8 GPUs** (reference used 2×8=16):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen/Qwen3-30B-A3B-Base | MoE: 128 experts, 8 active |
| GPUs | 8 | Half of reference (16) |
| Prompt batch size | 16 | Halved (reference: 32) |
| Responses per prompt (n) | 16 | Same |
| Train batch size | 16 | Halved |
| PPO mini batch size | 16 | Halved |
| Max response length | 20480 | Same |
| FSDP2 offload | param + optimizer | Essential for memory |
| SP size | 4 | Ulysses sequence parallelism |
| vLLM gpu_memory_util | 0.5 | Lower for MoE memory |
| Rollout TP | 1 | ROCm constraint |
| Total steps | 500 | Same |

**Lumen FP8 E2E config** (for run 2C):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `FP8_PARAM_MANAGER` | 1 | Enables Lumen on-the-fly FP8 quantization |
| Entry point | `lumen.rl.verl.verl_entry` | Patches FSDP2 worker with LumenConfig.enable() |
| Training precision | BF16 GEMM, FP8 in autograd graph | Not TE FP8 GEMM |

**Expected outcome**: 2A ≈ 2B ≈ 2C (aligned curves). Expected mismatch KL ordering: 2B > 2C > 2A.
Higher overall mismatch KL for MoE vs dense (matching reference observation).

---

## Metrics to Track

For each run, log and compare:

| Metric | Description | WandB key (expected) |
|--------|-------------|---------------------|
| val_score | AIME-2024 accuracy | `val/score` or `test/score` |
| critic/rewards | Reward from DAPO scoring | `reward/mean` or `critic/rewards` |
| mismatch | Rollout/training distribution KL | `rollout_correction/mismatch_kl` |
| response_length | Average generation length | `response_length/mean` |

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
- Run 1A (8B BF16 baseline), 2A (30B MoE BF16 + TIS baseline)
- Verify training converges, metrics are logged correctly
- Confirm curves match expected DAPO behavior (improving val_score, increasing response_length)

### Step 4: FP8 tests
- Run 1B, 1C (FP8 rollout ± TIS for 8B)
- Run 1D (Lumen FP8 E2E for 8B)
- Run 2B (FP8 rollout + TIS for 30B MoE)
- Run 2C (Lumen FP8 E2E for 30B MoE)

### Step 5: Compare
- Experiment 1 chart: overlay 1A vs 1B vs 1C vs 1D
- Experiment 2 chart: overlay 2A vs 2B vs 2C
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

---

## File Layout

```
Lumen/
├── data/
│   ├── dapo-math-17k.parquet        # Training data
│   └── aime-2024.parquet            # Validation data
├── examples/rl/verl/dapo/
│   ├── common.sh                                        # Shared DAPO config
│   ├── smoke_test.sh                                    # 2-step validation
│   ├── run_dapo_qwen3_8b_bf16.sh                       # Exp 1A
│   ├── run_dapo_qwen3_8b_fp8_rollout_tis.sh            # Exp 1B
│   ├── run_dapo_qwen3_8b_fp8_rollout_no_tis.sh         # Exp 1C
│   ├── run_dapo_qwen3_8b_fp8_e2e_lumen.sh              # Exp 1D (FP8 E2E)
│   ├── run_dapo_qwen3_30b_moe_bf16_tis.sh              # Exp 2A
│   ├── run_dapo_qwen3_30b_moe_fp8_rollout_tis.sh       # Exp 2B
│   └── run_dapo_qwen3_30b_moe_fp8_e2e_lumen.sh         # Exp 2C (FP8 E2E)
└── outputs/fp8_training_alignment/
    ├── FP8_TRAINING_ALIGNMENT_PLAN.md                   # This document
    └── FP8_TRAINING_ALIGNMENT_RESULTS.md                # Final comparison (after runs)
```
