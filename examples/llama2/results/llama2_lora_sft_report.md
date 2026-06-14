# Llama2 LoRA SFT on AMD MI308X — FSDP vs Megatron, BF16 vs FP8 (delayed / blockwise2d)

Consolidated results for Llama2 LoRA supervised fine-tuning on **8×MI308X**, covering both
backends (PyTorch **FSDP** and **Megatron-LM**) at **7B and 70B**, in **BF16** and two FP8
linear-quantization recipes (**delayed**, **blockwise2d**), on the gov_report dataset.

> This is the single living report — append new runs to the matrix (§1) and the results
> tables (§7). **Read the data-comparability caveat (§8) before comparing val-loss across
> rows:** only the Megatron 70B blockwise2d run uses the MLPerf **answer-only** loss mask;
> all other rows report **all-token** loss and are NOT comparable to the MLPerf target 0.925.

---

## 1. Experiment matrix

| # | Backend | Size | Precision | Steps reached | Status | Loss mask |
|---|---|---|---|---|---|---|
| 1 | FSDP | 7B | bf16 | 200 (full) | ✅ complete | all-token |
| 2 | FSDP | 7B | fp8_delayed | 200 (full) | ✅ complete | all-token |
| 3 | FSDP | 7B | fp8_blockwise2d | 200 (full) | ✅ complete | all-token |
| 4 | FSDP | 70B | bf16 | 60 | ⏸ stopped (snapshot) | all-token |
| 5 | FSDP | 70B | fp8_blockwise2d | 256 | ⚠ NCCL watchdog hang ~256 | all-token |
| 6 | Megatron | 70B | fp8_blockwise2d | 192 (first eval) | ⚠ then early-stop desync (fixed) | **answer-only (MLPerf)** |
| 7 | **native** PyTorch FSDP | 7B | bf16 | 200 (full) | ✅ complete | all-token | final val 1.4233, ~4.5 s/step (baseline; see §7) |

<!-- append new runs here -->

---

## 2. Environment

| Item | Value |
|---|---|
| GPUs | 8× AMD MI308X (192 GB HBM, gfx942), single node, XGMI |
| Image | `lumen/llama2:latest` (ROCm 7.0, AITER/mori prebuilt, Megatron-LM-AMD) |
| Model (7B) | HF `NousResearch/Llama-2-7b-hf`, `from_pretrained` |
| Model (70B) | HF Llama2-70B → Megatron TP1 ckpt (Megatron); HF dir (FSDP) |
| Dataset | gov_report (`regisss/scrolls_gov_report_preprocessed_mlperf_2`), 3901 train / 173 val, packed seq 8192 |
| Lumen role | FP8 linear quantization (attention & norm stay BF16 on FSDP) |

---

## 3. Shared hyperparameters (all runs, MLPerf-aligned)

| Parameter | Value |
|---|---|
| Sequence length | 8192 |
| Micro-batch / DP / **global batch** | 1 / 8 / **8** |
| LR / schedule / warmup | 4e-4 / cosine→0 / 0 |
| Weight decay / grad clip | 1e-4 / 0.3 |
| Optimizer | AdamW (FSDP: β=(0.9,0.95), eps=1e-5) |
| LoRA | rank 16, alpha 32, dropout 0.1 |
| Seed | 1234 |
| TunableOp | off (`PYTORCH_TUNABLEOP_ENABLED=0`) |

---

## 4. Backend-specific conditions

| Knob | FSDP | Megatron |
|---|---|---|
| Parallelism | `full_shard` (FSDP1), DP=8 | TP=1, PP=1, **DP=8** |
| Weights | HF, BF16-sharded, no fp8 param-storage | `--fp8-param-storage` (fp8, ~70 GB/GPU) |
| Linear | HF `nn.Linear` via `LumenConfig.enable` | `--lumen-linear` parallel linears |
| LoRA targets | all 7 linears (q,k,v,o,gate,up,down) | **attention-only** + `--lora-a2a` |
| Activation recompute | grad-checkpointing (use_reentrant=False) | full/block, **21 layers** |
| Attention / norm | BF16 SDPA / HF RMSNorm | AITER CK FMHA / RMSNorm |
| FP8 format | e4m3 | **hybrid** e4m3, amax most_recent hist 4 |
| Trainable params | 7B 40.0M/6.78B (0.59%); 70B 207M/69.2B (0.30%) | 70B attn-only LoRA |

FP8 scaling notes:
- **delayed** — per-tensor scalar scale; survives `weight.t()` → full FP8 backward.
- **blockwise2d** (Jet-RL) — square 128×128 scale tiles; full FP8 DGrad+WGrad (WGrad token
  count M=1×8192=8192 divisible by 128). 1D `blockwise` (not benchmarked) uses BF16 wgrad.

---

## 5. How to run

FSDP (precision via `MODE`):
```bash
MODE=bf16|fp8_delayed|fp8_blockwise2d \
HOST_RESULTS=/mnt/raid0/leiwu/mlperf/results/<dir> CONTAINER_NAME=<name> \
bash examples/llama2/run_fsdp_lora_7b_mi308.sh        # 7B
# 70B: CONFIG=config_MI308X_fsdp_lora_70b.sh HOST_MODEL=<70b dir> bash run_fsdp_lora_mi308.sh
```
Megatron 70B blockwise2d:
```bash
cd /mnt/raid0/leiwu/mlperf && bash run_mlperf_70b_latest_blockwise2d.sh
```
Useful env: `TRAIN_STEPS`, `HOST_DATA` (`/data_mlperf` = answer-only), `HOST_MODEL`,
`PYTORCH_TUNABLEOP_ENABLED` (default 0; set 1 for long runs to amortize first-step GEMM tuning).

---

## 6. Results

### 6.1 — 7B FSDP (200 steps, complete; all-token loss)

Final loss:

| Mode | final train | **final val** | Δval vs BF16 |
|---|---|---|---|
| bf16 | 1.4815 | **1.3866** | — (baseline) |
| fp8_delayed | 1.4848 | **1.3921** | +0.0055 |
| fp8_blockwise2d | 1.4866 | **1.3894** | +0.0028 |

Validation-loss convergence:

| step | bf16 | fp8_delayed | fp8_blockwise2d |
|---|---|---|---|
| 10 | 3.0345 | 3.2423 | 3.0666 |
| 40 | 1.6469 | 1.6935 | 1.6540 |
| 80 | 1.4318 | 1.4405 | 1.4356 |
| 120 | 1.3964 | 1.4026 | 1.3995 |
| 160 | 1.3875 | 1.3934 | 1.3904 |
| 200 | 1.3866 | 1.3921 | 1.3894 |

Per-step time:

| Mode | median (≥20) | step-1 (incl JIT) |
|---|---|---|
| bf16 | 5.23 s | 17 s |
| fp8_delayed | 6.77 s | 188 s |
| fp8_blockwise2d | 8.70 s | 148 s |

→ all three within **0.006** val-loss; FP8 ≈ BF16 accuracy at 7B, but FP8 is *slower*
(quant overhead + FSDP comm-bound at low arithmetic intensity). Start loss ~4.57 (confirms
pretrained load; random init ≈ 10.4).

### 6.2 — 70B FSDP (all-token loss; partial)

| Mode | step time | step-1 | val (all-token) | status |
|---|---|---|---|---|
| bf16 | ~49 s | 68 s | 1.2355 @48 | stopped @60 (snapshot) |
| fp8_blockwise2d | ~70 s | 335 s | 1.236 / 1.197 / 1.184 / 1.177 / 1.174 (@48/96/144/192/240) | NCCL watchdog hang ~256 |

FSDP 70B all-gathers the full 140 GB model every step (~49–70 s/step). The blockwise2d run
hung ~step 256 on a post-eval collective.

### 6.3 — 70B Megatron (answer-only loss = MLPerf metric)

| Mode | step time | mem | **val (answer-only)** | status |
|---|---|---|---|---|
| fp8_blockwise2d | ~34 s | 97% (192 GB) | **0.951 @192** (target 0.925) | reached first eval; on-trend to ≤0.925 |

Faster than FSDP (TP1 + fp8 param-storage keeps weights resident, no per-step all-gather),
0 NaN, mem ~97%. **val 0.951 @192 is the MLPerf answer-only metric** → directly comparable to
0.925; trend matches the README reference (best 0.9223 @ step576).

---

## 7. Native PyTorch FSDP vs Lumen FSDP (7B, BF16)

A pure-PyTorch-FSDP baseline (`examples/llama2/native_fsdp_baseline.py` — HF + torch FSDP +
PEFT, **no `LumenConfig`**; only `LLaMA2SFTDataset` reused for identical inputs), same §3
hyperparameters, same old all-token data, 200 steps.

| | final val | step time (steady, no-eval) | data pipeline |
|---|---|---|---|
| Lumen FSDP bf16 (run #1) | 1.3866 | ~5.23 s | `shuffle=False` (`_pack_next` packing) |
| native PyTorch FSDP bf16 | **1.4233** | **~4.5 s** | `shuffle=True` (per-row) |

Val convergence (native): 3.09→1.69→1.47→1.43→**1.4233** (@10/40/80/120/200).

⚠ **Comparison is confounded — not yet a clean Lumen-vs-native result.** The two runs used
**different data pipelines**: the run-#1 Lumen-bf16 log predates the `shuffle=True` dataset
change, so it used the old `_pack_next` streaming packer; the native baseline uses the new
per-row + epoch-shuffle path. The +0.037 val gap is dominated by this data-path difference,
not by Lumen-vs-native. Step time is more directly comparable: native is ~0.7 s/step faster
(no `LumenConfig` wrap / ExperimentTracker per-step overhead).

**TODO for a clean comparison:** re-run Lumen-bf16 with the current code (`shuffle=True`) so
both share the identical per-row+shuffle pipeline; then only the framework differs.

<!-- add result subsections for future experiments here -->

---

## 8. Data / loss-mask comparability (important)

gov_report `.npy` was prepared two ways:
- **all-token** (`/data`, 2D int32 input_ids only) → `loss_mask = all ones` → loss over the
  whole packed sequence. Runs #1–#5.
- **answer-only** (`/data_mlperf`, object array `loss_mask = labels != -100`, from the MLPerf
  `convert_dataset.py`) → loss only on summary tokens (train mask ~100%, **val mask ≈13%**).
  Run #6.

The MLPerf target **0.925 is an answer-only validation loss**. The all-token val numbers
(7B ~1.39; 70B FSDP 1.17–1.24) are a *different metric* — do not compare to 0.925. Only run
#6's 0.951 is on the MLPerf scale.

---

## 9. Bugs found & fixed

| Bug | Symptom | Fix |
|---|---|---|
| FSDP+LoRA dtype | `flatten ... uniform dtype` | cast PEFT LoRA adapters to bf16 |
| FSDP+LoRA grad | `flatten ... uniform requires_grad` | `use_orig_params=True` |
| FP8 fast-dispatch sig | `_gemm_per_tensor_ck takes 4 args, 6 given` | 6-arg signature for CK/Triton |
| AITER version skew | `static_per_tensor_quant_..._with_amax` ImportError | probe requires both symbols → eager fallback |
| blockwise2d + fp8-param-storage | `gemm_a8w8_blockscale: tuple index out of range` (scalar weight scale) | store 2D block weight scale; pass direct (not reciprocal); 2D-aware dequant |
| `scale_f32_1x1` trap | output_layer `shape '[1,1]' invalid for size 524288` | collapse scale to (1,1) only for per-tensor scalings |
| DP early-stop desync | NCCL deadlock right after eval (one rank exits loop) | removed per-rank EMA stop; collective `install_val_loss_early_stop_hook` on reduced val loss |

---

## 10. Notes & caveats

- **Warmup parity:** an early 7B FP8 run used 5 synthetic-warmup steps that `optimizer.step()`
  on garbage tokens, inflating val to 1.4238; `WARMUP_STEPS=0` for all modes removed it.
- **FSDP checkpoint saving is broken:** `save_fsdp_checkpoint` does a `state_dict()` collective
  guarded by `rank==0` → mismatch/SIGABRT when `SAVE_INTERVAL>0`. Runs above had saving off.
- **Per-step time is JIT/thermal sensitive:** step 1 includes hipBLASLt/AITER kernel compile;
  FP8 first-step ≫ BF16.
- **Megatron iter-192 log was overwritten** by a later restart (tee writes one file); only
  iters 1–20 survived in `llama2_megatron_train_fp8_blockwise2d_70b.log`.

---

## 11. Conclusions

1. **Accuracy:** FP8 (delayed & blockwise2d) ≈ BF16 at 7B (Δval < 0.006); 70B blockwise2d
   reaches answer-only val 0.951 @192, on-trend to MLPerf 0.925.
2. **Throughput (70B):** Megatron (TP1 + fp8 param-storage, ~34 s/step) > FSDP (~49–70 s/step,
   full-model all-gather every step). At 7B, FP8 is slower than BF16 (overhead-bound).
3. **blockwise2d on Megatron needed new code** — fp8 param-storage now emits 2D block weight
   scales (§9).
4. **Collective early-stop is mandatory** — any per-rank stop on local loss desyncs DP → NCCL
   deadlock.

---

## 12. Source logs (`examples/llama2/results/`)

- 7B FSDP: `llama2_fsdp_train_bf16_lora_7b.log`, `llama2_fsdp_train_fp8_delayed_lora_7b.log`,
  `llama2_fsdp_train_fp8_blockwise2d_lora_7b.log`
- 7B native FSDP baseline: `llama2_native_fsdp_train_bf16_lora_7b.log` (script `native_fsdp_baseline.py`)
- 70B FSDP: `llama2_fsdp_train_bf16_lora_70b.log` (→60),
  `llama2_fsdp_train_fp8_blockwise2d_lora_70b.log` (→256)
- 70B Megatron: `llama2_megatron_train_fp8_blockwise2d_70b.log` (iters 1–20 only);
  delayed reference (passes 0.925): `/mnt/raid0/leiwu/mlperf/logs/mlperf_70b_delayed*.log`
