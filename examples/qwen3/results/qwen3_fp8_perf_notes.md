# Qwen3-8B FSDP FP8 blockwise2d — performance notes

8×MI308X (gfx942), seq 2048, micro-batch 1, global batch 8. Steady-state step time:
**FP8 blockwise2d ~2784 ms/step** vs **BF16 ~1730 ms/step** (FP8 ~61% slower at this scale —
overhead-bound, consistent with the llama2-7B finding).

## Profile breakdown (torch.profiler, steps 12-15, Self CUDA total 12.74 s / 4 steps)

| Category | Kernel(s) | Self CUDA | Share |
|---|---|---|---|
| FP8 GEMM (fwd) | `aiter::gemm_a8w8_blockscale_ck` + ck xdl | 4.70 s | **37%** |
| **copy** | `aten::copy_` (18988 calls) | 2.93 s | **23%** |
| elementwise | `elementwise_kernel_manual_unroll` | ~1.96 s | 15% |
| **comm** | rccl all-gather / reduce-scatter | 1.71 s | **13%** |
| FP8 GEMM (bwd DGrad/WGrad) | `_gemm_a8w8_blockscale_kernel` | ~0.97 s | 8% |
| attention | flash fwd+bwd | ~1.23 s | 10% |
| FP8 quant | `dynamic_per_token_scaled_quant` | ~0.22 s | 2% |

## GEMM tuning result (tried — marginal)

Root cause investigated: the image's `a8w8_blockscale_tuned_gemm.csv` had **zero gfx942 entries**
(all gfx950) → blockscale GEMM fell to the default heuristic kernel. Tuned the 5 transformer-layer
shapes (M=2048; N/K over 1024/4096/12288) on gfx942/80CU via
`csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py --libtype both`.

| Metric | default | tuned | Δ |
|---|---|---|---|
| `gemm_a8w8_blockscale_ck` avg | 1.549 ms | 1.497 ms | −3.4% |
| Self CUDA total (4 steps) | 12.74 s | 11.97 s | −6.1% |
| steady step_time | 2784 ms | 2785 ms | ~0% (noise) |

**Conclusion: GEMM kernel selection is NOT the bottleneck.** The tuner picked `kernelId 0` for all
5 shapes — essentially what the default heuristic already used — so the default was already
near-optimal. Tuning is a ~3-6% free win at best but does not move step time. This **rules out GEMM
kernel choice** and points the next effort at copy/elementwise.

Tuned CSV: host `/mnt/raid0/leiwu/mlperf/aiter_tune/tuned_qwen3_gfx942.csv`. Enable via launcher:
`AITER_TUNED_CSV=<path> bash run_qwen3_fsdp_mi308.sh` (gated, default off; first step JIT-compiles
the tuned kernels). lm_head (N/K=151936) left untuned (tuner too slow on it; ~1% of GEMM time).

## Top optimization opportunities (ranked by ROI)

1. **copy/contiguous 23% — highest ROI.** Pinpointed with `LUMEN_COPY_TRACE=1`
   (see `qwen3_copy_trace.txt`). The dominant source is **base-weight FP8 quantization in the
   forward**, NOT the backward:

   | MB/step | calls/step | site |
   |---|---|---|
   | **29028** | 505 | `_quant_blockwise2d_weight:326` — BF16 tile-major `.contiguous()` |
   | **14514** | 505 | `_quant_blockwise2d_weight:334` — FP8 invert-reshape `.contiguous()` |
   | 7568 | 253 | DGrad `weight_data.t().contiguous()` (`linear.py:1384`) |
   | 3180+2726 | 253 | WGrad `g_col/x_col.t().contiguous()` |

   - **Root cause:** `_quant_blockwise2d_weight` (`linear.py:300-337`) does two full-tensor
     contiguous copies (BF16 in tile-major + FP8 out) per call. With LoRA the base weight is
     **frozen** so its FP8 quant is constant, yet it is **re-quantized every forward**, and
     **505 ≈ 2×253** because **gradient checkpointing re-runs the forward** in backward →
     the weight is quantized twice per step per linear. ~43.5 GB/step of copies.
   - **Fix (IMPLEMENTED): `cache_frozen_weight`.** Quantize the frozen base weight to FP8 once
     on the first forward (where FSDP has gathered the full weight) and cache `_fp8_weight_data`
     /`_fp8_weight_scale` on the module; the existing `fp8_weight_cache` path (`linear.py:1084`)
     then feeds it straight to the GEMM, skipping `_quant_blockwise2d_weight` on every forward
     and grad-ckpt recompute. Config: `LumenConfig(cache_frozen_weight=True)` /
     `--cache-frozen-weight` (default OFF — caches the full FP8 weight resident, so keep off for
     models too large to fit, e.g. 70B FSDP). Frozen detection is done at **patch time** (before
     FSDP wrap) because under FSDP a frozen view can falsely report `requires_grad=True`.

     The same flag also caches the **transposed** FP8 weight + scale (frozen → constant) so the
     blockwise2d **DGrad** reuses it instead of `weight_data.t().contiguous()` (`linear.py:1384`,
     ~7.6 GB/step) every backward. The transpose is attached to the stable cache tensor and
     threaded onto `ctx` in the forward — no autograd-signature churn.

     The same `cache_frozen_weight` flag also **skips the frozen base weight's entire WGrad**
     (its grad is discarded under LoRA anyway): the backward returns `grad_weight=None` and
     never runs the dequant→requant + transpose + WGrad GEMM. This is the biggest single win
     after the fwd cache. DGrad (grad to the input) is unaffected — verified bit-identical.

     | | copy_ Self CUDA | step_time | loss |
     |---|---|---|---|
     | baseline | 2.934 s | 2785 ms | — |
     | + weight-quant cache | 1.762 s (−40%) | 2645 ms (−5%) | bit-identical |
     | + DGrad transpose cache | 0.980 s (−67%) | 2390 ms (−14%) | bit-identical |
     | + skip frozen WGrad | **0.336 s (−89%)** | **1912 ms (−31%)** | **bit-identical** |

     Net: FP8 blockwise2d goes from +61% slower than BF16 (1730 ms) to **+11%**. Loss is
     unchanged at every step (frozen base weight contributes no grad; only DGrad + LoRA grads
     matter). Tests: `tests/quantize/test_cache_frozen_weight.py` (incl. DGrad-preservation).
   - Note: a first attempt to *optimize* (not skip) the WGrad requant path gave zero gain —
     under frozen-weight LoRA the base WGrad is discarded, so the right move is to skip it
     entirely, not speed it up.

2. **gradient-checkpointing recompute — biggest training-config lever.** The backward bubbles
   around `FullyShardedDataParallel._post_backward_hook` come from grad-ckpt: before each layer's
   backward, FSDP re-all-gathers params and **re-runs the forward** (activations weren't saved) —
   a serial block injected into the backward. It also doubles the forward GEMMs
   (`gemm_a8w8_blockscale_ck` = 758/step = fwd 253 + **recompute 253** + DGrad 253).
   - **Action (config, not a lib fix): disable grad-ckpt when memory allows.** `--no-grad-checkpointing`
     / `GRAD_CKPT=0` + `--no-limit-all-gathers` + `--forward-prefetch`. Removes the recompute
     (GEMM 758→506/step) and the backward bubbles.

     | | step_time | GEMM/step | loss |
     |---|---|---|---|
     | + skip frozen WGrad (grad-ckpt on) | 1912 ms | 758 | — |
     | + no grad-ckpt + overlap | **~1610 ms (−16%)** | **506** | **bit-identical** |

     Cumulative from baseline: **2785 → 1610 ms (−42%)**, below the BF16-with-ckpt baseline
     (1730 ms). **Default keeps grad-ckpt ON** — disabling needs activation memory (fits at 8B
     mb1 seq2048 / 192 GB; **will OOM at 70B or large batch/seq**).

3. **comm — FULL_SHARD is the right choice here.** Tried `SHARD_GRAD_OP` (ZeRO-2): **slower**
   (step 1912→2059 ms, all_gather 0.79→1.55 s). Under grad-ckpt + FP8 the resident full params
   gather in fewer/larger, less-overlapped ops. Keep FULL_SHARD. (After disabling grad-ckpt the
   backward all-gathers also drop, so comm is no longer the bottleneck.)

4. **elementwise** — largely already removed by the frozen-weight cache (the biggest
   elementwise kernel fell 490→161 ms/step). Remainder is mostly FSDP `reduce_dtype=float32`
   grad casts (framework, not the quant path) — low ROI.

5. **GEMM (now ~50-57% after the above) — near-optimal.** tuning confirmed only ~3% headroom.
   Remaining GEMM volume reduction would mean lm_head→BF16 (tune too slow; ~1%).

> Reproduce the profile: `LUMEN_PROF_START=12 LUMEN_PROF_END=15 MODE=fp8_blockwise2d MAX_STEPS=16 \
> bash run_qwen3_fsdp_mi308.sh`. Baseline vs tuned profiles:
> `qwen3_profile_fp8_blockwise2d.txt` / `qwen3_profile_fp8_blockwise2d_tuned.txt`.
> Attribute copies to call sites: add `LUMEN_COPY_TRACE=1` → `<out>_copy_trace.txt`
> (see `qwen3_copy_trace.txt`).
