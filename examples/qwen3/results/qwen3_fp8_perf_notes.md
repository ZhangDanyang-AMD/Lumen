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

3. **comm — `SHARD_GRAD_OP` once grad-ckpt is off.** FULL_SHARD frees params after forward and
   **re-all-gathers them in backward** for DGrad (the backward bubbles around `_post_backward_hook`;
   all_gather 73/step = 37 fwd + 36 bwd). `SHARD_GRAD_OP` (ZeRO-2) keeps params resident → **no
   backward all-gather** (73→37/step). step_time 1610→**1542 ms**, loss bit-identical.
   - Caveat: `SHARD_GRAD_OP` is only a win **with grad-ckpt off**. With grad-ckpt ON it was
     *slower* (1912→2059 ms) — the recompute still re-gathers and the resident full params hurt
     overlap. So pair them: `GRAD_CKPT=0 SHARDING=shard_grad_op`.
   - `NO_SHARD` (remove the fwd all-gather too) is deprecated and errors here
     ("Cannot writeback when the parameter shape changes" on lm_head) — not viable.
   - Note: backward all-gather of the frozen base weight is itself wasted (our DGrad uses the
     cached FP8 transpose, not the gathered BF16) — `SHARD_GRAD_OP` sidesteps it by not freeing.

4. **elementwise** — largely already removed by the frozen-weight cache (the biggest
   elementwise kernel fell 490→161 ms/step). Remainder is mostly FSDP `reduce_dtype=float32`
   grad casts (framework, not the quant path) — low ROI.

5. **GEMM — B-preshuffle kernel is the big win (~2.6x).** The blockscale CK kernel's per-128×128
   dequant has poor MFMA/memory layout. AITER's **bpreshuffle** blockscale GEMM pre-shuffles the
   weight (`shuffle_weight(w,(16,16))`) into an engine-friendly layout — **bit-identical** math,
   ~2.5-2.8x faster on gfx942 (micro-bench). The base weight is frozen, so the shuffle is done
   **once and cached** (alongside the FP8 cache); the activation scale is supplied in the kernel's
   transposed layout. Forward + DGrad both use it.
   - Enable: `bpreshuffle_gemm` (needs `cache_frozen_weight`) / `--bpreshuffle` / `BPRESHUFFLE=1`.
   - Qwen3-8B end-to-end: FP8 GEMM **1028 → 396 ms/step (~2.6x)**, step_time **1542 → 960 ms
     (−38%)**, loss **bit-identical**. Tests: `test_cache_frozen_weight.py::test_bpreshuffle_matches_blockscale`.
   - Regular kernel-id tuning was only ~3% (CK instances near-equal); bpreshuffle is a *different*
     kernel family, hence the large gain.
   - lm_head→BF16 (~1%) is the only remaining GEMM-volume lever; not worth it.

## Cumulative (Qwen3-8B FSDP fp8 blockwise2d, 8×MI308X)

| stage | step_time | note |
|---|---|---|
| FP8 baseline | 2785 ms | — |
| + frozen-weight cache + skip WGrad | 1912 ms | lossless |
| + no grad-ckpt + overlap | 1610 ms | memory-for-speed |
| + SHARD_GRAD_OP | 1542 ms | memory-for-speed |
| **+ bpreshuffle GEMM** | **960 ms** | lossless; **−66% vs baseline, 44% faster than BF16 (1730)** |

> Reproduce the profile: `LUMEN_PROF_START=12 LUMEN_PROF_END=15 MODE=fp8_blockwise2d MAX_STEPS=16 \
> bash run_qwen3_fsdp_mi308.sh`. Baseline vs tuned profiles:
> `qwen3_profile_fp8_blockwise2d.txt` / `qwen3_profile_fp8_blockwise2d_tuned.txt`.
> Attribute copies to call sites: add `LUMEN_COPY_TRACE=1` → `<out>_copy_trace.txt`
> (see `qwen3_copy_trace.txt`).
