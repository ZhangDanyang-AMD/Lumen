# Kimi-K3 / K3.1 KDA via aiter

`LUMEN_KDA_AITER=1` makes `LumenConfig.enable(model)` route every KDA attention
module (HF `KimiDeltaAttention` layout) through `aiter.ops.kda_fused_layer.kda_layer_fwd`.
`LUMEN_KDA_AITER_BACKEND=torch|fused` (default `fused`) picks the aiter layer backend;
per-piece kernels follow aiter's `AITER_KDA_PIECES` / `set_kda_pieces`. Packed
sequences pass `cu_seqlens` through the attention kwargs. Module parameters are
handed to aiter as views, so no weight is copied.

`A_log` is per head and padded (K3: `[128]` for 96 heads; aiter reads `[:H]`), and
`gate_lower_bound` (K3: -5) is passed as `kda_layer_fwd(lower_bound=...)` for the safe gate
`lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))`. Only KV/conv-cache calls and
padded `attention_mask` keep the original module forward.

## FLA (Megatron reference, kda_backend checks)

Megatron-LM#6872's container pins `flash-linear-attention==0.5.2` (Kimi-K3 Dockerfile). `fla.ops.kda`
lives in `fla-core`, which is pinned to the same version. Install it without dependencies so that
pip keeps the ROCm torch and triton:

```bash
pip install --no-deps "fla-core==0.5.2" && pip install einops
```

`check_fla_rocm()` (and the `test_fla_on_rocm_check_reports` test) report whether FLA's Triton
`causal_conv1d` and `rms_norm_gated` actually run on the GPU with the installed triton.

## Required aiter branches

| What | aiter PR / branch |
| --- | --- |
| `kda_layer_fwd` / `kda_layer_bwd`, `KDALayerWeights`, `set_kda_pieces`, `cu_seqlens` | ROCm/aiter#8739, `cursor/kda-fused-layer-bwd-8350` at `bbed79a` or later (needs `lower_bound`, padded `A_log`) |
| Fused pieces it composes (optional; torch fallback otherwise) | #8744 chunk fwd, #8740/#6498 chunk bwd, #8726 conv1d bwd, #8741 gated RMSNorm bwd |

The Tiny-L13 mapping in `tiny_l13.py` is ported from the closed ROCm/aiter#8749
(`cursor/kda-tiny-l13-4bc0`).

## Tiny-L13 findings

* KDA layers (0-indexed): 0-2, 4-6, 8-10; gated MLA: 3, 7, 11, 12.
* `A_log` is `[128]`: one decay per head, padded past 96 heads. It is used as-is.
* `gate_lower_bound = -5` (safe gate); the parity test uses it on both sides.
* Conv weights, `A_log`, `dt_bias` and `o_norm` are fp32 in the checkpoint and stay fp32.

## Parity test

```bash
LUMEN_KIMI_TINY_L13_PATH=/path/to/Kimi-K3-Tiny-L13 \
  pytest tests/models/test_kimi_kda_aiter.py   # one gfx950 GPU
```

Without the env var, random weights at the real shapes are used.

## Tier 1: KDA-only stack (no checkpoint)

`lumen/models/kimi/kda_stack.py` builds the 9 Tiny-L13 KDA layers as a stack with random init from
the Tiny-L13 config: embedding, RMSNorm + KDA + residual blocks, final RMSNorm and LM head. MLA and
MoE are omitted. It keeps the real widths (hidden 7168, 96 × 128 heads, vocab 163840,
`gate_lower_bound=-5`, padded `A_log`) and fp32 conv, `A_log`, `dt_bias` and `o_norm`, in BF16. With
`LUMEN_KDA_AITER=0` each KDA layer runs aiter's `torch` reference; with `=1` it runs the fused layer.
It needs no download, compressed-tensors or fla-core.

```bash
# benchmark (tier kda-only is the default): step-1 loss and all grads, 50-step loss parity, step-time table
python benchmarks/bench_kimi_tiny_l13_kda.py --tier kda-only --seq-len 2048 --steps 50 --warmup 5 --parity-steps 32
AITER_KDA_PIECES=chunk_fwd=flydsl python benchmarks/bench_kimi_tiny_l13_kda.py --tier kda-only
# pytest: loss, logits, all grads, 5-step SGD loss curve (seq 512)
pytest -s tests/models/test_kimi_kda_stack.py
```

## Whole-network Tiny-L13 test

This runs the full 13-layer model (remote code; only the 9 KDA layers change) in BF16
fwd+bwd on one gfx950 GPU. It compares loss, logits and every parameter gradient with
`LUMEN_KDA_AITER=1` vs `0`, then checks that the loss curve over a few SGD steps matches.
To enable one aiter piece at a time, set `AITER_KDA_PIECES`:

```bash
export LUMEN_KIMI_TINY_L13_PATH=/path/to/Kimi-K3-Tiny-L13   # huggingface-cli download Qinghao/Kimi-K3-Tiny-L13
pytest -s tests/models/test_kimi_tiny_l13_whole_net.py                             # all pieces: aiter defaults
AITER_KDA_PIECES=chunk_fwd=flydsl pytest -s tests/models/test_kimi_tiny_l13_whole_net.py
AITER_KDA_PIECES=chunk_fwd=flydsl,chunk_bwd=flydsl pytest -s tests/models/test_kimi_tiny_l13_whole_net.py
```

`LUMEN_KIMI_WHOLE_NET_SEQLEN` (default 256) and `LUMEN_KIMI_WHOLE_NET_STEPS` (default 3)
size the run. The test prints the selected pieces and the pieces that actually ran.
It needs `transformers`, `fla-core` (the eager KDA reference) and `compressed-tensors`
(MXFP4 experts).

The checkpoint is 210 GB, mostly MXFP4-packed routed experts (896 experts × 12 MoE layers).
The packed experts are not trainable, so "all parameter gradients" covers the floating-point
params that get a gradient. Whether weights, activations and the CPU copies of the gradients
fit on one 288 GB MI355X has not been measured.

## Remote-code check (`modeling_kimi_linear.py`)

* `KimiDeltaAttention.forward(hidden_states, attention_mask=None, cache_params=None, **kwargs)`
  returns `o` (a tensor, not a tuple). `attention_mask` is the 2D padding mask; a non-2D mask
  is replaced by `kwargs["padding_mask"]`. `cu_seqlens` arrives in kwargs.
* Parameter names match the mapping: `q/k/v_proj`, `q/k/v_conv1d` (FLA `ShortConvolution`,
  weight `[C,1,W]`), `f_a_proj`, `f_b_proj`, `A_log`, `dt_bias`, `b_proj`, `g_proj` (only with
  `use_full_rank_gate=True`, which K3 sets; otherwise `g_a_proj`/`g_b_proj`), `o_norm`
  (FLA `FusedRMSNormGated`, sigmoid), `o_proj`. The output gate is `o_norm(o, g_proj(x))`.
* The remote code initializes `A_log` as `[num_heads]` (96), while the checkpoint holds `[128]`.
  The whole-network test loads with `ignore_mismatched_sizes=True` and then restores the
  checkpoint's `[128]` tensor.

## Whole-network benchmark (modeled on NVIDIA/Megatron-LM#6872)

```bash
export LUMEN_KIMI_TINY_L13_PATH=/path/to/Kimi-K3-Tiny-L13
python benchmarks/bench_kimi_tiny_l13_kda.py --seq-len 4096 --steps 50 --warmup 5 --parity-steps 32 \
    --json kda_bench.json
AITER_KDA_PIECES=chunk_fwd=flydsl python benchmarks/bench_kimi_tiny_l13_kda.py   # one FlyDSL piece at a time
```

Both modes, `eager` (`LUMEN_KDA_AITER=0`) and `aiter` (`=1`), start from the same weights and the
same fixed-length mock token stream. The script prints a table in the issue's
"Released recipes and benchmarks" format: recipe, hardware, TP/PP/EP/CP, GBS, median iteration
time, median TFLOP/s/GPU, plus tokens/s/GPU and peak allocated memory. It then prints an LM-loss
parity line.

| Megatron-LM#6872 | This benchmark |
| --- | --- |
| Median iteration time and logged TFLOP/s/GPU, steps 6–50 | Same medians over steps `warmup+1`..`steps` (default 6–50). FLOPs are an analytic fwd+bwd model count: dense GEMMs, top-16 MoE from config shapes, causal MLA scores. KDA recurrence is not counted. |
| Fixed-length mock documents | Same (`--data mock`). |
| 32-step continuation, max LM-loss difference within `1e-3` | Max per-step \|Δloss\| over `--parity-steps` (default 32), tolerance `1e-3`, plus relative grad-norm trajectory. This uses mock data from the checkpoint, not LongAlign-10k. |
| BF16 module-level fwd/bwd parity | `tests/models/test_kimi_kda_aiter.py` (per KDA layer) and `tests/models/test_kimi_tiny_l13_whole_net.py`. |

What one gfx950 GPU cannot match:

* Parallelism and scale: TP/PP/EP/CP are all 1, GBS = micro-batch. Megatron uses 64–512 GB300 GPUs,
  16.7M tokens per step, CP8–32 and EP64.
* Context: Megatron runs at 64K/256K/1M with THD packing. Here the limit is one GPU's memory after
  210 GB of weights; the default is 4K and it is not yet measured.
* Model: Tiny-L13 has 13 layers, versus the 93-block full model or 9-block proxy. It has no MTP depth.
* Numerics: Megatron trains MXFP8 recipes. Here the run is BF16 compute with frozen MXFP4-packed
  routed experts, so it is not QAT.
* Optimizer: Megatron uses per-head Muon. Here it is SGD by default, because AdamW state for about
  10B floating-point params does not fit next to the weights; `--optimizer adamw` is available.
* Kernels and stack: cuDNN KDA, MoK, latent CP, fused AttnRes and Quantile Balancing have no
  counterpart. MLA, MoE and AttnRes run the remote-code eager path.
* Real-data continuation and checkpoint resume are out of scope.

## K3.1 coverage

I checked `moonshotai/Kimi-K3.1-Pro`'s `config.json` and the safetensors headers of layer 0.

| | K3 (Tiny-L13 / config) | K3.1-Pro |
| --- | --- | --- |
| Layers | 13 (Tiny) | 93 |
| KDA layers (1-indexed) | 1-3, 5-7, 9-11 (9) | every layer except 4k and 93 (69) |
| Full-attention layers | gated MLA at 4, 8, 12, 13 | DSA (`dsa_config.dsa_layers`, 32-head top-2048 indexer) at 4, 8, …, 92, 93 (24) |
| KDA widths | hidden 7168, 96 × 128, conv 4, gate rank 128, `use_full_rank_gate` | identical |
| KDA params | `q/k/v_proj`, `q/k/v_conv1d` `[12288,1,4]` F32, `f_a_proj` `[128,7168]`, `f_b_proj` `[12288,128]`, `A_log` `[128]` F32, `dt_bias` `[12288]` F32, `b_proj` `[96,7168]`, `g_proj` `[12288,7168]`, `o_norm` `[128]` F32, `o_proj` | identical names, shapes and dtypes |
| Gate | `gate_lower_bound = -5` | `-5` |
| `rms_norm_eps` | 1e-5 | 1e-5 |
| Remote code | `modeling_kimi_linear.py` (`KimiDeltaAttention`) | not published (`auto_map` has only `AutoConfig`; architecture `KimiK31ForConditionalGeneration`) |

KDA itself is unchanged in K3.1, so `kda_layer_fwd` with the same mapping covers every K3.1 KDA layer.
`patch_kda_layers` reads `linear_attn_config.kda_layers` from the model config and patches only those
`layer_idx`. The tier-1 stack takes `--model k3.1-pro`; its default is the first 9 of the 69 KDA layers,
because all 69 (about 41B params plus grads) do not fit on one GPU:

```bash
python benchmarks/bench_kimi_tiny_l13_kda.py --tier kda-only --model k3.1-pro --num-layers 9 --seq-len 2048
```

Open: K3.1's KDA class name and forward are unverified until its modeling code is published. Patching
matches the `KimiDeltaAttention` name or the parameter attributes, and assumes K3's forward signature.
The DSA layers and the K3.1 full-network tier are out of scope here.

## Localizing an eager-vs-aiter mismatch

`benchmarks/diag_kimi_kda_stages.py` runs one KDA layer with the same weights and input on both sides.
It prints the norm-relative error of each forward stage (q/k after conv + l2norm, v, gate g, beta,
chunk o, y) and of the chunk-forward piece alone on eager inputs. It then prints every backward
gradient with the selected pieces and with each stage forced to torch: `chunk_fwd`, `chunk_bwd`,
`conv_bwd`, `proj` (FlyDSL gate and output-gate GEMM epilogues) and `rowops` (FlyDSL l2norm and
gated-RMSNorm backward). The stage whose torch override removes the error is the one that is wrong;
`all=torch` must match eager.

```bash
python benchmarks/diag_kimi_kda_stages.py --dtype fp32 --widths small --seq-len 256
python benchmarks/diag_kimi_kda_stages.py --dtype bf16 --widths full --seq-len 512
python benchmarks/diag_kimi_kda_stages.py --dtype bf16 --widths full --seq-len 512 --no-lower-bound
```

Since aiter#8739 `63497ee`, the q/k L2 norm uses a fixed eps of 1e-6, the same as FLA, on both sides.
The gated RMSNorm uses `o_norm.eps` (1e-5).

BF16 parity is graded against fp32. The torch backend computes in fp32 from bf16 inputs, while the
fused path feeds bf16 q/k/v to the chunk kernels, so aiter-bf16 is expected to sit further from fp32
than eager-bf16 (#8739 measures about 2% grad deviation at K=128, and 6–15% at K=16). The per-layer
and tier-1 stack tests run an fp32 eager copy. For each tensor they require
`err(aiter_bf16) <= max(4 × err(eager_bf16), 3e-2)`, plus zero grads on the padded `A_log` entries.
fp32 runs keep the strict `1e-2` direct comparison. The whole-net test and the benchmark have no fp32
copy, so they compare aiter against eager directly at 3e-2. Tests, the diagnostic and the benchmark
print `LAST_PIECES['gemm']`.

## Loss-trajectory grading

The tier-1 tests follow the Megatron-LM#6872 continuation protocol: 32 SGD steps from the same weights
and data, with lr 1e-5 cosine to 1e-6 over 50 iterations and no warm-up. The primary check is
per-step `|aiter − eager| ≤ 1e-3`.

In BF16 the tests also run an fp32 trajectory and require, per step,
`|aiter − fp32| ≤ max(4 × |eager_bf16 − fp32|, 1e-3)`:

- **Factor 4:** the same bound as the grads. On gfx950, aiter measures 1.4–1.5× eager-bf16's grad
  error against fp32.
- **Floor 1e-3:** the #6872 tolerance, i.e. the loss difference Megatron accepts between two
  implementations.

All three trajectories are printed. The small-width fp32 wiring check keeps 5 steps at lr 0.1 so the
trajectory moves.

## MXFP8 numerics: Lumen vs TE

`lumen/ops/quantize/mxfp8_te_reference.py` reproduces TE `MXFP8BlockScaling` in PyTorch:

- **E8M0 scale:** the exponent is `ceil(log2(amax / 448))`, i.e. TE's `float_to_e8m0` rounding toward +inf. It is clamped to biased [0, 254], and a zero amax gives byte 0.
- **E4M3 cast:** round-to-nearest-even with satfinite.
- **Operand axes:** fprop uses X and W rowwise. dgrad uses dY rowwise × W columnwise. wgrad uses dY columnwise × X columnwise.
- **Accumulation:** fp32, with optional accumulation into an fp32 `main_grad`.

Lumen's fast `quantized_linear(scaling_type="mxfp8")` quantizes with aiter's
`_convert_to_mxfp8_kernel`, which uses torchao's rule: `floor(log2(rne_3bit(amax))) - 8`. This gives
the same byte or one byte less than TE. When it is one less, the block's largest values clip at 448:
for example amax 460 or 230, whose mantissa rounds down. The Megatron KDA reference therefore defaults
to the TE-faithful helper (`--precision mxfp8`). `--precision mxfp8-lumen` keeps the fast path, and
`tests/ops/test_mxfp8_te_reference.py` reports the gap on gfx950.

In the benchmark, `--main-grad` (on by default) keeps fp32 master weights. With `--precision mxfp8`,
the reference accumulates the KDA projections' wgrad into fp32 `main_grad`.

## aiter in Megatron precision

#8739 `95a82cd` adds aiter's Megatron rounding mode, `AITER_KDA_PRECISION=megatron`. In this mode
aiter rounds the following to bf16: the projection outputs, the conv result before SiLU, q/k after
the L2 norm, and o; the gated norm is rounded before the gate. Conv and norm weights are bf16, while
A_log and dt_bias stay fp32.

Every comparison against the Megatron reference sets the aiter side to this mode. That covers
diag `[4]`, the vs-megatron tier-1 and per-layer tests, the whole-net tests, and the benchmark
(`--aiter-precision`, which defaults to `megatron` with `--reference megatron-fla`).

Lumen sets the mode through `patch_kda_layers(kda_precision=)` or `LUMEN_KDA_AITER_PRECISION`, and
`unpatch_kda_layers` resets it. It combines with `gemm_precision="mxfp8"` once that aiter mode
lands. The aiter-torch comparisons keep aiter's default precision.

## aiter as Megatron `kda_backend` (aiter#8755)

`LUMEN_KDA_MEGATRON_BACKEND=aiter` makes Lumen's `install_all()` call `install_kda_aiter_backend()`.
That is `lumen.models.kimi.megatron_kda_backend` (re-exported in `lumen.models.megatron_patches`),
and it does three things:

1. **Registers the backend.** It registers `kda_backend="aiter"` through
   `aiter.ops.kda.patch_megatron_kda_backend`. The recurrence comes from `LUMEN_KDA_AITER_RECURRENCE`,
   which is `triton` by default, or `flydsl` or `auto`.
2. **Substitutes the CP context.** Megatron's `build_cp_context` is `None` without FLA's CP extension.
   In that case it is replaced by `aiter.ops.kda.build_kda_cp_context` (contiguous layout only).
3. **Checks FLA on this GPU.** It checks FLA `causal_conv1d` (plain, varlen, and single-rank
   `cp_context`) and `rms_norm_gated` against PyTorch. If `rms_norm_gated` fails, it is replaced by the
   Megatron gated-norm formula. `causal_conv1d` has no substitute, so a failure is reported.

Recipe knobs: `examples/kimi_k3/kda_aiter_cp2_overrides.yaml` sets `kda_backend: aiter`,
`linear_cp_mode: chunkwise` and CP2.

`--reference megatron-aiter` runs the same Megatron layer math with `aiter.ops.kda.chunk_kda` as the
recurrence, for the tier-1 stack and the benchmark.

Tests and run commands:

```bash
pytest -s tests/models/test_kimi_megatron_kda_backend.py           # registration, fla vs aiter, CP2/TP2 if 2 GPUs
torchrun --nproc_per_node 2 benchmarks/dist_kimi_kda_cp_tp.py      # CP2 vs CP1, TP2 vs TP1, 32-step loss curve
```

## FLA (Megatron reference, kda_backend checks)

Megatron-LM#6872's container pins `flash-linear-attention==0.5.2` (Kimi-K3 Dockerfile). `fla.ops.kda`
lives in `fla-core`, which is pinned to the same version. Install it without dependencies so that
pip keeps the ROCm torch and triton:

```bash
pip install --no-deps "fla-core==0.5.2" && pip install einops
```

`check_fla_rocm()` (and the `test_fla_on_rocm_check_reports` test) report whether FLA's Triton
`causal_conv1d` and `rms_norm_gated` actually run on the GPU with the installed triton.

## Required aiter branches

| Feature | aiter PR / branch | Base |
| --- | --- | --- |
| `kda_layer_fwd`, safe gate, padded A_log, Megatron rounding (`AITER_KDA_PRECISION=megatron`) | #8739 `cursor/kda-fused-layer-bwd-8350` at `95a82cd` or later | main |
| `gemm_precision="mxfp8"` (TE MXFP8BlockScaling, `convert_to_mxfp8(scale_mode="te")`) | #8757 `cursor/kda-mxfp8-gemm-precision-9e7f` | #8739 `95a82cd` |
| Recurrence API `aiter.ops.kda`, chunkwise CP, `patch_megatron_kda_backend` | #8755 `cursor/kimi-k3-kda-recurrence-cp-backend-5470` | #8744 |

The Megatron-reference comparisons default to `gemm_precision="mxfp8"` together with
`AITER_KDA_PRECISION=megatron`. They need #8757, which contains #8739, and they skip when the
installed aiter's `kda_layer_fwd` has no `gemm_precision`. The `kda_backend` tests need #8755.

## Known gap: FLA `chunk_kda` backward on ROCm

With `fla-core==0.5.2` on triton 3.6 (ROCm, gfx950), FLA's `chunk_kda` **forward** compiles but the
**backward** does not. `chunk_kda_bwd_intra`'s Triton kernel `chunk_kda_bwd_kernel_intra`
(`fla/ops/kda/chunk_intra.py`) fails in the AMD backend with `PassManager::run failed`, raised from
`chunk.py → chunk_bwd.py → chunk_intra.py:929`.

The likely trigger is that kernel's safe-gate path. It calls `tl.gather` with an int16 index, which is
used when `fla.utils.IS_GATHER_SUPPORTED` is true. `kda_reference.fla_backward_status()` first probes
FLA's backward as-is. If that fails, it retries with FLA's own non-gather path
(`chunk_intra.IS_GATHER_SUPPORTED = False`, a `tl.load` of the same value).

When FLA's backward still fails, the Megatron reference keeps FLA for forward-only calls. For
differentiable calls it keeps the Megatron layer and swaps only the recurrence, by default for an
independent PyTorch recurrence with FLA `chunk_kda`'s precision. That means chunk 64, gate cumsum,
decays, A matrices and solve in fp32, every MMA operand (gated q/k, v, A_qk, u, state) rounded to
bf16, and fp32 accumulation and state. `LUMEN_KDA_REFERENCE_FALLBACK=torch-fp32` selects an
fp32-throughout recurrence, which is more precise than Megatron's. `LUMEN_KDA_REFERENCE_FALLBACK=aiter`
selects aiter#8755's `chunk_kda`, as a diagnostic only, because it makes the reference partly aiter.
`kda_reference.LAST_BACKEND` reports which recurrence ran, and the tests and diag print it.

Lumen's fast `quantized_linear(scaling_type="mxfp8")` follows TE as well. The scale rule is TE's by
default (`LUMEN_MXFP8_SCALE_MODE=te`), through aiter's `convert_to_mxfp8(scale_mode="te")` from #8757,
or the PyTorch TE reference when that aiter build is missing. Forward quantizes X and W rowwise. It
also quantizes them columnwise directly from the bf16 tensors and saves those columnwise copies for
backward. dgrad then uses dY rowwise × W columnwise, and wgrad uses dY columnwise × X columnwise, with
fp32 output that is optionally added into `main_grad`. No operand is re-quantized from a dequantized
copy. `test_lumen_mxfp8_vs_te` requires every quantized operand to be bit-exact against
`mxfp8_te_reference`.

## aiter weight caches (#8739 `2130550`)

aiter's KDA weight caches follow TE's `is_first_microbatch` semantics. `None`, Lumen's default, means
no reuse across calls. `True` rebuilds the caches and `False` reuses them.

- **Patched layers:** they forward a caller's `is_first_microbatch`, or the value set with
  `kda_aiter.set_kda_is_first_microbatch(model, flag)`. The benchmark sets `True` on each step's
  single microbatch.
- **Optimizer hook:** `kda_aiter.register_kda_optimizer_hooks(optimizer)` calls
  `invalidate_weight_caches()` after every `optimizer.step()`. Lumen's FSDP
  `register_quant_optimizer_hooks` installs it whenever `LUMEN_KDA_AITER=1`.
- **Tests:** the tests and the benchmark also invalidate after every in-place parameter update.

## Megatron reference under recompute

`megatron_kda_forward(recompute=True)` mirrors `recompute_gdn` with `torch.utils.checkpoint`. The
recurrence backend, FLA or the PyTorch fallback when FLA's backward fails, is resolved before the
checkpointed region, including the one-time FLA probe. The PyTorch recurrence has no inner
per-chunk checkpoints. The forward and its recomputation therefore run the same ops and save the
same tensors.
