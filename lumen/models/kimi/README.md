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
