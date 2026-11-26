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

Both sides use one `eps` (`o_norm.eps`, 1e-5 for K3 and K3.1) for the q/k l2norm and the gated RMSNorm,
because `kda_layer_fwd` takes a single `eps`. FLA's remote-code KDA uses its own l2norm eps (1e-6 by
default) together with the config's 1e-5 RMSNorm. That does not affect eager-vs-aiter parity, but it
is a small difference from the HF model.
