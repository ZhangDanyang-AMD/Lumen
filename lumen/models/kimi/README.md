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
