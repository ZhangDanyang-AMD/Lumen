# FP8 Padding/Unpadding Utilities — Design Spec

**Date:** 2026-03-23
**Status:** Draft
**Scope:** Layer 1 (refactor ad-hoc padding) + Layer 2 (fix MXFP8 `quantize_input`)

## 1. Problem

FP8 block-scaled quantization (blockwise, blockwise2d, MXFP8) requires tensor
dimensions to be multiples of the block size.  Lumen handles this today with
**ad-hoc inline padding** scattered across three files:

| Location | What it pads | Block size | Mechanism |
|----------|-------------|-----------|-----------|
| `scaling_manager._round_to_mxfp8` | Last dim N | 32 | `F.pad` + slice |
| `scaling_manager.quantize_block_mxfp8` (thd 2D) | Sequence S | config | Zero buffer + slice |
| `grouped_gemm._quant_blockwise_2d` | K and N | 128 | `torch.zeros` + copy + slice |

Meanwhile, `linear.quantize_input` for MXFP8 calls `convert_to_mxfp8`
**without** padding, causing an assert failure when the input dimension is not
a multiple of the MXFP8 block size.  Note that `convert_to_mxfp8` asserts
alignment on **both** the flattened M and N dimensions (lines 186-187 of
`ops/quantize/ops.py`), not just the quantization axis.

There is no shared utility, no central alignment-size mapping, and the
default block sizes (32, 64, 128) drift between call sites.

## 2. Prior Art

### Megatron-Core

- **`get_fp8_align_size(fp8_recipe)`** — returns the alignment size required
  for FP8 GEMM given a recipe (delayed, tensorwise, blockwise, mxfp8).
- **`get_padding(seq_len, cp_size, tp_size, ...)`** — computes sequence-level
  padding at initialization time, considering SP, CP, TP overlap, and FP8
  jointly.  Sequences are padded once; no runtime unpadding at GEMM level.

### TransformerEngine

- **Userbuffers MXFP8** requires tensor dimensions to be multiples of 128.
  Non-compliant shapes are rejected, not auto-padded.
- FP8 alignment is enforced **upfront** — TE does not do dynamic padding
  inside GEMM paths.

### Key insight

Neither TE nor Megatron-Core pads at the GEMM level dynamically.  They
require alignment upfront (via `get_padding` or architecture constraints).
Lumen's approach should provide the same alignment-query API while also
offering a `pad_to_block` utility for the cases where runtime padding is
unavoidable (MXFP8 round-trip, MoE weight quantization, standalone FSDP
trainers without Megatron-Core's `get_padding`).

## 3. Design

### 3.1 New File: `lumen/ops/quantize/padding.py`

Two public functions, no side effects, no state.

#### `get_fp8_align_size(scaling_type, block_size=128) -> int`

Returns the alignment size that tensor dimensions must satisfy for a given
FP8 scaling recipe.  Analogous to Megatron-Core's
`get_fp8_align_size(fp8_recipe)`.

Mapping:

| `scaling_type` | Returned alignment |
|----------------|--------------------|
| `"delayed"`, `"dynamic"`, `"per_token"`, `"none"` | 1 |
| `"blockwise"`, `"blockwise2d"` | `block_size` (default 128) |
| `"mxfp8"` | `32 if block_size > 64 else block_size` |

The MXFP8 rule preserves the existing logic in `linear.quantize_input`
(`mxfp8_block = 32 if block_size > 64 else block_size`) without change.
For the common default `block_size=128`, this returns 32.

Raises `ValueError` for unknown `scaling_type`.

#### `pad_to_block(tensor, align_size, dim=-1) -> Tuple[Tensor, int]`

Pads `tensor` along `dim` so its size is a multiple of `align_size`.

- Returns `(padded_tensor, orig_size)`.
- **Zero-copy fast path:** when no padding is needed, returns the original
  tensor object (no allocation, `padded_tensor is tensor`).
- Uses `torch.nn.functional.pad` with zero-fill.
- Supports negative `dim` indexing.
- Callers unpad via `tensor.narrow(dim, 0, orig_size)` or equivalent slice.

### 3.2 Layer 1 — Replace Ad-Hoc Padding (Pure Refactor)

#### 3.2.1 `scaling_manager._round_to_mxfp8`

Replace inline `F.pad` / slice with `pad_to_block`.  Padding runs
**before** `.to(torch.bfloat16)` to preserve identical behavior:

```python
from lumen.ops.quantize.padding import pad_to_block

def _round_to_mxfp8(tensor, block_size=32):
    orig_dtype = tensor.dtype
    orig_shape = tensor.shape
    flat = tensor.reshape(-1, orig_shape[-1]).contiguous()
    flat, orig_n = pad_to_block(flat, block_size, dim=-1)  # pad in orig dtype
    data_bf16 = flat.to(torch.bfloat16)
    # ... convert_to_mxfp8 / convert_from_mxfp8 ...
    if data_hp.size(-1) != orig_n:
        data_hp = data_hp[:, :orig_n]
    return data_hp.reshape(orig_shape).to(orig_dtype)
```

Behavior: identical.  M dimension is not padded here; callers of
`_round_to_mxfp8` are responsible for M alignment (guaranteed by Megatron
`get_padding` or architecture).

#### 3.2.2 `grouped_gemm._quant_blockwise_2d`

Replace manual `torch.zeros` buffer + copy with two `pad_to_block` calls:

```python
from lumen.ops.quantize.padding import pad_to_block

def _quant_blockwise_2d(w, fp8_dtype, block_size=128):
    E, K, N = w.shape
    # ...
    for e in range(E):
        we = w[e]
        we, orig_k = pad_to_block(we, block_size, dim=0)
        we, orig_n = pad_to_block(we, block_size, dim=1)
        # ... reshape to blocks, compute scales, quantize ...
        w_fp8[e] = q_full[:orig_k, :orig_n].to(fp8_dtype)
```

Behavior: identical.  `F.pad` zero-fills the same way as `torch.zeros` + copy.

#### 3.2.3 `scaling_manager.quantize_block_mxfp8` (thd 2D block) — NO CHANGE

This function pre-allocates a zero buffer `(B, H, padded, D)` and copies
tokens in.  Its padding pattern (pre-allocated buffer, not padding an existing
tensor) does not match `pad_to_block`'s API.  Refactoring here would increase
complexity without benefit.  Left as-is.

### 3.3 Layer 2 — Fix `quantize_input` MXFP8 Assert Crash

In `linear.quantize_input`, the `"mxfp8"` branch currently calls
`convert_to_mxfp8` without padding, which asserts on unaligned input.

Fix: insert `pad_to_block` before `convert_to_mxfp8` for **both** M and K
dimensions, and **do not unpad** the output:

```python
if scaling_type == "mxfp8":
    from lumen.ops.quantize.ops import convert_to_mxfp8
    from lumen.ops.quantize.padding import pad_to_block

    mxfp8_block = 32 if block_size > 64 else block_size
    x_2d, _orig_m = pad_to_block(x_2d, mxfp8_block, dim=0)   # M axis
    x_2d, _orig_n = pad_to_block(x_2d, mxfp8_block, dim=-1)  # K axis
    return convert_to_mxfp8(
        x_2d, block_size=mxfp8_block, axis=-1, float8_dtype_pt=fp8_dtype,
    )
```

#### Why both dimensions?

`convert_to_mxfp8` reshapes its input to 2D `(M, N)` and asserts **both**
`M % block_size == 0` and `N % block_size == 0` (lines 186-187 of
`ops/quantize/ops.py`).  Padding only the last dimension is insufficient.

In the Megatron-Core path, M (= batch × seq_len) is pre-padded by
Megatron's `get_padding()`, so the `dim=0` pad hits the zero-copy fast path.
In standalone FSDP trainers without Megatron's sequence padding, the `dim=0`
pad provides the safety net.

#### Why no unpadding?

The MXFP8 `convert_to_mxfp8` returns `(x_quant, x_scales)` where
`x_scales.shape[-1] = x_quant.shape[-1] // block_size`.  The scale tensor's
shape is derived from the padded data shape.  If we unpadded `x_quant` but
not `x_scales`, they would be inconsistent.  If we unpadded both, we would
need to recompute the scale boundary — defeating the purpose.

Instead, the padded FP8 tensor flows into the GEMM.  In the TN-layout GEMM
`Y = X @ W^T` (where X is `(M, K)` and W is `(N, K)`):
- Padding on K does not affect output shape (output is M×N, determined by W).
- Padding on M adds extra rows to the output; these are zeroed-out padding
  rows that are masked away by the attention mask or loss mask at the output
  level.

This is consistent with TE/Megatron-Core's strategy: pad once, let padded
data flow through, handle validity via attention mask at the loss level.

**Important caveat:** K must match between X and W.  If activations are
padded on K, the weight quantization path must use the same effective K.
In practice, K = hidden_size (or hidden_size / tp_size) is architecture-
determined and typically a multiple of 32.  The `pad_to_block` on dim=-1
serves as a safety net for unusual configurations; for standard LLM
architectures this is a zero-copy pass-through.

#### Unpadding strategy by call site

| Call site | Pad dims | Unpad | Reason |
|-----------|----------|-------|--------|
| `_round_to_mxfp8` (quant→dequant round-trip) | N (dim=-1) | Yes (slice) | Output is high-precision; must restore original shape. M is assumed aligned (Megatron `get_padding` or architecture). |
| `quantize_input` mxfp8 (forward quantization) | M (dim=0) + K (dim=-1) | **No** | Padded FP8 + scales flow into GEMM; extra M rows masked by loss; K padding transparent to output shape |
| `_quant_blockwise_2d` (MoE weight) | K (dim=0) + N (dim=1) | Yes (slice) | Weight must match original `(K, N)` for GEMM dispatch |

### 3.4 Megatron-Core Integration Compatibility

Lumen integrates with Megatron-Core via module replacement (`apply_fp8_training`
→ `quant.enable`).  The padding responsibility splits across three dimensions:

| Dimension | Owner | Mechanism |
|-----------|-------|-----------|
| M (seq_len × batch) | Megatron-Core | `get_padding()` at initialization |
| K, N (hidden / intermediate) | Model architecture | Typically multiples of 128 for standard TP sizes |
| MXFP8 internal ops | Lumen (this feature) | `pad_to_block` at runtime |

`get_fp8_align_size` maps `scaling_type` → alignment the same way
Megatron-Core's `get_fp8_align_size(fp8_recipe)` does.  In the Megatron path,
sequences are pre-padded by Megatron-Core, so `pad_to_block` hits the
zero-copy fast path (already aligned).  In standalone FSDP trainers (no
Megatron-Core `get_padding`), `pad_to_block` is the safety net.

Future extension: `get_fp8_align_size` can accept additional parameters
(tp_size, sp, cp_size) to evolve into Lumen's equivalent of Megatron-Core's
`get_padding`.

## 4. File Changes

| File | Change type | Lines (est.) | Risk |
|------|------------|-------------|------|
| `lumen/ops/quantize/padding.py` (NEW) | New file | ~50 | Low |
| `lumen/quantize/scaling_manager.py` | Refactor `_round_to_mxfp8` | ~6 | Low |
| `lumen/ops/gemm/grouped_gemm.py` | Refactor `_quant_blockwise_2d` | ~8 | Low |
| `lumen/ops/quantize/linear.py` | Fix `quantize_input` mxfp8 (pad M+K) | ~10 | **Medium** |
| `tests/ops/test_padding.py` (NEW) | New test file | ~80 | Low |
| Existing test files | Add regression tests | ~20 | Low |
| **Total** | | **~170** | |

## 5. Testing

### 5.1 Unit Tests — `tests/ops/test_padding.py`

**`get_fp8_align_size`:**
- Returns 1 for `delayed`, `dynamic`, `per_token`, `none`
- Returns `block_size` for `blockwise`, `blockwise2d`
- Returns 32 for `mxfp8` with `block_size=128` (the default)
- Returns `block_size` for `mxfp8` with `block_size` ∈ {32, 64}
- Raises `ValueError` for unknown scaling_type

**`pad_to_block`:**
- Already-aligned tensor → zero-copy (`padded is tensor`)
- Unaligned tensor → correct padded shape, padded region is zero
- Various `dim` values (0, 1, -1) on 2D, 3D, 4D tensors
- `align_size=1` → always zero-copy
- Returned `orig_size` matches input size
- `align_size` must be ≥ 1 (document and guard)

### 5.2 Integration Regression Tests

- `_round_to_mxfp8` with non-aligned N shape produces same output as before
- `_quant_blockwise_2d` round-trip produces same `(w_fp8, w_scales)` as before
- `quantize_input("mxfp8", ...)` with non-aligned K (dim=-1) no longer raises
- `quantize_input("mxfp8", ...)` with non-aligned M (dim=0) no longer raises
- Returned `(x_quant, x_scales)` have consistent shapes (scale count matches
  data block count)

### 5.3 GEMM Smoke Test (if GPU available)

- Small forward pass with intentionally unaligned K and aligned M through
  `quantize_input("mxfp8") → gemm_mxfp8`, comparing output to BF16 reference
  within expected FP8 tolerance

## 6. Out of Scope

- `quantize_block_mxfp8` thd 2D block buffer pre-allocation — left as-is
- `quant_fp8_blockwise_impl` — Triton kernel handles partial blocks via `cdiv`
- Fused pipeline overlap FP8 — deferred to pipeline overlap FP8 phase
- Sequence-level padding (Megatron-Core `get_padding` equivalent) — separate feature
