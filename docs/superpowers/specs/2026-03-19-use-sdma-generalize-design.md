# `--use-sdma` Generalization — Design Spec

**Date**: 2026-03-19
**Scope**: Promote `--use-sdma` from FP8-amax-only flag to a general SDMA
switch that gates all supported SDMA communication paths.

---

## Problem

`--use-sdma` is defined under the `linear-fp8` arg group in `megatron.py` with
help text "Use mori SDMA for amax all-reduce instead of torch.distributed."
This is too narrow:

1. **CP A2A never uses SDMA via CLI** — `attention_megatron.py` and
   `attention_mla.py` build `cp_param_bundle` without `"use_sdma"`, so CP A2A
   always falls back to `torch.distributed.all_to_all_single` even when
   `--use-sdma` is set. The SDMA A2A path (`SdmaAll2all`) exists and is tested,
   but unreachable from training scripts.

2. **`get_scale` amax reduction ignores SDMA** — `ScalingManager.get_scale`
   does per-step `all_reduce(MAX)` via `torch.distributed` for both delayed and
   dynamic scaling. The batch amax path (`quantize_fp8_params`) already
   switches on `_use_sdma`, but the per-tensor path does not.

3. **FSDP trainers lack `--use-sdma`** — Neither `add_common_fsdp_args` nor
   the individual trainer arg parsers define `--use-sdma`, so FSDP runs cannot
   opt into SDMA at all.

---

## Design

### 1. Move `--use-sdma` to `lumen` arg group + update help

**`lumen/models/megatron.py`**:

- Remove `--use-sdma` from `lfp8` (linear-fp8) arg group.
- Add it to `lumen` arg group (alongside `--lumen-tp-comm-overlap-method`).
- New help text:

```
"Use mori SDMA instead of torch.distributed for supported collectives "
"(TP comm, amax all-reduce, CP all-to-all) when available."
```

**`lumen/models/fsdp.py`** — `add_common_fsdp_args`:

- Add `--use-sdma` (`store_true`, `default=False`) after the FSDP arg group,
  with the same help text.

**`lumen/models/llama2/fsdp/sft.py`** and **`lumen/models/llama31/fsdp/pretrain.py`**:

- These trainers define their own argparsers (they don't call
  `add_common_fsdp_args`). Add `--use-sdma` to each trainer's FSDP arg group.

### 2. Wire CP A2A SDMA via `--use-sdma`

**`lumen/modules/attention_megatron.py`** (line ~174-181):

```python
cp_param_bundle = {
    "cp_group": cp_group,
    "cp_comm_type": cp_comm_type,
    "use_sdma": _use_sdma_from_args(),   # NEW
}
```

**`lumen/modules/attention_mla.py`** (line ~148-155):

Same change.

Both files import `_use_sdma_from_args` from `lumen.modules.parallel_linear`.
This function calls `megatron.training.get_args()` and returns
`getattr(args, "use_sdma", False)`, falling back to `False` when Megatron args
are unavailable. This means **HF/FSDP-only stacks** that never install
Megatron args are unaffected (SDMA stays off), which is correct since those
trainers get their own `--use-sdma` flag wired through `ScalingManager`.

**Downstream effect**: `attention.py` already reads
`cp_param_bundle.get("use_sdma", False) and is_sdma_available()` and routes to
`SdmaAll2all` when True. No changes needed in `attention.py` or
`attention_with_cp_a2a.py`.

### 3. Add SDMA path to `ScalingManager.get_scale`

**`lumen/quantize/scaling_manager.py`** — `get_scale` method:

Currently (delayed and dynamic branches):

```python
if self.config.reduce_amax and self._dp_group is not None:
    torch.distributed.all_reduce(
        amax, op=torch.distributed.ReduceOp.MAX, group=self._dp_group,
    )
```

Replace with:

```python
if self.config.reduce_amax and self._dp_group is not None:
    if self._use_sdma:
        amax = self._reduce_single_amax_sdma(amax)
    else:
        torch.distributed.all_reduce(
            amax, op=torch.distributed.ReduceOp.MAX, group=self._dp_group,
        )
```

Add a new private method `_reduce_single_amax_sdma`:

```python
def _reduce_single_amax_sdma(self, amax: torch.Tensor) -> torch.Tensor:
    """Reduce a single amax scalar via SDMA (allgather + max)."""
    from lumen.ops.sdma import sdma_allgather_max

    packed = amax.float().unsqueeze(0)  # sdma_allgather_max requires float32
    if self._sdma_allgather is None:
        from lumen.ops.sdma import SdmaAllgather
        self._sdma_allgather = SdmaAllgather()
    result = sdma_allgather_max(packed, self._sdma_allgather)
    return result[0]
```

Notes:

- `SdmaAllgather()` takes no size argument; its internal `_ensure_handle`
  allocates/reuses based on element count at call time.
- This reuses the existing `self._sdma_allgather` handle from
  `_reduce_fp8_amax_sdma`. The handle transparently supports different element
  counts (1 element here vs batch in `quantize_fp8_params`) because
  `_ensure_handle` reallocates when `n_elems > capacity` and reuses otherwise.
- `sdma_allgather_max` requires **float32** input. The `.float()` cast ensures
  correctness when `amax` originates from a BF16 tensor.

---

## Files changed

| File | Change |
|------|--------|
| `lumen/models/megatron.py` | Move `--use-sdma` from `lfp8` to `lumen` group, update help |
| `lumen/models/fsdp.py` | Add `--use-sdma` to `add_common_fsdp_args` |
| `lumen/models/llama2/fsdp/sft.py` | Add `--use-sdma` to trainer argparser |
| `lumen/models/llama31/fsdp/pretrain.py` | Add `--use-sdma` to trainer argparser |
| `lumen/modules/attention_megatron.py` | Add `"use_sdma"` to `cp_param_bundle` |
| `lumen/modules/attention_mla.py` | Same |
| `lumen/quantize/scaling_manager.py` | Add SDMA branch in `get_scale`, add `_reduce_single_amax_sdma` |

## Tests

| What | File |
|------|------|
| `cp_param_bundle` contains `"use_sdma"` when flag is set (megatron attn) | `tests/module/test_attention_megatron_module.py` |
| `cp_param_bundle` contains `"use_sdma"` when flag is set (MLA attn) | `tests/module/test_attention_mla_module.py` |
| `get_scale` uses SDMA path when `_use_sdma=True` (mock); verify float32 cast | `tests/quantize/test_scaling_manager.py` |
| `--use-sdma` parseable in FSDP trainers | `tests/models/test_fsdp.py` |
| Existing SDMA tests still pass | `tests/ops/test_sdma.py` |

## Out of scope

- FSDP2 DP all-gather / reduce-scatter via SDMA (deferred per FSDP2 spec)
- New SDMA primitives (all existing)
- `--lumen-tp-comm-overlap-method` changes (its `"auto"` logic already respects
  `--use-sdma`)
- FSDP trainer metric sync `all_reduce(AVG)` — SDMA only supports SUM, not AVG
