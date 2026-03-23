# MoE Fused TopK / Aux Loss -- Design Spec

**Date:** 2026-03-23
**Status:** Draft
**Scope:** Practical subset -- softmax top-k + batch-level aux loss (Mixtral/DeepSeek configs)

---

## Problem

Lumen's MoE routing is partially implemented. The existing `fused_topk` in `fused_routing.py` wraps AITER ASM `topk_softmax` but has a narrow API (`[N, k]` output, softmax-only, no autograd). Megatron-Core's `TopKRouter` uses three TE fused ops when `moe_router_fusion=True`:

1. `fused_topk_with_score_function` -- softmax + topk + scatter to `[N, E]` dense format with custom backward
2. `fused_compute_score_for_moe_aux_loss` -- score recomputation for aux loss with custom backward
3. `fused_moe_aux_loss` -- Switch load-balancing loss with custom backward

Without TE, Megatron-Core falls back to unfused PyTorch ops. Lumen provides no replacement, so `moe_router_fusion=True` cannot work on AMD GPUs. The `--lumen-fused-moe-routing` flag exists in the CLI but is never read.

## Goal

Implement Lumen-native replacements for all three TE fused router ops. Monkey-patch them into Megatron-Core's `moe_utils` module so `--moe-router-fusion` works end-to-end without TE on AMD GPUs. Delete the dead `--lumen-fused-moe-routing` flag.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope | Practical subset: softmax score function, standard top-k, batch-level aux loss | Covers Mixtral, DeepSeek-V2/V3. Sigmoid, group routing, expert bias deferred. |
| Integration | Monkey-patch TE extension imports in `moe_utils` | Follows Lumen's established `LumenSpecProvider` pattern. Zero Megatron-Core changes. |
| Kernel strategy | All bottom-layer compute goes through AITER (HIP + Triton) | Components 1 & 2: reuse existing HIP `moeSoftmax` + `moeTopK` via new pybind `softmax_topk`. Component 3: new AITER Triton `moe_aux_loss`. Lumen provides autograd glue only. |
| Output format | Convert `[N, k]` to `[N, E]` inside fused autograd function | Maintains Megatron-Core API compatibility. Scatter cost is negligible. |
| Dead flag | Delete `--lumen-fused-moe-routing` from `megatron.py` and `fsdp.py` | Never wired, replaced by standard `--moe-router-fusion`. |

## Architecture

### AITER kernel surface

All three components call AITER kernels at the bottom layer. Lumen owns only the `torch.autograd.Function` glue and monkey-patch wiring.

| Component | AITER kernel | Status | Location |
|-----------|-------------|--------|----------|
| Component 1 | `softmax_topk` (HIP -- reuses `moeSoftmax` + `moeTopK`) | **New pybind** | `csrc/kernels/topk_softmax_kernels.cu` (existing code) + new binding in `rocm_ops.hpp` |
| Component 2 | `softmax_topk` (HIP -- shared with Component 1) | **New pybind** (same) | same |
| Component 3 | `moe_aux_loss_fwd` / `moe_aux_loss_bwd` (Triton) | **New kernel** | `aiter/ops/triton/moe/moe_aux_loss.py` |

**Kernel reuse rationale:** AITER's `topk_softmax_kernels.cu` already contains two standalone device kernels -- `moeSoftmax` (full row-wise `[N, E]` softmax) and `moeTopK` (top-k selection from softmax output). In the existing `topk_softmax` entry point, these run as the fallback path when `num_experts` is not in `{1,2,4,8,...,512}` (the `default` switch branch). For power-of-two expert counts, the fused `topkGatingSoftmax` kernel runs instead -- but that fused kernel does NOT materialize full `[N, E]` softmax scores, which Components 1 & 2 need for backward.

The new `softmax_topk` binding always calls `moeSoftmax` then `moeTopK`, regardless of expert count. This means:
- For pow2 E (8, 64, 128): `softmax_topk` takes a different code path than `topk_softmax` (split vs fused). Both compute mathematically identical softmax + top-k, but numerical results may differ at float32 epsilon level due to different reduction order. Tests should compare against PyTorch reference, not against `topk_softmax`.
- For non-pow2 E: `softmax_topk` uses the exact same kernels as `topk_softmax`'s fallback.

The existing `topk_softmax` (fused path for inference, no autograd) is unchanged. The new `softmax_topk` serves the training autograd path only.

#### New AITER pybind: `softmax_topk`

Exposes existing internal HIP kernels `moeSoftmax` + `moeTopK` as a single Python-callable function.

**C++ wrapper** (new function in `topk_softmax_kernels.cu` or adjacent file):

```cpp
void softmax_topk(
    torch::Tensor& scores,              // [N, E] float32, output: full softmax probs
    torch::Tensor& topk_weights,        // [N, k] float32, output: top-k softmax values
    torch::Tensor& topk_indices,        // [N, k] int32, output: top-k expert indices
    torch::Tensor& token_expert_indices, // [N, k] int32, output: routing metadata
    const torch::Tensor& gating_output, // [N, E] input logits
    int k,
    bool need_renorm
);
```

Implementation:
1. Allocate `scores` as `[N, E]` float32 if not provided
2. Call existing `moeSoftmax<DTYPE, 256><<<N, 256, 0, stream>>>(gating_output, nullptr, scores, E)` -- writes full softmax to `scores`
3. Call existing `moeTopK<256><<<N, 256, 0, stream>>>(scores, nullptr, topk_weights, topk_indices, token_expert_indices, E, k, 0, E, need_renorm)` -- reads `scores`, writes top-k outputs
4. Return: caller gets both `scores [N, E]` (full softmax) and `topk_indices [N, k]`

**Python API:**

```python
def softmax_topk(
    gating_output: torch.Tensor,  # [N, E] float32 (caller casts bf16->fp32)
    k: int,                       # 1 <= k <= E
    need_renorm: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full softmax + top-k selection, reusing existing HIP kernels.

    Compute dtype: float32 (matches internal moeSoftmax).
    Input must be contiguous in the last dimension (stride(0) == E).
    Tie-breaking: iterative argmax (matches existing moeTopK behavior, may differ from torch.topk on ties).

    Edge cases:
    - N=0: returns empty tensors [0, E], [0, k], [0, k]
    - k=E: all experts selected
    - E=1: trivial softmax

    Returns:
        scores: [N, E] float32, full softmax probabilities
        topk_weights: [N, k] float32, top-k values (renormed if need_renorm=True)
        topk_indices: [N, k] int32, top-k expert indices
    """
```

**Pybind registration** (add to `MOE_OP_PYBIND` in `rocm_ops.hpp`):

```cpp
m.def("softmax_topk", &aiter::softmax_topk,
      py::arg("scores"), py::arg("topk_weights"),
      py::arg("topk_indices"), py::arg("token_expert_indices"),
      py::arg("gating_output"), py::arg("k"), py::arg("need_renorm"),
      "Full softmax + top-k via moeSoftmax + moeTopK.");
```

**Key constraints:**
- **Contiguity:** Input must be contiguous with `stride(0) == num_experts` (kernel indexes as `blockIdx.x * num_cols + c`). Non-contiguous inputs require `.contiguous()` before calling.
- **N=0 guard:** The C++ wrapper must check `num_tokens == 0` and return empty output tensors without launching any kernels (HIP does not allow zero-grid launches).

#### New AITER Triton kernel: `moe_aux_loss_fwd` / `moe_aux_loss_bwd`

Fuses the entire Switch load-balancing loss into one kernel launch each for forward and backward.

```python
def moe_aux_loss_fwd(
    probs: torch.Tensor,            # [N, E] float32, same device as output
    tokens_per_expert: torch.Tensor, # [E] float32, same device as probs
    coeff_scaled: float,             # Python float: E * coeff / (topk * T * T)
) -> torch.Tensor:
    """Fused column-sum + dot-product + scalar multiply.

    Edge cases:
    - N=0: returns 0-dim tensor with value 0.0
    - E=1: single-expert loss (trivial but valid)

    Returns: 0-dim float32 tensor on same device as probs.
    """

def moe_aux_loss_bwd(
    tokens_per_expert: torch.Tensor, # [E] float32
    coeff_scaled: float,             # same Python float from forward
    grad_aux_loss: torch.Tensor,     # 0-dim float32 tensor
    num_tokens: int,                 # N, must match probs.shape[0] from forward
    num_experts: int,                # E, must match probs.shape[1] from forward
) -> torch.Tensor:
    """Broadcast multiply: grad_probs[i,j] = tokens_per_expert[j] * C * grad_aux_loss.

    Edge cases:
    - num_tokens=0: returns empty [0, E] tensor

    Returns: [N, E] float32 grad_probs on same device as tokens_per_expert.
    """
```

Forward kernel: parallel column-wise reduction over `N` rows (each thread block handles a subset of experts), then dot-product + scalar multiply. Backward kernel: simple broadcast of `tokens_per_expert * C * grad_aux_loss` to `[N, E]`.

### New file: `lumen/ops/moe/fused_router.py`

Three `torch.autograd.Function` subclasses + three public API functions, all in one file. They share AITER backend access and format conversion logic. Imports: AITER HIP `softmax_topk` (new pybind, Components 1 & 2) and AITER Triton `moe_aux_loss_*` (new kernel, Component 3) via probe-guarded imports (same pattern as existing AITER bindings in `dispatch.py`).

### Component 1: `LumenFusedTopkWithScoreFunction`

Replaces `te.router.fused_topk_with_score_function`.

**Forward -- named tensors:**

The forward computes three named intermediate tensors:

1. **`s [N, E]`** -- full softmax: `s = softmax(logits.float(), dim=-1)`
2. **`s_k [N, k]`** -- gathered top-k softmax masses: `s_k = s.gather(dim=1, topk_indices)`
3. **`w [N, k]`** -- renormalized routing weights: `w = s_k / s_k.sum(dim=-1, keepdim=True)`

**Bottom-layer kernel:** AITER HIP `softmax_topk(gating_output, k, need_renorm=True)` reuses existing `moeSoftmax` + `moeTopK` kernels. Returns `(s [N, E], w [N, k], topk_indices [N, k])` -- full softmax, renormalized top-k weights, and indices in two HIP kernel launches.

**Forward steps:**

1. **`use_pre_softmax=True` (common path):** Call AITER `softmax_topk(logits.float(), topk, need_renorm=True)` -> `(s [N, E], w [N, k], topk_indices [N, k])`
2. `s_k = s.gather(dim=1, topk_indices.long())` -- gather selected scores (needed for backward)
3. `V = s_k.sum(dim=-1, keepdim=True)` -- save renorm denominator for backward
5. **`use_pre_softmax=False`:** Use pure PyTorch -- `topk` on raw logits to get `topk_indices`, then `softmax(gathered_logits, dim=-1)` over the k selected values. This branch delegates to `torch.autograd` (no custom backward) for the first version.
6. Scatter `w [N, k]` -> `routing_probs [N, E]` dense format (zeros for non-selected experts)
7. Build `routing_map [N, E]` boolean mask via `scatter` from `topk_indices`
8. Save `s`, `s_k`, `topk_indices`, and `V` for backward

**Backward (custom, for `use_pre_softmax=True` only):**

The derivative chain has three stages, working backward through `logits -> s -> s_k -> w`:

1. **Renorm backward** (`w = s_k / V` where `V = s_k.sum(-1, keepdim=True)`):
   `grad_s_k = (grad_w - (grad_w * w).sum(dim=-1, keepdim=True)) / V`
   (standard `x / x.sum()` backward -- same shape as softmax JVP over the k-simplex)

2. **Gather backward** (scatter `grad_s_k` from `[N, k]` into `grad_s [N, E]`):
   `grad_s = zeros(N, E).scatter_add(dim=1, index=topk_indices.long(), src=grad_s_k)`
   (Use `scatter_add` with `dim=1` not plain `scatter` for correctness if indices overlap;
   standard `topk` produces unique indices per row, but `scatter_add` is defensive.)

3. **Softmax backward** (full softmax over E experts, using saved `s` from forward):
   `grad_logits = s * (grad_s - (grad_s * s).sum(dim=-1, keepdim=True))`

All three stages use PyTorch standard ops. AITER kernels are not needed for backward -- the ops are element-wise or reduce over small dimensions (k or E), and PyTorch's fused kernels handle them efficiently. Component 3's backward IS an AITER Triton kernel because it broadcasts over the full `[N, E]` output.

If `scaling_factor` is applied, multiply `w` by it before scatter, and scale `grad_w` accordingly.

**Scope limits:**

- `score_function="sigmoid"` -> `NotImplementedError`
- `num_groups` / `group_topk` non-None/non-zero -> `NotImplementedError`
- `expert_bias` must be None
- `scaling_factor`: always a scalar float (or None); applied as simple multiply
- `use_pre_softmax=False`: implemented via pure PyTorch ops (no custom backward); `torch.autograd` handles gradients for this branch

### Component 2: `LumenFusedComputeScoreForMoEAuxLoss`

Replaces `te.router.fused_compute_score_for_moe_aux_loss`.

This computes **raw softmax scores** (not renormalized) for the aux loss path. Megatron-Core's unfused reference (`compute_routing_scores_for_aux_loss` in `moe_utils`) does: softmax -> topk to get routing_map -> return `(routing_map, scores)`. The full `[N, E]` softmax scores are returned (not just the top-k slice), and no renormalization is applied -- this differs from Component 1.

**Bottom-layer kernel:** AITER HIP `softmax_topk(gating_output, k, need_renorm=False)` -- reuses existing `moeSoftmax` + `moeTopK` kernels. Returns full `[N, E]` softmax scores and top-k indices.

**Forward:**

1. Call AITER `softmax_topk(logits.float(), topk, need_renorm=False)` -> `(scores [N, E], _, topk_indices [N, k])` (discard topk_weights, not needed for aux loss path)
2. Build `routing_map [N, E]` boolean mask via scatter from `topk_indices`
3. Return `(routing_map, scores)` -- note: `scores` is the full softmax, not gathered/renormed
4. Save `scores` for backward

**Backward:**

- Receive `(None, grad_scores)` -- `routing_map` has no gradient (boolean)
- Standard full-softmax backward: `grad_logits = scores * (grad_scores - sum(grad_scores * scores, dim=-1, keepdim=True))`

### Component 3: `LumenFusedMoEAuxLoss`

Replaces `te.router.fused_moe_aux_loss`.

Must match Megatron-Core's `switch_load_balancing_loss_func` (unfused path) exactly. Reference formula from `megatron/core/transformer/moe/moe_utils.py`:

```
aggregated_probs_per_expert = probs.sum(dim=0)  # [E]
aux_loss = sum(aggregated_probs_per_expert * tokens_per_expert) * (E * coeff / (topk * T * T))
```

`tokens_per_expert` is treated as a constant w.r.t. `probs` -- it comes from `routing_map.sum(dim=0)` which is detached (no gradient flows through it).

**Bottom-layer kernel:** New AITER Triton `moe_aux_loss_fwd` / `moe_aux_loss_bwd` -- fuses column-wise reduction + dot product + scalar multiply into single kernel launches.

**Forward:**

- Inputs: `probs [N, E]`, `tokens_per_expert [E]`, `total_num_tokens (T)`, `num_experts (E)`, `topk`, `coeff`
- Compute `C = E * coeff / (topk * T * T)` in Python
- Call AITER `moe_aux_loss_fwd(probs, tokens_per_expert, C)` -> scalar `aux_loss`
- Save `C` and `tokens_per_expert` for backward

**Backward:**

- Call AITER `moe_aux_loss_bwd(tokens_per_expert, C, grad_aux_loss, N, E)` -> `grad_probs [N, E]`
- Kernel computes: `grad_probs[i, j] = tokens_per_expert[j] * C * grad_aux_loss`
- `tokens_per_expert` has no gradient (constant input)

### Public API

Three thin wrappers with signatures matching TE exactly:

```python
def fused_topk_with_score_function(
    logits, topk, use_pre_softmax, num_groups, group_topk,
    scaling_factor, score_function, expert_bias,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...

def fused_compute_score_for_moe_aux_loss(
    logits, topk, score_function,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...

def fused_moe_aux_loss(
    probs, tokens_per_expert, total_num_tokens, num_experts, topk, coeff,
) -> torch.Tensor:
    ...
```

## Wiring

### Monkey-patching in `lumen/models/megatron.py`

During Lumen initialization (where other TE replacements are already patched):

```python
import megatron.core.transformer.moe.moe_utils as moe_utils
from lumen.ops.moe.fused_router import (
    fused_topk_with_score_function,
    fused_compute_score_for_moe_aux_loss,
    fused_moe_aux_loss,
)

moe_utils.fused_topk_with_score_function = fused_topk_with_score_function
moe_utils.fused_compute_score_for_moe_aux_loss = fused_compute_score_for_moe_aux_loss
moe_utils.fused_moe_aux_loss = fused_moe_aux_loss
```

Also patch the same three symbols in `megatron.core.extensions.transformer_engine`:

```python
import megatron.core.extensions.transformer_engine as te_ext

te_ext.fused_topk_with_score_function = fused_topk_with_score_function
te_ext.fused_compute_score_for_moe_aux_loss = fused_compute_score_for_moe_aux_loss
te_ext.fused_moe_aux_loss = fused_moe_aux_loss
```

This covers both import paths: `moe_utils` binds from `te_ext` at import time, so patching both ensures correctness regardless of import order.

### Import binding safety

Megatron-Core's `moe_utils.py` uses module-level `from ... import` to bind these functions. The `fused=True` call sites in `moe_utils` call them via local names (e.g. `fused_moe_aux_loss(...)`) not via `moe_utils.fused_moe_aux_loss(...)`. Since these names are module-level globals in `moe_utils`, patching `moe_utils.fused_moe_aux_loss = ...` updates the binding correctly **as long as Lumen patches before any module captures the old reference**.

**Patch timing:** The monkey-patch must execute during Lumen initialization (in `_apply_lumen_args` or equivalent), which runs before `TopKRouter` is instantiated. This is the same discipline used by `LumenSpecProvider`.

**Known failure modes to guard against:**

1. **Early `from` binding:** If any module does `from moe_utils import fused_moe_aux_loss` before Lumen patches, it holds a stale reference. Implementation must verify no such early imports exist in the target Megatron-Core version.
2. **Default argument capture:** `def foo(fn=fused_moe_aux_loss):` captures at definition time. Rare but a real Python footgun -- verify no such patterns exist.
3. **`HAVE_TE` side effects:** The fused code paths check `fused_topk_with_score_function is not None`, not `HAVE_TE` directly. Setting `HAVE_TE = True` is **not required** and should be avoided to prevent activating unrelated TE gates. Implementation should verify the exact guards in the target Megatron-Core version.

**Megatron-Core version pinning:** The implementation PR must state which Megatron-Core commit these patches target, since import structure and guard logic may change across versions.

### Activation path

With `--moe-router-fusion`:

1. `TopKRouter.routing()` -> `topk_routing_with_score_function(..., fused=True)` -> Lumen's `fused_topk_with_score_function`
2. `compute_routing_scores_for_aux_loss(..., fused=True)` -> Lumen's `fused_compute_score_for_moe_aux_loss`
3. `switch_load_balancing_loss_func(..., fused=True)` -> Lumen's `fused_moe_aux_loss`

### Deletions

- Delete `--lumen-fused-moe-routing` argument from `lumen/models/megatron.py`
- Delete `--lumen-fused-moe-routing` argument from `lumen/models/fsdp.py`

## Testing

### New file: `tests/ops/test_moe_fused_router.py`

**Test 1: `test_fused_topk_with_score_function`**

- Reference: `topk_routing_with_score_function(logits, topk, ..., fused=False)` from `moe_utils`
- Lumen: `fused_topk_with_score_function(logits, topk, ...)`
- Forward check: `routing_probs` values match (rtol=1e-5 primary, SNR > 20 dB secondary), `routing_map` identical (boolean exact match)
- Backward check: `grad_logits` match (rtol=1e-4, SNR > 15 dB)
- Configs: `num_tokens` in {32, 512, 4096}, `num_experts` in {8, 64, 128}, `topk` in {1, 2, 8}, `use_pre_softmax` in {True, False}
- Tests that compare to `fused=False` reference must use the same `use_pre_softmax` flag

**Test 2: `test_fused_compute_score_for_moe_aux_loss`**

- Reference: `compute_routing_scores_for_aux_loss(logits, topk, "softmax", fused=False)`
- Lumen: `fused_compute_score_for_moe_aux_loss(logits, topk, "softmax")`
- Forward check: `scores` rtol=1e-5, `routing_map` exact match
- Backward check: `grad_logits` rtol=1e-4

**Test 3: `test_fused_moe_aux_loss`**

- Reference: unfused `switch_load_balancing_loss_func(probs, tokens_per_expert, T, topk, E, coeff, fused=False)`
- Lumen: `fused_moe_aux_loss(probs, tokens_per_expert, T, E, topk, coeff)`
- Forward check: scalar loss rtol=1e-5
- Backward check: `grad_probs` rtol=1e-5

**Test 4: `test_unsupported_score_function`**

- `score_function="sigmoid"` -> `NotImplementedError`
- `group_topk` non-zero -> `NotImplementedError`

**Test 5: `test_gradcheck`**

- Use `torch.autograd.gradcheck` on all three fused ops with small tensors (float64)
- Tests run on thin PyTorch wrappers that replicate the autograd function math (softmax+gather+renorm, softmax+topk, aux loss) in pure PyTorch float64 -- this validates the backward math independently of AITER kernels
- Component 1: must use `use_pre_softmax=True` with non-trivial `topk` to exercise custom backward
- AITER Triton kernels do not support float64; gradcheck targets the autograd logic, not the kernels

**Test 6: `test_end_to_end_patch_smoke`**

- Import Lumen's megatron init path, apply the monkey-patch
- Call all three entry points once on small tensors
- Verifies import-order correctness and patch wiring

**Test 7: `test_aiter_forward_parity`**

- **7a: `softmax_topk` HIP parity** -- When AITER is available, compare AITER HIP `softmax_topk(logits, k, need_renorm=False)` full softmax output against PyTorch `softmax` reference. Checks: `scores` match (rtol=1e-5). For `topk_indices`: compare as multiset per row (same expert set selected), not positional identity -- `moeTopK` uses iterative argmax which may differ from `torch.topk` tie-breaking. Test logits are seeded with distinct values to avoid ties in practice.
- **7b: `softmax_topk` renorm parity** -- Compare `softmax_topk(logits, k, need_renorm=True)` top-k weights against PyTorch `softmax -> topk -> renorm` reference. Same multiset comparison for indices.
- **7c: `topk_softmax` ASM parity** -- Compare existing AITER ASM `topk_softmax(need_renorm=True)` output against PyTorch reference. Confirms the inference-path kernel matches the mathematical `s, s_k, w` story.

**Test conventions:**

- Input logits: bf16 (cast to float32 internally, matching production path). Tests must not use only fp32.
- Seeded distinct logits (avoid equal values causing ambiguous topk tie-breaking)
- `scaling_factor`: tested with None and with a scalar float (e.g. 0.5)
- Primary checks: `rtol`/`atol` against unfused reference. SNR as secondary diagnostic.
- Single-process init order only -- distributed patch ordering is out of scope for unit tests.

**AITER guard:** Tests using AITER-accelerated path guarded with `pytest.mark.skipif(not _probe_aiter_moe_topk_softmax(), ...)`.

## Implementation Phases

Implementation is split into two phases because AITER changes are a prerequisite for Lumen.

### Phase 1: AITER (upstream prerequisite)

Changes land in AITER repo first. Lumen cannot proceed until these are merged and the AITER submodule is updated.

| File | Change |
|------|--------|
| `csrc/kernels/topk_softmax_kernels.cu` | Add `softmax_topk` C++ wrapper calling existing `moeSoftmax` + `moeTopK` |
| `csrc/include/rocm_ops.hpp` | Add `softmax_topk` pybind entry to `MOE_OP_PYBIND` |
| `aiter/ops/moe_op.py` | Add Python `softmax_topk` wrapper |
| `aiter/ops/triton/moe/moe_aux_loss.py` | **New** -- Triton MoE aux loss forward + backward kernels |
| `op_tests/test_softmax_topk.py` | **New** -- kernel-level tests for `softmax_topk` HIP binding |
| `op_tests/test_moe_aux_loss.py` | **New** -- kernel-level tests for Triton `moe_aux_loss` |

**AITER PR deliverables:**
1. `softmax_topk` pybind exposing existing `moeSoftmax` + `moeTopK` (C++ wrapper + Python API + tests)
2. `moe_aux_loss_fwd` / `moe_aux_loss_bwd` Triton kernels (new kernel + tests)
3. All AITER-level tests passing

### Phase 2: Lumen (depends on Phase 1)

After AITER submodule is updated with the new APIs, Lumen implements the autograd wrappers and monkey-patch wiring.

| File | Change |
|------|--------|
| `lumen/ops/moe/fused_router.py` | **New** -- three autograd functions + three public API wrappers |
| `lumen/ops/moe/__init__.py` | Re-export new public APIs |
| `lumen/ops/dispatch.py` | Add probes: `_probe_aiter_softmax_topk`, `_probe_aiter_triton_moe_aux_loss` |
| `lumen/models/megatron.py` | Monkey-patch `moe_utils` fused ops; delete `--lumen-fused-moe-routing` |
| `lumen/models/fsdp.py` | Delete `--lumen-fused-moe-routing` |
| `tests/ops/test_moe_fused_router.py` | **New** -- Lumen-level unit tests |

**Lumen PR deliverables:**
1. Autograd wrappers consuming AITER `softmax_topk` and `moe_aux_loss_*`
2. Monkey-patch wiring for Megatron-Core `moe_utils` + `transformer_engine` extension
3. Dead flag deletion (`--lumen-fused-moe-routing`)
4. All Lumen-level tests passing

No changes to Megatron-Core. Existing `fused_routing.py` and `fused_moe.py` unchanged.

## Future Work (out of scope)

- Sigmoid score function support
- Group routing (`num_groups` / `group_topk`)
- Expert bias for sigmoid routing
- Sequence-level and global aux loss variants
- z-loss integration (currently handled by Megatron-Core's unfused path, which continues to work)
- Triton softmax backward kernel in AITER (if profiling shows PyTorch softmax backward is a bottleneck)
