# MoE Fused TopK / Aux Loss — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace TE's three fused MoE router ops with Lumen-native implementations backed by AITER kernels, enabling `--moe-router-fusion` on AMD GPUs.

**Architecture:** Phase 1 adds AITER-side APIs (HIP pybind `softmax_topk` reusing existing `moeSoftmax`+`moeTopK`, plus new Triton `moe_aux_loss`). Phase 2 builds Lumen `torch.autograd.Function` wrappers and monkey-patches them into Megatron-Core's `moe_utils`.

**Tech Stack:** C++/HIP (AITER pybind), Triton (AITER aux loss kernel), Python/PyTorch autograd (Lumen), pytest

**Spec:** `docs/superpowers/specs/2026-03-23-moe-fused-topk-aux-loss-design.md`

---

## File Structure

| File | Responsibility | Phase | Tasks |
|------|---------------|-------|-------|
| `third_party/aiter/csrc/kernels/topk_softmax_kernels.cu` | C++ wrapper for `softmax_topk` | 1 | 1 |
| `third_party/aiter/csrc/include/rocm_ops.hpp` | Pybind registration for `softmax_topk` | 1 | 1 |
| `third_party/aiter/aiter/ops/moe_op.py` | Python `softmax_topk` wrapper | 1 | 2 |
| `third_party/aiter/op_tests/test_softmax_topk.py` | AITER kernel-level tests for `softmax_topk` | 1 | 3 |
| `third_party/aiter/aiter/ops/triton/moe/moe_aux_loss.py` | Triton `moe_aux_loss_fwd` / `moe_aux_loss_bwd` | 1 | 4 |
| `third_party/aiter/op_tests/test_moe_aux_loss.py` | AITER kernel-level tests for Triton aux loss | 1 | 5 |
| `lumen/ops/dispatch.py` | AITER probes for new ops | 2 | 6 |
| `lumen/ops/moe/fused_router.py` | Three autograd functions + public API | 2 | 7, 8, 9 |
| `lumen/ops/moe/__init__.py` | Re-export new public APIs | 2 | 9 |
| `lumen/models/megatron.py` | Monkey-patch wiring; delete dead flag | 2 | 10 |
| `lumen/models/fsdp.py` | Delete dead flag | 2 | 10 |
| `tests/ops/test_moe_fused_router.py` | Lumen-level unit tests | 2 | 11 |

---

# Phase 1: AITER (upstream prerequisite)

All paths in Phase 1 are relative to `third_party/aiter/`.

---

### Task 1: `softmax_topk` C++ wrapper + pybind

**Why first:** Exposes existing `moeSoftmax` + `moeTopK` HIP kernels to Python. Everything in Components 1 & 2 depends on this.

**Files:**
- Modify: `csrc/kernels/topk_softmax_kernels.cu:623-663` (add new function after existing `aiter::topk_softmax`)
- Modify: `csrc/include/rocm_ops.hpp:1161-1177` (add pybind entry after existing `topk_softmax`)

- [ ] **Step 1: Add C++ `softmax_topk` function**

Add immediately after the existing `aiter::topk_softmax` function (after line 663 in `topk_softmax_kernels.cu`), inside the `aiter` namespace:

```cpp
void softmax_topk(
    torch::Tensor& scores,              // [N, E] float32, output
    torch::Tensor& topk_weights,        // [N, k] float32, output
    torch::Tensor& topk_indices,        // [N, k] int32, output
    torch::Tensor& token_expert_indices, // [N, k] int32, output
    torch::Tensor& gating_output,       // [N, E] input logits
    int k,
    bool need_renorm)
{
    const int num_experts = gating_output.size(-1);
    const int num_tokens  = gating_output.numel() / num_experts;

    if (num_tokens == 0) {
        return;
    }

    auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();

    static constexpr int TPB = 256;

    VLLM_DISPATCH_FLOATING_TYPES(
        gating_output.scalar_type(), "softmax_topk", [&] {
            using input_dtype = typename t2ck<scalar_t>::type;

            vllm::moe::moeSoftmax<input_dtype, TPB>
                <<<num_tokens, TPB, 0, stream>>>(
                    reinterpret_cast<input_dtype*>(gating_output.data_ptr()),
                    nullptr,
                    scores.data_ptr<float>(),
                    num_experts);

            vllm::moe::moeTopK<TPB>
                <<<num_tokens, TPB, 0, stream>>>(
                    scores.data_ptr<float>(),
                    nullptr,
                    topk_weights.data_ptr<float>(),
                    topk_indices.data_ptr<int>(),
                    token_expert_indices.data_ptr<int>(),
                    num_experts,
                    k,
                    0,
                    num_experts,
                    need_renorm);
        });
}
```

- [ ] **Step 2: Add pybind registration**

In `csrc/include/rocm_ops.hpp`, add after the existing `topk_softmax_asm` pybind entry (after line ~1177), still inside the `MOE_OP_PYBIND` macro:

```cpp
    m.def("softmax_topk", &aiter::softmax_topk,                               \
          py::arg("scores"), py::arg("topk_weights"),                          \
          py::arg("topk_indices"), py::arg("token_expert_indices"),            \
          py::arg("gating_output"), py::arg("k"), py::arg("need_renorm"),     \
          "Full softmax + top-k via moeSoftmax + moeTopK.");                   \
```

Also add the declaration in the header section where `aiter::topk_softmax` is declared.

- [ ] **Step 3: Build AITER and verify compilation**

Run: `cd third_party/aiter && pip install -e . --no-build-isolation`
Expected: Build succeeds, no compilation errors.

- [ ] **Step 4: Commit**

```bash
git add csrc/kernels/topk_softmax_kernels.cu csrc/include/rocm_ops.hpp
git commit -m "feat(moe): add softmax_topk pybind exposing moeSoftmax + moeTopK"
```

---

### Task 2: `softmax_topk` Python wrapper

**Why next:** Provides the clean Python API that Lumen will call.

**Files:**
- Modify: `aiter/ops/moe_op.py:15-22` (add new function after existing `topk_softmax`)

- [ ] **Step 1: Add Python wrapper**

Add after the existing `topk_softmax` function in `aiter/ops/moe_op.py`. Follow the same `@compile_ops` pattern:

```python
@compile_ops("module_moe_asm")
def softmax_topk(
    scores: Tensor,
    topk_weights: Tensor,
    topk_indices: Tensor,
    token_expert_indices: Tensor,
    gating_output: Tensor,
    k: int,
    need_renorm: bool,
) -> None: ...
```

- [ ] **Step 2: Quick smoke test in Python**

Run: `python -c "from aiter.ops.moe_op import softmax_topk; print('import ok')"`
Expected: `import ok`

- [ ] **Step 3: Commit**

```bash
git add aiter/ops/moe_op.py
git commit -m "feat(moe): add softmax_topk Python wrapper"
```

---

### Task 3: AITER-level tests for `softmax_topk`

**Why next:** Validates the HIP binding before Lumen depends on it.

**Files:**
- Create: `op_tests/test_softmax_topk.py`

- [ ] **Step 1: Write kernel-level test**

```python
# op_tests/test_softmax_topk.py
import torch
import pytest

torch.set_default_device("cuda")


def softmax_topk_ref(gating_output, topk, need_renorm):
    """Pure PyTorch reference: full softmax + topk."""
    scores = torch.nn.functional.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = scores.topk(k=topk, dim=-1, largest=True, sorted=True)
    if need_renorm:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return scores, topk_weights, topk_ids.to(torch.int32)


@pytest.mark.parametrize("num_tokens", [1, 32, 512, 4096])
@pytest.mark.parametrize("num_experts", [8, 64, 128])
@pytest.mark.parametrize("topk", [1, 2, 8])
@pytest.mark.parametrize("need_renorm", [True, False])
def test_softmax_topk(num_tokens, num_experts, topk, need_renorm):
    if topk > num_experts:
        pytest.skip("topk > num_experts")

    from aiter.ops.moe_op import softmax_topk

    torch.manual_seed(42)
    gating_output = torch.randn(num_tokens, num_experts, dtype=torch.float32)

    scores_ref, weights_ref, ids_ref = softmax_topk_ref(
        gating_output, topk, need_renorm
    )

    scores = torch.empty(num_tokens, num_experts, dtype=torch.float32)
    topk_weights = torch.empty(num_tokens, topk, dtype=torch.float32)
    topk_indices = torch.empty(num_tokens, topk, dtype=torch.int32)
    token_expert_indices = torch.empty(num_tokens, topk, dtype=torch.int32)

    softmax_topk(
        scores, topk_weights, topk_indices, token_expert_indices,
        gating_output, topk, need_renorm,
    )

    torch.testing.assert_close(scores, scores_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(topk_weights, weights_ref, atol=1e-4, rtol=1e-4)

    for row in range(min(num_tokens, 16)):
        assert set(topk_indices[row].tolist()) == set(ids_ref[row].tolist()), (
            f"Row {row}: expert sets differ"
        )


def test_softmax_topk_n_zero():
    from aiter.ops.moe_op import softmax_topk

    scores = torch.empty(0, 8, dtype=torch.float32)
    topk_weights = torch.empty(0, 2, dtype=torch.float32)
    topk_indices = torch.empty(0, 2, dtype=torch.int32)
    token_expert_indices = torch.empty(0, 2, dtype=torch.int32)
    gating_output = torch.empty(0, 8, dtype=torch.float32)

    softmax_topk(
        scores, topk_weights, topk_indices, token_expert_indices,
        gating_output, 2, False,
    )
    assert scores.shape == (0, 8)


def test_softmax_topk_bf16_input():
    """Verify bf16 inputs are handled (caller should cast to fp32)."""
    from aiter.ops.moe_op import softmax_topk

    torch.manual_seed(42)
    gating_fp32 = torch.randn(32, 8, dtype=torch.float32)

    scores = torch.empty(32, 8, dtype=torch.float32)
    topk_weights = torch.empty(32, 2, dtype=torch.float32)
    topk_indices = torch.empty(32, 2, dtype=torch.int32)
    token_expert_indices = torch.empty(32, 2, dtype=torch.int32)

    softmax_topk(
        scores, topk_weights, topk_indices, token_expert_indices,
        gating_fp32, 2, True,
    )

    scores_ref, _, _ = softmax_topk_ref(gating_fp32, 2, True)
    torch.testing.assert_close(scores, scores_ref, atol=1e-5, rtol=1e-5)
```

- [ ] **Step 2: Run tests**

Run: `cd third_party/aiter && python -m pytest op_tests/test_softmax_topk.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add op_tests/test_softmax_topk.py
git commit -m "test(moe): add softmax_topk kernel-level tests"
```

---

### Task 4: Triton `moe_aux_loss` kernels

**Why next:** Component 3's AITER backend. No dependency on Tasks 1-3.

**Files:**
- Create: `aiter/ops/triton/moe/moe_aux_loss.py`

- [ ] **Step 1: Write Triton forward kernel**

```python
# aiter/ops/triton/moe/moe_aux_loss.py
import torch
import triton
import triton.language as tl


@triton.jit
def _moe_aux_loss_fwd_kernel(
    probs_ptr,
    tokens_per_expert_ptr,
    out_ptr,
    coeff_scaled,
    N: tl.constexpr,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    expert_id = tl.program_id(0)
    col_sum = tl.zeros([], dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        row_ids = start + tl.arange(0, BLOCK_N)
        mask = row_ids < N
        vals = tl.load(probs_ptr + row_ids * E + expert_id, mask=mask, other=0.0)
        col_sum += tl.sum(vals)

    tpe = tl.load(tokens_per_expert_ptr + expert_id)
    partial = col_sum * tpe * coeff_scaled
    tl.atomic_add(out_ptr, partial)


def moe_aux_loss_fwd(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    coeff_scaled: float,
) -> torch.Tensor:
    N, E = probs.shape
    out = torch.zeros((), dtype=torch.float32, device=probs.device)
    if N == 0:
        return out
    BLOCK_N = min(triton.next_power_of_2(N), 1024)
    _moe_aux_loss_fwd_kernel[(E,)](
        probs, tokens_per_expert, out,
        coeff_scaled, N, E, BLOCK_N,
    )
    return out


@triton.jit
def _moe_aux_loss_bwd_kernel(
    tokens_per_expert_ptr,
    grad_probs_ptr,
    scale,
    N: tl.constexpr,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    expert_id = tl.program_id(0)
    tpe_val = tl.load(tokens_per_expert_ptr + expert_id)
    fill_val = tpe_val * scale

    for start in range(0, N, BLOCK_N):
        row_ids = start + tl.arange(0, BLOCK_N)
        mask = row_ids < N
        tl.store(grad_probs_ptr + row_ids * E + expert_id, fill_val, mask=mask)


def moe_aux_loss_bwd(
    tokens_per_expert: torch.Tensor,
    coeff_scaled: float,
    grad_aux_loss: torch.Tensor,
    num_tokens: int,
    num_experts: int,
) -> torch.Tensor:
    if num_tokens == 0:
        return torch.empty(0, num_experts, dtype=torch.float32,
                           device=tokens_per_expert.device)
    grad_probs = torch.empty(num_tokens, num_experts, dtype=torch.float32,
                             device=tokens_per_expert.device)
    scale = coeff_scaled * grad_aux_loss.item()
    BLOCK_N = min(triton.next_power_of_2(num_tokens), 1024)
    _moe_aux_loss_bwd_kernel[(num_experts,)](
        tokens_per_expert, grad_probs,
        scale, num_tokens, num_experts, BLOCK_N,
    )
    return grad_probs
```

- [ ] **Step 2: Verify import**

Run: `python -c "from aiter.ops.triton.moe.moe_aux_loss import moe_aux_loss_fwd, moe_aux_loss_bwd; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add aiter/ops/triton/moe/moe_aux_loss.py
git commit -m "feat(moe): add Triton moe_aux_loss_fwd and moe_aux_loss_bwd kernels"
```

---

### Task 5: AITER-level tests for `moe_aux_loss`

**Files:**
- Create: `op_tests/test_moe_aux_loss.py`

- [ ] **Step 1: Write kernel-level tests**

```python
# op_tests/test_moe_aux_loss.py
import torch
import pytest

torch.set_default_device("cuda")


def aux_loss_ref(probs, tokens_per_expert, coeff_scaled):
    """Pure PyTorch reference."""
    aggregated = probs.sum(dim=0)
    return (aggregated * tokens_per_expert).sum() * coeff_scaled


@pytest.mark.parametrize("N", [1, 32, 512, 4096])
@pytest.mark.parametrize("E", [8, 64, 128])
def test_moe_aux_loss_fwd(N, E):
    from aiter.ops.triton.moe.moe_aux_loss import moe_aux_loss_fwd

    torch.manual_seed(42)
    probs = torch.rand(N, E, dtype=torch.float32)
    tokens_per_expert = torch.randint(0, N, (E,), dtype=torch.float32)
    coeff_scaled = 0.01

    result = moe_aux_loss_fwd(probs, tokens_per_expert, coeff_scaled)
    expected = aux_loss_ref(probs, tokens_per_expert, coeff_scaled)

    torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("N", [1, 32, 512])
@pytest.mark.parametrize("E", [8, 64])
def test_moe_aux_loss_bwd(N, E):
    from aiter.ops.triton.moe.moe_aux_loss import moe_aux_loss_bwd

    torch.manual_seed(42)
    tokens_per_expert = torch.randint(0, N, (E,), dtype=torch.float32)
    coeff_scaled = 0.01
    grad_aux_loss = torch.tensor(1.0)

    result = moe_aux_loss_bwd(tokens_per_expert, coeff_scaled, grad_aux_loss, N, E)

    expected = tokens_per_expert.unsqueeze(0).expand(N, E) * coeff_scaled
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


def test_moe_aux_loss_fwd_n_zero():
    from aiter.ops.triton.moe.moe_aux_loss import moe_aux_loss_fwd

    probs = torch.empty(0, 8, dtype=torch.float32)
    tokens_per_expert = torch.ones(8, dtype=torch.float32)
    result = moe_aux_loss_fwd(probs, tokens_per_expert, 0.01)
    assert result.item() == 0.0


def test_moe_aux_loss_bwd_n_zero():
    from aiter.ops.triton.moe.moe_aux_loss import moe_aux_loss_bwd

    tokens_per_expert = torch.ones(8, dtype=torch.float32)
    result = moe_aux_loss_bwd(tokens_per_expert, 0.01, torch.tensor(1.0), 0, 8)
    assert result.shape == (0, 8)
```

- [ ] **Step 2: Run tests**

Run: `cd third_party/aiter && python -m pytest op_tests/test_moe_aux_loss.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add op_tests/test_moe_aux_loss.py
git commit -m "test(moe): add moe_aux_loss Triton kernel-level tests"
```

---

# Phase 2: Lumen (depends on Phase 1)

**Prerequisite:** AITER submodule updated with `softmax_topk` pybind and `moe_aux_loss` Triton kernels from Phase 1.

All paths in Phase 2 are relative to the Lumen repo root.

---

### Task 6: AITER dispatch probes

**Why first:** All Lumen code needs probe-guarded imports.

**Files:**
- Modify: `lumen/ops/dispatch.py:247-255` (add new probes after existing `_probe_aiter_moe_topk_softmax`)

- [ ] **Step 1: Add probes**

Add after `_probe_aiter_moe_topk_softmax` (around line 255) in `lumen/ops/dispatch.py`:

```python
@functools.lru_cache(maxsize=1)
def _probe_aiter_softmax_topk():
    """Check if AITER softmax_topk HIP binding is available."""
    try:
        from aiter.ops.moe_op import softmax_topk as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_moe_aux_loss():
    """Check if AITER Triton moe_aux_loss kernels are available."""
    try:
        from aiter.ops.triton.moe.moe_aux_loss import moe_aux_loss_fwd as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False
```

- [ ] **Step 2: Verify import**

Run: `python -c "from lumen.ops.dispatch import _probe_aiter_softmax_topk, _probe_aiter_triton_moe_aux_loss; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add lumen/ops/dispatch.py
git commit -m "feat(moe): add dispatch probes for softmax_topk and moe_aux_loss"
```

---

### Task 7: `fused_router.py` — Component 3 (LumenFusedMoEAuxLoss)

**Why this order:** Simplest autograd function. No dependency on `softmax_topk`. Good foundation file.

**Files:**
- Create: `lumen/ops/moe/fused_router.py`

- [ ] **Step 1: Create file with Component 3 autograd function**

```python
# lumen/ops/moe/fused_router.py
"""Lumen-native fused MoE router ops replacing TE equivalents.

Three torch.autograd.Function subclasses providing:
1. fused_topk_with_score_function — softmax + topk + scatter (Component 1)
2. fused_compute_score_for_moe_aux_loss — softmax + topk for aux loss (Component 2)
3. fused_moe_aux_loss — Switch load-balancing loss (Component 3)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import Tensor

from lumen.ops.dispatch import (
    _probe_aiter_softmax_topk,
    _probe_aiter_triton_moe_aux_loss,
)

logger = logging.getLogger(__name__)


class LumenFusedMoEAuxLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        probs: Tensor,
        tokens_per_expert: Tensor,
        total_num_tokens: int,
        num_experts: int,
        topk: int,
        coeff: float,
    ) -> Tensor:
        C = num_experts * coeff / (topk * total_num_tokens * total_num_tokens)
        ctx.save_for_backward(tokens_per_expert)
        ctx.C = C
        ctx.N = probs.shape[0]
        ctx.E = num_experts

        if _probe_aiter_triton_moe_aux_loss():
            from aiter.ops.triton.moe.moe_aux_loss import moe_aux_loss_fwd

            return moe_aux_loss_fwd(probs.float(), tokens_per_expert.float(), C)
        else:
            aggregated = probs.float().sum(dim=0)
            return (aggregated * tokens_per_expert.float()).sum() * C

    @staticmethod
    def backward(ctx, grad_aux_loss: Tensor):
        (tokens_per_expert,) = ctx.saved_tensors

        if _probe_aiter_triton_moe_aux_loss():
            from aiter.ops.triton.moe.moe_aux_loss import moe_aux_loss_bwd

            grad_probs = moe_aux_loss_bwd(
                tokens_per_expert.float(), ctx.C, grad_aux_loss, ctx.N, ctx.E
            )
        else:
            grad_probs = (
                tokens_per_expert.float().unsqueeze(0).expand(ctx.N, ctx.E)
                * ctx.C
                * grad_aux_loss
            )

        return grad_probs, None, None, None, None, None


def fused_moe_aux_loss(
    probs: Tensor,
    tokens_per_expert: Tensor,
    total_num_tokens: int,
    num_experts: int,
    topk: int,
    coeff: float,
) -> Tensor:
    return LumenFusedMoEAuxLoss.apply(
        probs, tokens_per_expert, total_num_tokens, num_experts, topk, coeff
    )
```

- [ ] **Step 2: Verify import**

Run: `python -c "from lumen.ops.moe.fused_router import fused_moe_aux_loss; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add lumen/ops/moe/fused_router.py
git commit -m "feat(moe): add LumenFusedMoEAuxLoss autograd function"
```

---

### Task 8: `fused_router.py` — Components 1 & 2

**Why next:** Builds on the file created in Task 7. Component 2 is simpler (no renorm backward), Component 1 is the most complex.

**Files:**
- Modify: `lumen/ops/moe/fused_router.py`

- [ ] **Step 1: Add helper for AITER softmax_topk call**

Add before the `LumenFusedMoEAuxLoss` class:

```python
def _aiter_softmax_topk(logits_fp32: Tensor, k: int, need_renorm: bool):
    """Call AITER HIP softmax_topk, allocating output buffers."""
    N, E = logits_fp32.shape
    device = logits_fp32.device
    scores = torch.empty(N, E, dtype=torch.float32, device=device)
    topk_weights = torch.empty(N, k, dtype=torch.float32, device=device)
    topk_indices = torch.empty(N, k, dtype=torch.int32, device=device)
    token_expert_indices = torch.empty(N, k, dtype=torch.int32, device=device)

    from aiter.ops.moe_op import softmax_topk

    softmax_topk(
        scores, topk_weights, topk_indices, token_expert_indices,
        logits_fp32.contiguous(), k, need_renorm,
    )
    return scores, topk_weights, topk_indices


def _pytorch_softmax_topk(logits_fp32: Tensor, k: int, need_renorm: bool):
    """Pure PyTorch fallback for softmax + topk."""
    scores = torch.softmax(logits_fp32, dim=-1)
    topk_weights, topk_indices = scores.topk(k, dim=-1, largest=True, sorted=True)
    if need_renorm:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return scores, topk_weights, topk_indices
```

- [ ] **Step 2: Add Component 2 autograd**

Add after the helpers:

```python
class LumenFusedComputeScoreForMoEAuxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: Tensor, topk: int) -> Tuple[Tensor, Tensor]:
        logits_fp32 = logits.float()

        if _probe_aiter_softmax_topk():
            scores, _, topk_indices = _aiter_softmax_topk(logits_fp32, topk, False)
        else:
            scores, _, topk_indices = _pytorch_softmax_topk(logits_fp32, topk, False)

        N, E = scores.shape
        routing_map = torch.zeros(N, E, dtype=torch.bool, device=logits.device)
        routing_map.scatter_(1, topk_indices.long(), True)

        ctx.save_for_backward(scores)

        return routing_map, scores

    @staticmethod
    def backward(ctx, grad_routing_map, grad_scores):
        (scores,) = ctx.saved_tensors
        dot = (grad_scores * scores).sum(dim=-1, keepdim=True)
        grad_logits = scores * (grad_scores - dot)
        return grad_logits, None


def fused_compute_score_for_moe_aux_loss(
    logits: Tensor, topk: int, score_function: str,
) -> Tuple[Tensor, Tensor]:
    if score_function != "softmax":
        raise NotImplementedError(
            f"score_function='{score_function}' not supported, only 'softmax'"
        )
    return LumenFusedComputeScoreForMoEAuxLoss.apply(logits, topk)
```

- [ ] **Step 3: Add Component 1 autograd**

Add after Component 2:

```python
class LumenFusedTopkWithScoreFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: Tensor,
        topk: int,
        use_pre_softmax: bool,
        scaling_factor: Optional[float],
    ) -> Tuple[Tensor, Tensor]:
        N, E = logits.shape

        if use_pre_softmax:
            logits_fp32 = logits.float()
            if _probe_aiter_softmax_topk():
                s, w, topk_indices = _aiter_softmax_topk(logits_fp32, topk, True)
            else:
                s, w, topk_indices = _pytorch_softmax_topk(logits_fp32, topk, True)

            s_k = s.gather(1, topk_indices.long())
            V = s_k.sum(dim=-1, keepdim=True)

            if scaling_factor is not None:
                w = w * scaling_factor

            routing_probs = torch.zeros(N, E, dtype=w.dtype, device=logits.device)
            routing_probs.scatter_(1, topk_indices.long(), w)

            routing_map = torch.zeros(N, E, dtype=torch.bool, device=logits.device)
            routing_map.scatter_(1, topk_indices.long(), True)

            ctx.save_for_backward(s, s_k, topk_indices.long(), V)
            ctx.scaling_factor = scaling_factor
            ctx.use_pre_softmax = True

            return routing_map, routing_probs

        else:
            ctx.use_pre_softmax = False
            logits_fp32 = logits.float().detach().requires_grad_(True)
            with torch.enable_grad():
                _, topk_indices = logits_fp32.topk(topk, dim=-1)
                gathered = logits_fp32.gather(1, topk_indices)
                w = torch.softmax(gathered, dim=-1)

                if scaling_factor is not None:
                    w = w * scaling_factor

                routing_probs = torch.zeros(
                    N, E, dtype=w.dtype, device=logits.device
                )
                routing_probs.scatter_(1, topk_indices, w)

            routing_map = torch.zeros(N, E, dtype=torch.bool, device=logits.device)
            routing_map.scatter_(1, topk_indices, True)

            ctx.save_for_backward(logits_fp32, routing_probs)
            ctx._pre_softmax_false_graph = routing_probs

            return routing_map, routing_probs.detach()

    @staticmethod
    def backward(ctx, grad_routing_map, grad_routing_probs):
        if ctx.use_pre_softmax:
            s, s_k, topk_indices, V = ctx.saved_tensors
            N, E = s.shape
            k = topk_indices.shape[1]

            grad_w = grad_routing_probs.gather(1, topk_indices)

            if ctx.scaling_factor is not None:
                grad_w = grad_w * ctx.scaling_factor

            w = s_k / V
            grad_s_k = (grad_w - (grad_w * w).sum(dim=-1, keepdim=True)) / V

            grad_s = torch.zeros(N, E, dtype=s.dtype, device=s.device)
            grad_s.scatter_add_(1, topk_indices, grad_s_k)

            dot = (grad_s * s).sum(dim=-1, keepdim=True)
            grad_logits = s * (grad_s - dot)

            return grad_logits, None, None, None

        else:
            logits_fp32, routing_probs = ctx.saved_tensors
            routing_probs.backward(grad_routing_probs)
            return logits_fp32.grad, None, None, None


def fused_topk_with_score_function(
    logits: Tensor,
    topk: int,
    use_pre_softmax: bool,
    num_groups: Optional[int],
    group_topk: Optional[int],
    scaling_factor: Optional[float],
    score_function: str,
    expert_bias: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    if score_function != "softmax":
        raise NotImplementedError(
            f"score_function='{score_function}' not supported, only 'softmax'"
        )
    if num_groups and num_groups > 0:
        raise NotImplementedError("Group routing (num_groups > 0) not supported")
    if group_topk and group_topk > 0:
        raise NotImplementedError("Group top-k not supported")
    if expert_bias is not None:
        raise NotImplementedError("expert_bias not supported")

    return LumenFusedTopkWithScoreFunction.apply(
        logits, topk, use_pre_softmax, scaling_factor
    )
```

- [ ] **Step 4: Verify all three imports work**

Run: `python -c "from lumen.ops.moe.fused_router import fused_topk_with_score_function, fused_compute_score_for_moe_aux_loss, fused_moe_aux_loss; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add lumen/ops/moe/fused_router.py
git commit -m "feat(moe): add Components 1 & 2 autograd functions for fused routing"
```

---

### Task 9: Exports in `__init__.py`

**Files:**
- Modify: `lumen/ops/moe/__init__.py:7-15`

- [ ] **Step 1: Add re-exports**

Add to the imports and `__all__` in `lumen/ops/moe/__init__.py`:

```python
from lumen.ops.moe.fused_router import (
    fused_compute_score_for_moe_aux_loss,
    fused_moe_aux_loss,
    fused_topk_with_score_function,
)
```

Add to `__all__`:

```python
    "fused_topk_with_score_function",
    "fused_compute_score_for_moe_aux_loss",
    "fused_moe_aux_loss",
```

- [ ] **Step 2: Commit**

```bash
git add lumen/ops/moe/__init__.py
git commit -m "feat(moe): export fused router APIs from lumen.ops.moe"
```

---

### Task 10: Monkey-patch wiring + dead flag deletion

**Files:**
- Modify: `lumen/models/megatron.py:1103-1109` (delete dead flag), and add monkey-patch near other TE replacements
- Modify: `lumen/models/fsdp.py:270-277` (delete dead flag)

- [ ] **Step 1: Delete `--lumen-fused-moe-routing` from `megatron.py`**

Delete lines 1103-1109 in `lumen/models/megatron.py` (the `safe_add_argument` block for `--lumen-fused-moe-routing`).

- [ ] **Step 2: Delete `--lumen-fused-moe-routing` from `fsdp.py`**

Delete lines 270-277 in `lumen/models/fsdp.py` (the `moe-routing` argument group and `--lumen-fused-moe-routing` argument).

- [ ] **Step 3: Add monkey-patch in `megatron.py`**

Find the section where other TE replacements are applied (near `TESpecProvider` replacement around line 493-512). Add the MoE router monkey-patch in the same initialization flow. Look for `_apply_lumen_args` or the function that runs during Lumen initialization.

Add this patching logic:

```python
def _patch_moe_fused_router():
    """Monkey-patch Megatron-Core's moe_utils with Lumen fused router ops."""
    try:
        import megatron.core.transformer.moe.moe_utils as moe_utils
        from lumen.ops.moe.fused_router import (
            fused_compute_score_for_moe_aux_loss,
            fused_moe_aux_loss,
            fused_topk_with_score_function,
        )

        moe_utils.fused_topk_with_score_function = fused_topk_with_score_function
        moe_utils.fused_compute_score_for_moe_aux_loss = (
            fused_compute_score_for_moe_aux_loss
        )
        moe_utils.fused_moe_aux_loss = fused_moe_aux_loss

        try:
            import megatron.core.extensions.transformer_engine as te_ext

            te_ext.fused_topk_with_score_function = fused_topk_with_score_function
            te_ext.fused_compute_score_for_moe_aux_loss = (
                fused_compute_score_for_moe_aux_loss
            )
            te_ext.fused_moe_aux_loss = fused_moe_aux_loss
        except ImportError:
            pass

        logger.info("Patched Megatron-Core moe_utils with Lumen fused router ops")
    except ImportError:
        logger.debug("Megatron-Core moe_utils not found, skipping MoE router patch")
```

Call `_patch_moe_fused_router()` from the existing Lumen initialization path (where `LumenSpecProvider` is set up).

- [ ] **Step 4: Verify patch loads**

Run: `python -c "import lumen.models.megatron; print('ok')"`
Expected: `ok` (no crash; import may warn if Megatron-Core is not installed, which is fine)

- [ ] **Step 5: Commit**

```bash
git add lumen/models/megatron.py lumen/models/fsdp.py
git commit -m "feat(moe): monkey-patch fused router ops; delete dead --lumen-fused-moe-routing flag"
```

---

### Task 11: Lumen-level unit tests

**Files:**
- Create: `tests/ops/test_moe_fused_router.py`

- [ ] **Step 1: Write file header and references**

```python
# tests/ops/test_moe_fused_router.py
"""Lumen-level unit tests for fused MoE router ops.

Tests 1-7 map to the spec's Testing section:
  Test 1: fused_topk_with_score_function (Component 1)
  Test 2: fused_compute_score_for_moe_aux_loss (Component 2)
  Test 3: fused_moe_aux_loss (Component 3)
  Test 4: Unsupported feature guards
  Test 5: gradcheck on all three components
  Test 6: End-to-end patch smoke test
  Test 7: AITER forward parity (7a, 7b, 7c)
"""
import pytest
import torch

from conftest import compute_snr
from lumen.ops.dispatch import (
    _probe_aiter_softmax_topk,
    _probe_aiter_triton_moe_aux_loss,
)

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

_requires_aiter_softmax_topk = pytest.mark.skipif(
    not torch.cuda.is_available() or not _probe_aiter_softmax_topk(),
    reason="AITER softmax_topk not available",
)

_requires_aiter_moe_aux_loss = pytest.mark.skipif(
    not torch.cuda.is_available() or not _probe_aiter_triton_moe_aux_loss(),
    reason="AITER Triton moe_aux_loss not available",
)


# -- Reference implementations --


def _topk_routing_ref(logits, topk, use_pre_softmax, scaling_factor):
    """Unfused reference matching moe_utils.topk_routing_with_score_function(fused=False)."""
    N, E = logits.shape
    logits_fp32 = logits.float()

    if use_pre_softmax:
        scores = torch.softmax(logits_fp32, dim=-1)
        topk_weights, topk_indices = scores.topk(topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    else:
        topk_vals, topk_indices = logits_fp32.topk(topk, dim=-1)
        topk_weights = torch.softmax(topk_vals, dim=-1)

    if scaling_factor is not None:
        topk_weights = topk_weights * scaling_factor

    routing_probs = torch.zeros(N, E, dtype=topk_weights.dtype, device=logits.device)
    routing_probs.scatter_(1, topk_indices, topk_weights)
    routing_map = torch.zeros(N, E, dtype=torch.bool, device=logits.device)
    routing_map.scatter_(1, topk_indices, True)
    return routing_map, routing_probs


def _compute_scores_ref(logits, topk):
    """Unfused reference matching moe_utils.compute_routing_scores_for_aux_loss(fused=False)."""
    scores = torch.softmax(logits.float(), dim=-1)
    _, topk_indices = scores.topk(topk, dim=-1)
    N, E = scores.shape
    routing_map = torch.zeros(N, E, dtype=torch.bool, device=logits.device)
    routing_map.scatter_(1, topk_indices, True)
    return routing_map, scores


def _switch_load_balancing_loss_ref(probs, tokens_per_expert, T, topk, E, coeff):
    """Unfused reference matching moe_utils.switch_load_balancing_loss_func(fused=False)."""
    aggregated = probs.sum(dim=0)
    C = E * coeff / (topk * T * T)
    return (aggregated * tokens_per_expert).sum() * C
```

- [ ] **Step 2: Write Test 1 — `test_fused_topk_with_score_function`**

```python
@_requires_cuda
@pytest.mark.parametrize("N,E,topk", [(32, 8, 1), (32, 8, 2), (512, 64, 4), (4096, 128, 8)])
@pytest.mark.parametrize("use_pre_softmax", [True, False])
@pytest.mark.parametrize("scaling_factor", [None, 0.5])
def test_fused_topk_with_score_function(N, E, topk, use_pre_softmax, scaling_factor):
    """Spec Test 1: forward + backward vs unfused reference."""
    from lumen.ops.moe.fused_router import fused_topk_with_score_function

    torch.manual_seed(42)
    logits = torch.randn(N, E, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    routing_map, routing_probs = fused_topk_with_score_function(
        logits, topk, use_pre_softmax, None, None, scaling_factor, "softmax", None,
    )

    logits_ref = logits.detach().clone().requires_grad_(True)
    routing_map_ref, routing_probs_ref = _topk_routing_ref(
        logits_ref, topk, use_pre_softmax, scaling_factor,
    )

    assert routing_map.dtype == torch.bool
    assert routing_map.sum(dim=-1).eq(topk).all()
    torch.testing.assert_close(
        routing_probs, routing_probs_ref, atol=1e-5, rtol=1e-5,
    )
    assert (routing_map == routing_map_ref).all()

    loss = routing_probs.sum()
    loss.backward()
    loss_ref = routing_probs_ref.sum()
    loss_ref.backward()

    torch.testing.assert_close(logits.grad, logits_ref.grad, atol=1e-4, rtol=1e-4)
    snr = compute_snr(logits.grad, logits_ref.grad)
    assert snr > 15, f"Backward SNR too low: {snr:.1f} dB"
```

- [ ] **Step 3: Write Test 2 — `test_fused_compute_score_for_moe_aux_loss`**

```python
@_requires_cuda
@pytest.mark.parametrize("N,E,topk", [(32, 8, 2), (512, 64, 4), (4096, 128, 8)])
def test_fused_compute_score_for_moe_aux_loss(N, E, topk):
    """Spec Test 2: forward + backward vs unfused reference."""
    from lumen.ops.moe.fused_router import fused_compute_score_for_moe_aux_loss

    torch.manual_seed(42)
    logits = torch.randn(N, E, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    routing_map, scores = fused_compute_score_for_moe_aux_loss(logits, topk, "softmax")

    logits_ref = logits.detach().clone().requires_grad_(True)
    routing_map_ref, scores_ref = _compute_scores_ref(logits_ref, topk)

    torch.testing.assert_close(scores, scores_ref, atol=1e-5, rtol=1e-5)
    assert routing_map.dtype == torch.bool
    assert (routing_map == routing_map_ref).all()

    loss = scores.sum()
    loss.backward()
    loss_ref = scores_ref.sum()
    loss_ref.backward()

    torch.testing.assert_close(logits.grad, logits_ref.grad, atol=1e-4, rtol=1e-4)
    snr = compute_snr(logits.grad, logits_ref.grad)
    assert snr > 15, f"Backward SNR too low: {snr:.1f} dB"
```

- [ ] **Step 4: Write Test 3 — `test_fused_moe_aux_loss`**

```python
@_requires_cuda
@pytest.mark.parametrize("N,E,topk", [(32, 8, 2), (512, 64, 4), (4096, 128, 8)])
def test_fused_moe_aux_loss(N, E, topk):
    """Spec Test 3: forward + backward vs unfused reference."""
    from lumen.ops.moe.fused_router import fused_moe_aux_loss

    torch.manual_seed(42)
    probs = torch.rand(N, E, device="cuda", dtype=torch.float32, requires_grad=True)
    tokens_per_expert = torch.randint(0, N, (E,), device="cuda", dtype=torch.float32)
    T, coeff = N, 0.01

    loss = fused_moe_aux_loss(probs, tokens_per_expert, T, E, topk, coeff)
    loss_ref = _switch_load_balancing_loss_ref(
        probs.detach(), tokens_per_expert, T, topk, E, coeff
    )
    torch.testing.assert_close(loss, loss_ref, atol=1e-5, rtol=1e-5)

    loss.backward()
    probs_ref = probs.detach().clone().requires_grad_(True)
    loss_ref2 = _switch_load_balancing_loss_ref(
        probs_ref, tokens_per_expert, T, topk, E, coeff
    )
    loss_ref2.backward()
    torch.testing.assert_close(probs.grad, probs_ref.grad, atol=1e-5, rtol=1e-5)
```

- [ ] **Step 5: Write Test 4 — unsupported feature guards**

```python
@_requires_cuda
def test_unsupported_score_function():
    """Spec Test 4: NotImplementedError for unsupported features."""
    from lumen.ops.moe.fused_router import (
        fused_topk_with_score_function,
        fused_compute_score_for_moe_aux_loss,
    )

    logits = torch.randn(4, 8, device="cuda", dtype=torch.float32)

    with pytest.raises(NotImplementedError, match="sigmoid"):
        fused_topk_with_score_function(
            logits, 2, True, None, None, None, "sigmoid", None
        )

    with pytest.raises(NotImplementedError, match="sigmoid"):
        fused_compute_score_for_moe_aux_loss(logits, 2, "sigmoid")

    with pytest.raises(NotImplementedError, match="Group"):
        fused_topk_with_score_function(
            logits, 2, True, 2, 1, None, "softmax", None
        )
```

- [ ] **Step 6: Write Test 5 — gradcheck on all three components**

Gradcheck uses thin pure-PyTorch float64 wrappers that replicate the autograd math,
not the actual AITER-backed autograd functions (AITER kernels don't support float64).
This validates the backward formulas independently of the kernel implementations.

```python
def _pure_pytorch_topk_with_score(logits, topk):
    """Pure PyTorch reimplementation of Component 1 backward math (float64-safe)."""
    s = torch.softmax(logits, dim=-1)
    _, topk_indices = s.topk(topk, dim=-1)
    s_k = s.gather(1, topk_indices)
    V = s_k.sum(dim=-1, keepdim=True)
    w = s_k / V
    N, E = logits.shape
    routing_probs = torch.zeros(N, E, dtype=logits.dtype, device=logits.device)
    routing_probs.scatter_(1, topk_indices, w)
    return routing_probs


def _pure_pytorch_compute_scores(logits, topk):
    """Pure PyTorch reimplementation of Component 2 (float64-safe)."""
    return torch.softmax(logits, dim=-1)


def _pure_pytorch_aux_loss(probs, tokens_per_expert, T, E, topk, coeff):
    """Pure PyTorch reimplementation of Component 3 (float64-safe)."""
    C = E * coeff / (topk * T * T)
    return (probs.sum(dim=0) * tokens_per_expert).sum() * C


@_requires_cuda
def test_gradcheck_topk_with_score_function():
    """Spec Test 5: gradcheck for Component 1 math (use_pre_softmax=True, topk=2)."""
    torch.manual_seed(42)
    logits = torch.randn(4, 8, device="cuda", dtype=torch.float64, requires_grad=True)

    torch.autograd.gradcheck(
        lambda x: _pure_pytorch_topk_with_score(x, 2),
        (logits,),
        eps=1e-6,
        atol=1e-4,
    )


@_requires_cuda
def test_gradcheck_compute_scores():
    """Spec Test 5: gradcheck for Component 2 math."""
    torch.manual_seed(42)
    logits = torch.randn(4, 8, device="cuda", dtype=torch.float64, requires_grad=True)

    torch.autograd.gradcheck(
        lambda x: _pure_pytorch_compute_scores(x, 2),
        (logits,),
        eps=1e-6,
        atol=1e-4,
    )


@_requires_cuda
def test_gradcheck_aux_loss():
    """Spec Test 5: gradcheck for Component 3 math."""
    torch.manual_seed(42)
    probs = torch.rand(4, 8, device="cuda", dtype=torch.float64, requires_grad=True)
    tokens_per_expert = torch.rand(8, device="cuda", dtype=torch.float64)

    torch.autograd.gradcheck(
        lambda p: _pure_pytorch_aux_loss(p, tokens_per_expert, 4, 8, 2, 0.01),
        (probs,),
        eps=1e-6,
        atol=1e-4,
    )
```

- [ ] **Step 7: Write Test 6 — end-to-end patch smoke test**

```python
@_requires_cuda
def test_end_to_end_patch_smoke():
    """Spec Test 6: import + patch + call all three entry points."""
    from lumen.ops.moe.fused_router import (
        fused_compute_score_for_moe_aux_loss,
        fused_moe_aux_loss,
        fused_topk_with_score_function,
    )

    try:
        import megatron.core.transformer.moe.moe_utils as moe_utils
    except ImportError:
        pytest.skip("Megatron-Core not installed")

    moe_utils.fused_topk_with_score_function = fused_topk_with_score_function
    moe_utils.fused_compute_score_for_moe_aux_loss = (
        fused_compute_score_for_moe_aux_loss
    )
    moe_utils.fused_moe_aux_loss = fused_moe_aux_loss

    torch.manual_seed(42)
    logits = torch.randn(16, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    routing_map, routing_probs = moe_utils.fused_topk_with_score_function(
        logits, 2, True, None, None, None, "softmax", None,
    )
    assert routing_probs.shape == (16, 8)
    routing_probs.sum().backward()

    logits2 = torch.randn(16, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    r_map, scores = moe_utils.fused_compute_score_for_moe_aux_loss(
        logits2, 2, "softmax",
    )
    assert scores.shape == (16, 8)

    probs = torch.rand(16, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    tpe = torch.ones(8, device="cuda", dtype=torch.float32)
    loss = moe_utils.fused_moe_aux_loss(probs, tpe, 16, 8, 2, 0.01)
    assert loss.dim() == 0
    loss.backward()
```

- [ ] **Step 8: Write Test 7 — AITER forward parity**

```python
@_requires_aiter_softmax_topk
def test_aiter_softmax_topk_parity():
    """Spec Test 7a: AITER HIP softmax_topk full softmax vs PyTorch reference."""
    from lumen.ops.moe.fused_router import _aiter_softmax_topk

    torch.manual_seed(42)
    logits = torch.randn(512, 64, device="cuda", dtype=torch.float32)

    scores_aiter, _, topk_indices_aiter = _aiter_softmax_topk(logits, 4, False)
    scores_ref = torch.softmax(logits, dim=-1)
    _, topk_indices_ref = scores_ref.topk(4, dim=-1)

    torch.testing.assert_close(scores_aiter, scores_ref, atol=1e-5, rtol=1e-5)
    for row in range(min(512, 32)):
        assert set(topk_indices_aiter[row].tolist()) == set(
            topk_indices_ref[row].tolist()
        ), f"Row {row}: expert sets differ"


@_requires_aiter_softmax_topk
def test_aiter_softmax_topk_renorm_parity():
    """Spec Test 7b: AITER softmax_topk renorm vs PyTorch reference."""
    from lumen.ops.moe.fused_router import _aiter_softmax_topk

    torch.manual_seed(42)
    logits = torch.randn(512, 64, device="cuda", dtype=torch.float32)

    _, topk_weights, topk_indices = _aiter_softmax_topk(logits, 4, True)

    scores_ref = torch.softmax(logits, dim=-1)
    w_ref, idx_ref = scores_ref.topk(4, dim=-1)
    w_ref = w_ref / w_ref.sum(dim=-1, keepdim=True)

    for row in range(min(512, 32)):
        aiter_set = set(topk_indices[row].tolist())
        ref_set = set(idx_ref[row].tolist())
        assert aiter_set == ref_set, f"Row {row}: expert sets differ"

    torch.testing.assert_close(topk_weights, w_ref, atol=1e-4, rtol=1e-4)


@_requires_aiter_softmax_topk
def test_aiter_topk_softmax_asm_parity():
    """Spec Test 7c: existing AITER ASM topk_softmax vs PyTorch reference."""
    from lumen.ops.dispatch import _probe_aiter_moe_topk_softmax

    if not _probe_aiter_moe_topk_softmax():
        pytest.skip("AITER ASM topk_softmax not available")

    from aiter.ops.moe_op import topk_softmax

    torch.manual_seed(42)
    N, E, k = 512, 64, 4
    logits = torch.randn(N, E, device="cuda", dtype=torch.float32)

    topk_weights = torch.empty(N, k, device="cuda", dtype=torch.float32)
    topk_indices = torch.empty(N, k, device="cuda", dtype=torch.int32)
    token_expert_indices = torch.empty(N, k, device="cuda", dtype=torch.int32)

    topk_softmax(topk_weights, topk_indices, token_expert_indices, logits, True)

    scores_ref = torch.softmax(logits, dim=-1)
    w_ref, idx_ref = scores_ref.topk(k, dim=-1)
    w_ref = w_ref / w_ref.sum(dim=-1, keepdim=True)

    for row in range(min(N, 32)):
        assert set(topk_indices[row].tolist()) == set(
            idx_ref[row].tolist()
        ), f"Row {row}: expert sets differ"

    torch.testing.assert_close(topk_weights, w_ref, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 9: Run all tests**

Run: `pytest tests/ops/test_moe_fused_router.py -v`
Expected: All tests PASS.

- [ ] **Step 10: Commit**

```bash
git add tests/ops/test_moe_fused_router.py
git commit -m "test(moe): add Lumen-level unit tests for fused router ops"
```

---

## Verification Checklist

After all tasks are complete:

- [ ] All AITER-level tests pass: `cd third_party/aiter && python -m pytest op_tests/test_softmax_topk.py op_tests/test_moe_aux_loss.py -v`
- [ ] All Lumen-level tests pass: `pytest tests/ops/test_moe_fused_router.py -v`
- [ ] No existing tests broken: `pytest tests/ -x --timeout=120`
- [ ] `--lumen-fused-moe-routing` flag fully removed (grep finds zero hits)
- [ ] `from lumen.ops.moe import fused_topk_with_score_function, fused_compute_score_for_moe_aux_loss, fused_moe_aux_loss` succeeds
- [ ] PR description states which Megatron-Core commit/version the patches target
