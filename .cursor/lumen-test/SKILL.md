# Lumen Test Generation Guide

## Mission

Generate tests for the Lumen project that follow established conventions, use the shared test infrastructure, and meet the accuracy/coverage standards defined in the TE feature parity plan. Every new feature, kernel, or module must have tests before it can be marked "Supported".

## Test Architecture

```
tests/
├── ops/              # Op-level: stateless functions vs PyTorch/torchao references
│   └── conftest.py   # Shared references, metrics, fixtures, config dataclasses
├── module/           # Module-level: nn.Module wrappers, mock-based TP/SP, CUDA required
├── models/           # Model-level: Megatron/FSDP integration, CLI args, patching
├── quantize/         # FP8 scaling manager, config, param lifecycle
├── core/             # Float8 dtypes, grad quantization
├── env/              # Multi-GPU NCCL/RCCL collectives
└── utils/            # HIP graphs, checkpoint, CPU offload
```

### Which directory to use

| You are testing… | Directory | Pattern |
|------------------|-----------|---------|
| A function in `lumen/ops/` | `tests/ops/` | Compare against PyTorch reference using `compute_snr` |
| An `nn.Module` in `lumen/modules/` | `tests/module/` | Mock distributed primitives, test construction + forward + backward |
| Megatron/FSDP integration | `tests/models/` | Mock `MegatronConfig`, test patching + CLI args |
| `QuantConfig` / `ScalingManager` | `tests/quantize/` | Test config parsing, scale computation, amax history |
| Utilities (graphs, checkpoint, offload) | `tests/utils/` | Test lifecycle, context managers |

## Shared Infrastructure (`tests/ops/conftest.py`)

### Reference Implementations

Always compare Lumen ops against these pure-PyTorch references:

| Reference | Use for | Import |
|-----------|---------|--------|
| `attention_ref(q, k, v, sm_scale, causal)` | Multi-head attention (BSHD, f32 compute) | `from conftest import attention_ref` |
| `rmsnorm_ref(x, weight, eps)` | RMSNorm | `from conftest import rmsnorm_ref` |
| `layernorm_ref(x, weight, bias, eps)` | LayerNorm | `from conftest import layernorm_ref` |
| `grouped_gemm_ref(lhs, rhs, group_sizes, bias)` | Per-expert GEMM (MoE) | `from conftest import grouped_gemm_ref` |
| `cross_entropy_ref(logits, target, label_smoothing, ignore_idx)` | Cross-entropy | `from conftest import cross_entropy_ref` |
| `fp8_quant_dequant_ref(tensor, fp8_dtype)` | Per-tensor FP8 round-trip | `from conftest import fp8_quant_dequant_ref` |
| `fp8_blockwise_quant_dequant_ref(tensor, block_size, fp8_dtype)` | Blockwise FP8 round-trip | `from conftest import fp8_blockwise_quant_dequant_ref` |

If no shared reference exists for your op, write a pure-PyTorch reference **inside the test file** (not in conftest). Prefix it with `_` (e.g., `_rope_reference_neox`).

### Accuracy Metrics

```python
from conftest import compute_snr, check_close
```

| Metric | Use | Thresholds |
|--------|-----|------------|
| `compute_snr(ref, test)` | SNR in dB — primary metric | See table below |
| `check_close(a, b, atol, rtol, tol_err_ratio)` | Element-wise with outlier tolerance | Default: 5% outliers allowed |
| `torch.testing.assert_close(a, b, atol, rtol)` | Strict element-wise (for exact/BF16 paths) | Use for non-quantized paths |

#### SNR Thresholds (empirical, MI300X / MI355X)

| Path | Forward | Backward (dX) | Backward (dW) |
|------|:-------:|:--------------:|:--------------:|
| BF16 (attention) | ≥ 20 dB | ≥ 15 dB | ≥ 15 dB |
| BF16 causal bwd | — | ≥ 10 dB | ≥ 10 dB |
| FP8 blockwise | ≥ 15 dB | ≥ 8 dB | — |
| FP8 delayed/dynamic | ≥ 12 dB | ≥ 6 dB | ≥ 6 dB |
| FP8 per_token | ≥ 12 dB | — | — |
| MXFP8 | ≥ 10 dB | ≥ 5 dB | — |
| BF16 GEMM (none) | ≥ 25 dB | ≥ 20 dB | — |
| Fused MLP | `assert_close` | `assert_close` | — |

When adding relaxed thresholds, add a comment explaining why:
```python
# FP8 blockwise adds ~5 dB quantization noise vs f32 reference
assert snr > 8, f"FP8 blockwise bwd SNR too low: {snr:.1f} dB"
```

### Config Dataclasses

```python
from conftest import AttnConfig, NormConfig, LinearConfig
```

Use dataclasses for parametrize configs — they produce readable test IDs via `__repr__`:

```python
CONFIGS = [
    AttnConfig(128, 128, 8, 8, 64, 64),   # standard MHA
    AttnConfig(256, 256, 16, 4, 64, 64),   # GQA
]
CONFIG_IDS = [repr(c) for c in CONFIGS]

@pytest.mark.parametrize("config", CONFIGS, ids=CONFIG_IDS)
def test_something(config):
    ...
```

### Fixtures

| Fixture | Scope | Behavior |
|---------|-------|----------|
| `seed_rng` | autouse | Sets `torch.manual_seed(0)` + `torch.cuda.manual_seed(0)` |
| `device` | function | Returns `"cuda"` |

The `seed_rng` fixture is autouse in `tests/ops/`. For other directories, set seeds manually if determinism is required.

## Test File Template

### Op-level test (tests/ops/)

```python
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Tests for lumen.ops.<module>: <brief description>.

Covers:
  - Forward vs PyTorch reference (BF16, compute_snr)
  - Forward + backward: gradient correctness vs reference
  - Edge cases: <list relevant edge cases>

Reference: <reference_fn> from conftest (pure PyTorch).

SNR threshold rationale (empirical, measured on MI300X / MI355X):
  - BF16 forward:  >= XX dB
  - BF16 backward: >= XX dB
"""

import pytest
import torch
from conftest import SomeConfig, some_ref, compute_snr

import lumen.ops.<module> as ops

# ---------------------------------------------------------------------------
# Hardware / backend detection
# ---------------------------------------------------------------------------

def _is_aiter_available():
    try:
        import aiter  # noqa: F401
        return True
    except ImportError:
        return False

_aiter = _is_aiter_available()
_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

CONFIGS = [
    SomeConfig(...),  # standard
    SomeConfig(...),  # edge case
]
CONFIG_IDS = [repr(c) for c in CONFIGS]

# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------

@_CUDA
@pytest.mark.skipif(not _aiter, reason="AITER required")
@pytest.mark.parametrize("config", CONFIGS, ids=CONFIG_IDS)
def test_op_fwd(config):
    # 1. Create inputs
    x = torch.randn(..., device="cuda", dtype=torch.bfloat16)
    # 2. Reference
    ref = some_ref(x, ...)
    # 3. Lumen op
    out = ops.some_op(x, ...)
    # 4. Verify
    snr = compute_snr(ref, out)
    assert snr > 20, f"Forward SNR too low: {snr:.1f} dB"

# ---------------------------------------------------------------------------
# Forward + backward tests
# ---------------------------------------------------------------------------

@_CUDA
@pytest.mark.skipif(not _aiter, reason="AITER required")
@pytest.mark.parametrize("config", CONFIGS, ids=CONFIG_IDS)
def test_op_fwd_bwd(config):
    # 1. Reference path
    x_ref = torch.randn(..., requires_grad=True)
    out_ref = some_ref(x_ref, ...)
    out_ref.sum().backward()
    # 2. Lumen path (same seed)
    torch.manual_seed(0)
    x = torch.randn(..., requires_grad=True)
    out = ops.some_op(x, ...)
    out.sum().backward()
    # 3. Compare
    snr_fwd = compute_snr(out_ref, out)
    snr_dx = compute_snr(x_ref.grad, x.grad)
    assert snr_fwd > 20
    assert snr_dx > 15
```

### Module-level test (tests/module/)

```python
import pytest
import torch
from unittest import mock

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

@_CUDA
class TestSomeModule:

    def test_construction(self):
        m = SomeModule(64, 128)
        assert m.weight.shape == (128, 64)

    def test_forward_shape(self):
        m = SomeModule(64, 128).to("cuda")
        x = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
        out = m(x)
        assert out.shape == (2, 16, 128)

    def test_backward_runs(self):
        m = SomeModule(64, 128).to("cuda")
        x = torch.randn(2, 16, 64, device="cuda", requires_grad=True)
        out = m(x)
        out.sum().backward()
        assert x.grad is not None
        assert m.weight.grad is not None
```

For modules requiring TP/SP mocking (e.g., `LumenColumnParallelLinear`), stack `@mock.patch` decorators:

```python
@mock.patch("lumen.modules.parallel_linear._get_tp_group", return_value=None)
@mock.patch("lumen.modules.parallel_linear._pg_size", return_value=1)
@mock.patch("lumen.modules.parallel_linear._pg_rank", return_value=0)
class TestLumenColumnParallelLinear:
    ...
```

### Distributed test (multi-GPU spawn)

```python
import functools
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

_MULTI_GPU = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Need ≥2 GPUs"
)

def _worker(rank, world_size, fn, *args):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    device = torch.device("cuda", rank)
    torch.cuda.set_device(rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank,
                            world_size=world_size, device_id=device)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()

@_MULTI_GPU
def test_distributed_op():
    def _check(rank, world_size):
        ...  # actual test logic
    mp.spawn(_worker, args=(2, _check), nprocs=2, join=True)
```

For SDMA tests, also register the process group for mori:
```python
torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
```

## Hardware Skip Markers

Use consistently across all test files:

```python
_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
_AITER = pytest.mark.skipif(not _is_aiter_available(), reason="AITER required")
_GFX950 = pytest.mark.skipif(not _is_gfx950(), reason="gfx950+ required")
_MULTI_GPU = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need ≥2 GPUs")
```

Apply at class level for entire test classes, or at function level for individual tests.

## What to Test per Feature Category

### Attention Features

| Feature | Required Tests |
|---------|---------------|
| New attention variant | Forward SNR vs `attention_ref`, backward dQ/dK/dV SNR, causal mode, GQA |
| Sliding window (`window_size`) | Forward correctness with window mask, compare masked vs full attention |
| Attention bias | All `core_attention_bias_type` variants, forward SNR, backward SNR |
| ALiBi slopes | Forward with slopes tensor, compare vs manual bias addition |
| Return LSE | Check `lse` shape and dtype, verify `exp(lse)` matches row softmax sums |
| CP (context parallelism) | Multi-GPU spawn, output matches non-CP reference (same global input) |

### FP8 Quantization Features

| Feature | Required Tests |
|---------|---------------|
| New scaling type | Forward GEMM SNR, backward dX SNR, round-trip quant→dequant vs `fp8_quant_dequant_ref` |
| Hybrid (e4m3/e5m2) | Forward uses e4m3, backward uses e5m2, verify dtype of intermediate tensors |
| Amax/delayed scale | `delayed_scale_ref` comparison, history buffer length, algorithm (most_recent/max) |
| FP8 param all-gather | Distributed test: quant on each rank, all-gather, dequant, compare to BF16 all-gather |

### Linear / GEMM Features

| Feature | Required Tests |
|---------|---------------|
| New GEMM path | Forward SNR vs `torch.mm`, backward dX and dW SNR, parametrize all scaling types |
| Fused MLP | `torch.testing.assert_close` vs decomposed reference, backward correctness |
| TP overlap | Multi-GPU: verify output matches non-overlapped path, measure overlap ratio |
| Grad accum fusion | Compare accumulated gradient vs sequential accumulation |

### Normalization Features

| Feature | Required Tests |
|---------|---------------|
| Norm op | Forward+backward SNR vs `rmsnorm_ref`/`layernorm_ref`, parametrize shapes |
| Fused norm+quant | Forward output SNR, verify quantized output dtype, scale shape per scaling type |
| Zero-centered gamma | Forward with `zero_centered_gamma=True`, compare vs `(1 + gamma) * norm(x)` |

### Communication Features

| Feature | Required Tests |
|---------|---------------|
| SDMA collective | Multi-GPU: output matches NCCL reference, dtype variants (bf16, fp32) |
| Sequence parallelism | Verify SP scatter/gather round-trip preserves tensor values |
| SDMA cross-entropy | Multi-GPU: SDMA path output matches non-SDMA reference |

### Module Features

| Feature | Required Tests |
|---------|---------------|
| Any `nn.Module` | Construction (shapes, defaults), forward shape, backward runs, FP8 variant |
| Checkpoint/recompute | Verify activation recomputation produces same output, memory reduction |
| CPU offload | Hooks registered, tensors moved to CPU, restored to GPU on forward |
| HIP graphs | Graph capture succeeds, replayed output matches eager output |

## Scaling Type Parametrization

Standard parametrization sets for linear/GEMM tests:

```python
ALL_SCALING_TYPES = ["delayed", "dynamic", "per_token", "blockwise", "blockwise2d", "mxfp8", "none"]

# Only per-tensor (scalar) scales survive weight transposition in backward
BWD_SCALING_TYPES = ["delayed", "dynamic", "none"]
```

For attention FP8 tests:
```python
ATTN_FP8_TYPES = ["blockwise", "blockwise2d", "dynamic", "delayed", "per_token"]
ATTN_MXFP8 = ["mxfp8"]  # gfx950+ only
```

## Test Naming Conventions

| Pattern | Meaning |
|---------|---------|
| `test_<op>_fwd` | Forward-only correctness |
| `test_<op>_fwd_bwd` | Forward + backward correctness |
| `test_<op>_<variant>` | Specific variant (causal, gqa, fp8, mxfp8) |
| `test_<op>_edge_<case>` | Edge case (batch1, empty, zero_groups) |
| `test_construction` | Module construction, shapes, defaults |
| `test_forward_shape` | Output shape verification |
| `test_backward_runs` | Backward completes, grads are not None |
| `test_matches_reference` | SNR or assert_close vs reference |

## Coverage Gaps (Known)

These features need tests added (as of 2026-03-18):

### P0 — Supported but untested
- Sliding window attention (`window_size` parameter)
- Attention bias (`core_attention_bias_type` variants)
- ALiBi slopes (csrc + Triton backends)
- CP P2P distributed ring test (only existence checks exist)

### P1 — Plan compliance
- End-to-end training smoke test (short loop, Lumen FP8 vs BF16)
- Hybrid FP8 (e4m3 fwd / e5m2 bwd) GEMM round-trip
- MoE auxiliary loss
- Multi-GPU SP/TP integration tests

### P2 — Hardening thin tests
- `test_fp8_activation_store.py` — needs multi-dtype, large-tensor, SNR comparison
- `test_fsdp2.py` — needs FP8 integration, multi-GPU
- `test_delay_wgrad.py` — needs wiring into `LumenRowParallelLinear`
- `test_cp_p2p.py` — needs actual distributed ring send/recv

## Anti-Patterns

```python
# BAD: testing on CPU when the op requires CUDA kernels
x = torch.randn(2, 16, 64)  # defaults to CPU
out = lumen_op(x)  # will fail or use wrong backend

# GOOD: explicit CUDA placement
x = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
```

```python
# BAD: no reference comparison — only checking shape
def test_attention():
    out = attention(q, k, v, sm_scale)
    assert out.shape == q.shape  # proves nothing about correctness

# GOOD: compare against reference
def test_attention():
    ref = attention_ref(q, k, v, sm_scale)
    out = attention(q, k, v, sm_scale)
    assert compute_snr(ref, out) > 20
```

```python
# BAD: backward test without checking grads exist
out.sum().backward()
# test ends here — doesn't verify gradient flow

# GOOD: verify gradient flow
out.sum().backward()
assert x.grad is not None
assert w.grad is not None
snr_dx = compute_snr(x_ref.grad, x.grad)
assert snr_dx > 15
```

```python
# BAD: block_size > seqlen causes kernel assertion failures
q = torch.randn(1, 128, 4, 256, ...)  # seqlen=128
attention_fp8_quant(q, k, v, quant_block_size=256, block_m_fwd=256)  # FAILS

# GOOD: ensure seqlen >= max(block_sizes)
seqlen = 256  # accommodates all parametrized block sizes
q = torch.randn(1, seqlen, 4, 256, ...)
```

```python
# BAD: torch.autograd.grad outside enable_grad in custom Function backward
# (backward runs with grad mode disabled by default)
with torch.enable_grad():
    act_val = act_fn(x.detach().requires_grad_(True))
act_grad = torch.autograd.grad(act_val.sum(), x)  # FAILS

# GOOD: keep autograd.grad inside enable_grad block
with torch.enable_grad():
    x_g = x.detach().requires_grad_(True)
    act_val = act_fn(x_g)
    act_grad = torch.autograd.grad(act_val.sum(), x_g)[0]
```
