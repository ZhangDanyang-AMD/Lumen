---
name: lumen-coding
description: "Comprehensive development guide for the Lumen library — a TE replacement for Megatron-Core on AMD GPUs. Covers architecture, dispatch patterns, FP8 conventions, testing, and TE feature parity. Use when writing, reviewing, or designing any Lumen code including ops, modules, kernels, or tests."
---

# Lumen Development Guide

## Mission

Lumen is a drop-in replacement for NVIDIA TransformerEngine (TE) within Megatron-Core, targeting AMD GPUs (gfx94x / gfx950). Every module, op, and kernel provides a feature-equivalent alternative to TE's interfaces so Megatron-Core runs on AMD hardware without source modifications. All design decisions should be evaluated against this goal.

## Architecture

```
Megatron-Core
  └─ lumen/models/megatron.py  (LumenSpecProvider — returns Lumen modules for Megatron layer specs)
       └─ lumen/modules/        (nn.Module wrappers: LumenColumnParallelLinear, LumenAttention, etc.)
            └─ lumen/ops/       (stateless dispatch functions: attention(), gemm(), rmsnorm(), etc.)
                 └─ lumen/kernels/   (Lumen-owned Triton kernels, e.g. attention_impl.py)
                 └─ third_party/aiter/  (AITER: CK, ASM, Triton kernels — NOT Lumen code)
```

### Code Boundary: Lumen vs AITER

**New GPU kernels belong in `third_party/aiter/`, not in Lumen.** Lumen only implements dispatch and calling logic — choosing the right backend, marshalling arguments, handling fallback. If a feature needs a new kernel, add it to AITER and call it from Lumen via `try_backends()`.

AITER is always available on AMD targets. No PyTorch fallbacks for production paths — raise `RuntimeError` if AITER is missing.

### Layer Responsibilities

| Layer | Owns | Does NOT own |
|-------|------|-------------|
| `modules/` | State (params, buffers), Megatron API shape | Kernel calls |
| `ops/` | Backend dispatch, argument marshalling, fallback | Param management, nn.Module concerns |
| `kernels/` | Lumen-specific Triton kernels | CK/ASM kernels (those go in AITER) |

## Dispatch Architecture

Use `try_backends()` from `lumen/ops/dispatch.py` to fall through backends on failure. It catches `RuntimeError`, `NotImplementedError`, `TypeError`, `ValueError`, Triton `CompilationError`, and `OutOfResources`.

**Backend priority:** ASM → CK → Triton → torch (last resort, not for production)

```python
from lumen.ops.dispatch import try_backends

def my_op(x, ...):
    return try_backends(
        lambda: _asm_path(x, ...),
        lambda: _ck_path(x, ...),
        lambda: _triton_path(x, ...),
    )
```

**Rules:**
- All AITER imports: guarded by `try/except (ImportError, OSError)` via `_probe_aiter_*()` functions
- New backend: add `@functools.lru_cache(maxsize=1)` probe in `dispatch.py`
- Always `torch.cuda.synchronize()` after GPU kernels before fallback

## FP8 Quantization

### Scaling Types

| Type | Backends (priority) | Backward Constraint |
|------|--------------------|--------------------|
| `delayed` | hipBLASLt → CK → Triton | Full FP8 bwd OK |
| `dynamic` | hipBLASLt → CK → Triton | Full FP8 bwd OK |
| `per_token` | Triton only | No FP8 wgrad (scale misalignment after transpose) |
| `blockwise` | CK → Triton | No FP8 wgrad |
| `mxfp8` | Triton only | No FP8 wgrad |

**Scale misalignment:** Per-tensor scalar scales survive `weight.t()`. Per-token `(N,1)`, blockwise `(N,K/bs)`, and mxfp8 block scales become misaligned — FP8 backward restricted to `["delayed", "dynamic", "none"]`.

### Deriving Constants

```python
# GOOD: derived from spec
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
scale = amax / FP8_E4M3_MAX

# BAD: magic number
scale = 448.0
```

## BF16 GEMM

All GEMM kernels use TN layout: `Y = A @ W^T`, weight is always `(N, K)`.

- `tuned_gemm` / `hipb_mm` / `F.linear` require matching dtypes — cast after dequant: `(w_fp8.to(bf16) * scale).to(bf16)`
- Triton `gemm_a16w16`: BLOCK_SIZE_K ≥ 128 on gfx942
- ASM: requires K % 64 == 0 and N % 64 == 0
- Backward dgrad: pass `weight.t().contiguous()` as `w`

## Normalization

- Backward: prefer Triton when `requires_grad=True` (CK fwd doesn't save intermediates for CK bwd)
- Fused norm+quant: available for all scaling types in both LayerNorm and RMSNorm
- ASM: `layernorm2d_with_add_asm` for LayerNorm only (no RMSNorm ASM)

## Testing Conventions

### Reference Implementations

Op-level tests compare against pure PyTorch references in `tests/ops/conftest.py`:

| Reference | For |
|-----------|-----|
| `attention_ref` | Multi-head attention (BSHD, f32 compute, causal mask, GQA) |
| `rmsnorm_ref` | RMSNorm |
| `layernorm_ref` | LayerNorm (`F.layer_norm`) |
| `grouped_gemm_ref` | Sequential per-expert GEMM |
| `cross_entropy_ref` | `F.cross_entropy` |

### Accuracy Metrics

- `compute_snr(x, y)`: Signal-to-Noise Ratio in dB. Forward: SNR > 20 dB, Backward: SNR > 15 dB.
- `check_close(a, b)`: Element-wise closeness with outlier tolerance (default 5%).
- Fixed seed via `seed_rng` fixture (autouse).

### Test Structure

```python
from conftest import SomeConfig, some_ref, compute_snr

CONFIGS = [SomeConfig(...), ...]

@pytest.mark.parametrize("config", CONFIGS, ids=[repr(c) for c in CONFIGS])
def test_some_op_fwd(config):
    out_ref = some_ref(...)
    out = lumen_op(...)
    snr = compute_snr(out_ref, out)
    assert snr > 20
```

## Coding Conventions

### Documentation

Do NOT describe Lumen modules as "drop-in replacements" for TE in docstrings/comments:

```python
# BAD
"""Drop-in replacement for TE's TEColumnParallelLinear."""

# GOOD
"""Column-parallel linear using Lumen GEMM."""
```

Exception: actual code referencing TE class names (imports, runtime logic) is fine.

### First-Principles Codegen

1. Clarify the problem in one sentence before writing code
2. Identify constraints (memory bandwidth, register pressure, API contract)
3. Derive solution from constraints, not analogy
4. Eliminate unnecessary abstraction — add a layer only when it removes duplication across ≥2 call sites

### Anti-Patterns

```python
# BAD: abstraction with one subclass
class QuantScaleManager(ABC): ...

# GOOD: plain function until second strategy appears
def compute_delayed_scale(amax_history, fp8_max): ...
```

```python
# BAD: copying TE API without understanding
def forward(self, inp, weight, bias=None, **te_kwargs): ...

# GOOD: API shaped by Lumen's own requirements
def forward(self, inp: Tensor, *, quant_config: QuantConfig) -> Tensor: ...
```

## TE Feature Parity

For the complete feature parity tracker, see [reference.md](reference.md).

**Current scorecard:** ~61 features total, ~46 supported (75%), 4 Lumen-only, ~3 missing, ~8 partial.

**Key remaining work:**
- CP A2A+P2P (missing)
- TP comm-GEMM overlap full pipelining (partial)
- MoE fused permute/unpermute via AITER (partial)
- Sliding window attention on Triton backend (partial)
