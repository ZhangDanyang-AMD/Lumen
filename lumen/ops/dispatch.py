###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Backend dispatcher with automatic ASM → CK → Triton fallback.

All backends are AITER implementations. No torch.nn.functional fallbacks.
Each operator registers its available backends via :func:`try_backends`.
On each call the dispatcher walks the priority chain and returns the
first successful result, logging fallbacks as warnings.
"""

import functools
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Backend(Enum):
    ASM = "asm"
    CK = "ck"
    TRITON = "triton"


FALLBACK_ORDER = [Backend.ASM, Backend.CK, Backend.TRITON]


# ---------------------------------------------------------------------------
# Lazy import helpers — detect available AITER sub-packages once
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _probe_aiter_ck_norm():
    """Check if AITER CK norm ops are available."""
    try:
        from aiter.ops.norm import layernorm2d_fwd as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_ck_rmsnorm():
    try:
        from aiter.ops.rmsnorm import rmsnorm2d_fwd_ck as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_norm():
    try:
        from aiter.ops.triton.normalization.norm import layer_norm as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_rmsnorm():
    try:
        from aiter.ops.triton.normalization.rmsnorm import rms_norm as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_ck_gemm():
    try:
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8 as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_gemm():
    try:
        from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8 as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_gemm_bf16():
    try:
        from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16 as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_asm_norm():
    try:
        from aiter.ops.norm import layernorm2d_with_add_asm as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_quant():
    """Check if AITER CK/HIP quant ops are available."""
    try:
        from aiter.ops.quant import per_tensor_quant_hip as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_quant():
    """Check if AITER Triton quant ops are available."""
    try:
        from aiter.ops.quant import per_tensor_quant_triton as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_fused_quant():
    try:
        from aiter.ops.triton.quant.fused_fp8_quant import fused_rms_fp8_per_tensor_static_quant as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_gmm():
    try:
        from aiter.ops.triton.gmm import gmm as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_deepgemm():
    try:
        from aiter.ops.deepgemm import deepgemm as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_gemm_mxfp8():
    """Check if AITER Triton MXFP8 GEMM is available."""
    try:
        from aiter.ops.triton.gemm.basic.gemm_mxfp8 import gemm_mxfp8 as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_moe_gemm_per_token():
    """Check if AITER fused per-token MOE GEMM is available."""
    try:
        from aiter.ops.triton.moe.moe_gemm_per_token import moe_gemm_per_token as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_moe_gemm_mxfp8():
    """Check if AITER fused MXFP8 MOE GEMM is available."""
    try:
        from aiter.ops.triton.moe.moe_gemm_mxfp8 import moe_gemm_mxfp8 as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


# ---------------------------------------------------------------------------
# Core fallback dispatcher
# ---------------------------------------------------------------------------


def try_backends(
    backends: List[Tuple[Backend, Callable]],
    *args,
    op_name: str = "op",
    **kwargs,
) -> Any:
    """Try each ``(backend, fn)`` pair in order; return first success.

    On ``RuntimeError`` or ``NotImplementedError`` from a backend, logs a
    warning and falls through to the next.  If all fail, raises the last
    exception.
    """
    last_exc = None
    for backend, fn in backends:
        try:
            return fn(*args, **kwargs)
        except (RuntimeError, NotImplementedError, TypeError) as exc:
            logger.warning(
                "%s: %s backend failed (%s), trying next...",
                op_name,
                backend.value,
                exc,
            )
            last_exc = exc
    raise RuntimeError(f"{op_name}: all AITER backends exhausted. Last error: {last_exc}") from last_exc


def build_fallback_chain(
    candidates: Dict[Backend, Optional[Callable]],
    order: List[Backend] = FALLBACK_ORDER,
) -> List[Tuple[Backend, Callable]]:
    """Build an ordered list of (backend, callable) from candidates.

    ``None`` values (unavailable backends) are skipped.
    """
    chain = []
    for b in order:
        fn = candidates.get(b)
        if fn is not None:
            chain.append((b, fn))
    return chain
