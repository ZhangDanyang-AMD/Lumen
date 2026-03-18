###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for lumen.ops.dispatch — backend dispatcher and fallback chain.

Covers:
  - Backend enum values
  - FALLBACK_ORDER ordering
  - build_fallback_chain: filters None, preserves order
  - try_backends: returns first success
  - try_backends: falls through on RuntimeError
  - try_backends: raises when all backends fail
  - Probe functions: lru_cache returns consistent bool
"""

import pytest

from lumen.ops.dispatch import (
    FALLBACK_ORDER,
    Backend,
    build_fallback_chain,
    try_backends,
)

# ===================================================================
# Backend enum
# ===================================================================


def test_backend_values():
    assert Backend.ASM.value == "asm"
    assert Backend.CK.value == "ck"
    assert Backend.TRITON.value == "triton"


def test_fallback_order():
    assert FALLBACK_ORDER == [Backend.ASM, Backend.CK, Backend.TRITON]


# ===================================================================
# build_fallback_chain
# ===================================================================


def test_build_chain_filters_none():
    """None values (unavailable backends) are skipped."""
    candidates = {
        Backend.ASM: None,
        Backend.CK: lambda: "ck_result",
        Backend.TRITON: lambda: "triton_result",
    }
    chain = build_fallback_chain(candidates)
    assert len(chain) == 2
    assert chain[0][0] == Backend.CK
    assert chain[1][0] == Backend.TRITON


def test_build_chain_preserves_order():
    """Chain follows FALLBACK_ORDER."""

    def fn_a():
        return "a"

    def fn_b():
        return "b"

    def fn_c():
        return "c"

    candidates = {
        Backend.TRITON: fn_c,
        Backend.ASM: fn_a,
        Backend.CK: fn_b,
    }
    chain = build_fallback_chain(candidates)
    assert [b for b, _ in chain] == [Backend.ASM, Backend.CK, Backend.TRITON]


def test_build_chain_empty():
    """All-None candidates produce an empty chain."""
    candidates = {Backend.ASM: None, Backend.CK: None, Backend.TRITON: None}
    chain = build_fallback_chain(candidates)
    assert chain == []


def test_build_chain_custom_order():
    """Custom order is respected."""

    def fn():
        return "ok"

    candidates = {Backend.ASM: fn, Backend.CK: fn, Backend.TRITON: fn}
    chain = build_fallback_chain(candidates, order=[Backend.TRITON, Backend.ASM])
    assert [b for b, _ in chain] == [Backend.TRITON, Backend.ASM]


# ===================================================================
# try_backends
# ===================================================================


def test_try_backends_first_success():
    """Returns result from first successful backend."""
    chain = [
        (Backend.ASM, lambda: "asm_ok"),
        (Backend.CK, lambda: "ck_ok"),
    ]
    result = try_backends(chain, op_name="test")
    assert result == "asm_ok"


def test_try_backends_fallthrough():
    """Falls through on RuntimeError to next backend."""

    def fail_asm():
        raise RuntimeError("ASM not supported")

    chain = [
        (Backend.ASM, fail_asm),
        (Backend.CK, lambda: "ck_ok"),
    ]
    result = try_backends(chain, op_name="test")
    assert result == "ck_ok"


def test_try_backends_all_fail():
    """Raises RuntimeError when all backends exhausted."""

    def fail():
        raise RuntimeError("nope")

    chain = [
        (Backend.ASM, fail),
        (Backend.CK, fail),
        (Backend.TRITON, fail),
    ]
    with pytest.raises(RuntimeError, match="all AITER backends exhausted"):
        try_backends(chain, op_name="test")


def test_try_backends_not_implemented():
    """NotImplementedError triggers fallback."""

    def not_impl():
        raise NotImplementedError("missing")

    chain = [
        (Backend.ASM, not_impl),
        (Backend.TRITON, lambda: "triton_ok"),
    ]
    result = try_backends(chain, op_name="test")
    assert result == "triton_ok"


def test_try_backends_type_error():
    """TypeError triggers fallback."""

    def bad_types():
        raise TypeError("wrong args")

    chain = [
        (Backend.ASM, bad_types),
        (Backend.CK, lambda: "ck_ok"),
    ]
    result = try_backends(chain, op_name="test")
    assert result == "ck_ok"


def test_try_backends_value_error():
    """ValueError triggers fallback."""

    def bad_value():
        raise ValueError("bad config")

    chain = [
        (Backend.ASM, bad_value),
        (Backend.TRITON, lambda: "ok"),
    ]
    result = try_backends(chain, op_name="test")
    assert result == "ok"


def test_try_backends_passes_args():
    """Arguments are forwarded to backend callables."""

    def backend_fn(a, b, c=None):
        return a + b + (c or 0)

    chain = [(Backend.ASM, backend_fn)]
    result = try_backends(chain, 1, 2, c=3, op_name="test")
    assert result == 6
