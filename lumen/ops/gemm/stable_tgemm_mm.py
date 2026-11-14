###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""BF16 ``tgemm.mm`` padded onto a splitK=1 kernel bucket.

Some ``(M, N, K)`` shapes resolve via ``getPaddedM`` to a tuned row with
``splitK > 1``. The atomic reduction used by those FlyDSL kernels is
non-deterministic in BF16. Padding the token dimension to the next stable M
forces a splitK=1 kernel without changing the result for the original rows.

``call_stable_tgemm_mm`` is the only copy of that scan. Training calls it through
:func:`stable_tgemm_mm`. ATOM and vLLM call ``TunedGemm.mm`` directly, so
``lumenrl``'s ``sitecustomize`` wraps that method with the same function and
marks the wrapper ``_lumen_stable_tgemm``. A marked ``mm`` is already padded, and
:func:`stable_tgemm_mm` calls it once.
"""

from __future__ import annotations

import os

# Per-shape cache: (M, N, K) -> stable_M with splitK <= 1.
_stable_m_cache: dict[tuple[int, int, int], int] = {}


def call_stable_tgemm_mm(raw_mm, gemm, x, weight, bias, otype):
    """Call unpatched ``raw_mm`` with M padded onto a splitK=1 bucket.

    ``raw_mm`` is the unbound ``TunedGemm.mm`` from before any wrapper. It is
    invoked as ``raw_mm(gemm, x, weight, bias, otype=otype)``. Passing the
    wrapper back in here recurses.

    ``x`` may be ``(B, T, K)`` from Hugging Face or ``(M, K)`` from ATOM.
    Every leading dimension is the token count because each row is independent.
    The lookup is cached per ``(M, N, K)``.
    """
    import torch
    from aiter.tuned_gemm import get_GEMM_A16W16_config

    m = x.shape[0] if x.dim() == 2 else x.numel() // x.shape[-1]
    n, k = weight.shape
    dtype_str = str(x.dtype)
    orig_shape = x.shape
    x_2d = x.reshape(m, k)

    key = (m, n, k)
    stable_m = _stable_m_cache.get(key)
    if stable_m is None:
        # Walk upward from m until we find an M with splitK <= 1.
        candidate = m
        for _ in range(512):
            cfg = get_GEMM_A16W16_config(
                candidate, n, k, bias is not None, dtype_str, dtype_str
            )
            sk = cfg.get("splitK") if cfg else None
            if sk is None or int(sk) <= 1:
                break
            candidate += 1
        stable_m = candidate
        _stable_m_cache[key] = stable_m

    if stable_m == m:
        out = raw_mm(gemm, x_2d, weight, bias, otype=otype)
        return out.reshape(*orig_shape[:-1], n)

    pad = x_2d.new_zeros(stable_m - m, k)
    x_padded = torch.cat([x_2d, pad], dim=0)
    out_padded = raw_mm(gemm, x_padded, weight, bias, otype=otype)
    return out_padded[:m].reshape(*orig_shape[:-1], n).contiguous()


def stable_tgemm_mm(tgemm, x, weight, bias, otype):
    """Call ``tgemm.mm`` with M padded to the nearest splitK=1 stable bucket.

    Gated by ``LUMENRL_STABLE_GEMM=1``. When the env var is absent or 0,
    falls through directly to ``tgemm.mm``.

    A ``_lumen_stable_tgemm`` mark means this ``mm`` already pads, which is how the
    ATOM and vLLM workers install :func:`call_stable_tgemm_mm`. Calling the scan
    again would pad a second time.
    """
    if os.environ.get("LUMENRL_STABLE_GEMM", "0") != "1":
        return tgemm.mm(x, weight, bias, otype=otype)
    mm = getattr(type(tgemm), "mm", None)
    if getattr(mm, "_lumen_stable_tgemm", False) or getattr(tgemm.mm, "_lumen_stable_tgemm", False):
        return tgemm.mm(x, weight, bias, otype=otype)
    return call_stable_tgemm_mm(mm, tgemm, x, weight, bias, otype)
