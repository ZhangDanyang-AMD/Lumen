# ruff: noqa: BLE001, F401
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dispatch pre-routed quantized experts without silently changing precision."""

import logging

from lumen.ops.dispatch import (
    Backend,
    _probe_aiter_pack_pre_routed_weights,
    _probe_aiter_pre_routed_a4w4,
    _probe_aiter_pre_routed_a4w4_bf16_main_grad,
    _probe_aiter_pre_routed_a4w4_wgrad,
    _probe_aiter_pre_routed_a4w4_wgrad_gluon,
    _probe_aiter_pre_routed_a8w4,
    _probe_aiter_pre_routed_a8w4f,
    _probe_aiter_pre_routed_mapped,
    _probe_aiter_pre_routed_w4a16,
    try_backends,
)

logger = logging.getLogger(__name__)

_PROBES = {
    "w4a16": ("_probe_aiter_pre_routed_w4a16", "moe_pre_routed_w4a16"),
    "a4w4": ("_probe_aiter_pre_routed_a4w4", "moe_pre_routed_a4w4"),
    "a8w4": ("_probe_aiter_pre_routed_a8w4", "moe_pre_routed_a8w4"),
    "a8w4f": ("_probe_aiter_pre_routed_a8w4f", "moe_pre_routed_a8w4f"),
}


def _resolve(recipe):
    """Look the probe up by name so tests can patch it on this module."""
    probe_name, call_name = _PROBES[recipe]
    return globals()[probe_name](), call_name


def pack_pre_routed_weights(w1, w2):
    """Pack expert weights for the quantized recipes; returns a caller-owned cache."""
    operation = globals()["_probe_aiter_pack_pre_routed_weights"]()
    if operation is None:
        raise RuntimeError("weight packing requires the AITER pre-routed MXFP4 API")
    return operation(w1, w2)


def _run(operation, name, x, scores, counts, w1, w2, packed=None):
    if operation is None:
        raise RuntimeError(f"{name} requires the AITER pre-routed MXFP4 API")
    # No BF16 fallback: failure must remain visible for this requested recipe.
    return try_backends(
        [(Backend.TRITON, operation)],
        x,
        scores,
        counts,
        w1,
        w2,
        packed,
        op_name=name,
    )


def pre_routed_w4a16(x, scores, counts, w1, w2, packed=None):
    operation, name = _resolve("w4a16")
    return _run(operation, name, x, scores, counts, w1, w2, packed)


def pre_routed_a4w4(
    x,
    scores,
    counts,
    w1,
    w2,
    packed=None,
    *,
    wgrad_mode="bf16",
    main_grads=None,
    wgrad_fallback="error",
    wgrad_backend="triton",
):
    """A4W4 experts with opt-in native Wgrad and caller-owned FP32 main_grads.

    Missing native capability raises unless wgrad_fallback='bf16' is explicit.
    Backward failures are not retried after possible writes to main_grad.
    """
    if wgrad_mode not in ("bf16", "mxfp4"):
        raise ValueError("wgrad_mode must be bf16 or mxfp4")
    if wgrad_fallback not in ("error", "bf16"):
        raise ValueError("wgrad_fallback must be error or bf16")
    if wgrad_backend not in ("triton", "gluon"):
        raise ValueError("wgrad_backend must be triton or gluon")
    if wgrad_mode == "bf16" and main_grads is None:
        operation, name = _resolve("a4w4")
        return _run(operation, name, x, scores, counts, w1, w2, packed)
    if wgrad_mode == "bf16":
        operation = _probe_aiter_pre_routed_a4w4_bf16_main_grad()
        if operation is None:
            raise RuntimeError(
                "BF16 Wgrad main_grad accumulation requires "
                "BF16_WGRAD_MAIN_GRAD_API_VERSION=1"
            )
        return try_backends(
            [(Backend.TRITON, operation)],
            x,
            scores,
            counts,
            w1,
            w2,
            packed,
            main_grads=main_grads,
            op_name="moe_pre_routed_a4w4:bf16_wgrad_main_grad",
        )
    probe = (
        _probe_aiter_pre_routed_a4w4_wgrad_gluon
        if wgrad_backend == "gluon"
        else _probe_aiter_pre_routed_a4w4_wgrad
    )
    operation = probe()
    if operation is None:
        if wgrad_fallback != "bf16":
            raise RuntimeError(
                f"native MXFP4 Wgrad ({wgrad_backend}) requires AITER "
                "MXFP4_WGRAD_API_VERSION=1 and the requested backend"
            )
        # No native graph exists yet, so this explicit pre-forward fallback is safe.
        logger.warning(
            "native MXFP4 Wgrad (%s) unavailable; explicit fallback to BF16 Wgrad",
            wgrad_backend,
        )
        operation, name = _resolve("a4w4")
        return _run(
            operation, name + ":explicit_bf16_wgrad", x, scores, counts, w1, w2, packed
        )
    # Backend selection is explicit; never retry a partial backward on another backend.
    backend = Backend.GLUON if wgrad_backend == "gluon" else Backend.TRITON
    label = "moe_pre_routed_a4w4:native_wgrad"
    if wgrad_backend == "gluon":
        label += ":gluon"
    return try_backends(
        [(backend, operation)],
        x,
        scores,
        counts,
        w1,
        w2,
        packed,
        main_grads=main_grads,
        op_name=label,
    )


def pre_routed_a8w4(x, scores, counts, w1, w2, packed=None):
    operation, name = _resolve("a8w4")
    return _run(operation, name, x, scores, counts, w1, w2, packed)


def pre_routed_a8w4f(x, scores, counts, w1, w2, packed=None):
    operation, name = _resolve("a8w4f")
    return _run(operation, name, x, scores, counts, w1, w2, packed)


def pre_routed_quantized(
    recipe,
    x,
    scores,
    counts,
    w1,
    w2,
    packed=None,
    *,
    wgrad_mode="bf16",
    main_grads=None,
    wgrad_fallback="error",
    wgrad_backend="triton",
):
    """Dispatch an explicit recipe; native MXFP4 Wgrad is supported only for a4w4."""
    if recipe == "a4w4":
        return pre_routed_a4w4(
            x,
            scores,
            counts,
            w1,
            w2,
            packed,
            wgrad_mode=wgrad_mode,
            main_grads=main_grads,
            wgrad_fallback=wgrad_fallback,
            wgrad_backend=wgrad_backend,
        )
    if wgrad_mode != "bf16" or main_grads is not None:
        raise ValueError("native MXFP4 Wgrad requires the a4w4 recipe")
    operation, name = _resolve(recipe)
    return _run(operation, name, x, scores, counts, w1, w2, packed)


def mapped_pre_routed_available():
    """Whether AITER exposes the exact mapped dual-quant contract Lumen requires."""
    return _probe_aiter_pre_routed_mapped() is not None


def pre_routed_quantized_mapped(
    recipe,
    source_tokens,
    scores,
    counts,
    gather_map,
    w1,
    w2,
    packed=None,
    *,
    wgrad_mode="bf16",
    main_grads=None,
    wgrad_fallback="error",
    wgrad_backend="triton",
):
    """Run expert-major MXFP4 work from unsorted source rows.

    The versioned AITER operation owns gather-aware forward quantization and
    source-order dX. Its output stays expert-major for the dispatcher to combine.
    MORI may reuse the source buffer after forward, so AITER must retain private
    backward operands rather than saving that mutable buffer by reference.
    """
    if recipe not in _PROBES:
        raise ValueError(f"Unsupported expert precision: {recipe}")
    if wgrad_mode not in ("bf16", "mxfp4"):
        raise ValueError("wgrad_mode must be bf16 or mxfp4")
    if wgrad_mode == "mxfp4" and recipe != "a4w4":
        raise ValueError("native MXFP4 Wgrad requires the a4w4 recipe")
    operation = _probe_aiter_pre_routed_mapped()
    if operation is None:
        raise RuntimeError(
            "mapped MORI handoff requires AITER MAPPED_DUAL_QUANT_API_VERSION=1"
        )
    # Once source reordering is skipped, a failed mapped call cannot safely
    # retry the non-mapped operation with a different input layout.
    return try_backends(
        [(Backend.TRITON, operation)],
        source_tokens,
        scores,
        counts,
        gather_map,
        w1,
        w2,
        packed,
        recipe=recipe,
        wgrad_mode=wgrad_mode,
        main_grads=main_grads,
        wgrad_fallback=wgrad_fallback,
        wgrad_backend=wgrad_backend,
        op_name="moe_pre_routed_quantized_mapped:v1",
    )


# Preserve the optional Round-80 dual-orientation weight pack.
_DUAL_PACK_PROBE = None


def _probe_aiter_pack_pre_routed_weights_dual():
    """Resolve the optional public dual-pack API once; None means unavailable."""
    global _DUAL_PACK_PROBE
    if _DUAL_PACK_PROBE is None:
        try:
            from aiter.ops.triton.gemm.grouped.sonic_mxfp4 import (
                pack_pre_routed_weights_dual as _op,
            )

            _DUAL_PACK_PROBE = _op
        except (
            Exception
        ) as exc:  # optional feature: fall back to the old loop, with a log
            logger.warning(
                "R80 dual weight pack unavailable (%r); using old 4-call loop", exc
            )
            _DUAL_PACK_PROBE = False
    return _DUAL_PACK_PROBE or None


def pack_pre_routed_weights_dual(w1, w2):
    """Optional one-read pack with the historical four-orientation contract.

    R80_DUAL_WEIGHT_PACK=1 opts in. Missing capability or pack failure logs
    and uses the historical loop without changing the numerical recipe.
    """
    operation = _probe_aiter_pack_pre_routed_weights_dual()
    if operation is None:
        return pack_pre_routed_weights(w1, w2)
    try:
        return operation(w1, w2)
    except Exception as exc:
        logger.warning(
            "R80 dual weight pack failed (%r); falling back to old 4-call loop", exc
        )
        return pack_pre_routed_weights(w1, w2)
