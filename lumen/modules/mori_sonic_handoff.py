"""Opt-in mapped handoff for Megatron's MORI flex dispatcher."""

from __future__ import annotations

import inspect
import logging
import os

from lumen.utils.sonic_moe_handoff import (
    mapped_handoff_enabled,
    mapped_handoff_scope,
    mapped_handoff_scope_for_source,
    publish_mapped_handoff,
    register_mapped_handoff_source,
    unregister_mapped_handoff_source,
)

logger = logging.getLogger(__name__)

_ENV = "LUMEN_MORI_SONIC_MXFP4_HANDOFF"
_MANAGER_SCOPE_ATTR = "_lumen_mapped_handoff_scope_v1"
_FALLBACK_NOTICES: set[str] = set()
_EXPECTED_SIGNATURES = {
    "_MoriManager.dispatch": (
        "self",
        "hidden_states",
        "async_finish",
        "allocate_on_comm_stream",
    ),
    "_MoriManager._permute_by_experts": ("self", "hidden_states"),
    "_MoriManager.combine": (
        "self",
        "hidden_states",
        "async_finish",
        "allocate_on_comm_stream",
    ),
}
_EMPTY = inspect.Parameter.empty
_EXPECTED_DEFAULTS = {
    "_MoriManager.dispatch": (_EMPTY, _EMPTY, True, True),
    "_MoriManager._permute_by_experts": (_EMPTY, _EMPTY),
    "_MoriManager.combine": (_EMPTY, _EMPTY, True, True),
}
_ROUTED_EXPERT_SIGNATURES = {
    False: (("self", "hidden_states", "probs"), (_EMPTY, _EMPTY, _EMPTY)),
    True: (
        ("self", "hidden_states", "probs", "residual"),
        (_EMPTY, _EMPTY, _EMPTY, _EMPTY),
    ),
}


def _notice(reason: str) -> None:
    if reason in _FALLBACK_NOTICES:
        return
    _FALLBACK_NOTICES.add(reason)
    logger.warning(
        "MORI mapped MXFP4 handoff unavailable (%s); "
        "using Megatron's BF16 expert permutation",
        reason,
    )


def _has_signature(
    operation,
    expected_names: tuple[str, ...],
    expected_defaults: tuple[object, ...],
) -> bool:
    try:
        parameters = tuple(inspect.signature(operation).parameters.values())
    except (TypeError, ValueError):
        return False
    return (
        tuple(parameter.name for parameter in parameters) == expected_names
        and tuple(parameter.default for parameter in parameters) == expected_defaults
        and all(
            parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in parameters
        )
    )


def _resolve_targets():
    try:
        from megatron.core.transformer.moe import moe_layer, token_dispatcher
        from megatron.core.transformer.moe.moe_utils import permute
    except (ImportError, AttributeError) as exc:
        return None, f"Megatron MORI imports failed: {exc!r}"

    manager = getattr(token_dispatcher, "_MoriManager", None)
    if manager is None or manager.__module__ != token_dispatcher.__name__:
        return (
            None,
            "exact megatron.core.transformer.moe.token_dispatcher._MoriManager missing",
        )
    if moe_layer.MoELayer.__module__ != moe_layer.__name__:
        return (
            None,
            "exact megatron.core.transformer.moe.moe_layer.MoELayer missing",
        )
    fused_indices_to_multihot = getattr(
        token_dispatcher, "fused_indices_to_multihot", None
    )
    checks = (
        ("_MoriManager.dispatch", manager.dispatch),
        ("_MoriManager._permute_by_experts", manager._permute_by_experts),
        ("_MoriManager.combine", manager.combine),
    )
    for name, operation in checks:
        if not _has_signature(
            operation,
            _EXPECTED_SIGNATURES[name],
            _EXPECTED_DEFAULTS[name],
        ):
            return None, f"{name} signature mismatch"

    routed_operation = moe_layer.MoELayer.routed_experts_compute
    routed_has_residual = next(
        (
            has_residual
            for has_residual, (names, defaults) in _ROUTED_EXPERT_SIGNATURES.items()
            if _has_signature(routed_operation, names, defaults)
        ),
        None,
    )
    if routed_has_residual is None:
        return None, "MoELayer.routed_experts_compute signature mismatch"
    return (
        manager,
        moe_layer.MoELayer,
        permute,
        fused_indices_to_multihot,
        routed_has_residual,
    ), ""


def _prepare_mapped_routing(
    manager,
    hidden_states,
    permute,
    fused_indices_to_multihot,
):
    import torch

    if manager.config.moe_router_padding_for_quantization:
        raise NotImplementedError("router padding requires Megatron's full permutation")
    if manager.router_dtype == "fp64":
        raise NotImplementedError("FP64 router scores are unsupported by Sonic MXFP4")
    if hidden_states.shape[0] >= 2**24:
        raise NotImplementedError("dispatch capacity exceeds exact FP32 row ids")

    if manager.permute_fusion:
        if not callable(fused_indices_to_multihot):
            raise NotImplementedError("fused_indices_to_multihot is unavailable")
        manager.dispatched_routing_map, multihot_probs = fused_indices_to_multihot(
            manager.dispatched_indices,
            manager.dispatched_probs,
            manager.num_local_experts,
        )
    else:
        manager.dispatched_routing_map, multihot_probs = manager._indices_to_multihot(
            manager.dispatched_indices,
            manager.dispatched_probs,
        )

    counts = manager._synced_tokens_per_expert()
    num_out_tokens = int(counts.sum())
    row_ids = torch.arange(
        hidden_states.shape[0],
        dtype=torch.float32,
        device=hidden_states.device,
    ).unsqueeze(1)
    (
        permuted_rows,
        permuted_probs,
        reversed_mapping,
        _pad_offsets,
        _padded_counts,
    ) = permute(
        row_ids,
        manager.dispatched_routing_map,
        probs=multihot_probs,
        num_out_tokens=num_out_tokens,
        fused=manager.permute_fusion,
    )
    gather_map = permuted_rows.reshape(-1).to(torch.int32).contiguous()
    if gather_map.numel() != num_out_tokens:
        raise RuntimeError("small MORI permutation returned the wrong row count")
    return counts, gather_map, permuted_probs, reversed_mapping


def install_mori_sonic_handoff() -> bool:
    """Patch the exact Megatron MORI manager when the opt-in contract is available."""
    if os.environ.get(_ENV, "0") != "1":
        return False

    from lumen.ops.moe.mxfp4_sonic import mapped_pre_routed_available

    if not mapped_pre_routed_available():
        _notice("AITER mapped dual-quant API version 1 is unavailable")
        return False

    targets, reason = _resolve_targets()
    if targets is None:
        _notice(reason)
        return False
    (
        manager_type,
        moe_layer_type,
        permute,
        fused_indices_to_multihot,
        routed_has_residual,
    ) = targets
    if getattr(manager_type, "_lumen_mapped_handoff_v1", False):
        return True

    original_permute = manager_type._permute_by_experts
    original_routed_experts_compute = moe_layer_type.routed_experts_compute

    def _patched_permute(self, hidden_states):
        scope = mapped_handoff_scope_for_source(hidden_states)
        if scope is None:
            scope = getattr(self, _MANAGER_SCOPE_ATTR, None)
        if not mapped_handoff_enabled(scope):
            return original_permute(self, hidden_states)
        try:
            counts, gather_map, permuted_probs, reversed_mapping = (
                _prepare_mapped_routing(
                    self,
                    hidden_states,
                    permute,
                    fused_indices_to_multihot,
                )
            )
        except (
            AssertionError,
            AttributeError,
            NotImplementedError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            _notice(f"small routing permutation failed: {exc}")
            return original_permute(self, hidden_states)

        self.hidden_shape_before_permute = hidden_states.shape
        self.reversed_mapping_for_combine = reversed_mapping
        self.dispatched_probs = permuted_probs
        publish_mapped_handoff(
            hidden_states,
            permuted_probs,
            counts,
            gather_map,
            scope=scope,
        )
        return hidden_states

    def _call_routed_experts(self, hidden_states, probs, residual=_EMPTY):
        call_args = (
            (self, hidden_states, probs, residual)
            if routed_has_residual
            else (self, hidden_states, probs)
        )
        manager = getattr(self.token_dispatcher, "_comm_manager", None)
        if type(manager) is not manager_type:
            return original_routed_experts_compute(*call_args)
        capability = getattr(self.experts, "mori_mapped_handoff_capability", None)
        if not callable(capability):
            _notice(f"{type(self.experts).__name__} is not a SonicMoEExperts consumer")
            return original_routed_experts_compute(*call_args)
        enabled, reason = capability()
        if not enabled:
            _notice(reason)
            return original_routed_experts_compute(*call_args)
        with mapped_handoff_scope(True) as scope:
            if hasattr(manager, _MANAGER_SCOPE_ATTR):
                raise RuntimeError("MORI manager already owns a mapped handoff scope")
            register_mapped_handoff_source(scope, hidden_states)
            setattr(manager, _MANAGER_SCOPE_ATTR, scope)
            try:
                return original_routed_experts_compute(*call_args)
            finally:
                delattr(manager, _MANAGER_SCOPE_ATTR)
                unregister_mapped_handoff_source(scope, hidden_states)

    if routed_has_residual:

        def _patched_routed_experts_compute(self, hidden_states, probs, residual):
            return _call_routed_experts(self, hidden_states, probs, residual)

    else:

        def _patched_routed_experts_compute(self, hidden_states, probs):
            return _call_routed_experts(self, hidden_states, probs)

    manager_type._permute_by_experts = _patched_permute
    moe_layer_type.routed_experts_compute = _patched_routed_experts_compute
    manager_type._lumen_mapped_handoff_v1 = True
    logger.info("Installed MORI-to-Sonic mapped MXFP4 handoff v1")
    return True


__all__ = ["install_mori_sonic_handoff"]
