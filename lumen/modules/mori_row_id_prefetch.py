"""Build TE's permute row-id map before the MORI count sync so its launches overlap the wait."""

from __future__ import annotations

import inspect
import logging
import os

logger = logging.getLogger(__name__)

_ENV = "LUMEN_MORI_ROW_ID_PREFETCH"
_STATE = {"installed": False, "reason": None}
# The patched path relies on _permute_by_experts building the routing map, then syncing
# the counts, then calling TE's mask-map permute, which builds the row-id map itself.
_REQUIRED_PERMUTE_STEPS = (
    "self.dispatched_routing_map, self.dispatched_probs = fused_indices_to_multihot(",
    "num_out_tokens = int(self._synced_tokens_per_expert().sum())",
    "permute(",
)
_REQUIRED_TE_CALL = "triton_permutation.make_row_id_map(routing_map, num_tokens, num_experts)"


class _Prefetch:
    """One routing map's precomputed row-id map, handed to TE at most once."""

    def __init__(self, make_row_id_map):
        self.make_row_id_map = make_row_id_map
        self.routing_map = None
        self.version = None
        self.row_id_map = None

    def prefetch(self, routing_map):
        if routing_map is self.routing_map:
            return
        self.routing_map = routing_map
        self.version = routing_map._version
        self.row_id_map = self.make_row_id_map(
            routing_map, routing_map.shape[0], routing_map.shape[1]
        )

    def take(self, routing_map, num_tokens, num_experts):
        if (
            self.row_id_map is None
            or routing_map is not self.routing_map
            or routing_map._version != self.version
            or (num_tokens, num_experts) != tuple(routing_map.shape)
        ):
            return None
        row_id_map, self.row_id_map = self.row_id_map, None
        return row_id_map


def _eligible(manager, routing_map) -> bool:
    import torch

    return (
        getattr(manager, "permute_fusion", False)
        and not getattr(manager.config, "moe_router_padding_for_quantization", False)
        and isinstance(routing_map, torch.Tensor)
        and routing_map.is_cuda
        and routing_map.dim() == 2
    )


def install_mori_row_id_prefetch() -> bool:
    """Patch the exact Megatron MORI manager and TE permute path; any mismatch keeps both."""
    if _STATE["installed"]:
        return True
    if os.environ.get(_ENV, "1") == "0":
        _STATE["reason"] = f"{_ENV}=0"
        return False
    try:
        import transformer_engine.pytorch.permutation as te_permutation
        import transformer_engine.pytorch.triton.permutation as triton_permutation
        from megatron.core.transformer.moe import token_dispatcher
    except ImportError as exc:
        _STATE["reason"] = f"Megatron/TE imports failed: {exc!r}"
        return False
    manager = getattr(token_dispatcher, "_MoriManager", None)
    mask_map = getattr(te_permutation, "_moe_permute_mask_map", None)
    make_row_id_map = getattr(triton_permutation, "make_row_id_map", None)
    try:
        permute_source = inspect.getsource(manager._permute_by_experts)
        te_source = inspect.getsource(mask_map.forward)
        make_names = tuple(inspect.signature(make_row_id_map).parameters)
        sync = manager._synced_tokens_per_expert
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        _STATE["reason"] = f"MORI/TE permute sources unavailable: {exc!r}"
        return False
    if (
        any(step not in permute_source for step in _REQUIRED_PERMUTE_STEPS)
        or _REQUIRED_TE_CALL not in te_source
        or make_names != ("routing_map", "num_tokens", "num_experts")
        or tuple(inspect.signature(sync).parameters) != ("self",)
    ):
        _STATE["reason"] = "MORI _permute_by_experts or TE mask-map permute layout changed"
        logger.warning("MORI row-id prefetch unavailable: %s", _STATE["reason"])
        return False

    prefetch = _Prefetch(make_row_id_map)

    def _synced_tokens_per_expert(self):
        routing_map = getattr(self, "dispatched_routing_map", None)
        if _eligible(self, routing_map):
            prefetch.prefetch(routing_map)
        return sync(self)

    def _make_row_id_map(routing_map, num_tokens, num_experts):
        row_id_map = prefetch.take(routing_map, num_tokens, num_experts)
        if row_id_map is not None:
            return row_id_map
        return make_row_id_map(routing_map, num_tokens, num_experts)

    manager._synced_tokens_per_expert = _synced_tokens_per_expert
    triton_permutation.make_row_id_map = _make_row_id_map
    _STATE["installed"] = True
    _STATE["prefetch"] = prefetch
    logger.info("Installed MORI row-id map prefetch")
    return True


__all__ = ["install_mori_row_id_prefetch"]
