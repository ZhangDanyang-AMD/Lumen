###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Static per-layer expert placement for SonicMoE layers behind the MORI dispatcher.

Megatron gives EP rank ``r`` the contiguous expert slots ``[r * L, (r + 1) * L)``. A
placement table moves each logical expert to another slot so that the heavy experts of a
layer spread over the EP ranks. ``LUMEN_EXPERT_PLACEMENT`` names a JSON file whose
``slot[layer][logical_expert]`` is the physical slot (layer is ``layer_number - 1``).

The router computes probabilities, the routing map, aux loss, z-loss and their logging in
logical expert ids; only its returned ``probs``/``routing_map`` columns are reordered into
physical slots, inside the router forward, so the graphed router replays the reorder too.
The expert weights are reordered once, while the checkpoint is loaded (see
``SonicMoEExperts.set_expert_layout``).
"""

from __future__ import annotations

import json
import os

import torch

_ENV = "LUMEN_EXPERT_PLACEMENT"
_TABLES: dict[str, list[list[int]]] = {}


def placement_table(path: str | None = None) -> list[list[int]] | None:
    """``slot[layer][logical]`` from ``path`` or ``LUMEN_EXPERT_PLACEMENT``; None when unset."""
    path = path or os.environ.get(_ENV)
    if not path:
        return None
    if path not in _TABLES:
        with open(path) as f:
            data = json.load(f)
        _TABLES[path] = [list(map(int, row)) for row in data["slot"]]
    return _TABLES[path]


def physical_to_logical(slot: list[int]) -> list[int]:
    """Invert one layer's ``slot[logical] -> physical`` map."""
    if sorted(slot) != list(range(len(slot))):
        raise ValueError(f"expert placement row is not a permutation of {len(slot)} experts")
    layout = [0] * len(slot)
    for logical, physical in enumerate(slot):
        layout[physical] = logical
    return layout


def _install_router_reorder() -> None:
    from megatron.core.transformer.moe.router import TopKRouter

    if getattr(TopKRouter, "_lumen_expert_placement", False):
        return
    routing = TopKRouter.routing

    def placed_routing(self, logits, *args, **kwargs):
        probs, routing_map = routing(self, logits, *args, **kwargs)
        layout = getattr(self, "lumen_expert_layout", None)
        if layout is None:
            return probs, routing_map
        return probs.index_select(1, layout), routing_map.index_select(1, layout)

    TopKRouter.routing = placed_routing
    TopKRouter._lumen_expert_placement = True


def install_expert_placement(moe_layer, table: list[list[int]] | None = None) -> bool:
    """Place one MoE layer's experts by the table; a no-op when no table is configured."""
    table = table if table is not None else placement_table()
    if table is None:
        return False
    layer = getattr(moe_layer, "layer_number", None)
    if layer is None or not 1 <= layer <= len(table):
        raise ValueError(f"expert placement has {len(table)} layers, MoE layer_number {layer}")
    layout = physical_to_logical(table[layer - 1])
    router = moe_layer.router
    if len(layout) != router.config.num_moe_experts:
        raise ValueError(
            f"expert placement row has {len(layout)} experts, model has "
            f"{router.config.num_moe_experts}"
        )
    moe_layer.experts.set_expert_layout(layout)
    router.register_buffer(
        "lumen_expert_layout",
        torch.tensor(layout, dtype=torch.long, device=router.weight.device),
        persistent=False,
    )
    _install_router_reorder()
    return True


__all__ = ["install_expert_placement", "physical_to_logical", "placement_table"]
