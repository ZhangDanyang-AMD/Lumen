"""Runtime expert placement for Qwen3 MoE.

Each rank still owns a contiguous slot range. ``slot[original]`` is the
dispatch id. The map is a permutation computed from token counts collected
during training, not a table baked into the source. A token keeps the expert
it routed to; only the rank that owns that expert changes.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

_COUNTS: dict[int, torch.Tensor] = {}
_SLOTS: dict[int, list[int]] | None = None


def pack_slots(loads: list[int], experts_per_rank: int) -> list[int]:
    """Pack experts onto ranks so each rank has the same count and a similar load.

    The busiest expert is seated first, on the rank with the smallest load so
    far. Equal loads go to the lower rank index, and equal expert loads keep
    the smaller expert id first. Seats inside a rank are the contiguous range
    ``[rank * experts_per_rank, (rank + 1) * experts_per_rank)``, ordered by
    the original expert id.
    """
    num_experts = len(loads)
    if experts_per_rank <= 0 or num_experts % experts_per_rank != 0:
        raise ValueError(
            f"cannot pack {num_experts} experts into groups of {experts_per_rank}"
        )
    num_ranks = num_experts // experts_per_rank
    order = sorted(range(num_experts), key=lambda expert: (-int(loads[expert]), expert))
    rank_load = [0] * num_ranks
    members: list[list[int]] = [[] for _ in range(num_ranks)]
    for expert in order:
        rank = min(
            (item for item in range(num_ranks) if len(members[item]) < experts_per_rank),
            key=lambda item: (rank_load[item], item),
        )
        members[rank].append(expert)
        rank_load[rank] += int(loads[expert])
    slot = [0] * num_experts
    for rank, group in enumerate(members):
        for local, expert in enumerate(sorted(group)):
            slot[expert] = rank * experts_per_rank + local
    return slot


def accumulate_layer_counts(layer: int, counts: torch.Tensor) -> None:
    """Add one forward's per-expert token counts. ``counts`` is indexed by original id."""
    if counts.ndim != 1:
        raise RuntimeError(f"expert counts must be a vector, got shape {tuple(counts.shape)}")
    values = counts.detach().to(dtype=torch.float64)
    stored = _COUNTS.get(layer)
    if stored is None:
        _COUNTS[layer] = values
        return
    if stored.shape != values.shape:
        raise RuntimeError(
            f"layer {layer} expert counts changed from {tuple(stored.shape)} "
            f"to {tuple(values.shape)}"
        )
    stored.add_(values)


def slots_ready() -> bool:
    return _SLOTS is not None


def expert_slot(layer: int) -> list[int]:
    """Return original_expert -> contiguous slot for one decoder layer."""
    if _SLOTS is None or layer not in _SLOTS:
        raise RuntimeError(f"expert placement for layer {layer} has not been computed")
    return _SLOTS[layer]


def inverse_ids(layer: int) -> list[int]:
    """Return contiguous slot -> original expert id."""
    slot = expert_slot(layer)
    inverse = [0] * len(slot)
    for original, new_slot in enumerate(slot):
        inverse[new_slot] = original
    if sorted(inverse) != list(range(len(slot))):
        raise RuntimeError(f"expert placement for layer {layer} is not a permutation")
    return inverse


def commit_slots(table: dict[int, list[int]]) -> None:
    """Publish a placement. Routing must not observe it before the weights move."""
    global _SLOTS
    _SLOTS = table


def build_slots(
    layers: list[int],
    experts_per_rank: int,
) -> tuple[dict[int, list[int]], float]:
    """All-reduce the collected counts and pack one permutation per layer.

    Returns the slot table and the worst rank's load as a multiple of the
    mean. Every rank must call this. The table is not visible to routing
    until ``commit_slots``.
    """
    if not layers:
        raise RuntimeError("expert remap has no layers to place")
    missing = [layer for layer in layers if layer not in _COUNTS]
    if missing:
        preview = ", ".join(str(layer) for layer in missing[:8])
        raise RuntimeError(f"expert remap is missing token counts for layers {preview}")
    ref = _COUNTS[layers[0]]
    stacked = torch.stack(
        [_COUNTS[layer].to(device=ref.device, dtype=torch.float64) for layer in layers]
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stacked)
    table: dict[int, list[int]] = {}
    worst = 1.0
    num_experts = int(stacked.shape[1])
    num_ranks = num_experts // experts_per_rank
    for row, layer in enumerate(layers):
        loads = [int(round(float(value))) for value in stacked[row].tolist()]
        slot = pack_slots(loads, experts_per_rank)
        if sorted(slot) != list(range(num_experts)):
            raise RuntimeError(f"packed placement for layer {layer} is not a permutation")
        table[layer] = slot
        rank_load = [0] * num_ranks
        for expert, dest in enumerate(slot):
            rank_load[dest // experts_per_rank] += loads[expert]
        mean = sum(rank_load) / float(num_ranks)
        if mean > 0:
            worst = max(worst, max(rank_load) / mean)
    _COUNTS.clear()
    return table, worst
