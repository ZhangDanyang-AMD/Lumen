"""Packing rules for the runtime expert placement."""

from lumen.models.qwen3_30b_a3b.fsdp.expert_balance import pack_slots


def _rank_loads(loads, slot, experts_per_rank):
    num_ranks = len(loads) // experts_per_rank
    totals = [0] * num_ranks
    for expert, dest in enumerate(slot):
        totals[dest // experts_per_rank] += loads[expert]
    return totals


def test_pack_slots_is_a_balanced_permutation():
    loads = [10, 10, 10, 10, 0, 0, 0, 0]
    slot = pack_slots(loads, experts_per_rank=2)
    assert sorted(slot) == list(range(8))
    for rank in range(4):
        assert sum(dest // 2 == rank for dest in slot) == 2
    packed = _rank_loads(loads, slot, 2)
    identity = _rank_loads(loads, list(range(8)), 2)
    assert max(packed) == 10
    assert max(identity) == 20


def test_pack_slots_breaks_load_ties_by_expert_id():
    # Equal positive loads alternate ranks. A zero load does not, because
    # seating it leaves every rank tied, so the lowest rank fills first.
    assert pack_slots([5, 5, 5, 5], experts_per_rank=2) == [0, 2, 1, 3]
    assert pack_slots([0, 0, 0, 0], experts_per_rank=2) == [0, 1, 2, 3]
