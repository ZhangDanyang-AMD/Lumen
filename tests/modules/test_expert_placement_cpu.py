"""Static expert placement: router column reorder and layout-aware expert checkpoints."""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn


def _import_modules():
    try:
        from lumen.modules import expert_placement
        from lumen.modules.sonic_moe import SonicMoEExperts
    except Exception:
        # lumen/__init__ imports AITER ops that need a GPU; load just these modules by path, then
        # restore sys.modules so test files collected in the same session keep their lumen modules.
        def owned(name):
            return name == "lumen" or name.startswith(("lumen.", "aiter"))

        saved = {name: module for name, module in sys.modules.items() if owned(name)}
        for name in saved:
            del sys.modules[name]
        root = Path(__file__).resolve().parents[2] / "lumen"
        for name, path in (("lumen", root), ("lumen.utils", root / "utils"), ("lumen.modules", root / "modules")):
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
        try:
            from lumen.modules import expert_placement
            from lumen.modules.sonic_moe import SonicMoEExperts
        finally:
            for name in [m for m in sys.modules if owned(m)]:
                del sys.modules[name]
            sys.modules.update(saved)
    return expert_placement, SonicMoEExperts


placement, SonicMoEExperts = _import_modules()

HIDDEN, FFN = 8, 4


def _source(local, group=None, seed=0):
    gen = torch.Generator().manual_seed(seed)
    module = nn.Module()
    module.config = SimpleNamespace(
        gated_linear_unit=True, add_bias_linear=False, expert_tensor_parallel_size=1
    )
    module.num_local_experts = local
    module.ep_group = group
    module.local_experts = nn.ModuleList()
    for _ in range(local):
        expert = nn.Module()
        expert.linear_fc1 = nn.Linear(HIDDEN, 2 * FFN, bias=False)
        expert.linear_fc2 = nn.Linear(FFN, HIDDEN, bias=False)
        for linear in (expert.linear_fc1, expert.linear_fc2):
            linear.weight.data = torch.randn(linear.weight.shape, generator=gen)
        module.local_experts.append(expert)
    return module


def _experts(local, layout=None, group=None, seed=0):
    model = SonicMoEExperts(_source(local, group, seed))
    if layout is not None:
        model.set_expert_layout(layout)
    return model


def _logical_weights(num_experts):
    """Expert e's w1/w2 are filled with e, so a slot's value names the expert it holds."""
    w1 = torch.arange(num_experts, dtype=torch.float32).view(-1, 1, 1).expand(-1, HIDDEN, 2 * FFN)
    w2 = torch.arange(num_experts, dtype=torch.float32).view(-1, 1, 1).expand(-1, FFN, HIDDEN)
    return w1.contiguous(), w2.contiguous()


def _held(model):
    w1 = model.w1.detach()[:, 0, 0].long().tolist()
    w2 = model.w2.detach()[:, 0, 0].long().tolist()
    assert w1 == w2
    return w1


@pytest.fixture(autouse=True)
def _no_megatron_args(monkeypatch):
    training = types.ModuleType("megatron.training")

    def get_args():
        raise AssertionError("args not initialized")

    training.get_args = get_args
    monkeypatch.setitem(sys.modules, "megatron.training", training)


def test_physical_to_logical_inverts_slot_table(tmp_path, monkeypatch):
    slot = [2, 0, 3, 1]
    assert placement.physical_to_logical(slot) == [1, 3, 0, 2]
    with pytest.raises(ValueError):
        placement.physical_to_logical([0, 0, 1, 2])
    path = tmp_path / "table.json"
    path.write_text(json.dumps({"slot": [slot, [0, 1, 2, 3]]}))
    monkeypatch.setenv("LUMEN_EXPERT_PLACEMENT", str(path))
    assert placement.placement_table() == [slot, [0, 1, 2, 3]]
    monkeypatch.delenv("LUMEN_EXPERT_PLACEMENT")
    assert placement.placement_table() is None


def _fake_router_module(monkeypatch):
    class TopKRouter(nn.Module):
        def __init__(self, num_experts):
            super().__init__()
            self.config = SimpleNamespace(num_moe_experts=num_experts)
            self.weight = nn.Parameter(torch.zeros(num_experts, HIDDEN))
            self.seen = None

        def routing(self, logits, padding_mask=None):
            self.seen = logits
            probs = torch.softmax(logits, dim=-1)
            return probs, probs > 0.3

    module = types.ModuleType("megatron.core.transformer.moe.router")
    module.TopKRouter = TopKRouter
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.moe.router", module)
    return TopKRouter


def test_router_returns_physical_columns_after_logical_routing(monkeypatch):
    TopKRouter = _fake_router_module(monkeypatch)
    layer = nn.Module()
    layer.layer_number = 2
    layer.router = TopKRouter(4)
    layer.experts = _experts(4)
    table = [[0, 1, 2, 3], [2, 0, 3, 1]]
    assert placement.install_expert_placement(layer, table)

    logits = torch.randn(5, 4, requires_grad=True)
    probs, routing_map = layer.router.routing(logits)
    ref = torch.softmax(logits, dim=-1)
    layout = [1, 3, 0, 2]
    assert layer.router.seen is logits
    torch.testing.assert_close(probs, ref[:, layout])
    assert torch.equal(routing_map, (ref > 0.3)[:, layout])
    (probs * torch.arange(4.0)).sum().backward()
    expected = torch.autograd.grad((ref[:, layout] * torch.arange(4.0)).sum(), logits)[0]
    torch.testing.assert_close(logits.grad, expected)
    assert layer.experts._expert_layout == layout
    assert "lumen_expert_layout" not in layer.router.state_dict()


def _logical_state(num_experts):
    w1, w2 = _logical_weights(num_experts)
    return {"w1": w1, "w2": w2}


def test_logical_checkpoint_loads_into_placed_slots_and_round_trips():
    layout = [2, 0, 3, 1]
    placed = _experts(4, layout)
    placed.load_state_dict(_logical_state(4))
    assert _held(placed) == layout

    saved = placed.state_dict()
    assert saved["expert_layout"].tolist() == layout

    same = _experts(4, layout, seed=1)
    same.load_state_dict(saved)
    assert _held(same) == layout

    plain = _experts(4, seed=2)
    plain.load_state_dict(dict(saved))
    assert _held(plain) == [0, 1, 2, 3]
    assert "expert_layout" not in plain.state_dict()

    other = _experts(4, [3, 2, 1, 0], seed=3)
    other.load_state_dict(dict(saved))
    assert _held(other) == [3, 2, 1, 0]
    assert other.state_dict()["expert_layout"].tolist() == [3, 2, 1, 0]


def test_reload_of_the_same_state_dict_does_not_permute_twice():
    layout = [1, 2, 3, 0]
    model = _experts(4, layout)
    state = _logical_state(4)
    model.load_state_dict(state)
    model.load_state_dict(state)
    assert _held(model) == layout


def test_layout_change_with_optimizer_state_is_refused(monkeypatch):
    sys.modules["megatron.training"].get_args = lambda: SimpleNamespace(
        finetune=False, no_load_optim=False
    )
    model = _experts(4, [1, 0, 3, 2])
    with pytest.raises(RuntimeError, match="optimizer"):
        model.load_state_dict(_logical_state(4))

    resumed = _experts(4, [1, 0, 3, 2])
    saved = {**_logical_state(4), "expert_layout": torch.tensor([1, 0, 3, 2])}
    resumed.load_state_dict(saved)
    assert _held(resumed) == [0, 1, 2, 3]  # already in the saved layout: loaded as stored


def test_layout_rejects_wrong_size():
    with pytest.raises(ValueError):
        _experts(4, [0, 1, 2])


def _ep_worker(rank, world, init_file, layout, out):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        local = len(layout) // world
        w1, w2 = _logical_weights(len(layout))
        logical = {"w1": w1[rank * local : (rank + 1) * local], "w2": w2[rank * local : (rank + 1) * local]}
        group = dist.group.WORLD
        placed = _experts(local, layout, group=group)
        placed.load_state_dict(logical)
        saved = placed.state_dict()
        plain = _experts(local, group=group, seed=5)
        plain.load_state_dict(dict(saved))
        torch.save({"placed": _held(placed), "plain": _held(plain)}, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


def test_expert_weights_move_across_ep_ranks(tmp_path):
    world, layout = 4, [5, 2, 7, 0, 3, 6, 1, 4]
    mp.start_processes(
        _ep_worker,
        args=(world, str(tmp_path / "init"), layout, str(tmp_path / "out")),
        nprocs=world,
        start_method="fork",
    )
    for rank in range(world):
        held = torch.load(tmp_path / f"out.{rank}")
        assert held["placed"] == layout[2 * rank : 2 * rank + 2]
        assert held["plain"] == [2 * rank, 2 * rank + 1]


def test_distributed_checkpoint_keys_name_logical_experts():
    pytest.importorskip("megatron.core.dist_checkpointing")
    layout = [2, 0, 3, 1]
    model = _experts(4, layout)
    model.load_state_dict(_logical_state(4))
    factory = model.sharded_state_dict(prefix="experts.")["experts.w2"]
    shards = factory.build_fn(factory.key, factory.data, factory.replica_id, None)
    offsets = [shard.global_offset[0] for shard in shards]
    assert offsets == layout
    for shard, expert in zip(shards, layout):
        assert torch.all(shard.data == expert)
    merged = factory.merge_fn([shard.data for shard in shards])
    assert model._sharded_layout_loaded
    fresh = _experts(4, layout, seed=4)
    fresh._sharded_layout_loaded = True
    fresh.load_state_dict({"w1": model.w1.detach().clone(), "w2": merged})
    assert _held(fresh) == layout
