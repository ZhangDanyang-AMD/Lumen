"""Mocked CPU integration for MORI's mapped dispatcher handoff."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _module(name):
    module = ModuleType(name)
    module.__path__ = []
    return module


class Source:
    shape = (3, 8)


class GatherMap:
    _version = 4

    @staticmethod
    def numel():
        return 5


@pytest.fixture
def mori(monkeypatch):
    torch = _module("torch")
    distributed = _module("torch.distributed")
    distributed.ProcessGroup = object
    torch.distributed = distributed
    torch.float32 = "float32"
    torch.long = "long"
    torch.tensor = mock.Mock(side_effect=lambda value, **kwargs: (value, kwargs))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)

    for name in (
        "megatron",
        "megatron.core",
        "megatron.core.process_groups_config",
        "megatron.core.transformer",
        "megatron.core.transformer.moe",
        "megatron.core.transformer.moe.moe_utils",
        "megatron.core.transformer.moe.token_dispatcher",
        "megatron.core.transformer.transformer_config",
        "lumen",
        "lumen.utils",
        "lumen.utils.sonic_moe_handoff",
    ):
        monkeypatch.setitem(sys.modules, name, _module(name))

    core = sys.modules["megatron.core"]
    core.parallel_state = _module("megatron.core.parallel_state")
    sys.modules["megatron.core.process_groups_config"].ProcessGroupCollection = object
    sys.modules["megatron.core.transformer.moe.moe_utils"].sort_chunks_by_idxs = (
        mock.Mock()
    )
    sys.modules["megatron.core.transformer.moe.token_dispatcher"].MoETokenDispatcher = (
        object
    )
    sys.modules["megatron.core.transformer.transformer_config"].TransformerConfig = (
        object
    )

    handoff = sys.modules["lumen.utils.sonic_moe_handoff"]
    handoff.mapped_handoff_enabled = mock.Mock(return_value=True)
    handoff.mapped_handoff_scope = mock.MagicMock()
    handoff.publish_mapped_handoff = mock.Mock(return_value=True)

    name = "_test_moe_mori_mapped"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "lumen/models/dsv4/megatron/moe_mori.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module, torch, handoff


def _dispatcher(module):
    dispatcher = module.MoEMoriTokenDispatcher.__new__(module.MoEMoriTokenDispatcher)
    dispatcher._recv_topk_weights = "received-scores"
    dispatcher._recv_topk_ids = "received-experts"
    dispatcher.local_expert_indices = list(range(8, 16))
    dispatcher.num_local_experts = 8
    dispatcher._tokens_per_expert = "legacy-counts"
    dispatcher._mapped_combine = None
    return dispatcher


def test_dispatcher_publishes_source_without_materializing_reorder(mori, monkeypatch):
    module, torch, handoff = mori
    dispatcher = _dispatcher(module)
    source, scores, gather = Source(), object(), GatherMap()
    route = mock.Mock(return_value=(scores, (1, 0, 1, 0, 1, 0, 1, 1), gather))
    monkeypatch.setattr(module, "_mapped_local_routing", route)

    result = dispatcher.dispatch_postprocess(source, "legacy-probs")

    assert result[0] is source
    assert result[2] is scores
    handoff.publish_mapped_handoff.assert_called_once_with(
        source, scores, (1, 0, 1, 0, 1, 0, 1, 1), gather
    )
    torch.tensor.assert_called_once_with((1, 0, 1, 0, 1, 0, 1, 1), dtype="long")
    assert dispatcher._mapped_combine.source_rows == 3
    assert dispatcher._mapped_combine.gather_version == 4


def test_disabled_handoff_preserves_legacy_dispatch_result(mori, monkeypatch):
    module, _, handoff = mori
    dispatcher = _dispatcher(module)
    dispatcher.num_local_experts = 1
    handoff.mapped_handoff_enabled.return_value = False
    route = mock.Mock()
    monkeypatch.setattr(module, "_mapped_local_routing", route)
    source = Source()

    assert dispatcher.dispatch_postprocess(source, "legacy-probs") == (
        source,
        "legacy-counts",
        "legacy-probs",
    )
    route.assert_not_called()
    handoff.publish_mapped_handoff.assert_not_called()
