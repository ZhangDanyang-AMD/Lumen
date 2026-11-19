"""Mocked CPU integration for Megatron's real ``_MoriManager``."""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _module(name):
    module = ModuleType(name)
    module.__path__ = []
    return module


class FakeTensor:
    def __init__(self, shape, dtype="bf16"):
        self.shape = shape
        self.dtype = dtype
        self.device = "cuda"
        self._version = 0

    def unsqueeze(self, _dim):
        return self

    def reshape(self, *_shape):
        return self

    def to(self, dtype):
        self.dtype = dtype
        return self

    def contiguous(self):
        return self

    def numel(self):
        return self.shape[0]


class Counts:
    def sum(self):
        return 5

    def tolist(self):
        return [1, 0, 1, 0, 1, 0, 1, 1]


@pytest.fixture
def mori(monkeypatch):
    torch = _module("torch")
    torch.float32 = "float32"
    torch.int32 = "int32"
    row_ids = FakeTensor((12, 1), "float32")
    torch.arange = mock.Mock(return_value=row_ids)
    monkeypatch.setitem(sys.modules, "torch", torch)

    for name in (
        "megatron",
        "megatron.core",
        "megatron.core.transformer",
        "megatron.core.transformer.moe",
        "megatron.core.transformer.moe.moe_layer",
        "megatron.core.transformer.moe.moe_utils",
        "megatron.core.transformer.moe.token_dispatcher",
        "lumen",
        "lumen.ops",
        "lumen.ops.moe",
        "lumen.ops.moe.mxfp4_sonic",
        "lumen.utils",
        "lumen.utils.sonic_moe_handoff",
    ):
        monkeypatch.setitem(sys.modules, name, _module(name))

    token_dispatcher = sys.modules["megatron.core.transformer.moe.token_dispatcher"]

    class MoriManager:
        def dispatch(
            self,
            hidden_states,
            async_finish=True,
            allocate_on_comm_stream=True,
        ):
            return hidden_states

        def _permute_by_experts(self, hidden_states):
            del hidden_states
            return "legacy-permuted"

        def combine(
            self,
            hidden_states,
            async_finish=True,
            allocate_on_comm_stream=True,
        ):
            return hidden_states

    MoriManager.__module__ = token_dispatcher.__name__
    token_dispatcher._MoriManager = MoriManager
    fused_indices_to_multihot = mock.Mock(
        return_value=(object(), FakeTensor((12, 8), "float32"))
    )
    token_dispatcher.fused_indices_to_multihot = fused_indices_to_multihot

    moe_layer = sys.modules["megatron.core.transformer.moe.moe_layer"]

    class MoELayer:
        def routed_experts_compute(self, hidden_states, probs, residual):
            return hidden_states, probs, residual

    MoELayer.__module__ = moe_layer.__name__
    moe_layer.MoELayer = MoELayer
    sys.modules["megatron.core.transformer.moe"].moe_layer = moe_layer
    sys.modules["megatron.core.transformer.moe"].token_dispatcher = token_dispatcher

    permuted_rows = FakeTensor((5, 1), "float32")
    permuted_probs = FakeTensor((5,), "float32")
    reversed_mapping = object()
    permute = mock.Mock(
        return_value=(
            permuted_rows,
            permuted_probs,
            reversed_mapping,
            None,
            None,
        )
    )
    sys.modules["megatron.core.transformer.moe.moe_utils"].permute = permute
    sys.modules["lumen.ops.moe.mxfp4_sonic"].mapped_pre_routed_available = lambda: True

    handoff = sys.modules["lumen.utils.sonic_moe_handoff"]
    handoff.mapped_handoff_enabled = mock.Mock(return_value=True)
    handoff.publish_mapped_handoff = mock.Mock(return_value=True)
    scope_entries = []

    @contextmanager
    def scope(_enabled):
        scope_entries.append("enter")
        try:
            yield
        finally:
            scope_entries.append("exit")

    handoff.mapped_handoff_scope = scope

    name = "_test_real_mori_handoff"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "lumen/modules/mori_sonic_handoff.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setenv("LUMEN_MORI_SONIC_MXFP4_HANDOFF", "1")
    assert module.install_mori_sonic_handoff()
    return SimpleNamespace(
        module=module,
        manager_type=MoriManager,
        layer_type=MoELayer,
        torch=torch,
        row_ids=row_ids,
        permute=permute,
        permuted_probs=permuted_probs,
        reversed_mapping=reversed_mapping,
        handoff=handoff,
        scope_entries=scope_entries,
        fused_indices_to_multihot=fused_indices_to_multihot,
    )


def _manager(fixture):
    manager = fixture.manager_type()
    manager.config = SimpleNamespace(moe_router_padding_for_quantization=False)
    manager.router_dtype = "fp32"
    manager.permute_fusion = False
    manager.num_local_experts = 8
    manager.dispatched_indices = object()
    manager.dispatched_probs = object()
    manager._routing_handle = object()
    manager._raw_dispatched_probs = object()
    manager._indices_to_multihot = mock.Mock(
        return_value=(object(), FakeTensor((12, 8), "float32"))
    )
    manager._synced_tokens_per_expert = mock.Mock(return_value=Counts())
    return manager


def test_real_manager_publishes_small_permutation_without_copying_hidden(mori):
    manager = _manager(mori)
    source = FakeTensor((12, 4096))
    routing_handle = manager._routing_handle
    raw_probs = manager._raw_dispatched_probs

    assert manager._permute_by_experts(source) is source

    call = mori.permute.call_args
    assert call.args[0] is mori.row_ids
    assert call.args[0] is not source
    assert call.kwargs["num_out_tokens"] == 5
    assert manager.reversed_mapping_for_combine is mori.reversed_mapping
    assert manager.dispatched_probs is mori.permuted_probs
    assert manager._routing_handle is routing_handle
    assert manager._raw_dispatched_probs is raw_probs
    mori.handoff.publish_mapped_handoff.assert_called_once()
    published = mori.handoff.publish_mapped_handoff.call_args.args
    assert published[0] is source
    assert published[1] is mori.permuted_probs
    assert published[3].dtype == "int32"


def test_disabled_scope_preserves_original_manager_path(mori):
    manager = _manager(mori)
    mori.handoff.mapped_handoff_enabled.return_value = False
    assert manager._permute_by_experts(FakeTensor((12, 4096))) == "legacy-permuted"
    mori.permute.assert_not_called()
    mori.handoff.publish_mapped_handoff.assert_not_called()


def test_fused_manager_uses_megatron_fused_index_conversion(mori):
    manager = _manager(mori)
    manager.permute_fusion = True
    dispatched_indices = manager.dispatched_indices
    dispatched_probs = manager.dispatched_probs
    source = FakeTensor((12, 4096))
    assert manager._permute_by_experts(source) is source
    mori.fused_indices_to_multihot.assert_called_once_with(
        dispatched_indices,
        dispatched_probs,
        8,
    )
    manager._indices_to_multihot.assert_not_called()
    assert mori.permute.call_args.kwargs["fused"] is True


def test_unsupported_padding_falls_back_before_publication(mori):
    manager = _manager(mori)
    manager.config.moe_router_padding_for_quantization = True
    assert manager._permute_by_experts(FakeTensor((12, 4096))) == "legacy-permuted"
    mori.handoff.publish_mapped_handoff.assert_not_called()
    assert any("router padding" in reason for reason in mori.module._FALLBACK_NOTICES)


def test_exact_signature_gate_rejects_method_drift(mori):
    def drifted(self, hidden_states, extra=None):
        del self, hidden_states, extra

    assert not mori.module._has_signature(
        drifted,
        mori.module._EXPECTED_SIGNATURES["_MoriManager._permute_by_experts"],
        mori.module._EXPECTED_DEFAULTS["_MoriManager._permute_by_experts"],
    )


def test_moe_layer_scope_is_limited_to_exact_mori_sonic_pair(mori):
    layer = mori.layer_type()
    layer.token_dispatcher = SimpleNamespace(_comm_manager=_manager(mori))
    layer.experts = SimpleNamespace(mori_mapped_handoff_capability=lambda: (True, ""))
    result = layer.routed_experts_compute("hidden", "probs", "residual")
    assert result == ("hidden", "probs", "residual")
    assert mori.scope_entries == ["enter", "exit"]
