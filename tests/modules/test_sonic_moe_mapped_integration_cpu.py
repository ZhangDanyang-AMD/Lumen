"""Mocked CPU integration for Sonic's one-shot mapped handoff."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


class CpuTensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.device = "cpu"
        self._version = 0

    def reshape(self, *_shape):
        return self

    def float(self):
        return self


class CpuCounts:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


def _module(name):
    module = ModuleType(name)
    module.__path__ = []
    return module


@pytest.fixture
def integration(monkeypatch):
    torch = _module("torch")
    nn = _module("torch.nn")
    nn.Module = object
    torch.nn = nn
    torch.is_grad_enabled = lambda: False
    torch.as_tensor = lambda value, **_kwargs: value
    torch.int32 = "int32"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.nn", nn)

    for name in (
        "megatron",
        "megatron.core",
        "megatron.core.dist_checkpointing",
        "megatron.core.dist_checkpointing.mapping",
        "megatron.core.utils",
        "lumen",
        "lumen.modules",
        "lumen.ops",
        "lumen.ops.moe",
        "lumen.utils",
    ):
        monkeypatch.setitem(sys.modules, name, _module(name))
    mapping = sys.modules["megatron.core.dist_checkpointing.mapping"]
    mapping.ShardedTensor = object
    mapping.ShardedTensorFactory = object
    sys.modules["megatron.core.utils"].get_pg_rank = lambda _group: 0

    handoff_name = "lumen.utils.sonic_moe_handoff"
    handoff_spec = importlib.util.spec_from_file_location(
        handoff_name, ROOT / "lumen/utils/sonic_moe_handoff.py"
    )
    handoff = importlib.util.module_from_spec(handoff_spec)
    monkeypatch.setitem(sys.modules, handoff_name, handoff)
    handoff_spec.loader.exec_module(handoff)

    mapped = mock.Mock(return_value="mapped-output")
    regular = mock.Mock(return_value="regular-output")
    ops = _module("lumen.ops.moe.mxfp4_sonic")
    ops.pre_routed_quantized_mapped = mapped
    ops.pre_routed_quantized = regular
    ops.mapped_pre_routed_available = mock.Mock(return_value=False)
    monkeypatch.setitem(sys.modules, ops.__name__, ops)

    sonic_name = "_test_sonic_moe_mapped"
    sonic_spec = importlib.util.spec_from_file_location(
        sonic_name, ROOT / "lumen/modules/sonic_moe.py"
    )
    sonic = importlib.util.module_from_spec(sonic_spec)
    monkeypatch.setitem(sys.modules, sonic_name, sonic)
    sonic_spec.loader.exec_module(sonic)

    experts = sonic.SonicMoEExperts.__new__(sonic.SonicMoEExperts)
    experts.precision_recipe = "a8w4f"
    experts.w1 = "w1"
    experts.w2 = "w2"
    experts._cached_packed_weights = lambda: "packed"
    return handoff, experts, mapped, regular


def test_sonic_consumes_mapped_handoff_once(integration):
    handoff, experts, mapped, regular = integration
    source = CpuTensor((3, 8), "bfloat16")
    scores = CpuTensor((4,), "float32")
    gather = CpuTensor((4,), "int32")
    counts = CpuCounts((1, 3))
    with handoff.mapped_handoff_scope(True):
        assert handoff.publish_mapped_handoff(source, scores, counts, gather)
        assert experts.forward(source, counts, scores) == ("mapped-output", None)
    regular.assert_not_called()
    mapped.assert_called_once_with(
        "a8w4f",
        source,
        scores,
        (1, 3),
        gather,
        "w1",
        "w2",
        "packed",
        wgrad_mode="bf16",
        main_grads=None,
        wgrad_fallback="error",
        wgrad_backend="triton",
    )


def test_disabled_scope_preserves_regular_path(integration):
    handoff, experts, mapped, regular = integration
    source = CpuTensor((3, 8), "bfloat16")
    scores = CpuTensor((3,), "float32")
    with handoff.mapped_handoff_scope(False):
        assert experts.forward(source, CpuCounts((1, 2)), scores) == (
            "regular-output",
            None,
        )
    mapped.assert_not_called()
    regular.assert_called_once()


def test_missing_aiter_capability_requests_dispatcher_fallback(integration):
    _, experts, _, _ = integration
    enabled, reason = experts.mori_mapped_handoff_capability()
    assert not enabled
    assert "API version 1 is unavailable" in reason
