"""Install guard of the fused MORI receive-routing patch, with fake Megatron/AITER modules."""

import importlib
import sys
from types import ModuleType

import pytest


def _package(name):
    module = ModuleType(name)
    module.__path__ = []
    return module


class _FakeDispatch:
    @staticmethod
    def forward(ctx, x):
        return x


@pytest.fixture
def patch_module(monkeypatch):
    module = importlib.reload(importlib.import_module("lumen.modules.mori_fused_recv"))
    fused_a2a = _package("megatron.core.transformer.moe.fused_a2a")
    fused_a2a.MoriDispatch = _FakeDispatch
    fused_a2a.get_mori_op = lambda **_: None
    fused_a2a._run_mori_op_on_stream = lambda fn, *_: fn()
    original = _FakeDispatch.__dict__["forward"]
    names = ("megatron", "megatron.core", "megatron.core.transformer", "megatron.core.transformer.moe")
    for name in names:
        monkeypatch.setitem(sys.modules, name, _package(name))
    sys.modules["megatron.core.transformer.moe"].fused_a2a = fused_a2a
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.moe.fused_a2a", fused_a2a)
    recv = ModuleType("aiter.ops.triton.moe.mori_recv_routing")
    recv.mori_recv_local_routing = lambda *args: args
    for name in ("aiter", "aiter.ops", "aiter.ops.triton", "aiter.ops.triton.moe"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, _package(name))
    monkeypatch.setitem(sys.modules, "aiter.ops.triton.moe.mori_recv_routing", recv)
    yield module
    _FakeDispatch.forward = original


def test_disabled_by_env(patch_module, monkeypatch):
    monkeypatch.setenv("LUMEN_MORI_FUSED_RECV", "0")
    assert not patch_module.install_mori_fused_recv()
    assert _FakeDispatch.forward(None, 3) == 3


def test_unknown_forward_keeps_original(patch_module, monkeypatch):
    monkeypatch.delenv("LUMEN_MORI_FUSED_RECV", raising=False)
    assert not patch_module.install_mori_fused_recv()
    assert "fingerprint" in patch_module._STATE["reason"]
    assert _FakeDispatch.forward(None, 3) == 3


def test_matching_forward_installs_once(patch_module, monkeypatch):
    monkeypatch.delenv("LUMEN_MORI_FUSED_RECV", raising=False)
    fingerprint = patch_module.forward_fingerprint(_FakeDispatch.forward)
    monkeypatch.setattr(patch_module, "_EXPECTED_FORWARD_SHA256", frozenset({fingerprint}))
    assert patch_module.install_mori_fused_recv()
    assert _FakeDispatch.forward.__name__ == "forward"
    assert _FakeDispatch.forward.__code__.co_argcount == 13
    assert patch_module.install_mori_fused_recv()
