"""CPU tests for the ``LUMEN_ROPE_IMPL`` switch of the Megatron fused-RoPE hook."""

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

if "lumen" not in sys.modules:
    _lumen = types.ModuleType("lumen")
    _lumen.__path__ = [str(Path(__file__).resolve().parents[2] / "lumen")]
    sys.modules["lumen"] = _lumen

import lumen.patches.runtime.megatron_rope as megatron_rope  # noqa: E402


@pytest.fixture
def rope(monkeypatch):
    for name in ("LUMEN_ROPE_IMPL", "LUMEN_FUSED_ROPE", "ROPE_FUSION_DEBUG"):
        monkeypatch.delenv(name, raising=False)
    module = importlib.reload(megatron_rope)
    monkeypatch.setattr(module.atexit, "register", lambda fn: fn)
    return module


def _fake_rope_utils():
    calls = []

    def unfused(t, freqs, rotary_interleaved=False, mscale=1.0):
        calls.append((rotary_interleaved, mscale))
        return t + 1

    return types.SimpleNamespace(_apply_rotary_pos_emb_bshd=unfused, fused_apply_rotary_pos_emb=None), calls


def _fake_apex(monkeypatch, aiter_backend):
    calls = []

    def fused(t, freqs, transpose_output_memory=False):
        calls.append(tuple(t.shape))
        return t * 2

    mod = types.ModuleType("apex.transformer.functional.fused_rope")
    mod.AITER_ROPE_BACKEND = aiter_backend
    mod.fused_apply_rotary_pos_emb = fused
    for name in ("apex", "apex.transformer", "apex.transformer.functional"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "apex.transformer.functional.fused_rope", mod)
    sys.modules["apex.transformer.functional"].fused_rope = mod
    sys.modules["apex.transformer"].functional = sys.modules["apex.transformer.functional"]
    sys.modules["apex"].transformer = sys.modules["apex.transformer"]
    return calls


@pytest.mark.parametrize(
    "value, expected",
    [(None, "aiter"), ("", "aiter"), ("aiter", "aiter"), ("apex", "apex"), (" APEX ", "apex")],
)
def test_rope_impl_parses_env(rope, monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv("LUMEN_ROPE_IMPL", value)
    assert rope.rope_impl() == expected


def test_rope_impl_rejects_unknown_value(rope, monkeypatch):
    monkeypatch.setenv("LUMEN_ROPE_IMPL", "te")
    with pytest.raises(ValueError, match="LUMEN_ROPE_IMPL"):
        rope.rope_impl()
    with pytest.raises(ValueError):
        rope.install(_fake_rope_utils()[0])
    assert not rope.report()["installed"]


def test_default_install_keeps_aiter_and_does_not_import_apex(rope, monkeypatch):
    monkeypatch.setitem(sys.modules, "apex", None)
    utils, _ = _fake_rope_utils()
    assert rope.install(utils)
    assert utils.fused_apply_rotary_pos_emb is rope.fused_apply_rotary_pos_emb
    assert rope.report()["impl"] == "aiter"
    assert rope._APEX["fn"] is None


def test_apex_install_raises_when_apex_picked_aiter_backend(rope, monkeypatch):
    monkeypatch.setenv("LUMEN_ROPE_IMPL", "apex")
    _fake_apex(monkeypatch, aiter_backend=True)
    utils, _ = _fake_rope_utils()
    with pytest.raises(RuntimeError, match="USE_ROCM_AITER_ROPE_BACKEND=0"):
        rope.install(utils)
    assert utils.fused_apply_rotary_pos_emb is None
    assert not rope.report()["installed"]


def test_apex_path_dispatches_sbhd_and_falls_back(rope, monkeypatch):
    monkeypatch.setenv("LUMEN_ROPE_IMPL", "apex")
    apex_calls = _fake_apex(monkeypatch, aiter_backend=False)
    utils, unfused_calls = _fake_rope_utils()
    assert rope.install(utils)
    assert rope.report()["impl"] == "apex"

    t = torch.ones(8, 1, 4, 16)
    freqs = torch.zeros(8, 1, 1, 16)
    assert torch.equal(utils.fused_apply_rotary_pos_emb(t, freqs), t * 2)
    assert apex_calls == [(8, 1, 4, 16)]

    # Interleaved, scaled, and mismatched-sequence calls take Megatron's unfused path.
    utils.fused_apply_rotary_pos_emb(t, freqs, interleaved=True)
    utils.fused_apply_rotary_pos_emb(t, freqs, mscale=0.5)
    utils.fused_apply_rotary_pos_emb(t, torch.zeros(4, 1, 1, 16))
    assert unfused_calls == [(True, 1.0), (False, 0.5), (False, 1.0)]
    state = rope.report()
    assert (state["calls"], state["sbhd"], state["fallbacks"]) == (1, 1, 3)


def test_aiter_path_refuses_interleaved_and_unknown_layout(rope):
    utils, unfused_calls = _fake_rope_utils()
    assert rope.install(utils)
    t = torch.ones(8, 1, 4, 16)
    utils.fused_apply_rotary_pos_emb(t, torch.zeros(8, 1, 1, 16), interleaved=True)
    utils.fused_apply_rotary_pos_emb(t, torch.zeros(5, 1, 1, 16))
    utils.fused_apply_rotary_pos_emb(t, torch.zeros(8, 16))
    assert len(unfused_calls) == 3
    state = rope.report()
    assert (state["refused"], state["fallbacks"], state["calls"]) == (1, 2, 0)


def test_aiter_path_picks_layout_from_sequence_axis(rope, monkeypatch):
    seen = []

    class _Recorder:
        @staticmethod
        def apply(core, cos, sin, layout):
            seen.append((layout, tuple(cos.shape)))
            return core

    monkeypatch.setattr(rope, "_FusedRoPE", _Recorder)
    utils, _ = _fake_rope_utils()
    assert rope.install(utils)
    utils.fused_apply_rotary_pos_emb(torch.ones(8, 1, 4, 16), torch.zeros(8, 1, 1, 16))
    utils.fused_apply_rotary_pos_emb(torch.ones(1, 4, 8, 16), torch.zeros(1, 1, 8, 16))
    out = utils.fused_apply_rotary_pos_emb(torch.ones(8, 1, 4, 24), torch.zeros(8, 1, 1, 16))
    assert out.shape == (8, 1, 4, 24)
    assert seen == [("sbhd", (8, 1, 1, 16)), ("bhsd", (8, 1, 1, 16)), ("sbhd", (8, 1, 1, 16))]
    assert rope.report()["cos_sin_computed"] == 3


def test_cos_sin_helper_is_wrappable(rope, monkeypatch):
    wrapped = []
    original = rope._cos_sin_4d

    def cached(freqs, rot_dim):
        wrapped.append(rot_dim)
        return original(freqs, rot_dim)

    monkeypatch.setattr(rope, "_FusedRoPE", types.SimpleNamespace(apply=lambda c, *a: c))
    monkeypatch.setattr(rope, "_cos_sin_4d", cached)
    utils, _ = _fake_rope_utils()
    rope.install(utils)
    utils.fused_apply_rotary_pos_emb(torch.ones(8, 1, 4, 16), torch.zeros(8, 1, 1, 16))
    assert wrapped == [16]


def test_disable_switch_and_enable_on_config(rope, monkeypatch):
    model = types.SimpleNamespace(config=types.SimpleNamespace(apply_rope_fusion=False))
    assert not rope.enable_on_config(model)

    monkeypatch.setenv("LUMEN_FUSED_ROPE", "0")
    utils, _ = _fake_rope_utils()
    assert not rope.install(utils)
    assert utils.fused_apply_rotary_pos_emb is None

    monkeypatch.delenv("LUMEN_FUSED_ROPE")
    assert rope.install(utils)
    assert rope.enable_on_config(model)
    assert model.config.apply_rope_fusion is True
    assert not rope.enable_on_config(types.SimpleNamespace(config=None))
