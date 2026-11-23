###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""LUMEN_ATTN_AITER_TRITON_MHA / LUMEN_ATTN_AITER_FUSED_BWD: switch handling (CPU) and
the resulting aiter_csrc attention gradients against a PyTorch reference (GPU)."""

import importlib
import math
import sys
import types
from pathlib import Path

import pytest
import torch

if "lumen" not in sys.modules:
    _lumen = types.ModuleType("lumen")
    _lumen.__path__ = [str(Path(__file__).resolve().parents[2] / "lumen")]
    sys.modules["lumen"] = _lumen

import lumen.ops.attention.aiter_mha_config as aiter_mha_config  # noqa: E402

_ENV = ("LUMEN_ATTN_AITER_TRITON_MHA", "LUMEN_ATTN_AITER_FUSED_BWD")


@pytest.fixture
def config(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    return importlib.reload(aiter_mha_config)


def _fake_aiter(monkeypatch):
    calls = []
    mha = types.ModuleType("aiter.ops.mha")
    mha.ENABLE_CK = True
    triton_mha = types.ModuleType("aiter.ops.triton.attention.mha")
    triton_mha.mha_set_use_fused_bwd_kernel = calls.append
    names = ("aiter", "aiter.ops", "aiter.ops.triton", "aiter.ops.triton.attention")
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "aiter.ops.mha", mha)
    monkeypatch.setitem(sys.modules, "aiter.ops.triton.attention.mha", triton_mha)
    sys.modules["aiter"].ops = sys.modules["aiter.ops"]
    sys.modules["aiter.ops"].mha = mha
    sys.modules["aiter.ops"].triton = sys.modules["aiter.ops.triton"]
    sys.modules["aiter.ops.triton"].attention = sys.modules["aiter.ops.triton.attention"]
    sys.modules["aiter.ops.triton.attention"].mha = triton_mha
    return mha, calls


def test_defaults_leave_aiter_untouched(config, monkeypatch):
    monkeypatch.setitem(sys.modules, "aiter", None)
    assert config.configure_aiter_mha() == {"triton_mha": False, "fused_bwd": False}


def test_triton_mha_switch_is_module_scoped(config, monkeypatch):
    mha, calls = _fake_aiter(monkeypatch)
    monkeypatch.setenv("LUMEN_ATTN_AITER_TRITON_MHA", "1")
    assert config.configure_aiter_mha() == {"triton_mha": True, "fused_bwd": False}
    assert mha.ENABLE_CK is False
    assert calls == []


def test_fused_bwd_switch(config, monkeypatch):
    mha, calls = _fake_aiter(monkeypatch)
    monkeypatch.setenv("LUMEN_ATTN_AITER_FUSED_BWD", "1")
    assert config.configure_aiter_mha() == {"triton_mha": False, "fused_bwd": True}
    assert calls == [True]
    assert mha.ENABLE_CK is True


@pytest.mark.parametrize("value", ["0", "", "true"])
def test_only_one_enables(config, monkeypatch, value):
    mha, calls = _fake_aiter(monkeypatch)
    for name in _ENV:
        monkeypatch.setenv(name, value)
    assert config.configure_aiter_mha() == {"triton_mha": False, "fused_bwd": False}
    assert mha.ENABLE_CK is True and calls == []


def _attention_ref(q, k, v, do, scale):
    qf, kf, vf = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    rep = q.shape[2] // k.shape[2]
    s = q.shape[1]
    att = torch.einsum("bshd,bthd->bhst", qf, kf.repeat_interleave(rep, 2)) * scale
    mask = torch.ones(s, s, device=q.device, dtype=torch.bool).tril()
    att = att.masked_fill(~mask, float("-inf")).softmax(-1)
    out = torch.einsum("bhst,bthd->bshd", att, vf.repeat_interleave(rep, 2))
    out.backward(do.float())
    return out, qf.grad, kf.grad, vf.grad


def _snr_db(ref, x):
    err = (x.float() - ref.float()).pow(2).mean().item()
    return 10 * math.log10(ref.float().pow(2).mean().item() / max(err, 1e-30))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("fused", [False, True])
def test_aiter_csrc_triton_mha_gradients(monkeypatch, fused):
    pytest.importorskip("aiter")
    from aiter.ops.triton.attention.mha import mha_set_use_fused_bwd_kernel

    import aiter.ops.mha as aiter_mha
    from lumen.ops.attention.attention import attention

    monkeypatch.setenv("LUMEN_ATTN_AITER_TRITON_MHA", "1")
    monkeypatch.setenv("LUMEN_ATTN_AITER_FUSED_BWD", "1" if fused else "0")
    enable_ck = aiter_mha.ENABLE_CK
    importlib.reload(aiter_mha_config).configure_aiter_mha()
    try:
        torch.manual_seed(0)
        b, s, hq, hk, d = 1, 2048, 32, 4, 128
        q = torch.randn(b, s, hq, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(b, s, hk, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(b, s, hk, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        do = torch.randn(b, s, hq, d, device="cuda", dtype=torch.bfloat16)
        scale = d**-0.5
        out = attention(q, k, v, softmax_scale=scale, causal=True, backend_type="aiter_csrc")
        # Stale allocator blocks must not leak into the atomically accumulated dK/dV.
        stale = [torch.full_like(t, float("nan")) for t in (k, v, k, v)]
        del stale
        out.backward(do)
        ref = _attention_ref(q, k, v, do, scale)
        for name, ref_t, got in zip(("out", "dq", "dk", "dv"), ref, (out, q.grad, k.grad, v.grad)):
            assert _snr_db(ref_t, got) > 40, name
    finally:
        aiter_mha.ENABLE_CK = enable_ck
        mha_set_use_fused_bwd_kernel(False)
