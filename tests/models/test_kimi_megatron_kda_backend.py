###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""aiter as Megatron-Core ``kda_backend`` (aiter#8755) wired by Lumen.

* registration and ``build_cp_context`` substitution against stub Megatron modules (CPU);
* recurrence parity with Megatron's keywords, FLA ``chunk_kda`` vs aiter ``chunk_kda`` (gfx950);
* the Megatron KDA layer math (``megatron_kda_forward``) with backend fla vs aiter (gfx950);
* CP2-vs-CP1 and TP2-vs-TP1 plus the #6872 loss curve via ``torchrun`` on 2 GPUs.
"""

import os
import subprocess
import sys
import types

import pytest
import torch
import torch.nn.functional as F

from lumen.models.kimi import kda_reference, megatron_kda_backend

aiter_kda = pytest.importorskip("aiter.ops.kda", reason="aiter.ops.kda (ROCm/aiter#8755) missing")


def _gfx950():
    return torch.cuda.is_available() and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


@pytest.fixture
def fake_megatron(monkeypatch):
    """Just enough of megatron.core for patch_megatron_kda_backend and the CP-context substitution."""
    calls = {}

    def select_kda_recurrence(name):
        calls["select"] = name
        return f"orig:{name}"

    class KDALayerConfig:
        def __init__(self, kda_backend):
            self.kda_backend = kda_backend

        def validate_kda(self):
            if self.kda_backend not in ("fla", "cudnn"):
                raise ValueError(self.kda_backend)

    mods = {
        "megatron": types.ModuleType("megatron"),
        "megatron.core": types.ModuleType("megatron.core"),
        "megatron.core.ops": types.ModuleType("megatron.core.ops"),
        "megatron.core.ops.ssm": types.ModuleType("megatron.core.ops.ssm"),
        "megatron.core.ops.ssm.kda": types.ModuleType("megatron.core.ops.ssm.kda"),
        "megatron.core.ops.ssm.kda.backends": types.ModuleType("megatron.core.ops.ssm.kda.backends"),
        "megatron.core.ssm": types.ModuleType("megatron.core.ssm"),
        "megatron.core.ssm.kda_layer_config": types.ModuleType("megatron.core.ssm.kda_layer_config"),
        "megatron.core.ssm.gated_delta_net": types.ModuleType("megatron.core.ssm.gated_delta_net"),
        "megatron.core.ssm.gated_delta_net.kda": types.ModuleType("megatron.core.ssm.gated_delta_net.kda"),
        "megatron.core.ssm.gated_delta_net.common": types.ModuleType("megatron.core.ssm.gated_delta_net.common"),
    }
    mods["megatron.core.ops.ssm.kda.backends"].select_kda_recurrence = select_kda_recurrence
    mods["megatron.core.ssm.kda_layer_config"].KDALayerConfig = KDALayerConfig
    for m in ("megatron.core.ssm.gated_delta_net.kda", "megatron.core.ssm.gated_delta_net.common"):
        mods[m].select_kda_recurrence = select_kda_recurrence
        mods[m].build_cp_context = None
        mods[m].rms_norm_gated = "fla"
    # parent links so `from megatron.core.ssm import kda_layer_config` resolves
    mods["megatron.core.ops.ssm.kda"].backends = mods["megatron.core.ops.ssm.kda.backends"]
    mods["megatron.core.ssm"].kda_layer_config = mods["megatron.core.ssm.kda_layer_config"]
    mods["megatron.core.ssm"].gated_delta_net = mods["megatron.core.ssm.gated_delta_net"]
    mods["megatron.core.ssm.gated_delta_net"].kda = mods["megatron.core.ssm.gated_delta_net.kda"]
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    return mods, KDALayerConfig


def test_install_registers_backend_and_cp_context(fake_megatron):
    mods, KDALayerConfig = fake_megatron
    megatron_kda_backend.install_kda_aiter_backend("aiter", rocm_check=False)
    backends = mods["megatron.core.ops.ssm.kda.backends"]
    kda = mods["megatron.core.ssm.gated_delta_net.kda"]
    assert callable(backends.select_kda_recurrence("aiter"))
    assert kda.select_kda_recurrence is backends.select_kda_recurrence
    assert backends.select_kda_recurrence("fla") == "orig:fla"
    KDALayerConfig("aiter").validate_kda()  # admitted
    with pytest.raises(ValueError):
        KDALayerConfig("bogus").validate_kda()
    assert kda.build_cp_context is megatron_kda_backend.build_cp_context_adapter
    assert mods["megatron.core.ssm.gated_delta_net.common"].build_cp_context is megatron_kda_backend.build_cp_context_adapter
    assert kda.rms_norm_gated == "fla"  # untouched without a failing ROCm check


def test_cp_context_adapter_partition():
    cu = torch.tensor([0, 700, 1600, 2048])
    ctx = aiter_kda.build_kda_cp_context(cu, None, conv1d_kernel_size=4, rank=1, world_size=2)
    assert ctx.cu_seqlens_cpu.tolist() == [0, 576, 1024]  # rank 1 holds tokens 1024..2047
    assert ctx.pre_num_conv_tokens == 1024 - 700
    with pytest.raises(NotImplementedError):
        megatron_kda_backend.build_cp_context_adapter(cu, None, layout="zigzag")


def test_rms_norm_gated_substitute_matches_megatron_formula():
    o = torch.randn(8, 2, 16).bfloat16()
    g = torch.randn_like(o)
    w = torch.ones(16).bfloat16()
    torch.testing.assert_close(
        megatron_kda_backend._rms_norm_gated_megatron(o, g, w, eps=1e-5), kda_reference._rms_norm_gated(o, g, w, 1e-5)
    )


@pytest.mark.skipif(not _gfx950(), reason="gfx950")
def test_fla_on_rocm_check_reports():
    res = megatron_kda_backend.check_fla_rocm()
    print("FLA on ROCm:", res)
    assert set(res) >= {"causal_conv1d", "causal_conv1d_cp", "rms_norm_gated"}


def _megatron_inputs(T=1024, H=8, K=128, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, device="cuda")  # noqa: E731
    return dict(
        q=F.normalize(r(1, T, H, K), dim=-1).bfloat16(), k=F.normalize(r(1, T, H, K), dim=-1).bfloat16(),
        v=r(1, T, H, K).bfloat16(), g=r(1, T, H, K).bfloat16(), beta=torch.sigmoid(r(1, T, H)),
        A_log=torch.log(torch.empty(H, device="cuda").uniform_(1, 16, generator=g)), dt_bias=r(H * K),
    )  # fmt: skip


@pytest.mark.skipif(not _gfx950(), reason="gfx950")
@pytest.mark.skipif(not kda_reference.fla_available(), reason="fla-core")
def test_recurrence_fla_vs_aiter_megatron_kwargs():
    from fla.ops.kda import chunk_kda as fla_chunk_kda

    x = _megatron_inputs()
    cu = torch.tensor([0, 300, 1024], device="cuda", dtype=torch.int32)
    kw = dict(use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, use_qk_l2norm_in_kernel=False, cu_seqlens=cu)
    status = kda_reference.fla_backward_status()
    with torch.no_grad():
        o_fla, _ = fla_chunk_kda(x["q"], x["k"], x["v"], x["g"], x["beta"], A_log=x["A_log"], dt_bias=x["dt_bias"], **kw)
    ref_fn = fla_chunk_kda
    if not status.startswith("ok"):
        # FLA forward still graded below; backward against the fp32 PyTorch recurrence instead
        print(f"FLA chunk_kda backward unavailable ({status}); backward reference = PyTorch recurrence")
        ref_fn = kda_reference._torch_chunk_kda
    outs = []
    for fn in (ref_fn, aiter_kda.chunk_kda):
        t = {n: (v.clone().requires_grad_(True) if n in "qkvg" else v) for n, v in x.items()}
        o, _ = fn(t["q"], t["k"], t["v"], t["g"], t["beta"], A_log=t["A_log"], dt_bias=t["dt_bias"], **kw)
        o.float().pow(2).sum().backward()
        outs.append([o.detach()] + [t[n].grad for n in "qkvg"])
    e = ((outs[1][0].float() - o_fla.float()).norm() / o_fla.float().norm()).item()
    print(f"aiter vs fla forward o: {e:.3e}")
    assert e < 2e-2, "forward o"
    for name, a, f in zip(["o", "dq", "dk", "dv", "dg"], outs[1], outs[0]):
        e = ((a.float() - f.float()).norm() / f.float().norm()).item()
        print(f"aiter vs fla {name}: {e:.3e}")
        assert e < 2e-2, name


@pytest.mark.skipif(not _gfx950(), reason="gfx950")
@pytest.mark.skipif(not kda_reference.fla_available(), reason="fla-core")
def test_megatron_layer_fla_vs_aiter():
    """The Megatron KDA layer math with kda_backend fla vs aiter, same weights, bf16 projections."""
    from aiter.ops.kda_fused_layer import KDALayerWeights

    from lumen.models.kimi import tiny_l13

    cfg = dict(tiny_l13.load_config(None), heads=8)
    ts = tiny_l13.random_tensors(cfg, torch.float32, seed=0)
    w = KDALayerWeights(**{n: t.cuda().to(torch.float32 if n in ("A_log", "dt_bias") else torch.bfloat16) for n, t in ts.items()})
    x = torch.randn(1, 512, cfg["hidden"], device="cuda").bfloat16()
    ys = [kda_reference.megatron_kda_forward(x, w, 8, backend=b, recompute=False) for b in ("fla", "aiter")]
    e = ((ys[1].float() - ys[0].float()).norm() / ys[0].float().norm()).item()
    print(f"Megatron layer aiter vs fla: {e:.3e}")
    assert e < 2e-2


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() >= 2 and _gfx950()), reason="2 gfx950 GPUs"
)
def test_cp2_tp2_via_torchrun():
    script = os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "dist_kimi_kda_cp_tp.py")
    r = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", script],
        capture_output=True, text=True, timeout=1800,
    )  # fmt: skip
    print(r.stdout[-4000:], r.stderr[-4000:])
    assert r.returncode == 0


def _recurrence_inputs(T=96, H=2, K=32, device="cpu", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, device=device)  # noqa: E731
    return dict(
        q=F.normalize(r(1, T, H, K), dim=-1).bfloat16(), k=F.normalize(r(1, T, H, K), dim=-1).bfloat16(),
        v=r(1, T, H, K).bfloat16(), g=r(1, T, H, K).bfloat16(), beta=torch.sigmoid(r(1, T, H)),
        A_log=torch.log(torch.empty(H, device=device).uniform_(1, 16, generator=g)), dt_bias=r(H * K),
    )  # fmt: skip


def test_bf16_mma_recurrence_vs_fp32():
    """The FLA-precision reference recurrence rounds its MMA operands: close to fp32, not equal;
    fp32 inputs take the fp32 path exactly."""
    x = _recurrence_inputs()
    kw = dict(safe_gate=True, lower_bound=-5.0)
    o16, _ = kda_reference._torch_chunk_kda(x["q"], x["k"], x["v"], x["g"], x["beta"], x["A_log"], x["dt_bias"], **kw)
    o32, _ = kda_reference._torch_chunk_kda(x["q"], x["k"], x["v"], x["g"], x["beta"], x["A_log"], x["dt_bias"],
                                            mma_bf16=False, **kw)  # fmt: skip
    e = ((o16.float() - o32.float()).norm() / o32.float().norm()).item()
    print(f"bf16-operand vs fp32 recurrence: {e:.3e}")
    assert 0 < e < 2e-2
    f32 = {n: (t.float() if t.is_floating_point() else t) for n, t in x.items()}
    a, _ = kda_reference._torch_chunk_kda(**f32, **kw)
    b, _ = kda_reference._torch_chunk_kda(**f32, mma_bf16=False, **kw)
    assert torch.equal(a, b)


@pytest.mark.skipif(not _gfx950(), reason="FLA forward needs the GPU")
@pytest.mark.skipif(not kda_reference.fla_available(), reason="fla-core")
def test_bf16_mma_recurrence_matches_fla_forward():
    """FLA's chunk_kda forward (which compiles on ROCm) vs the independent FLA-precision reference."""
    from fla.ops.kda import chunk_kda as fla_chunk_kda

    x = _recurrence_inputs(T=1024, H=8, K=128, device="cuda")
    cu = torch.tensor([0, 300, 1024], device="cuda", dtype=torch.int32)
    kw = dict(safe_gate=True, lower_bound=-5.0)
    with torch.no_grad():
        o_fla, _ = fla_chunk_kda(x["q"], x["k"], x["v"], x["g"], x["beta"], A_log=x["A_log"], dt_bias=x["dt_bias"],
                                 use_gate_in_kernel=True, use_qk_l2norm_in_kernel=False, cu_seqlens=cu, **kw)  # fmt: skip
        out = {}
        for name, mma in (("bf16-operand", True), ("fp32", False)):
            o, _ = kda_reference._torch_chunk_kda(x["q"], x["k"], x["v"], x["g"], x["beta"], x["A_log"],
                                                  x["dt_bias"], cu_seqlens=cu, mma_bf16=mma, **kw)  # fmt: skip
            out[name] = ((o.float() - o_fla.float()).norm() / o_fla.float().norm()).item()
    print("reference recurrence vs FLA forward:", out)
    assert out["bf16-operand"] < 2e-3, out  # ~1e-3: bf16 output rounding and summation order only
