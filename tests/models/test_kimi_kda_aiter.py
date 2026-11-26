###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kimi-K3 Tiny-L13 KDA layers through aiter ``kda_layer_fwd``.

Requires aiter with ROCm/aiter#8739 (branch ``cursor/kda-fused-layer-bwd-8350``,
``aiter/ops/kda_fused_layer.py``). The parity test needs one gfx950 GPU; set
``LUMEN_KIMI_TINY_L13_PATH`` to a ``Qinghao/Kimi-K3-Tiny-L13`` snapshot to use
real weights (including the padded [128] ``A_log``), otherwise random weights at
the real shapes are used. Both sides use the safe gate (``gate_lower_bound=-5``).
"""

import os

import pytest
import torch

from lumen.models.kimi import kda_reference, tiny_l13
from lumen.models.kimi.kda_aiter import kda_layer_weights, patch_kda_layers, unpatch_kda_layers
from lumen.models.kimi.kda_stack import (
    FP32_KDA_PARAMS,
    MEGATRON_FP32_KDA_PARAMS,
    KimiDeltaAttention,
    grad_parity_vs_fp32,
)
from lumen.ops.dispatch import _probe_aiter_kda_fused_layer

pytestmark = pytest.mark.skipif(not _probe_aiter_kda_fused_layer(), reason="aiter kda_fused_layer (aiter#8739) missing")


def _to_train_dtype(m, device, dtype, fp32_params=FP32_KDA_PARAMS):
    m.to(device, dtype)
    for name, p in m.named_parameters():
        if name.endswith(tuple(fp32_params)):
            p.data = p.data.float()
    return m


def _small_cfg():
    return dict(tiny_l13.load_config(None), hidden=64, heads=2, head_dim=16, gate_rank=16)


def test_patch_follows_config_kda_layers():
    class _Cfg:
        linear_attn_config = {"kda_layers": [1]}  # 1-indexed

    root = torch.nn.Module()
    root.config = _Cfg()
    root.a = KimiDeltaAttention(_small_cfg(), layer_idx=0)
    root.b = KimiDeltaAttention(_small_cfg(), layer_idx=3)
    assert patch_kda_layers(root, backend="torch") == 1
    assert unpatch_kda_layers(root) == 1


def test_layer_split_and_checkpoint_findings():
    cfg = tiny_l13.load_config(None)
    assert cfg["kda_layers"] == (0, 1, 2, 4, 5, 6, 8, 9, 10)
    assert cfg["mla_layers"] == (3, 7, 11, 12)
    assert cfg["gate_lower_bound"] == -5.0
    with pytest.raises(ValueError, match="not a KDA layer"):
        tiny_l13.load_kda_tensors(3)


def test_weights_are_views_of_module_params():
    m = KimiDeltaAttention(_small_cfg())
    w = kda_layer_weights(m)
    assert w.conv_q.data_ptr() == m.q_conv1d.weight.data_ptr()
    assert w.conv_q.shape == (m.q_conv1d.weight.shape[0], m.q_conv1d.weight.shape[-1])
    assert w.w_o.data_ptr() == m.o_proj.weight.data_ptr()
    assert w.A_log.data_ptr() == m.A_log.data_ptr()


def test_padded_a_log_accepted():
    m = KimiDeltaAttention(_small_cfg())
    assert m.A_log.numel() == tiny_l13.A_LOG_LEN > m.num_heads  # checkpoint layout: per head, padded
    assert kda_layer_weights(m).A_log.numel() == tiny_l13.A_LOG_LEN


def test_patched_matches_eager_with_safe_gate():
    cfg = _small_cfg()
    ts = tiny_l13.random_tensors(cfg, torch.float32)
    ref, m = (KimiDeltaAttention(cfg).load_fields(ts) for _ in range(2))
    assert patch_kda_layers(m, backend="fused") == 1
    x = torch.randn(1, 70, cfg["hidden"])
    outs = []
    for mod in (ref, m):
        xs = x.clone().requires_grad_(True)
        y = mod(xs, attention_mask=None)
        y.sum().backward()
        outs.append([y.detach(), xs.grad] + [p.grad for p in mod.parameters()])
    for r, o in zip(*outs):
        assert r is not None and o is not None
        torch.testing.assert_close(o, r, rtol=1e-4, atol=1e-4)
    assert unpatch_kda_layers(m) == 1


def test_cache_keeps_module_forward():
    m = KimiDeltaAttention(_small_cfg())
    calls = []
    m.forward = lambda h, attention_mask=None, cache_params=None, **kw: calls.append(cache_params) or h
    patch_kda_layers(m, backend="fused")
    m(torch.randn(1, 8, 64), cache_params="cache")
    assert calls == ["cache"]


def _gfx950():
    return torch.cuda.is_available() and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


@pytest.mark.skipif(not _gfx950(), reason="Tiny-L13 KDA parity targets one gfx950 GPU")
@pytest.mark.parametrize("seqlen", [100, 256])
@pytest.mark.parametrize("layer", tiny_l13.KDA_LAYERS)
def test_tiny_l13_kda_parity_bf16(layer, seqlen):
    """BF16 fwd+bwd of the patched module (aiter fused) and the eager module (torch), both graded
    against an fp32 eager run of the same bf16-rounded weights and inputs."""
    cfg = tiny_l13.load_config()
    ts = tiny_l13.load_kda_tensors(layer, cfg=cfg)
    lb = cfg["gate_lower_bound"]
    assert lb == tiny_l13.GATE_LOWER_BOUND
    ref = _to_train_dtype(KimiDeltaAttention(cfg, 0, lb).load_fields(ts), "cuda", torch.bfloat16)
    test = _to_train_dtype(KimiDeltaAttention(cfg, 0, lb).load_fields(ts), "cuda", torch.bfloat16)
    ref32 = KimiDeltaAttention(cfg, 0, lb).load_fields({n: t.to(torch.bfloat16).float() for n, t in ts.items()}).cuda()
    assert patch_kda_layers(test, backend="fused") == 1

    g = torch.Generator(device="cuda").manual_seed(layer)
    x = torch.randn(1, seqlen, cfg["hidden"], generator=g, device="cuda").to(torch.bfloat16)
    dy = torch.randn(x.shape, generator=g, device="cuda").to(torch.bfloat16)
    outs = []
    for m in (ref32, ref, test):
        xs = (x.float() if m is ref32 else x).clone().requires_grad_(True)
        y = m(xs, attention_mask=None)
        y.backward(dy.to(y.dtype))
        torch.cuda.synchronize()
        outs.append([y.detach(), xs.grad] + [p.grad for p in m.parameters()])

    names = ["y", "dx"] + [n for n, _ in ref.named_parameters()]
    for name, *vals in zip(names, *outs):  # fp32 reference, eager bf16, aiter bf16
        assert all(v is not None for v in vals), f"layer {layer}: {name} has no gradient"
    from aiter.ops import kda_fused_layer as kfl

    r32, e16, a16 = ({n: t.detach().float().cpu() for n, t in zip(names, ts)} for ts in outs)
    rows, problems = grad_parity_vs_fp32(r32, e16, a16, cfg["heads"])
    print(f"layer {layer} T={seqlen} gemm={kfl.LAST_PIECES.get('gemm')} worst vs fp32 (eager_bf16, aiter_bf16, limit):",
          list(rows.items())[:5])  # fmt: skip
    assert not problems, f"layer {layer}: {problems}"


@pytest.mark.skipif(not _gfx950(), reason="Tiny-L13 KDA parity targets one gfx950 GPU")
@pytest.mark.skipif(not kda_reference.fla_available(), reason="Megatron-LM#6872 reference needs fla-core (fla.ops.kda)")
@pytest.mark.parametrize("seqlen", [100, 256])
@pytest.mark.parametrize("layer", tiny_l13.KDA_LAYERS)
def test_tiny_l13_kda_vs_megatron_fla(layer, seqlen):
    """aiter fused and Megatron-FLA, both bf16 activations with projections in ``LUMEN_KIMI_KDA_PRECISION``
    (default mxfp8, as in the Megatron-LM#6872 recipes), graded against Megatron-FLA in fp32 with
    unquantized projections (same rounded weights): aiter's error <= max(4 x Megatron's, 3e-2).

    aiter torch bf16 is reported against the same reference as a secondary diagnostic.
    """
    from lumen.models.kimi.kda_aiter import aiter_supports_gemm_precision

    prec = os.environ.get("LUMEN_KIMI_KDA_PRECISION", "mxfp8")
    if prec != "bf16" and not aiter_supports_gemm_precision():
        pytest.skip(f"aiter kda_layer_fwd has no gemm_precision={prec} (MXFP8 PR stacked on #8739)")
    cfg = tiny_l13.load_config()
    ts = tiny_l13.load_kda_tensors(layer, cfg=cfg)
    lb = cfg["gate_lower_bound"]

    def module(dtype, reference, precision="bf16"):
        m = KimiDeltaAttention(cfg, 0, lb)
        m.gemm_precision = precision
        if dtype == torch.float32:
            m.load_fields({n: t.to(torch.bfloat16).float() for n, t in ts.items()}).cuda()
        else:
            # Megatron runtime dtypes on every bf16 side: only A_log / dt_bias stay fp32
            _to_train_dtype(m.load_fields(ts), "cuda", dtype, MEGATRON_FP32_KDA_PARAMS)
        m.reference = reference
        return m

    meg32, meg16 = module(torch.float32, "megatron-fla"), module(torch.bfloat16, "megatron-fla", prec)
    torch16, fused16 = module(torch.bfloat16, "aiter-torch"), module(torch.bfloat16, "aiter-torch")
    assert patch_kda_layers(fused16, backend="fused", gemm_precision=prec, kda_precision="megatron") == 1
    g = torch.Generator(device="cuda").manual_seed(layer)
    x = torch.randn(1, seqlen, cfg["hidden"], generator=g, device="cuda").to(torch.bfloat16)
    dy = torch.randn(x.shape, generator=g, device="cuda").to(torch.bfloat16)
    names = ["y", "dx"] + [n for n, _ in meg32.named_parameters()]
    res = []
    for m in (meg32, meg16, fused16, torch16):
        xs = (x.float() if m is meg32 else x).clone().requires_grad_(True)
        y = m(xs, attention_mask=None)
        y.backward(dy.to(y.dtype))
        torch.cuda.synchronize()
        res.append({n: t.detach().float().cpu() for n, t in zip(names, [y, xs.grad] + [p.grad for p in m.parameters()])})
    status = kda_reference.fla_backward_status()
    if not status.startswith("ok"):
        # backward graded against the PyTorch recurrence (Megatron layer unchanged); forward against FLA
        print(f"FLA chunk_kda backward unavailable ({status}); Megatron reference recurrence: {kda_reference.LAST_BACKEND}")
        with torch.no_grad():
            y_fla = meg16(x, attention_mask=None)
        e = ((res[2]["y"] - y_fla.float().cpu()).norm() / y_fla.float().cpu().norm()).item()
        print(f"  forward aiter vs FLA (bf16): {e:.3e}")
        assert e < 3e-2, f"forward aiter vs FLA {e:.3e}"
    rows, problems = grad_parity_vs_fp32(res[0], res[1], res[2], cfg["heads"])
    rows_t, _ = grad_parity_vs_fp32(res[0], res[1], res[3], cfg["heads"])
    print(f"layer {layer} T={seqlen} precision={prec} vs Megatron-FLA fp32 (megatron_bf16, aiter_fused_bf16, limit):", list(rows.items())[:5])
    print("  [secondary] aiter-torch bf16:", list(rows_t.items())[:5])
    unpatch_kda_layers(fused16)
    assert not problems, f"layer {layer}: {problems}"


def test_is_first_microbatch_and_optimizer_invalidation(monkeypatch):
    """TE semantics (#8739 2130550): the flag set on the module reaches kda_layer_fwd; step() invalidates."""
    from aiter.ops import kda_fused_layer as kfl

    from lumen.models.kimi import kda_aiter

    seen, calls = [], []
    orig = kfl.kda_layer_fwd

    import functools

    @functools.wraps(orig)  # keeps the signature patch_kda_layers inspects
    def spy(*a, is_first_microbatch=None, **k):
        seen.append(is_first_microbatch)
        return orig(*a, is_first_microbatch=is_first_microbatch, **k)

    monkeypatch.setattr(kfl, "kda_layer_fwd", spy)
    monkeypatch.setattr(kfl, "invalidate_weight_caches", lambda weights=None: calls.append(weights))
    m = KimiDeltaAttention(_small_cfg())
    patch_kda_layers(m, backend="torch")
    x = torch.randn(1, 8, 64)
    m(x)
    kda_aiter.set_kda_is_first_microbatch(m, True)
    m(x)
    kda_aiter.set_kda_is_first_microbatch(m, False)
    m(x)
    assert seen == [None, True, False]
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    assert kda_aiter.register_kda_optimizer_hooks(opt)
    m(x).sum().backward()
    opt.step()
    assert calls == [None]
    unpatch_kda_layers(m)
