###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tier 1: KDA-only K3 Tiny-L13 / K3.1-Pro stack (random init), ``LUMEN_KDA_AITER=1`` vs ``0``.

Loss, logits and every parameter gradient, then an SGD trajectory following
the Megatron-LM#6872 continuation protocol (32 steps, lr 1e-5 cosine to 1e-6;
primary: per-step |aiter - eager| <= 1e-3). In bf16 the trajectories are also
compared like #6872's continuation checks, two implementations of the same
recipe: aiter fused vs aiter torch, both in aiter's Megatron rounding mode,
per step |dloss| <= 1e-3. The fp32 trajectory is printed as the bf16 noise floor. The GPU test uses the real Tiny-L13 widths (9 KDA layers,
hidden 7168, 96 x 128 heads, vocab 163840) in BF16 on one gfx950 GPU; the CPU
test uses tiny widths in fp32. Per-piece aiter kernels follow ``AITER_KDA_PIECES``.

    pytest -s tests/models/test_kimi_kda_stack.py
"""

import contextlib
import copy
import os

import pytest
import torch

from lumen.models.kimi import kda_reference, kda_stack, tiny_l13
from lumen.models.kimi.kda_aiter import (
    KDA_AITER_ENV,
    invalidate_aiter_kda_weight_caches,
    maybe_patch_kda_layers,
    set_aiter_kda_precision,
    unpatch_kda_layers,
)
from lumen.ops.dispatch import _probe_aiter_kda_fused_layer

pytestmark = pytest.mark.skipif(not _probe_aiter_kda_fused_layer(), reason="aiter kda_fused_layer (aiter#8739) missing")

SEQLEN = int(os.environ.get("LUMEN_KIMI_KDA_STACK_SEQLEN", "512"))
# GEMM precision of the Megatron-LM#6872 comparison; mxfp8 matches its recipes (TE MXFP8BlockScaling)
MEGATRON_PRECISION = os.environ.get("LUMEN_KIMI_KDA_PRECISION", "mxfp8")


def _skip_unless_precision(precision):
    from lumen.models.kimi.kda_aiter import aiter_supports_gemm_precision

    if precision != "bf16" and not aiter_supports_gemm_precision():
        pytest.skip(f"aiter kda_layer_fwd has no gemm_precision={precision} (MXFP8 PR stacked on #8739)")
STEPS = int(os.environ.get("LUMEN_KIMI_KDA_STACK_STEPS", str(kda_stack.MEGATRON_PARITY_STEPS)))


def _gfx950():
    return torch.cuda.is_available() and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


def _step(model, ids, aiter_kda, gemm_precision="bf16", kda_precision=None):
    os.environ[KDA_AITER_ENV] = "1" if aiter_kda else "0"
    unpatch_kda_layers(model)
    invalidate_aiter_kda_weight_caches()  # params change in place between calls
    patched = maybe_patch_kda_layers(model, gemm_precision=gemm_precision, kda_precision=kda_precision)
    assert patched == (len(model.layers) if aiter_kda else 0)
    if kda_precision is not None:
        set_aiter_kda_precision(kda_precision)  # the eager (aiter torch) side uses the same rounding
    model.zero_grad(set_to_none=True)
    loss, logits = model(ids, labels=ids)
    loss.backward()
    if ids.is_cuda:
        torch.cuda.synchronize()
    grads = {n: p.grad.detach().float().cpu() for n, p in model.named_parameters() if p.grad is not None}
    unpatch_kda_layers(model)
    return loss.item(), logits.detach().float().cpu(), grads


def _fp32_reference(model, ids):
    """Eager grads of an fp32 copy (bf16 weights upcast exactly)."""
    ref = copy.deepcopy(model).float()
    _, _, g = _step(ref, ids, False)
    del ref
    return g


def _sgd_curve(model, ids, aiter_kda, init, lr, kda_precision=None):
    """SGD trajectory from ``init``; ``lr=None`` follows the #6872 continuation schedule."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(init[n])
    curve = []
    for step in range(STEPS):
        l, _, grads = _step(model, ids, aiter_kda, kda_precision=kda_precision)
        curve.append(l)
        step_lr = kda_stack.megatron_continuation_lr(step) if lr is None else lr
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.add_(grads[n].to(p), alpha=-step_lr)
    return curve


def _check(model, ids, tol, loss_tol, lr, grad_tol=kda_stack.GRAD_REL_NORM_TOL, vs_fp32=False):
    init = {n: p.detach().clone() for n, p in model.named_parameters()}
    ref_loss, ref_logits, ref_g = _step(model, ids, False)
    loss, logits, g = _step(model, ids, True)
    from aiter.ops import kda_fused_layer as kfl

    print("aiter pieces:", dict(sorted(kfl.LAST_PIECES.items())), "gemm:", kfl.LAST_PIECES.get("gemm"))
    assert abs(loss - ref_loss) <= loss_tol, f"loss eager {ref_loss} vs aiter {loss}"
    s = ref_logits.abs().max()
    torch.testing.assert_close(logits / s, ref_logits / s, rtol=tol, atol=tol)
    assert ref_g.keys() == g.keys() == {n for n, _ in model.named_parameters()}
    heads = model.layers[0].self_attn.num_heads
    if vs_fp32:
        rows, problems = kda_stack.grad_parity_vs_fp32(_fp32_reference(model, ids), ref_g, g, heads)
        print("worst grad vs fp32 (eager_bf16, aiter_bf16, limit):", list(rows.items())[:5])
        assert not problems, problems
    else:
        rel, problems, diag = kda_stack.grad_parity(ref_g, g, heads)
        print("worst grad ||a-b||/||b||:", list(rel.items())[:5])
        for n, rows in list(diag.items())[:3]:
            print(f"{n} largest |diff| heads (idx, eager, aiter):", rows)
        assert not problems, problems
        bad = {n: v for n, v in rel.items() if v > grad_tol}
        assert not bad, f"grad rel-norm over {grad_tol}: {bad}"

    # bf16: both sides in aiter's Megatron rounding mode, i.e. the same precision recipe, which is
    # what #6872's 1e-3 continuation criterion compares (two implementations, same MXFP8/bf16 recipe).
    mode = "megatron" if vs_fp32 else None
    curves = {"eager": _sgd_curve(model, ids, False, init, lr, mode), "aiter": _sgd_curve(model, ids, True, init, lr, mode)}
    set_aiter_kda_precision(None)
    if vs_fp32:
        ref = copy.deepcopy(model).float()
        curves["fp32"] = _sgd_curve(ref, ids, False, {n: t.float() for n, t in init.items()}, lr, "default")
        set_aiter_kda_precision(None)
        del ref
    for name, c in curves.items():
        print(f"loss {name:6s}:", [f"{v:.6f}" for v in c])
    d = [abs(o - r) for r, o in zip(curves["eager"], curves["aiter"])]
    if not vs_fp32:
        # fp32 wiring check: both sides compute in fp32, so they must agree directly
        print(f"[primary] max |aiter - eager| {max(d):.3e} at step {d.index(max(d)) + 1} (tol {loss_tol:g})")
        assert max(d) <= loss_tol, f"step {d.index(max(d)) + 1}: |aiter - eager| {max(d):.3e} > {loss_tol:g}"
        return
    # #6872's criterion compares two runs of the same recipe (e.g. a memory/CP-layout change against
    # its baseline, both MXFP8/bf16): max per-step |dloss| <= 1e-3. bf16 against fp32 is not that
    # comparison: bf16 rounding alone moves the loss by several 1e-3 here, so fp32 is reported only.
    de = [abs(e - f) for f, e in zip(curves["fp32"], curves["eager"])]
    da = [abs(a - f) for f, a in zip(curves["fp32"], curves["aiter"])]
    for i in range(len(d)):
        print(f"  step {i + 1:2d} |aiter-eager| {d[i]:.3e} |eager-fp32| {de[i]:.3e} |aiter-fp32| {da[i]:.3e}")
    print(f"[primary] #6872 same-recipe parity: max |aiter - eager| {max(d):.3e} at step {d.index(max(d)) + 1} "
          f"(tol {loss_tol:g}, both in aiter megatron precision)")  # fmt: skip
    print(f"[secondary] bf16 noise floor vs fp32: max |eager-fp32| {max(de):.3e}, max |aiter-fp32| {max(da):.3e}")
    assert max(d) <= loss_tol, f"step {d.index(max(d)) + 1}: |aiter - eager| {max(d):.3e} > {loss_tol:g}"


def test_k31_preset_layer_split():
    cfg = tiny_l13.load_config(None, preset="k3.1-pro")
    assert cfg["num_layers"] == 93 and len(cfg["kda_layers"]) == 69 and len(cfg["mla_layers"]) == 24
    assert cfg["kda_layers"][:9] == (0, 1, 2, 4, 5, 6, 8, 9, 10)
    assert cfg["mla_layers"][-2:] == (91, 92)
    k3 = tiny_l13.load_config(None)
    for key in ("hidden", "heads", "head_dim", "conv", "gate_rank", "eps", "gate_lower_bound", "vocab"):
        assert cfg[key] == k3[key], key


@contextlib.contextmanager
def _steps(n):
    """Shorter trajectory for the small-width wiring check (a large lr makes 5 steps informative)."""
    global STEPS
    saved, STEPS = STEPS, n
    try:
        yield
    finally:
        STEPS = saved


def _small_device():
    """GPU when present (ROCm torch builds may lack CPU LAPACK for triangular_solve), else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    try:
        torch.linalg.solve_triangular(torch.eye(2), torch.ones(2, 1), upper=False)
    except RuntimeError as e:
        pytest.skip(f"no GPU and CPU LAPACK unavailable: {e}")
    return "cpu"


@pytest.mark.parametrize("preset", tiny_l13.PRESETS)
def test_kda_stack_small_fp32(preset):
    device = _small_device()
    cfg = dict(tiny_l13.load_config(None, preset=preset), hidden=64, heads=2, head_dim=16, gate_rank=16)
    model = kda_stack.build_kda_stack(cfg, vocab=97, num_layers=2, device=device, dtype=torch.float32)
    ids = torch.randint(0, 97, (1, 70), generator=torch.Generator().manual_seed(0)).to(device)
    with _steps(5):
        _check(model, ids, tol=1e-3, loss_tol=1e-4, lr=1e-1, grad_tol=1e-4)


@pytest.mark.skipif(not _gfx950(), reason="tier-1 KDA stack targets one gfx950 GPU")
@pytest.mark.parametrize("preset,num_layers", [("k3-tiny-l13", None), ("k3.1-pro", 9)])
def test_kda_stack_bf16(preset, num_layers):
    model = kda_stack.build_kda_stack(preset=preset, num_layers=num_layers)
    assert len(model.layers) == (num_layers or len(tiny_l13.KDA_LAYERS))
    ids = torch.randint(0, kda_stack.TINY_L13_VOCAB, (1, SEQLEN), device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(0))  # fmt: skip
    _check(model, ids, tol=5e-2, loss_tol=kda_stack.MEGATRON_LOSS_PARITY_TOL, lr=None, vs_fp32=True)


@pytest.mark.skipif(not _gfx950(), reason="tier-1 KDA stack targets one gfx950 GPU")
@pytest.mark.skipif(not kda_reference.fla_available(), reason="Megatron-LM#6872 reference needs fla-core (fla.ops.kda)")
@pytest.mark.parametrize("preset,num_layers", [("k3-tiny-l13", None), ("k3.1-pro", 9)])
def test_kda_stack_vs_megatron_fla(preset, num_layers):
    """Primary: Megatron-LM#6872 loss parity (per-step |dloss| <= 1e-3) of aiter vs the Megatron KDA
    layer (FLA recurrence), both with KDA projections in ``LUMEN_KIMI_KDA_PRECISION`` (default mxfp8).
    Secondary diagnostics: step-1 grads vs Megatron and vs aiter-torch (bf16 GEMMs)."""
    prec = MEGATRON_PRECISION
    _skip_unless_precision(prec)
    model = kda_stack.build_kda_stack(
        preset=preset, num_layers=num_layers, fp32_params=kda_stack.MEGATRON_FP32_KDA_PARAMS
    )
    kda_stack.set_reference(model, "megatron-fla", prec)
    ids = torch.randint(0, kda_stack.TINY_L13_VOCAB, (1, SEQLEN), device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(0))  # fmt: skip
    init = {n: p.detach().clone() for n, p in model.named_parameters()}
    heads = model.layers[0].self_attn.num_heads

    _, _, g_meg = _step(model, ids, False)
    # aiter in Megatron rounding mode against the Megatron reference
    _, _, g_aiter = _step(model, ids, True, prec, "megatron")
    kda_stack.set_reference(model, "aiter-torch", "bf16")
    _, _, g_torch = _step(model, ids, False)
    kda_stack.set_reference(model, "megatron-fla", prec)
    print(f"precision={prec} megatron recurrence={kda_reference.LAST_BACKEND} "
          f"(FLA backward: {kda_reference.fla_backward_status()})")
    rel_a, _, _ = kda_stack.grad_parity(g_meg, g_aiter, heads)
    rel_t, _, _ = kda_stack.grad_parity(g_meg, g_torch, heads)
    print("[secondary] step-1 grad ||x-megatron||/||megatron||, worst aiter-fused:", list(rel_a.items())[:5])
    print("[secondary] worst aiter-torch:", list(rel_t.items())[:5])

    curves = {}
    for aiter_kda in (False, True):
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(init[n])
        curve = []
        for step in range(STEPS):
            l, _, grads = _step(model, ids, aiter_kda, prec, "megatron" if aiter_kda else None)
            curve.append(l)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.add_(grads[n].to(p), alpha=-kda_stack.megatron_continuation_lr(step))
        curves[aiter_kda] = curve
    d = [abs(a - b) for a, b in zip(curves[False], curves[True])]
    print("[primary] loss megatron-fla:", curves[False])
    print("[primary] loss aiter       :", curves[True])
    print(f"[primary] max |dloss| {max(d):.3e} (tol {kda_stack.MEGATRON_LOSS_PARITY_TOL:g})")
    assert max(d) <= kda_stack.MEGATRON_LOSS_PARITY_TOL


def test_kda_varlen_cu_seqlens_matches_separate_sequences():
    """Packed THD (cu_seqlens) through the patched layer equals running each sequence alone."""
    device = _small_device()
    cfg = dict(tiny_l13.load_config(None), hidden=64, heads=2, head_dim=16, gate_rank=16)
    attn = kda_stack.build_kda_stack(cfg, vocab=97, num_layers=1, device=device, dtype=torch.float32).layers[0].self_attn
    from lumen.models.kimi.kda_aiter import patch_kda_layers

    patch_kda_layers(attn, backend="fused")
    lens = [37, 70, 21]
    xs = [torch.randn(1, n, 64, device=device) for n in lens]
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device=device, dtype=torch.int32)
    packed = attn(torch.cat(xs, 1), cu_seqlens=cu)
    separate = torch.cat([attn(x) for x in xs], 1)
    torch.testing.assert_close(packed, separate, rtol=1e-4, atol=1e-5)
    unpatch_kda_layers(attn)
