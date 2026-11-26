###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tier 1: KDA-only K3 Tiny-L13 / K3.1-Pro stack (random init), ``LUMEN_KDA_AITER=1`` vs ``0``.

Loss, logits and every parameter gradient, then a short SGD run whose loss
curves must agree. The GPU test uses the real Tiny-L13 widths (9 KDA layers,
hidden 7168, 96 x 128 heads, vocab 163840) in BF16 on one gfx950 GPU; the CPU
test uses tiny widths in fp32. Per-piece aiter kernels follow ``AITER_KDA_PIECES``.

    pytest -s tests/models/test_kimi_kda_stack.py
"""

import os

import pytest
import torch

from lumen.models.kimi import kda_stack, tiny_l13
from lumen.models.kimi.kda_aiter import KDA_AITER_ENV, maybe_patch_kda_layers, unpatch_kda_layers
from lumen.ops.dispatch import _probe_aiter_kda_fused_layer

pytestmark = pytest.mark.skipif(not _probe_aiter_kda_fused_layer(), reason="aiter kda_fused_layer (aiter#8739) missing")

SEQLEN = int(os.environ.get("LUMEN_KIMI_KDA_STACK_SEQLEN", "512"))
STEPS = int(os.environ.get("LUMEN_KIMI_KDA_STACK_STEPS", "5"))


def _gfx950():
    return torch.cuda.is_available() and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


def _step(model, ids, aiter_kda):
    os.environ[KDA_AITER_ENV] = "1" if aiter_kda else "0"
    unpatch_kda_layers(model)
    assert maybe_patch_kda_layers(model) == (len(model.layers) if aiter_kda else 0)
    model.zero_grad(set_to_none=True)
    loss, logits = model(ids, labels=ids)
    loss.backward()
    if ids.is_cuda:
        torch.cuda.synchronize()
    grads = {n: p.grad.detach().float().cpu() for n, p in model.named_parameters() if p.grad is not None}
    unpatch_kda_layers(model)
    return loss.item(), logits.detach().float().cpu(), grads


def _check(model, ids, tol, loss_tol, lr):
    init = {n: p.detach().clone() for n, p in model.named_parameters()}
    ref_loss, ref_logits, ref_g = _step(model, ids, False)
    loss, logits, g = _step(model, ids, True)
    assert abs(loss - ref_loss) <= loss_tol, f"loss eager {ref_loss} vs aiter {loss}"
    s = ref_logits.abs().max()
    torch.testing.assert_close(logits / s, ref_logits / s, rtol=tol, atol=tol)
    assert ref_g.keys() == g.keys() == {n for n, _ in model.named_parameters()}
    for n in ref_g:
        s = ref_g[n].abs().max().clamp_min(1e-6)
        torch.testing.assert_close(g[n] / s, ref_g[n] / s, rtol=tol, atol=tol, msg=lambda m: f"{n}: {m}")

    curves = {}
    for aiter_kda in (False, True):
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(init[n])
        curve = []
        for _ in range(STEPS):
            l, _, grads = _step(model, ids, aiter_kda)
            curve.append(l)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.add_(grads[n].to(p), alpha=-lr)
        curves[aiter_kda] = curve
    print("loss curve eager:", curves[False])
    print("loss curve aiter:", curves[True])
    assert curves[False][-1] < curves[False][0]
    for r, o in zip(curves[False], curves[True]):
        assert abs(o - r) <= loss_tol, f"eager {curves[False]} vs aiter {curves[True]}"


def test_k31_preset_layer_split():
    cfg = tiny_l13.load_config(None, preset="k3.1-pro")
    assert cfg["num_layers"] == 93 and len(cfg["kda_layers"]) == 69 and len(cfg["mla_layers"]) == 24
    assert cfg["kda_layers"][:9] == (0, 1, 2, 4, 5, 6, 8, 9, 10)
    assert cfg["mla_layers"][-2:] == (91, 92)
    k3 = tiny_l13.load_config(None)
    for key in ("hidden", "heads", "head_dim", "conv", "gate_rank", "eps", "gate_lower_bound", "vocab"):
        assert cfg[key] == k3[key], key


@pytest.mark.parametrize("preset", tiny_l13.PRESETS)
def test_kda_stack_cpu_small(preset):
    cfg = dict(tiny_l13.load_config(None, preset=preset), hidden=64, heads=2, head_dim=16, gate_rank=16)
    model = kda_stack.build_kda_stack(cfg, vocab=97, num_layers=2, device="cpu", dtype=torch.float32)
    ids = torch.randint(0, 97, (1, 70), generator=torch.Generator().manual_seed(0))
    _check(model, ids, tol=1e-4, loss_tol=1e-5, lr=1e-1)


@pytest.mark.skipif(not _gfx950(), reason="tier-1 KDA stack targets one gfx950 GPU")
@pytest.mark.parametrize("preset,num_layers", [("k3-tiny-l13", None), ("k3.1-pro", 9)])
def test_kda_stack_bf16(preset, num_layers):
    model = kda_stack.build_kda_stack(preset=preset, num_layers=num_layers)
    assert len(model.layers) == (num_layers or len(tiny_l13.KDA_LAYERS))
    ids = torch.randint(0, kda_stack.TINY_L13_VOCAB, (1, SEQLEN), device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(0))  # fmt: skip
    _check(model, ids, tol=5e-2, loss_tol=1e-3, lr=1e-3)
