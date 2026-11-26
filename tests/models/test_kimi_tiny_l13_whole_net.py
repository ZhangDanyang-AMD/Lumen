###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Whole-network Tiny-L13 graded against the Megatron-LM#6872 KDA layer and protocol.

The full 13-layer ``Qinghao/Kimi-K3-Tiny-L13`` (remote code) runs on one gfx950 GPU.
Reference: every KDA layer runs the Megatron-Core KDA layer with the FLA recurrence
(``kda_reference.megatron_kda_forward``), projections in ``LUMEN_KIMI_KDA_PRECISION``
(default ``mxfp8``, TE MXFP8BlockScaling as in the #6872 recipes). Candidate: the
same layers through aiter ``kda_layer_fwd`` (``LUMEN_KDA_AITER=1``) at the same
GEMM precision. Everything outside KDA is unchanged on both sides.

Primary (pass/fail), the #6872 continuation protocol: per-step LM-loss difference
within 1e-3 over ``LUMEN_KIMI_WHOLE_NET_STEPS`` (default 32) steps from the same
weights and data, with the recipe's learning-rate schedule (1e-5 cosine to 1e-6
over 50 iterations, no warm-up). Secondary diagnostics (printed): step-1 logits
and per-parameter gradients vs the reference.

``AITER_KDA_PIECES`` selects aiter's per-piece kernels; the selection is printed.

    LUMEN_KIMI_TINY_L13_PATH=/path/to/Kimi-K3-Tiny-L13 \\
    pytest -s tests/models/test_kimi_tiny_l13_whole_net.py
"""

import os

import pytest
import torch

from lumen.models.kimi import kda_reference, tiny_l13
from lumen.models.kimi.kda_aiter import (
    KDA_AITER_ENV,
    invalidate_aiter_kda_weight_caches,
    aiter_supports_gemm_precision,
    maybe_patch_kda_layers,
    unpatch_kda_layers,
)
from lumen.models.kimi.kda_stack import MEGATRON_LOSS_PARITY_TOL, MEGATRON_PARITY_STEPS, grad_parity
from lumen.ops.dispatch import _probe_aiter_kda_fused_layer

SEQLEN = int(os.environ.get("LUMEN_KIMI_WHOLE_NET_SEQLEN", "256"))
STEPS = int(os.environ.get("LUMEN_KIMI_WHOLE_NET_STEPS", str(MEGATRON_PARITY_STEPS)))
PRECISION = os.environ.get("LUMEN_KIMI_KDA_PRECISION", "mxfp8")
# continuation recipe (kimi_k3_proxy_9layer ... TP4PP1EP64CP2_muon.yaml): lr 1e-5 -> 1e-6 cosine over 50 iters
LR_MAX, LR_MIN, LR_DECAY_ITERS = 1e-5, 1e-6, 50


def _gfx950():
    return torch.cuda.is_available() and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


pytestmark = [
    pytest.mark.skipif(not _probe_aiter_kda_fused_layer(), reason="aiter kda_fused_layer (aiter#8739) missing"),
    pytest.mark.skipif(not _gfx950(), reason="whole-network Tiny-L13 targets one gfx950 GPU"),
    pytest.mark.skipif(tiny_l13.checkpoint_path() is None, reason="set LUMEN_KIMI_TINY_L13_PATH"),
]


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("transformers")
    if not kda_reference.fla_available():
        pytest.skip("the Megatron KDA reference needs fla-core (fla.ops.kda)")
    if PRECISION != "bf16" and not aiter_supports_gemm_precision():
        pytest.skip(f"aiter kda_layer_fwd has no gemm_precision={PRECISION} (MXFP8 PR stacked on #8739)")
    m = tiny_l13.load_hf_model()
    assert kda_reference.patch_kda_reference(m, PRECISION) == len(tiny_l13.KDA_LAYERS)
    m.train()
    return m


def _lr(step):
    import math

    t = min(step, LR_DECAY_ITERS) / LR_DECAY_ITERS
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * t))


def _batch(model):
    vocab = model.config.get_text_config().vocab_size
    g = torch.Generator(device="cuda").manual_seed(0)
    ids = torch.randint(0, vocab, (1, SEQLEN), generator=g, device="cuda")
    return ids


def _trainable(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad and p.is_floating_point()]


def _run(model, ids, aiter_kda):
    os.environ[KDA_AITER_ENV] = "1" if aiter_kda else "0"
    unpatch_kda_layers(model)
    invalidate_aiter_kda_weight_caches()  # params change in place between calls
    patched = maybe_patch_kda_layers(model, gemm_precision=PRECISION, kda_precision="megatron")
    assert patched == (len(tiny_l13.KDA_LAYERS) if aiter_kda else 0)
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    torch.cuda.synchronize()
    grads = {n: p.grad.detach().float().cpu() for n, p in _trainable(model) if p.grad is not None}
    return out.loss.detach().float().item(), out.logits.detach().float().cpu(), grads


def _rel_norm(ref, out):
    return ((out - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def _report_pieces():
    from aiter.ops import kda_fused_layer as kfl

    print(f"AITER_KDA_PIECES={os.environ.get('AITER_KDA_PIECES', '')!r}")
    print("selected:", kfl.kda_piece_choices())
    print("ran:", dict(sorted(kfl.LAST_PIECES.items())))


def test_whole_net_step1_vs_megatron(model):
    """Primary: step-1 LM loss within 1e-3 of the Megatron KDA reference. Secondary: logits and grads."""
    ids = _batch(model)
    ref_loss, ref_logits, ref_grads = _run(model, ids, aiter_kda=False)
    loss, logits, grads = _run(model, ids, aiter_kda=True)
    _report_pieces()
    unpatch_kda_layers(model)
    from aiter.ops import kda_fused_layer as kfl

    heads = model.config.get_text_config().linear_attn_config["num_heads"]
    rel, problems, diag = grad_parity(ref_grads, grads, heads)
    print(f"precision={PRECISION} gemm={kfl.LAST_PIECES.get('gemm')}")
    print(f"[secondary] logits ||a-b||/||b|| = {_rel_norm(ref_logits, logits):.3e}")
    print("[secondary] worst grad ||a-b||/||b||:", list(rel.items())[:5])
    for n, rows in list(diag.items())[:3]:
        print(f"[secondary] {n} largest |diff| heads (idx, megatron, aiter):", rows)
    if problems:
        print("[secondary] padded A_log:", problems)
    print(f"[primary] loss megatron={ref_loss:.6f} aiter={loss:.6f} |d|={abs(loss - ref_loss):.3e}")
    assert abs(loss - ref_loss) <= MEGATRON_LOSS_PARITY_TOL


def _train(model, ids, aiter_kda, init):
    with torch.no_grad():
        for n, p in _trainable(model):
            p.copy_(init[n])
    losses = []
    for step in range(STEPS):
        loss, _, _ = _run(model, ids, aiter_kda)
        losses.append(loss)
        with torch.no_grad():
            for _, p in _trainable(model):
                if p.grad is not None:
                    p.add_(p.grad, alpha=-_lr(step))
    return losses


def test_whole_net_continuation_vs_megatron(model):
    """Primary: Megatron-LM#6872 continuation protocol, max per-step |dloss| <= 1e-3 over STEPS."""
    ids = _batch(model)
    init = {n: p.detach().cpu().clone() for n, p in _trainable(model)}
    ref = _train(model, ids, False, init)
    got = _train(model, ids, True, init)
    _report_pieces()
    unpatch_kda_layers(model)
    with torch.no_grad():
        for n, p in _trainable(model):
            p.copy_(init[n])
    d = [abs(o - r) for r, o in zip(ref, got)]
    print("loss megatron:", ref)
    print("loss aiter   :", got)
    print(f"[primary] max |dloss| {max(d):.3e} at step {d.index(max(d)) + 1} (tol {MEGATRON_LOSS_PARITY_TOL:g})")
    assert max(d) <= MEGATRON_LOSS_PARITY_TOL
