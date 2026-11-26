###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Whole-network Tiny-L13: KDA layers via aiter (``LUMEN_KDA_AITER=1``) vs the model's own KDA.

The full 13-layer ``Qinghao/Kimi-K3-Tiny-L13`` (remote code, everything except
the 9 KDA layers unchanged) runs BF16 forward + backward on one gfx950 GPU. Loss,
logits and every parameter gradient are compared between the two KDA paths on
the same weights, then a few SGD steps check that the loss curves match.

Per-piece kernels inside aiter's fused layer follow ``AITER_KDA_PIECES`` (e.g.
``AITER_KDA_PIECES=chunk_fwd=flydsl``) so FlyDSL pieces can be enabled one at a
time; the selection and the pieces that actually ran are printed.

Run (see ``lumen/models/kimi/README.md``)::

    LUMEN_KIMI_TINY_L13_PATH=/path/to/Kimi-K3-Tiny-L13 \\
    AITER_KDA_PIECES=chunk_fwd=flydsl \\
    pytest -s tests/models/test_kimi_tiny_l13_whole_net.py

``LUMEN_KIMI_WHOLE_NET_SEQLEN`` (default 256) and ``LUMEN_KIMI_WHOLE_NET_STEPS``
(default 3) size the run.
"""

import os

import pytest
import torch

from lumen.models.kimi import tiny_l13
from lumen.models.kimi.kda_aiter import KDA_AITER_ENV, maybe_patch_kda_layers, unpatch_kda_layers
from lumen.ops.dispatch import _probe_aiter_kda_fused_layer

SEQLEN = int(os.environ.get("LUMEN_KIMI_WHOLE_NET_SEQLEN", "256"))
STEPS = int(os.environ.get("LUMEN_KIMI_WHOLE_NET_STEPS", "3"))
LR = 1e-4
TOL = 5e-2


def _gfx950():
    return torch.cuda.is_available() and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


pytestmark = [
    pytest.mark.skipif(not _probe_aiter_kda_fused_layer(), reason="aiter kda_fused_layer (aiter#8739) missing"),
    pytest.mark.skipif(not _gfx950(), reason="whole-network Tiny-L13 targets one gfx950 GPU"),
    pytest.mark.skipif(tiny_l13.checkpoint_path() is None, reason="set LUMEN_KIMI_TINY_L13_PATH"),
]


def _load_real_a_log(model, path):
    """The checkpoint's per-head A_log is padded to [128]; the remote-code init is [num_heads]."""
    cfg = tiny_l13.load_config(path)
    names = {i: tiny_l13.HF_PREFIX.format(i) + "A_log" for i in cfg["kda_layers"]}
    raw = tiny_l13._read(path, names.values())
    for name, mod in model.named_modules():
        if type(mod).__name__ == "KimiDeltaAttention":
            t = raw[names[mod.layer_idx]].reshape(-1).float().to(mod.A_log.device)
            mod.A_log = torch.nn.Parameter(t)


@pytest.fixture(scope="module")
def model():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("fla", reason="the eager KDA reference needs fla-core")
    path = tiny_l13.checkpoint_path()
    m = transformers.AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype="auto", device_map="cuda", ignore_mismatched_sizes=True
    )
    _load_real_a_log(m, path)
    m.train()
    return m


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
    patched = maybe_patch_kda_layers(model)
    assert patched == (len(tiny_l13.KDA_LAYERS) if aiter_kda else 0)
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    torch.cuda.synchronize()
    grads = {n: p.grad.detach().float().cpu() for n, p in _trainable(model) if p.grad is not None}
    return out.loss.detach().float().item(), out.logits.detach().float().cpu(), grads


def _scaled_close(name, ref, out):
    s = ref.abs().max().clamp_min(1e-6)
    torch.testing.assert_close(out / s, ref / s, rtol=TOL, atol=TOL, msg=lambda m: f"{name}: {m}")


def _report_pieces():
    from aiter.ops import kda_fused_layer as kfl

    print(f"AITER_KDA_PIECES={os.environ.get('AITER_KDA_PIECES', '')!r}")
    print("selected:", kfl.kda_piece_choices())
    print("ran:", dict(sorted(kfl.LAST_PIECES.items())))


def test_whole_net_fwd_bwd_parity(model):
    ids = _batch(model)
    ref_loss, ref_logits, ref_grads = _run(model, ids, aiter_kda=False)
    loss, logits, grads = _run(model, ids, aiter_kda=True)
    _report_pieces()
    unpatch_kda_layers(model)

    print(f"loss eager={ref_loss:.6f} aiter={loss:.6f}")
    assert abs(loss - ref_loss) <= TOL * max(1.0, abs(ref_loss))
    _scaled_close("logits", ref_logits, logits)
    assert ref_grads.keys() == grads.keys(), "parameters with gradients differ between paths"
    kda_names = [n for n in grads if ".self_attn." in n and any(f"layers.{i}." in n for i in tiny_l13.KDA_LAYERS)]
    assert kda_names, "no KDA parameter received a gradient"
    for n in ref_grads:
        _scaled_close(n, ref_grads[n], grads[n])


def _train(model, ids, aiter_kda, init):
    with torch.no_grad():
        for n, p in _trainable(model):
            p.copy_(init[n])
    losses = []
    for _ in range(STEPS):
        loss, _, _ = _run(model, ids, aiter_kda)
        losses.append(loss)
        with torch.no_grad():
            for _, p in _trainable(model):
                if p.grad is not None:
                    p.add_(p.grad, alpha=-LR)
    return losses


def test_whole_net_loss_curve(model):
    ids = _batch(model)
    init = {n: p.detach().cpu().clone() for n, p in _trainable(model)}
    ref = _train(model, ids, False, init)
    got = _train(model, ids, True, init)
    _report_pieces()
    unpatch_kda_layers(model)
    with torch.no_grad():
        for n, p in _trainable(model):
            p.copy_(init[n])
    print("loss curve eager:", ref)
    print("loss curve aiter:", got)
    assert ref[-1] < ref[0], "eager loss did not decrease; the comparison is not meaningful"
    for step, (r, o) in enumerate(zip(ref, got)):
        assert abs(o - r) <= TOL * max(1.0, abs(r)), f"step {step}: eager {r:.6f} vs aiter {o:.6f}"
