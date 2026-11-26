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

import pytest
import torch
import torch.nn as nn

from lumen.models.kimi import tiny_l13
from lumen.models.kimi.kda_aiter import kda_layer_weights, patch_kda_layers, unpatch_kda_layers
from lumen.ops.dispatch import _probe_aiter_kda_fused_layer

pytestmark = pytest.mark.skipif(not _probe_aiter_kda_fused_layer(), reason="aiter kda_fused_layer (aiter#8739) missing")


class _RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps


class KimiDeltaAttention(nn.Module):
    """HF-named KDA parameter layout; the eager forward is aiter's torch reference."""

    def __init__(self, cfg, gate_lower_bound=tiny_l13.GATE_LOWER_BOUND):
        super().__init__()
        D, H, K, W, R = (cfg[n] for n in ("hidden", "heads", "head_dim", "conv", "gate_rank"))
        HK = H * K
        self.num_heads, self.head_dim, self.gate_lower_bound = H, K, gate_lower_bound
        self.q_proj, self.k_proj, self.v_proj = (nn.Linear(D, HK, bias=False) for _ in range(3))
        self.q_conv1d, self.k_conv1d, self.v_conv1d = (
            nn.Conv1d(HK, HK, W, groups=HK, padding=W - 1, bias=False) for _ in range(3)
        )
        self.f_a_proj = nn.Linear(D, R, bias=False)
        self.f_b_proj = nn.Linear(R, HK, bias=False)
        self.A_log = nn.Parameter(torch.zeros(max(H, tiny_l13.A_LOG_LEN)))
        self.dt_bias = nn.Parameter(torch.zeros(HK))
        self.b_proj = nn.Linear(D, H, bias=False)
        self.g_proj = nn.Linear(D, HK, bias=False)
        self.o_norm = _RMSNorm(K, cfg["eps"])
        self.o_proj = nn.Linear(HK, D, bias=False)

    def load_fields(self, ts):
        params = {
            "w_q": self.q_proj.weight, "w_k": self.k_proj.weight, "w_v": self.v_proj.weight,
            "conv_q": self.q_conv1d.weight, "conv_k": self.k_conv1d.weight, "conv_v": self.v_conv1d.weight,
            "f_a": self.f_a_proj.weight, "f_b": self.f_b_proj.weight, "A_log": self.A_log,
            "dt_bias": self.dt_bias, "w_beta": self.b_proj.weight, "g_proj": self.g_proj.weight,
            "norm_w": self.o_norm.weight, "w_o": self.o_proj.weight,
        }  # fmt: skip
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(ts[n].reshape(p.shape))
        return self

    def to_train_dtype(self, device, dtype):
        self.to(device, dtype)
        for m in (self.q_conv1d, self.k_conv1d, self.v_conv1d, self.o_norm):
            m.weight.data = m.weight.data.float()
        self.A_log.data, self.dt_bias.data = self.A_log.data.float(), self.dt_bias.data.float()
        return self

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        from aiter.ops.kda_fused_layer import kda_layer_fwd

        o = kda_layer_fwd(
            hidden_states,
            kda_layer_weights(self),
            self.num_heads,
            eps=self.o_norm.eps,
            backend="torch",
            lower_bound=self.gate_lower_bound,
        )
        return o, None, past_key_values


def _small_cfg():
    return dict(tiny_l13.load_config(None), hidden=64, heads=2, head_dim=16, gate_rank=16)


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
        y, _, _ = mod(xs, attention_mask=None)
        y.sum().backward()
        outs.append([y.detach(), xs.grad] + [p.grad for p in mod.parameters()])
    for r, o in zip(*outs):
        assert r is not None and o is not None
        torch.testing.assert_close(o, r, rtol=1e-4, atol=1e-4)
    assert unpatch_kda_layers(m) == 1


def test_cache_keeps_module_forward():
    m = KimiDeltaAttention(_small_cfg())
    patch_kda_layers(m, backend="fused")
    _, _, past = m(torch.randn(1, 8, 64), past_key_values="cache")
    assert past == "cache"


def _gfx950():
    return torch.cuda.is_available() and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


@pytest.mark.skipif(not _gfx950(), reason="Tiny-L13 KDA parity targets one gfx950 GPU")
@pytest.mark.parametrize("seqlen", [100, 256])
@pytest.mark.parametrize("layer", tiny_l13.KDA_LAYERS)
def test_tiny_l13_kda_parity_bf16(layer, seqlen):
    """BF16 fwd+bwd: patched module (aiter fused) vs the eager module forward (torch)."""
    cfg = tiny_l13.load_config()
    ts = tiny_l13.load_kda_tensors(layer, cfg=cfg)
    lb = cfg["gate_lower_bound"]
    assert lb == tiny_l13.GATE_LOWER_BOUND
    ref = KimiDeltaAttention(cfg, lb).load_fields(ts).to_train_dtype("cuda", torch.bfloat16)
    test = KimiDeltaAttention(cfg, lb).load_fields(ts).to_train_dtype("cuda", torch.bfloat16)
    assert patch_kda_layers(test, backend="fused") == 1

    g = torch.Generator(device="cuda").manual_seed(layer)
    x = torch.randn(1, seqlen, cfg["hidden"], generator=g, device="cuda").to(torch.bfloat16)
    dy = torch.randn(x.shape, generator=g, device="cuda").to(torch.bfloat16)
    outs = []
    for m in (ref, test):
        xs = x.clone().requires_grad_(True)
        y, _, _ = m(xs, attention_mask=None)
        y.backward(dy)
        torch.cuda.synchronize()
        outs.append([y.detach(), xs.grad] + [p.grad for p in m.parameters()])

    names = ["y", "dx"] + [n for n, _ in ref.named_parameters()]
    for name, r, o in zip(names, *outs):
        assert r is not None and o is not None, f"layer {layer}: {name} has no gradient"
        # grade each output against its own scale so a small gradient is not hidden
        s = r.float().abs().max().clamp_min(1e-6)
        torch.testing.assert_close(o.float() / s, r.float() / s, rtol=5e-2, atol=5e-2, msg=f"layer {layer}: {name}")
