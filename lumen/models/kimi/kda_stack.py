###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""KDA-only Kimi-K3 / K3.1 network built from a config preset, randomly initialized.

Embedding -> N x (RMSNorm -> KDA -> residual) -> RMSNorm -> LM head. MLA and MoE
(and K3.1 DSA) blocks are omitted, so no checkpoint or compressed-tensors is
needed. K3 Tiny-L13 and K3.1-Pro share the KDA widths (hidden 7168, 96 heads x
128, conv 4, gate rank 128, vocab 163840, ``gate_lower_bound=-5``, per-head
``A_log`` padded to 128); they differ in which layers are KDA (9 of 13 vs 69 of 93).

``KimiDeltaAttention`` here has the remote-code parameter names and fp32
conv/``A_log``/``dt_bias``/``o_norm``; its own forward is aiter's ``torch``
reference (``kda_layer_fwd(backend="torch")``). ``patch_kda_layers`` swaps it
for the aiter fused layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumen.models.kimi import tiny_l13
from lumen.models.kimi.kda_aiter import kda_layer_weights

TINY_L13_VOCAB = 163840
FP32_KDA_PARAMS = ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight", "A_log", "dt_bias", "o_norm.weight")


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(x.dtype)


class KimiDeltaAttention(nn.Module):
    def __init__(self, cfg, layer_idx=0, gate_lower_bound=tiny_l13.GATE_LOWER_BOUND):
        super().__init__()
        D, H, K, W, R = (cfg[n] for n in ("hidden", "heads", "head_dim", "conv", "gate_rank"))
        HK = H * K
        self.layer_idx, self.num_heads, self.head_dim, self.gate_lower_bound = layer_idx, H, K, gate_lower_bound
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
        self.o_norm = RMSNorm(K, cfg["eps"])
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

    def forward(self, hidden_states, attention_mask=None, cache_params=None, **kwargs):
        from aiter.ops.kda_fused_layer import kda_layer_fwd

        return kda_layer_fwd(
            hidden_states,
            kda_layer_weights(self),
            self.num_heads,
            eps=self.o_norm.eps,
            backend="torch",
            cu_seqlens=kwargs.get("cu_seqlens"),
            lower_bound=self.gate_lower_bound,
        )


class KDABlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg["hidden"], cfg["eps"])
        self.self_attn = KimiDeltaAttention(cfg, layer_idx, cfg["gate_lower_bound"])

    def forward(self, x):
        return x + self.self_attn(self.input_layernorm(x))


class KDAStackLM(nn.Module):
    def __init__(self, cfg, vocab=None, num_layers=None):
        vocab = vocab or cfg.get("vocab") or TINY_L13_VOCAB
        super().__init__()
        layers = cfg["kda_layers"][:num_layers] if num_layers else cfg["kda_layers"]
        self.embed_tokens = nn.Embedding(vocab, cfg["hidden"])
        self.layers = nn.ModuleList(KDABlock(cfg, i) for i in layers)
        self.norm = RMSNorm(cfg["hidden"], cfg["eps"])
        self.lm_head = nn.Linear(cfg["hidden"], vocab, bias=False)

    def forward(self, input_ids, labels=None):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        logits = self.lm_head(self.norm(h))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten())
        return loss, logits


def build_kda_stack(
    cfg=None, vocab=None, num_layers=None, device="cuda", dtype=torch.bfloat16, seed=0, preset="k3-tiny-l13"
):
    """Random-init KDA-only network over the first ``num_layers`` config KDA layers (all by default)."""
    cfg = cfg or tiny_l13.load_config(None, preset=preset)
    torch.manual_seed(seed)
    model = KDAStackLM(cfg, vocab, num_layers)
    for block in model.layers:
        attn = block.self_attn
        attn.load_fields(tiny_l13.random_tensors(cfg, torch.float32, seed=seed + attn.layer_idx))
    nn.init.normal_(model.embed_tokens.weight, std=1.0)
    nn.init.normal_(model.lm_head.weight, std=cfg["hidden"] ** -0.5)
    model.to(device=device, dtype=dtype)
    for name, p in model.named_parameters():
        if name.endswith(FP32_KDA_PARAMS):
            p.data = p.data.float()
    return model
