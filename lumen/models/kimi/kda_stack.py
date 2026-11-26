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
conv/``A_log``/``dt_bias``/``o_norm``. Its own forward is selected by
``reference``: ``"megatron-fla"`` (the Megatron-LM#6872 KDA layer with the FLA
recurrence, ``kda_reference.megatron_kda_forward``) or ``"aiter-torch"``
(``kda_layer_fwd(backend="torch")``). ``patch_kda_layers`` swaps it for the aiter
fused layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumen.models.kimi import tiny_l13
from lumen.models.kimi.kda_aiter import kda_layer_weights

TINY_L13_VOCAB = 163840
FP32_KDA_PARAMS = ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight", "A_log", "dt_bias", "o_norm.weight")
# Megatron-LM#6872: only the gate params stay fp32 at runtime; conv and out-norm weights are bf16
# params with fp32 masters (precision map section 3).
MEGATRON_FP32_KDA_PARAMS = ("A_log", "dt_bias")


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
        self.reference = "aiter-torch"
        self.gemm_precision = "bf16"
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
        if self.reference in ("megatron-fla", "megatron-aiter"):
            from lumen.models.kimi.kda_reference import megatron_kda_forward

            return megatron_kda_forward(
                hidden_states,
                kda_layer_weights(self),
                self.num_heads,
                lower_bound=self.gate_lower_bound,
                eps=self.o_norm.eps,
                cu_seqlens=kwargs.get("cu_seqlens"),
                precision=self.gemm_precision,
                backend="aiter" if self.reference == "megatron-aiter" else "fla",
            )
        from aiter.ops.kda_fused_layer import kda_layer_fwd

        extra = {} if self.gemm_precision == "bf16" else {"gemm_precision": self.gemm_precision}
        return kda_layer_fwd(
            hidden_states,
            kda_layer_weights(self),
            self.num_heads,
            eps=self.o_norm.eps,
            backend="torch",
            cu_seqlens=kwargs.get("cu_seqlens"),
            lower_bound=self.gate_lower_bound,
            **extra,
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
    cfg=None,
    vocab=None,
    num_layers=None,
    device="cuda",
    dtype=torch.bfloat16,
    seed=0,
    preset="k3-tiny-l13",
    fp32_params=FP32_KDA_PARAMS,
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
        if name.endswith(tuple(fp32_params)):
            p.data = p.data.float()
    return model


REFERENCES = ("aiter-torch", "megatron-fla", "megatron-aiter")


def set_reference(model, reference, precision=None):
    """Select the unpatched KDA forward (and its GEMM precision) of every ``KimiDeltaAttention``."""
    if reference not in REFERENCES:
        raise ValueError(f"unknown KDA reference {reference!r}, expected {REFERENCES}")
    for m in model.modules():
        if isinstance(m, KimiDeltaAttention):
            m.reference = reference
            if precision is not None:
                m.gemm_precision = precision
    return model


# Megatron-LM#6872 continuation protocol: 32 steps, max per-step LM-loss difference within 1e-3.
MEGATRON_LOSS_PARITY_TOL = 1e-3
MEGATRON_PARITY_STEPS = 32

GRAD_REL_NORM_TOL = 1e-2


def grad_parity(ref, out, num_heads, top=5):
    """Per-parameter ||out - ref|| / ||ref|| and the padded-``A_log`` check.

    A max-abs metric is dominated by a few elements of reduction-heavy gradients
    (``A_log`` sums over every token and channel of a head), so parity is judged
    on the norm ratio. ``A_log`` entries past ``num_heads`` are padding and must
    get exactly zero gradient on both sides.
    Returns (rel_by_param sorted worst-first, list of problems, A_log diagnostics).
    """
    rel, problems, diag = {}, [], {}
    for n, r in ref.items():
        r, o = r.float(), out[n].float()
        rel[n] = ((o - r).norm() / r.norm().clamp_min(1e-30)).item()
        if n.endswith("A_log"):
            pad_r, pad_o = r[num_heads:].abs().max().item(), o[num_heads:].abs().max().item()
            if r.numel() > num_heads and (pad_r != 0 or pad_o != 0):
                problems.append(f"{n}: padded entries {num_heads}..{r.numel() - 1} have grad (eager {pad_r}, aiter {pad_o})")
            d = (o[:num_heads] - r[:num_heads]).abs()
            idx = d.topk(min(top, num_heads)).indices.tolist()
            diag[n] = [(i, r[i].item(), o[i].item()) for i in idx]
    rel = dict(sorted(rel.items(), key=lambda kv: -kv[1]))
    return rel, problems, diag


# bf16 grading against an fp32 reference: the torch backend computes in fp32 from bf16 inputs while
# the fused path feeds bf16 q/k/v to the chunk kernels, so aiter-bf16 may sit further from fp32 than
# eager-bf16 does. ROCm/aiter#8739 measures about 2% grad deviation at K=128 from that quantization.
BF16_VS_FP32_FACTOR = 4.0
BF16_VS_FP32_FLOOR = 3e-2


def grad_parity_vs_fp32(ref32, eager16, aiter16, num_heads, factor=BF16_VS_FP32_FACTOR, floor=BF16_VS_FP32_FLOOR):
    """Grade both bf16 paths against fp32: aiter's error must stay within ``max(factor * eager_err, floor)``.

    Returns ({name: (eager_err, aiter_err, limit)} sorted by aiter_err / limit, problems).
    """
    e_rel, problems, _ = grad_parity(ref32, eager16, num_heads)
    a_rel, a_problems, _ = grad_parity(ref32, aiter16, num_heads)
    rows = {n: (e_rel[n], a_rel[n], max(factor * e_rel[n], floor)) for n in ref32}
    rows = dict(sorted(rows.items(), key=lambda kv: -kv[1][1] / kv[1][2]))
    problems = problems + a_problems + [f"{n}: aiter {a:.3e} > limit {lim:.3e} (eager {e:.3e})"
                                        for n, (e, a, lim) in rows.items() if a > lim]  # fmt: skip
    return rows, problems


def megatron_continuation_lr(step, lr_max=1e-5, lr_min=1e-6, decay_iters=50):
    """LR at 0-indexed ``step`` of the #6872 continuation recipe: cosine 1e-5 -> 1e-6 over 50 iters, no warm-up."""
    import math

    t = min(step, decay_iters) / decay_iters
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))


# bf16 loss trajectories graded against an fp32 one, like the grads: |aiter - fp32| must stay within
# max(LOSS_VS_FP32_FACTOR * |eager_bf16 - fp32|, MEGATRON_LOSS_PARITY_TOL) per step. The factor reuses
# the grad bound (aiter measures 1.4-1.5x eager-bf16's grad error on gfx950); the floor is the #6872
# tolerance itself, the loss difference Megatron accepts between two implementations.
LOSS_VS_FP32_FACTOR = BF16_VS_FP32_FACTOR


def loss_curve_vs_fp32(fp32, eager16, aiter16, factor=LOSS_VS_FP32_FACTOR, floor=MEGATRON_LOSS_PARITY_TOL):
    """Per-step (|eager-fp32|, |aiter-fp32|, limit) and the steps that exceed the limit."""
    rows = []
    for f, e, a in zip(fp32, eager16, aiter16):
        de, da = abs(e - f), abs(a - f)
        rows.append((de, da, max(factor * de, floor)))
    bad = [i + 1 for i, (_, da, lim) in enumerate(rows) if da > lim]
    return rows, bad
