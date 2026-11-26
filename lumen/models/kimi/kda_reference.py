###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""KDA layer as Megatron-Core runs it for the NVIDIA/Megatron-LM#6872 Kimi-K3 recipes.

Megatron-Core ``KimiDeltaAttention`` (``megatron/core/ssm/gated_delta_net/kda.py`` at
the ``kimi_k3_preview_20261126`` commit ``6e4ebe2``) selects the recurrence with
``kda_backend`` (``megatron/core/ops/ssm/kda/backends.py``):

* ``"fla"`` (Megatron's default, and the backend of the historical FLA-KDA runs):
  ``fla.ops.kda.chunk_kda``;
* ``"cudnn"`` (the 2026-11-26 recipes): cuDNN ``kimi_delta_attention``, SM100+ only.

Either way the preprocessing and output norm are shared: depthwise causal conv1d
(no bias) + SiLU on q/k/v, raw decay ``g = f_b(f_a(x))`` passed to the kernel with
``A_log`` / ``dt_bias`` (``use_gate_in_kernel``), ``beta = sigmoid(b_proj(x))`` in
fp32, ``kda_safe_gate=True`` / ``kda_lower_bound=-5``, then the gated
RMSNorm of Megatron's ``fused_gated_norm`` (eps ``norm_epsilon`` = 1e-5) and ``out_proj``.
Op-by-op precision follows ``docs/megatron-kda-precision-map.md`` (F1-F14): conv and
out-norm weights are used in the activation dtype (bf16 params in Megatron), ``A_log``
and ``dt_bias`` in fp32. The recipes set ``gdn_pre_gated_delta_rule_fusion: true``, which
L2-normalizes q/k before the kernel (``use_qk_l2norm_in_kernel=False``); the
reference does the same with FLA's ``l2norm`` (eps 1e-6).

``megatron_kda_forward`` runs exactly that with ``chunk_kda`` (Triton, runs on
ROCm). The cuDNN recurrence is CUDA/SM100-only and has no ROCm build; it computes
the same recurrence, so the FLA backend stands in for it here.

Precision: the #6872 recipes run every linear in MXFP8 (TE ``MXFP8BlockScaling``:
1x32 blocks, E8M0 scales, fprop/dgrad/wgrad) and the recurrence in bf16.
``precision="mxfp8"`` routes the five projections and ``out_proj`` through Lumen's
``quantized_linear(scaling_type="mxfp8", block_size=32)``; ``"bf16"`` uses ``F.linear``.
"""

import torch
import torch.nn.functional as F

MEGATRON_KDA_BACKENDS = ("fla", "cudnn")
PRECISIONS = ("bf16", "mxfp8")
MXFP8_BLOCK = 32  # TE MXFP8BlockScaling: one E8M0 scale per 1x32 block


def linear_fn(precision):
    if precision == "bf16":
        return F.linear
    if precision != "mxfp8":
        raise ValueError(f"unknown precision {precision!r}, expected {PRECISIONS}")
    from lumen.ops.quantize.linear import quantized_linear

    def mxfp8_linear(x, w):
        return quantized_linear(x, w, scaling_type="mxfp8", block_size=MXFP8_BLOCK)

    return mxfp8_linear


def fla_available():
    try:
        from fla.ops.kda import chunk_kda  # noqa: F401

        return True
    except ImportError:
        return False


def _conv_silu(x, w):
    # Megatron's deterministic / no-causal_conv1d path: F.conv1d with left padding, then SiLU.
    T = x.shape[1]
    y = F.conv1d(x.transpose(1, 2), w.unsqueeze(1), padding=w.shape[1] - 1, groups=w.shape[0])
    return F.silu(y[..., :T]).transpose(1, 2)


L2NORM_EPS = 1e-6  # FLA l2norm default, used by Megatron's pre-GDR fusion


def _l2norm(x):
    try:
        from fla.modules.l2norm import l2norm

        return l2norm(x, eps=L2NORM_EPS)
    except ImportError:
        return F.normalize(x.float(), dim=-1, eps=L2NORM_EPS).to(x.dtype)


def _rms_norm_gated(o, gate, weight, eps):
    """Megatron ``fused_gated_norm``: fp32 ``x * rsqrt(mean(x^2) + eps) * w``, rounded to the activation
    dtype, then multiplied by ``sigmoid(gate)`` in fp32 and rounded again (precision map F13)."""
    of = o.float()
    normed = (of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(o.dtype)
    return (normed.float() * torch.sigmoid(gate.float())).to(o.dtype)


def megatron_kda_forward(
    x, w, num_heads, lower_bound=-5.0, eps=1e-5, backend="fla", cu_seqlens=None, precision="bf16", recompute=True
):
    """``recompute=True`` mirrors ``recompute_gdn``: only the KDA input is kept for backward."""
    if recompute and torch.is_grad_enabled():
        return torch.utils.checkpoint.checkpoint(
            _megatron_kda_forward, x, w, num_heads, lower_bound, eps, backend, cu_seqlens, precision,
            use_reentrant=False,
        )  # fmt: skip
    return _megatron_kda_forward(x, w, num_heads, lower_bound, eps, backend, cu_seqlens, precision)


def _megatron_kda_forward(x, w, num_heads, lower_bound, eps, backend, cu_seqlens, precision):
    """y [B, T, D] for x [B, T, D] and ``KDALayerWeights`` w, differentiable through FLA's autograd."""
    if backend == "cudnn":
        raise NotImplementedError("cuDNN KDA needs CUDA SM100+; use backend='fla' (same recurrence) on ROCm")
    if backend not in MEGATRON_KDA_BACKENDS:
        raise ValueError(f"unknown Megatron KDA backend {backend!r}, expected {MEGATRON_KDA_BACKENDS}")
    from fla.ops.kda import chunk_kda

    B, T, _ = x.shape
    H = num_heads
    dt = x.dtype
    lin = linear_fn(precision)
    q = _l2norm(_conv_silu(lin(x, w.w_q.to(dt)), w.conv_q.to(dt)).reshape(B, T, H, -1))
    k = _l2norm(_conv_silu(lin(x, w.w_k.to(dt)), w.conv_k.to(dt)).reshape(B, T, H, -1))
    v = _conv_silu(lin(x, w.w_v.to(dt)), w.conv_v.to(dt)).reshape(B, T, H, -1)
    raw_g = lin(lin(x, w.f_a.to(dt)), w.f_b.to(dt)).reshape(B, T, H, -1)
    beta = lin(x, w.w_beta.to(dt)).float().sigmoid()
    o, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=raw_g,
        beta=beta,
        A_log=w.A_log[:H],
        dt_bias=w.dt_bias,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=True,
        safe_gate=lower_bound is not None,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
    )
    gate = lin(x, w.g_proj.to(dt)).reshape(B, T, H, -1)
    o = _rms_norm_gated(o.to(dt), gate, w.norm_w.to(dt), eps).reshape(B, T, -1)
    return lin(o.to(dt), w.w_o.to(dt))


def patch_kda_reference(model, precision="bf16"):
    """Run every KDA module's unpatched forward as ``megatron_kda_forward`` (for remote-code models).

    Returns the count patched. ``lumen.models.kimi.kda_aiter.patch_kda_layers`` may be applied on top
    and restores to this forward on ``unpatch_kda_layers``.
    """
    import types

    from lumen.models.kimi.kda_aiter import _eps, _gate_lower_bound, _num_heads, is_kda_module, kda_layer_weights

    n = 0
    for mod in model.modules():
        if not is_kda_module(mod):
            continue

        def forward(self, hidden_states, attention_mask=None, cache_params=None, **kwargs):
            if cache_params is not None or (attention_mask is not None and attention_mask.dim() == 2
                                            and not bool(attention_mask.all())):  # fmt: skip
                raise NotImplementedError("Megatron KDA reference: no decoding cache or padded masks")
            return megatron_kda_forward(
                hidden_states,
                kda_layer_weights(self),
                _num_heads(self),
                lower_bound=_gate_lower_bound(self),
                eps=_eps(self),
                cu_seqlens=kwargs.get("cu_seqlens"),
                precision=precision,
            )

        mod.forward = types.MethodType(forward, mod)
        n += 1
    return n
