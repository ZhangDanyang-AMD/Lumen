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
``precision="mxfp8"`` runs the projections with TE-faithful MXFP8 numerics
(``lumen.ops.quantize.mxfp8_te_reference``: E8M0 exponent ``ceil(log2(amax/448))``, fprop
X/W rowwise, dgrad dY rowwise x W columnwise, wgrad dY/X columnwise, fp32 accumulation,
optional fp32 ``main_grad``). ``"mxfp8-lumen"`` uses Lumen's fast ``quantized_linear(
scaling_type="mxfp8", block_size=32)``, whose aiter quantizer rounds the scale the torchao
way (see that module). ``"bf16"`` uses ``F.linear``.
"""

import functools
import os

import torch
import torch.nn.functional as F

MEGATRON_KDA_BACKENDS = ("fla", "cudnn", "aiter", "torch", "torch-fp32")
# Recurrence used for differentiable reference calls when FLA's backward cannot compile:
# "torch" (default; FLA-precision PyTorch), "torch-fp32" (fp32 throughout) or "aiter" (diagnostic).
REFERENCE_FALLBACK_ENV = "LUMEN_KDA_REFERENCE_FALLBACK"
LAST_BACKEND = None  # recurrence the last megatron_kda_forward call used (printed by tests / diag)
PRECISIONS = ("bf16", "mxfp8", "mxfp8-lumen")
MXFP8_BLOCK = 32  # TE MXFP8BlockScaling: one E8M0 scale per 1x32 block


def linear_fn(precision):
    if precision == "bf16":
        return F.linear
    if precision == "mxfp8":
        # TE MXFP8BlockScaling numerics (scale rounding, operand axes); see mxfp8_te_reference
        from lumen.ops.quantize.mxfp8_te_reference import te_mxfp8_linear

        return te_mxfp8_linear
    if precision != "mxfp8-lumen":
        raise ValueError(f"unknown precision {precision!r}, expected {PRECISIONS}")
    from lumen.ops.quantize.linear import quantized_linear

    def mxfp8_linear(x, w):
        return quantized_linear(x, w, scaling_type="mxfp8", block_size=MXFP8_BLOCK)

    return mxfp8_linear


def aiter_recurrence_available():
    try:
        from aiter.ops.kda import chunk_kda  # noqa: F401

        return True
    except ImportError:
        return False


def _torch_chunk_kda(q, k, v, g, beta, A_log, dt_bias, safe_gate, lower_bound, cu_seqlens=None, mma_bf16=True, **_):
    """The KDA recurrence in plain PyTorch (aiter's differentiable ``chunk_kda_torch``) with the
    in-kernel gate of the FLA contract; used where FLA's backward does not compile."""
    from aiter.ops.kda_fused_layer import kda_gate

    B, T, H, K = q.shape
    mma_bf16 = mma_bf16 and v.dtype == torch.bfloat16  # FLA on fp32 inputs computes in fp32
    z = g.float() + dt_bias.float().view(H, K)
    gate = kda_gate(z, A_log.float(), lower_bound if safe_gate else None)
    bounds = [0, T] if cu_seqlens is None else [int(b) for b in cu_seqlens.tolist()]
    outs = [
        _chunk_recurrence(q[:, s:e].float(), k[:, s:e].float(), v[:, s:e].float(), gate[:, s:e],
                          beta[:, s:e].float(), K**-0.5, mma_bf16=mma_bf16)  # fmt: skip
        for s, e in zip(bounds[:-1], bounds[1:])
    ]
    return torch.cat(outs, 1).to(v.dtype), None


def _r16(t):
    return t.to(torch.bfloat16).float()


def _chunk_step_bf16_mma(S, q, k, v, g, beta):
    """One chunk of the gated delta rule with FLA ``chunk_kda``'s precision: gate cumsum, decays,
    the intra-chunk A matrices and the triangular solve in fp32; every MMA operand (the gated q/k,
    v, A_qk, the WY-transformed values u, and the state S) rounded to bf16; fp32 accumulation and
    an fp32 state. Same math as aiter's ``_chunk_step``."""
    C = q.shape[-2]
    G = g.cumsum(-2)
    strict = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
    incl = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    dG = (G.unsqueeze(-2) - G.unsqueeze(-3)).masked_fill(~incl[..., None], float("-inf"))
    decay = dG.exp()
    qb, kb, vb = _r16(q), _r16(k), _r16(v)
    A_kk = torch.einsum("bhik,bhjk,bhijk->bhij", kb, kb, decay) * strict
    A_qk = torch.einsum("bhik,bhjk,bhijk->bhij", qb, kb, decay)
    L = torch.eye(C, device=q.device, dtype=q.dtype) + beta.unsqueeze(-1) * A_kk
    eg = G.exp()
    Sb = _r16(S)
    rhs = beta.unsqueeze(-1) * (vb - _r16(kb * eg) @ Sb)
    u = torch.linalg.solve_triangular(L, rhs, upper=False, unitriangular=True)
    ub = _r16(u)
    o = _r16(qb * eg) @ Sb + _r16(A_qk) @ ub
    k_end = _r16(kb * (G[..., -1:, :] - G).exp())
    S = G[..., -1, :, None].exp() * S + k_end.transpose(-1, -2) @ ub
    return o, S


def _chunk_recurrence(q, k, v, g, beta, scale, chunk_size=64, mma_bf16=True):
    """Chunked gated delta rule without per-chunk checkpoints, so the graph is the same in the
    forward and in an outer checkpoint's recomputation. ``mma_bf16`` rounds the MMA operands like
    FLA; otherwise aiter's fp32 ``_chunk_step``."""
    from aiter.ops.kda_fused_layer import _chunk_step

    step = _chunk_step_bf16_mma if mma_bf16 else _chunk_step
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    pad = (-T) % C
    if pad:
        q, k, v, g = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q, k, v, g))
        beta = F.pad(beta, (0, 0, 0, pad))
    N = (T + pad) // C

    def chunks(t):  # [B, T, H, ...] -> [B, H, N, C, ...]
        return t.reshape(B, N, C, H, *t.shape[3:]).transpose(1, 3).transpose(2, 3)

    q, k, v, g, beta = map(chunks, (q, k, v, g, beta))
    S = q.new_zeros(B, H, K, V)
    out = []
    for n in range(N):
        o, S = step(S, q[:, :, n], k[:, :, n], v[:, :, n], g[:, :, n], beta[:, :, n])
        out.append(o)
    o = torch.stack(out, 2) * scale
    return o.permute(0, 2, 3, 1, 4).reshape(B, N * C, H, V)[:, :T]


@functools.lru_cache(maxsize=None)
def _warn_once(msg):
    import logging

    logging.getLogger(__name__).warning(msg)


@functools.lru_cache(maxsize=1)
def fla_backward_status():
    """``"ok"``, ``"ok-nogather"`` (FLA's ``IS_GATHER_SUPPORTED`` fallback needed), or the error
    string when FLA ``chunk_kda``'s backward does not run on this GPU (e.g. triton 3.6 on ROCm:
    ``chunk_kda_bwd_intra`` fails with "PassManager::run failed")."""
    if not (fla_available() and torch.cuda.is_available()):
        return "unavailable"
    from fla.ops.kda import chunk_kda

    def probe():
        T, H, K = 64, 2, 128
        q = F.normalize(torch.randn(1, T, H, K, device="cuda"), dim=-1).bfloat16().requires_grad_(True)
        k = F.normalize(torch.randn(1, T, H, K, device="cuda"), dim=-1).bfloat16().requires_grad_(True)
        v = torch.randn(1, T, H, K, device="cuda").bfloat16().requires_grad_(True)
        g = torch.randn(1, T, H, K, device="cuda").bfloat16().requires_grad_(True)
        beta = torch.rand(1, T, H, device="cuda")
        o, _ = chunk_kda(q=q, k=k, v=v, g=g, beta=beta, A_log=torch.zeros(H, device="cuda"),
                         dt_bias=torch.zeros(H * K, device="cuda"), use_gate_in_kernel=True, safe_gate=True,
                         lower_bound=-5.0)  # fmt: skip
        o.float().sum().backward()
        torch.cuda.synchronize()

    try:
        probe()
        return "ok"
    except Exception as first:  # noqa: BLE001 — reported to the caller
        try:
            import fla.ops.kda.chunk_intra as ci

            ci.IS_GATHER_SUPPORTED = False  # use FLA's tl.load path instead of tl.gather
            probe()
            return "ok-nogather"
        except Exception:  # noqa: BLE001
            return f"{type(first).__name__}: {str(first).splitlines()[0][:200]}"


def fla_available():
    try:
        from fla.ops.kda import chunk_kda  # noqa: F401

        return True
    except ImportError:
        return False


def _conv_silu(x, w, cu_seqlens=None):
    # Megatron's deterministic / no-causal_conv1d path: F.conv1d with left padding, then SiLU; with
    # packed sequences the causal window restarts at every cu_seqlens boundary (as causal_conv1d does).
    if cu_seqlens is not None:
        b = [int(v) for v in cu_seqlens.tolist()]
        return torch.cat([_conv_silu(x[:, s:e], w) for s, e in zip(b[:-1], b[1:])], 1)
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
    """y [B, T, D] for x [B, T, D] and ``KDALayerWeights`` w.

    ``recompute=True`` mirrors ``recompute_gdn``: only the KDA input is kept for backward and the
    layer reruns in backward. The backend is resolved here, before the checkpointed region, so the
    forward and its recomputation run the same ops and save the same tensors.
    """
    if backend == "fla" and torch.is_grad_enabled() and not fla_backward_status().startswith("ok"):
        # FLA's chunk_kda backward does not compile here. Keep the Megatron layer and swap only the
        # recurrence, by default for an independent PyTorch one with FLA's precision (bf16 MMA
        # operands, fp32 state and accumulation, chunk 64). aiter#8755's chunk_kda (Megatron
        # kda_backend="aiter") is an opt-in diagnostic only: it would make the reference our own kernel.
        backend = os.environ.get(REFERENCE_FALLBACK_ENV, "torch")
        if backend == "aiter" and not aiter_recurrence_available():
            backend = "torch"
        _warn_once(f"Megatron KDA reference: FLA chunk_kda backward unavailable ({fla_backward_status()}); "
                   f"using the {backend} recurrence for differentiable calls")  # fmt: skip
    global LAST_BACKEND
    LAST_BACKEND = backend
    if recompute and torch.is_grad_enabled():
        return torch.utils.checkpoint.checkpoint(
            _megatron_kda_forward, x, w, num_heads, lower_bound, eps, backend, cu_seqlens, precision,
            use_reentrant=False,
        )  # fmt: skip
    return _megatron_kda_forward(x, w, num_heads, lower_bound, eps, backend, cu_seqlens, precision)


def _megatron_kda_forward(x, w, num_heads, lower_bound, eps, backend, cu_seqlens, precision):
    if backend == "cudnn":
        raise NotImplementedError("cuDNN KDA needs CUDA SM100+; use backend='fla' (same recurrence) on ROCm")
    if backend not in MEGATRON_KDA_BACKENDS:
        raise ValueError(f"unknown Megatron KDA backend {backend!r}, expected {MEGATRON_KDA_BACKENDS}")
    if backend == "aiter":
        from aiter.ops.kda import chunk_kda  # aiter#8755, FLA contract (Megatron kda_backend="aiter")
    elif backend == "torch":
        chunk_kda = functools.partial(_torch_chunk_kda, mma_bf16=True)
    elif backend == "torch-fp32":
        chunk_kda = functools.partial(_torch_chunk_kda, mma_bf16=False)
    else:
        from fla.ops.kda import chunk_kda

    B, T, _ = x.shape
    H = num_heads
    dt = x.dtype
    lin = linear_fn(precision)
    q = _l2norm(_conv_silu(lin(x, w.w_q.to(dt)), w.conv_q.to(dt), cu_seqlens).reshape(B, T, H, -1))
    k = _l2norm(_conv_silu(lin(x, w.w_k.to(dt)), w.conv_k.to(dt), cu_seqlens).reshape(B, T, H, -1))
    v = _conv_silu(lin(x, w.w_v.to(dt)), w.conv_v.to(dt), cu_seqlens).reshape(B, T, H, -1)
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


def patch_kda_reference(model, precision="bf16", backend="fla"):
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
                backend=backend,
            )

        mod.forward = types.MethodType(forward, mod)
        n += 1
    return n
