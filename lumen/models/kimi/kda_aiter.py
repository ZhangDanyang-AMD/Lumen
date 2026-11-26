###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Route Kimi Delta Attention (KDA) layers through ``aiter.ops.kda_fused_layer``.

``patch_kda_layers(model)`` swaps the forward of every KDA attention module
(``KimiDeltaAttention`` in the ``Qinghao/Kimi-K3-Tiny-L13`` remote code
``modeling_kimi_linear.py``: ``forward(hidden_states, attention_mask=None,
cache_params=None, **kwargs) -> o``; parameter names ``q_proj``, ``q_conv1d``, ``f_a_proj``,
``f_b_proj``, ``A_log``, ``dt_bias``, ``b_proj``, ``g_proj``, ``o_norm``,
``o_proj``) for ``kda_layer_fwd``, whose autograd covers the whole layer
backward. The module parameters are passed as views, so gradients land on the
original parameters and no weight is copied.

Enable with ``LUMEN_KDA_AITER=1`` (or ``patch_kda_layers(model)`` directly).
``LUMEN_KDA_AITER_BACKEND`` picks the aiter layer backend (``torch`` or
``fused``, default ``fused``); ``LUMEN_KDA_AITER_GEMM_PRECISION=mxfp8`` passes
``gemm_precision="mxfp8"`` (needs the MXFP8 aiter PR stacked on #8739);
``LUMEN_KDA_AITER_PRECISION=megatron`` (or ``kda_precision=``) selects aiter's Megatron
rounding mode (#8739 95a82cd); per-piece kernels inside the fused backend are
chosen by aiter's ``AITER_KDA_PIECES`` / ``set_kda_pieces``.

``A_log`` is per head and may be padded past ``num_heads`` (K3 checkpoints
store [128] for 96 heads; aiter reads ``[:H]``). A ``gate_lower_bound`` from the
module or its ``linear_attn_config`` (K3: -5) selects aiter's safe gate
``lower_bound * sigmoid(exp(A_log) * z)``. Calls the aiter layer cannot express
(``cache_params`` decoding, padded attention masks, the low-rank
``g_a_proj``/``g_b_proj`` gate) run the original forward.
"""

import inspect
import logging
import os
import types

import torch

from lumen.ops.dispatch import _probe_aiter_kda_fused_layer

logger = logging.getLogger(__name__)

KDA_AITER_ENV = "LUMEN_KDA_AITER"
KDA_AITER_BACKEND_ENV = "LUMEN_KDA_AITER_BACKEND"
KDA_AITER_GEMM_PRECISION_ENV = "LUMEN_KDA_AITER_GEMM_PRECISION"
KDA_AITER_PRECISION_ENV = "LUMEN_KDA_AITER_PRECISION"


def set_aiter_kda_precision(mode):
    """aiter's KDA rounding mode (``"default"`` or ``"megatron"``, #8739 95a82cd+); ``None`` resets it.

    ``"megatron"`` rounds like Megatron-LM#6872: bf16 projection outputs, conv rounded to bf16
    before SiLU, bf16 q/k after the L2 norm, bf16 o, gated norm rounded before the gate,
    bf16 conv/norm weights, fp32 A_log/dt_bias. The mode is process-global in aiter, so it
    covers the backward pass too.
    """
    from aiter.ops import kda_fused_layer as kfl

    if not hasattr(kfl, "set_kda_precision"):
        if mode in (None, "default"):
            return
        raise ImportError(f"aiter kda_fused_layer has no set_kda_precision; {mode!r} needs #8739 at 95a82cd or later")
    kfl.set_kda_precision(mode)
_ORIG_FORWARD = "_lumen_kda_orig_forward"
_KDA_ATTRS = ("q_proj", "k_proj", "v_proj", "q_conv1d", "f_a_proj", "f_b_proj", "A_log", "o_norm", "o_proj")


def kda_aiter_enabled() -> bool:
    return os.environ.get(KDA_AITER_ENV, "0").lower() in ("1", "true", "yes", "on")


def is_kda_module(mod: torch.nn.Module) -> bool:
    return type(mod).__name__ == "KimiDeltaAttention" or all(hasattr(mod, a) for a in _KDA_ATTRS)


def _num_heads(mod):
    return int(getattr(mod, "num_heads", None) or mod.A_log.numel())


def _eps(mod):
    for name in ("eps", "variance_epsilon"):
        if hasattr(mod.o_norm, name):
            return float(getattr(mod.o_norm, name))
    return 1e-6


def _gate_lower_bound(mod):
    if hasattr(mod, "gate_lower_bound"):
        return mod.gate_lower_bound
    cfg = getattr(mod, "config", None)
    la = getattr(cfg, "linear_attn_config", None) or {}
    return la.get("gate_lower_bound") if isinstance(la, dict) else getattr(la, "gate_lower_bound", None)


def _flat_weight(layer, rows, name):
    w = getattr(layer, "weight", None)
    if w is None:
        raise ValueError(f"{name} has no single weight (low-rank or composite projection is not supported)")
    if w.shape[0] != rows:
        raise ValueError(f"{name}.weight has {w.shape[0]} rows, expected {rows}")
    return w.view(rows, -1)


def kda_layer_weights(mod: torch.nn.Module):
    """``KDALayerWeights`` of views onto ``mod``'s parameters (no copies)."""
    from aiter.ops.kda_fused_layer import KDALayerWeights

    H = _num_heads(mod)
    hk = mod.q_proj.weight.shape[0]
    hv = mod.v_proj.weight.shape[0]
    a_log = mod.A_log.view(-1)
    if a_log.numel() < H:
        raise ValueError(f"A_log has {a_log.numel()} values, expected at least one per head ({H})")
    return KDALayerWeights(
        w_q=mod.q_proj.weight,
        w_k=mod.k_proj.weight,
        w_v=mod.v_proj.weight,
        conv_q=_flat_weight(mod.q_conv1d, hk, "q_conv1d"),
        conv_k=_flat_weight(mod.k_conv1d, hk, "k_conv1d"),
        conv_v=_flat_weight(mod.v_conv1d, hv, "v_conv1d"),
        f_a=mod.f_a_proj.weight,
        f_b=mod.f_b_proj.weight,
        A_log=a_log,
        dt_bias=mod.dt_bias.view(-1),
        w_beta=mod.b_proj.weight,
        g_proj=_flat_weight(mod.g_proj, hv, "g_proj"),
        norm_w=mod.o_norm.weight,
        w_o=mod.o_proj.weight,
    )


def _unsupported_reason(mod, attention_mask, cache_params, kwargs):
    if cache_params is not None:
        return "cache_params (decoding)"
    if not hasattr(mod, "g_proj"):
        return "low-rank output gate (use_full_rank_gate=False)"
    if attention_mask is not None and attention_mask.dim() != 2:
        attention_mask = kwargs.get("padding_mask")
    if attention_mask is not None and not bool(attention_mask.all()):
        return "padded attention_mask (pack with cu_seqlens instead)"
    return None


def aiter_supports_gemm_precision():
    from aiter.ops.kda_fused_layer import kda_layer_fwd

    return "gemm_precision" in inspect.signature(kda_layer_fwd).parameters


def _make_forward(mod, backend, chunk_size, gemm_precision="bf16"):
    from aiter.ops.kda_fused_layer import kda_layer_fwd

    orig = getattr(mod, _ORIG_FORWARD)
    if "lower_bound" not in inspect.signature(kda_layer_fwd).parameters:
        raise ImportError("aiter kda_layer_fwd lacks lower_bound; need ROCm/aiter#8739 at bbed79a or later")
    extra = {}
    gemm_precision = "mxfp8" if gemm_precision.startswith("mxfp8") else gemm_precision
    # TE semantics for aiter's weight caches (#8739): None = no cross-call reuse, True = rebuild,
    # False = reuse. Lumen never relies on cross-call reuse unless the caller passes the flag.
    takes_microbatch = "is_first_microbatch" in inspect.signature(kda_layer_fwd).parameters
    if gemm_precision != "bf16":
        if not aiter_supports_gemm_precision():
            raise ImportError(
                f"aiter kda_layer_fwd has no gemm_precision; {gemm_precision} needs the MXFP8 PR stacked on #8739"
            )
        extra["gemm_precision"] = gemm_precision
    warned = set()

    def forward(self, hidden_states, attention_mask=None, cache_params=None, **kwargs):
        reason = _unsupported_reason(self, attention_mask, cache_params, kwargs)
        if reason is not None:
            # The aiter layer cannot express this call; keep the model's own KDA forward.
            if reason not in warned:
                warned.add(reason)
                logger.warning("KDA aiter path: %s, falling back to the module forward", reason)
            return orig(hidden_states, attention_mask=attention_mask, cache_params=cache_params, **kwargs)
        # Megatron passes is_first_microbatch to TE layers per microbatch; HF-style callers set it
        # on the module (set_kda_is_first_microbatch) since the decoder does not forward it.
        flag = kwargs.get("is_first_microbatch", getattr(self, "_lumen_is_first_microbatch", None))
        mb = {"is_first_microbatch": flag} if takes_microbatch else {}
        return kda_layer_fwd(
            hidden_states,
            kda_layer_weights(self),
            _num_heads(self),
            chunk_size=chunk_size,
            eps=_eps(self),
            backend=backend,
            cu_seqlens=kwargs.get("cu_seqlens"),
            lower_bound=_gate_lower_bound(self),
            **extra,
            **mb,
        )

    return types.MethodType(forward, mod)


def kda_layer_ids(model: torch.nn.Module):
    """0-indexed KDA decoder layers from ``linear_attn_config.kda_layers`` (1-indexed in K3 / K3.1), else None."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    if hasattr(cfg, "get_text_config"):
        cfg = cfg.get_text_config()
    la = getattr(cfg, "linear_attn_config", None)
    if not isinstance(la, dict) or "kda_layers" not in la:
        return None
    return frozenset(i - 1 for i in la["kda_layers"])


def patch_kda_layers(
    model: torch.nn.Module,
    backend: str | None = None,
    chunk_size: int = 64,
    gemm_precision: str | None = None,
    kda_precision: str | None = None,
) -> int:
    """Replace every KDA layer forward in ``model`` with aiter's; returns the count patched.

    When the model config lists ``kda_layers``, only modules whose ``layer_idx`` is in it are patched.
    """
    if not _probe_aiter_kda_fused_layer():
        raise ImportError(
            "aiter.ops.kda_fused_layer is not available; install aiter with ROCm/aiter#8739 "
            "(branch cursor/kda-fused-layer-bwd-8350)"
        )
    backend = backend or os.environ.get(KDA_AITER_BACKEND_ENV, "fused")
    gemm_precision = gemm_precision or os.environ.get(KDA_AITER_GEMM_PRECISION_ENV, "bf16")
    kda_precision = kda_precision or os.environ.get(KDA_AITER_PRECISION_ENV)
    if kda_precision is not None:
        set_aiter_kda_precision(kda_precision)
    layer_ids = kda_layer_ids(model)
    n = 0
    for name, mod in model.named_modules():
        if not is_kda_module(mod) or hasattr(mod, _ORIG_FORWARD):
            continue
        if layer_ids is not None and getattr(mod, "layer_idx", None) not in layer_ids:
            logger.warning("KDA aiter path: %s (layer_idx=%s) is not in config kda_layers, left unpatched",
                           name, getattr(mod, "layer_idx", None))  # fmt: skip
            continue
        setattr(mod, _ORIG_FORWARD, mod.forward)
        mod.forward = _make_forward(mod, backend, chunk_size, gemm_precision)
        n += 1
        logger.debug("KDA aiter path: patched %s (backend=%s)", name, backend)
    logger.info("KDA aiter path: patched %d KDA layer(s), backend=%s", n, backend)
    return n


def invalidate_aiter_kda_weight_caches():
    """Drop aiter's KDA weight caches (``invalidate_weight_caches``, #8739) if this aiter has them.

    Megatron-style updates (``param.data`` copies, distributed-optimizer buffers) do not bump
    ``_version``, so call this after every ``optimizer.step()``. Returns whether aiter had caches.
    """
    if not _probe_aiter_kda_fused_layer():
        return False
    from aiter.ops import kda_fused_layer as kfl

    fn = getattr(kfl, "invalidate_weight_caches", None)
    if fn is None:
        return False
    fn()
    return True


def set_kda_is_first_microbatch(model: torch.nn.Module, flag):
    """TE ``is_first_microbatch`` for every patched KDA layer: ``True`` on the first microbatch of an
    optimizer step (rebuild aiter's weight caches), ``False`` on the following ones (reuse), ``None``
    for no cross-call reuse (the default)."""
    n = 0
    for mod in model.modules():
        if hasattr(mod, _ORIG_FORWARD):
            mod._lumen_is_first_microbatch = flag
            n += 1
    return n


def register_kda_optimizer_hooks(optimizer):
    """Call ``invalidate_aiter_kda_weight_caches()`` after every ``optimizer.step()``.

    Belt and braces next to ``is_first_microbatch``: Megatron-style updates (``param.data`` copies,
    distributed-optimizer buffers) do not bump ``_version``. Returns whether a hook was installed.
    """
    if hasattr(optimizer, "register_step_post_hook"):
        optimizer.register_step_post_hook(lambda _opt, _a, _k: invalidate_aiter_kda_weight_caches())
        return True
    step = getattr(optimizer, "step", None)
    if step is None:
        return False

    def _step(*args, **kwargs):
        out = step(*args, **kwargs)
        invalidate_aiter_kda_weight_caches()
        return out

    optimizer.step = _step  # Megatron optimizers are not torch.optim.Optimizer subclasses
    return True


def unpatch_kda_layers(model: torch.nn.Module) -> int:
    n = 0
    if any(hasattr(m, _ORIG_FORWARD) for m in model.modules()) and _probe_aiter_kda_fused_layer():
        set_aiter_kda_precision(None)
    for mod in model.modules():
        if hasattr(mod, _ORIG_FORWARD):
            mod.forward = getattr(mod, _ORIG_FORWARD)
            delattr(mod, _ORIG_FORWARD)
            n += 1
    return n


def maybe_patch_kda_layers(
    model: torch.nn.Module, gemm_precision: str | None = None, kda_precision: str | None = None
) -> int:
    """``patch_kda_layers`` when ``LUMEN_KDA_AITER`` is set, else a no-op."""
    if not kda_aiter_enabled():
        return 0
    return patch_kda_layers(model, gemm_precision=gemm_precision, kda_precision=kda_precision)
