###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Route Kimi Delta Attention (KDA) layers through ``aiter.ops.kda_fused_layer``.

``patch_kda_layers(model)`` swaps the forward of every KDA attention module
(HF ``KimiDeltaAttention`` naming: ``q_proj``, ``q_conv1d``, ``f_a_proj``,
``f_b_proj``, ``A_log``, ``dt_bias``, ``b_proj``, ``g_proj``, ``o_norm``,
``o_proj``) for ``kda_layer_fwd``, whose autograd covers the whole layer
backward. The module parameters are passed as views, so gradients land on the
original parameters and no weight is copied.

Enable with ``LUMEN_KDA_AITER=1`` (or ``patch_kda_layers(model)`` directly).
``LUMEN_KDA_AITER_BACKEND`` picks the aiter layer backend (``torch`` or
``fused``, default ``fused``); per-piece kernels inside the fused backend are
chosen by aiter's ``AITER_KDA_PIECES`` / ``set_kda_pieces``.

Calls the aiter layer cannot express (KV/conv cache, padded attention masks,
a checkpoint ``A_log`` that is not one value per head, a low-rank ``g_proj``,
or ``gate_lower_bound`` while aiter lacks it) run the original forward.
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
    if a_log.numel() != H:
        # Published K3 checkpoints store A_log as [head_dim]; aiter wants one decay per head.
        raise ValueError(f"A_log has {a_log.numel()} values, expected one per head ({H})")
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


def _unsupported_reason(mod, attention_mask, past_key_values, use_cache, gate_kw):
    if use_cache or past_key_values is not None:
        return "KV/conv cache"
    if attention_mask is not None and not bool(attention_mask.all()):
        return "padded attention_mask (pack with cu_seqlens instead)"
    if _gate_lower_bound(mod) is not None and gate_kw is None:
        return "gate_lower_bound (safe_gate) is not in this aiter kda_layer_fwd yet"
    return None


def _make_forward(mod, backend, chunk_size):
    from aiter.ops.kda_fused_layer import kda_layer_fwd

    orig = getattr(mod, _ORIG_FORWARD)
    gate_kw = "gate_lower_bound" if "gate_lower_bound" in inspect.signature(kda_layer_fwd).parameters else None
    warned = set()

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        reason = _unsupported_reason(self, attention_mask, past_key_values, use_cache, gate_kw)
        weights = None
        if reason is None:
            try:
                weights = kda_layer_weights(self)
            except ValueError as e:
                reason = str(e)
        if reason is not None:
            # The aiter layer cannot express this call; keep the model's own KDA forward.
            if reason not in warned:
                warned.add(reason)
                logger.warning("KDA aiter path: %s, falling back to the module forward", reason)
            return orig(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        extra = {gate_kw: _gate_lower_bound(self)} if gate_kw and _gate_lower_bound(self) is not None else {}
        o = kda_layer_fwd(
            hidden_states,
            weights,
            _num_heads(self),
            chunk_size=chunk_size,
            eps=_eps(self),
            backend=backend,
            cu_seqlens=kwargs.get("cu_seqlens"),
            **extra,
        )
        return o, None, past_key_values

    return types.MethodType(forward, mod)


def patch_kda_layers(model: torch.nn.Module, backend: str | None = None, chunk_size: int = 64) -> int:
    """Replace every KDA layer forward in ``model`` with aiter's; returns the count patched."""
    if not _probe_aiter_kda_fused_layer():
        raise ImportError(
            "aiter.ops.kda_fused_layer is not available; install aiter with ROCm/aiter#8739 "
            "(branch cursor/kda-fused-layer-bwd-8350)"
        )
    backend = backend or os.environ.get(KDA_AITER_BACKEND_ENV, "fused")
    n = 0
    for name, mod in model.named_modules():
        if not is_kda_module(mod) or hasattr(mod, _ORIG_FORWARD):
            continue
        setattr(mod, _ORIG_FORWARD, mod.forward)
        mod.forward = _make_forward(mod, backend, chunk_size)
        n += 1
        logger.debug("KDA aiter path: patched %s (backend=%s)", name, backend)
    logger.info("KDA aiter path: patched %d KDA layer(s), backend=%s", n, backend)
    return n


def unpatch_kda_layers(model: torch.nn.Module) -> int:
    n = 0
    for mod in model.modules():
        if hasattr(mod, _ORIG_FORWARD):
            mod.forward = getattr(mod, _ORIG_FORWARD)
            delattr(mod, _ORIG_FORWARD)
            n += 1
    return n


def maybe_patch_kda_layers(model: torch.nn.Module) -> int:
    """``patch_kda_layers`` when ``LUMEN_KDA_AITER`` is set, else a no-op."""
    return patch_kda_layers(model) if kda_aiter_enabled() else 0
