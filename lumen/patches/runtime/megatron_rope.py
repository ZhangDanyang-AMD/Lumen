###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Autograd fused RoPE for Megatron's ``apply_rope_fusion`` path.

``install()`` replaces ``rope_utils.fused_apply_rotary_pos_emb`` and
``enable_on_config(model)`` turns ``apply_rope_fusion`` on for a built model, so
the flag cannot be lost when Megatron builds the config before the provider runs.

``LUMEN_ROPE_IMPL`` selects the kernel:

* ``aiter`` (default): AITER Triton ``rope_cached`` forward/backward kernels on
  SBHD ``[S, B, H, D]`` or BHSD ``[B, H, S, D]`` input, with the fp32 cos/sin
  tables built by :func:`_cos_sin_4d`.
* ``apex``: apex's native ``fused_rotary_positional_embedding`` kernel on SBHD
  input. It needs ``USE_ROCM_AITER_ROPE_BACKEND=0`` in the environment before
  apex is first imported; ``install()`` raises if apex picked its AITER backend
  instead.

The apex kernels are slower per call than ``aiter`` but their host path is
shorter, so the faster choice depends on how host-bound the step is.

Interleaved (GPT-J) rotation, ``mscale != 1`` and unrecognised layouts run
Megatron's unfused implementation.

Env switches: ``LUMEN_FUSED_ROPE=0`` leaves RoPE unfused,
``ROPE_FUSION_DEBUG=1`` prints the call/fallback counters at exit.
"""

from __future__ import annotations

import atexit
import os

import torch

ROPE_IMPLS = ("aiter", "apex")

_state = {
    "installed": False,
    "impl": None,
    "refused": 0,
    "fallbacks": 0,
    "calls": 0,
    "bhsd": 0,
    "sbhd": 0,
    "cos_sin_computed": 0,
}
_UNFUSED = {"fn": None}
_APEX = {"fn": None}


def rope_impl() -> str:
    """Return the ``LUMEN_ROPE_IMPL`` selection, ``aiter`` when unset."""
    value = os.environ.get("LUMEN_ROPE_IMPL", "").strip().lower() or "aiter"
    if value not in ROPE_IMPLS:
        raise ValueError(f"LUMEN_ROPE_IMPL={value!r} is not one of {ROPE_IMPLS}")
    return value


def _load_apex_rope():
    import apex.transformer.functional.fused_rope as apex_rope

    if getattr(apex_rope, "AITER_ROPE_BACKEND", False):
        raise RuntimeError(
            "LUMEN_ROPE_IMPL=apex but apex selected its AITER RoPE backend; "
            "export USE_ROCM_AITER_ROPE_BACKEND=0 before apex is imported"
        )
    return apex_rope.fused_apply_rotary_pos_emb


def _report_at_exit():
    if os.environ.get("ROPE_FUSION_DEBUG", "0") != "1":
        return
    print(
        "> rope_fusion report: impl=%s calls=%d bhsd=%d sbhd=%d fallbacks=%d refused=%d "
        "cos_sin_computed=%d"
        % (
            _state["impl"],
            _state["calls"],
            _state["bhsd"],
            _state["sbhd"],
            _state["fallbacks"],
            _state["refused"],
            _state["cos_sin_computed"],
        )
    )


def _fallback(t, freqs, interleaved=False, mscale=1.0, count_layout=True):
    if _UNFUSED["fn"] is None:
        raise RuntimeError("megatron_rope: unfused fallback is not installed")
    if count_layout:
        _state["fallbacks"] += 1
    return _UNFUSED["fn"](t, freqs, rotary_interleaved=interleaved, mscale=mscale)


def _cos_sin_4d(freqs, rot_dim):
    """Return fp32 ``(cos4, sin4)`` of shape ``[S, 1, 1, rot_dim]``, or ``(None, None)``.

    Module-level so a per-step cache can wrap it by attribute assignment.
    """
    if freqs.dim() != 4 or freqs.shape[-1] != rot_dim:
        return None, None
    # Megatron passes [S,1,1,D] (SBHD) or [1,1,S,D] (BHSD); both flatten to [S,1,1,D].
    cos = torch.cos(freqs).reshape(-1, 1, 1, rot_dim)
    sin = torch.sin(freqs).reshape(-1, 1, 1, rot_dim)
    _state["cos_sin_computed"] += 1
    return cos, sin


def _rope_apply(x, cos4, sin4, layout, backward=False):
    from lumen.ops.rope import (
        NEOX_STYLE,
        _get_aiter_rope_cached_bwd_lowlevel,
        _get_aiter_rope_cached_fwd_lowlevel,
    )

    low_level_fn = (
        _get_aiter_rope_cached_bwd_lowlevel() if backward else _get_aiter_rope_cached_fwd_lowlevel()
    )
    if layout == "sbhd":
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        low_level_fn(x, out, cos4, sin4, None, None, NEOX_STYLE, False, False, False, False)
        return out

    # The kernel is stride-aware: feed [S,B,H,D] views of the BHSD input and output.
    b, h, s, d = x.shape
    out = torch.empty((b, h, s, d), dtype=x.dtype, device=x.device)
    low_level_fn(
        x.permute(2, 0, 1, 3),
        out.permute(2, 0, 1, 3),
        cos4,
        sin4,
        None,
        None,
        NEOX_STYLE,
        False,
        False,
        False,
        False,
    )
    return out


class _FusedRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos4, sin4, layout):
        out = _rope_apply(x, cos4, sin4, layout, backward=False)
        ctx.save_for_backward(cos4, sin4)
        ctx.layout = layout
        return out

    @staticmethod
    def backward(ctx, grad_out):
        cos4, sin4 = ctx.saved_tensors
        return _rope_apply(grad_out, cos4, sin4, ctx.layout, backward=True), None, None, None


def _apex_rope(t, freqs, interleaved, mscale):
    if (
        interleaved
        or mscale != 1.0
        or t.dim() != 4
        or freqs.dim() != 4
        or freqs.shape[0] != t.shape[0]
    ):
        return _fallback(t, freqs, interleaved, mscale)
    _state["calls"] += 1
    _state["sbhd"] += 1
    return _APEX["fn"](t, freqs)


def fused_apply_rotary_pos_emb(t, freqs, interleaved=False, mscale=1.0, **kwargs):
    """Autograd-capable drop-in for ``rope_utils.fused_apply_rotary_pos_emb``."""
    if _APEX["fn"] is not None:
        return _apex_rope(t, freqs, interleaved, mscale)
    if interleaved or mscale != 1.0:
        _state["refused"] += 1
        return _fallback(t, freqs, interleaved, mscale, count_layout=False)

    rot_dim = freqs.shape[-1]
    core = t[..., :rot_dim]
    tail = t[..., rot_dim:]
    if core.dim() != 4:
        return _fallback(t, freqs, interleaved, mscale)
    cos, sin = _cos_sin_4d(freqs, rot_dim)
    if cos is None:
        return _fallback(t, freqs, interleaved, mscale)
    seq_len = cos.shape[0]
    if core.shape[2] == seq_len:
        layout = "bhsd"
    elif core.shape[0] == seq_len:
        layout = "sbhd"
    else:
        return _fallback(t, freqs, interleaved, mscale)

    out = _FusedRoPE.apply(core, cos, sin, layout)
    _state["calls"] += 1
    _state[layout] += 1
    return torch.cat((out, tail), dim=-1) if tail.shape[-1] else out


def install(rope_utils=None) -> bool:
    """Patch ``rope_utils.fused_apply_rotary_pos_emb`` with the selected implementation."""
    if _state["installed"]:
        return True
    if os.environ.get("LUMEN_FUSED_ROPE", "1") == "0":
        print("> rope_fusion: LUMEN_FUSED_ROPE=0, RoPE left unfused")
        return False
    impl = rope_impl()
    if rope_utils is None:
        import megatron.core.models.common.embeddings.rope_utils as rope_utils

    _APEX["fn"] = _load_apex_rope() if impl == "apex" else None
    _UNFUSED["fn"] = rope_utils._apply_rotary_pos_emb_bshd
    rope_utils.fused_apply_rotary_pos_emb = fused_apply_rotary_pos_emb
    _state["installed"] = True
    _state["impl"] = impl
    atexit.register(_report_at_exit)
    if impl == "apex":
        print("> rope_fusion: installed native apex fused RoPE (LUMEN_ROPE_IMPL=apex)")
    else:
        print("> rope_fusion: installed autograd fused RoPE (Lumen AITER rope_cached, BHSD+SBHD)")
    return True


def enable_on_config(model) -> bool:
    """Set ``apply_rope_fusion=True`` on a built model's config once ``install()`` ran."""
    if os.environ.get("LUMEN_FUSED_ROPE", "1") == "0" or not _state["installed"]:
        return False
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "apply_rope_fusion"):
        print("> rope_fusion: WARNING model.config has no apply_rope_fusion; left unfused")
        return False
    config.apply_rope_fusion = True
    print("> rope_fusion: enabled apply_rope_fusion=True on model.config")
    return True


def report() -> dict:
    return dict(_state)
