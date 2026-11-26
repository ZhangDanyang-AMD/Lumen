###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""aiter as Megatron-Core's KDA recurrence backend (``kda_backend: aiter``).

Megatron-Core's ``KimiDeltaAttention`` (``megatron/core/ssm/gated_delta_net/kda.py``)
selects its recurrence with ``select_kda_recurrence(config.kda_backend)``. aiter#8755's
``aiter.ops.kda.patch_megatron_kda_backend`` registers ``"aiter"`` there; the recurrence
is ``aiter.ops.kda.chunk_kda`` (FLA ``chunk_kda`` contract, chunkwise CP via ``cp_context``).

The rest of the layer still comes from FLA through ``gated_delta_net.common``:

* ``build_cp_context`` (``fla.ops.cp``): ``None`` without FLA's CP extension, which makes
  chunkwise CP raise. Substituted by ``aiter.ops.kda.build_kda_cp_context`` (same
  contiguous partition, carries ``conv1d_kernel_size`` / ``pre_num_conv_tokens``).
* ``causal_conv1d`` (FLA Triton, with ``cp_context``) and ``rms_norm_gated`` (FLA Triton):
  :func:`check_fla_rocm` runs both against PyTorch on the current GPU. A failing
  ``rms_norm_gated`` is replaced by the Megatron fused-gated-norm formula in PyTorch; a
  failing ``causal_conv1d`` has no substitute here and is reported.
"""

import logging
import os

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MEGATRON_KDA_BACKEND_ENV = "LUMEN_KDA_MEGATRON_BACKEND"
AITER_RECURRENCE_ENV = "LUMEN_KDA_AITER_RECURRENCE"
AITER_CP_CONTEXT_ENV = "LUMEN_KDA_AITER_CP_CONTEXT"
ROCM_CHECK_ENV = "LUMEN_KDA_FLA_ROCM_CHECK"

_KDA_MODULES = ("megatron.core.ssm.gated_delta_net.kda", "megatron.core.ssm.gated_delta_net.common")


def _modules():
    import importlib

    out = []
    for name in _KDA_MODULES:
        try:
            out.append(importlib.import_module(name))
        except ImportError:
            pass
    return out


def build_cp_context_adapter(cu_seqlens, group, conv1d_kernel_size=None, **kwargs):
    """``fla.ops.cp.build_cp_context`` keywords mapped onto aiter's contiguous partition."""
    from aiter.ops.kda import build_kda_cp_context

    layout = kwargs.pop("layout", "contiguous")
    if layout != "contiguous":
        raise NotImplementedError(f"aiter KDA CP supports the contiguous layout only, got {layout!r}")
    return build_kda_cp_context(
        cu_seqlens, group, conv1d_kernel_size=conv1d_kernel_size, cu_seqlens_cpu=kwargs.get("cu_seqlens_cpu")
    )


def substitute_build_cp_context(force=False):
    """Point Megatron's ``build_cp_context`` at aiter's when FLA lacks it (or when ``force``)."""
    n = 0
    for mod in _modules():
        if hasattr(mod, "build_cp_context") and (force or mod.build_cp_context is None):
            mod.build_cp_context = build_cp_context_adapter
            n += 1
    if n:
        logger.info("KDA: build_cp_context -> aiter.ops.kda.build_kda_cp_context in %d module(s)", n)
    return n


def _rms_norm_gated_megatron(x, gate, weight, bias=None, activation="sigmoid", eps=1e-5, **_):
    # Megatron fused_gated_norm numerics: fp32 norm, round to the activation dtype, then the fp32 gate.
    if activation not in ("sigmoid", "swish", "silu"):
        raise NotImplementedError(f"rms_norm_gated activation {activation!r}")
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    if bias is not None:
        normed = normed + bias.float()
    normed = normed.to(x.dtype).float()
    g = gate.float()
    act = torch.sigmoid(g) if activation == "sigmoid" else F.silu(g)
    return (normed * act).to(x.dtype)


def check_fla_rocm(device="cuda", dtype=torch.bfloat16, tol=2e-2):
    """Run FLA ``causal_conv1d`` (plain, varlen, single-rank ``cp_context``) and ``rms_norm_gated``
    on this GPU against PyTorch. Returns ``{check: "ok" | "missing" | "<error or rel-norm>"}``."""
    res = {}
    torch.manual_seed(0)
    T, C, W = 96, 64, 4
    x = torch.randn(1, T, C, device=device, dtype=dtype)
    w = torch.randn(C, W, device=device, dtype=dtype) * 0.5
    ref = F.silu(F.conv1d(x.transpose(1, 2).float(), w.float().unsqueeze(1), padding=W - 1, groups=C)[..., :T]).transpose(1, 2)
    try:
        from fla.modules.convolution import causal_conv1d
    except ImportError:
        causal_conv1d = None
    if causal_conv1d is None:
        res["causal_conv1d"] = res["causal_conv1d_cp"] = "missing"
    else:
        cu = torch.tensor([0, T], device=device, dtype=torch.int32)
        for name, kw in [("causal_conv1d", {}), ("causal_conv1d_varlen", {"cu_seqlens": cu})]:
            try:
                y, _ = causal_conv1d(x=x, weight=w, bias=None, activation="silu", **kw)
                err = ((y.float() - ref).norm() / ref.norm()).item()
                res[name] = "ok" if err < tol else f"rel {err:.2e}"
            except Exception as e:  # noqa: BLE001 — reported, not raised
                res[name] = f"{type(e).__name__}: {e}"
        try:
            from aiter.ops.kda import build_kda_cp_context

            ctx = build_kda_cp_context(cu, None, conv1d_kernel_size=W, rank=0, world_size=1)
            y, _ = causal_conv1d(x=x, weight=w, bias=None, activation="silu", cu_seqlens=ctx.cu_seqlens, cp_context=ctx)
            err = ((y.float() - ref).norm() / ref.norm()).item()
            res["causal_conv1d_cp"] = "ok" if err < tol else f"rel {err:.2e}"
        except Exception as e:  # noqa: BLE001
            res["causal_conv1d_cp"] = f"{type(e).__name__}: {e}"
    try:
        from fla.modules.fused_norm_gate import rms_norm_gated
    except ImportError:
        rms_norm_gated = None
    if rms_norm_gated is None:
        res["rms_norm_gated"] = "missing"
    else:
        o = torch.randn(T, 4, 128, device=device, dtype=dtype)
        g = torch.randn_like(o)
        wn = (1 + 0.1 * torch.randn(128, device=device)).to(dtype)
        try:
            y = rms_norm_gated(o, g, wn, None, activation="sigmoid", eps=1e-5)
            r = _rms_norm_gated_megatron(o, g, wn, eps=1e-5).float()
            err = ((y.float() - r).norm() / r.norm()).item()
            res["rms_norm_gated"] = "ok" if err < tol else f"rel {err:.2e}"
        except Exception as e:  # noqa: BLE001
            res["rms_norm_gated"] = f"{type(e).__name__}: {e}"
    return res


def substitute_rms_norm_gated():
    n = 0
    for mod in _modules():
        if hasattr(mod, "rms_norm_gated"):
            mod.rms_norm_gated = _rms_norm_gated_megatron
            n += 1
    return n


def install_kda_aiter_backend(name=None, recurrence=None, force_cp_context=None, rocm_check=None):
    """Register aiter as Megatron ``kda_backend=name`` and patch the FLA pieces chunkwise CP needs.

    Call before the Megatron model is built. ``name`` defaults to ``$LUMEN_KDA_MEGATRON_BACKEND``
    (``aiter``); ``recurrence`` to ``$LUMEN_KDA_AITER_RECURRENCE`` (``triton``; or ``flydsl`` /
    ``auto``). Returns the FLA ROCm check results (empty when skipped).
    """
    name = name or os.environ.get(MEGATRON_KDA_BACKEND_ENV) or "aiter"
    recurrence = recurrence or os.environ.get(AITER_RECURRENCE_ENV, "triton")
    try:
        from aiter.ops.kda import patch_megatron_kda_backend
    except ImportError as e:
        raise ImportError("kda_backend=aiter needs aiter.ops.kda (ROCm/aiter#8755)") from e
    patch_megatron_kda_backend(name, backend=recurrence)
    if force_cp_context is None:
        force_cp_context = os.environ.get(AITER_CP_CONTEXT_ENV, "0") == "1"
    substitute_build_cp_context(force=force_cp_context)
    if rocm_check is None:
        rocm_check = os.environ.get(ROCM_CHECK_ENV, "1") == "1" and torch.cuda.is_available()
    results = check_fla_rocm() if rocm_check else {}
    if results:
        logger.info("KDA FLA-on-ROCm check: %s", results)
        if results.get("rms_norm_gated") != "ok":
            # FLA rms_norm_gated unavailable or wrong on this GPU: use the Megatron formula in PyTorch.
            logger.warning("KDA: FLA rms_norm_gated %s; using the PyTorch Megatron gated norm", results["rms_norm_gated"])
            substitute_rms_norm_gated()
        bad = {k: v for k, v in results.items() if k.startswith("causal_conv1d") and v != "ok"}
        if bad:
            logger.warning("KDA: FLA causal_conv1d on this GPU: %s; chunkwise CP / packed KDA will not work", bad)
    logger.info("KDA: registered Megatron kda_backend=%r on aiter (%s)", name, recurrence)
    return results
