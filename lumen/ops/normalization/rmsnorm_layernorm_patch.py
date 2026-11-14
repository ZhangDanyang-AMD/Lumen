###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Replace RMSNorm and LayerNorm modules on a model.

``atom_mode`` installs :class:`lumen.ops.normalization.rmsnorm_with_residual.RMSNormWithResidual`.
Otherwise the installer uses ``LumenRMSNorm`` and ``LumenLayerNorm``.
"""

from __future__ import annotations

import os

import torch

from lumen.kernels.rmsnorm_autograd import rmsnorm_fns
from lumen.ops.normalization.rmsnorm_with_residual import RMSNormWithResidual


def _rank0_print(msg: str) -> None:
    from lumen.config import _rank0_print as _print

    _print(msg)


def patch_norms(model, *, atom_mode: bool = False, extra_classes: tuple[str, ...] = (), quantize_grad=None) -> None:
    """Replace RMSNorm / LayerNorm modules on *model*."""
    if atom_mode:
        if os.environ.get("ATOM_USE_TORCH_RMSNORM", "0") == "1":
            _rank0_print("> ATOM rollout: ATOM_USE_TORCH_RMSNORM=1, keeping original RMSNorm")
            return

        # Importing the kernels here, rather than on the first forward, keeps
        # a missing aiter install as a patch-time error.
        rmsnorm_fns()

        norm_classes = (
            "RMSNorm",
            "LlamaRMSNorm",
            "MistralRMSNorm",
            "Qwen2RMSNorm",
            "MegatronRMSNorm",
            "TENorm",
            "LumenRMSNorm",
            "_MegatronCompatibleTLNorm",
            "MegatronCompatibleTLNorm",
            "_MegatronCompatibleTLRMSNorm",
            "MegatronCompatibleTLRMSNorm",
            *extra_classes,
        )
        count = 0
        for _name, module in model.named_modules():
            for attr_name, child in list(module.named_children()):
                cls_name = type(child).__name__
                if cls_name in norm_classes:
                    w = getattr(child, "weight", None)
                    if w is None:
                        w = getattr(getattr(child, "_norm", None), "weight", None)
                    if w is None:
                        continue
                    eps = getattr(child, "eps", getattr(child, "variance_epsilon",
                          getattr(child, "epsilon", 1e-6)))
                    repl = RMSNormWithResidual(w, eps)
                    if w.is_meta:
                        repl.to(device="meta")
                    setattr(module, attr_name, repl)
                    count += 1
        if count:
            _rank0_print(f"> Replaced {count} RMSNorm modules with ATOM-aligned aiter.rmsnorm2d_fwd")
        return

    from lumen.ops.normalization import LumenLayerNorm, LumenRMSNorm

    count = 0
    for _name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            cls_name = type(child).__name__
            if cls_name in (
                "RMSNorm",
                "LlamaRMSNorm",
                "MistralRMSNorm",
                "Qwen2RMSNorm",
                "Qwen3RMSNorm",
            ):
                hidden = child.weight.shape[0]
                eps = getattr(child, "eps", getattr(child, "variance_epsilon", 1e-6))
                target_dtype = child.weight.dtype if not child.weight.is_meta else torch.bfloat16
                repl = LumenRMSNorm(hidden, eps=eps, grad_quant_type=quantize_grad)
                if not child.weight.is_meta:
                    repl.weight.data.copy_(child.weight.data)
                repl.to(target_dtype)
                if child.weight.is_meta:
                    repl.to(device="meta")
                setattr(module, attr_name, repl)
                count += 1
            elif cls_name in ("LayerNorm",):
                hidden = child.weight.shape[0] if child.weight is not None else child.normalized_shape[0]
                eps = getattr(child, "eps", 1e-5)
                repl = LumenLayerNorm(hidden, eps=eps, grad_quant_type=quantize_grad)
                is_meta = child.weight is not None and child.weight.is_meta
                target_dtype = child.weight.dtype if (child.weight is not None and not is_meta) else torch.bfloat16
                if child.weight is not None and not is_meta:
                    repl.weight.data.copy_(child.weight.data)
                if hasattr(child, "bias") and child.bias is not None and repl.bias is not None and not is_meta:
                    repl.bias.data.copy_(child.bias.data)
                repl.to(target_dtype)
                if is_meta:
                    repl.to(device="meta")
                setattr(module, attr_name, repl)
                count += 1

    if count:
        _rank0_print(f"> Replaced {count} norm modules with Lumen implementations")
