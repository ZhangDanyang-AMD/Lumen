###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Conv2d and Conv3d through AITER, applied by ``LumenConfig.enable``.

Only ``_conv_forward`` is rebound. A subclass that pads (or concatenates a
causal cache) and then calls ``super().forward`` keeps that logic; the
convolution underneath it is what changes. AITER ships the forward only, so
the backward is PyTorch's ``convNd_input`` / ``convNd_weight``.

A call AITER cannot run — fp32, grouped, non-zero ``padding_mode``, string
padding, CPU, or a kernel rejection — uses the original ``_conv_forward``.
"""

from __future__ import annotations

import logging
import os
import sys
import types

import torch
import torch.nn as nn
from torch.nn.grad import conv2d_input, conv2d_weight, conv3d_input, conv3d_weight

logger = logging.getLogger(__name__)

_FP16 = (torch.float16, torch.bfloat16)
_CONV3D = None
_CONV3D_TRIED = False


def _conv2d():
    from aiter.ops.triton.conv.conv2d import conv2d

    return conv2d


def _prepend(pkg, path: str) -> None:
    cur = list(pkg.__path__)
    if path not in cur:
        pkg.__path__ = [path, *cur]


def _conv3d_root() -> str | None:
    """Checkout that contains ``aiter/ops/triton/conv/conv3d.py``, if any."""
    roots = []
    for key in ("AITER_SRC", "AITER_DIR"):
        val = os.environ.get(key)
        if val:
            roots.append(val)
    for root in roots:
        if os.path.isfile(os.path.join(root, "aiter", "ops", "triton", "conv", "conv3d.py")):
            return root
    return None


def _conv3d():
    """``aiter`` conv3d. An install without that module can load it from ``AITER_SRC``."""
    global _CONV3D, _CONV3D_TRIED
    if _CONV3D is not None or _CONV3D_TRIED:
        return _CONV3D
    _CONV3D_TRIED = True
    try:
        from aiter.ops.triton.conv.conv3d import conv3d

        _CONV3D = conv3d
        return conv3d
    except ImportError:
        pass

    root = _conv3d_root()
    if root is None:
        logger.warning("aiter conv3d is not installed and AITER_SRC is unset; Conv3d stays on torch")
        return None

    import aiter.ops.triton.conv as conv_pkg
    import aiter.ops.triton._triton_kernels as kernels_pkg
    import aiter.ops.triton._triton_kernels.conv as conv_kernels_pkg
    import aiter.ops.triton.utils as utils_pkg

    src = os.path.join(root, "aiter", "ops", "triton")
    _prepend(conv_pkg, os.path.join(src, "conv"))
    _prepend(kernels_pkg, os.path.join(src, "_triton_kernels"))
    _prepend(conv_kernels_pkg, os.path.join(src, "_triton_kernels", "conv"))
    _prepend(utils_pkg, os.path.join(src, "utils"))

    # conv2d may already have bound the install's helpers. Swap those modules
    # only for this import, then put them back so conv2d's late imports stay put.
    names = (
        "aiter.ops.triton.utils.conv_config_utils",
        "aiter.ops.triton.conv._utils",
        "aiter.ops.triton.conv._launch",
        "aiter.ops.triton.conv._prepack",
    )
    saved = {}
    for name in names:
        mod = sys.modules.get(name)
        if mod is not None and root not in (getattr(mod, "__file__", "") or ""):
            saved[name] = mod
            del sys.modules[name]
    try:
        from aiter.ops.triton.conv.conv3d import conv3d
    except ImportError as exc:
        logger.warning("aiter conv3d failed to load from %s (%s); Conv3d stays on torch", root, exc)
        return None
    finally:
        for name, mod in saved.items():
            sys.modules[name] = mod
    _CONV3D = conv3d
    return conv3d


class _AiterConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, spatial):
        ctx.save_for_backward(x, weight)
        ctx.conf = (stride, padding, dilation, bias is not None, spatial)
        bias_arg = None if bias is None else bias.to(dtype=x.dtype)
        weight = weight.to(dtype=x.dtype)
        fn = _conv3d() if spatial == 3 else _conv2d()
        return fn(x.contiguous(), weight, bias_arg, stride=stride, padding=padding, dilation=dilation)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        stride, padding, dilation, has_bias, spatial = ctx.conf
        w = weight.to(dtype=grad_out.dtype)
        input_grad = conv3d_input if spatial == 3 else conv2d_input
        weight_grad = conv3d_weight if spatial == 3 else conv2d_weight
        reduce = (0, 2, 3, 4) if spatial == 3 else (0, 2, 3)
        gx = input_grad(x.shape, w, grad_out, stride, padding, dilation) if ctx.needs_input_grad[0] else None
        gw = weight_grad(x, weight.shape, grad_out, stride, padding, dilation) if ctx.needs_input_grad[1] else None
        gb = grad_out.sum(reduce) if has_bias and ctx.needs_input_grad[2] else None
        if gw is not None:
            gw = gw.to(dtype=weight.dtype)
        return gx, gw, gb, None, None, None, None


def _supported(conv: nn.modules.conv._ConvNd) -> bool:
    return conv.groups == 1 and conv.padding_mode == "zeros" and not isinstance(conv.padding, str)


def _bind(module, spatial: int) -> None:
    if getattr(module._conv_forward, "_lumen_conv", False):
        return
    orig = module._conv_forward

    def _conv_forward(self, inp, weight, bias, _orig=orig, _spatial=spatial):
        if not inp.is_cuda or inp.dtype not in _FP16 or self.groups != 1:
            return _orig(inp, weight, bias)
        if _spatial == 3 and _conv3d() is None:
            return _orig(inp, weight, bias)
        try:
            return _AiterConvFn.apply(
                inp, weight, bias, tuple(self.stride), tuple(self.padding), tuple(self.dilation), _spatial,
            )
        except Exception as exc:
            if not getattr(_conv_forward, "_logged", False):
                logger.warning(
                    "aiter conv%dd fell back to torch for weight %s: %s",
                    _spatial, tuple(self.weight.shape), exc,
                )
                _conv_forward._logged = True
            return _orig(inp, weight, bias)

    _conv_forward._lumen_conv = True
    module._conv_forward = types.MethodType(_conv_forward, module)


def patch_conv(model: nn.Module) -> tuple[int, int]:
    """Patch supported ``nn.Conv2d`` and ``nn.Conv3d`` in ``model``.

    Returns ``(n_conv2d, n_conv3d)``. Subclasses are included, but their
    ``forward`` is left alone.
    """
    _conv2d()
    _conv3d()
    n2 = n3 = 0
    for module in model.modules():
        if not isinstance(module, (nn.Conv2d, nn.Conv3d)) or not _supported(module):
            continue
        if isinstance(module, nn.Conv3d):
            if _conv3d() is None:
                continue
            _bind(module, 3)
            n3 += 1
        else:
            _bind(module, 2)
            n2 += 1
    return n2, n3
