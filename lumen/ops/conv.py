###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``nn.Conv2d`` forward via the AITER Triton conv2d router.

AITER only provides the forward pass, so the backward uses PyTorch's
``conv2d_input`` / ``conv2d_weight`` gradients. Convolutions AITER cannot run
(fp32 inputs, grouped / depthwise, non-zero padding modes, string padding,
CPU tensors) keep their original forward.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.grad import conv2d_input, conv2d_weight


def _aiter_conv2d():
    from aiter.ops.triton.conv.conv2d import conv2d

    return conv2d


class _AiterConv2dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation):
        ctx.save_for_backward(x, weight)
        ctx.conf = (stride, padding, dilation, bias is not None)
        return _aiter_conv2d()(x.contiguous(), weight.to(x.dtype), None if bias is None else bias.to(x.dtype),
                               stride=stride, padding=padding, dilation=dilation)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        stride, padding, dilation, has_bias = ctx.conf
        w = weight.to(grad_out.dtype)
        gx = conv2d_input(x.shape, w, grad_out, stride, padding, dilation) if ctx.needs_input_grad[0] else None
        gw = conv2d_weight(x, weight.shape, grad_out, stride, padding, dilation) if ctx.needs_input_grad[1] else None
        gb = grad_out.sum((0, 2, 3)) if has_bias and ctx.needs_input_grad[2] else None
        if gw is not None:
            gw = gw.to(weight.dtype)
        return gx, gw, gb, None, None, None


def aiter_conv2d_supported(conv: nn.Conv2d) -> bool:
    return conv.groups == 1 and conv.padding_mode == "zeros" and not isinstance(conv.padding, str)


def patch_conv2d(model: nn.Module) -> int:
    """Route every supported ``nn.Conv2d`` in ``model`` through AITER; returns the patch count."""
    _aiter_conv2d()
    count = 0
    for module in model.modules():
        if not isinstance(module, nn.Conv2d) or not aiter_conv2d_supported(module):
            continue

        def _make_forward(mod, orig):
            def _forward(x):
                if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
                    return orig(x)
                return _AiterConv2dFn.apply(x, mod.weight, mod.bias, tuple(mod.stride),
                                            tuple(mod.padding), tuple(mod.dilation))

            return _forward

        module.forward = _make_forward(module, module.forward)
        count += 1
    return count
