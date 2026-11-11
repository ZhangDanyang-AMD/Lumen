###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""``LumenConfig(aiter_conv=True)`` routes Conv2d and Conv3d through AITER."""

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lumen.config import LumenConfig


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("kernel,stride,padding", [(3, 1, 1), (1, 1, 0), (3, 2, 1), (4, 2, 1)])
def test_forward_backward_match_torch(kernel, stride, padding):
    torch.manual_seed(0)
    ref = nn.Conv2d(64, 96, kernel, stride=stride, padding=padding).cuda().bfloat16()
    patched = copy.deepcopy(ref)
    LumenConfig(aiter_conv=True).enable(patched)

    x = torch.randn(2, 64, 32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x2 = x.detach().clone().requires_grad_()
    y, y2 = ref(x), patched(x2)
    g = torch.randn_like(y)
    y.backward(g)
    y2.backward(g)

    assert _rel(y2, y) < 1e-2
    assert _rel(x2.grad, x.grad) < 1e-2
    assert _rel(patched.weight.grad, ref.weight.grad) < 1e-2
    assert _rel(patched.bias.grad, ref.bias.grad) < 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_unsupported_convs_keep_torch_forward():
    model = nn.Sequential(
        nn.Conv2d(8, 8, 3, padding=1, groups=8),
        nn.Conv2d(8, 8, 3, padding=1, padding_mode="reflect"),
        nn.Conv2d(8, 8, 3, padding="same"),
        nn.Conv2d(8, 8, 3, padding=1),
    ).cuda()
    forwards = [type(m).forward for m in model]
    LumenConfig(aiter_conv=True).enable(model)
    assert ["_conv_forward" in vars(m) for m in model] == [False, False, False, True]

    x = torch.randn(1, 8, 16, 16, device="cuda")
    ref = x
    for m, f in zip(model, forwards):
        ref = f(m, ref)
    torch.testing.assert_close(model(x), ref)


class _CausalConv3d(nn.Conv3d):
    """Same pad-then-conv shape as the Wan VAE. ``forward`` must stay ours."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            x = torch.cat([cache_x.to(x.device), x], dim=2)
            padding[4] -= cache_x.shape[2]
        return super().forward(F.pad(x, padding))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("kernel,padding", [((3, 3, 3), 1), ((1, 1, 1), 0), ((3, 1, 1), (1, 0, 0))])
def test_conv3d_matches_torch(kernel, padding):
    torch.manual_seed(0)
    ref = nn.Conv3d(8, 16, kernel, padding=padding).cuda().bfloat16()
    patched = copy.deepcopy(ref)
    LumenConfig(aiter_conv=True).enable(patched)
    x = torch.randn(1, 8, 4, 8, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x2 = x.detach().clone().requires_grad_()
    y, y2 = ref(x), patched(x2)
    g = torch.randn_like(y)
    y.backward(g)
    y2.backward(g)
    assert _rel(y2, y) < 2e-2
    assert _rel(x2.grad, x.grad) < 2e-2
    assert _rel(patched.weight.grad, ref.weight.grad) < 2e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_causal_conv3d_keeps_its_forward():
    torch.manual_seed(0)
    ref = _CausalConv3d(4, 4, 3, padding=1).cuda().bfloat16()
    patched = copy.deepcopy(ref)
    LumenConfig(aiter_conv=True).enable(patched)
    assert patched.forward.__func__ is _CausalConv3d.forward
    x = torch.randn(1, 4, 2, 8, 8, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(1, 4, 2, 8, 8, device="cuda", dtype=torch.bfloat16)
    y, y2 = ref(x, cache), patched(x, cache)
    assert _rel(y2, y) < 2e-2
