###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""``LumenConfig(aiter_conv=True)`` routes ``nn.Conv2d`` through AITER Triton conv2d."""

import copy

import pytest
import torch
import torch.nn as nn

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
    assert ["forward" in vars(m) for m in model] == [False, False, False, True]

    x = torch.randn(1, 8, 16, 16, device="cuda")
    ref = x
    for m, f in zip(model, forwards):
        ref = f(m, ref)
    torch.testing.assert_close(model(x), ref)
