###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

from types import SimpleNamespace

import torch
import torch.nn as nn

from lumen.utils.checkpoint import (
    _FP8ScalingContext,
    lumen_checkpoint,
    lumen_checkpoint_core_attention,
)


class TestFP8ScalingContext:

    def test_save_restore_none(self):
        ctx = _FP8ScalingContext()
        ctx.save(None)
        ctx.restore(None)

    def test_save_restore_dict_state(self):
        mgr = SimpleNamespace(
            amax_history={"w": torch.tensor([1.0, 2.0])},
            scale={"w": torch.tensor([0.5])},
            step=10,
        )
        ctx = _FP8ScalingContext()
        ctx.save(mgr)

        # Modify state
        mgr.amax_history["w"].fill_(99.0)
        mgr.scale["w"].fill_(99.0)

        # Restore should bring back original values
        ctx.restore(mgr)
        torch.testing.assert_close(mgr.amax_history["w"], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(mgr.scale["w"], torch.tensor([0.5]))


class TestLumenCheckpoint:

    def test_basic_checkpoint(self):
        layer = nn.Linear(16, 16)
        x = torch.randn(4, 16, requires_grad=True)

        out = lumen_checkpoint(layer, x)
        assert out.shape == (4, 16)

        out.sum().backward()
        assert x.grad is not None
        assert layer.weight.grad is not None

    def test_checkpoint_matches_no_checkpoint(self):
        torch.manual_seed(42)
        layer = nn.Linear(16, 16)
        x = torch.randn(4, 16, requires_grad=True)

        out_ckpt = lumen_checkpoint(layer, x)
        out_ckpt.sum().backward()
        grad_ckpt = x.grad.clone()

        x.grad = None
        out_direct = layer(x)
        out_direct.sum().backward()
        grad_direct = x.grad.clone()

        torch.testing.assert_close(grad_ckpt, grad_direct)

    def test_checkpoint_with_scaling_manager(self):
        mgr = SimpleNamespace(
            amax_history={"w": torch.tensor([1.0])},
            scale={"w": torch.tensor([0.5])},
        )

        layer = nn.Linear(16, 16)
        x = torch.randn(4, 16, requires_grad=True)

        out = lumen_checkpoint(layer, x, scaling_manager=mgr)
        assert out.shape == (4, 16)


class TestLumenCheckpointCoreAttention:

    def test_basic(self):
        def simple_attn(q, k, v):
            return torch.bmm(q, k.transpose(1, 2)).softmax(-1) @ v

        B, S, D = 2, 8, 16
        q = torch.randn(B, S, D, requires_grad=True)
        k = torch.randn(B, S, D, requires_grad=True)
        v = torch.randn(B, S, D, requires_grad=True)

        out = lumen_checkpoint_core_attention(simple_attn, q, k, v)
        assert out.shape == (B, S, D)
        out.sum().backward()
        assert q.grad is not None
