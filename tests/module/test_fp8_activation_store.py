###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

import pytest
import torch

DEVICE = "cuda"
_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@_CUDA
class TestFP8ActivationStoreGatedMLP:

    def test_forward_shape(self):
        from lumen.modules.fused_mlp import LumenGatedMLP

        mlp = LumenGatedMLP(64, 128, fp8_activation_store=True).to(DEVICE)
        x = torch.randn(2, 16, 64, device=DEVICE)
        out = mlp(x)
        assert out.shape == (2, 16, 64)

    def test_backward_runs(self):
        from lumen.modules.fused_mlp import LumenGatedMLP

        mlp = LumenGatedMLP(64, 128, fp8_activation_store=True).to(DEVICE)
        x = torch.randn(2, 16, 64, device=DEVICE, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert mlp.w_gate.grad is not None

    def test_matches_non_fp8_store_approximately(self):
        from lumen.modules.fused_mlp import LumenGatedMLP

        torch.manual_seed(42)
        mlp_ref = LumenGatedMLP(64, 128, fp8_activation_store=False).to(DEVICE)
        mlp_fp8 = LumenGatedMLP(64, 128, fp8_activation_store=True).to(DEVICE)
        with torch.no_grad():
            mlp_fp8.w_gate.copy_(mlp_ref.w_gate)
            mlp_fp8.w_up.copy_(mlp_ref.w_up)
            mlp_fp8.w_down.copy_(mlp_ref.w_down)
            if mlp_ref.bias_gate is not None:
                mlp_fp8.bias_gate.copy_(mlp_ref.bias_gate)
                mlp_fp8.bias_up.copy_(mlp_ref.bias_up)
                mlp_fp8.bias_down.copy_(mlp_ref.bias_down)

        x = torch.randn(2, 16, 64, device=DEVICE)
        out_ref = mlp_ref(x)
        out_fp8 = mlp_fp8(x)

        torch.testing.assert_close(out_ref, out_fp8, atol=1e-5, rtol=1e-5)


@_CUDA
class TestFP8ActivationStoreFusedMLP:

    def test_forward_shape(self):
        from lumen.modules.fused_mlp import LumenFusedMLP

        mlp = LumenFusedMLP(64, 128, fp8_activation_store=True).to(DEVICE)
        x = torch.randn(2, 16, 64, device=DEVICE)
        out = mlp(x)
        assert out.shape == (2, 16, 64)

    def test_backward_runs(self):
        from lumen.modules.fused_mlp import LumenFusedMLP

        mlp = LumenFusedMLP(64, 128, fp8_activation_store=True).to(DEVICE)
        x = torch.randn(2, 16, 64, device=DEVICE, requires_grad=True)
        out = mlp(x)
        out.sum().backward()
        assert x.grad is not None
