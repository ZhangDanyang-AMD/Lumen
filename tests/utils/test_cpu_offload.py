###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

import pytest
import torch

from lumen.utils.cpu_offload import CPUOffloadManager, lumen_cpu_offload_context


class TestCPUOffloadManager:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_pack_returns_cpu_for_cuda_tensor(self):
        mgr = CPUOffloadManager(enabled=True)
        t = torch.randn(64, 64, device="cuda")
        packed = mgr._pack(t)
        assert not packed.is_cuda
        assert packed.is_pinned()
        assert mgr.memory_saved_bytes == 64 * 64 * 4

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_pack_skips_tiny_tensors(self):
        mgr = CPUOffloadManager(enabled=True)
        tiny = torch.randn(4, device="cuda")  # 16 bytes
        packed = mgr._pack(tiny)
        assert packed.is_cuda
        assert mgr.memory_saved_bytes == 0

    def test_pack_skips_cpu_tensors(self):
        mgr = CPUOffloadManager(enabled=True)
        t = torch.randn(64, 64)
        packed = mgr._pack(t)
        assert packed is t

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_unpack_returns_cuda_tensor(self):
        mgr = CPUOffloadManager(enabled=True)
        t = torch.randn(64, 64, device="cuda")
        packed = mgr._pack(t)
        unpacked = mgr._unpack(packed)
        assert unpacked.is_cuda
        assert torch.allclose(t, unpacked)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_context_produces_correct_gradients(self):
        """Full forward+backward through context."""
        model = torch.nn.Linear(256, 128).cuda()
        x = torch.randn(32, 256, device="cuda", requires_grad=True)

        # Reference
        out_ref = model(x)
        loss_ref = out_ref.sum()
        loss_ref.backward()
        grad_ref = x.grad.clone()
        x.grad = None

        # With offload
        with lumen_cpu_offload_context(enabled=True) as mgr:
            out = model(x)
            loss = out.sum()
        loss.backward()
        grad_offload = x.grad.clone()

        assert torch.allclose(grad_ref, grad_offload)
        assert mgr.memory_saved_bytes > 0
