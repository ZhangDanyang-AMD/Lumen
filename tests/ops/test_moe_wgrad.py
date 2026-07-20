###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Correctness tests for the MoE Triton wgrad kernel and stride-swap dgrad."""

import pytest
import torch
import torch.nn.functional as F

DEVICE = "cuda"
_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _has_aiter_moe():
    try:
        from lumen.ops.moe.fused_moe import fused_moe_triton, moe_wgrad_triton
        from lumen.ops.dispatch import (
            _probe_aiter_triton_fused_moe,
            _probe_aiter_triton_moe_align,
        )
        return _probe_aiter_triton_fused_moe() and _probe_aiter_triton_moe_align()
    except ImportError:
        return False


_AITER = pytest.mark.skipif(not _has_aiter_moe(), reason="AITER MoE kernels required")


def _reference_wgrad(grad, input_act, expert_ids, num_experts, weight_shape):
    """Reference wgrad: dW[e] = sum_{t assigned to e} grad[t].T @ input[t]."""
    E, N, K = weight_shape
    dW = torch.zeros(weight_shape, dtype=grad.dtype, device=grad.device)
    for t in range(grad.shape[0]):
        e = expert_ids[t].item()
        if 0 <= e < E:
            dW[e] += grad[t].unsqueeze(1).float() @ input_act[t].unsqueeze(0).float()
    return dW.to(grad.dtype)


def _reference_moe_forward(hidden, w, expert_ids, num_experts):
    """Reference: per-expert hidden @ W[e].T, single top-k=1."""
    T = hidden.shape[0]
    N = w.shape[1]
    out = torch.zeros(T, N, dtype=hidden.dtype, device=hidden.device)
    for t in range(T):
        e = expert_ids[t].item()
        out[t] = hidden[t] @ w[e].T
    return out


@_CUDA
@_AITER
class TestMoeWgrad:
    """Test moe_wgrad_triton against PyTorch reference."""

    @pytest.mark.parametrize("num_tokens,num_experts,N,K", [
        (32, 4, 64, 32),
        (64, 8, 128, 64),
        (128, 16, 64, 64),
    ])
    def test_wgrad_correctness(self, num_tokens, num_experts, N, K):
        from lumen.ops.moe.fused_moe import _align_tokens, _FALLBACK_MOE_CONFIG, moe_wgrad_triton

        torch.manual_seed(42)
        grad = torch.randn(num_tokens, N, dtype=torch.bfloat16, device=DEVICE)
        input_act = torch.randn(num_tokens, K, dtype=torch.bfloat16, device=DEVICE)
        expert_ids = torch.randint(0, num_experts, (num_tokens,), device=DEVICE)

        topk_ids = expert_ids.unsqueeze(1).to(torch.int32)
        block_size_m = _FALLBACK_MOE_CONFIG["BLOCK_SIZE_M"]
        alignment = _align_tokens(topk_ids, num_experts, block_size_m)
        sorted_token_ids, expert_ids_aligned, num_tokens_post_pad = alignment

        dW = moe_wgrad_triton(
            grad, input_act,
            sorted_token_ids, expert_ids_aligned, num_tokens_post_pad,
            num_experts, 1, (num_experts, N, K),
        )
        dW_ref = _reference_wgrad(grad, input_act, expert_ids, num_experts, (num_experts, N, K))

        torch.testing.assert_close(dW.float(), dW_ref.float(), atol=0.05, rtol=0.02)

    def test_wgrad_empty(self):
        from lumen.ops.moe.fused_moe import _align_tokens, _FALLBACK_MOE_CONFIG, moe_wgrad_triton

        num_experts, N, K = 4, 64, 32
        grad = torch.randn(0, N, dtype=torch.bfloat16, device=DEVICE)
        input_act = torch.randn(0, K, dtype=torch.bfloat16, device=DEVICE)
        topk_ids = torch.zeros(0, 1, dtype=torch.int32, device=DEVICE)

        block_size_m = _FALLBACK_MOE_CONFIG["BLOCK_SIZE_M"]
        alignment = _align_tokens(topk_ids, num_experts, block_size_m)

        dW = moe_wgrad_triton(
            grad, input_act,
            *alignment, num_experts, 1, (num_experts, N, K),
        )
        assert dW.shape == (num_experts, N, K)
        assert (dW == 0).all()


@_CUDA
@_AITER
class TestStrideSwapDgrad:
    """Test that fused_moe with B.transpose(1,2) computes A @ B (not A @ B.T)."""

    @pytest.mark.parametrize("num_tokens,num_experts,N,K", [
        (32, 4, 64, 32),
        (64, 8, 128, 64),
    ])
    def test_stride_swap(self, num_tokens, num_experts, N, K):
        from lumen.ops.moe.fused_moe import _align_tokens, _FALLBACK_MOE_CONFIG, fused_moe_triton

        torch.manual_seed(42)
        A = torch.randn(num_tokens, N, dtype=torch.bfloat16, device=DEVICE)
        B = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device=DEVICE)
        expert_ids = torch.randint(0, num_experts, (num_tokens,), device=DEVICE)

        topk_ids = expert_ids.unsqueeze(1).to(torch.int32)
        ones = torch.ones(num_tokens, 1, dtype=torch.float32, device=DEVICE)
        block_size_m = _FALLBACK_MOE_CONFIG["BLOCK_SIZE_M"]
        alignment = _align_tokens(topk_ids, num_experts, block_size_m)

        # fused_moe with B.transpose(1,2) should compute A @ B (not A @ B.T)
        out = fused_moe_triton(
            A, B.transpose(1, 2), topk_ids, ones,
            num_experts, 1, mul_routed_weight=False,
            precomputed_alignment=alignment,
        ).squeeze(1)

        ref = _reference_moe_forward(A, B.transpose(1, 2), expert_ids, num_experts)

        torch.testing.assert_close(out.float(), ref.float(), atol=0.05, rtol=0.02)


@_CUDA
@_AITER
class TestMoeEndToEndGradient:
    """End-to-end gradient check: forward + backward through SwiGLU MoE."""

    def test_e2e_gradient(self):
        from lumen.ops.moe.fused_moe import (
            _align_tokens, _FALLBACK_MOE_CONFIG, fused_moe_triton, moe_wgrad_triton,
        )

        torch.manual_seed(42)
        T, E, N, K, top_k = 16, 4, 32, 16, 1
        block_size_m = _FALLBACK_MOE_CONFIG["BLOCK_SIZE_M"]

        hidden = torch.randn(T, K, dtype=torch.bfloat16, device=DEVICE)
        expert_ids = torch.randint(0, E, (T,), device=DEVICE)
        topk_ids = expert_ids.unsqueeze(1).to(torch.int32)
        ones = torch.ones(T, 1, dtype=torch.float32, device=DEVICE)

        w_gate = (torch.randn(E, N, K, device=DEVICE) * 0.02).to(torch.bfloat16).requires_grad_(True)
        w_up = (torch.randn(E, N, K, device=DEVICE) * 0.02).to(torch.bfloat16).requires_grad_(True)
        w_down = (torch.randn(E, K, N, device=DEVICE) * 0.02).to(torch.bfloat16).requires_grad_(True)

        alignment = _align_tokens(topk_ids, E, block_size_m)

        # Forward
        gate_out = fused_moe_triton(
            hidden, w_gate, topk_ids, ones, E, 1,
            mul_routed_weight=False, precomputed_alignment=alignment,
        ).squeeze(1)
        up_out = fused_moe_triton(
            hidden, w_up, topk_ids, ones, E, 1,
            mul_routed_weight=False, precomputed_alignment=alignment,
        ).squeeze(1)
        sg = F.silu(gate_out)
        act_out = sg * up_out
        down_out = fused_moe_triton(
            act_out, w_down, topk_ids, ones, E, 1,
            mul_routed_weight=False, precomputed_alignment=alignment,
        ).squeeze(1)
        loss = down_out.sum()

        # Reference backward via autograd (for input grad)
        loss.backward()

        # Verify gradients are non-None
        assert w_gate.grad is not None, "w_gate.grad is None"
        assert w_up.grad is not None, "w_up.grad is None"
        assert w_down.grad is not None, "w_down.grad is None"

        # Now compute wgrad via our kernel and compare
        grad_output = torch.ones_like(down_out)
        grad_down = grad_output

        # dgrad for down
        grad_act = fused_moe_triton(
            grad_down, w_down.detach().transpose(1, 2), topk_ids, ones,
            E, 1, mul_routed_weight=False, precomputed_alignment=alignment,
        ).squeeze(1)

        sg_detach = F.silu(gate_out.detach())
        grad_up = grad_act * sg_detach
        grad_silu = grad_act * up_out.detach()
        sig = torch.sigmoid(gate_out.detach())
        grad_gate = grad_silu * sig * (1.0 + gate_out.detach() * (1.0 - sig))

        sorted_token_ids, expert_ids_a, num_tokens_post_pad = alignment

        grad_w_gate_triton = moe_wgrad_triton(
            grad_gate, hidden,
            sorted_token_ids, expert_ids_a, num_tokens_post_pad,
            E, 1, w_gate.shape,
        )
        grad_w_up_triton = moe_wgrad_triton(
            grad_up, hidden,
            sorted_token_ids, expert_ids_a, num_tokens_post_pad,
            E, 1, w_up.shape,
        )
        grad_w_down_triton = moe_wgrad_triton(
            grad_down, act_out.detach(),
            sorted_token_ids, expert_ids_a, num_tokens_post_pad,
            E, 1, w_down.shape,
        )

        torch.testing.assert_close(
            grad_w_gate_triton.float(), w_gate.grad.float(), atol=0.1, rtol=0.05,
        )
        torch.testing.assert_close(
            grad_w_up_triton.float(), w_up.grad.float(), atol=0.1, rtol=0.05,
        )
        torch.testing.assert_close(
            grad_w_down_triton.float(), w_down.grad.float(), atol=0.1, rtol=0.05,
        )
