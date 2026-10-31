"""Correctness tests for the separate-input fused SwiGLU training kernel."""

import pytest
import torch
import torch.nn.functional as F


_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")


@_CUDA
@pytest.mark.parametrize("shape", [(8, 128), (4, 11008)])
def test_fused_swiglu_pair_forward_backward(shape):
    from lumen.ops.fused_swiglu import fused_swiglu_pair

    torch.manual_seed(123)
    gate_ref = torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    up_ref = torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gate = gate_ref.detach().clone().requires_grad_(True)
    up = up_ref.detach().clone().requires_grad_(True)
    grad = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    expected = F.silu(gate_ref) * up_ref
    actual = fused_swiglu_pair(gate, up)
    expected.backward(grad)
    actual.backward(grad)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(gate.grad, gate_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(up.grad, up_ref.grad, atol=2e-2, rtol=2e-2)


@_CUDA
def test_fused_swiglu_pair_falls_back_above_single_program_limit():
    from lumen.ops.fused_swiglu import fused_swiglu_pair

    gate = torch.randn(1, 65537, device="cuda", requires_grad=True)
    up = torch.randn(1, 65537, device="cuda", requires_grad=True)
    actual = fused_swiglu_pair(gate, up)
    expected = F.silu(gate) * up

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(gate.grad).all()
    assert torch.isfinite(up.grad).all()
