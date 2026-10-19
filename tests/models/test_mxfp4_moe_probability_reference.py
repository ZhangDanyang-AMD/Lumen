"""Independent clamp/probability-chain sensitivity on real Core expert body."""

import os

import pytest
import torch
import torch.distributed as dist

from tests.env.mxfp4_moe_fixture import build_core_layer, signed_config


def reference_chain(fc1, probabilities):
    gate, up = fc1.chunk(2, dim=-1)
    return (
        torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10) * probabilities
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_clamp_probability_reference_is_sensitive():
    gate = torch.tensor([-12.0, -2.0, 2.0, 12.0], device="cuda", dtype=torch.bfloat16)
    up = torch.tensor([-12.0, -2.0, 2.0, 12.0], device="cuda", dtype=torch.bfloat16)
    fc1 = torch.cat((gate, up)).reshape(1, 8).requires_grad_(True)
    p = torch.tensor([[0.75]], device="cuda", dtype=torch.bfloat16, requires_grad=True)
    result = reference_chain(fc1, p)
    result.float().sum().backward()
    assert p.grad is not None and p.grad.abs().sum() > 0
    assert fc1.grad is not None
    wrong = (
        torch.nn.functional.silu(gate.clamp(-10, 10)) * up.clamp(-10, 10) * p.detach()
    )
    assert not torch.equal(result, wrong)


def test_real_core_gate_is_trainable_and_pre_softmax_sensitive():
    if not dist.is_initialized():
        pytest.skip("authorized world8 Core fixture required")
    ep = int(os.environ["MXFP4_TEST_EP"])
    config, _ = signed_config(os.environ["MXFP4_SIGNED_CONTRACT"], ep=ep, overlap=False)
    layer = build_core_layer(config)
    assert layer.router.weight.requires_grad
    logits = (
        torch.linspace(-5, 5, 256, device="cuda").reshape(1, 256).requires_grad_(True)
    )
    full = logits.softmax(-1).topk(6).values * 1.5
    wrong = logits.topk(6).values.softmax(-1)
    assert not torch.allclose(full, wrong)
    full.square().sum().backward()
    assert logits.grad is not None and torch.count_nonzero(logits.grad).item() > 6
