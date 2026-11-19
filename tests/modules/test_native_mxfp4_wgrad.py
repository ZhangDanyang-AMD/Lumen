"""Verify native main_grad accumulation and fresh packs across model updates."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lumen.modules import mxfp4_pack_cache as lifecycle
from lumen.modules.sonic_moe import SonicMoEExperts

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def source():
    module = nn.Module()
    module.config = SimpleNamespace(
        gated_linear_unit=True, add_bias_linear=False, expert_tensor_parallel_size=1
    )
    module.num_local_experts = 4
    module.local_experts = nn.ModuleList()
    for _ in range(4):
        expert = nn.Module()
        expert.linear_fc1 = nn.Linear(
            128, 128, bias=False, device="cuda", dtype=torch.bfloat16
        )
        expert.linear_fc2 = nn.Linear(
            64, 128, bias=False, device="cuda", dtype=torch.bfloat16
        )
        module.local_experts.append(expert)
    return module


def model_with_buffers():
    model = SonicMoEExperts(source())
    model.precision_recipe = "a4w4"
    for p in (model.w1, model.w2):
        p.main_grad = torch.zeros_like(p, dtype=torch.float32)
        p.grad_added_to_main_grad = False
    return model


def mb(counts, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows = sum(counts)
    return (
        (torch.randn(rows, 128, device="cuda", generator=gen) * 0.1)
        .bfloat16()
        .requires_grad_(),
        torch.rand(rows, device="cuda", generator=gen).requires_grad_(),
        (torch.randn(rows, 128, device="cuda", generator=gen) * 0.01).bfloat16(),
        torch.tensor(counts, dtype=torch.int32),
    )


def run(model, data):
    x, score, cotangent, counts = data
    x.grad, score.grad = None, None
    out, bias = model(x, counts, score)
    assert bias is None
    out.backward(cotangent)
    torch.cuda.synchronize()
    assert x.grad is not None and score.grad is not None
    assert torch.isfinite(x.grad).all() and torch.isfinite(score.grad).all()
    assert all(p.grad is None and p.grad_added_to_main_grad for p in model.parameters())
    return out.detach()


@_CUDA
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_distinct_microbatches_accumulate_fp32_without_double_add(monkeypatch, backend):
    monkeypatch.setenv("SONIC_MOE_MXFP4_WGRAD_BACKEND", backend)
    monkeypatch.setenv("SONIC_MOE_MXFP4_WGRAD", "1")
    monkeypatch.setenv("SONIC_MOE_MXFP4_WGRAD_FALLBACK", "error")
    monkeypatch.setenv("SONIC_MOE_PACK_CACHE", "0")
    torch.manual_seed(8103)
    combined = model_with_buffers()
    independent = [model_with_buffers(), model_with_buffers()]
    for model in independent:
        model.load_state_dict(combined.state_dict())
    data = [mb([3, 0, 33, 129], 8104), mb([0, 31, 1, 257], 8105)]
    for model, item in zip(independent, data):
        run(model, item)
    for item in data:
        run(combined, item)
    for index, p in enumerate((combined.w1, combined.w2)):
        other = [(m.w1, m.w2)[index].main_grad for m in independent]
        torch.testing.assert_close(p.main_grad, other[0] + other[1], rtol=0, atol=0)
        assert p.main_grad.dtype == torch.float32
        assert torch.isfinite(p.main_grad).all()
    # The next update window starts from zero, not a retained previous contribution.
    for p in combined.parameters():
        p.main_grad.zero_()
        p.grad_added_to_main_grad = False
    run(combined, data[1])
    for p, ref in zip(combined.parameters(), independent[1].parameters()):
        torch.testing.assert_close(p.main_grad, ref.main_grad, rtol=0, atol=0)


@_CUDA
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_data_copy_update_consumes_new_packed_weights(monkeypatch, backend):
    monkeypatch.setenv("SONIC_MOE_MXFP4_WGRAD_BACKEND", backend)
    monkeypatch.setenv("SONIC_MOE_MXFP4_WGRAD", "1")
    monkeypatch.setenv("SONIC_MOE_PACK_CACHE", "1")
    monkeypatch.setenv("R80_CACHE_GENERATION", "1")
    torch.manual_seed(8106)
    cached = model_with_buffers()
    data = mb([1, 0, 32, 129], 8107)

    class CopyOptimizer:
        def step(self):
            # Reproduce Megatron's version-preserving main-to-model data write.
            for p in cached.parameters():
                p.data.copy_((p.float() * 1.125).to(p.dtype))

    optimizer = CopyOptimizer()
    handle = lifecycle.install_mxfp4_pack_cache_lifecycle(cached, optimizer)
    assert handle["strategy"] == "generation"
    first = run(cached, data)
    packed_id, stamp = id(cached._packed_weights), cached._packed_stamp
    run(cached, data)
    assert id(cached._packed_weights) == packed_id and cached._packed_stamp == stamp
    versions = tuple(p._version for p in cached.parameters())
    optimizer.step()
    assert tuple(p._version for p in cached.parameters()) == versions
    assert cached._pack_generation != stamp[0]
    updated = run(cached, data)
    assert cached._packed_stamp != stamp
    assert not torch.equal(first, updated)
    uncached = model_with_buffers()
    uncached.load_state_dict(cached.state_dict())
    uncached._pack_cache_enabled = False
    fresh = run(uncached, data)
    torch.testing.assert_close(updated, fresh, rtol=0, atol=0)
    # Loading a root checkpoint also invalidates the generation-stamped cache.
    old_generation = cached._pack_generation
    cached.load_state_dict(copy.deepcopy(cached.state_dict()))
    assert cached._pack_generation > old_generation
