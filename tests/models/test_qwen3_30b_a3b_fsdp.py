"""CPU tests for the Qwen3-30B-A3B Transformers/FSDP implementation."""

import argparse
import copy
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "lumen"
    / "models"
    / "qwen3_30b_a3b"
    / "fsdp"
    / "pretrain.py"
)
SPEC = importlib.util.spec_from_file_location("qwen3_fsdp_pretrain_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
EPShardedMoeBlock = MODULE.EPShardedMoeBlock
SonicLocalExperts = MODULE._SonicLocalExperts
create_parallel_groups = MODULE.create_parallel_groups
parallel_rank_layout = MODULE._parallel_rank_layout


def _sonic_blockwise_fp8_supported() -> bool:
    try:
        from aiter.ops.triton._triton_kernels.moe.sonicmoe.grouped_gemm_triton import (
            grouped_gemm as sonic_grouped_gemm,
        )
    except ImportError:
        return False
    return "A_scale" in inspect.signature(sonic_grouped_gemm).parameters


class _FakeGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 4
        self.top_k = 2
        self.norm_topk_prob = True
        self.weight = nn.Parameter(torch.randn(4, 6))

    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight)
        scores = F.softmax(logits, dim=-1, dtype=torch.float32)
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return logits, weights.to(hidden_states.dtype), indices


class _FakeExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 4
        self.gate_up_proj = nn.Parameter(torch.randn(4, 10, 6))
        self.down_proj = nn.Parameter(torch.randn(4, 6, 5))
        self.act_fn = F.silu


class _FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _FakeGate()
        self.experts = _FakeExperts()

    def forward(self, hidden_states):
        shape = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, shape[-1])
        _, weights, indices = self.gate(hidden_flat)
        output = torch.zeros_like(hidden_flat)
        for expert_id in range(self.experts.num_experts):
            token_ids, slots = torch.where(indices == expert_id)
            if token_ids.numel() == 0:
                continue
            gate, up = F.linear(
                hidden_flat[token_ids],
                self.experts.gate_up_proj[expert_id],
            ).chunk(2, dim=-1)
            expert_output = F.linear(
                F.silu(gate) * up,
                self.experts.down_proj[expert_id],
            )
            output.index_add_(
                0,
                token_ids,
                expert_output * weights[token_ids, slots].unsqueeze(-1),
            )
        return output.reshape(shape)


def test_ep_sharded_moe_ep1_matches_huggingface_layout():
    torch.manual_seed(7)
    reference = _FakeBlock()
    sharded_source = copy.deepcopy(reference)
    sharded = EPShardedMoeBlock(sharded_source, ep_rank=0, ep_size=1, ep_group=None)

    reference_input = torch.randn(2, 3, 6, requires_grad=True)
    sharded_input = reference_input.detach().clone().requires_grad_(True)
    reference_output = reference(reference_input)
    local_expert_calls = []
    hook = sharded.local_experts.register_forward_pre_hook(
        lambda _module, _args: local_expert_calls.append(1)
    )
    sharded_output = sharded(sharded_input)
    hook.remove()

    torch.testing.assert_close(sharded_output, reference_output)
    assert len(local_expert_calls) == 1

    reference_output.square().sum().backward()
    sharded_output.square().sum().backward()
    torch.testing.assert_close(
        sharded_input.grad, reference_input.grad, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(sharded.gate.weight.grad, reference.gate.weight.grad)
    sharded_gate_up_grad = torch.stack(
        [expert.gate_up_proj.weight.grad for expert in sharded.local_experts.experts]
    )
    sharded_down_grad = torch.stack(
        [expert.down_proj.weight.grad for expert in sharded.local_experts.experts]
    )
    torch.testing.assert_close(
        sharded_gate_up_grad,
        reference.experts.gate_up_proj.grad,
    )
    torch.testing.assert_close(
        sharded_down_grad,
        reference.experts.down_proj.grad,
    )


def test_ep_sharded_moe_dispatch_overlap_is_noop_for_ep1():
    torch.manual_seed(9)
    reference = _FakeBlock()
    sharded = EPShardedMoeBlock(
        copy.deepcopy(reference),
        ep_rank=0,
        ep_size=1,
        ep_group=None,
    )
    sharded.enable_lumen_moe_dispatch_overlap()

    reference_input = torch.randn(2, 3, 6, requires_grad=True)
    sharded_input = reference_input.detach().clone().requires_grad_(True)
    reference_output = reference(reference_input)
    sharded_output = sharded(sharded_input)

    torch.testing.assert_close(sharded_output, reference_output)
    reference_output.sum().backward()
    sharded_output.sum().backward()
    torch.testing.assert_close(sharded_input.grad, reference_input.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("global_expert_layout", [False, True])
def test_sonic_multistream_ep1_matches_huggingface_layout(
    monkeypatch,
    global_expert_layout,
):
    monkeypatch.setenv("SONIC_MOE_GEMM_BACKEND", "triton")
    monkeypatch.setenv("SONIC_MOE_GROUPED_GEMM_BACKEND", "multistream")
    torch.manual_seed(11)
    reference = _FakeBlock().cuda()
    sharded = EPShardedMoeBlock(
        copy.deepcopy(reference),
        ep_rank=0,
        ep_size=1,
        ep_group=None,
        expert_backend="sonic",
    )
    if global_expert_layout:
        sharded.enable_lumen_moe_global_expert_layout()

    reference_input = torch.randn(2, 3, 6, device="cuda", requires_grad=True)
    sharded_input = reference_input.detach().clone().requires_grad_(True)
    reference_output = reference(reference_input)
    sharded_output = sharded(sharded_input)
    torch.testing.assert_close(sharded_output, reference_output)

    reference_output.square().sum().backward()
    sharded_output.square().sum().backward()
    torch.testing.assert_close(
        sharded_input.grad, reference_input.grad, rtol=1e-4, atol=1e-4
    )
    reference_gate, reference_up = reference.experts.gate_up_proj.grad.chunk(2, dim=1)
    torch.testing.assert_close(
        sharded.local_experts.w1.grad,
        torch.stack((reference_gate, reference_up), dim=2)
        .flatten(1, 2)
        .transpose(1, 2),
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        sharded.local_experts.w2.grad,
        reference.experts.down_proj.grad.transpose(1, 2),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
def test_sonic_triton_forward_and_gradients(monkeypatch):
    pytest.importorskip("aiter.ops.triton.sonicmoe")
    monkeypatch.setenv("SONIC_MOE_GEMM_BACKEND", "triton")
    torch.manual_seed(17)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    experts = _FakeExperts()
    experts.gate_up_proj = nn.Parameter(
        torch.randn(4, 128, 128, device=device, dtype=dtype) * 0.02
    )
    experts.down_proj = nn.Parameter(
        torch.randn(4, 128, 64, device=device, dtype=dtype) * 0.02
    )
    sonic = SonicLocalExperts(experts, 0, 4)

    hidden = torch.randn(19, 128, device=device, dtype=dtype, requires_grad=True)
    expert_ids = torch.tensor(
        [0, 2, 1, 3, 1, 0, 2, 2, 3, 0, 1, 3, 2, 0, 1, 1, 3, 2, 0],
        device=device,
    )
    weights = torch.rand(19, device=device, dtype=dtype, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weights = weights.detach().clone().requires_grad_(True)
    reference_w1 = sonic.w1.detach().clone().requires_grad_(True)
    reference_w2 = sonic.w2.detach().clone().requires_grad_(True)

    reference_output = torch.empty_like(reference_hidden)
    for expert_id in range(4):
        positions = torch.where(expert_ids == expert_id)[0]
        gate = F.linear(
            reference_hidden[positions], reference_w1[expert_id, :, 0::2].T
        )
        up = F.linear(
            reference_hidden[positions], reference_w1[expert_id, :, 1::2].T
        )
        result = F.linear(F.silu(gate) * up, reference_w2[expert_id].T)
        reference_output[positions] = result * reference_weights[positions, None]
    sonic_output = sonic.forward_all(hidden, expert_ids, weights)
    torch.testing.assert_close(sonic_output, reference_output, rtol=0.05, atol=0.02)

    output_gradient = torch.randn_like(sonic_output)
    sonic_output.backward(output_gradient)
    reference_output.backward(output_gradient)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0.08, atol=0.03)
    torch.testing.assert_close(weights.grad, reference_weights.grad, rtol=0.08, atol=0.03)
    torch.testing.assert_close(sonic.w1.grad, reference_w1.grad, rtol=0.08, atol=0.03)
    torch.testing.assert_close(sonic.w2.grad, reference_w2.grad, rtol=0.08, atol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
def test_sonic_fp8_forward_and_gradients():
    pytest.importorskip("aiter.ops.triton.sonicmoe")
    if not _sonic_blockwise_fp8_supported():
        pytest.skip("AITER Sonic grouped GEMM has no in-kernel blockwise FP8 scales")
    from lumen.quantize.config import _get_float8_e4m3

    torch.manual_seed(17)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    experts = _FakeExperts()
    experts.gate_up_proj = nn.Parameter(
        torch.randn(4, 128, 128, device=device, dtype=dtype) * 0.02
    )
    experts.down_proj = nn.Parameter(
        torch.randn(4, 128, 64, device=device, dtype=dtype) * 0.02
    )
    sonic = SonicLocalExperts(experts, 0, 4)
    sonic.enable_fp8(scaling_type="blockwise2d", fp8_dtype=_get_float8_e4m3(), block_size=128)

    hidden = torch.randn(19, 128, device=device, dtype=dtype, requires_grad=True)
    expert_ids = torch.tensor(
        [0, 2, 1, 3, 1, 0, 2, 2, 3, 0, 1, 3, 2, 0, 1, 1, 3, 2, 0],
        device=device,
    )
    weights = torch.rand(19, device=device, dtype=dtype, requires_grad=True)
    reference_w1 = sonic.w1.detach().clone()
    reference_w2 = sonic.w2.detach().clone()

    reference = torch.empty(19, 128, device=device, dtype=dtype)
    for expert_id in range(4):
        positions = torch.where(expert_ids == expert_id)[0]
        tokens = hidden[positions].detach()
        gate = tokens @ reference_w1[expert_id, :, 0::2]
        up = tokens @ reference_w1[expert_id, :, 1::2]
        activated = (F.silu(gate.float()) * up.float()).to(dtype)
        reference[positions] = (activated @ reference_w2[expert_id]) * weights[
            positions
        ].detach().unsqueeze(-1)

    sonic_output = sonic.forward_all(hidden, expert_ids, weights)
    assert torch.isfinite(sonic_output.float()).all()
    # FP8 expert GEMMs must reproduce the BF16 SwiGLU, so a gate/up layout
    # mismatch cannot hide behind "output is finite".
    cosine = F.cosine_similarity(
        sonic_output.detach().float().flatten(), reference.float().flatten(), dim=0
    )
    assert cosine > 0.99, f"FP8 expert MLP vs BF16 reference cosine {cosine:.4f}"

    sonic_output.backward(torch.randn_like(sonic_output))
    torch.cuda.synchronize()
    assert torch.isfinite(hidden.grad.float()).all()
    assert torch.isfinite(sonic.w1.grad.float()).all()
    assert torch.isfinite(sonic.w2.grad.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
def test_sonic_fp8_zero_expert_counts():
    pytest.importorskip("aiter.ops.triton.sonicmoe")
    if not _sonic_blockwise_fp8_supported():
        pytest.skip("AITER Sonic grouped GEMM has no in-kernel blockwise FP8 scales")
    from lumen.quantize.config import _get_float8_e4m3

    torch.manual_seed(3)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    experts = _FakeExperts()
    experts.gate_up_proj = nn.Parameter(
        torch.randn(4, 128, 128, device=device, dtype=dtype) * 0.02
    )
    experts.down_proj = nn.Parameter(
        torch.randn(4, 128, 64, device=device, dtype=dtype) * 0.02
    )
    sonic = SonicLocalExperts(experts, 0, 4)
    sonic.enable_fp8(scaling_type="blockwise2d", fp8_dtype=_get_float8_e4m3(), block_size=128)

    hidden = torch.randn(16, 128, device=device, dtype=dtype, requires_grad=True)
    expert_ids = torch.zeros(16, device=device, dtype=torch.int64)
    weights = torch.ones(16, device=device, dtype=dtype)
    out = sonic.forward_all(hidden, expert_ids, weights)
    out.float().sum().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(hidden.grad.float()).all()
    assert sonic.w1.grad[1].abs().sum() == 0


def test_enable_lumen_rejects_te_grouped_fp8():
    args = argparse.Namespace(mode="fp8_blockwise2d", expert_backend="te_grouped")
    with pytest.raises(ValueError, match="does not cover TE grouped"):
        MODULE._enable_lumen(nn.Linear(2, 2), args)


def test_sonic_enable_fp8_rejects_non_blockwise2d():
    experts = _FakeExperts()
    sonic = SonicLocalExperts(experts, 0, 4)
    with pytest.raises(ValueError, match="blockwise2d"):
        sonic.enable_fp8(scaling_type="delayed")


def test_sonic_backend_rejects_modulelist_experts():
    class _ListBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = _FakeGate()
            self.experts = nn.ModuleList([nn.Linear(6, 6) for _ in range(4)])

    with pytest.raises(TypeError, match="packed HF expert"):
        EPShardedMoeBlock(
            _ListBlock(),
            ep_rank=0,
            ep_size=1,
            ep_group=None,
            expert_backend="sonic",
        )


def test_local_weight_unwraps_to_local():
    class _Sharded:
        def to_local(self):
            return torch.ones(2, 2)

    dense = torch.zeros(2, 2)
    assert MODULE._local_weight(dense).equal(dense)
    assert torch.equal(MODULE._local_weight(_Sharded()), torch.ones(2, 2))


def test_single_rank_parallel_groups_do_not_require_distributed_init():
    groups = create_parallel_groups(ep_size=1)
    assert groups.ep_group is None
    assert groups.dp_group is None
    assert groups.expert_dp_group is None
    assert groups.ep_rank == 0
    assert groups.dp_rank == 0
    assert groups.expert_dp_rank == 0
    assert groups.ep_size == 1
    assert groups.dp_size == 1
    assert groups.expert_dp_size == 1


def test_parallel_rank_layout_overlaps_dense_dp_and_ep():
    layout = parallel_rank_layout(
        world_size=8,
        global_rank=5,
        ep_size=8,
        dp_size=8,
    )

    assert layout.dp_ranks == tuple(range(8))
    assert layout.ep_ranks == tuple(range(8))
    assert layout.expert_dp_ranks == (5,)
    assert layout.dp_rank == 5
    assert layout.ep_rank == 5
    assert layout.expert_dp_rank == 0
    assert layout.expert_dp_size == 1


def test_parallel_rank_layout_builds_expert_dp_columns():
    layout = parallel_rank_layout(
        world_size=16,
        global_rank=13,
        ep_size=8,
        dp_size=16,
    )

    assert layout.dp_ranks == tuple(range(16))
    assert layout.ep_ranks == tuple(range(8, 16))
    assert layout.expert_dp_ranks == (5, 13)
    assert layout.dp_rank == 13
    assert layout.ep_rank == 5
    assert layout.expert_dp_rank == 1
    assert layout.expert_dp_size == 2


def test_parallel_rank_layout_rejects_non_overlapping_dense_dp():
    with pytest.raises(ValueError, match="must equal world_size"):
        parallel_rank_layout(
            world_size=8,
            global_rank=0,
            ep_size=8,
            dp_size=1,
        )
