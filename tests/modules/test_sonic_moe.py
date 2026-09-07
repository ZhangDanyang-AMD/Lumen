from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from lumen.config import LumenConfig
from lumen.modules.sonic_moe import SonicMoEExperts


class _Linear(nn.Module):
    def __init__(self, out_features, in_features, value):
        super().__init__()
        self.weight = nn.Parameter(torch.full((out_features, in_features), value))


class _Expert(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.linear_fc1 = _Linear(8, 4, value)
        self.linear_fc2 = _Linear(4, 4, value + 10)


class _SequentialExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            add_bias_linear=False,
            gated_linear_unit=True,
            expert_tensor_parallel_size=1,
        )
        self.num_local_experts = 2
        self.local_experts = nn.ModuleList([_Expert(1.0), _Expert(2.0)])


class _GroupedLinear(nn.Module):
    def __init__(self, out_features, in_features, values):
        super().__init__()
        for index, value in enumerate(values):
            self.register_parameter(
                f"weight{index}",
                nn.Parameter(torch.full((out_features, in_features), value)),
            )


class _GroupedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            add_bias_linear=False,
            gated_linear_unit=True,
            expert_tensor_parallel_size=1,
        )
        self.num_local_experts = 2
        self.linear_fc1 = _GroupedLinear(8, 4, [3.0, 4.0])
        self.linear_fc2 = _GroupedLinear(4, 4, [13.0, 14.0])


class MoELayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _SequentialExperts()


def test_sonic_moe_packs_megatron_expert_weights():
    sonic = SonicMoEExperts(_SequentialExperts())

    assert sonic.w1.shape == (2, 4, 8)
    assert sonic.w2.shape == (2, 4, 4)
    assert sonic.w1.is_contiguous()
    assert sonic.w2.is_contiguous()
    torch.testing.assert_close(sonic.w1[0], torch.ones(4, 8))
    torch.testing.assert_close(sonic.w1[1], torch.full((4, 8), 2.0))
    torch.testing.assert_close(sonic.w2[0], torch.full((4, 4), 11.0))


def test_lumen_config_enable_replaces_megatron_moe():
    model = nn.Sequential(MoELayer())
    manager, returned = LumenConfig(scaling="none", sonic_moe=True).enable(model)

    assert manager is None
    assert returned is model
    assert isinstance(model[0].experts, SonicMoEExperts)


def test_sonic_loads_sequential_expert_checkpoint_keys():
    checkpoint_experts = _SequentialExperts()
    sonic = SonicMoEExperts(_SequentialExperts())
    sonic.w1.data.zero_()
    sonic.w2.data.zero_()

    sonic.load_state_dict(checkpoint_experts.state_dict(), strict=True)

    torch.testing.assert_close(
        sonic.w1,
        torch.stack(
            [
                expert.linear_fc1.weight
                for expert in checkpoint_experts.local_experts
            ]
        ).transpose(1, 2),
    )
    torch.testing.assert_close(
        sonic.w2,
        torch.stack(
            [
                expert.linear_fc2.weight
                for expert in checkpoint_experts.local_experts
            ]
        ).transpose(1, 2),
    )


def test_sonic_loads_grouped_expert_checkpoint_keys():
    checkpoint_experts = _GroupedExperts()
    sonic = SonicMoEExperts(_SequentialExperts())
    sonic.w1.data.zero_()
    sonic.w2.data.zero_()

    sonic.load_state_dict(checkpoint_experts.state_dict(), strict=True)

    torch.testing.assert_close(
        sonic.w1,
        torch.stack(
            [
                checkpoint_experts.linear_fc1.weight0,
                checkpoint_experts.linear_fc1.weight1,
            ]
        ).transpose(1, 2),
    )
    torch.testing.assert_close(
        sonic.w2,
        torch.stack(
            [
                checkpoint_experts.linear_fc2.weight0,
                checkpoint_experts.linear_fc2.weight1,
            ]
        ).transpose(1, 2),
    )


def test_sonic_sharded_state_dict_uses_sequential_checkpoint_layout():
    class _Group:
        def size(self):
            return 2

    sonic = SonicMoEExperts(_SequentialExperts())
    sonic.ep_group = _Group()

    sharded = sonic.sharded_state_dict(prefix="layers.0.experts.")
    w1_shards = sharded["layers.0.experts.w1"].build()
    w2_shards = sharded["layers.0.experts.w2"].build()

    assert len(w1_shards) == 2 * sonic.num_local_experts
    assert len(w2_shards) == sonic.num_local_experts
    assert {
        shard.key for shard in w1_shards
    } == {"layers.0.experts.experts.linear_fc1.weight"}
    assert {
        shard.key for shard in w2_shards
    } == {"layers.0.experts.experts.linear_fc2.weight"}
    torch.testing.assert_close(
        sharded["layers.0.experts.w1"].merge_fn(
            [shard.data for shard in w1_shards]
        ),
        sonic.w1,
    )
    torch.testing.assert_close(
        sharded["layers.0.experts.w2"].merge_fn(
            [shard.data for shard in w2_shards]
        ),
        sonic.w2,
    )


def test_sonic_rejects_legacy_python_blas_backend(monkeypatch):
    monkeypatch.setenv("SONIC_MOE_GEMM_BACKEND", "blas")
    with pytest.raises(ValueError, match="SONIC_MOE_GROUPED_GEMM_BACKEND"):
        SonicMoEExperts(_SequentialExperts())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_sonic_pre_routed_matches_general_routing():
    from aiter.ops.triton.sonicmoe import (
        SonicMoEActivationType,
        moe_general_routing_inputs,
        moe_pre_routed_inputs,
    )

    torch.manual_seed(123)
    device = torch.device("cuda")
    num_tokens, hidden_size, intermediate_size, num_experts = 64, 64, 64, 2
    counts = torch.tensor([32, 32], dtype=torch.int32, device=device)
    expert_indices = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int32, device=device), counts
    )
    token_indices = torch.arange(num_tokens, dtype=torch.int32, device=device)

    x_general = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device, requires_grad=True
    )
    x_pre_routed = x_general.detach().clone().requires_grad_(True)
    scores_general = torch.rand(num_tokens, dtype=torch.float32, device=device, requires_grad=True)
    scores_pre_routed = scores_general.detach().clone().requires_grad_(True)

    w1_general_storage = torch.randn(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    w2_general_storage = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    w1_pre_routed_storage = (
        w1_general_storage.detach().transpose(1, 2).contiguous().requires_grad_(True)
    )
    w2_pre_routed_storage = (
        w2_general_storage.detach().transpose(1, 2).contiguous().requires_grad_(True)
    )
    w1_general = w1_general_storage.permute(1, 2, 0)
    w2_general = w2_general_storage.permute(1, 2, 0)
    w1_pre_routed = w1_pre_routed_storage
    w2_pre_routed = w2_pre_routed_storage

    output_general, _ = moe_general_routing_inputs(
        x_general,
        scores_general,
        token_indices,
        expert_indices,
        w1_general,
        None,
        w2_general,
        None,
        num_experts,
        torch.cuda.current_stream().cuda_stream,
        SonicMoEActivationType.SWIGLU,
        False,
        True,
    )
    output_pre_routed, _ = moe_pre_routed_inputs(
        x_pre_routed,
        scores_pre_routed,
        counts.cpu(),
        w1_pre_routed,
        None,
        w2_pre_routed,
        None,
        torch.cuda.current_stream().cuda_stream,
        SonicMoEActivationType.SWIGLU,
        False,
        True,
    )

    torch.testing.assert_close(output_pre_routed, output_general)
    grad = torch.randn_like(output_general)
    output_general.backward(grad)
    output_pre_routed.backward(grad)
    torch.testing.assert_close(x_pre_routed.grad, x_general.grad)
    torch.testing.assert_close(scores_pre_routed.grad, scores_general.grad)
    torch.testing.assert_close(
        w1_pre_routed_storage.grad, w1_general_storage.grad.transpose(1, 2)
    )
    torch.testing.assert_close(
        w2_pre_routed_storage.grad, w2_general_storage.grad.transpose(1, 2)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_sonic_pre_routed_matches_torch_reference():
    from aiter.ops.triton.sonicmoe import (
        SonicMoEActivationType,
        moe_pre_routed_inputs,
    )

    torch.manual_seed(321)
    device = torch.device("cuda")
    hidden_size, intermediate_size, num_experts = 64, 64, 2
    counts = torch.tensor([32, 32], dtype=torch.int32)
    num_tokens = int(counts.sum())

    x_sonic = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device, requires_grad=True
    )
    x_reference = x_sonic.detach().clone().requires_grad_(True)
    scores_sonic = torch.rand(
        num_tokens, dtype=torch.float32, device=device, requires_grad=True
    )
    scores_reference = scores_sonic.detach().clone().requires_grad_(True)
    w1_sonic = (
        torch.randn(
            num_experts,
            hidden_size,
            2 * intermediate_size,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.02
    ).requires_grad_(True)
    w1_reference = w1_sonic.detach().clone().requires_grad_(True)
    w2_sonic = (
        torch.randn(
            num_experts,
            intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.02
    ).requires_grad_(True)
    w2_reference = w2_sonic.detach().clone().requires_grad_(True)

    output_sonic, _ = moe_pre_routed_inputs(
        x_sonic,
        scores_sonic,
        counts,
        w1_sonic,
        None,
        w2_sonic,
        None,
        torch.cuda.current_stream().cuda_stream,
        SonicMoEActivationType.SWIGLU,
        False,
        True,
    )

    outputs = []
    offset = 0
    for expert, count in enumerate(counts.tolist()):
        expert_input = x_reference[offset : offset + count]
        gate, up = (expert_input @ w1_reference[expert]).chunk(2, dim=-1)
        activated = torch.nn.functional.silu(gate) * up
        expert_output = activated @ w2_reference[expert]
        outputs.append(
            (
                expert_output
                * scores_reference[offset : offset + count].unsqueeze(-1)
            ).to(expert_output.dtype)
        )
        offset += count
    output_reference = torch.cat(outputs)

    torch.testing.assert_close(output_sonic, output_reference, rtol=2e-2, atol=2e-2)
    grad = torch.randn_like(output_sonic)
    output_sonic.backward(grad)
    output_reference.backward(grad)
    torch.testing.assert_close(x_sonic.grad, x_reference.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        scores_sonic.grad, scores_reference.grad, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(w1_sonic.grad, w1_reference.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(w2_sonic.grad, w2_reference.grad, rtol=2e-2, atol=2e-2)


def test_lumen_config_enable_allows_blockwise2d_with_sonic():
    model = nn.Sequential(MoELayer())
    manager, returned = LumenConfig(
        scaling="blockwise2d",
        block_size=128,
        sonic_moe=True,
    ).enable(model)

    assert returned is model
    assert manager is not None
    assert isinstance(model[0].experts, SonicMoEExperts)
    assert model[0].experts.scaling_type == "blockwise2d"
    assert model[0].experts.block_size == 128


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_sonic_fp8_blockwise2d_empty_expert_and_backward():
    hidden, intermediate = 128, 128

    class _WideLinear(nn.Module):
        def __init__(self, out_features, in_features):
            super().__init__()
            self.weight = nn.Parameter(
                torch.randn(out_features, in_features, dtype=torch.bfloat16)
            )

    class _WideExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_fc1 = _WideLinear(2 * intermediate, hidden)
            self.linear_fc2 = _WideLinear(hidden, intermediate)

    class _WideExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                add_bias_linear=False,
                gated_linear_unit=True,
                expert_tensor_parallel_size=1,
            )
            self.num_local_experts = 2
            self.local_experts = nn.ModuleList([_WideExpert(), _WideExpert()])

    sonic = SonicMoEExperts(_WideExperts()).cuda()
    sonic.enable_fp8(scaling_type="blockwise2d", block_size=128)
    hidden_states = torch.randn(
        32, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    counts = torch.tensor([32, 0], dtype=torch.int32)
    scores = torch.rand(32, device="cuda", dtype=torch.float32, requires_grad=True)

    output, extra = sonic(hidden_states, counts, scores)
    assert extra is None
    assert output.shape == (32, hidden)
    output.float().sum().backward()
    assert hidden_states.grad is not None
    assert sonic.w1.grad is not None
    assert sonic.w2.grad is not None
    assert torch.isfinite(output.float()).all()
    assert torch.isfinite(sonic.w1.grad.float()).all()
