###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

from unittest.mock import patch

import pytest
import torch

DEVICE = "cuda"
_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ── Pure-PyTorch references (test-only) ─────────────────────────────────────


def _topk_softmax_reference(logits, k, softmax_first=True):
    """Reference: softmax then top-k selection."""
    if softmax_first:
        probs = torch.softmax(logits.float(), dim=-1)
    else:
        probs = logits.float()
    weights, indices = probs.topk(k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, indices


def _permute_reference(tokens, indices, num_experts):
    """Reference: gather tokens by expert, return sort order."""
    num_tokens, k = indices.shape
    flat_indices = indices.reshape(-1)
    expert_order = torch.argsort(flat_indices, stable=True)
    token_ids = torch.arange(num_tokens, device=tokens.device).unsqueeze(1).expand(-1, k).reshape(-1)
    sorted_token_ids = token_ids[expert_order]
    return sorted_token_ids, expert_order


def _unpermute_reference(expert_output, sort_order, num_tokens, k):
    """Reference: unsort and average across top-k experts."""
    hidden_size = expert_output.shape[-1]
    unsort = torch.argsort(sort_order)
    unsorted = expert_output[unsort]
    return unsorted.reshape(num_tokens, k, hidden_size).sum(dim=1)


# ── Assertion / probe tests ────────────────────────────────────────────────


class TestFusedTopKAssertions:
    """Test fused_topk AITER-only dispatch."""

    def test_requires_aiter_and_cuda(self):
        from lumen.ops.moe.fused_routing import fused_topk

        logits = torch.randn(32, 8)
        with patch("lumen.ops.moe.fused_routing._probe_aiter_moe_topk_softmax", return_value=False):
            with pytest.raises(AssertionError):
                fused_topk(logits, k=2)

    def test_rejects_cpu_tensors(self):
        from lumen.ops.moe.fused_routing import fused_topk

        logits = torch.randn(32, 8)
        with patch("lumen.ops.moe.fused_routing._probe_aiter_moe_topk_softmax", return_value=True):
            with pytest.raises(AssertionError):
                fused_topk(logits, k=2)


class TestFusedPermuteAssertions:
    """Test fused_permute AITER-only dispatch."""

    def test_requires_aiter_and_cuda(self):
        from lumen.ops.moe.fused_routing import fused_permute

        tokens = torch.randn(16, 64)
        indices = torch.zeros(16, 2, dtype=torch.long)
        weights = torch.ones(16, 2)
        with patch("lumen.ops.moe.fused_routing._probe_aiter_moe_sorting", return_value=False):
            with pytest.raises(AssertionError):
                fused_permute(tokens, indices, weights, num_experts=4)

    def test_signature_no_use_aiter_param(self):
        import inspect

        from lumen.ops.moe.fused_routing import fused_permute

        sig = inspect.signature(fused_permute)
        assert "block_size" in sig.parameters
        assert "use_aiter" not in sig.parameters


class TestFusedUnpermuteAssertions:
    """Test fused_unpermute AITER-only dispatch."""

    def test_requires_aiter_and_cuda(self):
        from lumen.ops.moe.fused_routing import fused_unpermute

        expert_output = torch.randn(32, 64)
        sort_order = torch.arange(32)
        with patch("lumen.ops.moe.fused_routing._probe_aiter_moe_sum", return_value=False):
            with pytest.raises(AssertionError):
                fused_unpermute(expert_output, sort_order, num_tokens=16, k=2)


class TestAiterProbes:
    """Verify probe functions return bool."""

    def test_routing_probes_return_bool(self):
        from lumen.ops.dispatch import (
            _probe_aiter_moe_sorting,
            _probe_aiter_moe_sum,
            _probe_aiter_moe_topk_softmax,
        )

        assert isinstance(_probe_aiter_moe_topk_softmax(), bool)
        assert isinstance(_probe_aiter_moe_sorting(), bool)
        assert isinstance(_probe_aiter_moe_sum(), bool)

    def test_triton_moe_probes_return_bool(self):
        from lumen.ops.dispatch import (
            _probe_aiter_triton_fused_moe,
            _probe_aiter_triton_moe_align,
        )

        assert isinstance(_probe_aiter_triton_moe_align(), bool)
        assert isinstance(_probe_aiter_triton_fused_moe(), bool)


# ── Numerical correctness tests (require CUDA + AITER) ─────────────────────


@_CUDA
class TestFusedTopKCorrectness:
    """Numerical correctness for fused_topk."""

    @pytest.mark.parametrize(
        "num_tokens,num_experts,k",
        [
            (16, 8, 2),
            (64, 4, 1),
            (32, 16, 4),
        ],
    )
    def test_matches_reference(self, num_tokens, num_experts, k):
        from lumen.ops.moe.fused_routing import fused_topk

        logits = torch.randn(num_tokens, num_experts, device=DEVICE, dtype=torch.float32)
        weights, indices = fused_topk(logits, k=k)
        ref_weights, ref_indices = _topk_softmax_reference(logits, k)

        assert weights.shape == (num_tokens, k)
        assert indices.shape == (num_tokens, k)

        for i in range(num_tokens):
            actual_set = set(indices[i].tolist())
            expected_set = set(ref_indices[i].tolist())
            assert actual_set == expected_set, f"Token {i}: got experts {actual_set}, expected {expected_set}"

        actual_sorted = torch.sort(indices, dim=-1)
        ref_sorted = torch.sort(ref_indices, dim=-1)
        torch.testing.assert_close(
            weights.gather(1, actual_sorted.indices),
            ref_weights.gather(1, ref_sorted.indices),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_weights_sum_to_one(self):
        from lumen.ops.moe.fused_routing import fused_topk

        logits = torch.randn(64, 8, device=DEVICE, dtype=torch.float32)
        weights, _ = fused_topk(logits, k=2)

        row_sums = weights.sum(dim=-1)
        torch.testing.assert_close(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_weights_are_positive(self):
        from lumen.ops.moe.fused_routing import fused_topk

        logits = torch.randn(32, 8, device=DEVICE)
        weights, _ = fused_topk(logits, k=2)
        assert (weights > 0).all()

    def test_indices_in_range(self):
        from lumen.ops.moe.fused_routing import fused_topk

        num_experts = 8
        logits = torch.randn(32, num_experts, device=DEVICE)
        _, indices = fused_topk(logits, k=2)
        assert (indices >= 0).all()
        assert (indices < num_experts).all()


@_CUDA
class TestFusedPermuteCorrectness:
    """Numerical correctness for fused_permute."""

    def test_output_shapes(self):
        from lumen.ops.moe.fused_routing import fused_permute

        num_tokens, hidden, k, num_experts = 16, 64, 2, 4
        tokens = torch.randn(num_tokens, hidden, device=DEVICE)
        indices = torch.randint(0, num_experts, (num_tokens, k), device=DEVICE, dtype=torch.int64)
        weights = torch.ones(num_tokens, k, device=DEVICE)

        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = fused_permute(
            tokens, indices, weights, num_experts=num_experts
        )

        assert sorted_ids.dtype == torch.int32
        assert sorted_weights.dtype == torch.float32
        assert sorted_expert_ids.dtype == torch.int32
        assert num_valid_ids.dtype == torch.int32
        assert moe_buf.shape == (num_tokens, hidden)

    def test_all_tokens_covered(self):
        """Every token should appear in sorted output for each selected expert."""
        from lumen.ops.moe.fused_routing import fused_permute

        num_tokens, hidden, k, num_experts = 32, 64, 2, 8
        tokens = torch.randn(num_tokens, hidden, device=DEVICE)
        indices = torch.randint(0, num_experts, (num_tokens, k), device=DEVICE, dtype=torch.int64)
        weights = torch.ones(num_tokens, k, device=DEVICE)

        sorted_ids, _, _, num_valid_ids, _ = fused_permute(tokens, indices, weights, num_experts=num_experts)

        total_valid = num_valid_ids[0].item()
        valid_ids = sorted_ids[:total_valid].cpu()
        actual_tokens = set(valid_ids.tolist())

        assert (
            len(actual_tokens) >= num_tokens
        ), f"Only {len(actual_tokens)} unique token slots found, expected >= {num_tokens}"


@_CUDA
class TestFusedUnpermuteCorrectness:
    """Numerical correctness for fused_unpermute."""

    def test_output_shape(self):
        from lumen.ops.moe.fused_routing import fused_unpermute

        num_tokens, k, hidden = 16, 2, 64
        expert_output = torch.randn(num_tokens * k, hidden, device=DEVICE)
        sort_order = torch.randperm(num_tokens * k, device=DEVICE)

        out = fused_unpermute(expert_output, sort_order, num_tokens, k)
        assert out.shape == (num_tokens, hidden)

    def test_matches_reference_with_identity_permutation(self):
        """When sort_order is identity, moe_sum should reduce the k dim."""
        from lumen.ops.moe.fused_routing import fused_unpermute

        num_tokens, k, hidden = 8, 2, 32
        expert_output = torch.randn(num_tokens * k, hidden, device=DEVICE, dtype=torch.float32)
        sort_order = torch.arange(num_tokens * k, device=DEVICE)

        out = fused_unpermute(expert_output, sort_order, num_tokens, k)
        ref = _unpermute_reference(expert_output, sort_order, num_tokens, k)

        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_matches_reference_with_random_permutation(self):
        from lumen.ops.moe.fused_routing import fused_unpermute

        num_tokens, k, hidden = 16, 2, 64
        expert_output = torch.randn(num_tokens * k, hidden, device=DEVICE, dtype=torch.float32)
        sort_order = torch.randperm(num_tokens * k, device=DEVICE)

        out = fused_unpermute(expert_output, sort_order, num_tokens, k)
        ref = _unpermute_reference(expert_output, sort_order, num_tokens, k)

        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
