###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

import torch


class TestFusedTopK:

    def test_shape(self):
        from lumen.ops.moe.fused_routing import fused_topk

        logits = torch.randn(32, 8)  # 32 tokens, 8 experts
        weights, indices = fused_topk(logits, k=2)
        assert weights.shape == (32, 2)
        assert indices.shape == (32, 2)

    def test_weights_sum_to_one(self):
        from lumen.ops.moe.fused_routing import fused_topk

        logits = torch.randn(32, 8)
        weights, _ = fused_topk(logits, k=2)
        sums = weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(32), atol=1e-5, rtol=1e-5)

    def test_indices_in_range(self):
        from lumen.ops.moe.fused_routing import fused_topk

        logits = torch.randn(32, 8)
        _, indices = fused_topk(logits, k=3)
        assert (indices >= 0).all() and (indices < 8).all()

    def test_weights_positive(self):
        from lumen.ops.moe.fused_routing import fused_topk

        logits = torch.randn(32, 8)
        weights, _ = fused_topk(logits, k=2)
        assert (weights > 0).all()


class TestFusedPermuteUnpermute:

    def test_round_trip(self):
        from lumen.ops.moe.fused_routing import fused_permute, fused_topk, fused_unpermute

        num_tokens, hidden, num_experts, k = 16, 64, 4, 2
        tokens = torch.randn(num_tokens, hidden)
        logits = torch.randn(num_tokens, num_experts)

        weights, indices = fused_topk(logits, k)
        permuted, sort_order, offsets = fused_permute(tokens, indices, weights, num_experts)

        assert permuted.shape == (num_tokens * k, hidden)
        assert offsets.shape == (num_experts + 1,)
        assert offsets[-1] == num_tokens * k

        output = fused_unpermute(permuted, sort_order, num_tokens, k)
        assert output.shape == (num_tokens, hidden)

    def test_expert_offsets_partition(self):
        from lumen.ops.moe.fused_routing import fused_permute, fused_topk

        num_tokens, hidden, num_experts, k = 32, 64, 8, 2
        tokens = torch.randn(num_tokens, hidden)
        logits = torch.randn(num_tokens, num_experts)

        weights, indices = fused_topk(logits, k)
        _, _, offsets = fused_permute(tokens, indices, weights, num_experts)

        # Offsets should be monotonically non-decreasing
        for i in range(len(offsets) - 1):
            assert offsets[i] <= offsets[i + 1]
        assert offsets[0] == 0
        assert offsets[-1] == num_tokens * k

    def test_weighted_output(self):
        """Verify weights are applied during permute."""
        from lumen.ops.moe.fused_routing import fused_permute, fused_unpermute

        num_tokens, hidden, num_experts, k = 4, 8, 2, 1
        tokens = torch.ones(num_tokens, hidden)
        indices = torch.zeros(num_tokens, k, dtype=torch.long)
        weights = torch.full((num_tokens, k), 0.5)

        permuted, sort_order, _ = fused_permute(tokens, indices, weights, num_experts)
        output = fused_unpermute(permuted, sort_order, num_tokens, k)

        expected = torch.full((num_tokens, hidden), 0.5)
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6)
