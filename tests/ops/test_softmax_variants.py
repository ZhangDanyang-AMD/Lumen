###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

import pytest
import torch


class TestApplySoftmaxVariant:

    def test_vanilla_matches_torch(self):
        from lumen.ops.attention.attention import apply_softmax_variant

        logits = torch.randn(2, 4, 8, 16)
        out = apply_softmax_variant(logits, "vanilla")
        ref = torch.softmax(logits, dim=-1)
        torch.testing.assert_close(out, ref)

    def test_vanilla_sums_to_one(self):
        from lumen.ops.attention.attention import apply_softmax_variant

        logits = torch.randn(2, 4, 8, 16)
        out = apply_softmax_variant(logits, "vanilla")
        sums = out.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

    def test_off_by_one_sums_less_than_one(self):
        from lumen.ops.attention.attention import apply_softmax_variant

        logits = torch.randn(2, 4, 8, 16)
        out = apply_softmax_variant(logits, "off_by_one")
        sums = out.sum(dim=-1)
        # With +1 in denominator, sum should be < 1
        assert (sums < 1.0 + 1e-5).all()

    def test_off_by_one_shape(self):
        from lumen.ops.attention.attention import apply_softmax_variant

        logits = torch.randn(2, 4, 8, 16)
        out = apply_softmax_variant(logits, "off_by_one")
        assert out.shape == logits.shape

    def test_off_by_one_non_negative(self):
        from lumen.ops.attention.attention import apply_softmax_variant

        logits = torch.randn(2, 4, 8, 16)
        out = apply_softmax_variant(logits, "off_by_one")
        assert (out >= 0).all()

    def test_off_by_one_numerical_stability(self):
        from lumen.ops.attention.attention import apply_softmax_variant

        logits = torch.randn(2, 4, 8, 16) * 100  # large values
        out = apply_softmax_variant(logits, "off_by_one")
        assert torch.isfinite(out).all()

    def test_temperature_scaling(self):
        from lumen.ops.attention.attention import apply_softmax_variant

        logits = torch.randn(2, 4, 8, 16)
        out_normal = apply_softmax_variant(logits, "vanilla")
        out_hot = apply_softmax_variant(logits, "vanilla", temperature=0.5)
        # Lower temperature (0.5) scales logits down -> softer distribution -> higher entropy
        entropy_normal = -(out_normal * out_normal.log().clamp(min=-100)).sum(-1).mean()
        entropy_hot = -(out_hot * out_hot.log().clamp(min=-100)).sum(-1).mean()
        assert entropy_hot > entropy_normal

    def test_unknown_type_raises(self):
        from lumen.ops.attention.attention import apply_softmax_variant

        with pytest.raises(ValueError, match="Unknown softmax_type"):
            apply_softmax_variant(torch.randn(2, 4), "unknown")
