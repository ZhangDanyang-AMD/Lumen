###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

import pytest
import torch


def _make_cos_sin(seq_len, head_dim, device="cpu"):
    """Generate cos/sin frequencies for testing."""
    pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
    dim = torch.arange(0, head_dim, 2, device=device).float()
    freq = 1.0 / (10000.0 ** (dim / head_dim))
    angles = pos * freq
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


class TestApplyRotaryPosEmb:
    """Test PyTorch fallback RoPE implementation."""

    def test_shape_preserved(self):
        from lumen.ops.rope import _apply_rotary_pos_emb_torch

        B, S, H, D = 2, 16, 4, 64
        x = torch.randn(B, H, S, D)
        cos, sin = _make_cos_sin(S, D)
        out = _apply_rotary_pos_emb_torch(x, cos, sin)
        assert out.shape == x.shape

    def test_identity_at_zero_angles(self):
        from lumen.ops.rope import _apply_rotary_pos_emb_torch

        B, S, H, D = 2, 16, 4, 64
        x = torch.randn(B, H, S, D)
        cos = torch.ones(S, D // 2)
        sin = torch.zeros(S, D // 2)
        out = _apply_rotary_pos_emb_torch(x, cos, sin)
        # With cos=1, sin=0: x1*1 - x2*0 = x1, x2*1 + x1*0 = x2 => identity
        torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)

    def test_norm_preserved(self):
        """RoPE is a rotation — norms should be preserved."""
        from lumen.ops.rope import _apply_rotary_pos_emb_torch

        B, S, H, D = 2, 16, 4, 64
        x = torch.randn(B, H, S, D)
        cos, sin = _make_cos_sin(S, D)
        out = _apply_rotary_pos_emb_torch(x, cos, sin)
        x_norm = x.norm(dim=-1)
        out_norm = out.norm(dim=-1)
        torch.testing.assert_close(x_norm, out_norm, atol=1e-5, rtol=1e-5)


class TestFusedRoPE:
    """Test fused_rope function (uses PyTorch fallback in test env)."""

    def test_shape(self):
        from lumen.ops.rope import fused_rope

        B, S, H, D = 2, 16, 4, 64
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        cos, sin = _make_cos_sin(S, D)
        q_rot, k_rot = fused_rope(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_correctness_vs_reference(self):
        from lumen.ops.rope import _apply_rotary_pos_emb_torch, fused_rope

        B, S, H, D = 2, 16, 4, 64
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        cos, sin = _make_cos_sin(S, D)

        q_rot, k_rot = fused_rope(q, k, cos, sin)
        q_ref = _apply_rotary_pos_emb_torch(q, cos, sin)
        k_ref = _apply_rotary_pos_emb_torch(k, cos, sin)

        torch.testing.assert_close(q_rot, q_ref, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(k_rot, k_ref, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("head_dim", [32, 64, 128])
    def test_various_head_dims(self, head_dim):
        from lumen.ops.rope import fused_rope

        q = torch.randn(2, 4, 16, head_dim)
        k = torch.randn(2, 4, 16, head_dim)
        cos, sin = _make_cos_sin(16, head_dim)
        q_rot, k_rot = fused_rope(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert torch.isfinite(q_rot).all()
