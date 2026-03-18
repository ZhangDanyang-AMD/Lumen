###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

import torch


class TestOnlineSoftmaxUpdate:
    """Test the online softmax accumulation utility."""

    def test_two_equal_chunks(self):
        from lumen.ops.attention.attention_with_cp_p2p import _online_softmax_update

        B, S, H, D = 2, 8, 4, 64
        out1 = torch.randn(B, S, H, D)
        out2 = torch.randn(B, S, H, D)
        lse1 = torch.randn(B, S, H)
        lse2 = torch.randn(B, S, H)

        merged_out, merged_lse = _online_softmax_update(out1, lse1, out2, lse2)
        assert merged_out.shape == (B, S, H, D)
        assert merged_lse.shape == (B, S, H)
        assert torch.isfinite(merged_out).all()
        assert torch.isfinite(merged_lse).all()

    def test_identity_with_neg_inf_lse(self):
        """If one chunk has -inf LSE, the other dominates."""
        from lumen.ops.attention.attention_with_cp_p2p import _online_softmax_update

        B, S, H, D = 1, 4, 2, 32
        out1 = torch.randn(B, S, H, D)
        lse1 = torch.randn(B, S, H)
        out2 = torch.randn(B, S, H, D)
        lse2 = torch.full((B, S, H), float("-inf"))

        merged_out, merged_lse = _online_softmax_update(out1, lse1, out2, lse2)
        torch.testing.assert_close(merged_out, out1, atol=1e-5, rtol=1e-5)

    def test_commutativity(self):
        from lumen.ops.attention.attention_with_cp_p2p import _online_softmax_update

        B, S, H, D = 1, 4, 2, 32
        out1 = torch.randn(B, S, H, D)
        lse1 = torch.randn(B, S, H)
        out2 = torch.randn(B, S, H, D)
        lse2 = torch.randn(B, S, H)

        r1_out, r1_lse = _online_softmax_update(out1, lse1, out2, lse2)
        r2_out, r2_lse = _online_softmax_update(out2, lse2, out1, lse1)
        torch.testing.assert_close(r1_out, r2_out, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(r1_lse, r2_lse, atol=1e-5, rtol=1e-5)


class TestRingSendRecvKV:
    """Test P2P send/recv (mocked)."""

    def test_function_exists(self):
        from lumen.ops.attention.attention_with_cp_p2p import _ring_send_recv_kv

        assert callable(_ring_send_recv_kv)


class TestAttentionCPP2P:
    """Test ring attention autograd function."""

    def test_class_exists(self):
        from lumen.ops.attention.attention_with_cp_p2p import AttentionCPP2PFunction

        assert hasattr(AttentionCPP2PFunction, "forward")
        assert hasattr(AttentionCPP2PFunction, "backward")

    def test_api_function_exists(self):
        from lumen.ops.attention.attention_with_cp_p2p import attention_cp_p2p

        assert callable(attention_cp_p2p)
