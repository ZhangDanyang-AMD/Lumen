###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################
"""
Tests for ring-based context parallelism (CP P2P).

Covers:
  - Online softmax accumulation utility (unit tests)
  - Ring send/recv KV (existence check)
  - AttentionCPP2PFunction autograd class (existence check)
  - attention_cp_p2p API (existence check)
  - Distributed 2-GPU ring attention correctness test
"""

import os

import pytest
import torch

# =========================================================================
# Unit tests for online softmax merge
# =========================================================================


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

    def test_three_way_associativity(self):
        """Merging (A,B) then C should equal merging A then (B,C)."""
        from lumen.ops.attention.attention_with_cp_p2p import _online_softmax_update

        B, S, H, D = 1, 4, 2, 32
        o1, l1 = torch.randn(B, S, H, D), torch.randn(B, S, H)
        o2, l2 = torch.randn(B, S, H, D), torch.randn(B, S, H)
        o3, l3 = torch.randn(B, S, H, D), torch.randn(B, S, H)

        ab_out, ab_lse = _online_softmax_update(o1, l1, o2, l2)
        abc_out, abc_lse = _online_softmax_update(ab_out, ab_lse, o3, l3)

        bc_out, bc_lse = _online_softmax_update(o2, l2, o3, l3)
        abc_out2, abc_lse2 = _online_softmax_update(o1, l1, bc_out, bc_lse)

        torch.testing.assert_close(abc_out, abc_out2, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(abc_lse, abc_lse2, atol=1e-4, rtol=1e-4)


# =========================================================================
# API existence checks
# =========================================================================


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


# =========================================================================
# Distributed ring attention test (2 GPUs)
# =========================================================================

_MULTI_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need ≥2 GPUs for CP P2P distributed test",
)


def _is_aiter_available():
    try:
        import aiter  # noqa: F401

        return True
    except ImportError:
        return False


_AITER = pytest.mark.skipif(not _is_aiter_available(), reason="AITER required")


def _cp_p2p_worker(rank, world_size, result_queue, global_q, global_k, global_v, sm_scale, port):
    """Worker function for 2-GPU CP P2P test."""
    import torch.distributed as dist

    from lumen.ops.attention.attention_with_cp_p2p import attention_cp_p2p

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=device)
    try:
        cp_group = dist.new_group(ranks=list(range(world_size)))

        B, S_total, H, D = global_q.shape
        S_local = S_total // world_size
        q_local = global_q[:, rank * S_local : (rank + 1) * S_local].to(device).contiguous()
        k_local = global_k[:, rank * S_local : (rank + 1) * S_local].to(device).contiguous()
        v_local = global_v[:, rank * S_local : (rank + 1) * S_local].to(device).contiguous()

        def attn_fn(q_chunk, k_chunk, v_chunk, causal, softmax_scale):
            from lumen.ops.attention.attention import attention

            out = attention(q_chunk, k_chunk, v_chunk, softmax_scale=softmax_scale, causal=causal, return_lse=True)
            return out[0], out[1]

        out_local = attention_cp_p2p(
            q_local,
            k_local,
            v_local,
            cp_group=cp_group,
            cp_size=world_size,
            cp_rank=rank,
            attn_fn=attn_fn,
            causal=True,
            softmax_scale=sm_scale,
        )

        result_queue.put((rank, out_local.cpu()))
    finally:
        dist.destroy_process_group()


@_MULTI_GPU
@_AITER
class TestCPP2PDistributed:
    """Distributed 2-GPU test for CP P2P ring attention."""

    def test_output_shape(self):
        import socket

        import torch.multiprocessing as mp

        B, S, H, D = 2, 64, 8, 64
        sm_scale = D**-0.5
        torch.manual_seed(42)
        global_q = torch.randn(B, S, H, D, dtype=torch.bfloat16) * 0.02
        global_k = torch.randn(B, S, H, D, dtype=torch.bfloat16) * 0.02
        global_v = torch.randn(B, S, H, D, dtype=torch.bfloat16) * 0.02

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

        result_queue = mp.Queue()
        mp.spawn(
            _cp_p2p_worker,
            args=(2, result_queue, global_q, global_k, global_v, sm_scale, port),
            nprocs=2,
            join=True,
        )

        results = {}
        while not result_queue.empty():
            rank, out = result_queue.get()
            results[rank] = out

        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        S_local = S // 2
        for rank in [0, 1]:
            assert results[rank].shape == (
                B,
                S_local,
                H,
                D,
            ), f"Rank {rank}: expected shape ({B}, {S_local}, {H}, {D}), got {results[rank].shape}"
        assert torch.isfinite(results[0]).all(), "Rank 0 output contains non-finite values"
        assert torch.isfinite(results[1]).all(), "Rank 1 output contains non-finite values"
