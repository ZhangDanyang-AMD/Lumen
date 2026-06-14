###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################
"""Blockwise2DFP8Param: the FSDP2 all-gather wrapper for a frozen blockwise2d
base weight stored as FP8. The pre/post all-gather pair must reconstruct the
full FP8 weight + 2D scale bit-exactly (no re-quant, no dequant) when the
sharded shards are gathered back."""

import pytest
import torch

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _fp8():
    return (torch.float8_e4m3fnuz
            if "gfx94" in torch.cuda.get_device_properties(0).gcnArchName
            else torch.float8_e4m3fn)


@_CUDA
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_blockwise2d_fp8_param_allgather_roundtrip(world_size):
    """Simulate FSDP2 dim-0 sharding over `world_size` ranks: shard the FP8 weight
    + 2D scale, run pre_all_gather per shard, concat (= all-gather), run
    post_all_gather; the result must be bit-identical to the unsharded weight."""
    from lumen.quantize.comm_tensor import Blockwise2DFP8Param
    from lumen.ops.quantize.linear import _quant_blockwise2d_weight

    torch.manual_seed(0)
    block = 128
    N, K = block * world_size * 2, block * 3   # N % (block*world_size) == 0
    w = (torch.randn(N, K) * 0.05).cuda().to(torch.bfloat16)
    fp8, scale = _quant_blockwise2d_weight(w, _fp8(), block)   # (N,K), (N/128,K/128)

    # Shard along dim 0 (FSDP2 even split). Scale rows track weight rows.
    rows = N // world_size
    srows = (N // block) // world_size
    gathered_fp8, gathered_scale, meta = [], [], None
    for r in range(world_size):
        shard = Blockwise2DFP8Param(
            fp8[r * rows:(r + 1) * rows].contiguous(),
            scale[r * srows:(r + 1) * srows].contiguous(),
            orig_dtype=torch.bfloat16,
            block_size=block,
        )
        tensors, meta = Blockwise2DFP8Param.fsdp_pre_all_gather(shard)
        gathered_fp8.append(tensors[0])
        gathered_scale.append(tensors[1])

    full_fp8 = torch.cat(gathered_fp8, dim=0)
    full_scale = torch.cat(gathered_scale, dim=0)
    result = Blockwise2DFP8Param.fsdp_post_all_gather(
        (full_fp8, full_scale), meta, torch.bfloat16
    )[0]

    assert isinstance(result, Blockwise2DFP8Param)
    assert result._fp8.dtype == _fp8(), "gathered weight must stay FP8 (no dequant)"
    assert result.shape == (N, K)
    # Bit-exact reconstruction — pre/post move bytes, never re-quantize.
    assert torch.equal(result._fp8.view(torch.uint8), fp8.view(torch.uint8))
    assert torch.equal(result._scale, scale)


@_CUDA
def test_blockwise2d_fp8_param_dequant_matches_weight():
    """Dequantizing the gathered FP8 + 2D scale must approximate the BF16 weight."""
    from lumen.quantize.comm_tensor import Blockwise2DFP8Param
    from lumen.ops.quantize.linear import _quant_blockwise2d_weight
    from lumen.ops.quantize.gemm_primitives import _dequant_fp8_weight
    from conftest import compute_snr  # type: ignore

    torch.manual_seed(0)
    block, N, K = 128, 256, 384
    w = (torch.randn(N, K) * 0.05).cuda().to(torch.bfloat16)
    fp8, scale = _quant_blockwise2d_weight(w, _fp8(), block)
    p = Blockwise2DFP8Param(fp8, scale, torch.bfloat16, block)

    deq = _dequant_fp8_weight(p._fp8, p._scale, block)
    assert compute_snr(w.float(), deq.float()) > 30


@_CUDA
def test_blockwise2d_fp8_param_flatten_roundtrip():
    """__tensor_flatten__/__tensor_unflatten__ must preserve FP8 data + scale."""
    from lumen.quantize.comm_tensor import Blockwise2DFP8Param
    from lumen.ops.quantize.linear import _quant_blockwise2d_weight

    torch.manual_seed(0)
    fp8, scale = _quant_blockwise2d_weight(
        (torch.randn(256, 256) * 0.05).cuda().to(torch.bfloat16), _fp8(), 128
    )
    p = Blockwise2DFP8Param(fp8, scale, torch.bfloat16, 128)
    names, meta = p.__tensor_flatten__()
    inner = {n: getattr(p, n) for n in names}
    p2 = Blockwise2DFP8Param.__tensor_unflatten__(inner, meta, p.shape, p.stride())
    assert torch.equal(p2._fp8.view(torch.uint8), fp8.view(torch.uint8))
    assert torch.equal(p2._scale, scale)
    assert p2._block_size == 128
