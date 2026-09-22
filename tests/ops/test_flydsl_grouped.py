###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

import pytest
import torch

from lumen.ops.moe.flydsl_grouped import (
    build_pre_routed_metadata,
    flydsl_grouped_dgrad,
    flydsl_grouped_wgrad,
    padded_row_capacity,
)


def _require_flydsl_gfx950():
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    try:
        from flydsl.runtime.device import get_rocm_arch
        from kernels.moe.sonic_grouped_a16w16 import compile_sonic_grouped_a16w16_nn
    except ImportError:
        pytest.skip("FlyDSL grouped kernels are not available")
    del compile_sonic_grouped_a16w16_nn
    arch = str(get_rocm_arch())
    if "gfx950" not in arch:
        pytest.skip(f"FlyDSL grouped GEMMs require gfx950, found {arch}")
    return torch.device("cuda")


def test_padded_row_capacity_is_host_known():
    assert padded_row_capacity(0, 16) % 64 == 0
    assert padded_row_capacity(100, 8) >= 100
    assert padded_row_capacity(100, 8) >= 100 + 8 * 63


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_pre_routed_metadata_pads_and_skips_empty_experts():
    device = torch.device("cuda")
    counts = torch.tensor([17, 0, 15], dtype=torch.int32, device=device)
    cu = torch.zeros(4, dtype=torch.int32, device=device)
    cu[1:] = counts.cumsum(0)
    layout = build_pre_routed_metadata(cu, num_tokens=32)
    assert int(layout.frequency[1]) == 0
    assert int(layout.num_valid_ids[0]) == 128
    valid_blocks = int(layout.num_valid_ids[0]) // 64
    expert_ids = layout.sorted_expert_ids[:valid_blocks].tolist()
    assert expert_ids == [0, 2]
    assert layout.dest_index.tolist()[:17] == list(range(17))
    assert layout.dest_index[17:].tolist() == list(range(64, 79))
    token_ids = layout.sorted_token_ids.tolist()
    assert token_ids[:17] == list(range(17))
    assert token_ids[17:64] == [32] * 47
    assert token_ids[64:79] == list(range(17, 32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_pre_routed_metadata_supports_shared_bm128_layout():
    device = torch.device("cuda")
    counts = torch.tensor([129, 0, 127], dtype=torch.int32, device=device)
    cu = torch.zeros(4, dtype=torch.int32, device=device)
    cu[1:] = counts.cumsum(0)
    layout = build_pre_routed_metadata(cu, num_tokens=256, sort_unit=128)
    assert layout.sort_unit == 128
    assert int(layout.num_valid_ids[0]) == 384
    assert layout.sorted_expert_ids[:3].tolist() == [0, 0, 2]
    assert layout.dest_index[:129].tolist() == list(range(129))
    assert layout.dest_index[129:].tolist() == list(range(256, 383))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_flydsl_grouped_dgrad_and_wgrad_match_torch():
    device = _require_flydsl_gfx950()
    torch.manual_seed(0)
    counts = torch.tensor([17, 0, 15], dtype=torch.int32, device=device)
    cu = torch.zeros(4, dtype=torch.int32, device=device)
    cu[1:] = counts.cumsum(0)
    tokens = int(counts.sum())
    hidden, intermediate = 128, 128
    a = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
    weight = torch.randn(
        3, intermediate, hidden, dtype=torch.bfloat16, device=device
    )
    out = torch.empty(tokens, intermediate, dtype=torch.bfloat16, device=device)
    flydsl_grouped_dgrad(a, weight, cu, out)

    ref = torch.zeros_like(out, dtype=torch.float32)
    offset = 0
    for expert, count in enumerate(counts.tolist()):
        if count:
            ref[offset : offset + count] = (
                a[offset : offset + count].float() @ weight[expert].float().T
            )
            offset += count
    torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=3e-2)

    lhs = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
    rhs = torch.randn(tokens, 2 * intermediate, dtype=torch.bfloat16, device=device)
    dw = torch.empty(3, hidden, 2 * intermediate, dtype=torch.bfloat16, device=device)
    flydsl_grouped_wgrad(lhs, rhs, cu, dw)
    ref_dw = torch.zeros_like(dw, dtype=torch.float32)
    offset = 0
    for expert, count in enumerate(counts.tolist()):
        if count:
            ref_dw[expert] = (
                lhs[offset : offset + count].float().T
                @ rhs[offset : offset + count].float()
            )
            offset += count
    torch.testing.assert_close(dw.float(), ref_dw, rtol=3e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_flydsl_grouped_handles_empty_tokens():
    device = _require_flydsl_gfx950()
    cu = torch.zeros(3, dtype=torch.int32, device=device)
    a = torch.empty(0, 128, dtype=torch.bfloat16, device=device)
    weight = torch.randn(2, 128, 128, dtype=torch.bfloat16, device=device)
    out = torch.ones(0, 128, dtype=torch.bfloat16, device=device)
    flydsl_grouped_dgrad(a, weight, cu, out)
    rhs = torch.empty(0, 128, dtype=torch.bfloat16, device=device)
    dw = torch.ones(2, 128, 128, dtype=torch.bfloat16, device=device)
    flydsl_grouped_wgrad(a, rhs, cu, dw)
    assert dw.count_nonzero() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_flydsl_grouped_qwen3_local_shapes():
    device = _require_flydsl_gfx950()
    torch.manual_seed(1)
    hidden, intermediate, num_experts = 2048, 768, 16
    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    counts[0] = 65
    counts[3] = 1
    counts[7] = 64
    counts[15] = 17
    cu = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    cu[1:] = counts.cumsum(0)
    tokens = int(counts.sum())
    dy = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
    w2 = torch.randn(
        num_experts, intermediate, hidden, dtype=torch.bfloat16, device=device
    )
    da = torch.empty(tokens, intermediate, dtype=torch.bfloat16, device=device)
    flydsl_grouped_dgrad(dy, w2, cu, da)
    ref = torch.zeros_like(da, dtype=torch.float32)
    offset = 0
    for expert, count in enumerate(counts.tolist()):
        if count:
            ref[offset : offset + count] = (
                dy[offset : offset + count].float() @ w2[expert].float().T
            )
            offset += count
    torch.testing.assert_close(da.float(), ref, rtol=3e-2, atol=5e-2)

    dh = torch.randn(tokens, 2 * intermediate, dtype=torch.bfloat16, device=device)
    w1 = torch.randn(
        num_experts, hidden, 2 * intermediate, dtype=torch.bfloat16, device=device
    )
    dx = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=device)
    flydsl_grouped_dgrad(dh, w1, cu, dx)
    ref_dx = torch.zeros_like(dx, dtype=torch.float32)
    offset = 0
    for expert, count in enumerate(counts.tolist()):
        if count:
            ref_dx[offset : offset + count] = (
                dh[offset : offset + count].float() @ w1[expert].float().T
            )
            offset += count
    torch.testing.assert_close(dx.float(), ref_dx, rtol=3e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_flydsl_pre_routed_matches_torch(monkeypatch):
    device = _require_flydsl_gfx950()
    torch.manual_seed(2)
    from lumen.ops.moe.flydsl_grouped import flydsl_pre_routed

    monkeypatch.setenv("SONIC_MOE_GEMM_BACKEND", "flydsl")
    monkeypatch.setenv("SONIC_MOE_FLYDSL_NATIVE", "0")
    hidden, intermediate = 128, 128
    counts = torch.tensor([17, 0, 15], dtype=torch.int32, device=device)
    tokens = int(counts.sum())
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device, requires_grad=True)
    scores = torch.rand(tokens, dtype=torch.float32, device=device, requires_grad=True)
    w1 = (
        torch.randn(
            3, hidden, 2 * intermediate, dtype=torch.float32, device=device
        )
        / hidden ** 0.5
    ).to(torch.bfloat16)
    w1.requires_grad_(True)
    w2 = (
        torch.randn(
            3, intermediate, hidden, dtype=torch.float32, device=device
        )
        / intermediate ** 0.5
    ).to(torch.bfloat16)
    w2.requires_grad_(True)
    actual = flydsl_pre_routed(x, counts, scores, w1, w2)
    grad = torch.randn_like(actual)
    actual.backward(grad)

    ref_x = x.detach().clone().requires_grad_(True)
    ref_s = scores.detach().clone().requires_grad_(True)
    ref_w1 = w1.detach().clone().requires_grad_(True)
    ref_w2 = w2.detach().clone().requires_grad_(True)
    outputs = []
    offset = 0
    for expert, count in enumerate(counts.tolist()):
        if count:
            h = ref_x[offset : offset + count] @ ref_w1[expert]
            gate, up = h.chunk(2, dim=-1)
            outputs.append(
                (torch.nn.functional.silu(gate) * up)
                @ ref_w2[expert]
                * ref_s[offset : offset + count, None].to(ref_x.dtype)
            )
            offset += count
    reference = torch.cat(outputs)
    reference.backward(grad)
    torch.testing.assert_close(actual.float(), reference.float(), rtol=3e-2, atol=5e-2)
    torch.testing.assert_close(x.grad.float(), ref_x.grad.float(), rtol=3e-2, atol=5e-2)
    torch.testing.assert_close(w1.grad.float(), ref_w1.grad.float(), rtol=5e-2, atol=8e-2)
    torch.testing.assert_close(w2.grad.float(), ref_w2.grad.float(), rtol=5e-2, atol=8e-2)
    torch.testing.assert_close(scores.grad, ref_s.grad, rtol=3e-2, atol=5e-2)
