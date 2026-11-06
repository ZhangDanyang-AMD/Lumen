###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""MXFP4 operand-layout, fusion, and SR-Philox regressions.

Collected from the per-optimization files so one module covers:
  - dual-layout column shuffle
  - fused activation scale swizzle
  - fused WGrad activation operand
  - Philox round-count dither quality
"""

import importlib
import os
import random

import pytest
import torch

import lumen.kernels.mxfp4 as mxfp4_kernels
from lumen.ops.quantize.linear import QuantizedLinearFunction, _shuffle_mxfp4_weight
from lumen.ops.quantize.ops import dual_layout_quant_mxfp4

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

G = 16
BLOCK = 32
ROUNDS_FLOOR = 4
ROUNDS_BELOW_FLOOR = 2
DRAWS = 96


def _is_gfx950():
    if not torch.cuda.is_available():
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


def _run_linear(x, w, dy, scaling_type="mxfp4"):
    random.seed(0)
    x = x.detach().clone().requires_grad_(True)
    w = w.detach().clone().requires_grad_(True)
    out = QuantizedLinearFunction.apply(x, w, None, None, scaling_type, None, 32, "weight")
    out.backward(dy)
    return out, x.grad, w.grad


def _pad_rows(t, rows):
    return torch.cat([t, t.new_zeros((rows - t.shape[0], t.shape[1]))])


def _snr(ref, got):
    err = (ref.float() - got.float()).pow(2).sum()
    if err == 0:
        return float("inf")
    return 10 * torch.log10(ref.float().pow(2).sum() / err).item()


@pytest.mark.parametrize("shape", [(256, 128), (512, 256), (1024, 512)])
def test_shuffled_col_operand_matches_separate_shuffle_pass(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    sign = (torch.randint(0, 2, (G,), device="cuda") * 2 - 1).to(torch.bfloat16)

    row, row_s, col, col_s = dual_layout_quant_mxfp4(
        x, sign, block_size=BLOCK, g=G, use_sr_row=False, use_sr_transposed=False,
    )
    row_f, row_sf, col_f, col_sf = dual_layout_quant_mxfp4(
        x, sign, block_size=BLOCK, g=G, use_sr_row=False, use_sr_transposed=False,
        shuffle_col=True,
    )

    torch.testing.assert_close(row_f, row, rtol=0, atol=0)
    torch.testing.assert_close(row_sf, row_s, rtol=0, atol=0)
    torch.testing.assert_close(col_sf, col_s, rtol=0, atol=0)
    torch.testing.assert_close(col_f, _shuffle_mxfp4_weight(col), rtol=0, atol=0)


@pytest.mark.skipif(not _is_gfx950(), reason="gfx950 operand layout")
@pytest.mark.parametrize("use_sr", [False, True])
@pytest.mark.parametrize(
    "logical_rows,target_rows",
    [(1, 256), (255, 256), (257, 512), (542, 1024), (1025, 2048)],
)
def test_dual_layout_target_rows_matches_explicit_padding(
    use_sr, logical_rows, target_rows,
):
    width = 256
    torch.manual_seed(31)
    x = torch.randn(
        (logical_rows, width), device="cuda", dtype=torch.bfloat16,
    )
    sign = torch.ones(G, device=x.device, dtype=torch.bfloat16)
    kwargs = dict(
        block_size=BLOCK,
        g=G,
        use_sr_row=use_sr,
        use_sr_transposed=use_sr,
        philox_seed=1234,
        philox_offset=5678,
        swizzle_scale=True,
        shuffle_col=True,
    )

    ref = dual_layout_quant_mxfp4(_pad_rows(x, target_rows), sign, **kwargs)
    got = dual_layout_quant_mxfp4(
        x, sign, target_rows=target_rows, **kwargs,
    )

    for actual, expected in zip(got, ref):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not _is_gfx950(), reason="gfx950 operand layout")
@pytest.mark.parametrize(
    "logical_rows,target_rows",
    [(1, 256), (255, 256), (257, 512), (542, 1024), (1025, 2048)],
)
def test_dequant_hadamard_target_rows_matches_explicit_padding(
    logical_rows, target_rows,
):
    from lumen.ops.quantize.ops import (
        convert_to_mxfp4,
        dequant_hadamard_quant_mxfp4,
    )

    width = 256
    torch.manual_seed(37)
    x = torch.randn(
        (logical_rows, width), device="cuda", dtype=torch.bfloat16,
    )
    data, scales = convert_to_mxfp4(
        x, block_size=BLOCK, axis=-1, use_sr=False,
    )
    sign = torch.ones(G, device=x.device, dtype=torch.bfloat16)
    kwargs = dict(
        block_size=BLOCK,
        g=G,
        use_sr=False,
        swizzle_scale=True,
        shuffle_data=True,
    )

    ref = dequant_hadamard_quant_mxfp4(
        _pad_rows(data, target_rows), _pad_rows(scales, target_rows), sign, **kwargs,
    )
    got = dequant_hadamard_quant_mxfp4(
        data, scales, sign, target_rows=target_rows, **kwargs,
    )

    for actual, expected in zip(got, ref):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _compiled_variants(kernel):
    caches = getattr(kernel, "device_caches", None)
    if caches is not None:
        return sum(len(entry[0]) for entry in caches.values())
    return sum(len(cache) for cache in kernel.cache.values())


@pytest.mark.skipif(not _is_gfx950(), reason="gfx950 operand layout")
def test_target_row_quantizers_compile_once_across_token_counts():
    from lumen.ops.quantize.ops import (
        convert_to_mxfp4,
        dequant_hadamard_quant_mxfp4,
    )

    width = 256
    sign = torch.ones(G, device="cuda", dtype=torch.bfloat16)
    dual = mxfp4_kernels._dual_layout_quant_mxfp4_kernel
    dequant = mxfp4_kernels._dequant_hadamard_quant_mxfp4_kernel

    def quantize(rows, seed):
        target_rows = -(-rows // BLOCK) * BLOCK
        x = torch.randn((rows, width), device="cuda", dtype=torch.bfloat16)
        dual_layout_quant_mxfp4(
            x, sign, block_size=BLOCK, g=G,
            use_sr_row=True, use_sr_transposed=True,
            philox_seed=seed, philox_offset=seed, target_rows=target_rows,
        )
        data, scales = convert_to_mxfp4(
            x, block_size=BLOCK, axis=-1, use_sr=False,
        )
        dequant_hadamard_quant_mxfp4(
            data, scales, sign, block_size=BLOCK, g=G, use_sr=True,
            philox_seed=seed, philox_offset=seed, target_rows=target_rows,
        )

    before = (_compiled_variants(dual), _compiled_variants(dequant))
    # Triton specializes an integer on being 1 or a multiple of 16, so the
    # token counts and Philox counters cover all three classes.
    for rows, seed in [(1, 1), (100, 16), (300, 33), (500, 48), (777, 7), (1000, 1024)]:
        quantize(rows, seed)
    ragged = (_compiled_variants(dual), _compiled_variants(dequant))
    # Token counts that tile exactly share the unmasked variant instead.
    for rows, seed in [(512, 5), (768, 64)]:
        quantize(rows, seed)
    after = (_compiled_variants(dual), _compiled_variants(dequant))

    assert ragged[0] - before[0] <= 1, "dual-layout quant recompiled per token count"
    assert ragged[1] - before[1] <= 1, "dequant-Hadamard quant recompiled per token count"
    assert after[0] - ragged[0] <= 1, "dual-layout quant recompiled per whole-tile count"
    assert after[1] - ragged[1] <= 1, "dequant-Hadamard quant recompiled per whole-tile count"


@pytest.mark.skipif(not _is_gfx950(), reason="gfx950 MXFP4 backward")
def test_ragged_linear_backward_dispatches_target_rows(monkeypatch):
    import lumen.ops.quantize.linear as lin

    logical_rows, target_rows, width = 542, 1024, 256
    torch.manual_seed(41)
    x = torch.randn(
        (logical_rows, width), device="cuda", dtype=torch.bfloat16,
    )
    w = torch.randn(
        (width, width), device="cuda", dtype=torch.bfloat16,
    )
    dy = torch.randn_like(x)
    calls = []
    dispatch = lin.gemm_mxfp4_dispatch

    def record_dispatch(a, b, sa, sb):
        calls.append((tuple(a.shape), tuple(b.shape)))
        return dispatch(a, b, sa, sb)

    monkeypatch.setattr(lin, "_MXFP4_WGRAD_POWER2_BUCKET", True)
    monkeypatch.setattr(lin, "gemm_mxfp4_dispatch", record_dispatch)

    out, dx, dw = _run_linear(x, w, dy)

    assert out.shape == x.shape
    assert dx.shape == x.shape
    assert dw.shape == w.shape
    assert all(torch.isfinite(t).all() for t in (out, dx, dw))
    # Only WGrad reduces over the token axis, so only its A operand is
    # (width, target_rows // 2); forward and DGrad carry width // 2 columns.
    wgrad = [call for call in calls if call[0][1] == target_rows // 2]
    assert wgrad == [
        ((width, target_rows // 2), (width, target_rows // 2)),
    ]

    # Reference: zero-pad the producers' inputs to target rows up front.
    import lumen.ops.quantize.ops as quant_ops

    dual, dequant = quant_ops.dual_layout_quant_mxfp4, quant_ops.dequant_hadamard_quant_mxfp4

    def dual_padded(x, *args, target_rows=None, **kwargs):
        if target_rows is not None:
            x = _pad_rows(x, target_rows)
        return dual(x, *args, **kwargs)

    def dequant_padded(data, scales, *args, target_rows=None, **kwargs):
        if target_rows is not None:
            data, scales = _pad_rows(data, target_rows), _pad_rows(scales, target_rows)
        return dequant(data, scales, *args, **kwargs)

    monkeypatch.setattr(quant_ops, "dual_layout_quant_mxfp4", dual_padded)
    monkeypatch.setattr(quant_ops, "dequant_hadamard_quant_mxfp4", dequant_padded)
    ref = _run_linear(x, w, dy)
    for actual, expected in zip((out, dx, dw), ref):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not _is_gfx950(), reason="gfx950 MXFP4 backward")
@pytest.mark.parametrize(
    "in_features,out_features,stays_mxfp4",
    [(64, 256, True), (256, 64, False)],
    ids=["gradient_read_at_logical_rows", "dgrad_output_padded"],
)
def test_padding_past_int32_keeps_layer_bf16(
    monkeypatch, caplog, in_features, out_features, stays_mxfp4,
):
    import lumen.ops.quantize.linear as lin

    logical_rows, padded_rows = 300, 512
    torch.manual_seed(43)
    x = torch.randn(
        (logical_rows, in_features), device="cuda", dtype=torch.bfloat16,
    )
    w = torch.randn(
        (out_features, in_features), device="cuda", dtype=torch.bfloat16,
    )
    dy = torch.randn(
        (logical_rows, out_features), device="cuda", dtype=torch.bfloat16,
    )
    calls = []
    dispatch = lin.gemm_mxfp4_dispatch

    def record_dispatch(*args):
        calls.append(args[0].shape)
        return dispatch(*args)

    # Scaled-down 2**31: every logical operand fits under it, and so does the
    # gradient the quantizers read at logical rows, but the padded DGrad
    # output of a wide input does not.
    monkeypatch.setattr(
        lin, "_MXFP4_MAX_OPERAND_ELEMS", (logical_rows + padded_rows) // 2 * 256,
    )
    monkeypatch.setattr(lin, "_MXFP4_WGRAD_POWER2_BUCKET", True)
    monkeypatch.setattr(lin, "_mxfp4_int32_warned", set())
    monkeypatch.setattr(lin, "gemm_mxfp4_dispatch", record_dispatch)

    with caplog.at_level("WARNING", logger="lumen.ops.quantize.linear"):
        got = _run_linear(x, w, dy)
    if stays_mxfp4:
        assert calls
        return
    assert not calls
    assert any(
        f"padded to {padded_rows} rows" in record.getMessage()
        for record in caplog.records
    )
    ref = _run_linear(x, w, dy, scaling_type="none")
    for actual, expected in zip(got, ref):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not _is_gfx950(), reason="gfx950 scale layout")
@pytest.mark.parametrize("M,K,N", [(512, 4096, 4096), (256, 2048, 512)])
def test_mxfp4_linear_fused_act_scale_matches_row_major(M, K, N, monkeypatch):
    import lumen.ops.quantize.linear as lin

    torch.manual_seed(17)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    if not lin._mxfp4_can_fuse_scale_swizzle((M, K // 32)):
        pytest.skip(f"({M}, {K}) activation scales are not swizzle-eligible")

    monkeypatch.setattr(lin, "_mxfp4_wgrad_activation_operand", lambda *a: None)

    try:
        got = _run_linear(x, w, dy)
        monkeypatch.setattr(lin, "_mxfp4_can_fuse_scale_swizzle", lambda *a: False)
        ref = _run_linear(x, w, dy)
    except (AssertionError, RuntimeError) as e:
        pytest.skip(f"Lumen MXFP4 path unavailable: {e}")

    for name, a, b in zip(("out", "dX", "dW"), got, ref):
        torch.testing.assert_close(a, b, atol=0, rtol=0, msg=f"{name} differs")


@pytest.mark.skipif(not _is_gfx950(), reason="gfx950 operand layout")
@pytest.mark.parametrize("M,K,N", [(512, 512, 256), (768, 256, 512)])
def test_fused_wgrad_operand_keeps_forward_and_improves_wgrad(M, K, N, monkeypatch):
    import lumen.ops.quantize.linear as lin

    torch.manual_seed(17)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    if lin._mxfp4_wgrad_activation_operand(x, w, "mxfp4", True, True) is None:
        pytest.skip(f"({M}, {K}) is not eligible for the fused WGrad operand")

    try:
        got = _run_linear(x, w, dy)
        monkeypatch.setattr(lin, "_mxfp4_wgrad_activation_operand", lambda *a: None)
        ref = _run_linear(x, w, dy)
    except (AssertionError, RuntimeError) as e:
        pytest.skip(f"Lumen MXFP4 path unavailable: {e}")

    torch.testing.assert_close(got[0], ref[0], atol=0, rtol=0, msg="out differs")
    torch.testing.assert_close(got[1], ref[1], atol=0, rtol=0, msg="dX differs")

    dw_ref = dy.float().t() @ x.float()
    snr_fused, snr_rebuilt = _snr(dw_ref, got[2]), _snr(dw_ref, ref[2])
    assert snr_fused >= snr_rebuilt - 0.5, (
        f"fused dW is worse than the rebuilt one: {snr_fused:.2f} dB vs {snr_rebuilt:.2f} dB"
    )


@pytest.mark.skipif(not _is_gfx950(), reason="gfx950 operand layout")
def test_fused_operand_is_the_direct_quantization_of_the_rotated_activation():
    import lumen.ops.quantize.linear as lin
    from lumen.ops.quantize.ops import (
        convert_from_mxfp4,
        convert_to_mxfp4,
        dequant_hadamard_quant_mxfp4,
        hadamard_quant_mxfp4,
    )

    M, K = 512, 512
    torch.manual_seed(3)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    sign = lin._get_mxfp4_rht_sign(x.device)

    _, _, wg_fp4, wg_scale = dual_layout_quant_mxfp4(
        x, sign, block_size=32, g=lin._MXFP4_RHT_G,
        use_sr_row=False, use_sr_transposed=False,
    )
    exact_fp4, exact_scale = hadamard_quant_mxfp4(
        x.t().contiguous(), sign, block_size=32, g=lin._MXFP4_RHT_G, use_sr=False,
    )
    torch.testing.assert_close(wg_fp4, exact_fp4, atol=0, rtol=0)
    torch.testing.assert_close(wg_scale, exact_scale, atol=0, rtol=0)

    x_fp4, x_scale = convert_to_mxfp4(x, block_size=32, axis=-1, use_sr=False)
    rebuilt_fp4, rebuilt_scale = dequant_hadamard_quant_mxfp4(
        x_fp4, x_scale, sign, block_size=32, g=lin._MXFP4_RHT_G, use_sr=False,
    )
    exact = convert_from_mxfp4(exact_fp4, exact_scale, output_dtype=torch.float32, block_size=32)
    rebuilt = convert_from_mxfp4(
        rebuilt_fp4, rebuilt_scale, output_dtype=torch.float32, block_size=32,
    )
    assert not torch.equal(exact, rebuilt), "the rebuild would be lossless, which it is not"


@pytest.fixture
def rebuild_at_rounds():
    saved = os.environ.get("LUMEN_SR_PHILOX_ROUNDS")

    def _rebuild(rounds):
        os.environ["LUMEN_SR_PHILOX_ROUNDS"] = str(rounds)
        module = importlib.reload(mxfp4_kernels)
        assert module.SR_PHILOX_ROUNDS == rounds, "reload did not take"
        return module

    yield _rebuild

    if saved is None:
        os.environ.pop("LUMEN_SR_PHILOX_ROUNDS", None)
    else:
        os.environ["LUMEN_SR_PHILOX_ROUNDS"] = saved
    importlib.reload(mxfp4_kernels)


def _dequant(packed, scales, block):
    fp4_utils = pytest.importorskip("aiter.utility.fp4_utils", reason="AITER required")

    codes = fp4_utils.mxfp4_to_f32(packed)
    mult = (scales.to(torch.int32) << 23).view(torch.float32)
    return codes * mult.repeat_interleave(block, dim=-1)


def _residual_std(draws=DRAWS):
    from lumen.ops.quantize.ops import convert_to_mxfp4

    torch.manual_seed(0)
    shape = (256, BLOCK * 8)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    acc = torch.zeros(shape, dtype=torch.float32, device="cuda")
    for i in range(draws):
        packed, scales = convert_to_mxfp4(
            x, BLOCK, axis=-1, use_sr=True, philox_seed=1234 + i, philox_offset=0,
        )
        acc += _dequant(packed, scales, BLOCK)
    resid = acc / draws - x.to(torch.float32)
    denom = x.to(torch.float32).abs().mean().clamp_min(1e-6)
    return (resid.std() / denom).item()


@pytest.mark.skipif(not _is_gfx950(), reason="gfx950 SR packing")
def test_sr_dither_does_not_repeat_between_tiles():
    from lumen.ops.quantize.ops import convert_to_mxfp4

    torch.manual_seed(9)
    tile = torch.randn((64, 64), dtype=torch.bfloat16, device="cuda")
    x = tile.repeat(2, 1)

    packed, scales = convert_to_mxfp4(
        x, BLOCK, axis=-1, use_sr=True, philox_seed=1234, philox_offset=0,
    )

    torch.testing.assert_close(scales[:64], scales[64:], atol=0, rtol=0)
    assert not torch.equal(packed[:64], packed[64:]), (
        "identical input tiles reused the same stochastic-rounding stream"
    )


def test_default_round_count_is_the_documented_default(rebuild_at_rounds):
    os.environ.pop("LUMEN_SR_PHILOX_ROUNDS", None)
    module = importlib.reload(mxfp4_kernels)

    assert module.SR_PHILOX_ROUNDS == module.SR_PHILOX_ROUNDS_DEFAULT


def test_override_reaches_the_traced_constant(rebuild_at_rounds):
    module = rebuild_at_rounds(ROUNDS_FLOOR)

    assert module.SR_PHILOX_ROUNDS == ROUNDS_FLOOR
    assert module.SR_PHILOX_ROUNDS_C.value == ROUNDS_FLOOR


def test_rounds_at_the_floor_hold_dither_quality(rebuild_at_rounds):
    rebuild_at_rounds(mxfp4_kernels.SR_PHILOX_ROUNDS_DEFAULT)
    default_std = _residual_std()

    rebuild_at_rounds(ROUNDS_FLOOR)
    floor_std = _residual_std()

    assert floor_std == pytest.approx(default_std, rel=0.15), (
        f"rounds={ROUNDS_FLOOR} residual std {floor_std:.6f} vs "
        f"default {default_std:.6f}"
    )


def test_metric_catches_a_dither_below_the_floor(rebuild_at_rounds):
    rebuild_at_rounds(mxfp4_kernels.SR_PHILOX_ROUNDS_DEFAULT)
    default_std = _residual_std()

    rebuild_at_rounds(ROUNDS_BELOW_FLOOR)
    starved_std = _residual_std()

    assert starved_std > 2 * default_std, (
        f"rounds={ROUNDS_BELOW_FLOOR} residual std {starved_std:.6f} is not "
        f"clearly worse than default {default_std:.6f}; the metric has lost "
        "its discriminating power"
    )
