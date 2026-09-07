###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Tests for lumen.ops.gemm.grouped_gemm: grouped GEMM for MoE.

Covers:
  - BF16 grouped GEMM forward — all shapes, expert counts, N!=K
  - BF16 grouped GEMM forward with bias
  - BF16 grouped GEMM wgrad
  - Zero-group edge cases
  - FP8 scaling_type dispatch (blockwise, blockwise2d)
  - FP8 grouped Linear / expert MLP forward + dgrad/wgrad (no BF16 fallback)
  - Invalid scaling_type error
"""

import pytest
import torch
import importlib
from conftest import compute_snr, grouped_gemm_ref

from lumen.ops.gemm.grouped_gemm import grouped_gemm, grouped_gemm_wgrad


def _make_group_sizes(num_experts, tokens, device):
    """Create group_sizes that sum to tokens, distributed across experts."""
    base = tokens // num_experts
    remainder = tokens % num_experts
    sizes = [base + (1 if i < remainder else 0) for i in range(num_experts)]
    return torch.tensor(sizes, dtype=torch.int32, device=device)


def _wgrad_ref(grad_output, input_tensor, group_sizes):
    """Per-expert weight gradient reference: wgrad[g] = grad_g^T @ input_g."""
    num_experts = len(group_sizes)
    N = grad_output.shape[-1]
    K = input_tensor.shape[-1]
    wgrad = torch.zeros(num_experts, N, K, device=grad_output.device, dtype=grad_output.dtype)
    offset = 0
    for g, size in enumerate(group_sizes.tolist()):
        size = int(size)
        if size == 0:
            continue
        wgrad[g] = grad_output[offset : offset + size].T @ input_tensor[offset : offset + size]
        offset += size
    return wgrad


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

EXPERT_COUNTS = [1, 4, 8]
TOKEN_COUNTS = [64, 256]
# N != K shapes to verify independent input/output dims
SHAPES_NK = [(256, 256), (256, 512), (512, 256)]
SHAPE_IDS = [f"N{n}_K{k}" for n, k in SHAPES_NK]


# ===================================================================
# BF16 grouped GEMM forward
# ===================================================================


@pytest.mark.parametrize("num_experts", EXPERT_COUNTS)
@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
@pytest.mark.parametrize("N,K", SHAPES_NK, ids=SHAPE_IDS)
def test_grouped_gemm_fwd(num_experts, tokens, N, K):
    """BF16 grouped GEMM forward: compare against grouped_gemm_ref."""
    dtype = torch.bfloat16
    device = "cuda"
    group_sizes = _make_group_sizes(num_experts, tokens, device)
    total_tokens = int(group_sizes.sum().item())

    lhs = torch.randn(total_tokens, K, device=device, dtype=dtype) * 0.1
    rhs_ref = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.02

    out_ref = grouped_gemm_ref(lhs, rhs_ref, group_sizes)

    rhs_lumen = rhs_ref.transpose(1, 2)
    out_lumen = grouped_gemm(lhs, rhs_lumen, group_sizes, scaling_type="none")

    assert out_lumen.shape == (total_tokens, N)
    snr = compute_snr(out_ref, out_lumen)
    assert snr > 25, f"Grouped GEMM fwd SNR: {snr:.1f} dB (expected > 25)"


# ===================================================================
# BF16 grouped GEMM forward with bias
# ===================================================================


@pytest.mark.parametrize("num_experts", [4, 8])
@pytest.mark.parametrize("tokens", [64, 256])
def test_grouped_gemm_bias(num_experts, tokens):
    """Grouped GEMM forward with per-expert bias."""
    dtype = torch.bfloat16
    N, K = 256, 512
    device = "cuda"
    group_sizes = _make_group_sizes(num_experts, tokens, device)
    total_tokens = int(group_sizes.sum().item())

    lhs = torch.randn(total_tokens, K, device=device, dtype=dtype) * 0.1
    rhs_ref = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.02
    bias = torch.randn(num_experts, N, device=device, dtype=dtype) * 0.01

    out_ref = grouped_gemm_ref(lhs, rhs_ref, group_sizes, bias=bias)

    rhs_lumen = rhs_ref.transpose(1, 2)
    out_lumen = grouped_gemm(lhs, rhs_lumen, group_sizes, scaling_type="none", bias=bias)

    assert out_lumen.shape == (total_tokens, N)
    snr = compute_snr(out_ref, out_lumen)
    assert snr > 25, f"Grouped GEMM bias SNR: {snr:.1f} dB"


# ===================================================================
# Zero-group edge cases
# ===================================================================


@pytest.mark.parametrize(
    "sizes",
    [
        [16, 0, 16, 0],
        [0, 0, 32, 0],
    ],
    ids=["mixed_zeros", "leading_zeros"],
)
def test_grouped_gemm_zero_groups(sizes):
    """Experts with zero tokens should be handled without error."""
    dtype = torch.bfloat16
    N, K = 256, 256
    device = "cuda"
    num_experts = len(sizes)
    group_sizes = torch.tensor(sizes, dtype=torch.int32, device=device)
    total_tokens = int(group_sizes.sum().item())

    lhs = torch.randn(total_tokens, K, device=device, dtype=dtype) * 0.1
    rhs_ref = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.02

    out_ref = grouped_gemm_ref(lhs, rhs_ref, group_sizes)

    rhs_lumen = rhs_ref.transpose(1, 2)
    out_lumen = grouped_gemm(lhs, rhs_lumen, group_sizes, scaling_type="none")

    assert out_lumen.shape == (total_tokens, N)
    snr = compute_snr(out_ref, out_lumen)
    assert snr > 25, f"Grouped GEMM zero-groups SNR: {snr:.1f} dB"


# ===================================================================
# BF16 grouped GEMM wgrad
# ===================================================================


@pytest.mark.parametrize("num_experts", [1, 4, 8])
@pytest.mark.parametrize("tokens", [64, 256])
def test_grouped_gemm_wgrad(num_experts, tokens):
    """BF16 grouped GEMM wgrad: compare against per-expert grad^T @ input."""
    dtype = torch.bfloat16
    N, K = 256, 512
    device = "cuda"
    group_sizes = _make_group_sizes(num_experts, tokens, device)
    total_tokens = int(group_sizes.sum().item())

    grad_output = torch.randn(total_tokens, N, device=device, dtype=dtype) * 0.1
    input_tensor = torch.randn(total_tokens, K, device=device, dtype=dtype) * 0.1

    wgrad_ref = _wgrad_ref(grad_output, input_tensor, group_sizes)
    wgrad_lumen = grouped_gemm_wgrad(grad_output, input_tensor, group_sizes, scaling_type="none")

    assert wgrad_lumen.shape == (num_experts, N, K)
    snr = compute_snr(wgrad_ref, wgrad_lumen)
    assert snr > 20, f"Grouped GEMM wgrad SNR: {snr:.1f} dB (expected > 20)"


# ===================================================================
# FP8 scaling_type dispatch (blockwise, blockwise2d)
# ===================================================================


@pytest.mark.parametrize("scaling_type", ["blockwise", "blockwise2d"])
@pytest.mark.parametrize("num_experts", [4])
@pytest.mark.parametrize("tokens", [64, 256])
def test_grouped_gemm_fp8_scaling(num_experts, tokens, scaling_type):
    """Grouped GEMM forward with blockwise/blockwise2d scaling dispatch."""
    dtype = torch.bfloat16
    N, K = 256, 256
    device = "cuda"
    group_sizes = _make_group_sizes(num_experts, tokens, device)
    total_tokens = int(group_sizes.sum().item())

    lhs = torch.randn(total_tokens, K, device=device, dtype=dtype) * 0.1
    rhs_ref = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.02

    out_ref = grouped_gemm_ref(lhs, rhs_ref, group_sizes)

    rhs_lumen = rhs_ref.transpose(1, 2)
    out_lumen = grouped_gemm(lhs, rhs_lumen, group_sizes, scaling_type=scaling_type)

    assert out_lumen.shape == (total_tokens, N)
    snr = compute_snr(out_ref, out_lumen)
    assert snr > 8, f"Grouped GEMM {scaling_type} SNR: {snr:.1f} dB (expected > 8)"


# ===================================================================
# Invalid scaling_type
# ===================================================================


def test_grouped_gemm_invalid_scaling_type():
    """Unknown scaling_type should raise ValueError."""
    dtype = torch.bfloat16
    device = "cuda"
    lhs = torch.randn(32, 256, device=device, dtype=dtype)
    rhs = torch.randn(4, 256, 256, device=device, dtype=dtype)
    group_sizes = torch.tensor([8, 8, 8, 8], dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="Unknown scaling_type"):
        grouped_gemm(lhs, rhs, group_sizes, scaling_type="invalid")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_grouped_quantized_linear_fwd_bwd_matches_per_expert(monkeypatch):
    """Grouped FP8 Linear should match per-expert quantized_linear."""
    grouped_gemm_module = importlib.import_module("lumen.ops.gemm.grouped_gemm")
    from lumen.ops.gemm.grouped_gemm import grouped_quantized_linear
    from lumen.ops.quantize.linear import quantized_linear
    from lumen.quantize.config import _get_float8_e4m3

    device = "cuda"
    dtype = torch.bfloat16
    e, tokens, k, n = 2, 64, 128, 256
    group_sizes = torch.tensor([40, 24], dtype=torch.int32, device=device)
    torch.manual_seed(0)
    inp = torch.randn(tokens, k, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(e, k, n, device=device, dtype=dtype, requires_grad=True)
    fp8_dtype = _get_float8_e4m3()
    monkeypatch.setattr(
        grouped_gemm_module,
        "_sequential_grouped_linear_backward",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected BF16 expert backward fallback")
        ),
    )

    grouped_out = grouped_quantized_linear(
        inp,
        weight,
        group_sizes,
        scaling_type="blockwise2d",
        fp8_dtype=fp8_dtype,
        block_size=128,
    )
    ref_chunks = []
    offset = 0
    for expert, count in enumerate(group_sizes.tolist()):
        ref_chunks.append(
            quantized_linear(
                inp[offset : offset + count],
                weight[expert].transpose(0, 1),
                None,
                scaling_type="blockwise2d",
                fp8_dtype=fp8_dtype,
                block_size=128,
                tensor_id=f"ref.{expert}",
            )
        )
        offset += count
    ref_out = torch.cat(ref_chunks, dim=0)
    snr = compute_snr(ref_out.float(), grouped_out.float())
    assert snr > 20, f"grouped vs per-expert fwd SNR {snr:.1f} dB"

    grad = torch.randn_like(grouped_out)
    grouped_out.backward(grad)
    assert inp.grad is not None and weight.grad is not None
    assert torch.isfinite(inp.grad.float()).all()
    assert torch.isfinite(weight.grad.float()).all()
    input_grad_ref = torch.empty_like(inp)
    weight_grad_ref = torch.empty_like(weight)
    offset = 0
    for expert, count in enumerate(group_sizes.tolist()):
        x_e = inp.detach()[offset : offset + count]
        dy_e = grad[offset : offset + count]
        input_grad_ref[offset : offset + count] = dy_e @ weight.detach()[expert].T
        weight_grad_ref[expert] = x_e.T @ dy_e
        offset += count
    input_grad_snr = compute_snr(input_grad_ref.float(), inp.grad.float())
    weight_grad_snr = compute_snr(weight_grad_ref.float(), weight.grad.float())
    assert input_grad_snr > 15, f"FP8 grouped dgrad SNR {input_grad_snr:.1f} dB"
    assert weight_grad_snr > 15, f"FP8 grouped wgrad SNR {weight_grad_snr:.1f} dB"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_grouped_quantized_linear_zero_expert(monkeypatch):
    grouped_gemm_module = importlib.import_module("lumen.ops.gemm.grouped_gemm")
    from lumen.ops.gemm.grouped_gemm import grouped_quantized_linear
    from lumen.quantize.config import _get_float8_e4m3

    monkeypatch.setattr(
        grouped_gemm_module,
        "_sequential_grouped_linear_backward",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected BF16 expert backward fallback")
        ),
    )
    inp = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(2, 128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    group_sizes = torch.tensor([32, 0], dtype=torch.int32, device="cuda")
    out = grouped_quantized_linear(
        inp,
        weight,
        group_sizes,
        scaling_type="blockwise2d",
        fp8_dtype=_get_float8_e4m3(),
        block_size=128,
    )
    assert out.shape == (32, 128)
    out.float().sum().backward()
    assert inp.grad is not None
    assert weight.grad is not None
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_grouped_quantized_linear_qwen3_mbs2_shapes(monkeypatch):
    """Regression: MBS=2 token counts used to GPU-fault in FP8 dgrad/ptgmm."""
    grouped_gemm_module = importlib.import_module("lumen.ops.gemm.grouped_gemm")
    from lumen.ops.gemm.grouped_gemm import grouped_quantized_linear
    from lumen.quantize.config import _get_float8_e4m3

    device = "cuda"
    dtype = torch.bfloat16
    e, tokens, k, n = 16, 8192, 2048, 1536
    monkeypatch.setattr(
        grouped_gemm_module,
        "_sequential_grouped_linear_backward",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected BF16 expert backward fallback")
        ),
    )
    torch.manual_seed(1)
    raw = torch.randint(16, 1024, (e,), device=device)
    group_sizes = (raw * tokens // int(raw.sum().item())).to(torch.int32)
    group_sizes[-1] += tokens - int(group_sizes.sum().item())
    inp = torch.randn(tokens, k, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(e, k, n, device=device, dtype=dtype, requires_grad=True)
    out = grouped_quantized_linear(
        inp,
        weight,
        group_sizes,
        scaling_type="blockwise2d",
        fp8_dtype=_get_float8_e4m3(),
        block_size=128,
    )
    out.backward(torch.randn_like(out))
    torch.cuda.synchronize()
    assert torch.isfinite(inp.grad.float()).all()
    assert torch.isfinite(weight.grad.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_quant_blockwise_2d_gpu_and_cache():
    from lumen.ops.gemm.grouped_gemm import _quant_blockwise_2d, _weight_fp8_cache
    from lumen.ops.quantize.padding import pad_to_block
    from lumen.quantize.config import _get_float8_e4m3

    torch.manual_seed(0)
    w = torch.randn(4, 256, 128, device="cuda", dtype=torch.bfloat16)
    fp8_dtype = _get_float8_e4m3()
    w_fp8_a, scales_a = _quant_blockwise_2d(w, fp8_dtype, 128)
    w_fp8_b, scales_b = _quant_blockwise_2d(w, fp8_dtype, 128)
    assert w_fp8_a.data_ptr() == w_fp8_b.data_ptr()
    assert scales_a.data_ptr() == scales_b.data_ptr()

    bs = 128
    fp8_max = torch.finfo(fp8_dtype).max
    ref_scales = []
    for expert in range(w.shape[0]):
        we, _ = pad_to_block(w[expert], bs, dim=0)
        we, _ = pad_to_block(we, bs, dim=1)
        bk, bn = we.shape[0] // bs, we.shape[1] // bs
        blocks = we.reshape(bk, bs, bn, bs).permute(0, 2, 1, 3)
        amax = blocks.float().abs().amax(dim=(-2, -1)).clamp(min=1e-4)
        ref_scales.append(amax / fp8_max)
    ref = torch.stack(ref_scales, dim=0)
    torch.testing.assert_close(scales_a.float(), ref, rtol=1e-3, atol=1e-3)

    w.add_(0.25)
    w_fp8_c, _scales_c = _quant_blockwise_2d(w, fp8_dtype, 128)
    assert w_fp8_c.data_ptr() != w_fp8_a.data_ptr()
    assert w.data_ptr() in _weight_fp8_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_routing_data_stays_on_gpu():
    from lumen.ops.gemm.grouped_gemm import _build_routing_data_from_group_sizes

    group_sizes = torch.tensor([40, 24, 0, 64], dtype=torch.int32, device="cuda")
    routing = _build_routing_data_from_group_sizes(group_sizes, 128)
    assert routing.expt_hist.device.type == "cuda"
    assert routing.expt_data.token_offs_raw.device.type == "cuda"
    assert routing.expt_data.block_pid_map.device.type == "cuda"
    assert routing.expt_data.token_offs_raw.tolist() == [0, 40, 64, 64, 128]
    packed = routing.expt_data.block_pid_map.cpu()
    assert packed[0].item() == 0
    n_tiles0 = (40 + routing.block_m - 1) // routing.block_m
    assert packed[n_tiles0].item() & 0xFFFF == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_grouped_fp8_expert_mlp_matches_silu_path(monkeypatch):
    grouped_gemm_module = importlib.import_module("lumen.ops.gemm.grouped_gemm")
    from lumen.ops.gemm.grouped_gemm import (
        grouped_fp8_expert_mlp,
        grouped_quantized_linear,
    )
    from lumen.quantize.config import _get_float8_e4m3

    device = "cuda"
    dtype = torch.bfloat16
    e, tokens, k, n = 2, 64, 128, 256
    group_sizes = torch.tensor([40, 24], dtype=torch.int32, device=device)
    torch.manual_seed(2)
    hidden = torch.randn(tokens, k, device=device, dtype=dtype, requires_grad=True)
    w1 = torch.randn(e, k, n, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(e, n // 2, k, device=device, dtype=dtype, requires_grad=True)
    fp8_dtype = _get_float8_e4m3()
    monkeypatch.setattr(
        grouped_gemm_module,
        "_sequential_grouped_linear_backward",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected BF16 expert backward fallback")
        ),
    )
    kwargs = dict(scaling_type="blockwise2d", fp8_dtype=fp8_dtype, block_size=128)

    fused = grouped_fp8_expert_mlp(hidden, w1, w2, group_sizes, **kwargs)
    fc1 = grouped_quantized_linear(hidden, w1, group_sizes, **kwargs)
    gate, up = fc1.chunk(2, dim=-1)
    ref = grouped_quantized_linear(torch.nn.functional.silu(gate) * up, w2, group_sizes, **kwargs)
    snr = compute_snr(ref.float(), fused.float())
    assert snr > 20, f"fused SwiGLU MLP vs silu path SNR {snr:.1f} dB"

    grad = torch.randn_like(fused)
    fused.backward(grad)
    torch.cuda.synchronize()
    assert torch.isfinite(hidden.grad.float()).all()
    assert torch.isfinite(w1.grad.float()).all()
    assert torch.isfinite(w2.grad.float()).all()
