"""MXFP4 attention projections: recipe numerics, Megatron fallbacks and the per-step pack cache."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lumen.modules import mxfp4_attention_linear as attn_linear

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequant(payload, scales):
    lut = torch.tensor(_E2M1 + tuple(-v for v in _E2M1), device=payload.device)
    vals = torch.stack(
        (lut[(payload & 0xF).long()], lut[(payload >> 4).long()]), dim=-1
    ).reshape(payload.shape[0], -1)
    return vals * torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=1)


def _quant(t):
    from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant

    return _dequant(*dynamic_mxfp4_quant(t.contiguous()))


class _MegatronLinear(nn.Module):
    """Stand-in for Megatron's local-spec linear: BF16 ``_forward_impl``."""

    def __init__(self, out_features, in_features):
        super().__init__()
        self.weight = nn.Parameter(
            (torch.randn(out_features, in_features, device="cuda") * 0.02).bfloat16()
        )
        self.calls = 0

    def _forward_impl(self, input, weight, bias=None, **_kwargs):
        self.calls += 1
        out = torch.matmul(input, weight.t())
        return out if bias is None else out + bias

    def forward(self, x, **kwargs):
        return (
            self._forward_impl(input=x, weight=self.weight, bias=None, **kwargs),
            None,
        )


class ColumnParallelLinear(_MegatronLinear):
    pass


class RowParallelLinear(_MegatronLinear):
    pass


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self._pack_generation = 0
        self._pack_cache_enabled = True


def _model(layers=2, hidden=256, qkv=640, proj_in=512, experts=True):
    model = nn.Module()
    model.decoder = nn.Module()
    model.decoder.layers = nn.ModuleList()
    for _ in range(layers):
        layer = nn.Module()
        layer.self_attention = nn.Module()
        layer.self_attention.linear_qkv = ColumnParallelLinear(qkv, hidden)
        layer.self_attention.linear_proj = RowParallelLinear(hidden, proj_in)
        layer.mlp = nn.Module()
        if experts:
            layer.mlp.experts = _Experts()
        model.decoder.layers.append(layer)
    return model


@pytest.fixture
def installed(monkeypatch):
    monkeypatch.setenv("LUMEN_ATTN_LINEAR_RECIPE", "a4w4")
    torch.manual_seed(0)
    model = _model()
    assert attn_linear.install_mxfp4_attention_linears(model) == 4
    return model


def test_recipe_env_is_validated(monkeypatch):
    monkeypatch.delenv("LUMEN_ATTN_LINEAR_RECIPE", raising=False)
    assert attn_linear.attention_linear_recipe() == "bf16"
    monkeypatch.setenv("LUMEN_ATTN_LINEAR_RECIPE", "fp8")
    with pytest.raises(ValueError):
        attn_linear.attention_linear_recipe()


def test_config_rejects_global_linear_quantization(monkeypatch):
    from lumen.config import LumenConfig

    monkeypatch.setenv("LUMEN_ATTN_LINEAR_RECIPE", "a4w4")
    cfg = LumenConfig(lumen_linear=True)
    with pytest.raises(ValueError, match="must be disabled"):
        cfg._patch_attention_linears(nn.Module())


def test_config_bf16_recipe_is_a_no_op(monkeypatch):
    from lumen.config import LumenConfig

    monkeypatch.delenv("LUMEN_ATTN_LINEAR_RECIPE", raising=False)
    LumenConfig(lumen_linear=True)._patch_attention_linears(nn.Module())


@_CUDA
def test_installs_once_and_rejects_unknown_linears(installed):
    assert attn_linear.install_mxfp4_attention_linears(installed) == 0
    model = _model(layers=1)
    model.decoder.layers[0].self_attention.linear_qkv = nn.Linear(4, 4)
    with pytest.raises(TypeError):
        attn_linear.install_mxfp4_attention_linears(model)


@_CUDA
@pytest.mark.parametrize("name, k", [("linear_qkv", 256), ("linear_proj", 512)])
def test_recipe_matches_dequantized_reference(installed, name, k):
    linear = getattr(installed.decoder.layers[0].self_attention, name)
    x = torch.randn(128, 2, k, device="cuda").bfloat16().requires_grad_()
    y, bias = linear(x)
    assert bias is None and linear.calls == 0
    w = linear.weight.detach()
    ref = _quant(x.detach().reshape(-1, k)) @ _quant(w).t()
    err = (y.float().reshape(-1, w.shape[0]) - ref).abs().max() / ref.abs().max()
    assert err.item() < 8e-3

    dy = torch.randn_like(y)
    y.backward(dy)
    dy2d = dy.reshape(-1, w.shape[0])
    ref_dx = _quant(dy2d) @ _quant(w.t()).t()
    err = (x.grad.float().reshape(-1, k) - ref_dx).abs().max() / ref_dx.abs().max()
    assert err.item() < 8e-3
    # Wgrad stays Megatron's BF16 GEMM of the unquantized operands.
    assert torch.equal(linear.weight.grad, dy2d.t().matmul(x.detach().reshape(-1, k)))


@_CUDA
def test_autograd_gradient_flows_through_dgrad(installed):
    linear = installed.decoder.layers[0].self_attention.linear_qkv
    x = torch.randn(64, 1, 256, device="cuda").bfloat16().requires_grad_()
    y, _ = linear(x)
    (y.float() ** 2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert linear.weight.grad is not None and torch.isfinite(linear.weight.grad).all()


@_CUDA
@pytest.mark.parametrize(
    "kwargs",
    [
        {"allreduce_dgrad": True},
        {"sequence_parallel": True},
        {"grad_output_buffer": []},
    ],
)
def test_unsupported_calls_keep_the_megatron_path(installed, kwargs):
    linear = installed.decoder.layers[0].self_attention.linear_qkv
    x = torch.randn(16, 1, 256, device="cuda").bfloat16()
    y, _ = linear(x, **kwargs)
    assert linear.calls == 1
    assert torch.equal(y, torch.matmul(x, linear.weight.t()))


@_CUDA
def test_bias_and_frozen_weight_keep_the_megatron_path(installed):
    linear = installed.decoder.layers[0].self_attention.linear_qkv
    x = torch.randn(16, 1, 256, device="cuda").bfloat16()
    bias = torch.zeros(640, device="cuda", dtype=torch.bfloat16)
    linear._forward_impl(input=x, weight=linear.weight, bias=bias)
    linear.weight.requires_grad_(False)
    linear(x)
    assert linear.calls == 2


@_CUDA
def test_packs_follow_the_expert_generation(installed):
    layer = installed.decoder.layers[0]
    linear, experts = layer.self_attention.linear_qkv, layer.mlp.experts
    cache = linear._lumen_mxfp4_linear
    first = cache.get(linear.weight)
    assert cache.get(linear.weight) is first
    with torch.no_grad():
        linear.weight.data.mul_(2)  # optimizer-style write: _version unchanged
    assert cache.get(linear.weight) is first
    experts._pack_generation += 1
    fresh = cache.get(linear.weight)
    assert fresh is not first
    assert torch.equal(fresh[0][0], attn_linear._aiter_ops()[1](linear.weight)[0][0])


@_CUDA
def test_packs_are_fresh_without_an_active_lifecycle(monkeypatch):
    monkeypatch.setenv("LUMEN_ATTN_LINEAR_RECIPE", "a4w4")
    for model in (_model(layers=1, experts=False), _model(layers=1)):
        experts = getattr(model.decoder.layers[0].mlp, "experts", None)
        if experts is not None:
            experts._pack_cache_enabled = False
        attn_linear.install_mxfp4_attention_linears(model)
        linear = model.decoder.layers[0].self_attention.linear_qkv
        cache = linear._lumen_mxfp4_linear
        assert cache.get(linear.weight) is not cache.get(linear.weight)


@_CUDA
def test_gradient_accumulation_fusion_adds_into_main_grad(installed, monkeypatch):
    calls = []

    def accum_fp32(total_input, grad_output, main_grad):
        calls.append(1)
        main_grad.add_(grad_output.float().t() @ total_input.float())

    kernels = SimpleNamespace(wgrad_gemm_accum_fp32=accum_fp32)
    layers = types.ModuleType("megatron.core.tensor_parallel.layers")
    layers.fused_weight_gradient_mlp_cuda = kernels
    monkeypatch.setitem(sys.modules, "megatron.core.tensor_parallel.layers", layers)
    tp = types.ModuleType("megatron.core.tensor_parallel")
    tp.layers = layers
    monkeypatch.setitem(sys.modules, "megatron.core.tensor_parallel", tp)

    linear = installed.decoder.layers[0].self_attention.linear_qkv
    linear.weight.main_grad = torch.zeros_like(linear.weight, dtype=torch.float32)
    linear.weight.grad_added_to_main_grad = False
    x = torch.randn(32, 1, 256, device="cuda").bfloat16().requires_grad_()
    y, _ = linear(x, gradient_accumulation_fusion=True)
    dy = torch.randn_like(y)
    y.backward(dy)
    assert calls == [1] and linear.weight.grad_added_to_main_grad
    expected = dy.reshape(-1, 640).float().t() @ x.detach().reshape(-1, 256).float()
    torch.testing.assert_close(linear.weight.main_grad, expected)
