###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Unit tests for AITER calling interfaces in transformer_light.

These tests use ``unittest.mock`` to verify that the correct AITER functions
are invoked with correct arguments when the ``"aiter"`` backend is selected.
No actual GPU or running training is required -- all external AITER calls are
mocked.

Covers:
    1. Backend detection: ``is_aiter_available``, ``get_attention_backend``,
       ``get_quant_backend``
    2. Attention: ``aiter.ops.mha.flash_attn_func`` via ``attention()``
    3. Quantized linear: ``per_token_quant_hip`` + ``hipb_mm`` via
       ``quantized_linear()`` / ``QuantizedLinearFunction``
    4. RMSNorm: ``aiter.ops.triton.normalization.rmsnorm.rms_norm``
    5. Module wrappers: ``TransformerLightAttention``,
       ``TransformerLightLinear``
    6. ``quant.enable()`` / ``quant.disable()`` patching
    7. Gradient quantization utility dispatch
    8. API signature verification

Run::

    pytest tests/module/test_aiter_interfaces.py -v
"""

import inspect
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


# =========================================================================
# 1. Backend detection
# =========================================================================


class TestIsAiterAvailable:
    """Tests for ``transformer_light.quantize.is_aiter_available``."""

    def test_returns_true_when_aiter_importable(self):
        from transformer_light.quantize import is_aiter_available
        is_aiter_available.cache_clear()
        with patch.dict(sys.modules, {"aiter": MagicMock()}):
            assert is_aiter_available() is True
        is_aiter_available.cache_clear()

    def test_returns_false_when_aiter_missing(self):
        from transformer_light.quantize import is_aiter_available
        is_aiter_available.cache_clear()
        # Setting a module to None in sys.modules causes import to raise ImportError
        with patch.dict(sys.modules, {"aiter": None}):
            assert is_aiter_available() is False
        is_aiter_available.cache_clear()

    def test_result_is_cached(self):
        from transformer_light.quantize import is_aiter_available
        is_aiter_available.cache_clear()
        with patch.dict(sys.modules, {"aiter": MagicMock()}):
            first = is_aiter_available()
        # Second call should return cached result even though context exited
        second = is_aiter_available()
        assert first == second
        is_aiter_available.cache_clear()


class TestGetAttentionBackend:
    """Tests for ``transformer_light.quantize.get_attention_backend``."""

    def test_triton_explicit(self):
        from transformer_light.quantize import get_attention_backend
        assert get_attention_backend("triton") == "triton"

    def test_aiter_explicit_available(self):
        from transformer_light.quantize import get_attention_backend
        with patch("transformer_light.quantize.is_aiter_available", return_value=True):
            assert get_attention_backend("aiter") == "aiter"

    def test_aiter_explicit_unavailable_raises(self):
        from transformer_light.quantize import get_attention_backend
        with patch("transformer_light.quantize.is_aiter_available", return_value=False):
            with pytest.raises(RuntimeError, match="AITER is not installed"):
                get_attention_backend("aiter")

    def test_auto_with_aiter(self):
        from transformer_light.quantize import get_attention_backend
        with patch("transformer_light.quantize.is_aiter_available", return_value=True):
            assert get_attention_backend("auto") == "aiter"

    def test_auto_without_aiter(self):
        from transformer_light.quantize import get_attention_backend
        with patch("transformer_light.quantize.is_aiter_available", return_value=False):
            assert get_attention_backend("auto") == "triton"


class TestGetQuantBackend:
    """Tests for ``transformer_light.quantize.get_quant_backend``."""

    def test_triton_explicit(self):
        from transformer_light.quantize import get_quant_backend
        assert get_quant_backend("triton") == "triton"

    def test_aiter_explicit_available(self):
        from transformer_light.quantize import get_quant_backend
        with patch("transformer_light.quantize.is_aiter_available", return_value=True):
            assert get_quant_backend("aiter") == "aiter"

    def test_aiter_explicit_unavailable_raises(self):
        from transformer_light.quantize import get_quant_backend
        with patch("transformer_light.quantize.is_aiter_available", return_value=False):
            with pytest.raises(RuntimeError, match="AITER is not installed"):
                get_quant_backend("aiter")

    def test_auto_with_aiter(self):
        from transformer_light.quantize import get_quant_backend
        with patch("transformer_light.quantize.is_aiter_available", return_value=True):
            assert get_quant_backend("auto") == "aiter"

    def test_auto_without_aiter(self):
        from transformer_light.quantize import get_quant_backend
        with patch("transformer_light.quantize.is_aiter_available", return_value=False):
            assert get_quant_backend("auto") == "triton"


# =========================================================================
# 2. Attention -- aiter flash_attn_func interface
# =========================================================================


class TestAttentionAiterInterface:
    """Verify that ``attention()`` calls ``flash_attn_func`` correctly
    when ``backend_type="aiter"``."""

    def test_attention_calls_flash_attn_func_with_correct_kwargs(self):
        B, S, H, D = 2, 128, 8, 64
        q = torch.randn(B, S, H, D)
        k = torch.randn(B, S, H, D)
        v = torch.randn(B, S, H, D)
        mock_output = torch.randn(B, S, H, D)
        scale = D ** -0.5

        mock_flash = MagicMock(return_value=mock_output)

        with patch("transformer_light.ops.attention.attention.is_aiter_available",
                    return_value=True), \
             patch("transformer_light.ops.attention.attention.flash_attn_func",
                    mock_flash, create=True):
            from transformer_light.ops.attention.attention import attention
            result = attention(
                q, k, v,
                dropout_p=0.1,
                softmax_scale=scale,
                causal=True,
                window_size=(-1, -1),
                backend_type="aiter",
            )

        mock_flash.assert_called_once()
        kw = mock_flash.call_args[1]
        assert kw["dropout_p"] == 0.1
        assert kw["softmax_scale"] == scale
        assert kw["causal"] is True
        assert kw["window_size"] == (-1, -1)
        assert kw["deterministic"] is True
        assert kw["return_lse"] is False
        # return_attn_probs = False when dropout_p=0.1 and return_attn_probs=False
        assert kw["return_attn_probs"] is False
        assert result is mock_output

    def test_attention_passes_bias_and_alibi_slopes(self):
        B, S, H, D = 1, 64, 4, 32
        q = torch.randn(B, S, H, D)
        k = torch.randn(B, S, H, D)
        v = torch.randn(B, S, H, D)
        bias = torch.randn(H, S, S)
        alibi = torch.randn(H)

        mock_flash = MagicMock(return_value=torch.randn(B, S, H, D))
        with patch("transformer_light.ops.attention.attention.is_aiter_available",
                    return_value=True), \
             patch("transformer_light.ops.attention.attention.flash_attn_func",
                    mock_flash, create=True):
            from transformer_light.ops.attention.attention import attention
            attention(q, k, v, bias=bias, alibi_slopes=alibi,
                      backend_type="aiter")

        kw = mock_flash.call_args[1]
        assert torch.equal(kw["bias"], bias)
        assert torch.equal(kw["alibi_slopes"], alibi)

    def test_attention_aiter_raises_when_unavailable(self):
        with patch("transformer_light.ops.attention.attention.is_aiter_available",
                    return_value=False):
            from transformer_light.ops.attention.attention import attention
            with pytest.raises(RuntimeError, match="AITER is not installed"):
                attention(torch.randn(2, 128, 8, 64),
                          torch.randn(2, 128, 8, 64),
                          torch.randn(2, 128, 8, 64),
                          backend_type="aiter")

    def test_attention_auto_selects_aiter_when_available(self):
        B, S, H, D = 2, 64, 4, 32
        mock_flash = MagicMock(return_value=torch.randn(B, S, H, D))
        with patch("transformer_light.ops.attention.attention.is_aiter_available",
                    return_value=True), \
             patch("transformer_light.ops.attention.attention.flash_attn_func",
                    mock_flash, create=True):
            from transformer_light.ops.attention.attention import attention
            attention(torch.randn(B, S, H, D),
                      torch.randn(B, S, H, D),
                      torch.randn(B, S, H, D),
                      backend_type="auto")
        mock_flash.assert_called_once()

    def test_attention_return_lse_forwarded(self):
        B, S, H, D = 1, 32, 2, 16
        mock_out = torch.randn(B, S, H, D)
        mock_lse = torch.randn(B, H, S)
        mock_flash = MagicMock(return_value=(mock_out, mock_lse))
        with patch("transformer_light.ops.attention.attention.is_aiter_available",
                    return_value=True), \
             patch("transformer_light.ops.attention.attention.flash_attn_func",
                    mock_flash, create=True):
            from transformer_light.ops.attention.attention import attention
            attention(torch.randn(B, S, H, D),
                      torch.randn(B, S, H, D),
                      torch.randn(B, S, H, D),
                      return_lse=True, backend_type="aiter")
        assert mock_flash.call_args[1]["return_lse"] is True

    def test_attention_return_attn_probs_with_dropout(self):
        B, S, H, D = 1, 32, 2, 16
        mock_flash = MagicMock(return_value=torch.randn(B, S, H, D))
        with patch("transformer_light.ops.attention.attention.is_aiter_available",
                    return_value=True), \
             patch("transformer_light.ops.attention.attention.flash_attn_func",
                    mock_flash, create=True):
            from transformer_light.ops.attention.attention import attention
            attention(torch.randn(B, S, H, D),
                      torch.randn(B, S, H, D),
                      torch.randn(B, S, H, D),
                      dropout_p=0.5, return_attn_probs=True,
                      backend_type="aiter")
        # _return_softmax = True because dropout_p > 0 and return_attn_probs=True
        assert mock_flash.call_args[1]["return_attn_probs"] is True

    def test_attention_accepts_grad_quant_type(self):
        B, S, H, D = 2, 64, 4, 32
        mock_flash = MagicMock(return_value=torch.randn(B, S, H, D))
        with patch("transformer_light.ops.attention.attention.is_aiter_available",
                    return_value=True), \
             patch("transformer_light.ops.attention.attention.flash_attn_func",
                    mock_flash, create=True):
            from transformer_light.ops.attention.attention import attention
            attention(torch.randn(B, S, H, D),
                      torch.randn(B, S, H, D),
                      torch.randn(B, S, H, D),
                      backend_type="aiter", grad_quant_type="fp8")
        mock_flash.assert_called_once()

    def test_attention_fp8_quant_rejects_aiter_backend(self):
        from transformer_light.ops.attention.attention import attention_fp8_quant
        with pytest.raises(AssertionError):
            attention_fp8_quant(
                torch.randn(2, 64, 4, 32),
                torch.randn(2, 64, 4, 32),
                torch.randn(2, 64, 4, 32),
                backend_type="aiter",
            )


class TestAttentionAiterCPA2AInterface:
    """Verify that the CP A2A aiter variant calls ``flash_attn_func``."""

    def test_aiter_cpa2a_calls_flash_attn_func(self):
        B, S_local, H, D = 2, 64, 8, 32
        cp_size = 2
        q = torch.randn(B, S_local, H, D)
        k = torch.randn(B, S_local, H, D)
        v = torch.randn(B, S_local, H, D)
        scale = D ** -0.5

        mock_flash = MagicMock()
        mock_output = torch.randn(B, S_local, H // cp_size, D)
        mock_lse = torch.randn(B, H // cp_size, S_local)
        mock_flash.return_value = (mock_output, mock_lse)

        mock_cp_group = MagicMock()
        mock_cp_group.size.return_value = cp_size

        with patch("transformer_light.ops.attention.attention_with_cp_a2a.flash_attn_func",
                    mock_flash, create=True), \
             patch("transformer_light.ops.attention.attention_with_cp_a2a.torch.distributed.all_to_all_single",
                   side_effect=lambda out, inp, **kw: out.copy_(inp)):
            from transformer_light.ops.attention.attention_with_cp_a2a import (
                AttentionAiterFunctionCPA2A,
            )
            AttentionAiterFunctionCPA2A.apply(
                q, k, v,
                0.0, scale, True, (-1, -1),
                None, None, False, False, False,
                False, mock_cp_group,
            )

        mock_flash.assert_called_once()
        kw = mock_flash.call_args[1]
        assert kw["dropout_p"] == 0.0
        assert kw["softmax_scale"] == scale
        assert kw["causal"] is True
        assert kw["return_lse"] is True


# =========================================================================
# 3. Quantized Linear -- aiter quant + mm interface
# =========================================================================


class TestQuantizedLinearAiterInterface:
    """Verify that ``quantized_linear()`` calls AITER quant/mm helpers when
    backend='aiter'."""

    def test_forward_calls_aiter_quant_and_mm(self):
        in_f, out_f = 64, 32
        x = torch.randn(2, 8, in_f)
        w = torch.randn(out_f, in_f)

        x_fp8 = torch.randn(16, in_f)
        x_scale = torch.tensor([1.0])
        mock_quant = MagicMock(return_value=(x_fp8, x_scale))
        mock_mm = MagicMock(return_value=torch.randn(16, out_f))

        mock_sm = MagicMock()
        w_fp8 = torch.randn_like(w)
        w_scale = torch.tensor([1.0])
        mock_sm.quantize.return_value = (w_fp8, w_scale)

        with patch("transformer_light.ops.quantize.linear._aiter_quant", mock_quant), \
             patch("transformer_light.ops.quantize.linear._aiter_mm", mock_mm), \
             patch("transformer_light.ops.quantize.linear.is_aiter_available",
                    return_value=True):
            from transformer_light.ops.quantize.linear import quantized_linear
            quantized_linear(
                x, w, None,
                scaling_manager=mock_sm,
                backend="aiter",
                fp8_dtype=torch.float8_e4m3fn,
                block_size=128,
            )

        mock_quant.assert_called_once()
        assert mock_quant.call_args[0][1] == torch.float8_e4m3fn
        mock_sm.quantize.assert_called_once()

    def test_forward_aiter_raises_when_unavailable(self):
        from transformer_light.ops.quantize.linear import quantized_linear
        with patch("transformer_light.ops.quantize.linear.is_aiter_available",
                    return_value=False):
            with pytest.raises(RuntimeError, match="AITER is not installed"):
                quantized_linear(
                    torch.randn(2, 64),
                    torch.randn(32, 64),
                    None,
                    backend="aiter",
                )

    def test_aiter_quant_delegates_to_per_token_quant_hip(self):
        mock_ptq = MagicMock(return_value=(torch.randn(4, 8), torch.ones(4, 1)))
        with patch.dict(sys.modules, {
            "aiter": MagicMock(),
            "aiter.ops": MagicMock(),
            "aiter.ops.quant": MagicMock(per_token_quant_hip=mock_ptq),
        }):
            from transformer_light.ops.quantize.linear import _aiter_quant
            x = torch.randn(4, 8)
            _aiter_quant(x, torch.float8_e4m3fn)
        mock_ptq.assert_called_once_with(x, quant_dtype=torch.float8_e4m3fn)

    def test_aiter_mm_delegates_to_hipb_mm(self):
        mock_hipb = MagicMock(return_value=torch.randn(4, 8))
        with patch.dict(sys.modules, {
            "aiter": MagicMock(),
            "aiter.ops": MagicMock(),
            "aiter.ops.gradlib": MagicMock(hipb_mm=mock_hipb),
        }):
            from transformer_light.ops.quantize.linear import _aiter_mm
            a, b = torch.randn(4, 8), torch.randn(8, 16)
            sa, sb = torch.tensor(1.0), torch.tensor(1.0)
            _aiter_mm(a, b, sa, sb)
        mock_hipb.assert_called_once_with(a, b, scaleA=sa, scaleB=sb)


# =========================================================================
# 4. RMSNorm -- aiter rms_norm interface
# =========================================================================


class TestRMSNormAiterInterface:
    """Verify that ``rmsnorm()`` calls ``_aiter_rms_norm`` when the AITER
    backend is active."""

    def test_rmsnorm_calls_aiter_backend(self):
        hidden = 64
        x = torch.randn(2, 8, hidden)
        weight = torch.randn(hidden)

        mock_rms = MagicMock(return_value=torch.randn(16, hidden))
        with patch("transformer_light.ops.normalization.rmsnorm._USE_AITER", True), \
             patch("transformer_light.ops.normalization.rmsnorm._aiter_rms_norm", mock_rms):
            from transformer_light.ops.normalization.rmsnorm import rmsnorm
            rmsnorm(x, weight, eps=1e-6)

        mock_rms.assert_called_once()
        args = mock_rms.call_args[0]
        assert args[0].shape == (16, hidden)  # x reshaped to 2D
        assert torch.equal(args[1], weight)
        assert args[2] == 1e-6

    def test_rmsnorm_falls_back_to_torch(self):
        hidden = 64
        x = torch.randn(2, 8, hidden)
        weight = torch.ones(hidden)

        with patch("transformer_light.ops.normalization.rmsnorm._USE_AITER", False):
            from transformer_light.ops.normalization.rmsnorm import rmsnorm
            result = rmsnorm(x, weight, eps=1e-6)
        assert result.shape == x.shape

    def test_rmsnorm_grad_quant_uses_aiter_forward(self):
        hidden = 64
        x = torch.randn(2, 8, hidden, requires_grad=True)
        weight = torch.randn(hidden, requires_grad=True)

        mock_rms = MagicMock(return_value=torch.randn(16, hidden))
        with patch("transformer_light.ops.normalization.rmsnorm._USE_AITER", True), \
             patch("transformer_light.ops.normalization.rmsnorm._aiter_rms_norm", mock_rms):
            from transformer_light.ops.normalization.rmsnorm import rmsnorm
            rmsnorm(x, weight, eps=1e-6, grad_quant_type="fp8")
        # _RMSNormGradQuant.forward calls _aiter_rms_norm
        mock_rms.assert_called()

    def test_rmsnorm_module_delegates_to_functional(self):
        hidden = 64
        x = torch.randn(2, 8, hidden)

        mock_rms = MagicMock(return_value=torch.randn(16, hidden))
        with patch("transformer_light.ops.normalization.rmsnorm._USE_AITER", True), \
             patch("transformer_light.ops.normalization.rmsnorm._aiter_rms_norm", mock_rms):
            from transformer_light.ops.normalization.rmsnorm import TransformerLightRMSNorm
            norm = TransformerLightRMSNorm(hidden, eps=1e-5)
            norm(x)
        mock_rms.assert_called_once()

    def test_rmsnorm_module_extra_repr_with_grad_quant(self):
        from transformer_light.ops.normalization.rmsnorm import TransformerLightRMSNorm
        norm = TransformerLightRMSNorm(64, grad_quant_type="mxfp8")
        r = norm.extra_repr()
        assert "grad_quant=mxfp8" in r

    def test_rmsnorm_module_extra_repr_without_grad_quant(self):
        from transformer_light.ops.normalization.rmsnorm import TransformerLightRMSNorm
        norm = TransformerLightRMSNorm(64)
        assert "grad_quant" not in norm.extra_repr()


# =========================================================================
# 5. Module wrappers -- aiter availability checks
# =========================================================================


class TestTransformerLightAttentionModule:
    """Tests for ``modules.attention.TransformerLightAttention``."""

    def test_aiter_backend_requires_aiter(self):
        with patch("transformer_light.modules.attention.is_aiter_available",
                    return_value=False):
            from transformer_light.modules.attention import TransformerLightAttention
            with pytest.raises(RuntimeError, match="AITER is not installed"):
                TransformerLightAttention(backend_type="aiter", quant_type=None)

    def test_aiter_backend_succeeds_when_available(self):
        with patch("transformer_light.modules.attention.is_aiter_available",
                    return_value=True):
            from transformer_light.modules.attention import TransformerLightAttention
            attn = TransformerLightAttention(backend_type="aiter", quant_type=None)
            assert attn.backend_type == "aiter"

    def test_triton_backend_no_aiter_check(self):
        from transformer_light.modules.attention import TransformerLightAttention
        attn = TransformerLightAttention(backend_type="triton", quant_type=None)
        assert attn.backend_type == "triton"

    def test_aiter_plus_quant_type_raises(self):
        with patch("transformer_light.modules.attention.is_aiter_available",
                    return_value=True):
            from transformer_light.modules.attention import TransformerLightAttention
            with pytest.raises(AssertionError):
                TransformerLightAttention(
                    backend_type="aiter", quant_type="fp8_blockwise",
                )

    def test_grad_quant_type_stored(self):
        from transformer_light.modules.attention import TransformerLightAttention
        attn = TransformerLightAttention(
            backend_type="triton", quant_type="fp8_blockwise",
            grad_quant_type="mxfp8",
        )
        assert attn.grad_quant_type == "mxfp8"

    def test_forward_passes_grad_quant_type(self):
        from transformer_light.modules.attention import TransformerLightAttention
        attn = TransformerLightAttention(
            backend_type="triton", quant_type="fp8_blockwise",
            grad_quant_type="fp8",
        )
        mock_fn = MagicMock(return_value=torch.randn(2, 4, 8, 32))
        attn.attention_fn = mock_fn

        attn(torch.randn(2, 4, 8, 32),
             torch.randn(2, 4, 8, 32),
             torch.randn(2, 4, 8, 32))

        kw = mock_fn.call_args[1]
        assert kw["grad_quant_type"] == "fp8"

    def test_forward_no_grad_quant_when_none(self):
        from transformer_light.modules.attention import TransformerLightAttention
        attn = TransformerLightAttention(
            backend_type="triton", quant_type="fp8_blockwise",
        )
        mock_fn = MagicMock(return_value=torch.randn(2, 4, 8, 32))
        attn.attention_fn = mock_fn

        attn(torch.randn(2, 4, 8, 32),
             torch.randn(2, 4, 8, 32),
             torch.randn(2, 4, 8, 32))

        kw = mock_fn.call_args[1]
        assert "grad_quant_type" not in kw

    def test_attention_fn_is_attention_for_aiter(self):
        """When backend_type='aiter', attention_fn should be the plain
        ``attention`` function (not ``attention_fp8_quant``)."""
        with patch("transformer_light.modules.attention.is_aiter_available",
                    return_value=True):
            from transformer_light.modules.attention import TransformerLightAttention
            from transformer_light.ops.attention import attention
            attn = TransformerLightAttention(backend_type="aiter", quant_type=None)
            assert attn.attention_fn is attention

    def test_attention_fn_is_fp8_quant_for_triton(self):
        """When backend_type='triton', attention_fn should be
        ``attention_fp8_quant``."""
        from transformer_light.modules.attention import TransformerLightAttention
        from transformer_light.ops.attention import attention_fp8_quant
        attn = TransformerLightAttention(
            backend_type="triton", quant_type="fp8_blockwise",
        )
        assert attn.attention_fn is attention_fp8_quant


class TestTransformerLightLinearModule:
    """Tests for ``modules.quantize.TransformerLightLinear``."""

    def test_aiter_backend_requires_aiter(self):
        with patch("transformer_light.modules.quantize.is_aiter_available",
                    return_value=False):
            from transformer_light.modules.quantize import TransformerLightLinear
            with pytest.raises(RuntimeError, match="AITER is not installed"):
                TransformerLightLinear(64, 32, backend_type="aiter")

    def test_aiter_backend_succeeds_when_available(self):
        with patch("transformer_light.modules.quantize.is_aiter_available",
                    return_value=True):
            from transformer_light.modules.quantize import TransformerLightLinear
            linear = TransformerLightLinear(64, 32, backend_type="aiter")
            assert linear.backend_type == "aiter"
            assert linear.weight.shape == (32, 64)

    def test_triton_backend_no_aiter_check(self):
        from transformer_light.modules.quantize import TransformerLightLinear
        linear = TransformerLightLinear(64, 32, backend_type="triton")
        assert linear.backend_type == "triton"

    def test_bias_creation(self):
        from transformer_light.modules.quantize import TransformerLightLinear
        with_bias = TransformerLightLinear(64, 32, backend_type="triton", bias=True)
        assert with_bias.bias is not None
        assert with_bias.bias.shape == (32,)

        no_bias = TransformerLightLinear(64, 32, backend_type="triton", bias=False)
        assert no_bias.bias is None

    def test_extra_repr_content(self):
        from transformer_light.modules.quantize import TransformerLightLinear
        linear = TransformerLightLinear(64, 32, backend_type="triton")
        r = linear.extra_repr()
        assert "in_features=64" in r
        assert "out_features=32" in r
        assert "backend=triton" in r

    def test_forward_delegates_to_quantized_linear(self):
        from transformer_light.modules.quantize import TransformerLightLinear
        linear = TransformerLightLinear(128, 64, backend_type="triton")
        mock_ql = MagicMock(return_value=torch.randn(2, 64))
        with patch("transformer_light.modules.quantize.quantized_linear", mock_ql):
            linear(torch.randn(2, 128))
        mock_ql.assert_called_once()
        kw = mock_ql.call_args[1]
        assert kw["backend"] == "triton"


# =========================================================================
# 6. quant.enable() / quant.disable()
# =========================================================================


class TestQuantEnable:
    """Tests for ``transformer_light.quantize.enable()`` /
    ``transformer_light.quantize.disable()``."""

    def _make_model(self):
        return nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

    def test_enable_patches_all_linear_layers(self):
        import transformer_light.quantize as quant
        model = self._make_model()
        with patch("transformer_light.quantize.get_quant_backend",
                    return_value="triton"):
            quant.enable(model, format="fp8_e4m3", scaling="dynamic")

        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        assert len(linears) == 2
        for m in linears:
            assert m._quant_enabled is True

    def test_enable_sets_unique_tensor_ids(self):
        import transformer_light.quantize as quant
        model = self._make_model()
        with patch("transformer_light.quantize.get_quant_backend",
                    return_value="triton"):
            quant.enable(model, format="fp8_e4m3", scaling="dynamic")

        ids = [m._quant_tensor_id
               for m in model.modules() if isinstance(m, nn.Linear)]
        assert len(set(ids)) == 2  # all unique

    def test_enable_with_aiter_backend(self):
        import transformer_light.quantize as quant
        model = self._make_model()
        with patch("transformer_light.quantize.is_aiter_available",
                    return_value=True), \
             patch("transformer_light.quantize.get_quant_backend",
                    return_value="aiter"):
            quant.enable(model, format="fp8_e4m3", scaling="dynamic",
                         backend="aiter")

        for m in model.modules():
            if isinstance(m, nn.Linear):
                assert m._quant_backend == "aiter"

    def test_enable_with_triton_backend(self):
        import transformer_light.quantize as quant
        model = self._make_model()
        with patch("transformer_light.quantize.get_quant_backend",
                    return_value="triton"):
            quant.enable(model, format="fp8_e4m3", scaling="dynamic",
                         backend="triton")

        for m in model.modules():
            if isinstance(m, nn.Linear):
                assert m._quant_backend == "triton"

    def test_enable_returns_scaling_manager(self):
        import transformer_light.quantize as quant
        from transformer_light.quantize import ScalingManager
        model = self._make_model()
        with patch("transformer_light.quantize.get_quant_backend",
                    return_value="triton"):
            mgr = quant.enable(model, format="fp8_e4m3", scaling="dynamic")
        assert isinstance(mgr, ScalingManager)

    def test_enable_with_config_object(self):
        import transformer_light.quantize as quant
        from transformer_light.quantize import QuantConfig, ScalingManager
        model = self._make_model()
        config = QuantConfig(quantize_grad="fp8")
        with patch("transformer_light.quantize.get_quant_backend",
                    return_value="triton"):
            mgr = quant.enable(model, config=config)
        assert isinstance(mgr, ScalingManager)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                assert m._quant_enabled is True

    def test_disable_removes_hooks(self):
        import transformer_light.quantize as quant
        model = self._make_model()
        with patch("transformer_light.quantize.get_quant_backend",
                    return_value="triton"):
            quant.enable(model, format="fp8_e4m3", scaling="dynamic")

        quant.disable(model)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                assert m._quant_enabled is False
                assert not hasattr(m, "_quant_hook_handle")

    def test_disable_idempotent(self):
        """Calling disable on a model that was never enabled should not error."""
        import transformer_light.quantize as quant
        model = self._make_model()
        quant.disable(model)  # should not raise


# =========================================================================
# 7. Gradient quantization utility
# =========================================================================


class TestGradQuantUtility:
    """Tests for ``transformer_light.core.grad_quant.quantize_grad_tensor``."""

    def test_none_is_noop(self):
        from transformer_light.core.grad_quant import quantize_grad_tensor
        t = torch.randn(4, 8)
        assert quantize_grad_tensor(t, None) is t

    def test_fp8_dispatches_to_round_to_fp8(self):
        from transformer_light.core.grad_quant import quantize_grad_tensor
        t = torch.randn(4, 8)
        expected = torch.randn(4, 8)
        with patch("transformer_light.core.grad_quant._round_to_fp8",
                    return_value=expected) as mock_round:
            result = quantize_grad_tensor(t, "fp8",
                                          fp8_dtype=torch.float8_e4m3fn)
        mock_round.assert_called_once_with(t, torch.float8_e4m3fn)
        assert result is expected

    def test_fp8_auto_detects_dtype_when_none(self):
        from transformer_light.core.grad_quant import quantize_grad_tensor
        t = torch.randn(4, 8)
        mock_dtype = torch.float8_e4m3fn
        with patch("transformer_light.core.grad_quant._round_to_fp8",
                    return_value=t) as mock_round, \
             patch("transformer_light.quantize.config._get_float8_e4m3",
                    return_value=mock_dtype):
            quantize_grad_tensor(t, "fp8")
        assert mock_round.call_args[0][1] == mock_dtype

    def test_mxfp8_dispatches_to_round_to_mxfp8(self):
        from transformer_light.core.grad_quant import quantize_grad_tensor
        t = torch.randn(4, 8)
        expected = torch.randn(4, 8)
        with patch("transformer_light.core.grad_quant._round_to_mxfp8",
                    return_value=expected) as mock_round:
            result = quantize_grad_tensor(t, "mxfp8", block_size=64)
        mock_round.assert_called_once_with(t, block_size=64)
        assert result is expected

    def test_fp4_raises_not_implemented(self):
        from transformer_light.core.grad_quant import quantize_grad_tensor
        with pytest.raises(NotImplementedError, match="FP4"):
            quantize_grad_tensor(torch.randn(4, 8), "fp4")

    def test_unknown_type_raises_value_error(self):
        from transformer_light.core.grad_quant import quantize_grad_tensor
        with pytest.raises(ValueError, match="Unknown grad_quant_type"):
            quantize_grad_tensor(torch.randn(4, 8), "bfloat4")

    def test_grad_quant_types_constant(self):
        from transformer_light.core.grad_quant import GRAD_QUANT_TYPES
        assert None in GRAD_QUANT_TYPES
        assert "fp8" in GRAD_QUANT_TYPES
        assert "mxfp8" in GRAD_QUANT_TYPES
        assert "fp4" in GRAD_QUANT_TYPES


# =========================================================================
# 8. QuantConfig
# =========================================================================


class TestQuantConfig:
    """Tests for ``QuantConfig.quantize_grad`` field."""

    def test_default_is_none(self):
        from transformer_light.quantize import QuantConfig
        assert QuantConfig().quantize_grad is None

    def test_accepts_fp8(self):
        from transformer_light.quantize import QuantConfig
        assert QuantConfig(quantize_grad="fp8").quantize_grad == "fp8"

    def test_accepts_mxfp8(self):
        from transformer_light.quantize import QuantConfig
        assert QuantConfig(quantize_grad="mxfp8").quantize_grad == "mxfp8"

    def test_accepts_none_explicitly(self):
        from transformer_light.quantize import QuantConfig
        assert QuantConfig(quantize_grad=None).quantize_grad is None


# =========================================================================
# 9. API signature verification
# =========================================================================


class TestAPISignatures:
    """Verify that public API functions have the expected parameters,
    ensuring interface compatibility across modules."""

    def test_attention_signature(self):
        from transformer_light.ops.attention.attention import attention
        sig = inspect.signature(attention)
        params = sig.parameters
        assert "grad_quant_type" in params
        assert params["grad_quant_type"].default is None
        assert "backend_type" in params
        assert params["backend_type"].default == "auto"
        assert "cp_param_bundle" in params

    def test_attention_fp8_quant_signature(self):
        from transformer_light.ops.attention.attention import attention_fp8_quant
        sig = inspect.signature(attention_fp8_quant)
        params = sig.parameters
        assert "grad_quant_type" in params
        assert params["grad_quant_type"].default is None
        assert "quant_type" in params

    def test_quantized_linear_signature(self):
        from transformer_light.ops.quantize.linear import quantized_linear
        sig = inspect.signature(quantized_linear)
        expected = {
            "input", "weight", "bias", "scaling_manager", "backend",
            "fp8_dtype", "block_size", "tensor_id",
            "quantize_activation", "grad_quant_type",
        }
        for p in expected:
            assert p in sig.parameters, f"Missing parameter: {p}"
        assert sig.parameters["grad_quant_type"].default is None
        assert sig.parameters["backend"].default == "triton"

    def test_rmsnorm_signature(self):
        from transformer_light.ops.normalization.rmsnorm import rmsnorm
        sig = inspect.signature(rmsnorm)
        params = sig.parameters
        assert "grad_quant_type" in params
        assert params["grad_quant_type"].default is None
        assert "x" in params
        assert "weight" in params
        assert "eps" in params

    def test_transformer_light_attention_init_signature(self):
        from transformer_light.modules.attention import TransformerLightAttention
        sig = inspect.signature(TransformerLightAttention.__init__)
        params = sig.parameters
        assert "backend_type" in params
        assert "quant_type" in params
        assert "grad_quant_type" in params
        assert params["backend_type"].default == "aiter"

    def test_transformer_light_rmsnorm_init_signature(self):
        from transformer_light.ops.normalization.rmsnorm import TransformerLightRMSNorm
        sig = inspect.signature(TransformerLightRMSNorm.__init__)
        params = sig.parameters
        assert "hidden_size" in params
        assert "eps" in params
        assert "grad_quant_type" in params

    def test_quantize_grad_tensor_signature(self):
        from transformer_light.core.grad_quant import quantize_grad_tensor
        sig = inspect.signature(quantize_grad_tensor)
        params = sig.parameters
        assert "tensor" in params
        assert "grad_quant_type" in params
        assert "fp8_dtype" in params
        assert params["fp8_dtype"].default is None
        assert "block_size" in params
        assert params["block_size"].default == 32

    def test_enable_signature(self):
        import transformer_light.quantize as quant
        sig = inspect.signature(quant.enable)
        params = sig.parameters
        assert "model" in params
        assert "config" in params
        assert "format" in params
        assert "scaling" in params
        assert "backend" in params
        assert params["backend"].default == "auto"

    def test_is_aiter_available_signature(self):
        from transformer_light.quantize import is_aiter_available
        sig = inspect.signature(is_aiter_available)
        assert len(sig.parameters) == 0
