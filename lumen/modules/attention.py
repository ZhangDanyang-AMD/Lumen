###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch

from lumen.ops.attention import attention, attention_fp8_quant
from lumen.quantize import is_aiter_available

__all__ = ["LumenAttention"]


_FP8_BACKENDS = ("aiter_triton_fp8", "aiter_csrc_fp8", "aiter_asm_fp8")


class LumenAttention(torch.nn.Module):
    def __init__(
        self,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        alibi_slopes=None,
        deterministic=False,
        return_lse=False,
        return_attn_probs=False,
        backend_type: str = "aiter_csrc",
        quant_type: Optional[str] = None,
        block_m_fwd: int = 64,
        block_n_fwd: int = 64,
        block_m_dq_bwd: int = 64,
        block_n_dq_bwd: int = 64,
        block_m_dkv_bwd: int = 64,
        block_n_dkv_bwd: int = 64,
        quant_block_size: int = 32,
        grad_quant_type: Optional[str] = None,
    ):
        super().__init__()

        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.window_size = window_size
        self.alibi_slopes = alibi_slopes
        self.return_lse = return_lse
        self.return_attn_probs = return_attn_probs
        self.deterministic = deterministic
        self.backend_type = backend_type
        self.quant_type = quant_type
        self.block_m_fwd = block_m_fwd
        self.block_n_fwd = block_n_fwd
        self.block_m_dq_bwd = block_m_dq_bwd
        self.block_n_dq_bwd = block_n_dq_bwd
        self.block_m_dkv_bwd = block_m_dkv_bwd
        self.block_n_dkv_bwd = block_n_dkv_bwd
        self.quant_block_size = quant_block_size
        self.grad_quant_type = grad_quant_type

        self._is_fp8 = backend_type in _FP8_BACKENDS or quant_type is not None

        if not self._is_fp8:
            if backend_type == "aiter_csrc" and not is_aiter_available():
                raise RuntimeError(
                    "AITER is not installed. The aiter_csrc backend requires "
                    "'aiter' — install it or use backend_type='aiter_triton'."
                )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        if self._is_fp8:
            kwargs = {
                "backend_type": self.backend_type,
                "quant_type": self.quant_type or "blockwise",
                "block_m_fwd": self.block_m_fwd,
                "block_n_fwd": self.block_n_fwd,
                "block_m_dq_bwd": self.block_m_dq_bwd,
                "block_n_dq_bwd": self.block_n_dq_bwd,
                "block_m_dkv_bwd": self.block_m_dkv_bwd,
                "block_n_dkv_bwd": self.block_n_dkv_bwd,
                "quant_block_size": self.quant_block_size,
            }
            if self.grad_quant_type is not None:
                kwargs["grad_quant_type"] = self.grad_quant_type
            return attention_fp8_quant(
                q,
                k,
                v,
                dropout_p=self.dropout_p,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
                window_size=self.window_size,
                bias=bias,
                alibi_slopes=self.alibi_slopes,
                deterministic=self.deterministic,
                return_lse=self.return_lse,
                return_attn_probs=self.return_attn_probs,
                **kwargs,
            )

        kwargs = {}
        if self.grad_quant_type is not None:
            kwargs["grad_quant_type"] = self.grad_quant_type
        return attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            window_size=self.window_size,
            bias=bias,
            alibi_slopes=self.alibi_slopes,
            deterministic=self.deterministic,
            return_lse=self.return_lse,
            return_attn_probs=self.return_attn_probs,
            backend_type=self.backend_type,
            **kwargs,
        )
