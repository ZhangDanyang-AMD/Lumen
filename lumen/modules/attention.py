###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Literal, Optional

import torch

from lumen.ops.attention import attention, attention_fp8_quant
from lumen.quantize import is_aiter_available

__all__ = ["LumenAttention"]


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
        backend_type: str = "aiter_csrc",  # 'aiter_csrc', 'aiter_triton'
        # following parameters will be used in mxfp8
        quant_type: Optional[Literal["fp8_blockwise", "mxfp8"]] = None,  # "fp8", "mxfp8"
        block_m_fwd: int = 64,  # block of query seq len in fwd
        block_n_fwd: int = 64,  # block of key/value seq len in fwd
        block_m_dq_bwd: int = 64,  # block of dq seq len in bwd
        block_n_dq_bwd: int = 64,  # block of dq seq len in bwd
        block_m_dkv_bwd: int = 64,  # block of dkv seq len in bwd
        block_n_dkv_bwd: int = 64,  # block of dkv seq len in bwd
        quant_block_size: int = 32,
        grad_quant_type: Optional[str] = None,
    ):
        super().__init__()

        assert not (
            quant_type is not None and backend_type == "aiter_csrc"
        ), "When quant_type is not None, backend_type cannot be 'aiter_csrc'."

        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.window_size = window_size
        self.alibi_slopes = alibi_slopes
        self.return_lse = return_lse
        self.return_attn_probs = return_attn_probs
        self.deterministic = deterministic
        self.backend_type = backend_type
        # following parameters will be used in mxfp8
        self.quant_type = quant_type
        self.block_m_fwd = block_m_fwd
        self.block_n_fwd = block_n_fwd
        self.block_m_dq_bwd = block_m_dq_bwd
        self.block_n_dq_bwd = block_n_dq_bwd
        self.block_m_dkv_bwd = block_m_dkv_bwd
        self.block_n_dkv_bwd = block_n_dkv_bwd
        self.quant_block_size = quant_block_size
        self.grad_quant_type = grad_quant_type

        if backend_type == "aiter_csrc" and quant_type is None:
            if not is_aiter_available():
                raise RuntimeError(
                    "AITER is not installed. The aiter_csrc backend requires "
                    "'aiter' — install it or use backend_type='aiter_triton'."
                )
            self.attention_fn = attention
        elif backend_type == "aiter_triton":
            self.attention_fn = attention_fp8_quant
        else:
            raise ValueError(f"Unknown attention backend_type: {backend_type}")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        if self.quant_type is not None:
            kwargs = {
                "quant_type": self.quant_type,
                "block_m_fwd": self.block_m_fwd,
                "block_n_fwd": self.block_n_fwd,
                "block_m_dq_bwd": self.block_m_dq_bwd,
                "block_n_dq_bwd": self.block_n_dq_bwd,
                "block_m_dkv_bwd": self.block_m_dkv_bwd,
                "block_n_dkv_bwd": self.block_n_dkv_bwd,
                "quant_block_size": self.quant_block_size,
            }
        else:
            kwargs = {}

        if self.grad_quant_type is not None:
            kwargs["grad_quant_type"] = self.grad_quant_type

        return self.attention_fn(
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
