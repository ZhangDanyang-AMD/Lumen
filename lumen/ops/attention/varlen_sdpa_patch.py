###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Replace causal SDPA with ``flash_attn_varlen_func``.

Hugging Face calls SDPA with ``(batch, heads, tokens, dim)``. The varlen
kernel takes ``(tokens, heads, dim)`` plus cumulative sequence lengths.
"""

from __future__ import annotations

def _rank0_print(msg: str) -> None:
    from lumen.config import _rank0_print as _print

    _print(msg)


def patch_varlen_attn() -> None:
    import torch
    import torch.nn.functional as F

    import aiter

    original = F.scaled_dot_product_attention

    def varlen_sdpa(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        **kwargs,
    ):
        if attn_mask is not None or dropout_p != 0.0 or not is_causal or query.dim() != 4:
            return original(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                **kwargs,
            )
        batch, heads, tokens, dim = query.shape
        kv_tokens = key.shape[2]
        q = query.permute(0, 2, 1, 3).reshape(batch * tokens, heads, dim).contiguous()
        k = key.permute(0, 2, 1, 3).reshape(batch * kv_tokens, key.shape[1], key.shape[-1]).contiguous()
        v = value.permute(0, 2, 1, 3).reshape(batch * kv_tokens, value.shape[1], value.shape[-1]).contiguous()
        cu_q = torch.arange(0, (batch + 1) * tokens, tokens, device=query.device, dtype=torch.int32)
        cu_k = torch.arange(0, (batch + 1) * kv_tokens, kv_tokens, device=query.device, dtype=torch.int32)
        softmax_scale = dim ** -0.5 if scale is None else float(scale)
        # The varlen autograd function saves softmax_lse for the backward
        # kernel and asserts that the forward asked for it.
        needs_grad = torch.is_grad_enabled() and any(
            t.requires_grad for t in (q, k, v)
        )
        result = aiter.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=tokens,
            max_seqlen_k=kv_tokens,
            # Match ATOM's prefill metadata.  This value selects the
            # skip-zero FMHA variant and is not merely a shape hint.
            min_seqlen_q=0,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=(-1, -1, 0),
            return_lse=needs_grad,
        )
        out = result[0] if isinstance(result, tuple) else result
        return out.view(batch, tokens, heads, dim).permute(0, 2, 1, 3).contiguous()

    F.scaled_dot_product_attention = varlen_sdpa
    _rank0_print("> varlen_attn: causal SDPA uses aiter.flash_attn_varlen_func")
