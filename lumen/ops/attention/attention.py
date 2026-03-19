###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch

from lumen.core.grad_quant import quantize_grad_tensor


def _is_aiter_available() -> bool:
    try:
        import aiter  # noqa: F401

        return True
    except ImportError:
        return False


if _is_aiter_available():
    from aiter.ops.mha import flash_attn_func

from lumen.kernels.attention.attention_impl import FIXED_BLOCK_M
from lumen.kernels.attention.attention_impl import attention_backward as triton_fp8_backward
from lumen.kernels.attention.attention_impl import attention_forward as triton_fp8_forward
from lumen.kernels.attention.attention_impl import attention_mxfp8_backward as triton_mxfp8_backward
from lumen.kernels.attention.attention_impl import attention_mxfp8_forward as triton_mxfp8_forward
from lumen.ops.attention.attention_with_cp_p2p import attention_cp_p2p

__all__ = ["attention", "attention_fp8_quant"]

_FP8_BACKENDS = ("aiter_triton_fp8", "aiter_csrc_fp8", "aiter_asm_fp8")

_QUANT_TYPE_ALIAS = {
    "fp8_blockwise": "blockwise",
}


# ---------------------------------------------------------------------------
# torch.compile support
# ---------------------------------------------------------------------------
def _mark_allow_in_graph(*classes):
    try:
        from torch._dynamo import allow_in_graph

        for cls in classes:
            allow_in_graph(cls)
    except Exception:
        pass


def _extract_triton_lse(lse_raw, seqlen_q):
    """Extract contiguous LSE from the Triton kernel's interleaved buffer.

    The kernel allocates (B, H, SQ*2) and writes each block's LSE followed
    by BLOCK_M slots reserved for backward delta:
        [LSE_blk0(BM) | delta_blk0(BM) | LSE_blk1(BM) | delta_blk1(BM) | ...]

    Requires seqlen_q to be a multiple of FIXED_BLOCK_M (kernel constraint).
    """
    B, H, total = lse_raw.shape
    num_blocks = (seqlen_q + FIXED_BLOCK_M - 1) // FIXED_BLOCK_M
    reshaped = lse_raw.reshape(B, H, num_blocks, 2 * FIXED_BLOCK_M)
    lse_clean = reshaped[:, :, :, :FIXED_BLOCK_M].contiguous().reshape(B, H, num_blocks * FIXED_BLOCK_M)
    return lse_clean[:, :, :seqlen_q]


# ---------------------------------------------------------------------------
# Autograd Functions
# ---------------------------------------------------------------------------


class AttentionTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        return_lse,
        return_softmax,
        is_grad_enabled,
        use_fp8,
        grad_quant_type=None,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        (output, softmax_lse, exp_scores, q, k, v, q_scale, k_scale, v_scale, p_scale, rng_state) = triton_fp8_forward(
            q,
            k,
            v,
            use_fp8,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            bias,
            alibi_slopes,
            return_softmax,
            force_triton=True,
        )

        if is_grad:
            ctx.save_for_backward(
                q,
                k,
                v,
                output,
                softmax_lse,
                alibi_slopes,
                bias,
                q_scale,
                k_scale,
                v_scale,
                rng_state,
            )
            ctx.sm_scale = softmax_scale
            ctx.p_scale = p_scale
            ctx.causal = causal
            ctx.use_fp8 = use_fp8
            ctx.dropout_p = dropout_p
            ctx.window_size = window_size
            ctx.cu_seqlens_q = 0
            ctx.cu_seqlens_k = 0
            ctx.max_seqlens_q = q.shape[1]
            ctx.max_seqlens_k = k.shape[1]
            ctx.grad_quant_type = grad_quant_type

        result = [output]
        if return_lse:
            result.append(_extract_triton_lse(softmax_lse, q.shape[1]))
        if return_softmax:
            result.append(exp_scores)
        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        (q, k, v, o, softmax_lse, alibi_slopes, bias, q_scale, k_scale, v_scale, rng_state) = ctx.saved_tensors

        dq, dk, dv = triton_fp8_backward(
            do,
            q,
            k,
            v,
            o,
            q_scale,
            k_scale,
            v_scale,
            ctx.p_scale,
            softmax_lse,
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_k,
            ctx.max_seqlens_q,
            ctx.max_seqlens_k,
            ctx.sm_scale,
            ctx.causal,
            alibi_slopes,
            ctx.use_fp8,
            rng_state=rng_state,
            dropout_p=ctx.dropout_p,
            bias=bias,
            window_size=ctx.window_size,
            force_triton=True,
        )
        gqt = ctx.grad_quant_type
        dq = quantize_grad_tensor(dq, gqt)
        dk = quantize_grad_tensor(dk, gqt)
        dv = quantize_grad_tensor(dv, gqt)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None


class AttentionTritonMXFP8Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        return_lse,
        return_softmax,
        is_grad_enabled,
        use_mxfp8,
        block_m_fwd=64,
        block_n_fwd=64,
        block_m_dq_bwd=64,
        block_n_dq_bwd=64,
        block_m_dkv_bwd=64,
        block_n_dkv_bwd=64,
        quant_block_size=32,
        grad_quant_type=None,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        (output, softmax_lse, exp_scores, q, k, v, q_scale, k_scale, v_scale, p_scale, rng_state) = (
            triton_mxfp8_forward(
                q,
                k,
                v,
                use_mxfp8,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                bias,
                alibi_slopes,
                return_softmax,
                block_m_fwd,
                block_n_fwd,
                quant_block_size,
            )
        )

        if is_grad:
            ctx.save_for_backward(
                q,
                k,
                v,
                output,
                softmax_lse,
                alibi_slopes,
                bias,
                q_scale,
                k_scale,
                v_scale,
                rng_state,
            )
            ctx.use_mxfp8 = use_mxfp8
            ctx.p_scale = p_scale
            ctx.sm_scale = softmax_scale
            ctx.causal = causal
            ctx.dropout_p = dropout_p
            ctx.window_size = window_size
            ctx.layout = "bshd"
            ctx.block_m_dq_bwd = block_m_dq_bwd
            ctx.block_n_dq_bwd = block_n_dq_bwd
            ctx.block_m_dkv_bwd = block_m_dkv_bwd
            ctx.block_n_dkv_bwd = block_n_dkv_bwd
            ctx.quant_block_size = quant_block_size
            ctx.cu_seqlens_q = torch.tensor(0, device="cuda")
            ctx.cu_seqlens_k = torch.tensor(0, device="cuda")
            ctx.max_seqlens_q = q.shape[1]
            ctx.max_seqlens_k = k.shape[1]
            ctx.grad_quant_type = grad_quant_type

        result = [output]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(exp_scores)
        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        (q, k, v, o, softmax_lse, alibi_slopes, bias, q_scale, k_scale, v_scale, rng_state) = ctx.saved_tensors

        dq, dk, dv = triton_mxfp8_backward(
            do,
            q,
            k,
            v,
            o,
            softmax_lse,
            q_scale,
            k_scale,
            v_scale,
            ctx.sm_scale,
            ctx.p_scale,
            alibi_slopes,
            ctx.causal,
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_k,
            ctx.max_seqlens_q,
            ctx.max_seqlens_k,
            ctx.use_mxfp8,
            ctx.block_m_dq_bwd,
            ctx.block_n_dq_bwd,
            ctx.block_m_dkv_bwd,
            ctx.block_n_dkv_bwd,
            ctx.quant_block_size,
            rng_state=rng_state,
            dropout_p=ctx.dropout_p,
            bias=bias,
            window_size=ctx.window_size,
        )
        gqt = ctx.grad_quant_type
        dq = quantize_grad_tensor(dq, gqt)
        dk = quantize_grad_tensor(dk, gqt)
        dv = quantize_grad_tensor(dv, gqt)
        return (
            dq,
            dk,
            dv,
        ) + (None,) * 18


class AttentionTritonBlockwise2DFunction(torch.autograd.Function):
    """Autograd function for blockwise2d FP8 attention.

    Uses 2D block quantization to cache Q/K/V with scale shape
    ``[B, H, S//bm, D//bn]`` via a :class:`Blockwise2DScaleManager`.
    The actual kernel still uses 1D block scales (``quantize_block_fp8``
    for Q/K, per-tensor for V) since the underlying Triton kernel expects
    that layout.  The 2D cache enables backward to fully reuse forward's
    quantized tensors and to persist the dO scale across iterations.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        return_lse,
        return_softmax,
        is_grad_enabled,
        grad_quant_type=None,
        scale_manager=None,
    ):
        from lumen.quantize.scaling_manager import Blockwise2DScaleManager, ScalingManager

        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        fp8_dt = None
        try:
            from lumen.core.float8 import float8_e4m3

            fp8_dt = float8_e4m3
        except ImportError:
            from lumen.quantize.config import _get_float8_e4m3

            fp8_dt = _get_float8_e4m3()

        if scale_manager is not None and isinstance(scale_manager, Blockwise2DScaleManager):
            scale_manager.quantize_and_cache(q, k, v, fp8_dt)

        q_fp8, q_scale_1d = ScalingManager.quantize_block_fp8(q, FIXED_BLOCK_M, fp8_dt)
        k_fp8, k_scale_1d = ScalingManager.quantize_block_fp8(k, FIXED_BLOCK_M, fp8_dt)
        v_fp8, v_scale_1d, p_scale = ScalingManager.quantize_v_fp8(v, fp8_dt)

        (output, softmax_lse, exp_scores, q_out, k_out, _, _, _, _, _, rng_state) = triton_fp8_forward(
            q_fp8,
            k_fp8,
            v_fp8,
            True,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            bias,
            alibi_slopes,
            return_softmax,
            force_triton=True,
        )

        if is_grad:
            ctx.save_for_backward(
                q_out,
                k_out,
                v_fp8,
                output,
                softmax_lse,
                alibi_slopes,
                bias,
                q_scale_1d,
                k_scale_1d,
                v_scale_1d,
                rng_state,
            )
            ctx.sm_scale = softmax_scale
            ctx.p_scale = p_scale
            ctx.causal = causal
            ctx.use_fp8 = True
            ctx.dropout_p = dropout_p
            ctx.window_size = window_size
            ctx.cu_seqlens_q = 0
            ctx.cu_seqlens_k = 0
            ctx.max_seqlens_q = q.shape[1]
            ctx.max_seqlens_k = k.shape[1]
            ctx.grad_quant_type = grad_quant_type
            ctx.scale_manager = scale_manager

        result = [output]
        if return_lse:
            result.append(_extract_triton_lse(softmax_lse, q.shape[1]))
        if return_softmax:
            result.append(exp_scores)
        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        (q, k, v, o, softmax_lse, alibi_slopes, bias, q_scale, k_scale, v_scale, rng_state) = ctx.saved_tensors

        dq, dk, dv = triton_fp8_backward(
            do,
            q,
            k,
            v,
            o,
            q_scale,
            k_scale,
            v_scale,
            ctx.p_scale,
            softmax_lse,
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_k,
            ctx.max_seqlens_q,
            ctx.max_seqlens_k,
            ctx.sm_scale,
            ctx.causal,
            alibi_slopes,
            ctx.use_fp8,
            rng_state=rng_state,
            dropout_p=ctx.dropout_p,
            bias=bias,
            window_size=ctx.window_size,
            force_triton=True,
        )

        scale_mgr = ctx.scale_manager
        if scale_mgr is not None:
            from lumen.quantize.scaling_manager import Blockwise2DScaleManager

            if isinstance(scale_mgr, Blockwise2DScaleManager):
                do_abs_max = do.abs().amax()
                fp8_max = torch.finfo(q.dtype).max if q.is_floating_point() else 448.0
                do_scale = fp8_max / torch.where(
                    do_abs_max == 0,
                    torch.tensor(fp8_max, device=do.device),
                    do_abs_max,
                )
                scale_mgr.cache_do_scale(do_scale)

        gqt = ctx.grad_quant_type
        dq = quantize_grad_tensor(dq, gqt)
        dk = quantize_grad_tensor(dk, gqt)
        dv = quantize_grad_tensor(dv, gqt)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None


_mark_allow_in_graph(AttentionTritonFunction, AttentionTritonMXFP8Function, AttentionTritonBlockwise2DFunction)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def attention(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    bias=None,
    alibi_slopes=None,
    deterministic=True,
    return_lse=False,
    return_attn_probs=False,
    backend_type: str = "auto",
    cp_param_bundle=None,
    grad_quant_type: Optional[str] = None,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    # Resolve "auto": prefer CK csrc when available, fall back to Triton.
    if backend_type == "auto":
        backend_type = "aiter_csrc" if _is_aiter_available() else "aiter_triton"

    # Context-parallelism path
    if cp_param_bundle is not None:
        assert "cp_group" in cp_param_bundle
        assert "cp_comm_type" in cp_param_bundle
        from lumen.ops.attention.attention_with_cp_a2a import (
            AttentionAiterFunctionCPA2A,
            AttentionTritonFunctionCPA2A,
        )
        from lumen.ops.sdma import is_sdma_available

        cp_group = cp_param_bundle["cp_group"]
        cp_comm_type = cp_param_bundle["cp_comm_type"]
        cp_use_sdma = cp_param_bundle.get("use_sdma", False) and is_sdma_available()
        if backend_type == "aiter_csrc" and cp_comm_type == "a2a":
            return AttentionAiterFunctionCPA2A.apply(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                bias,
                alibi_slopes,
                deterministic,
                return_lse,
                return_attn_probs,
                torch.is_grad_enabled(),
                cp_group,
                cp_use_sdma,
            )
        elif backend_type == "aiter_triton" and cp_comm_type == "a2a":
            return AttentionTritonFunctionCPA2A.apply(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                bias,
                alibi_slopes,
                return_lse,
                return_attn_probs,
                torch.is_grad_enabled(),
                False,
                cp_group,
                grad_quant_type,
                cp_use_sdma,
            )
        elif cp_comm_type == "p2p":
            cp_size = cp_group.size()
            cp_rank = cp_group.rank()

            def _make_attn_fn():
                _dropout_p = dropout_p
                _window_size = window_size
                _bias = bias
                _alibi_slopes = alibi_slopes
                _backend_type = backend_type
                _grad_quant_type = grad_quant_type
                _deterministic = deterministic

                def attn_fn(q_chunk, k_chunk, v_chunk, causal, softmax_scale):
                    if _backend_type == "aiter_csrc" and _is_aiter_available():
                        out, lse = flash_attn_func(
                            q_chunk,
                            k_chunk,
                            v_chunk,
                            dropout_p=_dropout_p,
                            softmax_scale=softmax_scale,
                            causal=causal,
                            window_size=_window_size,
                            bias=_bias,
                            alibi_slopes=_alibi_slopes,
                            deterministic=_deterministic,
                            return_lse=True,
                            return_attn_probs=False,
                        )
                        return out, lse
                    else:
                        result = AttentionTritonFunction.apply(
                            q_chunk,
                            k_chunk,
                            v_chunk,
                            _dropout_p,
                            softmax_scale,
                            causal,
                            _window_size,
                            _bias,
                            _alibi_slopes,
                            True,
                            False,
                            torch.is_grad_enabled(),
                            False,
                            _grad_quant_type,
                        )
                        return result[0], result[1]

                return attn_fn

            attn_fn = _make_attn_fn()
            out = attention_cp_p2p(
                q,
                k,
                v,
                cp_group,
                cp_size,
                cp_rank,
                attn_fn,
                causal,
                softmax_scale,
            )
            if return_lse:
                raise NotImplementedError("return_lse not supported for cp_comm_type='p2p' yet")
            return out
        else:
            raise NotImplementedError(f"not supported backend_type {backend_type} cp_comm_type {cp_comm_type} yet")

    # Single-GPU path: prefer CK csrc, fall back to Triton.
    if backend_type == "aiter_csrc":
        if not _is_aiter_available():
            raise RuntimeError(
                "AITER is not installed. The aiter_csrc attention backend requires "
                "'aiter' — install it or use backend_type='aiter_triton'."
            )
        _return_softmax = return_attn_probs and dropout_p > 0
        _needs_grad = torch.is_grad_enabled() and any(t.requires_grad for t in [q, k, v])
        _internal_return_lse = return_lse or _needs_grad
        try:
            result = flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                bias=bias,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_lse=_internal_return_lse,
                return_attn_probs=_return_softmax,
            )
            if _internal_return_lse and not return_lse and isinstance(result, tuple):
                if len(result) == 2:
                    result = result[0]
                elif len(result) == 3:
                    # (out, lse, S_dmask) -> (out, S_dmask)
                    result = (result[0], result[2])
            return result
        except RuntimeError as _aiter_err:
            import warnings

            warnings.warn(
                f"AITER flash-attention kernel unavailable for this configuration "
                f"(q={tuple(q.shape)}, k={tuple(k.shape)}, causal={causal}): "
                f"{_aiter_err}. Falling back to aiter_triton backend.",
                RuntimeWarning,
                stacklevel=2,
            )
            return AttentionTritonFunction.apply(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                bias,
                alibi_slopes,
                return_lse,
                return_attn_probs,
                torch.is_grad_enabled(),
                False,
                grad_quant_type,
            )
    elif backend_type == "aiter_triton":
        return AttentionTritonFunction.apply(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            bias,
            alibi_slopes,
            return_lse,
            return_attn_probs,
            torch.is_grad_enabled(),
            False,
            grad_quant_type,
        )
    else:
        raise NotImplementedError(f"backend_type {backend_type} not supported")


def attention_fp8_quant(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    backend_type: str = "aiter_triton",
    cp_param_bundle=None,
    quant_type: str = "blockwise",
    block_m_fwd: int = 64,
    block_n_fwd: int = 64,
    block_m_dq_bwd: int = 64,
    block_n_dq_bwd: int = 64,
    block_m_dkv_bwd: int = 64,
    block_n_dkv_bwd: int = 64,
    quant_block_size: int = 32,
    grad_quant_type: Optional[str] = None,
    fp8_mha: bool = False,
    scale_manager=None,
):
    assert backend_type in (
        "aiter_triton",
        "aiter_triton_fp8",
        "aiter_csrc_fp8",
        "aiter_asm_fp8",
    ), (
        f"attention_fp8_quant only supports aiter_triton/aiter_triton_fp8/"
        f"aiter_csrc_fp8/aiter_asm_fp8 backends, got {backend_type}"
    )

    # Normalise legacy quant_type names
    quant_type = _QUANT_TYPE_ALIAS.get(quant_type, quant_type)

    # "none" means no FP8 quantisation — delegate to bf16 attention
    if quant_type == "none":
        _effective_backend = "aiter_csrc" if _is_aiter_available() else "aiter_triton"
        return attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            bias=bias,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_lse=return_lse,
            return_attn_probs=return_attn_probs,
            backend_type=_effective_backend,
            cp_param_bundle=cp_param_bundle,
            grad_quant_type=grad_quant_type,
        )

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    # Context-parallelism path
    if cp_param_bundle is not None:
        assert "cp_group" in cp_param_bundle
        assert "cp_comm_type" in cp_param_bundle
        from lumen.ops.attention.attention_with_cp_a2a import (
            AttentionTritonFunctionCPA2A,
            AttentionTritonMXFP8FunctionCPA2A,
        )
        from lumen.ops.sdma import is_sdma_available

        cp_group = cp_param_bundle["cp_group"]
        cp_comm_type = cp_param_bundle["cp_comm_type"]
        cp_use_sdma = cp_param_bundle.get("use_sdma", False) and is_sdma_available()
        if cp_comm_type != "a2a":
            raise NotImplementedError(f"not supported backend_type {backend_type} cp_comm_type {cp_comm_type} yet")
        if quant_type == "mxfp8":
            return AttentionTritonMXFP8FunctionCPA2A.apply(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                bias,
                alibi_slopes,
                return_lse,
                return_attn_probs,
                torch.is_grad_enabled(),
                True,
                cp_group,
                block_m_fwd,
                block_n_fwd,
                block_m_dq_bwd,
                block_n_dq_bwd,
                block_m_dkv_bwd,
                block_n_dkv_bwd,
                quant_block_size,
                grad_quant_type,
                cp_use_sdma,
            )
        elif quant_type in ("blockwise", "dynamic", "delayed", "per_token"):
            return AttentionTritonFunctionCPA2A.apply(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                bias,
                alibi_slopes,
                return_lse,
                return_attn_probs,
                torch.is_grad_enabled(),
                True,
                cp_group,
                grad_quant_type,
                cp_use_sdma,
            )
        else:
            raise NotImplementedError(f"not supported quant_type {quant_type}")

    # Single-GPU path
    if quant_type == "mxfp8":
        return AttentionTritonMXFP8Function.apply(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            bias,
            alibi_slopes,
            return_lse,
            return_attn_probs,
            torch.is_grad_enabled(),
            True,
            block_m_fwd,
            block_n_fwd,
            block_m_dq_bwd,
            block_n_dq_bwd,
            block_m_dkv_bwd,
            block_n_dkv_bwd,
            quant_block_size,
            grad_quant_type,
        )
    elif quant_type == "blockwise2d":
        return AttentionTritonBlockwise2DFunction.apply(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            bias,
            alibi_slopes,
            return_lse,
            return_attn_probs,
            torch.is_grad_enabled(),
            grad_quant_type,
            scale_manager,
        )
    elif quant_type in ("blockwise", "dynamic", "delayed", "per_token"):
        return AttentionTritonFunction.apply(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            bias,
            alibi_slopes,
            return_lse,
            return_attn_probs,
            torch.is_grad_enabled(),
            True,
            grad_quant_type,
        )
    else:
        raise NotImplementedError(f"not supported quant_type {quant_type} backend_type {backend_type} yet")
