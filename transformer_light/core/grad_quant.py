###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Gradient quantization utilities.

Provides :func:`quantize_grad_tensor` — a quant → dequant round-trip that
reduces a gradient tensor to the representable precision of a chosen
low-precision format.  Supported formats:

* ``"fp8"``   — per-tensor FP8 (E4M3 or E5M2, auto-detected from hardware)
* ``"mxfp8"`` — microscaling FP8 with per-block uint8 scales
* ``"fp4"``   — placeholder (raises ``NotImplementedError`` until kernels land)

The function is intentionally **stateless** and **pure** — it does not touch
any scaling manager or amax history.  It is used in autograd ``backward()``
methods to compress ``grad_weight`` / ``dq`` / ``dk`` / ``dv`` before they
are returned to the framework for accumulation or communication.
"""

from typing import Optional

import torch


# ── Valid grad quantization type names ────────────────────────────────────

GRAD_QUANT_TYPES = (None, "fp8", "mxfp8", "fp4")


# ── FP8 round-trip ───────────────────────────────────────────────────────

def _round_to_fp8(tensor: torch.Tensor, fp8_dtype: torch.dtype) -> torch.Tensor:
    orig_dtype = tensor.dtype
    amax = tensor.abs().amax().clamp(min=1e-12)
    fp8_max = torch.finfo(fp8_dtype).max
    scale = fp8_max / amax
    tensor_fp8 = (tensor.float() * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return tensor_fp8.to(orig_dtype) / scale


# ── MXFP8 round-trip ────────────────────────────────────────────────────

def _round_to_mxfp8(
    tensor: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    from transformer_light.ops.quantize.ops import convert_to_mxfp8, convert_from_mxfp8

    orig_dtype = tensor.dtype
    orig_shape = tensor.shape

    flat = tensor.reshape(-1, orig_shape[-1]).contiguous()
    M, N = flat.shape
    # Pad N to be divisible by block_size
    pad_n = (block_size - N % block_size) % block_size
    if pad_n > 0:
        flat = torch.nn.functional.pad(flat, (0, pad_n))

    data_bf16 = flat.to(torch.bfloat16)
    data_lp, scales = convert_to_mxfp8(data_bf16, block_size=block_size, axis=-1)
    data_hp = convert_from_mxfp8(
        data_lp, scales, output_dtype=torch.bfloat16,
        block_size=block_size, axis=-1,
    )

    if pad_n > 0:
        data_hp = data_hp[:, :N]

    return data_hp.reshape(orig_shape).to(orig_dtype)


# ── Public API ───────────────────────────────────────────────────────────

def quantize_grad_tensor(
    tensor: torch.Tensor,
    grad_quant_type: Optional[str],
    fp8_dtype: Optional[torch.dtype] = None,
    block_size: int = 32,
) -> torch.Tensor:
    """Quantize *tensor* to a low-precision format and dequantize back.

    This effectively rounds the values to the representable precision of
    the target format, reducing bit-width for gradient communication and
    accumulation.

    Args:
        tensor: The gradient tensor to quantize.
        grad_quant_type: One of ``"fp8"``, ``"mxfp8"``, ``"fp4"``, or
            ``None`` (no-op).
        fp8_dtype: Explicit FP8 dtype for the ``"fp8"`` path.  When
            ``None``, auto-detects E4M3 (FNUZ vs OCP) from the current GPU.
        block_size: Block size for ``"mxfp8"`` quantization.

    Returns:
        Tensor with the same shape and dtype, rounded to the target precision.
    """
    if grad_quant_type is None:
        return tensor

    if grad_quant_type == "fp8":
        if fp8_dtype is None:
            from transformer_light.quantize.config import _get_float8_e4m3
            fp8_dtype = _get_float8_e4m3()
        return _round_to_fp8(tensor, fp8_dtype)

    if grad_quant_type == "mxfp8":
        return _round_to_mxfp8(tensor, block_size=block_size)

    if grad_quant_type == "fp4":
        raise NotImplementedError(
            "FP4 gradient quantization is not yet implemented. "
            "Use 'fp8' or 'mxfp8' for now."
        )

    raise ValueError(
        f"Unknown grad_quant_type={grad_quant_type!r}. "
        f"Valid options: {GRAD_QUANT_TYPES}"
    )
