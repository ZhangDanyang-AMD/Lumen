###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Quantization ops — functional API and autograd-aware quantized linear."""

from lumen.ops.quantize.ops import (
    convert_from_mxfp8,
    convert_to_mxfp8,
    dequant_fp8_tensorwise_impl,
    is_cdna4,
    quant_fp8_blockwise_impl,
    quant_fp8_blockwise_segment_m_impl,
    quant_fp8_tensorwise_impl,
)
from lumen.ops.quantize.linear import (
    QuantizedLinearFunction,
    quantized_linear,
)

__all__ = [
    # Pure quant/dequant ops
    "convert_from_mxfp8",
    "convert_to_mxfp8",
    "dequant_fp8_tensorwise_impl",
    "is_cdna4",
    "quant_fp8_blockwise_impl",
    "quant_fp8_blockwise_segment_m_impl",
    "quant_fp8_tensorwise_impl",
    # Quantized linear (autograd)
    "QuantizedLinearFunction",
    "quantized_linear",
]
