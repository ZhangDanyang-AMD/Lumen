###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

import torch
from aiter.ops.quant import per_token_quant_hip
from aiter.ops.gradlib import hipb_mm


class QuantLinear(torch.autograd.Function):
    """Quantized linear: quantize -> GEMM -> dequant, with autograd support."""

    @staticmethod
    def forward(ctx, input, weight, scaling_manager):
        input_fp8, input_scale = per_token_quant_hip(
            input, quant_dtype=torch.float8_e4m3fn
        )
        weight_fp8, weight_scale = scaling_manager.quantize(
            "weight", weight
        )

        output = hipb_mm(
            input_fp8, weight_fp8,
            scaleA=input_scale, scaleB=weight_scale
        )

        ctx.save_for_backward(input_fp8, weight_fp8, input_scale, weight_scale)
        ctx.scaling_manager = scaling_manager
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_fp8, weight_fp8, input_scale, weight_scale = ctx.saved_tensors

        grad_fp8, grad_scale = per_token_quant_hip(
            grad_output, quant_dtype=torch.float8_e4m3fn
        )

        # dL/dX = dL/dY @ W^T
        grad_input = hipb_mm(
            grad_fp8, weight_fp8.t(),
            scaleA=grad_scale, scaleB=weight_scale
        )

        # dL/dW = dL/dY^T @ X
        grad_weight = hipb_mm(
            grad_fp8.reshape(-1, grad_output.shape[-1]).t(),
            input_fp8,
            scaleA=grad_scale, scaleB=input_scale
        )

        return grad_input, grad_weight, None


# Backward-compat alias
FP8Linear = QuantLinear
