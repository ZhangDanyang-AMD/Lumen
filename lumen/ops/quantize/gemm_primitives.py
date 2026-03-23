###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Standalone dgrad / wgrad primitives for BF16 GEMM.

Extracted from ``QuantizedLinearFunction.backward`` (``scaling_type='none'``
branch) to allow reuse in fused comm-GEMM overlap autograd functions.

These are additive — ``QuantizedLinearFunction`` is not modified.
"""

from typing import Callable

import torch

from lumen.ops.quantize.linear import dispatch_gemm


def compute_dgrad_bf16(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Compute input gradient: dgrad = grad_output @ weight.

    Args:
        grad_output: ``[*, N]`` where N = out_features.
        weight: ``[N, K]`` where K = in_features.

    Returns:
        Tensor with shape ``[*, K]``.
    """
    return dispatch_gemm(
        grad_output,
        weight.t().contiguous(),
        None,
        None,
        "none",
    )


def compute_wgrad_bf16(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    """Compute weight gradient: wgrad = grad_output^T @ input.

    Args:
        grad_output: ``[M, N]``.
        input_tensor: ``[M, K]``.

    Returns:
        Tensor with shape ``[N, K]``.
    """
    grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
    input_flat = input_tensor.reshape(-1, input_tensor.shape[-1]).contiguous()
    return dispatch_gemm(
        grad_flat.t().contiguous(),
        input_flat.t().contiguous(),
        None,
        None,
        "none",
    )


def make_wgrad_closure_bf16(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    weight_ref: torch.nn.Parameter,
    gradient_accumulation_fusion: bool,
) -> Callable[[], None]:
    """Build a deferred wgrad closure compatible with ``_DeferredWgrad``.

    The closure computes wgrad and accumulates into ``weight_ref.main_grad``
    (if *gradient_accumulation_fusion*) or ``weight_ref.grad``.
    """
    grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
    input_flat = input_tensor.reshape(-1, input_tensor.shape[-1]).contiguous()

    def _wgrad_fn():
        gw = dispatch_gemm(
            grad_flat.t().contiguous(),
            input_flat.t().contiguous(),
            None,
            None,
            "none",
        )
        if gradient_accumulation_fusion and hasattr(weight_ref, "main_grad"):
            weight_ref.main_grad.add_(gw)
        elif weight_ref.grad is not None:
            weight_ref.grad.add_(gw)
        else:
            weight_ref.grad = gw

    return _wgrad_fn
