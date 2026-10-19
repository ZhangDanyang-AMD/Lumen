# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0
"""Sequential routed FC execution using caller-bound AITER training APIs."""

import torch
from torch.autograd.function import once_differentiable

from lumen.ops.dispatch import _probe_aiter_mxfp4_training
from lumen.ops.mxfp4_moe_execution import RoutedFCExecution
from lumen.utils.mxfp4_moe_ledger import Mxfp4StateError


class TorchRuntimeOps:
    """Production operations only; no allocation or alternate execution schedule."""

    activation_dtype = torch.bfloat16
    current_stream = staticmethod(torch.cuda.current_stream)
    is_grad_enabled = staticmethod(torch.is_grad_enabled)
    probe_api = staticmethod(_probe_aiter_mxfp4_training)


class _RoutedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, execution, replay, *parameters):
        if len(parameters) != len(execution.zero_leases) or any(
            parameter is not lease.binding.parameter
            for parameter, lease in zip(parameters, execution.zero_leases)
        ):
            raise Mxfp4StateError("MAIN_GRAD_MISMATCH")
        ctx.execution = execution
        ctx.set_materialize_grads(False)
        return execution.forward(replay)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        if dy is None:
            raise Mxfp4StateError("MISSING_ROUTED_OUTPUT_GRADIENT")
        dx, emissions = ctx.execution.backward(dy)
        return dx, None, None, *emissions


def routed_linear(x, execution: RoutedFCExecution, *, replay=False):
    """Every original Parameter is an explicit edge, including all-empty FCs."""
    return _RoutedLinear.apply(
        x,
        execution,
        replay,
        *(lease.binding.parameter for lease in execution.zero_leases),
    )
