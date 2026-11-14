###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused linear, QKV, and gate/up GEMMs.

Each forward concatenates the weights, runs :func:`stable_tgemm_mm`, and the
backward is a dense matmul on the saved input. The factories close over one
``TunedGemm`` instance for the life of the patched model.
"""

from __future__ import annotations

from lumen.ops.gemm.stable_tgemm_mm import stable_tgemm_mm


def make_tuned_linear_fn(tgemm):
    import torch

    class TunedLinearFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            ctx.save_for_backward(input, weight)
            ctx.has_bias = bias is not None
            return stable_tgemm_mm(tgemm, input, weight, bias, input.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            input, weight = ctx.saved_tensors
            grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
            input_flat = input.reshape(-1, input.shape[-1])
            grad_input = grad_flat @ weight
            grad_input = grad_input.reshape(input.shape)
            grad_weight = grad_flat.t() @ input_flat
            grad_bias = (grad_flat.sum(0) if ctx.has_bias else None)
            return grad_input, grad_weight, grad_bias

    return TunedLinearFn


def make_fused_qkv_fn(tgemm):
    import torch

    class FusedQKVFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, wq, wk, wv, bq, bk, bv):
            weight = torch.cat((wq, wk, wv), dim=0)
            if bq is None and bk is None and bv is None:
                bias = None
            else:
                parts = [
                    part_bias if part_bias is not None else weight.new_zeros(part.shape[0])
                    for part, part_bias in ((wq, bq), (wk, bk), (wv, bv))
                ]
                bias = torch.cat(parts, dim=0)
            ctx.save_for_backward(x, wq, wk, wv)
            ctx.bias_flags = (bq is not None, bk is not None, bv is not None)
            return stable_tgemm_mm(tgemm, x, weight, bias, x.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            x, wq, wk, wv = ctx.saved_tensors
            weight = torch.cat((wq, wk, wv), dim=0)
            grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
            input_flat = x.reshape(-1, x.shape[-1])
            grad_input = (grad_flat @ weight).reshape(x.shape)
            grad_weight = grad_flat.t() @ input_flat
            sizes = (wq.shape[0], wk.shape[0], wv.shape[0])
            gq, gk, gv = grad_weight.split(sizes, dim=0)
            if any(ctx.bias_flags):
                gbq, gbk, gbv = grad_flat.sum(0).split(sizes, dim=0)
                gbq = gbq if ctx.bias_flags[0] else None
                gbk = gbk if ctx.bias_flags[1] else None
                gbv = gbv if ctx.bias_flags[2] else None
            else:
                gbq = gbk = gbv = None
            return grad_input, gq, gk, gv, gbq, gbk, gbv

    return FusedQKVFn


def make_fused_gate_up_fn(tgemm):
    import torch

    class FusedGateUpFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w_gate, w_up, b_gate, b_up):
            weight = torch.cat((w_gate, w_up), dim=0)
            if b_gate is None and b_up is None:
                bias = None
            else:
                parts = [
                    part_bias if part_bias is not None else weight.new_zeros(part.shape[0])
                    for part, part_bias in ((w_gate, b_gate), (w_up, b_up))
                ]
                bias = torch.cat(parts, dim=0)
            ctx.save_for_backward(x, w_gate, w_up)
            ctx.bias_flags = (b_gate is not None, b_up is not None)
            return stable_tgemm_mm(tgemm, x, weight, bias, x.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            x, w_gate, w_up = ctx.saved_tensors
            weight = torch.cat((w_gate, w_up), dim=0)
            grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
            input_flat = x.reshape(-1, x.shape[-1])
            grad_input = (grad_flat @ weight).reshape(x.shape)
            grad_weight = grad_flat.t() @ input_flat
            sizes = (w_gate.shape[0], w_up.shape[0])
            g_gate, g_up = grad_weight.split(sizes, dim=0)
            if any(ctx.bias_flags):
                gb_gate, gb_up = grad_flat.sum(0).split(sizes, dim=0)
                gb_gate = gb_gate if ctx.bias_flags[0] else None
                gb_up = gb_up if ctx.bias_flags[1] else None
            else:
                gb_gate = gb_up = None
            return grad_input, g_gate, g_up, gb_gate, gb_up

    return FusedGateUpFn
