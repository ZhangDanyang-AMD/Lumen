###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Replace ``nn.Linear`` forward with an AITER GEMM that has a backward.

``atom_mode=True`` uses ``TunedGemm().mm()``, the same entry ATOM inference
uses. Otherwise the forward goes through ``dispatch_gemm``.
"""

from __future__ import annotations

from lumen.ops.gemm.linear_autograd import make_tuned_linear_fn
from lumen.ops.gemm.stable_tgemm_mm import stable_tgemm_mm


def _rank0_print(msg: str) -> None:
    from lumen.config import _rank0_print as _print

    _print(msg)


def _auto_tune_bf16_gemm(model) -> None:
    """Placeholder. BF16 GEMM stays on the torch.mm fallback.

    hipBLASLt and ASM JIT both SIGABRT on MI350 gfx950, so tuned GEMM
    injection is skipped and ``tuned_gemm.gemm_a16w16`` falls back to
    ``torch.mm``.
    """
    del model


def patch_linear(model, *, atom_mode: bool = False) -> None:
    import torch
    import torch.nn as nn

    if atom_mode:
        from aiter.tuned_gemm import TunedGemm

        tgemm = TunedGemm()
        linear_fn = make_tuned_linear_fn(tgemm)

        count = 0
        for _name, module in model.named_modules():
            if isinstance(module, nn.Linear):

                def _make_forward(mod):
                    def _forward(input):
                        return linear_fn.apply(input, mod.weight, mod.bias)
                    return _forward

                module.forward = _make_forward(module)
                count += 1

        if count:
            _rank0_print(f"> Replaced {count} nn.Linear forward with ATOM TunedGemm.mm()")

        # Megatron path: patch _do_gemm only if the model contains Lumen parallel linears.
        try:
            from lumen.modules.parallel_linear import (
                LumenColumnParallelLinear,
                LumenRowParallelLinear,
            )
            has_parallel = any(
                isinstance(m, (LumenColumnParallelLinear, LumenRowParallelLinear))
                for m in model.modules()
            )
        except ImportError:
            has_parallel = False

        if has_parallel:
            import lumen.modules.parallel_linear as _pl

            _orig_do_gemm = _pl._do_gemm

            def _tuned_do_gemm(
                input_, weight, bias, scaling_manager, scaling_type,
                fp8_dtype, block_size, gradient_accumulation_fusion=False,
                delay_wgrad=False, deferred_wgrad=None,
                activation_tensor_id=None, pre_quantized_input=None,
                use_gemm_bf16=False,
            ):
                if scaling_type != "none" or delay_wgrad:
                    return _orig_do_gemm(
                        input_, weight, bias, scaling_manager, scaling_type,
                        fp8_dtype, block_size, gradient_accumulation_fusion,
                        delay_wgrad, deferred_wgrad, activation_tensor_id,
                        pre_quantized_input, use_gemm_bf16,
                    )
                return stable_tgemm_mm(
                    tgemm, input_, weight, bias, input_.dtype
                )

            _pl._do_gemm = _tuned_do_gemm
            _rank0_print("> ATOM rollout: patched parallel linear GEMM to use TunedGemm.mm()")

        return

    _auto_tune_bf16_gemm(model)

    from lumen.ops.quantize.linear import dispatch_gemm

    class _GemmA16W16Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            ctx.save_for_backward(input, weight)
            ctx.has_bias = bias is not None
            output = dispatch_gemm(input, weight, scaling_type="none", bias=bias)
            return output

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

    count = 0
    for _name, module in model.named_modules():
        if isinstance(module, nn.Linear):

            def _make_aiter_forward(mod):
                def _aiter_forward(input):
                    return _GemmA16W16Fn.apply(input, mod.weight, mod.bias)

                return _aiter_forward

            module.forward = _make_aiter_forward(module)
            count += 1

    if count:
        _rank0_print(f"> Replaced {count} nn.Linear forward with AITER GEMM (ASM→HIP→Triton)")
