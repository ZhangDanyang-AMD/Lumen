###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""FP8 parameter storage and all-gather hooks.

Quantizes model parameters to FP8 for storage and communication,
reducing memory footprint and all-gather bandwidth by ~2x.
Parameters are dequantized to BF16 after all-gather for computation.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def quantize_param_to_fp8(
    param: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple:
    """Quantize a parameter tensor to FP8 with per-tensor scale.

    Returns:
        Tuple of (fp8_tensor, scale) where scale is a scalar float32.
    """
    amax = param.abs().amax().clamp(min=1e-12)
    if fp8_dtype == torch.float8_e4m3fn:
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
    elif fp8_dtype == torch.float8_e4m3fnuz:
        fp8_max = torch.finfo(torch.float8_e4m3fnuz).max
    else:
        fp8_max = 448.0
    scale = fp8_max / amax
    fp8_param = (param.float() * scale).to(fp8_dtype)
    return fp8_param, scale


def dequantize_param_from_fp8(
    fp8_param: torch.Tensor,
    scale: torch.Tensor,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an FP8 parameter back to target dtype."""
    return (fp8_param.to(torch.float32) / scale).to(target_dtype)


class FP8ParamManager:
    """Manages FP8 parameter quantization and all-gather hooks.

    When enabled, linear layer weights are stored in FP8 and
    dequantized on-the-fly before computation. All-gather operations
    communicate FP8 data (half the bandwidth of BF16).

    Args:
        fp8_dtype: Target FP8 dtype for parameters.
    """

    def __init__(self, fp8_dtype: torch.dtype = torch.float8_e4m3fn):
        self.fp8_dtype = fp8_dtype
        self._param_scales: dict = {}
        self._original_dtypes: dict = {}
        self._hooks: list = []
        self._wrapped_modules: list = []

    def quantize_params(self, model: nn.Module) -> int:
        """Quantize all linear weights in the model to FP8.

        Returns:
            Number of parameters quantized.
        """
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or hasattr(module, "weight"):
                weight = getattr(module, "weight", None)
                if weight is None or not isinstance(weight, nn.Parameter):
                    continue
                if weight.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2):
                    continue

                self._original_dtypes[name] = weight.dtype
                fp8_weight, scale = quantize_param_to_fp8(weight.data, self.fp8_dtype)
                self._param_scales[name] = scale

                weight.data = fp8_weight.to(weight.device)
                weight._fp8_scale = scale
                weight._fp8_dtype = self.fp8_dtype
                weight._original_dtype = self._original_dtypes[name]
                count += 1

        logger.info("Quantized %d parameters to FP8 (%s)", count, self.fp8_dtype)
        return count

    def register_dequant_hooks(self, model: nn.Module) -> int:
        """Register forward pre-hooks to dequantize FP8 params before compute.

        Returns:
            Number of hooks registered.
        """
        count = 0
        for name, module in model.named_modules():
            if name in self._param_scales:
                hook = module.register_forward_pre_hook(self._make_dequant_hook(name))
                self._hooks.append(hook)
                self._wrap_forward_to_use_dequant(module)
                self._wrapped_modules.append(module)
                count += 1
        return count

    def _make_dequant_hook(self, param_name: str):
        original_dtype = self._original_dtypes[param_name]

        def hook(module, inputs):
            weight = module.weight
            if hasattr(weight, "_fp8_scale"):
                fp8_data = weight.data
                dequant = dequantize_param_from_fp8(fp8_data, weight._fp8_scale, original_dtype)
                module._dequantized_weight = dequant

        return hook

    def _wrap_forward_to_use_dequant(self, module: nn.Module) -> None:
        """Replace forward to use _dequantized_weight when set by pre-hook."""
        if hasattr(module, "_fp8_original_forward"):
            return
        original_forward = module.forward

        def fp8_aware_forward(*args, **kwargs):
            if hasattr(module, "_dequantized_weight"):
                weight = module._dequantized_weight
            else:
                weight = module.weight
            return torch.nn.functional.linear(args[0], weight, module.bias)

        module._fp8_original_forward = original_forward
        module.forward = fp8_aware_forward

    def remove_hooks(self):
        """Remove all registered hooks and restore original forwards."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        for module in self._wrapped_modules:
            if hasattr(module, "_fp8_original_forward"):
                module.forward = module._fp8_original_forward
                del module._fp8_original_forward
        self._wrapped_modules.clear()

    def memory_savings_bytes(self, model: nn.Module) -> int:
        """Estimate memory savings from FP8 params (bytes)."""
        saved = 0
        for name in self._param_scales:
            parts = name.split(".")
            mod = model
            for p in parts:
                mod = getattr(mod, p, None)
                if mod is None:
                    break
            if mod is not None and hasattr(mod, "weight"):
                numel = mod.weight.numel()
                saved += numel  # saving 1 byte per element (2 bytes BF16 -> 1 byte FP8)
        return saved
