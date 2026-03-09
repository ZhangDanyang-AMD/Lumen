###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""
transformer_light.quantize — low-precision training lifecycle for AMD GPUs.

Supports FP8 (E4M3 / E5M2), MXFP8, and FP4 formats.

Usage::

    import transformer_light.quantize as quant
    from transformer_light.quantize import QuantConfig, QuantFormat, ScalingType

    # Configure quantization
    config = QuantConfig(format=QuantFormat.FP8_E4M3,
                         scaling=ScalingType.DELAYED)

    # Non-invasive: patch existing model, no module replacement
    quant.enable(model, config=config)

    # Or use string shorthand
    quant.enable(model, format="fp8_e4m3", scaling="delayed")

    # Check backend availability
    backend = quant.get_attention_backend()   # "aiter" or "triton"
"""

import functools
import logging
from typing import Optional

import torch
import torch.nn as nn

from transformer_light.quantize.config import (
    AmaxAlgo,
    QuantConfig,
    QuantFormat,
    ScalingType,
    get_fp8_max,
    get_fp8_max_bwd,
)
from transformer_light.quantize.scaling_manager import (
    GRAD_QUANT_TYPES,
    ScalingManager,
)
from transformer_light.ops.quantize import (
    convert_to_mxfp8,
    convert_from_mxfp8,
    quant_fp8_blockwise_impl,
    quant_fp8_tensorwise_impl,
    dequant_fp8_tensorwise_impl,
    quant_fp8_blockwise_segment_m_impl,
    QuantizedLinearFunction,
    quantized_linear,
)

# Backward-compat aliases (autograd.py and communication.py removed)
FP8ScalingManager = ScalingManager
FP8Linear = QuantizedLinearFunction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def is_aiter_available() -> bool:
    """Return True if the AITER package (CK attention backend) is importable."""
    try:
        import aiter  # noqa: F401
        return True
    except ImportError:
        return False


def get_attention_backend(prefer: str = "auto") -> str:
    """Determine which attention backend to use.

    Args:
        prefer: One of ``"auto"``, ``"aiter"``, ``"triton"``.

    Returns:
        ``"aiter"`` or ``"triton"``
    """
    if prefer == "triton":
        return "triton"

    if prefer == "aiter":
        if not is_aiter_available():
            raise RuntimeError(
                "AITER is not installed. The aiter attention backend requires "
                "'aiter' — install it or set backend to 'triton'."
            )
        return "aiter"

    # auto
    if is_aiter_available():
        logger.info("AITER detected — using aiter attention backend")
        return "aiter"
    logger.info("AITER not found — falling back to Triton attention backend")
    return "triton"


def get_quant_backend(prefer: str = "auto") -> str:
    """Determine which quantization backend to use.

    Args:
        prefer: One of ``"auto"``, ``"aiter"``, ``"triton"``.

    Returns:
        ``"aiter"`` or ``"triton"``
    """
    if prefer == "triton":
        return "triton"
    if prefer == "aiter":
        if not is_aiter_available():
            raise RuntimeError(
                "AITER is not installed. Install it or use backend='triton'."
            )
        return "aiter"
    return "aiter" if is_aiter_available() else "triton"


# ---------------------------------------------------------------------------
# Quantization enablement
# ---------------------------------------------------------------------------

def enable(
    model,
    config: Optional[QuantConfig] = None,
    *,
    format: str = "fp8_e4m3",
    scaling: str = "delayed",
    backend: str = "auto",
    recipe: Optional[str] = None,
    dp_group=None,
    **kwargs,
) -> ScalingManager:
    """Non-invasive: patch existing model's ``nn.Linear`` layers with FP8
    quantized forward/backward.

    Every ``nn.Linear`` in *model* gets a forward hook that quantizes
    input and weight, runs an FP8 GEMM, and dequantizes the output — all
    transparently.  The training loop does not need to change.

    Args:
        model: The ``nn.Module`` to patch.
        config: A :class:`QuantConfig`. If provided, ``format``/``scaling``
                kwargs are ignored.
        format: Shorthand format string (ignored when *config* is given).
        scaling: Shorthand scaling string (ignored when *config* is given).
        backend: ``"auto"``, ``"aiter"``, or ``"triton"``.
        recipe: Legacy alias — maps to *scaling*. Deprecated.
        dp_group: Data-parallel process group for ``reduce_amax``.
            Required when ``config.reduce_amax=True``.
        **kwargs: Forwarded to :class:`QuantConfig` (e.g. ``block_size``).

    Returns:
        The :class:`ScalingManager` instance attached to the model.
    """
    if config is None:
        if recipe is not None:
            scaling = recipe
        config = QuantConfig.from_str(format=format, scaling=scaling, **kwargs)

    resolved_backend = get_quant_backend(backend)
    manager = ScalingManager(config)

    if config.reduce_amax and dp_group is not None:
        manager.set_dp_group(dp_group)

    _patch_linear_layers(model, manager, resolved_backend, config)
    return manager


def _get_megatron_linear_types():
    """Return Megatron's parallel linear types if available, else empty tuple."""
    try:
        from megatron.core.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )
        return ColumnParallelLinear, RowParallelLinear
    except ImportError:
        return ()


def _patch_linear_layers(
    model: nn.Module,
    manager: ScalingManager,
    backend: str,
    config: QuantConfig,
) -> None:
    """Hook quantized dispatch into every ``nn.Linear`` layer.

    Each layer gets a unique ``tensor_id`` derived from its module path so that
    :class:`ScalingManager` tracks independent amax histories per layer (fixes
    the shared-``"weight"`` bug where all layers polluted a single deque).

    Also handles Megatron's ``ColumnParallelLinear`` and ``RowParallelLinear``
    which do not inherit ``nn.Linear`` but expose the same ``.weight`` attribute.

    Gradient quantization is handled by the :class:`ScalingManager` itself
    (configured via ``config.quantize_grad``).
    """
    fp8_dtype = config.torch_dtype or torch.float8_e4m3fn
    block_size = config.block_size
    quant_act = config.quantize_activation

    megatron_types = _get_megatron_linear_types()
    quantizable_types = (nn.Linear,) + megatron_types

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, quantizable_types):
            tensor_id = f"{name}.weight" if name else "weight"
            module._quant_manager = manager
            module._quant_backend = backend
            module._quant_enabled = True
            module._quant_tensor_id = tensor_id

            is_megatron = megatron_types and isinstance(module, megatron_types)
            _replace_forward(module, manager, backend, fp8_dtype, block_size,
                             tensor_id, quant_act, is_megatron)
            count += 1

    act_str = "weight+activation" if quant_act else "weight-only"
    grad_quant_type = config.quantize_grad
    grad_str = f"+grad({grad_quant_type})" if grad_quant_type else ""
    logger.info(
        "Quantization enabled on %d nn.Linear layers "
        "(backend=%s, format=%s, scaling=%s, amax_algo=%s, %s%s)",
        count, backend, config.format.value, config.scaling.value,
        config.amax_algo.value, act_str, grad_str,
    )


def _replace_forward(module, manager, backend, fp8_dtype, block_size,
                     tensor_id, quantize_activation, is_megatron):
    """Replace the module's forward method with an FP8-quantized version.

    Unlike ``register_forward_hook``, this prevents the original (BF16) linear
    from running at all, saving both compute and peak memory — critical for
    70B-class models under FSDP.
    """
    original_forward = module.forward

    if not is_megatron:
        def quant_forward(input_tensor, *args, **kwargs):
            return quantized_linear(
                input_tensor,
                module.weight,
                module.bias,
                scaling_manager=manager,
                backend=backend,
                fp8_dtype=fp8_dtype,
                block_size=block_size,
                tensor_id=tensor_id,
                quantize_activation=quantize_activation,
            )
    else:
        def quant_forward(input_tensor, *args, **kwargs):
            skip_bias_add = getattr(module, "skip_bias_add", False)
            bias = getattr(module, "bias", None)
            bias_for_gemm = None if skip_bias_add else bias

            result = quantized_linear(
                input_tensor,
                module.weight,
                bias_for_gemm,
                scaling_manager=manager,
                backend=backend,
                fp8_dtype=fp8_dtype,
                block_size=block_size,
                tensor_id=tensor_id,
                quantize_activation=quantize_activation,
            )

            if getattr(module, "input_is_parallel", False):
                try:
                    from megatron.core.tensor_parallel.mappings import (
                        reduce_from_tensor_model_parallel_region,
                    )
                    result = reduce_from_tensor_model_parallel_region(result)
                except ImportError:
                    pass
            elif getattr(module, "gather_output", False):
                try:
                    from megatron.core.tensor_parallel.mappings import (
                        gather_from_tensor_model_parallel_region,
                    )
                    result = gather_from_tensor_model_parallel_region(result)
                except ImportError:
                    pass

            if skip_bias_add:
                return result, bias
            return result

    module._original_forward = original_forward
    module.forward = quant_forward


def disable(model: nn.Module) -> None:
    """Remove FP8 quantized forward from all patched layers."""
    for module in model.modules():
        if hasattr(module, "_original_forward"):
            module.forward = module._original_forward
            del module._original_forward
            module._quant_enabled = False
