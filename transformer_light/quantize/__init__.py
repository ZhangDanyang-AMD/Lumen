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

from transformer_light.quantize.config import (
    QuantConfig,
    QuantFormat,
    ScalingType,
)
from transformer_light.quantize.scaling_manager import ScalingManager
from transformer_light.quantize.autograd import QuantLinear
from transformer_light.quantize.communication import QuantAllGather
from transformer_light.quantize.ops import (
    convert_to_mxfp8,
    convert_from_mxfp8,
    quant_fp8_blockwise_impl,
    quant_fp8_tensorwise_impl,
    dequant_fp8_tensorwise_impl,
    quant_fp8_blockwise_segment_m_impl,
)

# Backward-compat aliases
FP8ScalingManager = ScalingManager
FP8Linear = QuantLinear
FP8AllGather = QuantAllGather

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


# ---------------------------------------------------------------------------
# Quantization enablement
# ---------------------------------------------------------------------------

def enable(model, config=None, *, format="fp8_e4m3", scaling="delayed",
           recipe=None, **kwargs):
    """Non-invasive: patch existing model, no module replacement.

    Args:
        model: The ``nn.Module`` to patch.
        config: A :class:`QuantConfig`. If provided, ``format``/``scaling``
                kwargs are ignored.
        format: Shorthand format string (ignored when *config* is given).
        scaling: Shorthand scaling string (ignored when *config* is given).
        recipe: Legacy alias — maps to *scaling*. Deprecated.
        **kwargs: Forwarded to :class:`QuantConfig` (e.g. ``block_size``).

    Returns:
        The :class:`ScalingManager` instance attached to the model.
    """
    if config is None:
        if recipe is not None:
            scaling = recipe
        config = QuantConfig.from_str(format=format, scaling=scaling, **kwargs)

    manager = ScalingManager(config)
    _patch_linear_layers(model, manager)
    return manager


def _patch_linear_layers(model, manager):
    """Hook quantized dispatch into existing nn.Linear layers."""
    import torch.nn as nn

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._quant_manager = manager
            module._quant_enabled = True
