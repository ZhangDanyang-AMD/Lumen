###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""
lumen.quantize — low-precision training lifecycle for AMD GPUs.

Supports FP8 (E4M3 / E5M2), MXFP8, and FP4 formats.

Usage::

    import lumen.quantize as quant
    from lumen.quantize import QuantConfig, QuantFormat, ScalingType

    # Configure quantization
    config = QuantConfig(format=QuantFormat.FP8_E4M3,
                         scaling=ScalingType.DELAYED)

    # Non-invasive: patch existing model, no module replacement
    quant.enable(model, config=config)

    # Or use string shorthand
    quant.enable(model, format="fp8_e4m3", scaling="delayed")

    # Check backend availability
    backend = quant.get_attention_backend()   # "aiter_csrc" or "aiter_triton"
"""

import functools
import logging
import re
from typing import Optional, Set

import torch
import torch.nn as nn

from lumen.ops.quantize import (
    QuantizedLinearFunction,
    convert_from_mxfp8,
    convert_to_mxfp8,
    dequant_fp8_tensorwise_impl,
    quant_fp8_blockwise_impl,
    quant_fp8_blockwise_segment_m_impl,
    quant_fp8_tensorwise_impl,
    quantized_linear,
)
from lumen.quantize.config import (
    AmaxAlgo,
    QuantConfig,
    QuantFormat,
    ScalingType,
    get_fp8_max,
    get_fp8_max_bwd,
)
from lumen.quantize.optimizer_manager import (
    FP32MasterWeightOptimizer,
    get_scaling_manager,
)
from lumen.quantize.scaling_manager import (
    GRAD_QUANT_TYPES,
    ScalingManager,
)

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
        prefer: One of ``"auto"``, ``"aiter_csrc"``, ``"aiter_triton"``.

    Returns:
        ``"aiter_csrc"`` or ``"aiter_triton"``
    """
    if prefer == "aiter_triton":
        return "aiter_triton"

    if prefer == "aiter_csrc":
        if not is_aiter_available():
            raise RuntimeError(
                "AITER is not installed. The aiter_csrc attention backend requires "
                "'aiter' — install it or set backend to 'aiter_triton'."
            )
        return "aiter_csrc"

    # auto
    if is_aiter_available():
        logger.info("AITER detected — using aiter_csrc attention backend")
        return "aiter_csrc"
    logger.info("AITER not found — falling back to aiter_triton attention backend")
    return "aiter_triton"


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
            raise RuntimeError("AITER is not installed. Install it or use backend='triton'.")
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


_LAYER_INDEX_RE = re.compile(r"layers\.(\d+)\b")


def _build_bf16_skip_prefixes(
    model: nn.Module,
    config: QuantConfig,
) -> Set[str]:
    """Return a set of module-name prefixes whose transformer layers should
    stay in BF16 (not be FP8-patched).

    Strategy:
    1. Walk the model looking for modules that expose a ``layer_number``
       attribute (Megatron ``TransformerLayer`` — 1-indexed global, correct
       even under pipeline parallelism).
    2. If none are found (HuggingFace / FSDP models), fall back to extracting
       the layer index from the module path (``layers.<N>``).
    """
    if not config.first_last_layers_bf16:
        return set()

    bf16_start = config.num_layers_at_start_in_bf16
    bf16_end = config.num_layers_at_end_in_bf16
    total = config.num_layers

    def _should_skip(global_idx: int) -> bool:
        return global_idx < bf16_start or global_idx >= total - bf16_end

    prefixes: Set[str] = set()

    # --- Strategy 1: Megatron layer_number (global, 1-indexed) ---
    for name, module in model.named_modules():
        layer_num = getattr(module, "layer_number", None)
        if layer_num is not None and isinstance(layer_num, int):
            global_idx = layer_num - 1
            if _should_skip(global_idx):
                prefixes.add(name)

    if prefixes:
        return prefixes

    # --- Strategy 2: path-based detection (HF / FSDP) ---
    layer_prefixes: dict[str, int] = {}
    for name, _ in model.named_modules():
        m = _LAYER_INDEX_RE.search(name)
        if m:
            prefix = name[: m.end()]
            idx = int(m.group(1))
            if prefix not in layer_prefixes or idx < layer_prefixes[prefix]:
                layer_prefixes[prefix] = idx

    if not layer_prefixes:
        return set()

    detected_max = max(
        int(m.group(1)) for name, _ in model.named_modules() for m in [_LAYER_INDEX_RE.search(name)] if m
    )
    effective_total = total if total > 0 else detected_max + 1

    for name, _ in model.named_modules():
        m = _LAYER_INDEX_RE.search(name)
        if m:
            idx = int(m.group(1))
            if idx < bf16_start or idx >= effective_total - bf16_end:
                prefixes.add(name[: m.end()])

    return prefixes


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
    fp8_wgrad = config.fp8_wgrad
    scaling_type = config.scaling.value

    megatron_types = _get_megatron_linear_types()
    quantizable_types = (nn.Linear,) + megatron_types

    bf16_prefixes = _build_bf16_skip_prefixes(model, config)

    count = 0
    skipped = 0
    for name, module in model.named_modules():
        if isinstance(module, quantizable_types):
            if bf16_prefixes and any(name.startswith(p) for p in bf16_prefixes):
                skipped += 1
                continue

            tensor_id = f"{name}.weight" if name else "weight"
            module._quant_manager = manager
            module._quant_backend = backend
            module._quant_enabled = True
            module._quant_tensor_id = tensor_id
            module._quant_scaling_type = scaling_type

            is_megatron = megatron_types and isinstance(module, megatron_types)
            _replace_forward(
                module,
                manager,
                backend,
                fp8_dtype,
                block_size,
                tensor_id,
                quant_act,
                fp8_wgrad,
                is_megatron,
                scaling_type,
            )
            count += 1

    act_str = "weight+activation" if quant_act else "weight-only"
    grad_quant_type = config.quantize_grad
    grad_str = f"+grad({grad_quant_type})" if grad_quant_type else ""
    bf16_str = f", bf16_layers_skipped={skipped}" if skipped else ""
    logger.info(
        "Quantization enabled on %d nn.Linear layers " "(backend=%s, format=%s, scaling=%s, amax_algo=%s, %s%s%s)",
        count,
        backend,
        config.format.value,
        config.scaling.value,
        config.amax_algo.value,
        act_str,
        grad_str,
        bf16_str,
    )


def _replace_forward(
    module,
    manager,
    backend,
    fp8_dtype,
    block_size,
    tensor_id,
    quantize_activation,
    fp8_wgrad,
    is_megatron,
    scaling_type="delayed",
):
    """Replace the module's forward method with an FP8-quantized version.

    Unlike ``register_forward_hook``, this prevents the original (BF16) linear
    from running at all, saving both compute and peak memory — critical for
    70B-class models under FSDP.
    """
    original_forward = module.forward

    _delay_wgrad = getattr(module, "delay_wgrad", False)
    _deferred_wgrad = getattr(module, "_deferred_wgrad", None) if _delay_wgrad else None
    _gaf = getattr(module, "gradient_accumulation_fusion", False)
    _fp8_act_store = getattr(module, "fp8_activation_store", False)

    if not is_megatron:

        def quant_forward(input_tensor, *args, **kwargs):
            return quantized_linear(
                input_tensor,
                module.weight,
                module.bias,
                scaling_manager=manager,
                backend=backend,
                scaling_type=scaling_type,
                fp8_dtype=fp8_dtype,
                block_size=block_size,
                tensor_id=tensor_id,
                quantize_activation=quantize_activation,
                fp8_wgrad=fp8_wgrad,
                gradient_accumulation_fusion=_gaf,
                delay_wgrad=_delay_wgrad,
                deferred_wgrad=_deferred_wgrad,
                fp8_activation_store=_fp8_act_store,
            )

    else:

        def quant_forward(input_tensor, *args, **kwargs):
            skip_bias_add = getattr(module, "skip_bias_add", False)
            bias = getattr(module, "bias", None)
            bias_for_gemm = None if skip_bias_add else bias

            is_row_parallel = getattr(module, "input_is_parallel", False)
            seq_parallel = getattr(module, "sequence_parallel", False)
            tp_group = getattr(module, "tp_group", None)

            if seq_parallel and not is_row_parallel:
                from megatron.core.tensor_parallel.mappings import (
                    gather_from_sequence_parallel_region,
                )

                input_tensor = gather_from_sequence_parallel_region(
                    input_tensor,
                    tensor_parallel_output_grad=True,
                    group=tp_group,
                )

            result = quantized_linear(
                input_tensor,
                module.weight,
                bias_for_gemm,
                scaling_manager=manager,
                backend=backend,
                scaling_type=scaling_type,
                fp8_dtype=fp8_dtype,
                block_size=block_size,
                tensor_id=tensor_id,
                quantize_activation=quantize_activation,
                fp8_wgrad=fp8_wgrad,
                gradient_accumulation_fusion=_gaf,
                delay_wgrad=_delay_wgrad,
                deferred_wgrad=_deferred_wgrad,
                fp8_activation_store=_fp8_act_store,
            )

            if is_row_parallel and seq_parallel:
                from megatron.core.tensor_parallel.mappings import (
                    reduce_scatter_to_sequence_parallel_region,
                )

                result = reduce_scatter_to_sequence_parallel_region(
                    result,
                    group=tp_group,
                )
            elif is_row_parallel:
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

            output_bias = bias if skip_bias_add else None
            return result, output_bias

    module._original_forward = original_forward
    module.forward = quant_forward


def disable(model: nn.Module) -> None:
    """Remove FP8 quantized forward from all patched layers."""
    for module in model.modules():
        if hasattr(module, "_original_forward"):
            module.forward = module._original_forward
            del module._original_forward
            module._quant_enabled = False
