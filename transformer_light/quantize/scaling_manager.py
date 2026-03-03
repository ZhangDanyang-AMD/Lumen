###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

from collections import defaultdict, deque
from typing import Optional, Union

import torch

from transformer_light.quantize.config import QuantConfig, QuantFormat, ScalingType
from transformer_light.pytorch.ops import (
    convert_to_mxfp8,
    quant_fp8_blockwise_impl,
)

FP8_MAX = 448.0  # E4M3 (240.0 for FNUZ E4M3)


class ScalingManager:
    """Tracks per-tensor scaling factors across training iterations.

    Accepts either a :class:`QuantConfig` or legacy keyword arguments.

    Examples::

        # With QuantConfig
        mgr = ScalingManager(QuantConfig(format=QuantFormat.MXFP8,
                                         scaling=ScalingType.BLOCKWISE))

        # Legacy style (still supported)
        mgr = ScalingManager(recipe="delayed")
    """

    def __init__(
        self,
        config: Optional[QuantConfig] = None,
        *,
        recipe: Optional[str] = None,
        history_len: int = 16,
        block_size: int = 32,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    ):
        if config is not None:
            self.config = config
        elif recipe is not None:
            self.config = QuantConfig.from_str(
                format="mxfp8" if recipe == "mxfp8" else "fp8_e4m3",
                scaling=recipe if recipe not in ("mxfp8", "block") else "blockwise",
                block_size=block_size,
                history_len=history_len,
            )
        else:
            self.config = QuantConfig(block_size=block_size, history_len=history_len)

        self.fp8_dtype = config.torch_dtype or fp8_dtype if config else fp8_dtype
        self.amax_history = defaultdict(lambda: deque(maxlen=self.config.history_len))
        self.scale_cache = {}

    @property
    def recipe(self) -> str:
        return self.config.recipe

    def get_scale(self, tensor_id: str, tensor: torch.Tensor):
        """Return the scale factor for this tensor (None for block/mxfp8)."""
        recipe = self.recipe
        if recipe == "delayed":
            history = self.amax_history[tensor_id]
            if len(history) == 0:
                amax = tensor.abs().amax()
            else:
                amax = max(history)
            return amax / FP8_MAX
        elif recipe == "dynamic":
            return tensor.abs().amax() / FP8_MAX
        elif recipe in ("blockwise", "mxfp8"):
            return None

    def update_amax(self, tensor_id: str, tensor: torch.Tensor):
        """Record amax for delayed scaling."""
        self.amax_history[tensor_id].append(tensor.abs().amax().item())

    def quantize(self, tensor_id: str, tensor: torch.Tensor):
        """Quantize tensor. Returns (quantized_tensor, scale)."""
        scale = self.get_scale(tensor_id, tensor)

        if scale is None and self.config.format == QuantFormat.MXFP8:
            return convert_to_mxfp8(
                tensor,
                block_size=self.config.block_size,
                axis=-1,
                float8_dtype_pt=self.fp8_dtype,
            )

        if scale is None and self.config.scaling == ScalingType.BLOCKWISE:
            orig_shape = tensor.shape
            flat = tensor.reshape(-1, orig_shape[-1])
            fp8_tensor, fp8_scales = quant_fp8_blockwise_impl(
                flat.contiguous(),
                dtype=self.fp8_dtype,
                axis=1,
                block_size=self.config.block_size,
            )
            return fp8_tensor.view(orig_shape), fp8_scales

        fp8_tensor = (tensor * (1.0 / scale)).clamp(
            -FP8_MAX, FP8_MAX
        ).to(self.fp8_dtype)
        self.update_amax(tensor_id, tensor)
        return fp8_tensor, scale

    def reset(self):
        """Clear all tracked state (e.g. after warmup)."""
        self.amax_history.clear()
        self.scale_cache.clear()


# Backward-compat alias
FP8ScalingManager = ScalingManager
