###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

from collections import defaultdict, deque
from typing import Optional

import torch

from transformer_light.quantize.config import (
    AmaxAlgo,
    QuantConfig,
    QuantFormat,
    ScalingType,
    get_fp8_max,
    get_fp8_max_bwd,
)
from transformer_light.pytorch.ops.quantize import (
    convert_to_mxfp8,
    quant_fp8_blockwise_impl,
)


class ScalingManager:
    """Tracks per-tensor scaling factors across training iterations.

    Accepts either a :class:`QuantConfig` or legacy keyword arguments.

    Examples::

        # With QuantConfig
        mgr = ScalingManager(QuantConfig(format=QuantFormat.MXFP8,
                                         scaling=ScalingType.BLOCKWISE))

        # MLPerf-style: most_recent amax, history=4
        mgr = ScalingManager(QuantConfig(amax_algo=AmaxAlgo.MOST_RECENT,
                                         history_len=4))

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
        self.fp8_dtype_bwd = (
            config.torch_dtype_bwd or self.fp8_dtype if config else self.fp8_dtype
        )
        self._fp8_max = self.config.fp8_max
        self._fp8_max_bwd = self.config.fp8_max_bwd
        self._margin = self.config.margin
        self.amax_history = defaultdict(lambda: deque(maxlen=self.config.history_len))
        self.scale_cache = {}
        self._dp_group = None

    @property
    def recipe(self) -> str:
        return self.config.recipe

    def set_dp_group(self, group) -> None:
        """Set the data-parallel process group for ``reduce_amax``."""
        self._dp_group = group

    def _compute_scale(self, amax: torch.Tensor, fp8_max: float) -> torch.Tensor:
        """Compute the quantization scale from *amax*, accounting for margin.

        Matches TE convention: ``sf = (fp8_max / amax) / (2 ** margin)``,
        returned as ``amax / (fp8_max / (2 ** margin))`` so that the caller
        can divide by the scale to quantize.
        """
        effective_max = fp8_max / (2 ** self._margin)
        scale = amax / effective_max
        scale = torch.where(amax > 0.0, scale, torch.ones_like(scale))
        return scale

    def get_scale(self, tensor_id: str, tensor: torch.Tensor,
                  *, backward: bool = False):
        """Return the scale factor for this tensor (None for block/mxfp8)."""
        recipe = self.recipe
        fp8_max = self._fp8_max_bwd if backward else self._fp8_max

        if recipe == "delayed":
            history = self.amax_history[tensor_id]
            if len(history) == 0:
                amax = tensor.abs().amax()
            elif self.config.amax_algo == AmaxAlgo.MOST_RECENT:
                amax = history[-1].to(device=tensor.device)
            else:
                amax = torch.stack(list(history)).amax().to(device=tensor.device)

            if self.config.reduce_amax and self._dp_group is not None:
                torch.distributed.all_reduce(
                    amax, op=torch.distributed.ReduceOp.MAX,
                    group=self._dp_group,
                )

            return self._compute_scale(amax, fp8_max)
        elif recipe == "dynamic":
            amax = tensor.abs().amax()
            if self.config.reduce_amax and self._dp_group is not None:
                torch.distributed.all_reduce(
                    amax, op=torch.distributed.ReduceOp.MAX,
                    group=self._dp_group,
                )
            return self._compute_scale(amax, fp8_max)
        elif recipe in ("blockwise", "mxfp8"):
            return None

    def update_amax(self, tensor_id: str, tensor: torch.Tensor):
        """Record amax for delayed scaling (tensor-based, no .item() sync)."""
        self.amax_history[tensor_id].append(tensor.detach().abs().amax())

    def quantize(self, tensor_id: str, tensor: torch.Tensor,
                 *, backward: bool = False):
        """Quantize tensor. Returns (quantized_tensor, scale).

        When *backward* is True and the format is HYBRID, E5M2 dtype and its
        corresponding FP8_MAX are used instead of the forward-pass values.
        """
        scale = self.get_scale(tensor_id, tensor, backward=backward)
        fp8_max = self._fp8_max_bwd if backward else self._fp8_max
        dtype = self.fp8_dtype_bwd if backward else self.fp8_dtype

        if scale is None and self.config.format == QuantFormat.MXFP8:
            return convert_to_mxfp8(
                tensor,
                block_size=self.config.block_size,
                axis=-1,
                float8_dtype_pt=dtype,
            )

        if scale is None and self.config.scaling == ScalingType.BLOCKWISE:
            orig_shape = tensor.shape
            flat = tensor.reshape(-1, orig_shape[-1])
            fp8_tensor, fp8_scales = quant_fp8_blockwise_impl(
                flat.contiguous(),
                dtype=dtype,
                axis=1,
                block_size=self.config.block_size,
            )
            return fp8_tensor.view(orig_shape), fp8_scales

        fp8_tensor = (tensor * (1.0 / scale)).clamp(
            -fp8_max, fp8_max
        ).to(dtype)
        self.update_amax(tensor_id, tensor)
        return fp8_tensor, scale

    def reset(self):
        """Clear all tracked state (e.g. after warmup)."""
        self.amax_history.clear()
        self.scale_cache.clear()


# Backward-compat alias
FP8ScalingManager = ScalingManager
