###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Quantization configuration for Transformer Light.

Supports FP8 (E4M3 / E5M2), MXFP8, and FP4 formats with multiple scaling
strategies.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch


class QuantFormat(Enum):
    """Supported low-precision number formats."""

    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    MXFP8 = "mxfp8"
    FP4 = "fp4"


class ScalingType(Enum):
    """How scaling factors are computed."""

    DYNAMIC = "dynamic"      # Scale from current tensor amax
    DELAYED = "delayed"      # Scale from amax history (TE-style)
    BLOCKWISE = "blockwise"  # Per-block scaling (e.g. per-128 elements)


class AmaxAlgo(Enum):
    """How delayed-scaling amax is derived from the history window."""

    MAX = "max"                    # max over the entire history window (TE default)
    MOST_RECENT = "most_recent"    # use only the latest recorded amax


# Mapping from QuantFormat to PyTorch dtype (where applicable)
_FORMAT_TO_DTYPE = {
    QuantFormat.FP8_E4M3: torch.float8_e4m3fn,
    QuantFormat.FP8_E5M2: torch.float8_e5m2,
    QuantFormat.MXFP8: torch.float8_e4m3fn,
    QuantFormat.FP4: None,  # no native torch dtype yet
}


@dataclass
class QuantConfig:
    """Unified quantization configuration.

    Examples::

        # FP8 with delayed scaling (default)
        cfg = QuantConfig()

        # MXFP8 with blockwise scaling
        cfg = QuantConfig(format=QuantFormat.MXFP8,
                          scaling=ScalingType.BLOCKWISE,
                          block_size=32)

        # MLPerf-style: most_recent amax, short history, activation quant on
        cfg = QuantConfig(amax_algo=AmaxAlgo.MOST_RECENT,
                          history_len=4,
                          quantize_activation=True)

        # From strings (handy for YAML / env-var configs)
        cfg = QuantConfig.from_str("fp8_e4m3", "delayed",
                                   amax_algo="most_recent")
    """

    format: QuantFormat = QuantFormat.FP8_E4M3
    scaling: ScalingType = ScalingType.DELAYED
    block_size: int = 32
    history_len: int = 16

    # Amax algorithm for delayed scaling
    amax_algo: AmaxAlgo = AmaxAlgo.MAX

    # Whether to all-reduce amax across data-parallel ranks before computing
    # the scale.  Useful for large-scale runs where per-rank amax can diverge.
    reduce_amax: bool = False

    # Whether to quantize activations (input to linear layers) in addition to
    # weights.  When False, only weights are quantized — activations stay in
    # the original precision (BF16/FP16).
    quantize_activation: bool = True

    @classmethod
    def from_str(cls, format: str = "fp8_e4m3", scaling: str = "delayed",
                 **kwargs) -> "QuantConfig":
        """Construct a QuantConfig from plain strings.

        Args:
            format: One of ``"fp8_e4m3"``, ``"fp8_e5m2"``, ``"mxfp8"``, ``"fp4"``.
            scaling: One of ``"dynamic"``, ``"delayed"``, ``"blockwise"``.
            **kwargs: Forwarded to :class:`QuantConfig`.  String values for
                enum fields (``amax_algo``) are auto-converted.
        """
        if "amax_algo" in kwargs and isinstance(kwargs["amax_algo"], str):
            kwargs["amax_algo"] = AmaxAlgo(kwargs["amax_algo"])
        return cls(
            format=QuantFormat(format),
            scaling=ScalingType(scaling),
            **kwargs,
        )

    @property
    def torch_dtype(self) -> Optional[torch.dtype]:
        """Return the PyTorch FP8 dtype for this format, or None if unavailable."""
        return _FORMAT_TO_DTYPE.get(self.format)

    @property
    def recipe(self) -> str:
        """Legacy recipe string expected by ScalingManager."""
        if self.format == QuantFormat.MXFP8:
            return "mxfp8"
        return self.scaling.value
