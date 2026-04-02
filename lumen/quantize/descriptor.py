###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

from dataclasses import dataclass, field
from typing import Optional

import torch

__all__ = ["FP8Descriptor"]


@dataclass
class FP8Descriptor:
    """Lightweight FP8 data bundle with lazy transpose cache.

    NOT a Tensor subclass — just a plain Python object that keeps
    FP8 data, its scale, and an optional cached transpose together.
    """

    data: torch.Tensor
    scale: torch.Tensor
    fp8_dtype: torch.dtype
    _transpose: Optional[torch.Tensor] = field(default=None, repr=False)

    @property
    def transpose_cached(self) -> torch.Tensor:
        if self._transpose is None:
            self._transpose = self.data.t().contiguous()
        return self._transpose

    def invalidate_transpose(self) -> None:
        self._transpose = None

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data, self.scale

    @classmethod
    def from_tensors(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        fp8_dtype: torch.dtype,
    ) -> "FP8Descriptor":
        return cls(data=data, scale=scale, fp8_dtype=fp8_dtype)
