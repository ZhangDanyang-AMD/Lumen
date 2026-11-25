###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Normalization operators.

The public names below still resolve through ``from lumen.ops.normalization
import ...``. Importing ``rmsnorm_with_residual`` or ``rmsnorm_layernorm_patch`` does not load the
Triton norm modules until a name from this package is requested.
"""

from __future__ import annotations

__all__ = [
    "LumenRMSNorm",
    "rmsnorm",
    "rmsnorm_with_quant",
    "LumenLayerNorm",
    "layernorm",
    "layernorm_with_quant",
]

_LAYERNORM = ("LumenLayerNorm", "layernorm", "layernorm_with_quant")
_RMSNORM = ("LumenRMSNorm", "rmsnorm", "rmsnorm_with_quant")


def __getattr__(name):
    if name in _LAYERNORM:
        from .layernorm import LumenLayerNorm, layernorm, layernorm_with_quant

        globals()["LumenLayerNorm"] = LumenLayerNorm
        globals()["layernorm"] = layernorm
        globals()["layernorm_with_quant"] = layernorm_with_quant
        return globals()[name]
    if name in _RMSNORM:
        from .rmsnorm import LumenRMSNorm, rmsnorm, rmsnorm_with_quant

        globals()["LumenRMSNorm"] = LumenRMSNorm
        globals()["rmsnorm"] = rmsnorm
        globals()["rmsnorm_with_quant"] = rmsnorm_with_quant
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
