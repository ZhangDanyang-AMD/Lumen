###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""MLP operators.

``fused_mlp`` and ``fused_gated_mlp`` stay available from this package.
Importing ``mlp_activation_patch`` does not load them. The SiLU-and-mul autograd
function lives in ``lumen.ops.silu_and_mul``.
"""

from __future__ import annotations

__all__ = ["fused_gated_mlp", "fused_mlp"]


def __getattr__(name):
    if name in ("fused_gated_mlp", "fused_mlp"):
        from .fused_mlp import fused_gated_mlp, fused_mlp

        globals()["fused_gated_mlp"] = fused_gated_mlp
        globals()["fused_mlp"] = fused_mlp
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
