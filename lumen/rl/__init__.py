###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""RL integrations for Lumen."""

import importlib

__all__ = ["trl"]


def __getattr__(name: str):
    if name == "trl":
        return importlib.import_module("lumen.rl.trl")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
