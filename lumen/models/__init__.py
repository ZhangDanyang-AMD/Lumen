###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Reusable model definitions and training utilities.

Common, model-agnostic helpers are available from ``lumen.models.utils``.
Shared backend-specific helpers live in :mod:`~lumen.models.megatron`,
:mod:`~lumen.models.fsdp`, and :mod:`~lumen.models.deepspeed`.
Model-specific implementations live in subpackages (e.g. ``llama2``).
"""

import importlib

from lumen.models.utils import (  # noqa: F401
    download_hf_dataset,
    download_hf_model,
    peek_backend,
    safe_add_argument,
    sha256_file,
)


def __getattr__(name: str):
    if name == "fsdp":
        return importlib.import_module("lumen.models.fsdp")
    if name == "megatron":
        return importlib.import_module("lumen.models.megatron")
    if name == "deepspeed":
        return importlib.import_module("lumen.models.deepspeed")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
