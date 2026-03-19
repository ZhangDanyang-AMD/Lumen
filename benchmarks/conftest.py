###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Shared pytest fixtures and markers for benchmarks."""

import pytest
import torch

try:
    import aiter  # noqa: F401

    _HAS_AITER = True
except ImportError:
    _HAS_AITER = False

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
AITER = pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_AITER,
    reason="CUDA + AITER required for benchmarks",
)
