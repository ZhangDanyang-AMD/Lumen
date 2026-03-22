###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Shared pytest fixtures and markers for benchmarks."""

import warnings

import pytest
import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Suppress noisy third-party warnings (Megatron-LM, Apex, PyTorch dist)
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", message=".*Aiter backend is selected for fused RoPE.*")
warnings.filterwarnings("ignore", message=".*will be removed in.*megatron-core.*")
warnings.filterwarnings("ignore", message=".*megatron.core.transformer.custom_layers.*")
warnings.filterwarnings("ignore", message=".*No device id is provided via.*init_process_group.*")
warnings.filterwarnings("ignore", message=".*destroy_process_group.*was not called.*")

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


@pytest.fixture(scope="session", autouse=True)
def _cleanup_dist():
    """Ensure NCCL process group is destroyed on exit to avoid SIGSEGV."""
    yield
    if dist.is_initialized():
        dist.destroy_process_group()
