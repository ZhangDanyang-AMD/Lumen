###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kimi-K3 / K3.1 training helpers."""

from lumen.models.kimi.kda_aiter import (  # noqa: F401
    KDA_AITER_BACKEND_ENV,
    KDA_AITER_ENV,
    kda_aiter_enabled,
    kda_layer_weights,
    maybe_patch_kda_layers,
    patch_kda_layers,
    unpatch_kda_layers,
)
