###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Environment switches for AITER's MHA behind the ``aiter_csrc`` attention backend.

``LUMEN_ATTN_AITER_TRITON_MHA=1``
    Serve ``aiter.ops.mha.flash_attn_func`` from AITER's Triton MHA, for AITER builds
    without the CK MHA kernels. Only the attention dispatcher changes; AITER's
    process-wide ``ENABLE_CK=0`` would also move other AITER ops off CK.
``LUMEN_ATTN_AITER_FUSED_BWD=1``
    Use AITER's fused dq/dk/dv Triton backward instead of its one-kernel backward.
    It needs an AITER that zero-initialises dK/dV for the fused kernel.

Both default to off.
"""

import os

_state = {"triton_mha": False, "fused_bwd": False}


def _flag(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def configure_aiter_mha() -> dict:
    """Apply the switches to the imported AITER modules; returns what was enabled."""
    if _flag("LUMEN_ATTN_AITER_TRITON_MHA"):
        import aiter.ops.mha as aiter_mha

        aiter_mha.ENABLE_CK = False
        _state["triton_mha"] = True
    if _flag("LUMEN_ATTN_AITER_FUSED_BWD"):
        from aiter.ops.triton.attention.mha import mha_set_use_fused_bwd_kernel

        mha_set_use_fused_bwd_kernel(True)
        _state["fused_bwd"] = True
    return dict(_state)
