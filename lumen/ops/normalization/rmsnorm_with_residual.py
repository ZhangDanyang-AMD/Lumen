###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""RMSNorm module backed by ``aiter.rmsnorm2d_fwd``."""

from __future__ import annotations

import torch

from lumen.kernels.rmsnorm_autograd import rmsnorm_fns


class RMSNormWithResidual(torch.nn.Module):
    """RMSNorm whose forward is ``aiter.rmsnorm2d_fwd``.

    A residual argument selects ``rmsnorm2d_fwd_with_add`` and returns
    ``(normed, residual_out)``.
    """

    def __init__(self, weight, eps):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.data.clone())
        self.eps = eps

    def forward(self, x, residual=None):
        plain, fused = rmsnorm_fns()
        if residual is None:
            return plain.apply(x, self.weight, self.eps)
        return fused.apply(x, residual, self.weight, self.eps)
