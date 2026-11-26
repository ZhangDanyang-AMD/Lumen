###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""tests/models fixtures. BLAS-free ``torch.linalg.solve_triangular`` for CPU tensors on torch builds without CPU LAPACK.

The ROCm test container's torch raises "torch.triangular_solve on a CPU tensor requires BLAS".
aiter's torch KDA reference (``_chunk_step``) solves a unit lower-triangular system per chunk;
this forward substitution covers that case on CPU and defers everything else to the original.
"""

import pytest
import torch

_orig = torch.linalg.solve_triangular


def cpu_lapack_available():
    try:
        _orig(torch.eye(2), torch.ones(2, 1), upper=False)
        return True
    except RuntimeError:
        return False


def _forward_substitution(A, B, *, upper, left=True, unitriangular=False, out=None):
    if A.is_cuda or upper or not left or out is not None:
        return _orig(A, B, upper=upper, left=left, unitriangular=unitriangular, out=out)
    n = A.shape[-1]
    rows = []
    for i in range(n):
        r = B[..., i, :]
        if i:
            r = r - (A[..., i, :i].unsqueeze(-2) @ torch.stack(rows, -2)).squeeze(-2)
        if not unitriangular:
            r = r / A[..., i, i].unsqueeze(-1)
        rows.append(r)
    return torch.stack(rows, -2)


@pytest.fixture(autouse=True)
def blas_free_cpu_solve(monkeypatch):
    if not cpu_lapack_available():
        monkeypatch.setattr(torch.linalg, "solve_triangular", _forward_substitution)
    yield
