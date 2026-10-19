"""Public caller-output numerical checks for both FC shapes and ragged rows."""

import os

import pytest
import torch

from lumen.ops.mxfp4_moe import _probe_aiter_mxfp4_training
from tests.ops.mxfp4_moe_numeric_reference import assert_metrics, decode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("k", [4096, 2048])
@pytest.mark.parametrize("m", [0, 1, 15, 16, 31, 32, 33, 63, 64, 127, 128, 129, 257])
@pytest.mark.parametrize("pass_id", [0, 1, 2])
def test_public_fixed_packed_numeric(k, m, pass_id):
    from aiter.ops.triton.utils.config_utils import resolve_config_dir

    api = _probe_aiter_mxfp4_training()
    cu = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    prepared = api.mxfp4_dense_prepare_kernel_v1(
        (1, 0, 950, cu, 1, 4096, k, 257, 0, 1, 1, 1, 2, 1, 1, 1, 2, 0, 1),
        os.path.join(
            resolve_config_dir("gemm", "GEMM-AFP4WFP4"),
            "GEMM-AFP4WFP4-CONTRACT-MANIFEST-V1.json",
        ),
    )
    p = (m + 31) // 32 * 32
    mg, ng, kg = ((m, 4096, k), (m, k, 4096), (4096, k, p))[pass_id]
    group = 1 if pass_id == 2 else 32
    a = torch.randint(0, 256, (mg, kg // 2), device="cuda", dtype=torch.uint8)
    b = torch.randint(0, 256, (ng, kg // 2), device="cuda", dtype=torch.uint8)
    sa = torch.full((mg, kg // 32), 124, dtype=torch.uint8, device="cuda")
    sb = torch.full((ng // group, kg // 32), 124, dtype=torch.uint8, device="cuda")
    out = torch.full(
        (mg, ng),
        0.001,
        dtype=torch.float32 if pass_id == 2 else torch.bfloat16,
        device="cuda",
    )
    before = out.clone()
    workspace = torch.empty((0,), dtype=torch.uint8, device="cuda")
    status = torch.zeros((1,), dtype=torch.int32, device="cuda")
    dtype, mode = (1, 1) if pass_id == 2 else (0, 0)
    config = prepared.records[
        (4, pass_id, 0, 0, dtype, group, 4096, k, mode, "any")
    ].config
    api.gemm_afp4wfp4_out_v1(
        a,
        b,
        sa,
        sb,
        out,
        workspace,
        status,
        mg,
        ng,
        kg,
        pass_id,
        group,
        mode,
        dtype,
        (1, 0, 4, 1, 2, 0),
        config,
    )
    if m == 0:
        assert torch.equal(out, before)
    else:
        reference = decode(a, sa, mg, kg) @ decode(b, sb, ng, kg, group).T
        if pass_id == 2:
            reference += before
        assert_metrics(reference, out)
    assert status.item() == 0
