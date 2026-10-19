"""Real public adapter checks; requires an explicitly authorized gfx950 allocation."""

import pytest
import torch

from lumen.ops.mxfp4_moe import _probe_aiter_mxfp4_training


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_public_training_capability_is_explicit():
    api = _probe_aiter_mxfp4_training()
    assert api is not None, "apply the reviewed AITER dependency patch first"
    cu = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    for k in (4096, 2048):
        spec = (1, 0, 950, cu, 1, 4096, k, 257, 0, 1, 1, 1, 2, 1, 1, 1, 2, 0, 1)
        capability = api.mxfp4_dense_query_static_v1(spec)
        assert capability.capability_status == 0
        assert not capability.capture_safe
