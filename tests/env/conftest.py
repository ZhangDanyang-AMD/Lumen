"""Explicit torchrun initialization for MXFP4-only isolated EP tests."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist


@pytest.fixture(autouse=True)
def initialized_core(request):
    if "mxfp4_moe" not in request.node.nodeid or request.node.nodeid.split("::", 1)[
        0
    ].endswith("_host.py"):
        yield
        return
    if os.environ.get("MXFP4_AUTHORIZED_RUNTIME") != "1":
        pytest.skip("runtime not authorized; source-only delivery")
    from tests.env.mxfp4_moe_pytest import fixture_options

    fixture_options()  # Explicit selector is required before the first GPU/PG call.
    if int(os.environ.get("WORLD_SIZE", "0")) != 8:
        pytest.fail("signed fixture requires world8")
    from megatron.core import parallel_state

    ep = int(os.environ["MXFP4_TEST_EP"])
    assert ep in (4, 8)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=600))
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=ep,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=1,
            order="tp-cp-ep-dp-pp",
        )
    yield
