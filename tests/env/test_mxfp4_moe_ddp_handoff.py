"""Real Core EP expert-DDP + Z/G handoff test; invoke via authorized torchrun only."""

import os

import pytest
import torch
import torch.distributed as dist

from lumen.ops.mxfp4_moe import _probe_aiter_mxfp4_training
from lumen.utils.mxfp4_moe_ddp import Mxfp4DDPHandoff, ZeroLease
from tests.mxfp4_callback_observers import gradient_emitter


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("zero_out", [False, True])
def test_real_expert_ddp_zero_handoff(overlap, zero_out):
    if not dist.is_initialized() or dist.get_world_size() != 8:
        pytest.skip("authorized world8 Core initialization required")
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )

    from tests.env.mxfp4_moe_fixture import build_core_layer, signed_config

    ep = int(os.environ["MXFP4_TEST_EP"])
    config, ddp_config = signed_config(
        os.environ["MXFP4_SIGNED_CONTRACT"], ep=ep, overlap=overlap
    )
    layer = build_core_layer(config)
    ddp = DistributedDataParallel(config=config, ddp_config=ddp_config, module=layer)
    parameters = tuple(
        parameter for parameter in layer.experts.parameters() if parameter.requires_grad
    )
    handoff = Mxfp4DDPHandoff(ddp, parameters)
    ddp.zero_grad_buffer()
    zero = torch.cuda.Event()
    zero.record()
    bindings = handoff.open_after_zero(0, 0, zero)
    api = _probe_aiter_mxfp4_training()
    cu = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    from aiter.ops.triton.utils.config_utils import resolve_config_dir

    manifest = os.path.join(
        resolve_config_dir("gemm", "GEMM-AFP4WFP4"),
        "GEMM-AFP4WFP4-CONTRACT-MANIFEST-V1.json",
    )
    prepared = api.mxfp4_dense_prepare_kernel_v1(
        (1, 0, 950, cu, 1, 4096, 2048, 257, 0, 1, 1, 1, 2, 1, 1, 1, 2, 0, 1), manifest
    )
    for binding in bindings:
        parameter = binding.parameter
        parameter.zero_out_wgrad = zero_out
        z, g = torch.empty_like(parameter), torch.empty_like(parameter)
        for tensor in (z, g):
            api.mxfp4_zero_output_out_v1(
                tensor,
                *parameter.shape,
                0,
                (1, 0, 5, 0, 0, 0),
                prepared.config_view.primitive(5, variant=4, output_dtype=0).config
            )
        events = [torch.cuda.Event() for _ in range(3)]
        for event in events:
            event.record()
        lease = ZeroLease(binding, z, g, *events)

        class Emit(torch.autograd.Function):
            @staticmethod
            def forward(ctx, weight):
                return weight.new_empty((0,))

            backward = staticmethod(gradient_emitter(handoff, lease))

        Emit.apply(parameter).sum().backward()
        assert lease.consumed and parameter.grad is None
        assert torch.count_nonzero(parameter.main_grad).item() == 0
        assert torch.count_nonzero(z.view(torch.int16)).item() == 0
        assert torch.count_nonzero(g.view(torch.int16)).item() == 0
    handoff.before_grad_sync()
    ddp.finish_grad_sync()
    end = torch.cuda.Event()
    end.record()
    handoff.close_after_grad_sync(end)
    handoff.remove()
