"""Actual two-layer Core router/AlltoAll/DDP/recompute fixture; runtime unrun."""

import os

import pytest
import torch
import torch.distributed as dist

from lumen.models.mxfp4_moe_startup import RoutedCoreRuntime
from tests.env.mxfp4_moe_fixture import build_core_layer, signed_config
from tests.env.mxfp4_moe_pytest import fixture_options


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("sender_tokens", [0, 32, 128])
def test_two_layer_core_probability_and_recompute(
    checkpoint, overlap, sender_tokens, tmp_path
):
    if not dist.is_initialized():
        pytest.skip("authorized Core world8 initialization required")
    ep = int(os.environ["MXFP4_TEST_EP"])
    config, ddp_config = signed_config(
        os.environ["MXFP4_SIGNED_CONTRACT"], ep=ep, overlap=overlap
    )
    torch.manual_seed(1729)
    layers = [build_core_layer(config) for _ in range(2)]
    module = torch.nn.ModuleList(layers)
    # Framework demand excludes routed weights/main_grad handled by each binding.
    framework = tuple(layer.router.weight for layer in layers)
    runtime = RoutedCoreRuntime(
        module,
        layers,
        options=fixture_options(),
        config=config,
        ddp_config=ddp_config,
        run_seed=1729,
        manifest_digest=os.environ["MXFP4_RUN_MANIFEST_DIGEST"],
        physical_run_id=os.environ["MXFP4_PHYSICAL_RUN_ID"]
        + "/"
        + str(checkpoint)
        + "/"
        + str(overlap)
        + "/"
        + str(sender_tokens),
        journal_path=tmp_path / ("runtime-" + str(dist.get_rank()) + ".jsonl"),
        sender_rows=(sender_tokens,) * ep,
        framework_resident_tensors=framework,
        framework_future_bytes=int(os.environ["MXFP4_FRAMEWORK_FUTURE_BYTES"]),
        host_limit_bytes=int(os.environ["MXFP4_PINNED_BUDGET_BYTES"]),
        max_expert_rows=1024,
    )
    runtime.zero_boundary(1)
    x = torch.randn(
        (sender_tokens, 1, 4096),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    output = runtime.forward(x, 0, checkpoint=checkpoint)
    output.float().square().sum().backward()
    assert x.grad is not None
    for layer in layers:
        assert layer.router.weight.requires_grad
        if sender_tokens:
            assert layer.router.weight.main_grad is not None
            assert torch.isfinite(layer.router.weight.main_grad).all()
            assert torch.count_nonzero(layer.router.weight.main_grad).item() > 0
        for parameter in layer.experts.parameters():
            assert parameter.grad is None
            assert parameter.grad_added_to_main_grad
            assert torch.isfinite(parameter.main_grad).all()
    runtime.finish_grad_sync()
    counts = runtime.ledger.counts()
    assert counts["open"] == 0 and counts["pending_physical"] == 0
    assert counts["declared"] == 12
    assert counts["EMPTY"] == (12 if sender_tokens == 0 else 0)
    runtime.optimizer_result(False)
    runtime.control.close()
