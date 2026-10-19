"""Two microbatches in one real Core window; authorized world8 execution UNRUN."""

import os

import pytest
import torch
import torch.distributed as dist

from lumen.models.mxfp4_moe_startup import RoutedCoreRuntime
from lumen.utils.mxfp4_moe_ledger import Mxfp4StateError
from tests.env.mxfp4_moe_fixture import build_core_layer, signed_config
from tests.env.mxfp4_moe_pytest import fixture_options


def _runtime(tmp_path, overlap, rows, suffix):
    ep = int(os.environ["MXFP4_TEST_EP"])
    config, ddp_config = signed_config(
        os.environ["MXFP4_SIGNED_CONTRACT"], ep=ep, overlap=overlap
    )
    torch.manual_seed(731)
    layers = torch.nn.ModuleList([build_core_layer(config) for _ in range(2)])
    runtime = RoutedCoreRuntime(
        layers,
        list(layers),
        options=fixture_options(),
        config=config,
        ddp_config=ddp_config,
        run_seed=731,
        manifest_digest=os.environ["MXFP4_RUN_MANIFEST_DIGEST"],
        physical_run_id=os.environ["MXFP4_PHYSICAL_RUN_ID"]
        + f"/microbatch/{overlap}/{rows}/{suffix}",
        journal_path=tmp_path / f"microbatch-{dist.get_rank()}.jsonl",
        sender_rows=(rows,) * ep,
        framework_resident_tensors=tuple(layer.router.weight for layer in layers),
        framework_future_bytes=int(os.environ["MXFP4_FRAMEWORK_FUTURE_BYTES"]),
        host_limit_bytes=int(os.environ["MXFP4_PINNED_BUDGET_BYTES"]),
        max_expert_rows=max(1, ep * rows),
        max_live_invocations_per_layer=2,
    )
    return runtime


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_two_microbatches_one_core_window(overlap, empty, checkpoint, tmp_path):
    rows = 0 if empty else 1
    runtime = _runtime(tmp_path, overlap, rows, f"success-{checkpoint}")
    groups = runtime.ddp.bucket_groups + runtime.ddp.expert_parallel_bucket_groups
    original_nodes = tuple(runtime.ddp.grad_accs)
    starts, methods = [], []
    for index, group in enumerate(groups):
        original = group.start_grad_sync

        def observe(*args, call=original, group_index=index, **kwargs):
            starts.append((group_index, runtime.microbatch))
            return call(*args, **kwargs)

        methods.append((group, original))
        group.start_grad_sync = observe  # Observe calls, never replace Core reduction.
    try:
        runtime.zero_boundary(2)
        first_values = None
        for microbatch in range(2):
            if first_values is not None:
                assert all(
                    torch.equal(parameter.main_grad, value)
                    for parameter, value in first_values
                ), "accumulation reset between microbatches"
            x = torch.randn(
                (rows, 1, 4096), device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            with runtime.microbatch_scope(microbatch):
                assert all(
                    group.is_last_microbatch == (microbatch == 1) for group in groups
                )
                y = runtime.forward(x, microbatch, checkpoint=checkpoint)
                assert runtime.microbatch == microbatch
                y.float().square().sum().backward()
                # Checkpoint replay and all original Core hooks execute before exit.
                assert all(
                    group.is_last_microbatch == (microbatch == 1) for group in groups
                )
                for layer in range(2):
                    invocation = runtime.lifecycle.active[
                        runtime.scopes[(microbatch, layer)]
                    ]
                    assert set(invocation.executions) == {1, 2}
                    for execution in invocation.executions.values():
                        assert execution.backward_started
                        assert all(
                            lease.armed and lease.consumed
                            for lease in execution.zero_leases
                        )
                        assert all(
                            lease.binding.parameter.grad is None
                            for lease in execution.zero_leases
                        )
                        # One W attempt per FC per microbatch, even for all-empty.
                        assert execution.obligations.attempts[2] == 0
                if not microbatch:
                    assert not starts
                    assert all(not group.params_with_grad for group in groups)
            assert runtime.microbatch is None
            assert all(group.is_last_microbatch for group in groups)
            assert x.grad is not None and x.grad.shape == x.shape
            if not microbatch:
                first_values = [
                    (parameter, parameter.main_grad.detach().clone())
                    for parameter in runtime.handoff.parameters
                ]
        if overlap:
            assert len(starts) == len(groups)
            assert all(microbatch == 1 for _, microbatch in starts)
            assert all(group.params_with_grad == set(group.params) for group in groups)
        else:
            assert not starts
        runtime.finish_grad_sync()
        assert len(starts) == len(
            groups
        ), "Core reducer dispatched more than once per group"
        assert tuple(runtime.ddp.grad_accs) == original_nodes
        counts = runtime.ledger.counts()
        assert (
            counts["declared"] == 24
            and counts["open"] == counts["pending_physical"] == 0
        )
        assert counts["EMPTY"] == (24 if empty else 0)
        assert not runtime.lifecycle.active and not runtime.handoff.pending
        for parameter in runtime.handoff.parameters:
            assert torch.isfinite(parameter.main_grad).all()
            if empty:
                assert torch.count_nonzero(parameter.main_grad) == 0
        runtime.optimizer_result(False)
    finally:
        for group, original in methods:
            group.start_grad_sync = original
        runtime.control.close()


@pytest.mark.parametrize("overlap", [False, True])
def test_microbatch_host_exception_restores_core_no_sync(overlap, tmp_path):
    runtime = _runtime(tmp_path, overlap, 0, "host-exception")
    groups = runtime.ddp.bucket_groups + runtime.ddp.expert_parallel_bucket_groups
    runtime.zero_boundary(2)
    failure = RuntimeError("injected healthy host interruption after forward")
    with pytest.raises(RuntimeError) as caught, runtime.microbatch_scope(0):
        x = torch.empty(
            (0, 1, 4096), device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        runtime.forward(x, 0)
        raise failure
    assert caught.value is failure
    assert all(group.is_last_microbatch for group in groups)
    assert all(not group.params_with_grad for group in groups)
    assert runtime.microbatch is None and runtime._next_microbatch == 0
    with pytest.raises(Mxfp4StateError, match="ABORTED_DRIVER_SCOPE"):
        runtime.finish_grad_sync()
    with pytest.raises(Mxfp4StateError, match="ABORTED_DRIVER_SCOPE"):
        runtime.optimizer_result(True)
    # No GPU cleanup/query/readback after injection; only close the CPU journal.
    runtime.control.close()
