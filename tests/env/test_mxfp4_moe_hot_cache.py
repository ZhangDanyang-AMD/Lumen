"""Two legal accumulation windows, real cold publication then hot QW reuse."""

import copy
import os

import pytest
import torch
import torch.distributed as dist

from lumen.models.mxfp4_moe_startup import RoutedCoreRuntime
from tests.env.mxfp4_moe_fixture import build_core_layer, signed_config
from tests.env.mxfp4_moe_pytest import fixture_options
from tests.env.test_mxfp4_moe_cross_mode import _equal_tree
from tests.mxfp4_callback_observers import capture_probability

_ROLES = ("w_fp4", "w_scales", "wt_fp4", "wt_scales")


def _arm(disabled, directory):
    from aiter.ops.triton.quant import quant as quant_module

    ep = int(os.environ["MXFP4_TEST_EP"])
    config, ddp = signed_config(
        os.environ["MXFP4_SIGNED_CONTRACT"], ep=ep, overlap=False
    )
    torch.manual_seed(2909)
    layers = torch.nn.ModuleList([build_core_layer(config) for _ in range(2)])
    inputs = tuple(
        torch.randn((1, 1, 4096), dtype=torch.bfloat16, device="cuda") for _ in layers
    )
    upstream = tuple(torch.randn_like(x) for x in inputs)
    runtime = RoutedCoreRuntime(
        layers,
        list(layers),
        options=fixture_options(),
        config=config,
        ddp_config=ddp,
        run_seed=2909,
        manifest_digest=os.environ["MXFP4_RUN_MANIFEST_DIGEST"],
        physical_run_id=os.environ["MXFP4_PHYSICAL_RUN_ID"] + "/hot/" + str(disabled),
        journal_path=directory / ("rank-" + str(dist.get_rank()) + ".jsonl"),
        sender_rows=(1,) * ep,
        framework_resident_tensors=tuple(layer.router.weight for layer in layers),
        framework_future_bytes=int(os.environ["MXFP4_FRAMEWORK_FUTURE_BYTES"]),
        host_limit_bytes=int(os.environ["MXFP4_PINNED_BUDGET_BYTES"]),
        max_expert_rows=ep,
    )
    calls, executions, original_materializers = [], {}, []
    weight_ids = {
        id(parameter) for layer in layers for parameter in layer.experts.parameters()
    }
    original_quant = quant_module.dynamic_mxfp4_quant_blockscale_out_v1

    def weight_spy(source, *args, **kwargs):
        if id(source) in weight_ids:
            calls.append(id(source))
        return original_quant(source, *args, **kwargs)

    quant_module.dynamic_mxfp4_quant_blockscale_out_v1 = weight_spy
    for index, binding in enumerate(runtime.bindings):
        binding.cache_disabled = disabled
        for fc, module in enumerate(binding.modules, 1):
            original = module._mxfp4_materializer
            original_materializers.append((module, original))

            def observe(
                x, invocation, which, *, call=original, key=(index, fc), owner=binding
            ):
                # Probe exactly the same production cache lookup before real bind.
                hit = tuple(
                    runtime.weights.cache(owner.cache_key(item), item.parameter)
                    for item in owner.bindings(which)
                )
                result = call(x, invocation, which)
                tokens = tuple(
                    token
                    for token in runtime.rng.receipts(
                        invocation.scope,
                        route=invocation.route,
                        weight_epoch=runtime.weights.committed_weight_epoch,
                    )
                    if token.fc == which
                )
                record = {
                    "hit": tuple(value is not None for value in hit),
                    "refs": result.candidate_caches,
                    "bounds": result.bounds,
                    "receipt": tuple(
                        (
                            token.scope.record(),
                            token.fc,
                            token.domain,
                            token.seed,
                            token.base_draw,
                            token.draws_reserved,
                            token.weight_epoch,
                            token.frontier_before,
                            token.frontier_after,
                            token.route.counts,
                            tuple(
                                token.expert_wire(e)
                                for e in range(len(token.route.counts))
                            ),
                        )
                        for token in tokens
                    ),
                    "qw": tuple(
                        {role: refs[role].detach().clone() for role in _ROLES}
                        for refs in result.candidate_caches
                    ),
                }
                backward = result.backward

                def capture_backward(dy):
                    dx, emissions = backward(dy)
                    record["w"] = tuple(
                        lease.binding.main_grad.detach().clone()
                        for lease in result.zero_leases
                    )
                    record["dy"] = dy.detach().clone()
                    record["dx"] = dx.detach().clone()
                    record["status"] = result.status.detach().clone()
                    return dx, emissions

                result.backward = capture_backward
                executions[key] = record
                return result

            module._mxfp4_materializer = observe
    snapshots, prior_refs, prior_bounds = [], None, None
    try:
        for window in range(2):
            runtime.zero_boundary(1)
            executions.clear()
            probabilities = {}

            handles = [
                layer.experts.register_forward_pre_hook(
                    capture_probability(probabilities, index)
                )
                for index, layer in enumerate(layers)
            ]
            x = tuple(source.clone().requires_grad_(True) for source in inputs)
            runtime.microbatch = 0
            try:
                outputs = tuple(layer(source)[0] for layer, source in zip(layers, x))
                torch.autograd.backward(outputs, upstream)
            finally:
                runtime.microbatch = None
                for handle in handles:
                    handle.remove()
            expected_weight_calls = (window + 1) if disabled else 1
            expected_parameters = len(weight_ids)
            assert len(calls) == expected_weight_calls * expected_parameters
            assert set(calls) == weight_ids
            refs_now = {key: record["refs"] for key, record in executions.items()}
            bounds_now = {key: record["bounds"] for key, record in executions.items()}
            for key, record in executions.items():
                assert record["hit"] == (window == 1 and not disabled,) * len(
                    record["hit"]
                )
                assert torch.count_nonzero(record["status"]) == 0
                if prior_refs is not None:
                    for old, new in zip(prior_refs[key], record["refs"]):
                        for role in _ROLES:
                            if not disabled:
                                assert new[role] is old[role], (key, role)
                            else:
                                assert new[role] is not old[role], (key, role)
                    # A new logical activation binding is required even for a QW hit.
                    assert all(
                        old is not new
                        for old, new in zip(prior_bounds[key], record["bounds"])
                    )
            snapshot = {
                "inputs": inputs,
                "upstream": upstream,
                "outputs": tuple(value.detach().clone() for value in outputs),
                "dx": tuple(value.grad.detach().clone() for value in x),
                "prob": tuple(
                    probabilities[index].detach().clone() for index in range(2)
                ),
                "prob_grad": tuple(
                    probabilities[index].grad.detach().clone() for index in range(2)
                ),
                "epoch": runtime.weights.committed_weight_epoch,
                "router_grad": tuple(
                    layer.router.weight.main_grad.detach().clone() for layer in layers
                ),
                "fc": {
                    key: {
                        name: record[name]
                        for name in ("receipt", "qw", "w", "dy", "dx", "status")
                    }
                    for key, record in executions.items()
                },
            }
            snapshots.append(snapshot)
            runtime.finish_grad_sync()
            runtime.optimizer_result(False)  # legal skipped update; preserves QW epoch
            prior_refs, prior_bounds = refs_now, bounds_now
        assert snapshots[0]["epoch"] == snapshots[1]["epoch"]
        for key in snapshots[0]["fc"]:
            first = snapshots[0]["fc"][key]["receipt"]
            second = snapshots[1]["fc"][key]["receipt"]
            for a, b in zip(first, second):
                assert a[0] != b[0] and a[7] < b[7]
                assert b[4] >= a[4]
                if a[5] or b[5]:
                    assert (
                        b[4] > a[4]
                    ), "nonempty new contribution must advance SR draws"
        return snapshots
    finally:
        quant_module.dynamic_mxfp4_quant_blockscale_out_v1 = original_quant
        for module, original in original_materializers:
            module._mxfp4_materializer = original
        runtime.control.close()


def test_actual_cold_then_hot_vs_disabled_same_logical_receipts(tmp_path):
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    if not dist.is_initialized():
        pytest.skip("authorized signed world8 Core fixture required")
    tracker = get_cuda_rng_tracker()
    cpu = torch.get_rng_state().clone()
    device = tuple(value.clone() for value in torch.cuda.get_rng_state_all())
    core = copy.deepcopy(tracker.get_states())
    try:
        arms = []
        for disabled in (False, True):
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state_all(device)
            tracker.set_states(copy.deepcopy(core))
            directory = tmp_path / str(disabled)
            directory.mkdir()
            arms.append(_arm(disabled, directory))
        # Compare corresponding logical windows, not distinct contributions with
        # legitimately different SR draws. Independent arm advances both windows.
        _equal_tree(arms[0], arms[1])
    finally:
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state_all(device)
        tracker.set_states(core)
