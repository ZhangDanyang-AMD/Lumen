"""Real world8 Core adapter versus independent recipe and BF16 Core arms; unrun."""

import json
import os

import pytest
import torch
import torch.distributed as dist

from lumen.models.mxfp4_moe_startup import RoutedCoreRuntime
from tests.env.mxfp4_moe_fixture import build_core_layer, signed_config
from tests.env.mxfp4_moe_paired_reference import _RECIPE, install_reference, metric
from tests.env.mxfp4_moe_pytest import fixture_options


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("cache_disabled", [False, True])
@pytest.mark.parametrize("clamp_active", [False, True])
@pytest.mark.parametrize(
    "sender_tokens", [0, 1, 15, 16, 31, 32, 33, 63, 64, 127, 128, 129, 257]
)
def test_real_core_adapter_paired_recipe(
    checkpoint, cache_disabled, clamp_active, sender_tokens, tmp_path
):
    _run_paired_cell(checkpoint, cache_disabled, clamp_active, sender_tokens, tmp_path)


def _run_paired_cell(
    checkpoint,
    cache_disabled,
    clamp_active,
    sender_tokens,
    tmp_path,
    *,
    snapshot_only=False,
    setup=None,
    expected_route_counts=None,
    resource_observer=None
):
    if not dist.is_initialized():
        pytest.skip("authorized signed world8 fixture required")
    ep = int(os.environ["MXFP4_TEST_EP"])
    config, ddp_config = signed_config(
        os.environ["MXFP4_SIGNED_CONTRACT"], ep=ep, overlap=False
    )
    assert config.activation_func_clamp_value == 10.0
    assert config.moe_router_pre_softmax is True
    assert config.moe_router_topk_scaling_factor == 1.5
    torch.manual_seed(1729)
    candidate = torch.nn.ModuleList([build_core_layer(config) for _ in range(2)])
    if clamp_active:
        with torch.no_grad():
            for layer in candidate:
                for parameter in layer.experts.linear_fc1.parameters():
                    parameter.mul_(32)
    initial = {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in candidate.state_dict().items()
    }
    # Explicit independent BF16 inputs: two strict layer cells, no propagation
    # of either candidate output into the next cell's router.
    hidden = tuple(
        torch.randn((sender_tokens, 1, 4096), device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    )
    incoming = tuple(torch.randn_like(source) for source in hidden)
    if setup is not None:
        setup(candidate, hidden, incoming)
        initial = {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in candidate.state_dict().items()
        }
    captures, probability = {}, {}
    runtime = RoutedCoreRuntime(
        candidate,
        list(candidate),
        options=fixture_options(),
        config=config,
        ddp_config=ddp_config,
        run_seed=1729,
        manifest_digest=os.environ["MXFP4_RUN_MANIFEST_DIGEST"],
        physical_run_id=os.environ["MXFP4_PHYSICAL_RUN_ID"]
        + "/paired/"
        + str(checkpoint)
        + str(cache_disabled)
        + str(clamp_active)
        + str(sender_tokens),
        journal_path=tmp_path / ("paired-" + str(dist.get_rank()) + ".jsonl"),
        sender_rows=(sender_tokens,) * ep,
        framework_resident_tensors=tuple(layer.router.weight for layer in candidate),
        framework_future_bytes=int(os.environ["MXFP4_FRAMEWORK_FUTURE_BYTES"]),
        host_limit_bytes=int(os.environ["MXFP4_PINNED_BUDGET_BYTES"]),
        max_expert_rows=ep * sender_tokens,
    )
    if resource_observer is not None:
        resource_observer("prepared", runtime)
    forward_observers = []
    materializer_observers = []
    for index, binding in enumerate(runtime.bindings):
        binding.cache_disabled = cache_disabled
        for fc, module in enumerate(binding.modules, 1):
            original = module._mxfp4_materializer

            def capture(x, invocation, which, *, call=original, key=(index, fc)):
                if expected_route_counts is not None:
                    expected = (
                        expected_route_counts[key[0]]
                        if isinstance(expected_route_counts, dict)
                        else expected_route_counts
                    )
                    assert (
                        invocation.route.counts == expected
                    ), "actual Core route profile mismatch"
                result = call(x, invocation, which)
                if resource_observer is not None:
                    resource_observer("materialized", runtime, result)
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
                    "counts": invocation.route.counts,
                    "tokens": tokens,
                    "input": x.detach().clone(),
                    "qw": tuple(
                        {
                            name: values[name].detach().clone()
                            for name in ("w_fp4", "w_scales", "wt_fp4", "wt_scales")
                        }
                        for values in result.candidate_caches
                    ),
                }
                captures[key] = record
                forward = result.forward

                def observe_forward(replay=False):
                    value = forward(replay)
                    observed = value.detach().clone()
                    if "forward" in record:
                        assert torch.equal(
                            record["forward"], observed
                        ), "recompute F changed"
                    record["forward"] = observed
                    return value

                forward_observers.append((result, forward))
                result.forward = observe_forward
                backward = result.backward

                def observe_backward(dy):
                    dx, emissions = backward(dy)
                    if resource_observer is not None:
                        resource_observer("backward", runtime, result, emissions)
                    record["dy"] = dy.detach().clone()
                    record["dx"] = dx.detach().clone()
                    record["weights"] = tuple(
                        lease.binding.main_grad.detach().clone()
                        for lease in result.zero_leases
                    )
                    return dx, emissions

                result.backward = observe_backward
                return result

            materializer_observers.append((module, original))
            module._mxfp4_materializer = capture

    def probability_hook(index, destination):
        def hook(module, args):
            probs = args[2]
            if torch.is_grad_enabled():
                destination[index] = probs
                if probs.requires_grad:
                    probs.retain_grad()

        return hook

    layer_edges, routing_maps = {}, {}

    def router_hook(index, expected=None):
        def hook(module, args, result):
            mapping = result[1].detach()
            if expected is None:
                if index in routing_maps:
                    assert torch.equal(
                        mapping, routing_maps[index]
                    ), "recompute route changed"
                routing_maps[index] = mapping.clone()
            else:
                assert torch.equal(
                    mapping, expected[index]
                ), "paired routing identities changed"

        return hook

    def layer_hook(index):
        def hook(module, args, result):
            if torch.is_grad_enabled():
                source, value = args[0], result[0]
                source.retain_grad()
                value.retain_grad()
                layer_edges[index] = (source, value)

        return hook

    handles = [
        layer.experts.register_forward_pre_hook(probability_hook(index, probability))
        for index, layer in enumerate(candidate)
    ]
    handles += [
        layer.register_forward_hook(layer_hook(index))
        for index, layer in enumerate(candidate)
    ]
    handles += [
        layer.router.register_forward_hook(router_hook(index))
        for index, layer in enumerate(candidate)
    ]
    runtime.zero_boundary(1)
    x = tuple(source.detach().clone().requires_grad_(True) for source in hidden)

    def independent_cells(*sources):
        return tuple(layer(source)[0] for layer, source in zip(candidate, sources))

    runtime.microbatch = 0
    try:
        if checkpoint:
            from megatron.core.tensor_parallel.random import (
                checkpoint as core_checkpoint,
            )

            from lumen.models.mxfp4_moe import CheckpointReceiptTape

            tape = CheckpointReceiptTape()
            outputs = core_checkpoint(tape.wrap(independent_cells), False, *x)
        else:
            outputs = independent_cells(*x)
        torch.autograd.backward(outputs, incoming)
        if resource_observer is not None:
            resource_observer("consumed", runtime)
    finally:
        runtime.microbatch = None
        for execution, original_forward in forward_observers:
            execution.forward = original_forward
        for module, original_materializer in materializer_observers:
            module._mxfp4_materializer = original_materializer
        for handle in handles:
            handle.remove()
    if sender_tokens == 0:
        assert len(captures) == 4
        for record in captures.values():
            assert not any(record["counts"])
            assert record["forward"].numel() == record["dx"].numel() == 0
            assert all(token.draws_reserved == 0 for token in record["tokens"])
            assert all(torch.count_nonzero(weight) == 0 for weight in record["weights"])
        assert all(source.grad is not None and source.grad.numel() == 0 for source in x)
    actual_weights = {
        (index, fc, expert): value
        for (index, fc), record in captures.items()
        for expert, value in enumerate(record["weights"])
    }
    actual_router = [
        layer.router.weight.main_grad.detach().clone() for layer in candidate
    ]
    actual_probs = [
        (
            probability[index].grad.detach().clone()
            if probability[index].grad is not None
            else torch.zeros_like(probability[index])
        )
        for index in range(2)
    ]
    edge_snapshots = {
        index: tuple(
            tensor.detach().clone()
            for tensor in (source, value, source.grad, value.grad)
        )
        for index, (source, value) in layer_edges.items()
    }
    snapshot = {
        "initial": initial,
        "inputs": tuple(source.detach().clone() for source in hidden),
        "upstream": tuple(value.detach().clone() for value in incoming),
        "outputs": tuple(value.detach().clone() for value in outputs),
        "dx": tuple(source.grad.detach().clone() for source in x),
        "router_grad": tuple(actual_router),
        "probability": tuple(probability[index].detach().clone() for index in range(2)),
        "probability_grad": tuple(actual_probs),
        "maps": routing_maps,
        "fc": captures,
        "scopes": tuple(scope.record() for scope in runtime.scopes.values()),
        "epoch": runtime.weights.committed_weight_epoch,
        "receipt_records": {
            key: tuple(
                (
                    token.scope.record(),
                    token.fc,
                    token.domain,
                    token.seed,
                    token.global_rank,
                    token.base_draw,
                    token.draws_reserved,
                    token.weight_epoch,
                    token.frontier_before,
                    token.frontier_after,
                    token.route.counts,
                    token.route.sender_rows,
                    token.route.sorted_rows,
                    token.route.routing_epoch,
                    token.schema,
                    token.allocation_mapping,
                    token.public_mapping,
                    tuple(
                        token.expert_wire(expert)
                        for expert in range(len(token.route.counts))
                    ),
                )
                for token in record["tokens"]
            )
            for key, record in captures.items()
        },
    }
    for handle in handles:
        handle.remove()
    runtime.finish_grad_sync()
    if resource_observer is not None:
        resource_observer("retired", runtime)
    runtime.optimizer_result(False)
    runtime.control.close()
    if snapshot_only:
        runtime.handoff.remove()
        for shell in runtime.shells:
            shell.experts.forward = shell.original_forward
        return snapshot
    reports = {}
    for (index, fc), record in captures.items():
        start = 0
        for expert, count in enumerate(record["counts"]):
            weight = getattr(
                (
                    candidate[index].experts.linear_fc1
                    if fc == 1
                    else candidate[index].experts.linear_fc2
                ),
                "weight" + str(expert),
            )
            source = record["input"][start : start + count]
            dy = record["dy"][start : start + count]
            actual_f = record["forward"][start : start + count]
            assert actual_f.shape == (count, weight.shape[0])
            assert source.shape == (count, weight.shape[1])
            if count:
                ref_f = (
                    _RECIPE.quantize_reference(source)
                    @ _RECIPE.quantize_reference(weight, 32).T
                )
                reports["isolated/F-recipe/" + str((index, fc, expert))] = metric(
                    ref_f, actual_f, 30, elements=True
                )
                reports["isolated/F-bf16/" + str((index, fc, expert))] = metric(
                    source.float() @ weight.float().T, actual_f, 12
                )
                row, transformed = record["tokens"]
                _, ref_dx, ref_dw = _RECIPE.quantized_recipe(
                    source,
                    weight,
                    dy,
                    row_seed=row.seed,
                    row_base=row.base_draw + row.route.prefixes(1)[expert],
                    h16_seed=transformed.seed,
                    h16_base=transformed.base_draw
                    + transformed.route.prefixes(2)[expert],
                )
                reports["isolated/D/" + str((index, fc, expert))] = metric(
                    ref_dx, record["dx"][start : start + count], 30, elements=True
                )
                reports["isolated/W/" + str((index, fc, expert))] = metric(
                    ref_dw, record["weights"][expert], 30, elements=True
                )
                reports["isolated/D-bf16/" + str((index, fc, expert))] = metric(
                    dy.float() @ weight.float(), record["dx"][start : start + count], 12
                )
                reports["isolated/W-bf16/" + str((index, fc, expert))] = metric(
                    dy.float().T @ source.float(), record["weights"][expert], 10
                )
            else:
                assert torch.count_nonzero(record["weights"][expert]) == 0
            start += count
    for mathematical in (False, True):
        reference = torch.nn.ModuleList([build_core_layer(config) for _ in range(2)])
        reference.load_state_dict(initial)
        install_reference(list(reference), captures, mathematical=mathematical)
        reference_probability = {}
        hooks = [
            layer.experts.register_forward_pre_hook(
                probability_hook(index, reference_probability)
            )
            for index, layer in enumerate(reference)
        ]
        hooks += [
            layer.router.register_forward_hook(router_hook(index, routing_maps))
            for index, layer in enumerate(reference)
        ]
        prefix = "bf16" if mathematical else "recipe"
        # Each strict cell uses the same independently sampled BF16 source and
        # explicit upstream gradient; candidate outputs are never reference inputs.
        for index, layer in enumerate(reference):
            _, candidate_value, candidate_dx, _ = edge_snapshots[index]
            rx = hidden[index].clone().requires_grad_(True)
            upstream = incoming[index]
            rout, _ = layer(rx)
            rout.backward(upstream)
            reports[prefix + "/out/" + str(index)] = metric(rout, candidate_value, 12)
            reports[prefix + "/dx/" + str(index)] = metric(rx.grad, candidate_dx, 12)
            reports[prefix + "/router/" + str(index)] = metric(
                layer.router.weight.grad, actual_router[index], 12
            )
            ref_probs = reference_probability[index]
            assert torch.equal(
                ref_probs.detach(), probability[index].detach()
            ), "paired probability values changed"
            ref_prob_grad = (
                ref_probs.grad
                if ref_probs.grad is not None
                else torch.zeros_like(ref_probs)
            )
            reports[prefix + "/probability/" + str(index)] = metric(
                ref_prob_grad, actual_probs[index], 12
            )
            for fc, module in enumerate(
                (layer.experts.linear_fc1, layer.experts.linear_fc2), 1
            ):
                for expert in range(module.num_gemms):
                    weight = getattr(module, "weight" + str(expert))
                    reports[prefix + "/W/" + str((index, fc, expert))] = metric(
                        weight._paired_fp32_grad,
                        actual_weights[index, fc, expert],
                        10,
                        elements=not mathematical,
                    )
        for hook in hooks:
            hook.remove()
        del reference
    (tmp_path / ("paired-metrics-" + str(dist.get_rank()) + ".json")).write_text(
        json.dumps(reports, indent=2)
    )
