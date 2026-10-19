"""Actual Core clamp/router mutations, not isolated toy activation tests."""

import copy
import os

import pytest
import torch

from tests.env.mxfp4_moe_fixture import build_core_layer, signed_config
from tests.env.mxfp4_moe_paired_reference import install_reference, metric
from tests.env.test_mxfp4_moe_paired_numeric import _run_paired_cell
from tests.mxfp4_callback_observers import capture_probability, capture_router_map


def _boundary_inputs(layers, hidden, upstream):
    # Exact BF16 values stay strictly away from +/-10 and TopK ties. Each FC1
    # output stripe includes both sides; no derivative at a clamp boundary is used.
    with torch.no_grad():
        for source, incoming, layer in zip(hidden, upstream, layers):
            source.zero_()
            source[..., 0] = 1
            incoming.fill_(1)
            coefficients = torch.tensor(
                [-12.0, -2.0, 2.0, 12.0], device=source.device, dtype=torch.bfloat16
            ).repeat(512)
            fc1, fc2 = layer.experts.linear_fc1, layer.experts.linear_fc2
            for slot in range(fc1.num_gemms):
                weight = getattr(fc1, "weight" + str(slot))
                weight.zero_()
                weight[:, 0] = torch.cat((coefficients, coefficients.flip(0)))
                second = getattr(fc2, "weight" + str(slot))
                second.zero_()
                second[:, 0] = 0.5

            def check_actual_boundary(module, args, result):
                value = result[0]
                if value.numel():
                    gate, up = value.chunk(2, -1)
                    assert (gate < -10).any() and (gate > 10).any()
                    assert (up < -10).any() and (up > 10).any()
                    assert not (gate.abs() == 10).any() and not (up.abs() == 10).any()

            fc1.register_forward_hook(check_actual_boundary)
            layer.router.weight.zero_()
            layer.router.weight[:, 0] = torch.linspace(
                -4, 4, 256, device=source.device
            ).to(torch.bfloat16)


def _core_arm(snapshot, mutation):
    ep = int(os.environ["MXFP4_TEST_EP"])
    config, _ = signed_config(os.environ["MXFP4_SIGNED_CONTRACT"], ep=ep, overlap=False)
    config = copy.deepcopy(config)
    if mutation == "no_clamp":
        config.activation_func_clamp_value = None
    elif mutation == "post_softmax":
        config.moe_router_pre_softmax = False
    elif mutation == "wrong_scale":
        config.moe_router_topk_scaling_factor = 1.0
    layers = torch.nn.ModuleList([build_core_layer(config) for _ in range(2)])
    layers.load_state_dict(snapshot["initial"])
    install_reference(list(layers), snapshot["fc"], mathematical=True)
    results = []
    for index, layer in enumerate(layers):
        state = {}

        probs_hook = capture_probability(state, "probs")
        router_hook = capture_router_map(state)

        def fc1_hook(module, args, result):
            value = result[0]
            if mutation == "gate_lower_clamp":
                gate, up = value.chunk(2, -1)
                return torch.cat((gate.clamp(min=-10), up), -1), result[1]

        handles = [
            layer.experts.register_forward_pre_hook(probs_hook),
            layer.router.register_forward_hook(router_hook),
            layer.experts.linear_fc1.register_forward_hook(fc1_hook),
        ]
        if mutation == "detach_probability":
            # Test-only subscription replaces probability at the real Core expert
            # call boundary while leaving its values/router/AlltoAll unchanged.
            def detach(module, args):
                return args[0], args[1], args[2].detach()

            handles.append(layer.experts.register_forward_pre_hook(detach))
        x = snapshot["inputs"][index].clone().requires_grad_(True)
        output, _ = layer(x)
        output.backward(snapshot["upstream"][index])
        probs = state["probs"]
        router_grad = layer.router.weight.grad
        results.append(
            {
                "out": output.detach(),
                "dx": x.grad.detach(),
                "map": state["map"],
                "prob": probs.detach(),
                "prob_grad": (
                    probs.grad.detach()
                    if probs.grad is not None
                    else torch.zeros_like(probs)
                ),
                "router_grad": (
                    router_grad.detach()
                    if router_grad is not None
                    else torch.zeros_like(layer.router.weight)
                ),
            }
        )
        for handle in handles:
            handle.remove()
    return results


@pytest.mark.parametrize(
    "mutation",
    [
        "no_clamp",
        "gate_lower_clamp",
        "post_softmax",
        "wrong_scale",
        "detach_probability",
    ],
)
def test_actual_core_clamp_router_gradient_sensitivity(mutation, tmp_path):
    snapshot = _run_paired_cell(
        False, False, False, 1, tmp_path, snapshot_only=True, setup=_boundary_inputs
    )
    baseline = _core_arm(snapshot, "signed")
    wrong = _core_arm(snapshot, mutation)
    for index, (reference, altered) in enumerate(zip(baseline, wrong)):
        assert torch.equal(reference["map"], snapshot["maps"][index])
        assert torch.equal(
            reference["map"], altered["map"]
        ), "TopK identity must be held comparable"
        metric(reference["out"], snapshot["outputs"][index], 12)
        metric(reference["dx"], snapshot["dx"][index], 12)
        metric(reference["prob_grad"], snapshot["probability_grad"][index], 12)
        metric(reference["router_grad"], snapshot["router_grad"][index], 12)
        assert torch.equal(reference["prob"], snapshot["probability"][index])
        # Negative variants must change an actual gradient, not merely configuration.
        assert any(
            not torch.equal(reference[key], altered[key])
            for key in ("dx", "prob_grad", "router_grad")
        ), mutation
        if mutation == "detach_probability":
            assert torch.count_nonzero(reference["router_grad"]) > 0
            assert torch.count_nonzero(altered["router_grad"]) == 0
