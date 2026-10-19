"""Exact receiver M and adjacent segment sentinel through real Core routing."""

import os

import torch

from tests.env.mxfp4_moe_resource_observer import ResourceObserver
from tests.env.mxfp4_moe_route_profiles import exact_plan
from tests.env.test_mxfp4_moe_paired_numeric import _run_paired_cell


def test_actual_exact_expert_m_and_segment_sentinel(tmp_path):
    from megatron.core import parallel_state

    ep = int(os.environ["MXFP4_TEST_EP"])
    rank = parallel_state.get_expert_model_parallel_rank()
    plans = tuple(exact_plan(ep, cell) for cell in (0, 1))
    rows = len(plans[0][0][rank])
    assert rows == len(plans[1][0][rank])
    local = 256 // ep
    expected = {
        cell: counts[rank * local : (rank + 1) * local]
        for cell, (_, counts) in enumerate(plans)
    }

    def setup(layers, inputs, upstream):
        with torch.no_grad():
            for cell, (layer, source) in enumerate(zip(layers, inputs)):
                source.zero_()
                router = layer.router.weight
                router.zero_()
                # Six real expert coordinates per token, diagonal router scores.
                router[:, :256] = torch.eye(
                    256, device=source.device, dtype=torch.bfloat16
                )
                for token, selected in enumerate(plans[cell][0][rank]):
                    for order, expert in enumerate(selected):
                        source[token, 0, expert] = 4 + order / 4

                def map_check(module, args, result, *, targets=plans[cell][0][rank]):
                    actual = result[1].reshape(rows, 256)
                    wanted = torch.zeros_like(actual, dtype=torch.bool)
                    for token, selected in enumerate(targets):
                        wanted[token, list(selected)] = True
                    assert torch.equal(
                        actual.bool(), wanted
                    ), "actual TopK map mismatch"

                layer.router.register_forward_hook(map_check)

    _run_paired_cell(
        False,
        False,
        False,
        rows,
        tmp_path,
        setup=setup,
        expected_route_counts=expected,
        resource_observer=ResourceObserver(),
    )


def test_all_empty_resources_and_original_core_hooks(tmp_path):
    local = 256 // int(os.environ["MXFP4_TEST_EP"])
    _run_paired_cell(
        False,
        False,
        False,
        0,
        tmp_path,
        expected_route_counts=(0,) * local,
        resource_observer=ResourceObserver(),
    )
