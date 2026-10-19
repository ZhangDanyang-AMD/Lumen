"""Named profiles defined at target receiver, not inferred from sender rows."""

import os

import pytest
import torch

from tests.env.mxfp4_moe_route_profiles import exact_plan
from tests.env.test_mxfp4_moe_paired_numeric import _run_paired_cell


@pytest.mark.parametrize(
    "profile",
    ["balanced", "half_empty", "many_empty", "single_hot", "zipf", "all_empty"],
)
def test_actual_receiver_count_profile(profile, tmp_path):
    from megatron.core import parallel_state

    ep = int(os.environ["MXFP4_TEST_EP"])
    rank = parallel_state.get_expert_model_parallel_rank()
    plans = tuple(exact_plan(ep, cell, profile) for cell in (0, 1))
    rows = len(plans[0][0][rank])
    local = 256 // ep
    expected = {
        cell: counts[rank * local : (rank + 1) * local]
        for cell, (_, counts) in enumerate(plans)
    }

    def setup(layers, inputs, upstream):
        with torch.no_grad():
            for cell, (layer, source) in enumerate(zip(layers, inputs)):
                source.zero_()
                layer.router.weight.zero_()
                layer.router.weight[:, :256] = torch.eye(
                    256, device=source.device, dtype=torch.bfloat16
                )
                for token, selected in enumerate(plans[cell][0][rank]):
                    source[token, 0, list(selected)] = torch.tensor(
                        [4 + index / 4 for index in range(6)],
                        device=source.device,
                        dtype=torch.bfloat16,
                    )

    _run_paired_cell(
        False, False, False, rows, tmp_path, setup=setup, expected_route_counts=expected
    )
