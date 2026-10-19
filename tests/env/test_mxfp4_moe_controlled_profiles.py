"""Actual trainable Core router executes controlled distinct-top6 profiles."""

import os

import pytest
import torch

from tests.env.mxfp4_moe_route_profiles import expected_counts, targets
from tests.env.test_mxfp4_moe_paired_numeric import _run_paired_cell


@pytest.mark.parametrize(
    "profile", ["balanced", "half_empty", "many_empty", "single_hot", "zipf"]
)
@pytest.mark.parametrize("tokens", [1, 31, 33, 129, 257])
def test_actual_adapter_controlled_route_profile(profile, tokens, tmp_path):
    from megatron.core import parallel_state

    ep = int(os.environ["MXFP4_TEST_EP"])
    sender = parallel_state.get_expert_model_parallel_rank()
    expected = expected_counts(profile, tokens, ep, sender)

    def setup(layers, inputs, upstream):
        with torch.no_grad():
            for layer, source in zip(layers, inputs):
                source.zero_()
                # One token basis coordinate per actual EP sender. Router stays a
                # real trainable BF16 matrix, not a routing-map override or stub.
                columns = torch.arange(tokens, device=source.device) + sender * tokens
                source[torch.arange(tokens, device=source.device), 0, columns] = 1
                router = layer.router.weight
                router.fill_(-4)
                for rank in range(ep):
                    for token in range(tokens):
                        selected = targets(profile, token, tokens, rank, ep)
                        for order, expert in enumerate(selected):
                            router[expert, rank * tokens + token] = 4 + order / 4

    _run_paired_cell(
        False,
        False,
        False,
        tokens,
        tmp_path,
        setup=setup,
        expected_route_counts=expected,
    )
