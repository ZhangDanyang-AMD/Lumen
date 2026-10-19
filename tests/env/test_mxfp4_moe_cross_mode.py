"""Independent real adapter instances, identical logical receipts across modes."""

import copy

import pytest
import torch

from tests.env.test_mxfp4_moe_paired_numeric import _run_paired_cell


def _equal_tree(left, right, path="result"):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert torch.equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            _equal_tree(left[key], right[key], path + "/" + str(key))
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            _equal_tree(a, b, path + "/" + str(index))
    else:
        assert left == right, path


@pytest.mark.parametrize("sender_tokens", [0, 1, 31, 33, 129, 257])
@pytest.mark.parametrize("clamp_active", [False, True])
def test_real_adapter_same_receipt_cache_recompute(
    sender_tokens, clamp_active, tmp_path
):
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    # Restore framework RNG, including Core's named model/expert RNG streams,
    # before every fresh constructor; never rewind a live adapter's SR allocator.
    tracker = get_cuda_rng_tracker()
    cpu_rng = torch.get_rng_state().clone()
    device_rng = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    core_rng = copy.deepcopy(tracker.get_states())
    baseline = None
    try:
        for cache_disabled, checkpoint in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(device_rng)
            tracker.set_states(copy.deepcopy(core_rng))
            mode = str(int(cache_disabled)) + str(int(checkpoint))
            directory = tmp_path / mode
            directory.mkdir()
            result = _run_paired_cell(
                checkpoint,
                cache_disabled,
                clamp_active,
                sender_tokens,
                directory,
                snapshot_only=True,
            )
            # FC snapshots retain tokens only for intra-arm reference; compare
            # their pointer-independent complete semantic records across instances.
            for record in result["fc"].values():
                del record["tokens"]
            if baseline is None:
                baseline = result
            else:
                _equal_tree(baseline, result)
            del result
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(device_rng)
        tracker.set_states(core_rng)
