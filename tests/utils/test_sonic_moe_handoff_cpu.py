"""CPU-only contract tests for the MORI-to-Sonic mapped handoff."""

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_handoff_module():
    path = (
        Path(__file__).resolve().parents[2] / "lumen" / "utils" / "sonic_moe_handoff.py"
    )
    spec = importlib.util.spec_from_file_location("_test_sonic_moe_handoff", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HANDOFF = _load_handoff_module()


class CpuTensor:
    """Minimal tensor metadata used by the dependency-free handoff contract."""

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.device = "cpu"
        self._version = 0

    def mutate(self):
        self._version += 1


def _routing():
    source = CpuTensor((3, 8), "bfloat16")
    scores = CpuTensor((4,), "float32")
    gather = CpuTensor((4,), "int32")
    counts = (1, 3)
    return source, scores, counts, gather


def test_handoff_lifetime_is_one_scope():
    source, scores, counts, gather = _routing()
    with HANDOFF.mapped_handoff_scope(True):
        assert HANDOFF.publish_mapped_handoff(source, scores, counts, gather)
        value = HANDOFF.consume_mapped_handoff(source, scores, counts)
        assert value.api_version == 1
        assert value.gather_map is gather
    assert HANDOFF.consume_mapped_handoff(source, scores, counts) is None


def test_two_microbatches_use_distinct_handoffs():
    consumed = []
    for _microbatch in range(2):
        source, scores, counts, gather = _routing()
        with HANDOFF.mapped_handoff_scope(True):
            HANDOFF.publish_mapped_handoff(source, scores, counts, gather)
            consumed.append(
                HANDOFF.consume_mapped_handoff(source, scores, counts).source_tokens
            )
    assert consumed[0] is not consumed[1]


def test_failed_scope_does_not_leak_into_next_microbatch():
    source, scores, counts, gather = _routing()
    with (
        pytest.raises(RuntimeError, match="expert failed"),
        HANDOFF.mapped_handoff_scope(True),
    ):
        HANDOFF.publish_mapped_handoff(source, scores, counts, gather)
        raise RuntimeError("expert failed")
    with HANDOFF.mapped_handoff_scope(True):
        assert HANDOFF.publish_mapped_handoff(source, scores, counts, gather)
        HANDOFF.consume_mapped_handoff(source, scores, counts)


def test_same_thread_fallback_survives_contextvar_boundary():
    source, scores, counts, gather = _routing()
    with HANDOFF.mapped_handoff_scope(True):
        token = HANDOFF._ACTIVE_SCOPE.set(None)
        try:
            assert HANDOFF.mapped_handoff_enabled()
            HANDOFF.publish_mapped_handoff(source, scores, counts, gather)
            value = HANDOFF.consume_mapped_handoff(source, scores, counts)
            assert value.source_tokens is source
        finally:
            HANDOFF._ACTIVE_SCOPE.reset(token)


@pytest.mark.parametrize("mutated", ["source", "scores", "gather"])
def test_mutation_is_rejected_before_consume(mutated):
    source, scores, counts, gather = _routing()
    with (
        pytest.raises(HANDOFF.MappedHandoffError, match="MUTATED"),
        HANDOFF.mapped_handoff_scope(True),
    ):
        HANDOFF.publish_mapped_handoff(source, scores, counts, gather)
        {"source": source, "scores": scores, "gather": gather}[mutated].mutate()
        HANDOFF.consume_mapped_handoff(source, scores, counts)


def test_double_consume_is_rejected():
    source, scores, counts, gather = _routing()
    with HANDOFF.mapped_handoff_scope(True):
        HANDOFF.publish_mapped_handoff(source, scores, counts, gather)
        HANDOFF.consume_mapped_handoff(source, scores, counts)
        with pytest.raises(
            HANDOFF.MappedHandoffError, match="HANDOFF_ALREADY_CONSUMED"
        ):
            HANDOFF.consume_mapped_handoff(source, scores, counts)


def test_unconsumed_handoff_is_rejected_at_scope_exit():
    source, scores, counts, gather = _routing()
    with (
        pytest.raises(HANDOFF.MappedHandoffError, match="UNCONSUMED_HANDOFF"),
        HANDOFF.mapped_handoff_scope(True),
    ):
        HANDOFF.publish_mapped_handoff(source, scores, counts, gather)


def test_disabled_scope_is_the_fallback_contract():
    source, scores, counts, gather = _routing()
    with HANDOFF.mapped_handoff_scope(False):
        assert not HANDOFF.publish_mapped_handoff(source, scores, counts, gather)
        assert HANDOFF.consume_mapped_handoff(source, scores, counts) is None


def test_count_mutation_cannot_change_published_contract():
    source, scores, _, gather = _routing()
    counts = [1, 3]
    with (
        pytest.raises(HANDOFF.MappedHandoffError, match="ROUTING_COUNTS_MISMATCH"),
        HANDOFF.mapped_handoff_scope(True),
    ):
        HANDOFF.publish_mapped_handoff(source, scores, counts, gather)
        counts[0] = 2
        HANDOFF.consume_mapped_handoff(source, scores, counts)
