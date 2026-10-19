"""Pure plan invariants only; no real router/runtime evidence."""

import ast
import importlib.util
from pathlib import Path


def test_exact_plan_conservation_and_sentinel():
    directory = Path(__file__).resolve().parent
    for name in (
        "mxfp4_moe_route_profiles.py",
        "test_mxfp4_moe_exact_expert_m.py",
        "test_mxfp4_moe_paired_numeric.py",
    ):
        ast.parse((directory / name).read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location(
        "exact_plan", directory / "mxfp4_moe_route_profiles.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for ep in (4, 8):
        vectors = []
        for cell in (0, 1):
            senders, counts = module.exact_plan(ep, cell)
            assert len(set(map(len, senders))) == 1
            assert sum(counts) == sum(map(len, senders)) * 6
            receiver = counts[: 256 // ep]
            assert set(module.EXACT_M) <= set(receiver)
            assert any(
                receiver[index : index + 3] == (31, 1, 33)
                for index in range(len(receiver) - 2)
            )
            for sender in senders:
                for token in sender:
                    assert len(token) == len(set(token)) == 6
                    assert sum(expert < 256 // ep for expert in token) <= 1
            observed = [0] * 256
            for sender in senders:
                for token in sender:
                    for expert in token:
                        observed[expert] += 1
            assert tuple(observed) == counts
            vectors.append(receiver)
        assert vectors[0] != vectors[1]
