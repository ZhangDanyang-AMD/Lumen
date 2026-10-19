"""No-runtime schedule invariants; not Core numerical execution evidence."""

import importlib.util
from pathlib import Path


def test_profile_schedule_conservation():
    directory = Path(__file__).resolve().parent
    for name in (
        "mxfp4_moe_route_profiles.py",
        "test_mxfp4_moe_controlled_profiles.py",
        "test_mxfp4_moe_paired_numeric.py",
    ):
        import ast

        ast.parse((directory / name).read_text(encoding="utf-8"))
    path = directory / "mxfp4_moe_route_profiles.py"
    spec = importlib.util.spec_from_file_location("profile_reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    for ep in (4, 8):
        for profile in ("balanced", "half_empty", "many_empty", "single_hot", "zipf"):
            for tokens in (1, 31, 33, 129, 257):
                counts = [
                    reference.expected_counts(profile, tokens, ep, receiver)
                    for receiver in range(ep)
                ]
                assert sum(map(sum, counts)) == ep * tokens * 6
                for sender in range(ep):
                    for token in range(tokens):
                        assert (
                            len(
                                set(
                                    reference.targets(
                                        profile, token, tokens, sender, ep
                                    )
                                )
                            )
                            == 6
                        )
