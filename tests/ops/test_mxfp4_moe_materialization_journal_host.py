"""Direct child boundary host behavior, not Torch/GPU submission proof."""

import ast
import importlib.util
from pathlib import Path


def test_direct_child_before_return_after_return_and_unknown():
    root = Path(__file__).resolve().parents[2]
    path = root / "lumen/utils/mxfp4_moe_submission.py"
    ast.parse((root / "lumen/ops/mxfp4_moe_materialize.py").read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("direct_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    descriptors = (
        {"expert": None, "op": "status_init"},
        {"expert": 0, "op": "Z_zero"},
        {"expert": 0, "op": "G_zero"},
    )
    before = module.SubmissionJournal(descriptors)
    assert all(item["submission"] == "NOT_ENTERED" for item in before.failure())
    returned = module.SubmissionJournal(descriptors)
    assert returned.call(0, lambda: 7) == 7
    assert returned.call(1, lambda: None) is None
    # Healthy external error after public return does not erase the prefix.
    facts = returned.failure()
    assert [item["submission"] for item in facts] == [
        "PUBLIC_RETURNED",
        "PUBLIC_RETURNED",
        "NOT_ENTERED",
    ]
    current = module.SubmissionJournal(descriptors)

    def opaque():
        raise RuntimeError("unknown driver acceptance")

    try:
        current.call(0, opaque)
    except RuntimeError:
        pass
    assert current.failure()[0]["submission"] == "ENTERED"
    assert current.unknown() and current.known_prefix() == ()
