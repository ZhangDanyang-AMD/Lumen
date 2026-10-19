"""Host submission facts only, not GPU completion or complete ledger repair."""

import importlib.util
from pathlib import Path


def test_known_prefix_unknown_and_failed_reuse():
    path = Path(__file__).resolve().parents[2] / "lumen/utils/mxfp4_moe_submission.py"
    spec = importlib.util.spec_from_file_location("submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    journal = module.SubmissionJournal(({"expert": 0}, {"expert": 1}, {"expert": 2}))
    journal.enter(0)
    journal.returned(0)
    journal.enter(1)
    facts = journal.failure()
    assert [fact["submission"] for fact in facts] == [
        "PUBLIC_RETURNED",
        "ENTERED",
        "NOT_ENTERED",
    ]
    assert journal.unknown()
    assert journal.known_prefix() == ({"expert": 0},)
    try:
        journal.enter(2)
    except RuntimeError:
        pass
    else:
        raise AssertionError("failed journal reused")
