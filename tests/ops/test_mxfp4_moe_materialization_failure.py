"""Source-level failure-disposition regression, not a GPU failure injection."""

import ast
from pathlib import Path


def test_materialization_postlaunch_is_not_candidate_miss():
    path = Path(__file__).resolve().parents[2] / "lumen/ops/mxfp4_moe_materialize.py"
    assert path.is_file(), path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wrapper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "materialize_fc"
    )
    text = ast.unparse(wrapper)
    assert "MATERIALIZATION_POSTLAUNCH" in text
    assert "begin_attempt" in text and "enqueue" in text
    assert "async_abort" in text and "asynchronous=True" in text
    assert "CANDIDATE_MISS" not in text
