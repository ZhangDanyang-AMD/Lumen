"""Actual custom Function parameter edge checks without a fake framework."""

import ast
from pathlib import Path


def test_all_weights_are_explicit_autograd_inputs():
    path = Path(__file__).resolve().parents[2] / "lumen/ops/mxfp4_moe.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "routed_linear"
    )
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    apply = next(
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "apply"
    )
    assert any(isinstance(arg, ast.Starred) for arg in apply.args)
    assert "mark_non_differentiable" not in path.read_text(encoding="utf-8")
