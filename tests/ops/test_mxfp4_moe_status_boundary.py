"""Affected source invariants; asynchronous device injection remains unrun."""

import ast
from pathlib import Path


def test_status_operations_share_failure_disposition():
    path = Path(__file__).resolve().parents[2] / "lumen/ops/mxfp4_moe_execution.py"
    assert path.is_file(), path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RoutedFCExecution"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "status_boundary"
    )
    guarded = next(node for node in method.body if isinstance(node, ast.Try))
    body = ast.unparse(guarded)
    for operation in ("wait_event", "copy_", "record", "synchronize", "vote"):
        assert operation in ast.unparse(ast.Module(body=guarded.body, type_ignores=[]))
    assert "async_abort" in body and "obligations.fail" in body


def test_driver_clears_microbatch_on_failure():
    path = Path(__file__).resolve().parents[2] / "lumen/models/mxfp4_moe_driver.py"
    assert path.is_file(), path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RoutedDriver"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    guarded = next(node for node in method.body if isinstance(node, ast.Try))
    assert "self.microbatch = previous" in ast.unparse(
        ast.Module(body=guarded.finalbody, type_ignores=[])
    )
    assert "self._driver_call(execute)" in ast.unparse(guarded)
    driver = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_driver_failure"
    )
    assert "async_abort" in ast.unparse(driver)
