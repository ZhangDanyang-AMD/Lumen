import ast
import importlib
import os
from pathlib import Path

import pytest

BENCH_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "bench_comm_overlap.py"


def _load_function(name: str):
    source = BENCH_PATH.read_text()
    tree = ast.parse(source)
    namespace: dict[str, object] = {"importlib": importlib, "os": os}

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            exec(ast.get_source_segment(source, node), namespace)
            return namespace[name]

    raise AssertionError(f"Missing function: {name}")


def test_sdma_available_treats_non_import_errors_as_unavailable(monkeypatch: pytest.MonkeyPatch):
    sdma_available = _load_function("_sdma_available")
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(RuntimeError("mori runtime init failed")),
    )

    assert sdma_available() is False
