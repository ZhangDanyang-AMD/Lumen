###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

import ast
from pathlib import Path

BENCH_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_fp8_param_allgather.py"


def _get_function_node(path: Path, function_name: str) -> ast.FunctionDef:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node

    raise AssertionError(f"Missing function: {function_name}")


def _has_wait_stream_call(function_node: ast.FunctionDef) -> bool:
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "wait_stream":
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "comm_stream":
            continue
        if len(node.args) != 1:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Name) and arg.id == "compute_stream":
            return True
    return False


def test_fp8_chunked_linear_pipelined_orders_comm_stream_after_default_stream():
    function_node = _get_function_node(BENCH_PATH, "_fp8_chunked_linear_pipelined")
    assert _has_wait_stream_call(function_node)


def test_fp8_multi_weight_chunked_pipelined_orders_comm_stream_after_default_stream():
    function_node = _get_function_node(BENCH_PATH, "_fp8_multi_weight_chunked_pipelined")
    assert _has_wait_stream_call(function_node)
