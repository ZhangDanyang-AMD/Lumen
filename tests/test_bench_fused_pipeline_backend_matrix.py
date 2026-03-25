import ast
from pathlib import Path

BENCH_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_fused_pipeline.py"


def _read_source() -> str:
    return BENCH_PATH.read_text(encoding="utf-8")


def _read_tree() -> ast.Module:
    return ast.parse(_read_source())


def _get_class_method(class_name: str, method_name: str) -> ast.FunctionDef:
    tree = _read_tree()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f"Missing method {class_name}.{method_name}")


def _get_function(function_name: str) -> ast.FunctionDef:
    tree = _read_tree()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"Missing function {function_name}")


def _dict_string_keys(node: ast.AST) -> set[str]:
    if not isinstance(node, ast.Dict):
        return set()
    keys: set[str] = set()
    for key in node.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.add(key.value)
    return keys


def test_sdma_suite_phase_map_covers_all_backend_matrix_phases():
    method = _get_class_method("TestFusedPipelineSdmaColumn", "test_nccl_vs_sdma_fused_suite")
    phase_dict = next(
        node for node in ast.walk(method) if isinstance(node, ast.Dict) and "column_fwd" in _dict_string_keys(node)
    )

    assert _dict_string_keys(phase_dict) == {
        "all",
        "column_fwd",
        "row_fwd",
        "column_fwd_bwd",
        "row_fwd_bwd",
    }


def test_sdma_suite_has_row_fwd_bwd_backend_benchmark_helper():
    method = _get_class_method("TestFusedPipelineSdmaColumn", "_bench_fused_row_fwd_bwd")

    assert method.name == "_bench_fused_row_fwd_bwd"


def test_backend_chunk_env_parser_exists():
    function_node = _get_function("_parse_backend_matrix_chunks")

    arg_names = [arg.arg for arg in function_node.args.args]
    assert arg_names == ["default"]


def test_backend_summary_row_builder_exists():
    function_node = _get_function("_build_backend_matrix_summary_row")

    arg_names = [arg.arg for arg in function_node.args.args]
    assert {"backend", "phase", "chunks", "avg_ms", "rccl_avg_ms"}.issubset(arg_names)


def test_benchmark_output_uses_rccl_backend_label():
    source = _read_source()

    assert "RCCL" in source
