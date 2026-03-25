import ast
from pathlib import Path

BENCH_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_wgrad_delay.py"


def _megatron_layerwise_backward_loop_body() -> list[ast.stmt]:
    tree = ast.parse(BENCH_PATH.read_text())

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_run_megatron_style_layerwise_backward":
            for stmt in node.body:
                if isinstance(stmt, ast.For):
                    return stmt.body

    raise AssertionError("Missing _run_megatron_style_layerwise_backward helper")


def _is_between_layers_wait(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Try):
        return any(_is_between_layers_wait(child) for child in stmt.body)
    if isinstance(stmt, ast.If) and ast.unparse(stmt.test) == "idx < len(modules) - 1":
        return any(
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "torch.cuda.current_stream().wait_stream"
            and len(node.args) == 1
            and ast.unparse(node.args[0]) == "wgrad_stream"
            for node in ast.walk(stmt)
        )
    return False


def _is_backward_call(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Try):
        return any(_is_backward_call(child) for child in stmt.body)
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and ast.unparse(stmt.value.func) == "torch.autograd.backward"
    )


def _load_bench_tree() -> ast.Module:
    return ast.parse(BENCH_PATH.read_text())


def _loop_execution_body(loop_body: list[ast.stmt]) -> list[ast.stmt]:
    if len(loop_body) == 1 and isinstance(loop_body[0], ast.Try):
        return loop_body[0].body
    return loop_body


def test_megatron_layerwise_backward_waits_for_wgrad_stream_between_layers():
    loop_body = _loop_execution_body(_megatron_layerwise_backward_loop_body())

    wait_idx = next((idx for idx, stmt in enumerate(loop_body) if _is_between_layers_wait(stmt)), None)
    backward_idx = next((idx for idx, stmt in enumerate(loop_body) if _is_backward_call(stmt)), None)

    assert wait_idx is not None
    assert backward_idx is not None
    assert wait_idx < backward_idx


def test_megatron_style_sdma_test_uses_rank_local_diagnostics_wrapper():
    tree = _load_bench_tree()

    helper_present = any(
        isinstance(node, ast.FunctionDef) and node.name == "_run_with_rank_local_diagnostics" for node in tree.body
    )
    validation_helper_present = any(
        isinstance(node, ast.FunctionDef) and node.name == "_validate_megatron_style_modules" for node in tree.body
    )

    test_calls_helper = False
    test_calls_validation = False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "TestMegatronStyleWgradDelay":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "test_megatron_style_pipeline_sdma":
                    test_calls_helper = any(
                        isinstance(call, ast.Call) and ast.unparse(call.func) == "_run_with_rank_local_diagnostics"
                        for call in ast.walk(child)
                    )
                    test_calls_validation = any(
                        isinstance(call, ast.Call) and ast.unparse(call.func) == "_validate_megatron_style_modules"
                        for call in ast.walk(child)
                    )

    assert helper_present
    assert validation_helper_present
    assert test_calls_helper
    assert test_calls_validation
