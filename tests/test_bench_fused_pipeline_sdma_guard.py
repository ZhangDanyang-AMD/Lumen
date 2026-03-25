import ast
from pathlib import Path

BENCH_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_fused_pipeline.py"


def _load_sdma_test_methods() -> list[str]:
    source = BENCH_PATH.read_text()
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "TestFusedPipelineSdmaColumn":
            return [
                child.name
                for child in node.body
                if isinstance(child, ast.FunctionDef) and child.name.startswith("test_")
            ]

    raise AssertionError("Missing TestFusedPipelineSdmaColumn benchmark class")


def test_fused_pipeline_sdma_benchmarks_share_one_test_method():
    methods = _load_sdma_test_methods()

    assert methods == ["test_nccl_vs_sdma_fused_suite"]


def test_fused_pipeline_sdma_suite_supports_phase_selector_env():
    source = BENCH_PATH.read_text()

    assert "LUMEN_FUSED_PIPELINE_SDMA_ONLY" in source
    assert "column_fwd" in source
    assert "row_fwd" in source
    assert "column_fwd_bwd" in source


def test_fused_pipeline_sdma_suite_checks_phase_consistency_across_ranks():
    source = BENCH_PATH.read_text()

    assert "ReduceOp.MIN" in source
    assert "ReduceOp.MAX" in source
    assert "phase_codes.get(selected_phase, -1)" in source
