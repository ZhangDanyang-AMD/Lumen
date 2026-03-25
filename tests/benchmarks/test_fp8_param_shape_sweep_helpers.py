import ast
from pathlib import Path

BENCH_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "bench_fp8_param_allgather.py"


def _module_source() -> str:
    return BENCH_PATH.read_text()


def _load_functions(*names: str):
    source = _module_source()
    tree = ast.parse(source)
    namespace: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(ast.get_source_segment(source, node), namespace)
    missing = [name for name in names if name not in namespace]
    assert not missing, f"Missing functions: {missing}"
    return namespace


def test_classify_fp8_tail_note_thresholds():
    funcs = _load_functions("classify_fp8_tail_note")
    classify_fp8_tail_note = funcs["classify_fp8_tail_note"]

    assert classify_fp8_tail_note(avg_ms=10.0, p95_ms=10.9) == "tail-stable"
    assert classify_fp8_tail_note(avg_ms=10.0, p95_ms=11.0) == "tail-heavy"


def test_classify_fp8_tail_note_stable_branch_below_ratio_threshold():
    funcs = _load_functions("classify_fp8_tail_note")
    classify_fp8_tail_note = funcs["classify_fp8_tail_note"]

    # ratio = 17.59 / 16.0 = 1.099375 < 1.10
    assert classify_fp8_tail_note(avg_ms=16.0, p95_ms=17.59) == "tail-stable"


def test_build_fp8_single_shape_row_forwards_all_fields():
    funcs = _load_functions("build_fp8_single_shape_row")
    build_fp8_single_shape_row = funcs["build_fp8_single_shape_row"]

    kwargs = {
        "profile_name": "default",
        "tokens": 4096,
        "ffn": 14336,
        "bf16_avg_ms": 5.0,
        "bf16_p95_ms": 5.5,
        "bf16_max_ms": 5.8,
        "fp8_avg_ms": 4.0,
        "fp8_p95_ms": 4.4,
        "fp8_max_ms": 4.8,
        "fp8_speedup": 1.25,
        "p95_bf16_over_fp8": 1.25,
    }
    row = build_fp8_single_shape_row(**kwargs)
    assert row == kwargs


def test_build_fp8_single_shape_row_docstring_mentions_p95_cross_path_meaning():
    funcs = _load_functions("build_fp8_single_shape_row")
    build_fp8_single_shape_row = funcs["build_fp8_single_shape_row"]
    doc = (build_fp8_single_shape_row.__doc__ or "").lower()
    assert "bf16" in doc and "fp8" in doc and "p95" in doc


def test_build_fp8_pipeline_shape_row_docstring_mentions_p95_bf16_over_fp8_pipe():
    funcs = _load_functions("build_fp8_pipeline_shape_row")
    build_fp8_pipeline_shape_row = funcs["build_fp8_pipeline_shape_row"]
    doc = (build_fp8_pipeline_shape_row.__doc__ or "").lower()
    assert "bf16" in doc and "fp8" in doc and "p95" in doc and "pipe" in doc


def test_bench_source_uses_p95_bf16_over_fp8_row_key():
    assert "p95_bf16_over_fp8" in _module_source()


def test_build_fp8_pipeline_shape_row_forwards_all_inputs_and_tail_heavy_note():
    funcs = _load_functions("classify_fp8_tail_note", "build_fp8_pipeline_shape_row")
    build_fp8_pipeline_shape_row = funcs["build_fp8_pipeline_shape_row"]

    kwargs = {
        "profile_name": "pipeline_gain",
        "tokens": 8192,
        "ffn": 28672,
        "bf16_avg_ms": 20.0,
        "bf16_p95_ms": 21.0,
        "fp8_seq_avg_ms": 19.0,
        "fp8_seq_p95_ms": 20.9,
        "fp8_pipe_avg_ms": 16.0,
        "fp8_pipe_p95_ms": 18.0,
        "fp8_pipe_max_ms": 18.7,
        "pipeline_speedup_vs_bf16": 1.25,
        "pipeline_speedup_vs_fp8_seq": 1.19,
        "p95_bf16_over_fp8_pipe": 1.17,
    }
    row = build_fp8_pipeline_shape_row(**kwargs)
    assert row["note"] == "tail-heavy"
    for key, value in kwargs.items():
        assert row[key] == value


def test_build_fp8_pipeline_shape_row_stable_note_matches_pipe_p95_over_avg():
    funcs = _load_functions("classify_fp8_tail_note", "build_fp8_pipeline_shape_row")
    build_fp8_pipeline_shape_row = funcs["build_fp8_pipeline_shape_row"]

    kwargs = {
        "profile_name": "default",
        "tokens": 2048,
        "ffn": 8192,
        "bf16_avg_ms": 30.0,
        "bf16_p95_ms": 45.0,
        "fp8_seq_avg_ms": 28.0,
        "fp8_seq_p95_ms": 40.0,
        "fp8_pipe_avg_ms": 20.0,
        "fp8_pipe_p95_ms": 21.0,
        "fp8_pipe_max_ms": 22.0,
        "pipeline_speedup_vs_bf16": 1.1,
        "pipeline_speedup_vs_fp8_seq": 1.05,
        "p95_bf16_over_fp8_pipe": 2.0,
    }
    row = build_fp8_pipeline_shape_row(**kwargs)
    assert row["note"] == "tail-stable"
    for key, value in kwargs.items():
        assert row[key] == value


def test_pipeline_row_note_uses_fp8_pipe_p95_over_avg_not_p95_bf16_over_fp8_pipe():
    """`note` classifies FP8 pipelined latency tail (p95/avg), not p95_bf16_over_fp8_pipe."""
    funcs = _load_functions("classify_fp8_tail_note", "build_fp8_pipeline_shape_row")
    build_fp8_pipeline_shape_row = funcs["build_fp8_pipeline_shape_row"]

    base = {
        "profile_name": "p",
        "tokens": 1,
        "ffn": 2,
        "bf16_avg_ms": 100.0,
        "bf16_p95_ms": 200.0,
        "fp8_seq_avg_ms": 50.0,
        "fp8_seq_p95_ms": 60.0,
        "fp8_pipe_avg_ms": 10.0,
        "fp8_pipe_p95_ms": 10.9,
        "fp8_pipe_max_ms": 11.0,
        "pipeline_speedup_vs_bf16": 1.0,
        "pipeline_speedup_vs_fp8_seq": 1.0,
        "p95_bf16_over_fp8_pipe": 50.0,
    }
    row_a = build_fp8_pipeline_shape_row(**base)
    assert row_a["note"] == "tail-stable"

    row_b = build_fp8_pipeline_shape_row(**{**base, "p95_bf16_over_fp8_pipe": 0.01})
    assert row_b["note"] == row_a["note"]

    row_c = build_fp8_pipeline_shape_row(**{**base, "fp8_pipe_p95_ms": 11.0, "p95_bf16_over_fp8_pipe": 50.0})
    assert row_c["note"] == "tail-heavy"


def test_bench_fp8_has_shape_sweep_class():
    tree = ast.parse(_module_source())

    assert any(isinstance(node, ast.ClassDef) and node.name == "TestFP8ParamShapeSweep" for node in tree.body)


def test_bench_fp8_source_mentions_shape_sweep_selector():
    assert "-k ShapeSweep" in _module_source()


def test_shape_sweep_uses_fixed_profile_iterator_directly():
    source = _module_source()

    assert "for profile in get_e2e_fusion_shape_sweep()" in source


def _test_pipelined_shape_sweep_function_source() -> str:
    """Source of ``TestFP8ParamShapeSweep.test_pipelined_shape_sweep`` (sweep path only)."""
    source = _module_source()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "TestFP8ParamShapeSweep":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "test_pipelined_shape_sweep":
                    seg = ast.get_source_segment(source, item)
                    assert seg is not None, "ast.get_source_segment failed for test_pipelined_shape_sweep"
                    return seg
    raise AssertionError("TestFP8ParamShapeSweep.test_pipelined_shape_sweep not found")


def test_pipelined_shape_sweep_summary_keeps_three_way_story():
    """Pipelined shape sweep must keep BF16 vs FP8-seq vs FP8-pipe metrics (not FP8-only)."""
    sweep_src = _test_pipelined_shape_sweep_function_source()
    assert "pipeline_speedup_vs_bf16" in sweep_src
    assert "pipeline_speedup_vs_fp8_seq" in sweep_src
    assert "p95_bf16_over_fp8_pipe" in sweep_src


def test_readme_mentions_fp8_shape_sweep():
    source = BENCH_PATH.with_name("README.md").read_text()

    assert (
        "torchrun --nproc_per_node=8 -m pytest " "benchmarks/bench_fp8_param_allgather.py -v -s -k ShapeSweep"
    ) in source
