import ast
from pathlib import Path

BENCH_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "bench_comm_overlap.py"


def _module_source() -> str:
    return BENCH_PATH.read_text()


def _module_ast() -> ast.Module:
    return ast.parse(_module_source())


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


def test_classify_overlap_note_thresholds():
    funcs = _load_functions("classify_overlap_note")
    classify_overlap_note = funcs["classify_overlap_note"]

    assert classify_overlap_note(1.03) == "latency win"
    assert classify_overlap_note(0.969) == "negative optimization"
    assert classify_overlap_note(1.00) == "neutral"


def test_shape_sweep_comm_mb_column_uses_hidden_width():
    funcs = _load_functions("shape_sweep_comm_mb")
    shape_sweep_comm_mb = funcs["shape_sweep_comm_mb"]

    assert shape_sweep_comm_mb(tokens=4096, width=4096) == (4096 * 4096 * 2) / 1e6


def test_shape_sweep_comm_mb_row_uses_ffn_width():
    funcs = _load_functions("shape_sweep_comm_mb")
    shape_sweep_comm_mb = funcs["shape_sweep_comm_mb"]

    assert shape_sweep_comm_mb(tokens=4096, width=14336) == (4096 * 14336 * 2) / 1e6


def test_build_overlap_shape_row_formats_missing_sdma():
    funcs = _load_functions("classify_overlap_note", "build_overlap_shape_row")
    build_overlap_shape_row = funcs["build_overlap_shape_row"]

    row = build_overlap_shape_row(
        profile_name="default",
        tokens=4096,
        ffn=14336,
        comm_mb=33.6,
        gemm_ms=2.5,
        nccl_comm_ms=1.2,
        nccl_seq_ms=3.4,
        nccl_ovl_ms=3.0,
        nccl_speedup=1.08,
        nccl_overlap_ratio=0.19,
        sdma_comm_ms=None,
        sdma_seq_ms=None,
        sdma_ovl_ms=None,
        sdma_speedup=None,
        sdma_overlap_ratio=None,
        sdma_vs_nccl=None,
    )

    assert row["nccl_note"] == "latency win"
    assert row["sdma_note"] == "n/a"
    assert row["sdma_vs_nccl"] is None


def test_build_overlap_shape_row_keeps_sdma_vs_nccl_ratio():
    funcs = _load_functions("classify_overlap_note", "build_overlap_shape_row")
    build_overlap_shape_row = funcs["build_overlap_shape_row"]

    row = build_overlap_shape_row(
        profile_name="backend_gap",
        tokens=8192,
        ffn=14336,
        comm_mb=67.1,
        gemm_ms=4.0,
        nccl_comm_ms=2.0,
        nccl_seq_ms=5.5,
        nccl_ovl_ms=5.0,
        nccl_speedup=1.10,
        nccl_overlap_ratio=0.17,
        sdma_comm_ms=1.4,
        sdma_seq_ms=5.2,
        sdma_ovl_ms=4.2,
        sdma_speedup=1.24,
        sdma_overlap_ratio=0.36,
        sdma_vs_nccl=1.19,
    )

    assert row["sdma_note"] == "latency win"
    assert row["sdma_vs_nccl"] == 1.19


def test_bench_comm_overlap_has_shape_sweep_test_class():
    tree = _module_ast()

    assert any(isinstance(node, ast.ClassDef) and node.name == "TestCommOverlapShapeSweep" for node in tree.body)


def test_bench_comm_overlap_source_mentions_shape_sweep_selector():
    assert "-k ShapeSweep" in _module_source()


def test_readme_mentions_comm_overlap_shape_sweep():
    source = BENCH_PATH.with_name("README.md").read_text()

    assert "bench_comm_overlap.py -v -s -k ShapeSweep" in source


def test_column_shape_sweep_shards_ffn_per_rank():
    source = _module_source()

    assert source.count("torch.randn(profile.ffn // self.world, profile.hidden") >= 2


def test_shape_sweep_validates_ffn_divisibility():
    assert "profile.ffn % self.world != 0" in _module_source()
