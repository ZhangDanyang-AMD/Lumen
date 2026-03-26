import ast
from pathlib import Path

BENCH_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_wgrad_delay.py"


def _load_bench_tree() -> ast.Module:
    return ast.parse(BENCH_PATH.read_text())


def _load_function(name: str) -> ast.FunctionDef:
    tree = _load_bench_tree()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Missing {name} helper")


def _load_method(class_name: str, method_name: str) -> ast.FunctionDef:
    tree = _load_bench_tree()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f"Missing {class_name}.{method_name}")


def test_tp_sharded_out_features_helper_validates_divisibility():
    fn = _load_function("_tp_sharded_out_features")
    source = ast.unparse(fn)

    assert "out_features % world_size != 0" in source
    assert "raise ValueError" in source
    assert "return out_features // world_size" in source


def test_wgrad_overlap_scaling_uses_tp_sharded_comm_geometry():
    fn = _load_method("TestNCCLvsSdmaWgradDelay", "test_wgrad_overlap_scaling")
    source = ast.unparse(fn)

    assert "local_n = _tp_sharded_out_features(gemm_n, self.world)" in source
    assert "ar_buf = torch.randn(local_n, K, device=self.device, dtype=torch.bfloat16)" in source
    assert "out_features=local_n" in source


def test_wgrad_overlap_scaling_summary_uses_tp_sharded_comm_geometry():
    fn = _load_method("TestNCCLvsSdmaWgradDelay", "test_wgrad_overlap_scaling_summary")
    source = ast.unparse(fn)

    assert "local_n = _tp_sharded_out_features(gemm_n, self.world)" in source
    assert "ar_buf = torch.randn(local_n, K, device=self.device, dtype=torch.bfloat16)" in source
    assert "out_features=local_n" in source
