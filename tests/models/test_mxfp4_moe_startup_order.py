"""Source-order regression; does not certify real runtime hook ordering."""

import ast
from pathlib import Path


def test_ddp_immediately_followed_by_handoff():
    path = Path(__file__).resolve().parents[2] / "lumen/models/mxfp4_moe_startup.py"
    path = (
        path
        if path.exists()
        else Path(__file__).resolve().parents[3] / "lumen/models/mxfp4_moe_startup.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RoutedCoreRuntime"
    )
    init = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    for index, node in enumerate(init.body):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "DistributedDataParallel"
        ):
            following = init.body[index + 1]
            assert isinstance(following, ast.Assign)
            assert following.value.func.id == "Mxfp4DDPHandoff"
            return
    raise AssertionError("controlled DDP construction absent")
