"""Pure public metadata mapping contract, not AITER prepare/runtime proof."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def test_child_mapping_original_replay_shared_producer_and_empty():
    path = Path(__file__).resolve().parents[2] / "lumen/utils/mxfp4_moe_children.py"
    spec = importlib.util.spec_from_file_location("child_mapping", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Metadata fixture only: no fake launch/torch/autograd framework.
    view = SimpleNamespace(
        w_output_mode=1,
        w_output_dtype=1,
        primitive=lambda op, **kw: SimpleNamespace(
            config=(op, kw["variant"], kw.get("output_dtype", 0))
        ),
        gemm=lambda pass_id: SimpleNamespace(config=(4, pass_id)),
    )
    counts = (0, 31, 1, 33)
    forward = module.dense_children(view, counts, 0)
    replay = module.dense_children(view, counts, 0, replay=True)
    d = module.dense_children(view, counts, 1)
    w = module.dense_children(view, counts, 2)
    assert [item["op"] for item in forward] == ["X_dual", "gemm"] * 3
    assert [item["op"] for item in replay] == ["gemm"] * 3
    assert sum(item["op"] == "dY_dual" for item in d + w) == 3
    assert all(item["expert"] != 0 for item in forward + replay + d + w)
    assert [item["ordinal"] for item in d] == list(range(6))
    assert module.dense_children(view, (0,), 2) == []
    view.w_output_mode = 0
    zero = module.dense_children(view, (0,), 2)
    assert zero == [{"op": "zero", "expert": 0, "config_id": [5, 4, 1], "ordinal": 0}]
