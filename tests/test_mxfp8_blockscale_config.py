"""Regression for the AITER GROUP_K/BLOCK_SIZE_K wrapper contract."""

import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class TestMXFP8BlockscaleConfig(unittest.TestCase):
    def test_32_value_scales_override_128_value_tuned_tile(self):
        path = Path(__file__).parents[1] / "lumen/ops/quantize/linear.py"
        tree = ast.parse(path.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_gemm_mxfp8_triton"
        )
        namespace = {
            "torch": types.SimpleNamespace(bfloat16="bf16"),
            "_e8m0_to_fp32": lambda scale: scale,
        }
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        gemm = types.ModuleType("aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale")
        loader = types.ModuleType("aiter.ops.triton.utils.gemm_config_utils")
        tuned = {"BLOCK_SIZE_K": 128, "NUM_KSPLIT": 1}
        loader.get_gemm_config = lambda *args, **kwargs: (tuned.copy(), True)
        expected = object()

        def blockscale(a, w, sa, sw, config=None, **kwargs):
            group_k = w.shape[1] // sw.shape[1]
            block_k = 128 if config is None else config["BLOCK_SIZE_K"]
            self.assertEqual(group_k, block_k, "GROUP_K must equal BLOCK_SIZE_K")
            self.assertEqual(kwargs["backend"], "triton")
            return expected

        gemm.gemm_a8w8_blockscale = blockscale
        a = types.SimpleNamespace(shape=(128, 4096))
        w = types.SimpleNamespace(shape=(4096, 4096))
        sa = types.SimpleNamespace(shape=(128, 128))
        sw = types.SimpleNamespace(shape=(4096, 128))
        with patch.dict(sys.modules, {gemm.__name__: gemm, loader.__name__: loader}):
            self.assertIs(namespace["_gemm_mxfp8_triton"](a, w, sa, sw), expected)
        self.assertEqual(tuned["BLOCK_SIZE_K"], 128)


if __name__ == "__main__":
    unittest.main()
