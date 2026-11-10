"""Probe contracts without importing GPU-dependent operator packages."""

import ast
import functools
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _probe(name):
    path = Path(__file__).parents[1] / "lumen/ops/dispatch.py"
    tree = ast.parse(path.read_text())
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    namespace = {"functools": functools}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[name]


class TestPublicProbes(unittest.TestCase):
    def test_swiglu_requires_both_public_functions(self):
        forward = types.ModuleType("aiter.ops.triton.fusions.activation")
        backward = types.ModuleType("aiter.ops.triton.fusions.silu_and_mul_backward")
        forward.fused_silu_mul = lambda x: x
        probe = _probe("_probe_aiter_swiglu")
        with patch.dict(
            sys.modules, {forward.__name__: forward, backward.__name__: backward}
        ):
            self.assertFalse(probe())
            backward.silu_and_mul_backward = lambda dy, x: dy
            probe.cache_clear()
            self.assertTrue(probe())
            del forward.fused_silu_mul
            probe.cache_clear()
            self.assertFalse(probe())

    def test_h16_probe_requires_final_upstream_symbol(self):
        module = types.ModuleType(
            "aiter.ops.triton.quant.fused_quant_transpose_hadamard_mxfp4"
        )
        probe = _probe("_probe_aiter_mxfp4_api")
        with patch.dict(sys.modules, {module.__name__: module}):
            self.assertFalse(probe("fused_quant_transpose_hadamard_mxfp4"))
            module.fused_quant_transpose_hadamard_mxfp4 = lambda *a: a
            probe.cache_clear()
            self.assertTrue(probe("fused_quant_transpose_hadamard_mxfp4"))


if __name__ == "__main__":
    unittest.main()
