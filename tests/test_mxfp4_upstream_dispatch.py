"""CPU checks of MXFP4 wrapper contracts; GPU numerics live in tests/ops."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class TestMXFP4UpstreamDispatch(unittest.TestCase):
    def test_rng_values_do_not_create_distinct_backend_cache_entries(self):
        torch = types.ModuleType("torch")
        dispatch = types.ModuleType("lumen.ops.dispatch")
        dispatch.Backend = types.SimpleNamespace(TRITON="triton")
        dispatch._probe_aiter_mxfp4_api = lambda name: True
        keys = []

        def run(chain, op_name, **kwargs):
            keys.append(op_name)
            return chain[0][1]()

        dispatch.try_backends = run
        with patch.dict(sys.modules, {"torch": torch, dispatch.__name__: dispatch}):
            path = Path(__file__).parents[1] / "lumen/ops/quantize/mxfp4.py"
            spec = importlib.util.spec_from_file_location("cache_under_test", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            kwargs = dict(use_sr=True, philox_seed=1, philox_offset=0)
            module._run("api", "quant", lambda: None, (object(),), kwargs)
            kwargs.update(philox_seed=2, philox_offset=256)
            module._run("api", "quant", lambda: None, (object(),), kwargs)
            self.assertEqual(keys[0], keys[1])
            kwargs.update(use_sr=False)
            module._run("api", "quant", lambda: None, (object(),), kwargs)
            self.assertNotEqual(keys[1], keys[2])

    def test_h16_layout_and_rng_arguments_reach_upstream(self):
        torch = types.ModuleType("torch")
        torch.Tensor = object
        dispatch = types.ModuleType("lumen.ops.dispatch")
        dispatch.Backend = types.SimpleNamespace(TRITON="triton")
        dispatch._probe_aiter_mxfp4_api = lambda name: True
        dispatch.try_backends = lambda chain, **kwargs: chain[0][1]()
        upstream = types.ModuleType(
            "aiter.ops.triton.quant.fused_quant_transpose_hadamard_mxfp4"
        )
        x, sign, expected = object(), object(), (object(), object(), object(), object())

        def quant(value, signs, **kwargs):
            self.assertIs(value, x)
            self.assertIs(signs, sign)
            self.assertEqual(
                kwargs,
                dict(
                    block_size=32,
                    g=16,
                    use_sr_row=True,
                    use_sr_transposed=False,
                    philox_seed=123,
                    philox_offset=456,
                    swizzle_scale=True,
                    shuffle_col=True,
                ),
            )
            return expected

        upstream.fused_quant_transpose_hadamard_mxfp4 = quant
        with patch.dict(
            sys.modules,
            {"torch": torch, dispatch.__name__: dispatch, upstream.__name__: upstream},
        ):
            path = Path(__file__).parents[1] / "lumen/ops/quantize/mxfp4.py"
            self.assertTrue(
                path.exists(), "upstream MXFP4 integration wrapper is missing"
            )
            spec = importlib.util.spec_from_file_location("mxfp4_under_test", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module._rng(False, 123, 456), (None, 0))
            self.assertEqual(module._rng(True, 123, 456), (123, 456))
            result = module.dual_layout_quant_mxfp4(
                x,
                sign,
                use_sr_row=True,
                use_sr_transposed=False,
                philox_seed=123,
                philox_offset=456,
                swizzle_scale=True,
                shuffle_col=True,
            )
            self.assertIs(result, expected)


if __name__ == "__main__":
    unittest.main()
