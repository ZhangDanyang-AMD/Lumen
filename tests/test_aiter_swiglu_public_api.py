"""Public SwiGLU API forwarding without requiring a GPU installation."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class TestPublicSwiGLU(unittest.TestCase):
    def test_forward_and_backward_out_use_public_api(self):
        torch = types.ModuleType("torch")
        torch.Tensor = object
        activation = types.ModuleType("aiter.ops.triton.fusions.activation")
        backward = activation
        x, grad, out = object(), object(), object()
        activation.fused_silu_mul = lambda value: value

        def run_backward(dy, value, out=None):
            self.assertIs(dy, grad)
            self.assertIs(value, x)
            return out

        backward.silu_and_mul_backward = run_backward
        with patch.dict(
            sys.modules,
            {
                "torch": torch,
                activation.__name__: activation,
                backward.__name__: backward,
            },
        ):
            path = Path(__file__).parents[1] / "lumen/ops/fused_swiglu.py"
            spec = importlib.util.spec_from_file_location("swiglu_under_test", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertIs(module.fused_swiglu(x), x)
            self.assertIs(module.fused_swiglu_backward(grad, x, out=out), out)


if __name__ == "__main__":
    unittest.main()
