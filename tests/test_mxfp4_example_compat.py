"""Version and mask contracts without importing GPU-dependent packages."""

import ast
import inspect
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]


def function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


class ExampleCompatibility(unittest.TestCase):
    def test_result_requires_twenty_finite_steps_and_no_skips(self):
        spec = importlib.util.spec_from_file_location("runner", ROOT / "examples/qwen3/scripts/run_mxfp4_examples.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        text = "\n".join(f"iteration {n}/ 20 elapsed time per iteration (ms): 350.0 lm loss: 3.0 grad norm: 39.0 number of skipped iterations: 0 number of nan iterations: 0" for n in range(1, 21))
        self.assertTrue(runner.parse_result("12", text, 0)["passed"])
        self.assertFalse(runner.parse_result("12", text.replace("grad norm: 39.0", "grad norm: nan"), 0)["passed"])
        self.assertFalse(runner.parse_result("12", text.replace("skipped iterations: 0", "skipped iterations: 1"), 0)["passed"])
        sft = "\n".join(f"step {n}/20 | loss 1.2 | grad_norm 2.7e-1 | lr 4e-4 | step_time_ms 560.0" for n in range(1, 21))
        self.assertTrue(runner.parse_result("13", sft, 0)["passed"])
        self.assertFalse(runner.parse_result("13", sft, 1)["passed"])

    def test_old_spec_drops_only_inactive_legacy_option(self):
        fn = function("lumen/models/megatron.py", "_legacy_moe_spec_kwargs", {"inspect": inspect})
        def old(num_experts=None):
            pass
        def new(moe_use_legacy_grouped_gemm=False):
            pass
        self.assertEqual(fn(old, False), {})
        with self.assertRaises(NotImplementedError):
            fn(old, True)
        self.assertEqual(fn(new, True), {"moe_use_legacy_grouped_gemm": True})

    def test_nonempty_mask_reaches_original_attention_and_mlp(self):
        calls = []
        mask, hidden, result = object(), object(), object()
        def attention(self, value, **kwargs):
            calls.append(("attention", kwargs["padding_mask"]))
            return result
        def mlp(self, value, inference_context=None, padding_mask=None):
            calls.append(("mlp", padding_mask))
            return result
        ns = {"_orig_fwd_attn": attention, "_orig_fwd_mlp": mlp,
              "_orig_mlp_accepts_padding_mask": True}
        attn = function("lumen/patches/runtime/megatron_import.py", "_patched_fwd_attn", ns)
        feedforward = function("lumen/patches/runtime/megatron_import.py", "_patched_fwd_mlp", ns)
        layer = SimpleNamespace(_lumen_deferred_bda=None)
        self.assertIs(attn(layer, hidden, padding_mask=mask), result)
        self.assertIs(feedforward(layer, hidden, padding_mask=mask), result)
        self.assertEqual(calls, [("attention", mask), ("mlp", mask)])
        ns["_orig_mlp_accepts_padding_mask"] = False
        with self.assertRaises(NotImplementedError):
            feedforward(layer, hidden, padding_mask=mask)


if __name__ == "__main__":
    unittest.main()
