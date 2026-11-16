"""Version and mask contracts without importing GPU-dependent packages."""

import ast
import importlib.util
import inspect
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

ROOT = Path(__file__).parents[1]


def function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


class ExampleCompatibility(unittest.TestCase):
    def test_result_requires_twenty_finite_steps_and_no_skips(self):
        spec = importlib.util.spec_from_file_location(
            "runner", ROOT / "examples/qwen3/scripts/run_mxfp4_examples.py"
        )
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        text = "\n".join(
            f"iteration {n}/ 20 elapsed time per iteration (ms): 350.0 lm loss: 3.0 grad norm: 39.0 number of skipped iterations: 0 number of nan iterations: 0"
            for n in range(1, 21)
        )
        self.assertTrue(runner.parse_result("12", text, 0)["passed"])
        self.assertFalse(
            runner.parse_result(
                "12", text.replace("grad norm: 39.0", "grad norm: nan"), 0
            )["passed"]
        )
        self.assertFalse(
            runner.parse_result(
                "12", text.replace("skipped iterations: 0", "skipped iterations: 1"), 0
            )["passed"]
        )
        sft = "\n".join(
            f"step {n}/20 | loss 1.2 | grad_norm 2.7e-1 | lr 4e-4 | step_time_ms 560.0"
            for n in range(1, 21)
        )
        self.assertTrue(runner.parse_result("13", sft, 0)["passed"])
        self.assertFalse(runner.parse_result("13", sft, 1)["passed"])

    def test_old_spec_drops_only_inactive_legacy_option(self):
        fn = function(
            "lumen/models/megatron.py", "_legacy_moe_spec_kwargs", {"inspect": inspect}
        )

        def old(num_experts=None):
            pass

        def new(moe_use_legacy_grouped_gemm=False):
            pass

        self.assertEqual(fn(old, False), {})
        with self.assertRaises(NotImplementedError):
            fn(old, True)
        self.assertEqual(fn(new, True), {"moe_use_legacy_grouped_gemm": True})

    def test_masked_mlp_completes_deferred_attention_residual(self):
        add = function(
            "lumen/ops/fused_residual_norm.py",
            "deferred_bda_add",
            {
                "torch": SimpleNamespace(Tensor=int),
                "Tuple": tuple,
                "Optional": Optional,
            },
        )
        module = types.ModuleType("lumen.ops.fused_residual_norm")
        module.deferred_bda_add = add
        mask = object()
        calls = []

        def mlp(self, value, inference_context=None, padding_mask=None):
            calls.append((value, padding_mask))
            return value

        ns = {"_orig_fwd_mlp": mlp, "_orig_mlp_accepts_padding_mask": True}
        forward = function(
            "lumen/patches/runtime/megatron_import.py", "_patched_fwd_mlp", ns
        )
        for bias in (None, 3):
            for defer_to_linear in (False, True):
                with self.subTest(bias=bias, defer_to_linear=defer_to_linear):
                    layer = SimpleNamespace(
                        _lumen_deferred_bda=((5, bias), 11),
                        _lumen_can_defer_bda_to_lnlinear=defer_to_linear,
                    )
                    with patch.dict(sys.modules, {module.__name__: module}):
                        result = forward(layer, 5 + (bias or 0), padding_mask=mask)
                    self.assertEqual(result, 16 + (bias or 0))
                    self.assertEqual(calls[-1], (result, mask))
                    self.assertIsNone(layer._lumen_deferred_bda)
                    # A second masked call must not add the residual twice.
                    self.assertEqual(forward(layer, result, padding_mask=mask), result)

    def test_builders_inspect_the_invoked_spec_factory(self):
        tree = ast.parse((ROOT / "lumen/models/megatron.py").read_text())
        helper = function(
            "lumen/models/megatron.py", "_legacy_moe_spec_kwargs", {"inspect": inspect}
        )

        def submodules(moe_use_legacy_grouped_gemm=False):
            pass

        def old_spec(*args, **kwargs):
            self.assertNotIn("moe_use_legacy_grouped_gemm", kwargs)
            return "built"

        for name in (
            "get_gpt_layer_local_spec",
            "get_gpt_layer_with_transformer_engine_spec",
        ):
            call = next(
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == name
            )
            gls = SimpleNamespace(
                get_gpt_layer_local_submodules=submodules,
                get_gpt_layer_with_transformer_engine_submodules=submodules,
            )
            ns = {
                "_legacy_moe_spec_kwargs": helper,
                "_gls": gls,
                name: old_spec,
                "args": SimpleNamespace(
                    num_experts=None,
                    moe_grouped_gemm=False,
                    qk_layernorm=False,
                    multi_latent_attention=False,
                    normalization="RMSNorm",
                    moe_use_legacy_grouped_gemm=False,
                ),
                "config": SimpleNamespace(use_kitchen=False),
            }
            with self.subTest(factory=name):
                self.assertEqual(
                    eval(compile(ast.Expression(call), "spec-call", "eval"), ns),
                    "built",
                )
                ns["args"].moe_use_legacy_grouped_gemm = True
                with self.assertRaises(NotImplementedError):
                    eval(compile(ast.Expression(call), "spec-call", "eval"), ns)

    def test_nonempty_mask_reaches_original_attention_and_mlp(self):
        calls = []
        mask, hidden, result = object(), object(), object()

        def attention(self, value, **kwargs):
            calls.append(("attention", kwargs["padding_mask"]))
            return result

        def mlp(self, value, inference_context=None, padding_mask=None):
            calls.append(("mlp", padding_mask))
            return result

        ns = {
            "_orig_fwd_attn": attention,
            "_orig_fwd_mlp": mlp,
            "_orig_mlp_accepts_padding_mask": True,
        }
        attn = function(
            "lumen/patches/runtime/megatron_import.py", "_patched_fwd_attn", ns
        )
        feedforward = function(
            "lumen/patches/runtime/megatron_import.py", "_patched_fwd_mlp", ns
        )
        layer = SimpleNamespace(_lumen_deferred_bda=None)
        self.assertIs(attn(layer, hidden, padding_mask=mask), result)
        self.assertIs(feedforward(layer, hidden, padding_mask=mask), result)
        self.assertEqual(calls, [("attention", mask), ("mlp", mask)])
        ns["_orig_mlp_accepts_padding_mask"] = False
        with self.assertRaises(NotImplementedError):
            feedforward(layer, hidden, padding_mask=mask)


if __name__ == "__main__":
    unittest.main()
