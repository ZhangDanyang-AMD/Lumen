"""Stdlib CLI behavior, exact extracted host functions; GPU imports are unrun."""

import argparse
import ast
import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load_host_modules():
    spec = importlib.util.spec_from_file_location(
        "mxfp4_host_imports", ROOT / "tests/mxfp4_host_imports.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_host_modules()


def namespace():
    modules = load_host_modules()
    return {
        **vars(modules.options),
        **vars(modules.pytest_options),
        "Mxfp4StateError": modules.ledger.Mxfp4StateError,
    }


class PytestParserProtocol:
    """Exercise addoption semantics with stdlib argparse, not a fake pytest import."""

    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def getgroup(self, name):
        group = self.parser.add_argument_group(name)
        return SimpleNamespace(addoption=group.add_argument)


class CLIHostTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.ns = namespace()
        self.parser = argparse.ArgumentParser()
        self.ns["add_mxfp4_moe_arguments"](self.parser)
        self.flags = [
            "--lumen-moe-mxfp4",
            "--lumen-moe-mxfp4-backend",
            "sequential",
            "--lumen-moe-mxfp4-strict",
        ]

    def test_defaults_preserved_and_p0_requires_explicit_selection(self):
        args = self.parser.parse_args([])
        self.assertFalse(args.lumen_moe_mxfp4)
        self.assertEqual(args.lumen_moe_mxfp4_backend, "auto")
        self.assertFalse(args.lumen_moe_mxfp4_strict)
        self.assertEqual(
            args.lumen_moe_mxfp4_grouped_tuning_policy, "require-certified-bucket"
        )
        with self.assertRaisesRegex(
            self.ns["Mxfp4StateError"], "FEATURE_NOT_REQUESTED"
        ):
            self.ns["validate_p0_selector"](args)
        for backend in ("auto", "grouped", "sequential"):
            for strict in (False, True):
                flags = ["--lumen-moe-mxfp4", "--lumen-moe-mxfp4-backend", backend]
                if strict:
                    flags.append("--lumen-moe-mxfp4-strict")
                args = self.parser.parse_args(flags)
                if backend == "sequential" and strict:
                    self.ns["validate_p0_selector"](args)
                else:
                    with self.assertRaisesRegex(
                        self.ns["Mxfp4StateError"],
                        "P0_REQUIRES_EXPLICIT_STRICT_SEQUENTIAL",
                    ):
                        self.ns["validate_p0_selector"](args)

    def test_plugin_options_reach_same_config_validator_and_reset(self):
        facade = PytestParserProtocol()
        self.ns["pytest_addoption"](facade)
        parsed = facade.parser.parse_args(self.flags)
        self.ns["pytest_configure"](
            SimpleNamespace(getoption=lambda name: getattr(parsed, name))
        )
        args = self.ns["fixture_options"]()
        self.assertEqual(vars(args), vars(parsed))
        config = SimpleNamespace(
            add_bias_linear=False,
            delay_wgrad_compute=False,
            expert_tensor_parallel_size=1,
        )
        receipt = self.ns["validate_p0_options"](args, config)
        self.assertEqual(receipt["requested_backend"], "sequential")
        self.assertTrue(receipt["strict"])
        self.assertEqual(receipt["grouped_tuning_policy"], "require-certified-bucket")
        self.assertFalse(receipt["grouped_tuning_policy_applies"])
        args.lumen_moe_mxfp4_backend = "auto"
        self.assertEqual(
            self.ns["fixture_options"]().lumen_moe_mxfp4_backend, "sequential"
        )
        self.ns["pytest_unconfigure"](None)
        with self.assertRaisesRegex(
            self.ns["Mxfp4StateError"], "P0_CLI_PLUGIN_REQUIRED"
        ):
            self.ns["fixture_options"]()

    def test_incompatible_flags_and_config_reject_without_fallback(self):
        args = self.parser.parse_args(self.flags)
        os.environ["LUMEN_DSV4_LINEAR_FP8"] = "1"
        with self.assertRaisesRegex(
            self.ns["Mxfp4StateError"], "INCOMPATIBLE_QUANTIZATION_FLAGS"
        ):
            self.ns["validate_p0_selector"](args)
        del os.environ["LUMEN_DSV4_LINEAR_FP8"]
        for field, value in (
            ("add_bias_linear", True),
            ("delay_wgrad_compute", True),
            ("expert_tensor_parallel_size", 2),
        ):
            config = SimpleNamespace(
                add_bias_linear=False,
                delay_wgrad_compute=False,
                expert_tensor_parallel_size=1,
            )
            setattr(config, field, value)
            with self.assertRaisesRegex(
                self.ns["Mxfp4StateError"], "UNSUPPORTED_ROUTED_FC"
            ):
                self.ns["validate_p0_options"](args, config)

    def test_startup_consumes_selector_before_ddp_and_all_callers_pass_options(self):
        path = ROOT / "lumen/models/mxfp4_moe_startup.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        runtime = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "RoutedCoreRuntime"
        )
        init = next(
            node
            for node in runtime.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        self.assertIn("options", [arg.arg for arg in init.args.kwonlyargs])
        selector = next(
            i
            for i, node in enumerate(init.body)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "validate_p0_options"
        )
        ddp = next(
            i
            for i, node in enumerate(init.body)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "DistributedDataParallel"
        )
        self.assertLess(selector, ddp)
        self.assertIn("self.handoff = Mxfp4DDPHandoff", ast.unparse(init.body[ddp + 1]))
        for name in ("paired_numeric", "hot_cache", "routed_runtime"):
            tree = ast.parse(
                (ROOT / ("tests/env/test_mxfp4_moe_" + name + ".py")).read_text(
                    encoding="utf-8"
                )
            )
            calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "RoutedCoreRuntime"
            ]
            self.assertEqual(len(calls), 1)
            selected = next(kw.value for kw in calls[0].keywords if kw.arg == "options")
            self.assertEqual(ast.unparse(selected), "fixture_options()")


if __name__ == "__main__":
    unittest.main()
