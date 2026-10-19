"""Run affected real host components without Torch/Core imports or GPU execution."""

import importlib.util
import inspect
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    "tests.models.test_mxfp4_moe_cli_host",
    "tests.models.test_mxfp4_moe_resource_driver_host",
    "tests.models.test_mxfp4_moe_microbatch_host",
    "tests.ops.test_mxfp4_moe_config_consumption_host",
    "tests.utils.test_mxfp4_moe_partial_failure",
    "tests.utils.test_mxfp4_moe_failure_diagnostics",
    "tests.utils.test_mxfp4_moe_diagnostics_host",
    "tests.utils.test_mxfp4_moe_callback_bindings_host",
    "tests.utils.test_mxfp4_moe_probe_contract_host",
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    sys.path.insert(0, str(ROOT))
    imports = load_module("mxfp4_host_imports", ROOT / "tests/mxfp4_host_imports.py")
    modules = imports.load_host_modules()
    forbidden = ("lumen", "lumen.ops", "torch", "aiter", "megatron")
    if any(name in sys.modules for name in forbidden):
        raise RuntimeError("isolated host runner imported a GPU initializer")
    print(
        json.dumps(
            {
                "real_host_modules": len(vars(modules)),
                "GPU_initializers_imported": False,
                "synthetic_parent_packages": False,
            }
        )
    )
    suite = unittest.TestSuite()
    for name in CASES:
        module = load_module(name, ROOT / (name.replace(".", "/") + ".py"))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        for function_name, function in inspect.getmembers(module, inspect.isfunction):
            if function_name.startswith("test_"):
                suite.addTest(unittest.FunctionTestCase(function))
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    print(
        json.dumps(
            {
                "tests_run": result.testsRun,
                "skipped": len(result.skipped),
                "failures": len(result.failures),
                "errors": len(result.errors),
                "Core_import_case": "skipped is UNRUN, not a pass",
            }
        )
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
