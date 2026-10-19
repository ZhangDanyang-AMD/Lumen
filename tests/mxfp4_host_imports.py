"""Load complete real host modules without importing GPU package initializers."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "ledger": "lumen.utils.mxfp4_moe_ledger",
    "rng": "lumen.quantize.mxfp4_moe_rng",
    "state": "lumen.quantize.mxfp4_moe_state",
    "diagnostics": "lumen.utils.mxfp4_moe_diagnostics",
    "control": "lumen.utils.mxfp4_moe_control",
    "children": "lumen.utils.mxfp4_moe_children",
    "submission": "lumen.utils.mxfp4_moe_submission",
    "execution_ledger": "lumen.utils.mxfp4_moe_execution_ledger",
    "resources": "lumen.utils.mxfp4_moe_resources",
    "ddp": "lumen.utils.mxfp4_moe_ddp",
    "lifecycle": "lumen.models.mxfp4_moe_lifecycle",
    "driver": "lumen.models.mxfp4_moe_driver",
    "execution": "lumen.ops.mxfp4_moe_execution",
    "materialization": "lumen.ops.mxfp4_moe_materialization_boundary",
    "options": "lumen.models.mxfp4_moe_options",
    "pytest_options": "tests.env.mxfp4_moe_pytest",
}


def load_host_modules():
    before = set(sys.modules)
    loaded = {}
    for alias, name in MODULES.items():
        path = ROOT / (name.replace(".", "/") + ".py")
        if name in sys.modules:
            module = sys.modules[name]
            if Path(module.__file__).resolve() != path.resolve():
                raise RuntimeError(f"host module origin mismatch: {name}")
        else:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            # This is the actual full file module, not a synthetic parent package.
            sys.modules[name] = module
            spec.loader.exec_module(module)
        loaded[alias] = module
    introduced = set(sys.modules) - before
    if introduced.intersection({"lumen", "lumen.ops", "torch", "aiter", "megatron"}):
        raise RuntimeError("host dependency imported a GPU initializer")
    return SimpleNamespace(**loaded)
