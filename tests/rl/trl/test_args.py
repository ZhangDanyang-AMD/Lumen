###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the TRL RL argument contract."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LUMEN_DIR = REPO_ROOT / "lumen"
RL_DIR = LUMEN_DIR / "rl"
TRL_DIR = RL_DIR / "trl"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_args_module():
    lumen_pkg = sys.modules.setdefault("lumen", types.ModuleType("lumen"))
    lumen_pkg.__path__ = [str(LUMEN_DIR)]

    rl_pkg = sys.modules.setdefault("lumen.rl", types.ModuleType("lumen.rl"))
    rl_pkg.__path__ = [str(RL_DIR)]

    trl_pkg = sys.modules.setdefault("lumen.rl.trl", types.ModuleType("lumen.rl.trl"))
    trl_pkg.__path__ = [str(TRL_DIR)]

    return _load_module("lumen.rl.trl.args", TRL_DIR / "args.py")


class TestTrlArgs:
    def test_beta_must_stay_zero_in_v1(self):
        mod = _load_args_module()

        with pytest.raises(ValueError, match="beta"):
            mod.TrlLumenArgs(
                model_name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM",
                dataset_name="trl-lib/Capybara",
                output_dir="out",
                beta=0.1,
            )

    def test_fsdp_version_selects_expected_accelerate_file(self):
        mod = _load_args_module()
        args = mod.TrlLumenArgs(
            model_name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM",
            dataset_name="trl-lib/Capybara",
            output_dir="out",
            fsdp_version=2,
        )

        path = mod.build_accelerate_config_path(args)

        assert path.endswith("examples/rl/trl/accelerate/fsdp2.yaml")

    def test_grpo_kwargs_preserve_lumen_batch_contract(self):
        mod = _load_args_module()
        args = mod.TrlLumenArgs(
            model_name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM",
            dataset_name="trl-lib/Capybara",
            output_dir="out",
            micro_batch_size=2,
            gradient_accumulation_steps=8,
            max_steps=10,
        )

        kwargs = mod.build_grpo_config_kwargs(args)

        assert kwargs["per_device_train_batch_size"] == 2
        assert kwargs["gradient_accumulation_steps"] == 8
        assert kwargs["max_steps"] == 10
        assert kwargs["bf16"] is True
        assert kwargs["beta"] == 0.0
