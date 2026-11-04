"""CPU contracts for the core split; provider execution uses explicit dependency doubles."""

import argparse
import ast
from pathlib import Path
from typing import Callable

import pytest
import torch
from torch import nn

from lumen.config import LumenConfig, linear_quant_enabled, linear_quant_requested
from lumen.quantize import QuantFormat, ScalingType, disable
from lumen.quantize import native
from lumen.rl.verl.config import VerlLumenArgs


@pytest.mark.parametrize("values,active,requested", [
    ({}, False, False),
    ({"linear_fp8": True, "linear_fp8_scaling": "none"}, False, True),
    ({"linear_fp8": True}, True, True),
    ({"linear_quant_format": "mxfp4"}, True, True),
    ({"linear_quant_format": "mxfp4", "linear_quant_scaling": "none"}, False, False),
])
def test_legacy_training_policy_and_effective_recipe_are_distinct(values, active, requested):
    args = argparse.Namespace(**values)
    assert linear_quant_enabled(args) is active
    assert linear_quant_requested(args) is requested


class NativeDouble(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(32, 32))
        self.scaling_type = "none"
        self.enable_calls = 0

    def enable_fp8(self, scaling_manager, scaling_type, fp8_dtype, block_size):
        self.enable_calls += 1
        self.scaling_manager = scaling_manager
        self.scaling_type = scaling_type
        self.block_size = block_size

    def forward(self, x):
        return x


@pytest.fixture
def native_model(monkeypatch):
    monkeypatch.setattr(native, "is_native_linear", lambda module: isinstance(module, NativeDouble))
    monkeypatch.delenv("LUMEN_WEIGHT_QUANT_ONCE", raising=False)
    model = nn.Module()
    model.proj = NativeDouble()
    return model


def test_public_mxfp4_enable_and_disable_are_paired(native_model):
    cfg = LumenConfig(format="mxfp4")
    original_forward = native_model.proj.forward
    cfg.enable(native_model)
    assert native_model.proj.scaling_type == "mxfp4"
    assert native_model.proj.block_size == 32
    assert native_model.proj.enable_calls == 1
    assert native_model.proj.forward == original_forward
    disable(native_model)
    assert native_model.proj.scaling_type == "none"
    cfg.enable(native_model)
    assert native_model.proj.scaling_type == "mxfp4"
    assert native_model.proj.enable_calls == 2


def test_public_fp8_does_not_acquire_new_native_enable_or_disable_policy(native_model):
    LumenConfig(format="fp8_e4m3").enable(native_model)
    assert native_model.proj.enable_calls == 0
    native_model.proj.scaling_type = "dynamic"  # represents the existing separate FP8 helper
    disable(native_model)
    assert native_model.proj.scaling_type == "dynamic"


@pytest.mark.parametrize("format,fp8_calls,native_calls", [
    ("mxfp4", 0, 1), ("fp8_e4m3", 1, 0),
])
def test_provider_has_one_owner_for_each_native_format(monkeypatch, native_model, format, fp8_calls, native_calls):
    # Compile the real provider function without importing the unavailable Megatron stack.
    # Model construction and legacy FP8 enable are doubles; cfg.enable is real CPU execution.
    source = Path(__file__).resolve().parents[2] / "lumen/models/megatron.py"
    node = next(node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "make_lumen_model_provider")
    calls = []
    args = argparse.Namespace(linear_quant_format=format, lumen_linear=True)
    scope = {
        "Callable": Callable, "get_args": lambda: args,
        "apply_lora": lambda *a, **kw: None, "apply_fp8_training": lambda *a, **kw: None,
        "_install_layernorm_linear_ckpt_hook": lambda model: None,
        "enable_fp8_for_parallel_linear": lambda *a, **kw: calls.append(kw),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    provider = scope["make_lumen_model_provider"](lambda *a, **kw: native_model)
    assert provider() is native_model
    assert len(calls) == fp8_calls
    assert native_model.proj.enable_calls == native_calls
    if calls:
        assert calls[0]["quant_config"].format.value == format


@pytest.mark.parametrize("disabled", [False, True])
def test_mxfp4_native_markers_reach_existing_fsdp_comm_consumer(native_model, disabled):
    from lumen.models.fsdp import _wrap_params_as_mxfp4_comm
    from lumen.quantize.comm_tensor import MXFP4CommTensor

    native_model.proj.weight = nn.Parameter(torch.zeros(32, 32, dtype=torch.bfloat16))
    LumenConfig(format="mxfp4").enable(native_model)
    if disabled:
        disable(native_model)
    # Actual CPU wrapper construction only, not all-gather or MXFP4 GPU quantization.
    count = _wrap_params_as_mxfp4_comm(native_model, block_size=32, world_size=1)
    assert count == (0 if disabled else 1)
    assert isinstance(native_model.proj.weight, MXFP4CommTensor) is (not disabled)


def test_existing_verl_fp8_boolean_fields_still_accept_explicit_values():
    args = VerlLumenArgs(model_name_or_path="test", linear_fp8=True,
                         linear_fp8_activation=False, linear_fp8_wgrad=False,
                         linear_fp8_reduce_amax=True)
    cfg = LumenConfig.from_args(args)
    assert not cfg.quantize_activation
    assert not cfg.fp8_wgrad
    assert cfg.reduce_amax


def _provider_scope(extra):
    source = Path(__file__).resolve().parents[2] / "lumen/models/megatron.py"
    node = next(node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "make_lumen_model_provider")
    scope = {
        "Callable": Callable,
        "apply_lora": lambda *a, **kw: None,
        "apply_fp8_training": lambda *a, **kw: None,
        "_install_layernorm_linear_ckpt_hook": lambda model: None,
    }
    scope.update(extra)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope["make_lumen_model_provider"]


def test_provider_resolves_the_recipe_after_the_builder_translates_megatron_args():
    # The builder runs _override_te_args_for_lumen, which sets lumen_fp8_format
    # from Megatron's --fp8-format; resolving earlier would keep fp8_e4m3.
    calls = []
    args = argparse.Namespace(linear_fp8=True, lumen_linear=True)

    def builder(*a, **kw):
        args.lumen_fp8_format = "hybrid"
        return nn.Linear(16, 16)

    provider = _provider_scope({
        "get_args": lambda: args,
        "enable_fp8_for_parallel_linear": lambda model, **kw: calls.append(kw),
    })(builder)
    provider()
    assert calls[0]["quant_config"].format == QuantFormat.HYBRID


@pytest.mark.parametrize("scaling", ["blockwise", "blockwise2d", "delayed", "dynamic"])
def test_legacy_fp4_gate_normalizes_hand_built_fp8_placeholders(scaling):
    # The old example and integrations built this Namespace shape; the gate used
    # to overwrite exactly the format/scaling/block_size these names carried.
    args = argparse.Namespace(
        linear_fp8=False, linear_fp4=True, linear_fp8_format="fp8_e4m3",
        linear_fp8_scaling=scaling, linear_fp8_block_size=128,
    )
    cfg = LumenConfig.from_args(args)
    assert cfg.format == "mxfp4"
    assert cfg.quant_config.block_size == 32
    assert cfg.quant_config.scaling == ScalingType.BLOCKWISE


def test_generic_recipe_fields_stay_strict_under_the_legacy_fp4_gate():
    args = argparse.Namespace(linear_fp4=True, linear_quant_block_size=64)
    with pytest.raises(ValueError, match="32-element blocks"):
        _ = LumenConfig.from_args(args).quant_config


@pytest.mark.parametrize("mode,active", [("bf16", False), ("mxfp4", True)])
def test_qwen3_preset_ignores_the_launcher_fp8_scaling_placeholder(mode, active):
    from examples.qwen3.train_qwen3_fsdp import parse_args

    # run_qwen3_fsdp_mi308.sh forwards --fp8-scaling unconditionally.
    args = parse_args(["--model-name-or-path", "test", "--train-data-path", "test.jsonl",
                       "--mode", mode, "--fp8-scaling", "blockwise2d"])
    cfg = LumenConfig.from_args(args)
    assert cfg.quant_config.is_quantized is active
    assert (cfg.format == "mxfp4") is active


@pytest.mark.parametrize("mode,scaling,mxfp4", [
    ("bf16", "none", False), ("mxfp4", "blockwise", True),
])
def test_qwen3_preset_keeps_a_generic_scaling_beside_the_placeholder(mode, scaling, mxfp4):
    from examples.qwen3.train_qwen3_fsdp import parse_args

    args = parse_args(["--model-name-or-path", "test", "--train-data-path", "test.jsonl",
                       "--mode", mode, "--fp8-scaling", "blockwise2d",
                       "--linear-quant-scaling", scaling])
    cfg = LumenConfig.from_args(args)
    assert (cfg.format == "mxfp4") is mxfp4
    assert cfg.quant_config.is_quantized is mxfp4


def test_fp8_param_storage_uses_the_resolved_recipe(monkeypatch):
    import sys
    import types

    from lumen.models import fp8_param_storage as storage

    megatron = types.ModuleType("megatron")
    training = types.ModuleType("megatron.training")
    training.get_args = lambda: argparse.Namespace(
        linear_quant_format="fp8_e4m3",
        linear_quant_scaling="blockwise",
        linear_quant_block_size=64,
    )
    megatron.training = training
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.training", training)
    # The stored weights must match the GEMMs the generic selector configured,
    # not the untouched legacy default.
    assert storage.get_fp8_store_recipe() == ("blockwise", 64)
    assert storage.get_fp8_store_scaling() == "blockwise"


def test_provider_rejects_fp8_param_storage_for_mxfp4():
    args = argparse.Namespace(
        linear_quant_format="mxfp4", lumen_linear=True, fp8_param_storage=True,
    )
    provider = _provider_scope({
        "get_args": lambda: args,
        "enable_fp8_for_parallel_linear": lambda model, **kw: None,
    })(lambda *a, **kw: nn.Linear(16, 16))
    with pytest.raises(ValueError, match="fp8-param-storage"):
        provider()

def test_fp8_store_recipe_propagates_config_errors(monkeypatch):
    import sys
    import types

    from lumen.models import fp8_param_storage as storage

    megatron = types.ModuleType("megatron")
    training = types.ModuleType("megatron.training")
    training.get_args = lambda: argparse.Namespace(
        linear_quant_format="mxfp4", linear_quant_scaling="dynamic",
    )
    megatron.training = training
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.training", training)
    # Storing delayed/128 weights while the GEMMs run another recipe is the
    # mismatch this resolution exists to prevent, so errors must not be swallowed.
    with pytest.raises(ValueError, match="fixed blockwise"):
        storage.get_fp8_store_recipe()
