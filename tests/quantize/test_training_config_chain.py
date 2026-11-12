"""CPU configuration/patch contracts; native modules use an explicit test double."""

import argparse
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from lumen.config import LumenConfig, linear_quant_enabled, mxfp4_enabled
from lumen.models.fsdp import add_common_fsdp_args, register_quant_optimizer_hooks, validate_fsdp_quant_args
from lumen.rl.trl.args import TrlLumenArgs
from lumen.rl.verl.config import VerlLumenArgs, from_verl_config


@pytest.mark.parametrize("flags", [
    ["--linear-fp4"], ["--linear-quant-format", "mxfp4"],
])
def test_generic_and_legacy_fsdp_select_the_same_active_recipe(flags):
    parser = argparse.ArgumentParser()
    add_common_fsdp_args(parser)
    args = parser.parse_args(flags + ["--fsdp-version", "2", "--fsdp-mxfp4-comm"])
    validate_fsdp_quant_args(args)
    assert linear_quant_enabled(args)
    assert mxfp4_enabled(args)
    assert LumenConfig.from_args(args).quant_config.recipe == "mxfp4"


def test_legacy_format_only_and_consumer_gates_agree():
    parser = argparse.ArgumentParser()
    add_common_fsdp_args(parser)
    args = parser.parse_args(["--linear-fp8-format", "hybrid"])
    assert not LumenConfig.from_args(args).quant_config.is_quantized
    assert not linear_quant_enabled(args)
    # A direct partial Namespace keeps the baseline constructor default, but is not a training selector.
    assert LumenConfig.from_args(SimpleNamespace()).quant_config.is_quantized
    assert not linear_quant_enabled(SimpleNamespace())


def test_qwen3_preset_still_enables_explicit_legacy_format():
    from examples.qwen3.train_qwen3_fsdp import parse_args

    args = parse_args(["--model-name-or-path", "test", "--train-data-path", "test.jsonl",
                       "--linear-fp8-format", "hybrid"])
    assert linear_quant_enabled(args)
    assert LumenConfig.from_args(args).format == "hybrid"
    assert LumenConfig.from_args(args).scaling == "blockwise2d"


def test_unsupported_mxfp4_output_width_remains_bf16(monkeypatch):
    monkeypatch.delenv("LUMEN_WEIGHT_QUANT_ONCE", raising=False)
    model = nn.Module()
    model.proj = nn.Linear(32, 32)
    model.score = nn.Linear(32, 1)
    score_forward = model.score.forward
    cfg = LumenConfig(format="mxfp4")
    cfg.enable(model, backend="triton")
    assert model.proj._quant_enabled
    assert model.score.forward == score_forward
    assert not getattr(model.score, "_quant_enabled", False)
    assert model.score(torch.zeros(2, 32)).shape == (2, 1)


def test_incompatible_models_disable_shared_swiglu_fusion(monkeypatch, caplog):
    from lumen.models import _swiglu_fp8_fuse as bridge

    monkeypatch.setattr(bridge, "_fused_scaling_configured", False)
    monkeypatch.setattr(bridge, "_fused_scaling_conflict", False)
    monkeypatch.setattr(bridge, "_fused_scaling", {"type": "dynamic", "block_size": 128})
    bridge.set_fused_swiglu_scaling("blockwise2d", 128)
    bridge._cache.valid = True
    bridge._cache.fp8_data = object()
    bridge._cache.scale = object()
    bridge.set_fused_swiglu_scaling("dynamic", 128)
    assert bridge._fused_scaling["type"] == "blockwise2d"
    assert bridge._fused_scaling_conflict
    assert bridge.pop_swiglu_fp8_cache() is None
    assert "disabling fused FP8" in caplog.text


def test_disabled_generic_recipe_does_not_enable_fsdp_comm():
    args = SimpleNamespace(linear_quant_format="mxfp4", linear_quant_scaling="none",
                           fsdp_mxfp4_comm=True, fsdp_version=2)
    assert not mxfp4_enabled(args)
    with pytest.raises(ValueError, match="requires"):
        validate_fsdp_quant_args(args)


def test_generic_selector_registers_real_cpu_optimizer_invalidation(monkeypatch):
    monkeypatch.delenv("LUMEN_WEIGHT_QUANT_ONCE", raising=False)
    args = SimpleNamespace(linear_quant_format="mxfp4")
    model = nn.Linear(32, 32)
    LumenConfig.from_args(args).enable(model, backend="triton")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model._mxfp4_w_cache = object()
    assert register_quant_optimizer_hooks(model, optimizer, args)
    optimizer.step()
    assert not hasattr(model, "_mxfp4_w_cache")


@pytest.mark.parametrize("kind", ["trl", "verl"])
def test_rl_generic_recipe_does_not_collide_with_legacy_defaults(kind):
    constructor = TrlLumenArgs if kind == "trl" else VerlLumenArgs
    args = constructor(model_name_or_path="test-model", output_dir="test-output",
                       linear_quant_format="mxfp4")
    assert LumenConfig.from_args(args).quant_config.recipe == "mxfp4"
    with pytest.raises(ValueError, match="Conflicting"):
        constructor(model_name_or_path="test-model", output_dir="test-output",
                    linear_quant_format="mxfp4", linear_fp8_format="hybrid")


def test_verl_config_preserves_generic_fields():
    args = from_verl_config({"lumen": {"linear_quant_format": "mxfp4", "linear_quant_block_size": 32}})
    assert LumenConfig.from_args(args).quant_config.block_size == 32
    assert mxfp4_enabled(args)


def test_public_enable_reaches_native_contract_with_per_layer_managers(monkeypatch):
    from lumen.quantize import native

    class NativeContractDouble(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(32, 32))
            self.scaling_type = "none"

        def enable_fp8(self, scaling_manager, scaling_type, fp8_dtype, block_size):
            self.scaling_manager = scaling_manager
            self.scaling_type = scaling_type
            self.block_size = block_size

        def forward(self, x):
            return x

    monkeypatch.setattr(native, "is_native_linear", lambda module: isinstance(module, NativeContractDouble))
    monkeypatch.delenv("LUMEN_WEIGHT_QUANT_ONCE", raising=False)
    model = nn.Module()
    model.layers = nn.ModuleList([NativeContractDouble() for _ in range(4)])
    model.lm_head = NativeContractDouble()
    forward = model.layers[1].forward
    cfg = LumenConfig(format="mxfp4", first_last_layers_bf16=True, num_layers=4)
    cfg.enable(model, backend="triton")
    assert [layer.scaling_type for layer in model.layers] == ["none", "mxfp4", "mxfp4", "none"]
    assert model.lm_head.scaling_type == "none"
    assert model.layers[1].forward == forward
    assert model.layers[1].block_size == 32
    assert model.layers[1].scaling_manager.config == cfg.quant_config
    assert model.layers[1].scaling_manager is not model.layers[2].scaling_manager


@pytest.mark.parametrize("flags", [["--mode", "mxfp4"], ["--linear-quant-format", "mxfp4"], ["--linear-fp4"]])
def test_qwen3_modes_and_generic_cli_resolve_equivalently(flags):
    from examples.qwen3.train_qwen3_fsdp import parse_args

    args = parse_args(["--model-name-or-path", "test", "--train-data-path", "test.jsonl", "--task", "pretrain", *flags])
    cfg = LumenConfig.from_args(args)
    assert cfg.quant_config.recipe == "mxfp4"
    assert cfg.quant_config.block_size == 32
    assert args.first_last_layers_bf16
    assert args.lora_rank == 0


@pytest.mark.parametrize("flags", [
    ["--mode", "bf16", "--linear-quant-scaling", "dynamic"],
    ["--mode", "mxfp4", "--linear-fp8-format", "hybrid"],
    ["--mode", "fp8_blockwise2d", "--linear-quant-format", "mxfp4"],
])
def test_qwen3_explicit_preset_conflicts_are_rejected(flags):
    from examples.qwen3.train_qwen3_fsdp import parse_args

    with pytest.raises(ValueError, match="conflicts"):
        parse_args(["--model-name-or-path", "test", "--train-data-path", "test.jsonl", *flags])
