"""Shared FP8/MXFP4 configuration and CPU model patching (no GPU kernels)."""

import argparse
from dataclasses import asdict, fields
from unittest import mock

import pytest
import torch
from torch import nn

from lumen.config import LumenConfig, _ARG_MAP
from lumen.models.training_contract import add_fsdp_fp8_contract_args
from lumen.quantize import QuantConfig, QuantFormat, ScalingType


def _assert_mxfp4(cfg):
    assert cfg.format == "mxfp4"
    qcfg = cfg.quant_config
    assert qcfg.format == QuantFormat.MXFP4
    assert qcfg.scaling == ScalingType.BLOCKWISE
    assert qcfg.block_size == 32
    assert qcfg.recipe == "mxfp4"
    assert qcfg.is_quantized


def test_cli_alias_is_not_a_second_model_configuration_field():
    assert "linear_fp4" not in {field.name for field in fields(LumenConfig)}
    assert "linear_fp4" not in _ARG_MAP
    assert {"format", "scaling", "block_size"} <= _ARG_MAP.keys()


@pytest.mark.parametrize(
    "make_config",
    [
        lambda: LumenConfig(format="mxfp4"),
        lambda: LumenConfig(format="mxfp4", scaling="blockwise"),
        lambda: LumenConfig.from_dict({"format": "mxfp4"}),
        lambda: LumenConfig.from_args(argparse.Namespace(linear_fp4=True)),
        lambda: LumenConfig.from_args(
            argparse.Namespace(linear_fp4=True, linear_fp8=False)
        ),
    ],
    ids=["constructor-default", "constructor-explicit", "dict", "cli", "cli-fp8-off"],
)
def test_mxfp4_uses_shared_format_recipe(make_config):
    _assert_mxfp4(make_config())


def test_constructor_dict_and_cli_resolve_to_same_quant_config():
    direct = LumenConfig(format="mxfp4", quantize_output_layer=True)
    restored = LumenConfig.from_dict(asdict(direct))
    cli = LumenConfig.from_args(
        argparse.Namespace(linear_fp4=True, quantize_output_layer=True)
    )
    assert direct == restored
    assert direct.quant_config == restored.quant_config == cli.quant_config


def test_mxfp4_cli_alias_ignores_only_parser_defaults():
    parser = argparse.ArgumentParser()
    add_fsdp_fp8_contract_args(parser)
    cfg = LumenConfig.from_args(parser.parse_args(["--linear-fp4"]))
    _assert_mxfp4(cfg)
    assert cfg.block_size is None
    with pytest.raises(ValueError, match="Conflicting"):
        LumenConfig.from_args(parser.parse_args([
            "--linear-fp4", "--linear-fp8-format", "hybrid",
        ]))
    with pytest.raises(ValueError, match="32-element blocks"):
        _ = LumenConfig.from_args(parser.parse_args([
            "--linear-fp4", "--linear-fp8-block-size", "64",
        ])).quant_config


def test_explicit_bad_mxfp4_block_size_is_not_silently_replaced():
    cfg = LumenConfig(format="mxfp4", block_size=128)
    assert cfg.block_size == 128
    with pytest.raises(ValueError, match="32-element blocks"):
        _ = cfg.quant_config


_UNSUPPORTED_MXFP4_OPTIONS = [
    ({"quantize_activation": False}, "quantize_activation=True"),
    ({"fp8_wgrad": False}, "fp8_wgrad=True"),
    ({"cache_frozen_weight": True}, "FP8-only options"),
    ({"bpreshuffle_gemm": True}, "FP8-only options"),
]


@pytest.mark.parametrize("options,message", _UNSUPPORTED_MXFP4_OPTIONS)
def test_unsupported_mxfp4_options_fail_in_shared_quant_config(options, message):
    with pytest.raises(ValueError, match=message):
        QuantConfig.from_str(format="mxfp4", **options)
    with pytest.raises(ValueError, match=message):
        QuantConfig(format=QuantFormat.MXFP4, **options)
    with pytest.raises(ValueError, match=message):
        _ = LumenConfig.from_dict({"format": "mxfp4", **options}).quant_config


@pytest.mark.parametrize(
    "options,message",
    [({"block_size": 128}, "32-element blocks"), *_UNSUPPORTED_MXFP4_OPTIONS],
)
def test_mxfp4_validation_precedes_model_mutation(options, message):
    cfg = LumenConfig(format="mxfp4", lumen_norm=True, **options)
    model = nn.Linear(32, 32)
    with mock.patch.object(cfg, "_patch_norms") as patch_norms:
        with pytest.raises(ValueError, match=message):
            cfg.enable(model)
    patch_norms.assert_not_called()
    assert not hasattr(model, "_quant_enabled")


@pytest.mark.parametrize("options,_message", _UNSUPPORTED_MXFP4_OPTIONS)
def test_fp8_options_and_disabled_mxfp4_remain_valid(options, _message):
    assert LumenConfig(**options).quant_config.is_quantized
    cfg = LumenConfig(format="mxfp4", scaling="none", **options)
    assert not cfg.quant_config.is_quantized


def test_mxfp4_rejects_fp8_storage_before_patching():
    cfg = LumenConfig(format="mxfp4", fp8_param_manager=True)
    with mock.patch.object(cfg, "_apply_fp8_param_manager") as patch_storage:
        with pytest.raises(ValueError, match="fp8_param_manager cannot be combined"):
            cfg.enable(nn.Linear(32, 32))
    patch_storage.assert_not_called()


@pytest.mark.parametrize(
    "flag", ["fp8_activation_store", "fp8_weight_cache", "fp8_param_gather"]
)
def test_mxfp4_rejects_fp8_storage_flags_before_model_mutation(monkeypatch, flag):
    monkeypatch.delenv("LUMEN_WEIGHT_QUANT_ONCE", raising=False)
    cfg = LumenConfig(format="mxfp4", lumen_norm=True, **{flag: True})
    model = nn.Linear(32, 32)
    original_forward = model.forward
    with mock.patch.object(cfg, "_patch_norms") as patch_norms:
        with pytest.raises(ValueError, match=flag):
            cfg.enable(model)
    patch_norms.assert_not_called()
    assert model.forward == original_forward
    assert not hasattr(model, "_quant_enabled")


def test_mxfp4_rejects_inherited_fp8_param_cache_before_patching(monkeypatch):
    monkeypatch.setenv("LUMEN_WEIGHT_QUANT_ONCE", "1")
    cfg = LumenConfig(format="mxfp4", lumen_norm=True)
    with mock.patch.object(cfg, "_patch_norms") as patch_norms:
        with pytest.raises(ValueError, match="LUMEN_WEIGHT_QUANT_ONCE=1"):
            cfg.enable(nn.Linear(32, 32))
    patch_norms.assert_not_called()


@pytest.mark.parametrize(
    "flag", ["fp8_activation_store", "fp8_weight_cache", "fp8_param_gather"]
)
def test_disabled_mxfp4_does_not_reject_inactive_storage_flags(monkeypatch, flag):
    monkeypatch.delenv("LUMEN_WEIGHT_QUANT_ONCE", raising=False)
    cfg = LumenConfig(format="mxfp4", scaling="none", **{flag: True})
    model = nn.Linear(32, 32)
    manager, patched = cfg.enable(model)
    assert manager is None
    assert patched is model
    assert not hasattr(model, "_quant_enabled")


def test_fp8_and_fp4_cli_selectors_remain_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        LumenConfig.from_args(argparse.Namespace(linear_fp8=True, linear_fp4=True))


def test_existing_fp8_defaults_and_positional_interface_are_unchanged():
    cfg = LumenConfig()
    assert cfg.quant_config.format == QuantFormat.FP8_E4M3
    assert cfg.quant_config.scaling == ScalingType.DELAYED
    assert cfg.quant_config.block_size == 128
    cfg = LumenConfig("hybrid", "dynamic")
    assert cfg.quant_config.format == QuantFormat.HYBRID
    assert cfg.quant_config.scaling == ScalingType.DYNAMIC


def test_existing_fp8_parser_and_disabled_paths_are_unchanged():
    parser = argparse.ArgumentParser()
    add_fsdp_fp8_contract_args(parser)
    cfg = LumenConfig.from_args(parser.parse_args(["--linear-fp8"]))
    assert cfg.quant_config.format == QuantFormat.FP8_E4M3
    assert cfg.quant_config.scaling == ScalingType.DELAYED
    assert cfg.quant_config.block_size == 128
    assert not LumenConfig.from_args(parser.parse_args([])).quant_config.is_quantized
    with pytest.raises(ValueError, match="MXFP4 must be enabled with --linear-fp4"):
        LumenConfig.from_args(
            argparse.Namespace(linear_fp8=True, linear_fp8_format="mxfp4")
        )


@pytest.mark.parametrize("format", ["fp8_e4m3", "mxfp4"])
@pytest.mark.parametrize("quantize_output_layer", [False, True])
def test_enable_applies_shared_model_policy(monkeypatch, format, quantize_output_layer):
    # Model patching is CPU-safe; this does not execute a quantization kernel.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("LUMEN_WEIGHT_QUANT_ONCE", raising=False)
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Linear(32, 32) for _ in range(4)])
    model.lm_head = nn.Linear(32, 32)
    weight = model.layers[1].weight
    old_forward = model.layers[1].forward
    cfg = LumenConfig.from_dict({
        "format": format,
        "quantize_output_layer": quantize_output_layer,
        "first_last_layers_bf16": True,
        "num_layers_at_start_in_bf16": 1,
        "num_layers_at_end_in_bf16": 1,
        "num_layers": 4,
        "quantize_grad": "mxfp4",
    })
    manager, patched = cfg.enable(model, backend="triton")
    assert patched is model
    assert model._lumen_config is cfg
    assert manager.config == cfg.quant_config
    assert manager.config.quantize_grad == "mxfp4"
    for index, layer in enumerate(model.layers):
        assert bool(getattr(layer, "_quant_enabled", False)) == (index in (1, 2))
    assert model.layers[1]._quant_manager is manager
    assert model.layers[1]._quant_scaling_type == cfg.quant_config.recipe
    assert model.layers[1]._quant_backend == "triton"
    assert model.layers[1].forward != old_forward
    assert model.layers[1].weight is weight
    assert bool(getattr(model.lm_head, "_quant_enabled", False)) == quantize_output_layer


@pytest.mark.parametrize("scaling", ["dynamic", "per_token", "blockwise2d"])
def test_mxfp4_refuses_unsupported_scaling(scaling):
    with pytest.raises(ValueError, match="fixed blockwise"):
        _ = LumenConfig(format="mxfp4", scaling=scaling).quant_config


def test_generic_cli_reaches_model_forward(monkeypatch):
    import lumen.quantize as quant

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("LUMEN_WEIGHT_QUANT_ONCE", raising=False)
    parser = argparse.ArgumentParser()
    add_fsdp_fp8_contract_args(parser)
    cfg = LumenConfig.from_args(parser.parse_args(["--linear-quant-format", "mxfp4"]))
    model = nn.Linear(32, 32)
    manager, _ = cfg.enable(model, backend="triton")
    captured = {}

    def record(x, weight, bias, **kwargs):
        captured.update(kwargs)
        return x

    monkeypatch.setattr(quant, "_mxfp4_cached_weight", lambda *a, **kw: (None, None))
    monkeypatch.setattr(quant, "quantized_linear", record)
    model(torch.zeros(2, 32))
    assert captured["scaling_type"] == "mxfp4"
    assert captured["block_size"] == 32
    assert captured["scaling_manager"] is manager
    assert manager.config == cfg.quant_config






def test_mxfp4_scale_query_has_consistent_return_contract():
    from lumen.quantize import ScalingManager

    manager = ScalingManager(QuantConfig.from_str("mxfp4"))
    tensor = torch.zeros(2, 32)
    assert manager.get_scale("activation", tensor) is None
    assert manager.get_scale("activation", tensor, return_amax=True) == (None, None)


def test_imported_fp8_cache_state_is_checked_even_after_environment_is_cleared(monkeypatch):
    import lumen.ops.quantize.linear as linear

    monkeypatch.setattr(linear, "_WEIGHT_QUANT_ONCE", True)
    monkeypatch.delenv("LUMEN_WEIGHT_QUANT_ONCE", raising=False)
    model = nn.Linear(32, 32)
    with pytest.raises(ValueError, match="imported state"):
        LumenConfig(format="mxfp4").enable(model)
    assert not hasattr(model, "_quant_enabled")


def test_disabled_mxfp4_recipe_does_not_patch_model():
    model = nn.Linear(32, 32)
    cfg = LumenConfig(format="mxfp4", scaling="none")
    assert cfg.quant_config.block_size == 32
    manager, patched = cfg.enable(model)
    assert manager is None
    assert patched is model
    assert not hasattr(model, "_quant_enabled")
