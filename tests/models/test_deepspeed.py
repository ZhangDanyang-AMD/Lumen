###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the optional DeepSpeed integration."""

import argparse
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lumen.config import LumenConfig
from lumen.models.deepspeed import (
    add_common_deepspeed_args,
    build_deepspeed_config,
    initialize_deepspeed,
    make_lumen_deepspeed_model_provider,
    prepare_deepspeed_model,
    validate_deepspeed_quant_config,
)


def _ds_config(stage=2, **zero_kwargs):
    return {
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": stage, **zero_kwargs},
    }


def _args(**overrides):
    values = {
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "zero_stage": 2,
        "max_grad_norm": 1.0,
        "log_interval": 5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("stage", [1, 2])
def test_validate_accepts_supported_zero_stages(stage):
    validate_deepspeed_quant_config(
        LumenConfig(format="fp8_e4m3", scaling="dynamic"),
        _ds_config(stage),
    )


@pytest.mark.parametrize("stage", [0, 3])
def test_validate_rejects_unvalidated_zero_stages(stage):
    with pytest.raises(ValueError, match="ZeRO stages 1 and 2"):
        validate_deepspeed_quant_config(
            LumenConfig(format="fp8_e4m3", scaling="dynamic"),
            _ds_config(stage),
        )


def test_validate_rejects_offload_and_weight_cache():
    with pytest.raises(ValueError, match="offload"):
        validate_deepspeed_quant_config(
            LumenConfig(scaling="dynamic"),
            _ds_config(2, offload_optimizer={"device": "cpu"}),
        )

    with pytest.raises(ValueError, match="fp8_weight_cache"):
        validate_deepspeed_quant_config(
            LumenConfig(scaling="dynamic", fp8_weight_cache=True),
            _ds_config(),
        )


def test_validate_rejects_unvalidated_quant_recipe():
    with pytest.raises(ValueError, match="dynamic.*delayed.*blockwise2d"):
        validate_deepspeed_quant_config(
            LumenConfig(format="fp8_e4m3", scaling="per_token"),
            _ds_config(),
        )


def test_validate_accepts_dynamic_e4m3_recipe():
    validate_deepspeed_quant_config(
        LumenConfig(format="fp8_e4m3", scaling="dynamic"),
        _ds_config(),
    )


def test_validate_accepts_hybrid_delayed():
    validate_deepspeed_quant_config(
        LumenConfig(
            format="hybrid",
            scaling="delayed",
            history_len=1024,
        ),
        _ds_config(),
    )


def test_validate_rejects_e4m3_delayed():
    with pytest.raises(ValueError, match="requires format='hybrid'"):
        validate_deepspeed_quant_config(
            LumenConfig(format="fp8_e4m3", scaling="delayed"),
            _ds_config(),
        )


def test_validate_accepts_supported_bf16_patches():
    validate_deepspeed_quant_config(
        LumenConfig(scaling="none", fused_mlp=True, lumen_linear=True),
        _ds_config(),
    )


def test_validate_rejects_bf16_linear_with_fp8_linear():
    with pytest.raises(ValueError, match="BF16 Linear path"):
        validate_deepspeed_quant_config(
            LumenConfig(scaling="dynamic", lumen_linear=True),
            _ds_config(),
        )


def test_model_provider_uses_lumen_config_enable():
    model = nn.Sequential(nn.Linear(4, 4))
    config = LumenConfig(scaling="none")
    provider = make_lumen_deepspeed_model_provider(
        lambda: model,
        config,
        _ds_config(),
    )

    manager, prepared = provider()

    assert manager is None
    assert prepared is model
    assert prepared._lumen_config is config


def test_hf_mlp_patch_preserves_parameters_and_matches_reference():
    class LlamaMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(4, 8, bias=False)
            self.up_proj = nn.Linear(4, 8, bias=False)
            self.down_proj = nn.Linear(8, 4, bias=False)
            self.act_fn = F.silu

        def forward(self, hidden_states):
            return self.down_proj(
                self.act_fn(self.gate_proj(hidden_states))
                * self.up_proj(hidden_states)
            )

    model = LlamaMLP()
    inputs = torch.randn(2, 3, 4)
    expected = model(inputs)
    parameter_ids = [id(param) for param in model.parameters()]

    LumenConfig._patch_hf_mlp(model)
    actual = model(inputs)

    torch.testing.assert_close(actual, expected)
    assert [id(param) for param in model.parameters()] == parameter_ids
    assert model._lumen_fused_mlp_patched is True
    assert model._lumen_fused_mlp_calls == 1


def test_prepare_forwards_strict_aiter_backend():
    model = nn.Linear(4, 4)
    config = LumenConfig(scaling="none")
    with patch.object(config, "enable", wraps=config.enable) as enable:
        prepare_deepspeed_model(
            model,
            config,
            _ds_config(),
            backend="aiter",
        )

    assert enable.call_args.kwargs["backend"] == "aiter"


def test_build_config_calculates_global_batch_and_bf16():
    config = build_deepspeed_config(_args(), world_size=8)

    assert config["train_batch_size"] == 64
    assert config["train_micro_batch_size_per_gpu"] == 2
    assert config["gradient_accumulation_steps"] == 4
    assert config["zero_optimization"]["stage"] == 2
    assert config["bf16"]["enabled"] is True
    assert config["fp16"]["enabled"] is False


def test_initialize_deepspeed_passes_external_optimizer_and_scheduler():
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = MagicMock()
    engine = SimpleNamespace(data_parallel_group=None, optimizer=optimizer)
    initialize = MagicMock(return_value=(engine, optimizer, None, scheduler))
    fake_deepspeed = types.ModuleType("deepspeed")
    fake_deepspeed.initialize = initialize

    with (
        patch.dict(sys.modules, {"deepspeed": fake_deepspeed}),
        patch("lumen.models.deepspeed.dist.is_initialized", return_value=False),
    ):
        result = initialize_deepspeed(
            model,
            _args(),
            optimizer=optimizer,
            scheduler=scheduler,
        )

    assert result == (engine, optimizer, scheduler)
    assert initialize.call_args.kwargs["optimizer"] is optimizer
    assert initialize.call_args.kwargs["lr_scheduler"] is scheduler


def test_deepspeed_parser_defaults_to_zero2_and_aiter():
    parser = argparse.ArgumentParser()
    add_common_deepspeed_args(parser)
    args = parser.parse_args([])

    assert args.zero_stage == 2
    assert args.linear_backend == "aiter"
    assert args.linear_fp8 is False
    assert args.gradient_checkpointing is True


def test_llama31_deepspeed_parser_uses_8b_defaults():
    from lumen.models.llama31.deepspeed.pretrain import get_args

    argv = [
        "pretrain_llama31.py",
        "--tokenizer-name-or-path",
        "/tmp/tokenizer",
        "--train-data-path",
        "/tmp/train.jsonl",
    ]
    with patch.object(sys, "argv", argv):
        args = get_args()

    assert args.model_label == "LLaMA3.1"
    assert args.hidden_size == 4096
    assert args.intermediate_size == 14336
    assert args.num_layers == 32
    assert args.num_attention_heads == 32
    assert args.num_key_value_heads == 8
    assert args.vocab_size == 128256
    assert args.seq_length == 8192
    assert args.max_position_embeddings == 131072
    assert args.rope_theta == 500000.0
    assert args.zero_stage == 2
    assert args.hf_attn_patch is True
    assert args.lumen_fused_rope is True
    assert args.lumen_fused_mlp is True
