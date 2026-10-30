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
    register_deepspeed_quant_hooks,
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


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_validate_accepts_supported_zero_stages(stage):
    validate_deepspeed_quant_config(
        LumenConfig(
            format="fp8_e4m3",
            scaling="none" if stage == 3 else "dynamic",
        ),
        _ds_config(stage),
    )


@pytest.mark.parametrize("stage", [0, 4])
def test_validate_rejects_unvalidated_zero_stages(stage):
    with pytest.raises(ValueError, match="ZeRO stages 1, 2, and 3"):
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


def test_validate_rejects_zero3_fp8_until_scaling_state_is_shard_safe():
    with pytest.raises(ValueError, match="ZeRO-3.*BF16 only"):
        validate_deepspeed_quant_config(
            LumenConfig(format="fp8_e4m3", scaling="dynamic"),
            _ds_config(3),
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_deepspeed_lumen_norm_patch_preserves_parameter_identity():
    class LlamaRMSNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))
            self.variance_epsilon = 1e-5

        def forward(self, hidden_states):
            variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
            return (
                hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            ).to(hidden_states.dtype) * self.weight

    model = nn.Sequential(LlamaRMSNorm()).cuda().to(torch.bfloat16)
    weight_id = id(model[0].weight)
    prepare_deepspeed_model(
        model,
        LumenConfig(scaling="none", lumen_norm=True),
        _ds_config(),
    )

    assert id(model[0].weight) == weight_id
    model(torch.randn(2, 3, 4, device="cuda", dtype=torch.bfloat16))
    assert model[0]._lumen_norm_calls == 1


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
    model.down_proj._quant_enabled = True
    model.down_proj._quant_scaling_type = "dynamic"
    inputs = torch.randn(2, 3, 4)
    expected = model(inputs)
    parameter_ids = [id(param) for param in model.parameters()]

    LumenConfig._patch_hf_mlp(model)
    actual = model(inputs)

    torch.testing.assert_close(actual, expected)
    assert [id(param) for param in model.parameters()] == parameter_ids
    assert model._lumen_fused_mlp_patched is True
    assert model._lumen_fused_mlp_calls == 1


def test_hf_mlp_patch_feeds_fused_fp8_activation_cache():
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
    model.down_proj._quant_enabled = True
    model.down_proj._quant_scaling_type = "dynamic"
    with (
        patch.dict("os.environ", {"LUMEN_FUSED_SWIGLU_QUANT": "1"}),
        patch(
            "lumen.models._swiglu_fp8_fuse.try_fused_swiglu_fp8",
        ) as fused_quant,
    ):
        LumenConfig._patch_hf_mlp(model)
        model(torch.randn(2, 3, 4))

    fused_quant.assert_called_once()


def test_hf_mlp_patch_skips_dynamic_bridge_for_delayed_scaling():
    class LlamaMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(4, 8, bias=False)
            self.up_proj = nn.Linear(4, 8, bias=False)
            self.down_proj = nn.Linear(8, 4, bias=False)
            self.down_proj._quant_enabled = True
            self.down_proj._quant_scaling_type = "delayed"
            self.act_fn = F.silu

    model = LlamaMLP()
    with (
        patch.dict("os.environ", {"LUMEN_FUSED_SWIGLU_QUANT": "1"}),
        patch(
            "lumen.models._swiglu_fp8_fuse.try_fused_swiglu_fp8",
        ) as fused_quant,
    ):
        LumenConfig._patch_hf_mlp(model)
        model(torch.randn(2, 3, 4))

    fused_quant.assert_not_called()


def test_fused_mlp_is_noop_for_non_hf_model():
    model = nn.Sequential(nn.Linear(4, 4))
    LumenConfig._patch_hf_mlp(model)


def test_shared_input_quant_cache_rejects_different_input():
    from lumen.quantize import _shared_input_quant_get, _shared_input_quant_put

    group = (object(), "layer.self_attn")
    first_input = torch.randn(2, 4)
    second_input = torch.randn(2, 4)
    cached = (torch.empty(0), torch.empty(0))
    _shared_input_quant_put(group, first_input, cached)

    assert _shared_input_quant_get(group, second_input, consume=False) is None
    assert _shared_input_quant_get(group, first_input, consume=True) is None


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


def test_build_config_adds_zero3_prefetch_settings():
    config = build_deepspeed_config(_args(zero_stage=3), world_size=8)
    zero = config["zero_optimization"]

    assert zero["stage"] == 3
    assert zero["stage3_prefetch_bucket_size"] == 50_000_000
    assert zero["stage3_param_persistence_threshold"] == 100_000


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


def test_registers_step_scale_hook_without_weight_cache():
    manager = MagicMock()
    manager.config.reduce_amax = False
    manager.needs_optimizer_step_hook = True
    engine = SimpleNamespace(
        data_parallel_group=None,
        optimizer=MagicMock(dp_process_group=None),
        module=nn.Linear(2, 2),
    )

    register_deepspeed_quant_hooks(
        engine,
        manager,
        enable_fp8_weight_cache=False,
    )

    manager.enable_fp8_params.assert_not_called()
    manager.register_fp8_optimizer_hook.assert_called_once_with(engine.optimizer)


def test_training_loader_exhaustion_requires_repeat_corpus():
    from lumen.models.llama2.deepspeed.pretrain import DeepSpeedTrainer

    trainer = DeepSpeedTrainer.__new__(DeepSpeedTrainer)
    trainer.args = SimpleNamespace(repeat_corpus=False)
    trainer.train_loader = []

    with pytest.raises(RuntimeError, match="Training corpus exhausted"):
        trainer._next_batch(iter(()))


def test_deepspeed_parser_defaults_to_zero2_and_auto_backend():
    parser = argparse.ArgumentParser()
    add_common_deepspeed_args(parser)
    args = parser.parse_args([])

    assert args.zero_stage == 2
    assert args.linear_backend == "auto"
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
    assert args.micro_batch_size == 2
    assert args.gradient_accumulation_steps == 8
    assert args.max_steps == 50
    assert args.lr == 1e-5
    assert args.min_lr == 0.0
    assert args.lr_warmup_steps == 2
    assert args.hf_attn_patch is False
    assert args.lumen_fused_rope is False
    assert args.lumen_fused_mlp is False


def test_llama31_scratch_model_uses_llama3_rope_scaling():
    from lumen.models.llama31.deepspeed.pretrain import build_model

    args = SimpleNamespace(
        model_name_or_path=None,
        hidden_size=32,
        intermediate_size=64,
        num_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=8192,
        rope_theta=500000.0,
        gradient_checkpointing=False,
    )
    model = build_model(args)

    assert model.config.rope_scaling["rope_type"] == "llama3"
    assert model.config.rope_scaling["factor"] == 8.0
    assert model.config.rope_scaling["original_max_position_embeddings"] == 8192
