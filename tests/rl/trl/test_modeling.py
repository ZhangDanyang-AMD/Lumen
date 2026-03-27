###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for TRL model builders."""

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LUMEN_DIR = REPO_ROOT / "lumen"
RL_DIR = LUMEN_DIR / "rl"
TRL_DIR = RL_DIR / "trl"


class _DummyModel:
    def __init__(self, kind: str):
        self.kind = kind
        self.events = []
        self.load_kwargs = {}
        self.gradient_checkpointing_kwargs = None
        self.is_eval = False

    def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs
        self.events.append("gradient_checkpointing")

    def eval(self):
        self.is_eval = True
        self.events.append("eval")
        return self


class _FakeAutoModelForCausalLM:
    calls = []

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        model = _DummyModel("causal_lm")
        model.load_kwargs = kwargs
        cls.calls.append((model_name_or_path, kwargs, model))
        return model


class _FakeAutoModelForSequenceClassification:
    calls = []

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        model = _DummyModel("sequence_classification")
        model.load_kwargs = kwargs
        cls.calls.append((model_name_or_path, kwargs, model))
        return model


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_modeling_module():
    torch_mod = types.ModuleType("torch")
    torch_mod.bfloat16 = "bfloat16"
    torch_nn_mod = types.ModuleType("torch.nn")
    torch_nn_mod.Module = object
    torch_mod.nn = torch_nn_mod
    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = torch_nn_mod

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoModelForCausalLM = _FakeAutoModelForCausalLM
    transformers_mod.AutoModelForSequenceClassification = _FakeAutoModelForSequenceClassification
    sys.modules["transformers"] = transformers_mod

    def _apply_lora(model, args):
        model.events.append("lora")
        model.lora_args = (args.lora_rank, args.lora_alpha, args.lora_dropout)
        return model

    def _apply_fp8_training(model, args):
        model.events.append("fp8")
        model.fp8_enabled = args.linear_fp8

    fsdp_mod = types.ModuleType("lumen.models.fsdp")
    fsdp_mod.apply_fp8_training = _apply_fp8_training
    fsdp_mod.apply_lora = _apply_lora
    sys.modules["lumen.models.fsdp"] = fsdp_mod

    lumen_pkg = sys.modules.setdefault("lumen", types.ModuleType("lumen"))
    lumen_pkg.__path__ = [str(LUMEN_DIR)]

    models_pkg = sys.modules.setdefault("lumen.models", types.ModuleType("lumen.models"))
    models_pkg.__path__ = [str(LUMEN_DIR / "models")]

    rl_pkg = sys.modules.setdefault("lumen.rl", types.ModuleType("lumen.rl"))
    rl_pkg.__path__ = [str(RL_DIR)]

    trl_pkg = sys.modules.setdefault("lumen.rl.trl", types.ModuleType("lumen.rl.trl"))
    trl_pkg.__path__ = [str(TRL_DIR)]

    return _load_module("lumen.rl.trl.modeling", TRL_DIR / "modeling.py")


class TestTrlModelBuilders:
    def test_build_actor_model_orders_gradient_checkpointing_lora_fp8(self):
        mod = _load_modeling_module()
        args = types.SimpleNamespace(
            model_name_or_path="meta-llama/Llama-2-70b-hf",
            lora_rank=8,
            lora_alpha=32.0,
            lora_dropout=0.1,
            linear_fp8=True,
            gradient_checkpointing=True,
        )

        model = mod.build_actor_model(args)

        assert model.events == ["gradient_checkpointing", "lora", "fp8"]
        assert model.gradient_checkpointing_kwargs == {"use_reentrant": False}
        assert model.load_kwargs["torch_dtype"] == "bfloat16"
        assert model.load_kwargs["attn_implementation"] == "sdpa"

    def test_build_reference_model_skips_lora_but_can_apply_fp8(self):
        mod = _load_modeling_module()
        args = types.SimpleNamespace(
            model_name_or_path="meta-llama/Llama-2-70b-hf",
            lora_rank=8,
            lora_alpha=32.0,
            lora_dropout=0.1,
            linear_fp8=True,
            gradient_checkpointing=True,
        )

        model = mod.build_reference_model(args)

        assert "lora" not in model.events
        assert "fp8" in model.events
        assert model.is_eval is True

    def test_build_reward_model_uses_sequence_classification_without_lora(self):
        mod = _load_modeling_module()
        args = types.SimpleNamespace(
            model_name_or_path="meta-llama/Llama-2-70b-hf",
            reward_model_name_or_path="reward-model",
            lora_rank=8,
            lora_alpha=32.0,
            lora_dropout=0.1,
            linear_fp8=True,
            gradient_checkpointing=True,
        )

        model = mod.build_reward_model(args)

        assert model.kind == "sequence_classification"
        assert "lora" not in model.events
        assert "fp8" in model.events
        assert model.load_kwargs["num_labels"] == 1
