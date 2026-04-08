###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Model builders for the TRL + Lumen FSDP integration."""

from dataclasses import replace

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

from lumen.config import LumenConfig

__all__ = [
    "build_actor_model",
    "build_reference_model",
    "build_reward_model",
]


def _load_causal_lm(model_name_or_path: str):
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def _maybe_enable_gradient_checkpointing(model, args):
    if getattr(args, "gradient_checkpointing", True) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    return model


def build_actor_model(args):
    """Build the trainable GRPO actor model.

    All Lumen features (FP8ParamManager, LoRA, Linear FP8, etc.) are
    orchestrated through a single ``LumenConfig.enable()`` call.
    """
    model = _load_causal_lm(args.model_name_or_path)
    model = _maybe_enable_gradient_checkpointing(model, args)
    cfg = LumenConfig.from_args(args)
    _manager, model = cfg.enable(model)
    return model


def build_reference_model(args):
    """Build the frozen causal LM reference model.

    Reference models skip LoRA and FP8ParamManager — they only get
    Linear FP8, norm patching, etc.
    """
    model = _load_causal_lm(args.model_name_or_path)
    cfg = LumenConfig.from_args(args)
    cfg_ref = replace(cfg, lora_rank=0, fp8_param_manager=False)
    cfg_ref.enable(model)
    model.eval()
    return model


def build_reward_model(args):
    """Build the sequence-classification reward model.

    Reward models skip LoRA and FP8ParamManager.
    """
    reward_path = getattr(args, "reward_model_name_or_path", None) or args.model_name_or_path
    model = AutoModelForSequenceClassification.from_pretrained(
        reward_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    cfg = LumenConfig.from_args(args)
    cfg_ref = replace(cfg, lora_rank=0, fp8_param_manager=False)
    cfg_ref.enable(model)
    model.eval()
    return model
