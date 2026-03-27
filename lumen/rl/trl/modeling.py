###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Model builders for the TRL + Lumen FSDP integration."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

from lumen.models.fsdp import (
    apply_fp8_training,
    apply_lora,
)

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


def _maybe_apply_fp8(model, args):
    if getattr(args, "linear_fp8", False):
        apply_fp8_training(model, args)
    return model


def build_actor_model(args):
    """Build the trainable GRPO actor model."""

    model = _load_causal_lm(args.model_name_or_path)
    model = _maybe_enable_gradient_checkpointing(model, args)
    if getattr(args, "lora_rank", 0) > 0:
        model = apply_lora(model, args)
    model = _maybe_apply_fp8(model, args)
    return model


def build_reference_model(args):
    """Build the frozen causal LM reference model."""

    model = _load_causal_lm(args.model_name_or_path)
    # v1 keeps LoRA exclusive to the actor so KL/reference math stays simple.
    model = _maybe_apply_fp8(model, args)
    model.eval()
    return model


def build_reward_model(args):
    """Build the sequence-classification reward model."""

    reward_path = getattr(args, "reward_model_name_or_path", None) or args.model_name_or_path
    model = AutoModelForSequenceClassification.from_pretrained(
        reward_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    # v1 keeps LoRA exclusive to the actor so reward evaluation stays stable.
    model = _maybe_apply_fp8(model, args)
    model.eval()
    return model
