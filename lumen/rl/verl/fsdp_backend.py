###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Lumen-aware FSDP role builders for VERL integration.

These functions build HuggingFace models with Lumen optimizations applied,
ready for VERL's FSDP wrapping. They are intentionally free of VERL imports
so they can be tested independently.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

from lumen.config import LumenConfig


def build_fsdp_actor(args, torch_dtype=torch.bfloat16) -> torch.nn.Module:
    """Build the trainable actor model with Lumen optimizations.

    All features (FP8ParamManager, LoRA, Linear FP8, etc.) are
    orchestrated through ``LumenConfig.enable()``.
    """
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
    )

    if getattr(args, "gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    cfg = LumenConfig.from_args(args)
    _manager, model = cfg.enable(model)
    return model


def build_fsdp_reference(args, torch_dtype=torch.bfloat16) -> torch.nn.Module:
    """Build the frozen reference model with optional Lumen FP8.

    Skips LoRA and FP8ParamManager — reference only gets Linear FP8, norms, etc.
    """
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
    )

    cfg = LumenConfig.from_args(args)
    cfg_ref = replace(cfg, lora_rank=0, fp8_param_manager=False)
    cfg_ref.enable(model)
    model.eval()
    return model


def build_fsdp_critic(args, torch_dtype=torch.bfloat16) -> torch.nn.Module:
    """Build the critic model for PPO with optional Lumen FP8.

    For GRPO (beta=0), the critic is not used.
    Skips LoRA and FP8ParamManager.
    """
    reward_path = getattr(args, "reward_model_name_or_path", None) or args.model_name_or_path

    model = AutoModelForSequenceClassification.from_pretrained(
        reward_path,
        num_labels=1,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
    )

    cfg = LumenConfig.from_args(args)
    cfg_ref = replace(cfg, lora_rank=0, fp8_param_manager=False)
    cfg_ref.enable(model)
    model.eval()
    return model
