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

import logging

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

from lumen.models.fsdp import apply_fp8_training, apply_lora

logger = logging.getLogger(__name__)


def _rank0_print(msg: str) -> None:
    try:
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass
    logger.info(msg)


def _has_lumen_optimizations(args) -> bool:
    if getattr(args, "linear_fp8", False):
        return True
    if getattr(args, "lumen_norm", False):
        return True
    if getattr(args, "lumen_fp8_attn", "none") != "none":
        return True
    if getattr(args, "lumen_fp8_activation_store", False):
        return True
    if getattr(args, "lumen_fp8_param_gather", False):
        return True
    return False


def build_fsdp_actor(args, torch_dtype=torch.bfloat16) -> torch.nn.Module:
    """Build the trainable actor model with Lumen optimizations.

    Order: load -> gradient_checkpointing -> LoRA -> Lumen FP8/norm/attn
    """
    _rank0_print(f"> Building Lumen FSDP actor: {args.model_name_or_path}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
    )

    if getattr(args, "gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    if getattr(args, "lora_rank", 0) > 0:
        model = apply_lora(model, args)

    if _has_lumen_optimizations(args):
        apply_fp8_training(model, args)

    _rank0_print(f"> Lumen FSDP actor ready (fp8={getattr(args, 'linear_fp8', False)}, "
                 f"lora={getattr(args, 'lora_rank', 0)})")
    return model


def build_fsdp_reference(args, torch_dtype=torch.bfloat16) -> torch.nn.Module:
    """Build the frozen reference model with optional Lumen FP8."""
    _rank0_print(f"> Building Lumen FSDP reference: {args.model_name_or_path}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
    )

    if _has_lumen_optimizations(args):
        apply_fp8_training(model, args)

    model.eval()
    _rank0_print("> Lumen FSDP reference ready (frozen)")
    return model


def build_fsdp_critic(args, torch_dtype=torch.bfloat16) -> torch.nn.Module:
    """Build the critic model for PPO with optional Lumen FP8.

    For GRPO (beta=0), the critic is not used.
    """
    reward_path = getattr(args, "reward_model_name_or_path", None) or args.model_name_or_path

    _rank0_print(f"> Building Lumen FSDP critic: {reward_path}")

    model = AutoModelForSequenceClassification.from_pretrained(
        reward_path,
        num_labels=1,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
    )

    if _has_lumen_optimizations(args):
        apply_fp8_training(model, args)

    model.eval()
    _rank0_print("> Lumen FSDP critic ready")
    return model
