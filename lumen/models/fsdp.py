###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared FSDP training helpers for Lumen models.

This module consolidates FP8 and LoRA helper functions that are common to
all PyTorch-FSDP-based training scripts (LLaMA2 SFT, LLaMA 3.1 pretraining,
etc.).  Model-specific code (model building, trainer class, CLI args) remains
in the per-model subpackages.
"""

import logging

import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


def _rank0_print(msg: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(msg)


# ---------------------------------------------------------------------------
# FP8 quantised training
# ---------------------------------------------------------------------------

def apply_fp8_training(model: nn.Module, args, dp_group=None) -> None:
    """Enable FP8 quantised training via Lumen.

    Args:
        dp_group: Data-parallel process group used for amax reduction.
            If ``None`` and ``args.fp8_reduce_amax`` is true, falls back
            to ``dist.group.WORLD``.
    """
    import lumen.quantize as quant
    from lumen.quantize import (
        AmaxAlgo, QuantConfig, QuantFormat, ScalingType,
    )

    config = QuantConfig(
        format=QuantFormat(args.fp8_format),
        scaling=ScalingType(args.fp8_scaling),
        block_size=args.fp8_block_size,
        amax_algo=AmaxAlgo(args.fp8_amax_algo),
        margin=getattr(args, "fp8_margin", 0),
        reduce_amax=args.fp8_reduce_amax,
        history_len=args.fp8_amax_history,
        quantize_activation=args.fp8_activation,
        quantize_grad=getattr(args, "grad_quant_type", None),
    )

    if dp_group is None and config.reduce_amax and dist.is_initialized():
        dp_group = dist.group.WORLD

    quant.enable(
        model, config=config,
        dp_group=dp_group if config.reduce_amax else None,
    )
    _rank0_print(
        f"> FP8 training enabled (format={args.fp8_format}, "
        f"scaling={args.fp8_scaling}, amax_algo={args.fp8_amax_algo}, "
        f"activation={args.fp8_activation}, grad_quant={config.quantize_grad})"
    )


def reset_fp8_state(model: nn.Module) -> None:
    """Reset FP8 scaling state after warmup.

    Unwraps nested ``.module`` attributes (e.g. from FSDP or DDP wrappers)
    before calling :meth:`apply`.
    """

    def _reset(m):
        if hasattr(m, "fp8_initialized"):
            m.fp8_initialized = False
        if hasattr(m, "_quant_manager"):
            m._quant_manager.reset()
        if hasattr(m, "_tl_scaling_manager"):
            m._tl_scaling_manager.reset()

    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    unwrapped.apply(_reset)
    _rank0_print("> FP8 state reset after warmup")


# ---------------------------------------------------------------------------
# LoRA (via HuggingFace PEFT)
# ---------------------------------------------------------------------------

def apply_lora(model: nn.Module, args) -> nn.Module:
    """Apply LoRA adapters via HuggingFace PEFT and freeze the base model."""
    from peft import LoraConfig, get_peft_model, TaskType

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    _rank0_print(
        f"> LoRA applied (rank={args.lora_rank}, alpha={args.lora_alpha}) — "
        f"trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )
    return model
