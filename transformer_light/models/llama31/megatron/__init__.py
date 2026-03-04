###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Megatron-LM-AMD backend for LLaMA 3.1 pretraining."""

from transformer_light.models.llama31.megatron.pretrain import (
    add_pretrain_args,
    apply_fp8_training,
    apply_lora,
    forward_step,
    get_batch,
    loss_func,
    reset_fp8_state,
    tl_gpt_builder,
    train_valid_test_datasets_provider,
)

__all__ = [
    "add_pretrain_args",
    "apply_fp8_training",
    "apply_lora",
    "forward_step",
    "get_batch",
    "loss_func",
    "reset_fp8_state",
    "tl_gpt_builder",
    "train_valid_test_datasets_provider",
]
