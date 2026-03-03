# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""LLaMA2 Supervised Fine-Tuning — entry point.

All reusable components (model builder, dataset, loss, forward step, CLI args)
live in ``transformer_light.models.llama2.sft``.  This script is a thin
wrapper that wires them into Megatron-LM-AMD's ``pretrain`` driver.

Prerequisites:
    pip install megatron-lm          # or: pip install git+https://github.com/ROCm/Megatron-LM.git
    pip install transformer_light    # or editable install from source

Usage:
    torchrun --nproc_per_node=8 finetune_llama2.py <args>

See run_finetune.sh for a complete launch example.
"""

import os

from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain, print_rank_0

from transformer_light.models.llama2 import (
    add_finetune_args,
    apply_fp8_training,
    apply_lora,
    forward_step,
    tl_gpt_builder,
    train_valid_test_datasets_provider,
)


def model_provider(pre_process=True, post_process=True, vp_stage=None):
    """Build GPT model with optional LoRA and FP8 training."""
    args = get_args()
    model = tl_gpt_builder(args, pre_process, post_process, vp_stage)

    if getattr(args, "lora_rank", 0) > 0:
        apply_lora(model, args)
        if getattr(args, "lora_a2a", False):
            os.environ["LORA_A2A"] = "1"
            print_rank_0("> LoRA A2A communication optimisation enabled")

    if getattr(args, "fp8_training", False):
        apply_fp8_training(model, args)

    return model


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_finetune_args,
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
    )
