# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""LLaMA 3.1 Pretraining — unified entry point.

Supports two backends, selected via ``--backend``:

- **megatron** — Megatron-LM-AMD ``pretrain`` driver with TP/PP/CP/VP/SP.
  Requires ``pip install megatron-lm``.
- **fsdp** — PyTorch FSDP + HuggingFace Transformers.
  Requires ``pip install transformers peft``.

Usage:
    # Megatron backend (default)
    torchrun --nproc_per_node=8 pretrain_llama31.py --backend megatron <args>

    # FSDP backend
    torchrun --nproc_per_node=8 pretrain_llama31.py --backend fsdp <args>

See run_pretrain.sh for a complete launch example.
"""

import sys

from lumen.models.utils import peek_backend


def _run_megatron():
    # Import TL megatron FIRST so _install_fused_layer_norm_patch() runs before
    # megatron.training loads TransformerBlock/FusedLayerNorm.
    import os

    from megatron.core.enums import ModelType
    from megatron.training import get_args, pretrain, print_rank_0

    from lumen.models.llama31.megatron import (
        add_pretrain_args,
        apply_fp8_training,
        apply_lora,
        forward_step,
        tl_gpt_builder,
        train_valid_test_datasets_provider,
    )

    def model_provider(pre_process=True, post_process=True, vp_stage=None):
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

    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_pretrain_args,
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
    )


def _run_fsdp():
    from lumen.models.llama31.fsdp import FSDPTrainer, get_args

    args = get_args()
    trainer = FSDPTrainer(args)
    trainer.train()


if __name__ == "__main__":
    backend = peek_backend()

    if backend == "megatron":
        _run_megatron()
    elif backend == "fsdp":
        _run_fsdp()
    else:
        print(f"ERROR: Unknown backend '{backend}'. Use 'megatron' or 'fsdp'.")
        sys.exit(1)
