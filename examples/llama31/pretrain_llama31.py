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

    import megatron.training.training as _mt_training
    from megatron.core.enums import ModelType
    from megatron.training import get_args, pretrain, print_rank_0

    from lumen.models.llama31.megatron import (
        add_pretrain_args,
        apply_fp8_training,
        apply_lora,
        enable_fp8_for_parallel_linear,
        forward_step,
        lumen_gpt_builder,
        train_valid_test_datasets_provider,
    )
    from lumen.models.megatron import (
        apply_lumen_post_quant,
        apply_lumen_pre_quant,
        register_fp8_param_optimizer_hook,
    )

    def model_provider(pre_process=True, post_process=True, vp_stage=None):
        args = get_args()
        model = lumen_gpt_builder(args, pre_process, post_process, vp_stage)

        if getattr(args, "lora_rank", 0) > 0:
            apply_lora(model, args)
            if getattr(args, "lora_a2a", False):
                os.environ["LORA_A2A"] = "1"
                print_rank_0("> LoRA A2A communication optimisation enabled")

        # Phase 1: module attributes before quant patching
        apply_lumen_pre_quant(model, args)

        if getattr(args, "linear_fp8", False):
            apply_fp8_training(model, args)
            if getattr(args, "lumen_linear", False):
                scaling_type = getattr(args, "linear_fp8_scaling", "dynamic")
                enable_fp8_for_parallel_linear(model, scaling_type=scaling_type)

        # Phase 2: features requiring ScalingManager
        apply_lumen_post_quant(model, args)

        return model

    # Wrap setup_model_and_optimizer to register FP8 param optimizer hook
    # after Megatron creates the optimizer (we have no other hook point).
    _original_setup = _mt_training.setup_model_and_optimizer

    def _setup_with_fp8_hook(*args, **kwargs):
        model, optimizer, scheduler = _original_setup(*args, **kwargs)
        train_args = get_args()
        if getattr(train_args, "lumen_fp8_param_gather", False) and model:
            target = model[0] if isinstance(model, list) else model
            register_fp8_param_optimizer_hook(target, optimizer)
        return model, optimizer, scheduler

    _mt_training.setup_model_and_optimizer = _setup_with_fp8_hook

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
