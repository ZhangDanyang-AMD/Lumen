###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""LLaMA 3.1 pretraining components for Transformer Light.

Two training backends are available:

- **Megatron** (``transformer_light.models.llama31.megatron``)
  Pretraining using Megatron-LM-AMD with TP/PP/CP/VP/SP parallelism,
  Transformer Light attention (AITER / Triton / FP8), FP8 hybrid training.

- **FSDP** (``transformer_light.models.llama31.fsdp``)
  Pretraining using PyTorch FSDP + HuggingFace LlamaForCausalLM,
  with LoRA (PEFT), FP8 training, and a standard PyTorch training loop.

Both backends share the same :class:`PretrainTextDataset`.

For backward compatibility, the Megatron APIs are re-exported at this level::

    from transformer_light.models.llama31 import tl_gpt_builder  # Megatron
"""

from transformer_light.models.llama31.dataset import PretrainTextDataset

from transformer_light.models.llama31.megatron import (
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
    # Shared
    "PretrainTextDataset",
    # Megatron (re-exports)
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
