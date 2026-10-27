###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""LLaMA2 model components for Lumen.

Three training backends are available:

- **Megatron** (``lumen.models.llama2.megatron``)
  Full-featured SFT using Megatron-LM-AMD with TP/PP/CP/VP/SP parallelism,
  Lumen attention, and the ``pretrain`` driver.

- **FSDP** (``lumen.models.llama2.fsdp``)
  SFT using PyTorch FSDP + HuggingFace LlamaForCausalLM, with LoRA (PEFT),
  FP8 training, and a standard PyTorch training loop.

- **DeepSpeed** (``lumen.models.llama2.deepspeed``)
  Causal-LM pretraining using ZeRO-1/2 and the shared ``LumenConfig`` path.

All backends share the same :class:`LLaMA2SFTDataset` (packed sequences,
answer-only loss masking, jsonl format).

For backward compatibility, the Megatron APIs are re-exported at this level::

    from lumen.models.llama2 import lumen_gpt_builder  # Megatron
"""

import importlib

from lumen.models.llama2.dataset import LLaMA2SFTDataset

_MEGATRON_EXPORTS = {
    "add_finetune_args",
    "apply_fp8_training",
    "apply_lora",
    "forward_step",
    "get_batch",
    "loss_func",
    "lumen_gpt_builder",
    "reset_fp8_state",
    "train_valid_test_datasets_provider",
}


def __getattr__(name: str):
    if name in _MEGATRON_EXPORTS:
        module = importlib.import_module("lumen.models.llama2.megatron")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in ("fsdp", "megatron", "deepspeed"):
        return importlib.import_module(f"lumen.models.llama2.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Shared
    "LLaMA2SFTDataset",
    # Megatron (backward-compatible, lazily loaded re-exports)
    *_MEGATRON_EXPORTS,
]
