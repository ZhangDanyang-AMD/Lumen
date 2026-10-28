###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""LLaMA 3.1 pretraining with DeepSpeed + Lumen."""

import argparse
import logging

import torch
import torch.nn as nn

from lumen.models.deepspeed import add_common_deepspeed_args
from lumen.models.llama2.deepspeed.pretrain import (
    DeepSpeedTrainer as _BaseDeepSpeedTrainer,
)

__all__ = ["DeepSpeedTrainer", "build_model", "get_args"]

logger = logging.getLogger(__name__)


def build_model(args) -> nn.Module:
    """Build a HuggingFace LLaMA 3.1 model."""
    from transformers import LlamaConfig, LlamaForCausalLM

    if args.model_name_or_path:
        model = LlamaForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    else:
        config = LlamaConfig(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
            vocab_size=args.vocab_size,
            max_position_embeddings=args.max_position_embeddings,
            rope_theta=args.rope_theta,
            rms_norm_eps=1e-5,
            attention_bias=False,
            hidden_act="silu",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            use_cache=False,
            tie_word_embeddings=False,
        )
        model = LlamaForCausalLM(config).to(dtype=torch.bfloat16)

    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    return model


class DeepSpeedTrainer(_BaseDeepSpeedTrainer):
    """DeepSpeed trainer using the LLaMA 3.1 model configuration."""

    def __init__(self, args):
        super().__init__(args, model_builder=build_model)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLaMA 3.1 pretraining with DeepSpeed + Lumen",
    )
    parser.add_argument("--backend", choices=["deepspeed"], default="deepspeed")

    model = parser.add_argument_group("model")
    model.add_argument("--model-name-or-path", type=str, default=None)
    model.add_argument("--tokenizer-name-or-path", type=str, required=True)
    model.add_argument("--hidden-size", type=int, default=4096)
    model.add_argument("--intermediate-size", type=int, default=14336)
    model.add_argument("--num-layers", type=int, default=32)
    model.add_argument("--num-attention-heads", type=int, default=32)
    model.add_argument("--num-key-value-heads", type=int, default=8)
    model.add_argument("--vocab-size", type=int, default=128256)
    model.add_argument("--max-position-embeddings", type=int, default=8192)
    model.add_argument("--rope-theta", type=float, default=500000.0)
    model.add_argument("--seq-length", type=int, default=8192)

    data = parser.add_argument_group("data")
    data.add_argument("--train-data-path", type=str, required=True)
    data.add_argument("--val-data-path", type=str, default=None)
    data.add_argument("--train-samples", type=int, default=None)
    data.add_argument("--val-samples", type=int, default=None)
    data.add_argument("--repeat-corpus", action="store_true", default=False)
    data.add_argument("--eval-iters", type=int, default=10)
    data.add_argument("--target-loss", type=float, default=0.0)

    add_common_deepspeed_args(parser)
    parser.set_defaults(
        model_label="LLaMA3.1",
        micro_batch_size=2,
        gradient_accumulation_steps=8,
        max_steps=50,
        lr=1e-5,
        min_lr=0.0,
        lr_warmup_steps=2,
        weight_decay=0.1,
        linear_fp8_format="fp8_e4m3",
        linear_fp8_scaling="dynamic",
        gradient_checkpointing=False,
        hf_attn_patch=False,
        lumen_fused_rope=False,
        lumen_fused_mlp=False,
    )
    return parser.parse_args()
