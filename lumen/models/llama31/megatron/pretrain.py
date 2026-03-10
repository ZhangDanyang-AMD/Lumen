###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""LLaMA 3.1 Pretraining components for Megatron-LM-AMD.

This module provides the building blocks for pretraining LLaMA 3.1 models
using Megatron-LM-AMD as the training backbone and Lumen for
the core dot-product attention (AITER / Triton / FP8).

Features:
    - LLaMA 3.1 8B architecture (GQA, RoPE theta=500000)
    - FP8 hybrid training (weight/activation quantisation via Lumen)
    - FP8 attention (MXFP8 / FP8 blockwise via Lumen)
    - Lumen attention backends: AITER, Triton, Triton-FP8
    - Fine-grained MXFP8 block configuration (6 independent block sizes)
    - Context Parallelism, Tensor Parallelism, Pipeline Parallelism, VP, SP
    - Distributed optimizer with FP8 param gather
    - Cosine annealing LR with warmup
    - Synthetic warmup with FP8 state reset
    - Early stopping based on validation loss target

Example::

    from lumen.models.llama31.megatron import (
        add_pretrain_args,
        forward_step,
        tl_gpt_builder,
        train_valid_test_datasets_provider,
    )
"""

import torch

from megatron.core import tensor_parallel
from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_ltor_masks_and_position_ids,
    is_first_or_last_pipeline_stage,
)
from lumen.models.llama31.dataset import PretrainTextDataset
from lumen.models.utils import safe_add_argument

# Re-export shared symbols so existing callers are not broken.
from lumen.models.megatron import (  # noqa: F401
    apply_fp8_training,
    apply_lora,
    loss_func,
    make_forward_step,
    reset_fp8_state,
    add_common_megatron_args,
)
from lumen.models.megatron import tl_gpt_builder as _tl_gpt_builder_generic

__all__ = [
    "PretrainTextDataset",
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


# ---------------------------------------------------------------------------
# LLaMA 3.1 architecture constants
# ---------------------------------------------------------------------------

LLAMA31_CONFIGS = {
    "8b": {
        "num_layers": 32,
        "hidden_size": 4096,
        "ffn_hidden_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "seq_length": 8192,
        "max_position_embeddings": 131072,
        "rotary_base": 500000,
    },
}


# ---------------------------------------------------------------------------
# Custom GPT builder (thin wrapper with model-specific log message)
# ---------------------------------------------------------------------------

def tl_gpt_builder(args, pre_process, post_process, vp_stage=None, config=None):
    return _tl_gpt_builder_generic(
        args, pre_process, post_process,
        vp_stage=vp_stage, config=config, model_name="LLaMA 3.1",
    )


# ---------------------------------------------------------------------------
# Dataset provider
# ---------------------------------------------------------------------------

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, validation, and test pretraining datasets."""
    args = get_args()

    tokenizer_obj = get_tokenizer()
    is_hf = hasattr(tokenizer_obj, "_tokenizer") and hasattr(
        tokenizer_obj._tokenizer, "encode"
    )
    raw_tokenizer = tokenizer_obj._tokenizer if is_hf else tokenizer_obj

    train_path = args.train_data_path[0] if args.train_data_path else None
    valid_path = args.valid_data_path[0] if args.valid_data_path else None
    test_path = args.test_data_path[0] if args.test_data_path else None

    print_rank_0("> building train, validation, and test pretraining datasets ...")

    train_ds = PretrainTextDataset(
        train_path, args.seq_length, raw_tokenizer, is_hf,
        max_samples=train_val_test_num_samples[0],
    )
    valid_ds = PretrainTextDataset(
        valid_path, args.seq_length, raw_tokenizer, is_hf,
        max_samples=train_val_test_num_samples[1],
    )
    test_ds = PretrainTextDataset(
        test_path, args.seq_length, raw_tokenizer, is_hf,
        max_samples=train_val_test_num_samples[2],
    )

    print_rank_0("> finished creating pretraining datasets ...")
    return train_ds, valid_ds, test_ds


# ---------------------------------------------------------------------------
# Batch construction (standard LM — all tokens contribute)
# ---------------------------------------------------------------------------

def get_batch(data_iterator, vp_stage=None):
    """Generate a batch for pretraining (standard LM, all tokens contribute)."""
    if not is_first_or_last_pipeline_stage(vp_stage):
        return None, None, None, None, None

    args = get_args()

    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_b = tensor_parallel.broadcast_data(
        ["input_ids", "labels"], data, torch.int64
    )

    tokens = data_b["input_ids"].contiguous()
    labels = data_b["labels"].contiguous()

    tokenizer = get_tokenizer()
    if hasattr(tokenizer, "_tokenizer") and hasattr(
        tokenizer._tokenizer, "eos_token_id"
    ):
        eod_token = tokenizer._tokenizer.eos_token_id
    else:
        eod_token = tokenizer.eod

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens, eod_token, eod_token,
        args.reset_position_ids, args.reset_attention_mask,
        args.eod_mask_loss, False,
    )

    batch = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    batch = get_batch_on_this_cp_rank(batch)
    return batch.values()


# ---------------------------------------------------------------------------
# Forward step (bound to pretraining get_batch)
# ---------------------------------------------------------------------------

forward_step = make_forward_step(get_batch, loss_func)


# ---------------------------------------------------------------------------
# Extra CLI arguments
# ---------------------------------------------------------------------------

def add_pretrain_args(parser):
    """Add pretrain-specific arguments."""
    # Pre-register args whose defaults differ from the shared module before
    # calling add_common_megatron_args (safe_add_argument skips duplicates).
    safe_add_argument(parser, "--tl-fp8-quant-type", type=str, default="mxfp8",
                       choices=["fp8_blockwise", "mxfp8"],
                       help="FP8 quantisation type for triton_fp8 backend.")
    safe_add_argument(parser, "--fp8-amax-algo", type=str, default="most_recent",
                       choices=["max", "most_recent"])
    safe_add_argument(parser, "--fp8-amax-history", type=int, default=4)

    add_common_megatron_args(parser)

    ckpt = parser.add_argument_group(title="checkpoint-management")
    ckpt.add_argument("--use-ckpt", action="store_true", default=False,
                       help="Resume from checkpoint.")
    ckpt.add_argument("--save-ckpt", action="store_true", default=False,
                       help="Save checkpoint at end of training.")
    ckpt.add_argument("--resume-from-hf", action="store_true", default=False,
                       help="Checkpoint is a weight-only HuggingFace format.")
    ckpt.add_argument("--continual-ckpt-path", type=str, default=None,
                       help="Path for saving/loading continual checkpoints.")
    ckpt.add_argument("--ckpt-start-step", type=int, default=0,
                       help="Steps already trained in the resumed checkpoint.")
    ckpt.add_argument("--fp8-params", action="store_true", default=False,
                       help="Load model parameters in FP8.")
    ckpt.add_argument("--initial-ckpt-path", type=str, default=None,
                       help="Path to initial checkpoint for resume.")

    mlperf = parser.add_argument_group(title="mlperf")
    mlperf.add_argument("--tag", type=str, default="",
                         help="Optional experiment tag.")
    mlperf.add_argument("--target-log-ppl", type=float, default=3.3,
                         help="Target log perplexity for convergence.")
    mlperf.add_argument("--step-time-atol", type=int, default=18000,
                         help="Maximum tolerable step time (ms).")
    mlperf.add_argument("--eval-every", type=int, default=0,
                         help="Evaluate every N training sequences.")
    mlperf.add_argument("--start-eval-at", type=int, default=0,
                         help="Start evaluation at N training sequences.")
    mlperf.add_argument("--size", type=str, default="8b",
                         choices=["8b"],
                         help="Model size (for Docker compatibility).")
    mlperf.add_argument("--nodes", type=int, default=None,
                         help="Number of nodes (Docker compat, unused by Megatron).")
    mlperf.add_argument("--gpus-per-node", type=int, default=None,
                         help="GPUs per node (Docker compat, unused by Megatron).")

    return parser
