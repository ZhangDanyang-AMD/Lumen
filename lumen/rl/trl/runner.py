###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""High-level GRPO runner for the TRL + Lumen integration."""

import os

import torch
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from lumen.rl.trl.args import build_grpo_config_kwargs
from lumen.rl.trl.eval_callback import GRPOEvalCallback
from lumen.rl.trl.modeling import build_actor_model
from lumen.rl.trl.warmup import maybe_run_synthetic_warmup

__all__ = ["run_grpo"]


def _build_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path or args.model_name_or_path,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def run_grpo(args, *, reward_fn):
    """Run GRPO using the current pinned prebuilt-model ownership mode.

    Current TRL accepts a pre-instantiated actor model, so Lumen keeps ownership
    of actor construction before handing the model to `GRPOTrainer`. Warmup runs
    after trainer initialization so it targets the trainer-visible model/device
    placement instead of materializing a full CPU actor onto CUDA early. Example
    launchers keep warmup opt-in (`warmup_steps=0` by default) until this path is
    validated on the target distributed stack. v1 also keeps `beta=0.0`, so the
    runner does not rely on TRL-owned reference-model construction. If a future
    pinned stack changes these assumptions, switch to a trainer-owned init flow
    and revisit the explicit reference builder path.
    """

    if getattr(args, "train_dataset", None) is None:
        raise ValueError("args.train_dataset must be populated before calling run_grpo().")

    tokenizer = _build_tokenizer(args)
    model = build_actor_model(args)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    config_kwargs = build_grpo_config_kwargs(args)
    config = GRPOConfig(**config_kwargs)

    eval_cb = GRPOEvalCallback(output_dir=config_kwargs["output_dir"])
    callbacks = [eval_cb]

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=args.train_dataset,
        eval_dataset=getattr(args, "eval_dataset", None),
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    maybe_run_synthetic_warmup(getattr(trainer, "model", model), args, device=device)
    trainer.train()
    return trainer
