###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""LLaMA 3.1 Pretraining with PyTorch FSDP.

This module provides a complete pretraining trainer using:

- **HuggingFace Transformers** for the LLaMA 3.1 model
- **PyTorch FSDP** for distributed sharding (no Megatron dependency)
- **Transformer Light** for FP8 quantised training

Features (parity with the Megatron backend):
    - Full pretraining or LoRA (via HuggingFace PEFT)
    - FP8 quantised training (weight/activation via ``quant.enable``)
    - Cosine annealing LR with warmup
    - Synthetic warmup with FP8 state reset
    - Early stopping based on validation loss target
    - Gradient accumulation + gradient clipping

Example::

    from transformer_light.models.llama31.fsdp import FSDPTrainer, get_args

    args = get_args()
    trainer = FSDPTrainer(args)
    trainer.train()
"""

import argparse
import logging
import math
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from transformer_light.models.llama31.dataset import PretrainTextDataset

__all__ = [
    "FSDPTrainer",
    "apply_fp8_training",
    "apply_lora",
    "build_model",
    "get_args",
    "reset_fp8_state",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(args) -> nn.Module:
    """Build LLaMA 3.1 model from HuggingFace."""
    from transformers import LlamaForCausalLM, LlamaConfig

    if args.model_name_or_path:
        logger.info("Loading model from %s", args.model_name_or_path)
        model = LlamaForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    else:
        logger.info("Building LLaMA 3.1 from scratch (%d layers)", args.num_layers)
        config = LlamaConfig(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads or args.num_attention_heads,
            vocab_size=args.vocab_size,
            max_position_embeddings=args.max_position_embeddings,
            rope_theta=args.rope_theta,
            rms_norm_eps=1e-5,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = LlamaForCausalLM(config)

    return model


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

def apply_lora(model: nn.Module, args) -> nn.Module:
    """Apply LoRA via HuggingFace PEFT."""
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# FP8
# ---------------------------------------------------------------------------

def apply_fp8_training(model: nn.Module, args, dp_group=None) -> None:
    """Enable FP8 quantised training via Transformer Light."""
    import transformer_light.quantize as quant
    from transformer_light.quantize import (
        AmaxAlgo, QuantConfig, QuantFormat, ScalingType,
    )

    config = QuantConfig(
        format=QuantFormat(args.fp8_format),
        scaling=ScalingType(args.fp8_scaling),
        block_size=args.fp8_block_size,
        amax_algo=AmaxAlgo(args.fp8_amax_algo),
        reduce_amax=args.fp8_reduce_amax,
        history_len=args.fp8_amax_history,
        quantize_activation=args.fp8_activation,
    )

    quant.enable(model, config=config, dp_group=dp_group)
    logger.info("FP8 training enabled: %s", config)


def reset_fp8_state(model: nn.Module) -> None:
    """Reset FP8 scaling managers after synthetic warmup."""

    def _reset(m):
        if hasattr(m, "fp8_initialized"):
            m.fp8_initialized = False
        if hasattr(m, "_quant_manager"):
            m._quant_manager.reset()
        if hasattr(m, "_tl_scaling_manager"):
            m._tl_scaling_manager.reset()

    model.apply(_reset)
    logger.info("FP8 state reset after warmup")


# ---------------------------------------------------------------------------
# FSDP Trainer
# ---------------------------------------------------------------------------

_SHARDING_MAP = {
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "no_shard": ShardingStrategy.NO_SHARD,
}


class FSDPTrainer:
    """Pretraining trainer using PyTorch FSDP + HuggingFace LLaMA 3.1."""

    def __init__(self, args):
        self.args = args
        self._setup_distributed()

        self.model = build_model(args)

        if args.lora_rank > 0:
            self.model = apply_lora(self.model, args)

        if args.fp8_training:
            dp_group = dist.group.WORLD if dist.is_initialized() else None
            apply_fp8_training(self.model, args, dp_group=dp_group)

        self.model = self._wrap_fsdp(self.model)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=1e-5,
            weight_decay=args.weight_decay,
        )
        self.scheduler = self._build_scheduler()

        self.train_loader = self._build_dataloader(args.train_data_path, args.train_samples)
        self.val_loader = (
            self._build_dataloader(args.val_data_path, args.val_samples)
            if args.val_data_path
            else None
        )

    # ------------------------------------------------------------------

    def _setup_distributed(self):
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(self.local_rank)

    def _wrap_fsdp(self, model: nn.Module) -> FSDP:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls={LlamaDecoderLayer},
        )
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        sharding = _SHARDING_MAP.get(
            self.args.sharding_strategy, ShardingStrategy.FULL_SHARD
        )
        return FSDP(
            model,
            sharding_strategy=sharding,
            mixed_precision=mp,
            auto_wrap_policy=wrap_policy,
            device_id=self.local_rank,
        )

    def _build_scheduler(self):
        args = self.args
        warmup = args.lr_warmup_steps

        def _lr_lambda(step):
            if step < warmup:
                return float(step) / max(warmup, 1)
            progress = float(step - warmup) / max(args.max_steps - warmup, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_ratio = args.min_lr / args.lr if args.lr > 0 else 0
            return min_ratio + (1.0 - min_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)

    def _build_dataloader(self, data_path, max_samples):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.args.tokenizer_name_or_path or self.args.model_name_or_path
        )

        ds = PretrainTextDataset(
            data_path=data_path,
            seq_length=self.args.seq_length,
            tokenizer=tokenizer,
            is_hf_tokenizer=True,
            max_samples=max_samples,
        )

        sampler = DistributedSampler(
            ds, num_replicas=self.world_size, rank=self.rank, shuffle=True
        )
        return DataLoader(
            ds,
            batch_size=self.args.micro_batch_size,
            sampler=sampler,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    # ------------------------------------------------------------------

    def train(self):
        args = self.args
        self.model.train()
        global_step = 0
        warmup_done = False

        data_iter = iter(self.train_loader)

        while global_step < args.max_steps:
            self.optimizer.zero_grad()
            accum_loss = 0.0

            for _micro in range(args.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].cuda(non_blocking=True)
                labels = batch["labels"].cuda(non_blocking=True)

                # Synthetic warmup
                if args.warmup_steps > 0 and global_step < args.warmup_steps:
                    input_ids = torch.ones_like(input_ids) * 3545
                    labels = input_ids.clone()

                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()
                accum_loss += loss.item()

            if args.max_grad_norm > 0:
                self.model.clip_grad_norm_(args.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()
            global_step += 1

            # FP8 state reset after warmup
            if (
                args.warmup_steps > 0
                and global_step == args.warmup_steps
                and not warmup_done
            ):
                if args.fp8_training:
                    reset_fp8_state(self.model)
                warmup_done = True
                if self.rank == 0:
                    logger.info(
                        "Synthetic warmup complete (%d steps). Real training begins.",
                        args.warmup_steps,
                    )

            if global_step % args.log_interval == 0 and self.rank == 0:
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    "step %d/%d | loss %.4f | lr %.2e",
                    global_step, args.max_steps, accum_loss, lr,
                )

            if (
                args.save_interval > 0
                and global_step % args.save_interval == 0
                and self.rank == 0
            ):
                save_path = os.path.join(args.save_dir, f"step_{global_step}")
                self.model.module.save_pretrained(save_path)
                logger.info("Checkpoint saved to %s", save_path)

            # Validation + early stopping
            if (
                self.val_loader
                and args.eval_interval > 0
                and global_step % args.eval_interval == 0
            ):
                val_loss = self._validate()
                if self.rank == 0:
                    logger.info("step %d | val_loss %.4f", global_step, val_loss)
                if args.val_loss_target and val_loss < args.val_loss_target:
                    if self.rank == 0:
                        logger.info(
                            "Val loss %.4f < target %.4f. Stopping.",
                            val_loss, args.val_loss_target,
                        )
                    break

        if self.rank == 0:
            logger.info("Training complete at step %d.", global_step)

    def _validate(self) -> float:
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].cuda(non_blocking=True)
                labels = batch["labels"].cuda(non_blocking=True)
                outputs = self.model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss.item()
                n_batches += 1
                if n_batches >= 10:
                    break
        self.model.train()

        avg = total_loss / max(n_batches, 1)
        if dist.is_initialized():
            t = torch.tensor([avg], device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            avg = t.item()
        return avg


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    """Parse command-line arguments for FSDP-based LLaMA 3.1 pretraining."""
    parser = argparse.ArgumentParser(
        description="LLaMA 3.1 Pretraining with FSDP + Transformer Light"
    )

    parser.add_argument("--backend", type=str, default="fsdp",
                        choices=["megatron", "fsdp"], help="Training backend.")

    # -- Model --
    m = parser.add_argument_group("model")
    m.add_argument("--model-name-or-path", type=str, default=None,
                    help="HuggingFace model name or local path.")
    m.add_argument("--hidden-size", type=int, default=4096)
    m.add_argument("--intermediate-size", type=int, default=14336)
    m.add_argument("--num-layers", type=int, default=32)
    m.add_argument("--num-attention-heads", type=int, default=32)
    m.add_argument("--num-key-value-heads", type=int, default=8)
    m.add_argument("--vocab-size", type=int, default=128256)
    m.add_argument("--max-position-embeddings", type=int, default=131072)
    m.add_argument("--rope-theta", type=float, default=500000.0)
    m.add_argument("--seq-length", type=int, default=8192)
    m.add_argument("--tokenizer-name-or-path", type=str, default=None)

    # -- Training --
    t = parser.add_argument_group("training")
    t.add_argument("--micro-batch-size", type=int, default=2)
    t.add_argument("--gradient-accumulation-steps", type=int, default=8)
    t.add_argument("--max-steps", type=int, default=1200000)
    t.add_argument("--lr", type=float, default=8e-4)
    t.add_argument("--min-lr", type=float, default=8e-5)
    t.add_argument("--lr-warmup-steps", type=int, default=128)
    t.add_argument("--weight-decay", type=float, default=0.1)
    t.add_argument("--max-grad-norm", type=float, default=1.0)
    t.add_argument("--log-interval", type=int, default=10)
    t.add_argument("--save-interval", type=int, default=0)
    t.add_argument("--eval-interval", type=int, default=0)
    t.add_argument("--save-dir", type=str, default="./checkpoints")
    t.add_argument("--num-workers", type=int, default=4)

    # -- Data --
    d = parser.add_argument_group("data")
    d.add_argument("--train-data-path", type=str, default=None)
    d.add_argument("--val-data-path", type=str, default=None)
    d.add_argument("--train-samples", type=int, default=None)
    d.add_argument("--val-samples", type=int, default=None)

    # -- FSDP --
    f = parser.add_argument_group("fsdp")
    f.add_argument("--sharding-strategy", type=str, default="full_shard",
                    choices=["full_shard", "shard_grad_op", "no_shard"])

    # -- LoRA --
    lora = parser.add_argument_group("lora")
    lora.add_argument("--lora-rank", type=int, default=0)
    lora.add_argument("--lora-alpha", type=float, default=32.0)
    lora.add_argument("--lora-dropout", type=float, default=0.1)

    # -- FP8 training --
    fp8 = parser.add_argument_group("fp8-training")
    fp8.add_argument("--fp8-training", action="store_true", default=False)
    fp8.add_argument("--fp8-format", type=str, default="fp8_e4m3",
                      choices=["fp8_e4m3", "fp8_e5m2", "mxfp8"])
    fp8.add_argument("--fp8-scaling", type=str, default="delayed",
                      choices=["dynamic", "delayed", "blockwise"])
    fp8.add_argument("--fp8-block-size", type=int, default=128)
    fp8.add_argument("--fp8-amax-algo", type=str, default="most_recent",
                      choices=["max", "most_recent"])
    fp8.add_argument("--fp8-reduce-amax", action="store_true", default=False)
    fp8.add_argument("--fp8-amax-history", type=int, default=4)
    fp8.add_argument("--fp8-activation", action="store_true", default=True)
    fp8.add_argument("--no-fp8-activation", dest="fp8_activation",
                      action="store_false")

    # -- Warmup + Early stopping --
    sft = parser.add_argument_group("pretrain")
    sft.add_argument("--warmup-steps", type=int, default=0)
    sft.add_argument("--val-loss-target", type=float, default=None)

    # -- Checkpoint management (Docker compat) --
    ckpt = parser.add_argument_group("checkpoint")
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

    # -- MLPerf / experiment management --
    mlp = parser.add_argument_group("mlperf")
    mlp.add_argument("--tag", type=str, default="",
                      help="Optional experiment tag.")
    mlp.add_argument("--target-log-ppl", type=float, default=3.3,
                      help="Target log perplexity for convergence.")
    mlp.add_argument("--step-time-atol", type=int, default=18000,
                      help="Maximum tolerable step time (ms).")
    mlp.add_argument("--eval-every", type=int, default=0,
                      help="Evaluate every N training sequences.")
    mlp.add_argument("--start-eval-at", type=int, default=0,
                      help="Start evaluation at N training sequences.")
    mlp.add_argument("--size", type=str, default="8b",
                      choices=["8b"],
                      help="Model size (for Docker compatibility).")

    # -- Primus Turbo attention (accepted for compat, unused in FSDP) --
    primus = parser.add_argument_group("primus-turbo")
    primus.add_argument("--primus-turbo-fp8-attention", type=int, default=0,
                         help="Enable Primus Turbo FP8 Attention.")
    primus.add_argument("--primus-turbo-mxfp8-attention", type=int, default=0,
                         help="Enable Primus Turbo MXFP8 Attention.")
    primus.add_argument("--dbg-attn-output", type=int, default=0,
                         help="Enable debug attention output.")

    return parser.parse_args()
