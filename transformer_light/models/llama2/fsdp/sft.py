###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""LLaMA2 Supervised Fine-Tuning with PyTorch FSDP.

This module provides a complete SFT trainer using:

- **HuggingFace Transformers** for the LLaMA2 model
- **PyTorch FSDP** for distributed sharding (no Megatron dependency)
- **Transformer Light** for FP8 quantised training and attention

Features (parity with the Megatron backend):
    - Full fine-tuning or LoRA (via HuggingFace PEFT)
    - FP8 quantised training (weight/activation via ``quant.enable``)
    - Packed sequences with answer-only loss masking
    - Synthetic warmup with FP8 state reset
    - Early stopping based on validation loss target
    - Gradient accumulation + gradient clipping

Example::

    from transformer_light.models.llama2.fsdp import FSDPTrainer, get_args

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

from transformer_light.models.llama2.dataset import LLaMA2SFTDataset

__all__ = [
    "FSDPTrainer",
    "apply_fp8_training",
    "apply_lora",
    "build_model",
    "get_args",
    "reset_fp8_state",
]

logger = logging.getLogger(__name__)


def _rank0_print(msg: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(msg)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(args) -> nn.Module:
    """Load a HuggingFace LlamaForCausalLM and return it (unwrapped).

    FSDP wrapping is done separately by :class:`FSDPTrainer`.
    """
    from transformers import LlamaForCausalLM, LlamaConfig

    if args.model_name_or_path:
        _rank0_print(f"> Loading LLaMA2 from {args.model_name_or_path} ...")
        model = LlamaForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    else:
        _rank0_print("> Building LLaMA2 from config ...")
        config = LlamaConfig(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=getattr(args, "num_key_value_heads", args.num_attention_heads),
            max_position_embeddings=args.seq_length,
            vocab_size=args.vocab_size,
        )
        model = LlamaForCausalLM(config)

    return model


# ---------------------------------------------------------------------------
# LoRA (via HuggingFace PEFT)
# ---------------------------------------------------------------------------

def apply_lora(model: nn.Module, args) -> nn.Module:
    """Apply LoRA adapters via HuggingFace PEFT and freeze the base model."""
    from peft import LoraConfig, get_peft_model, TaskType

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    _rank0_print(
        f"> LoRA applied (rank={args.lora_rank}, alpha={args.lora_alpha}) — "
        f"trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )
    return model


# ---------------------------------------------------------------------------
# FP8 quantised training
# ---------------------------------------------------------------------------

def apply_fp8_training(model: nn.Module, args) -> None:
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
        margin=getattr(args, "fp8_margin", 0),
        reduce_amax=args.fp8_reduce_amax,
        history_len=args.fp8_amax_history,
        quantize_activation=args.fp8_activation,
        quantize_grad=getattr(args, "grad_quant_type", None),
    )

    dp_group = dist.group.WORLD if dist.is_initialized() else None
    quant.enable(model, config=config, dp_group=dp_group if config.reduce_amax else None)
    _rank0_print(
        f"> FP8 training enabled (format={args.fp8_format}, "
        f"scaling={args.fp8_scaling}, amax_algo={args.fp8_amax_algo}, "
        f"activation={args.fp8_activation}, grad_quant={config.quantize_grad})"
    )


def reset_fp8_state(model: nn.Module) -> None:
    """Reset FP8 scaling state after warmup."""

    def _reset(m):
        if hasattr(m, "fp8_initialized"):
            m.fp8_initialized = False
        if hasattr(m, "_quant_manager"):
            m._quant_manager.reset()
        if hasattr(m, "_tl_scaling_manager"):
            m._tl_scaling_manager.reset()

    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    unwrapped.apply(_reset)
    _rank0_print("> FP8 state reset after warmup")


# ---------------------------------------------------------------------------
# FSDP Trainer
# ---------------------------------------------------------------------------

class FSDPTrainer:
    """Self-contained FSDP training loop for LLaMA2 SFT.

    Handles model building, FSDP wrapping, LoRA, FP8, dataset, optimizer,
    training loop with warmup, gradient accumulation, and early stopping.
    """

    def __init__(self, args):
        self.args = args
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        if not dist.is_initialized():
            dist.init_process_group("nccl")
        torch.cuda.set_device(self.local_rank)

        self.model = self._build_and_wrap_model()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.train_loader = self._build_dataloader(args.train_data_path, args.train_samples)
        self.val_loader = self._build_dataloader(args.val_data_path, args.val_samples) if args.val_data_path else None

        self._warmup_done = False
        self._warmup_counter = 0
        self._val_loss_ema: Optional[float] = None

    def _build_and_wrap_model(self) -> FSDP:
        args = self.args
        model = build_model(args)

        if getattr(args, "gradient_checkpointing", True):
            from torch.utils.checkpoint import checkpoint
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            _rank0_print("> Gradient checkpointing enabled")

        if args.lora_rank > 0:
            model = apply_lora(model, args)

        if args.fp8_training:
            apply_fp8_training(model, args)

        from functools import partial
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer},
        )

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        sharding = ShardingStrategy.FULL_SHARD
        if args.sharding_strategy == "shard_grad_op":
            sharding = ShardingStrategy.SHARD_GRAD_OP
        elif args.sharding_strategy == "no_shard":
            sharding = ShardingStrategy.NO_SHARD

        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            sharding_strategy=sharding,
            device_id=self.local_rank,
            limit_all_gathers=True,
        )

        _rank0_print(f"> FSDP model ready (sharding={args.sharding_strategy}, "
                      f"world_size={self.world_size})")
        return model

    def _build_optimizer(self) -> torch.optim.Optimizer:
        args = self.args
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )

    def _build_scheduler(self):
        args = self.args
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.max_steps,
            eta_min=args.min_lr,
        )

    def _build_dataloader(self, data_path: Optional[str], num_samples: int) -> DataLoader:
        args = self.args
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

        dataset = LLaMA2SFTDataset(
            num_samples=num_samples,
            data_path=data_path,
            seq_length=args.seq_length,
            tokenizer=tokenizer,
            is_hf_tokenizer=True,
        )

        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=True,
        ) if self.world_size > 1 else None

        return DataLoader(
            dataset,
            batch_size=args.micro_batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def _forward_backward(self, batch):
        """Run forward + backward on one micro-batch, return scalar loss."""
        input_ids = batch["input_ids"][:, :-1].to(self.local_rank)
        labels = batch["input_ids"][:, 1:].to(self.local_rank)
        loss_mask = batch["loss_mask"][:, 1:].to(self.local_rank).float()

        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits

        shift_logits = logits.view(-1, logits.size(-1))
        shift_labels = labels.reshape(-1)
        per_token_loss = nn.functional.cross_entropy(
            shift_logits, shift_labels, reduction="none",
        )
        masked_loss = (per_token_loss * loss_mask.reshape(-1)).sum()
        num_tokens = loss_mask.sum()
        loss = masked_loss / num_tokens.clamp(min=1)

        loss.backward()
        return loss.detach()

    def _synthetic_warmup_step(self):
        """Run one warmup step with synthetic data."""
        args = self.args
        seq_len = args.seq_length + 1
        fake_ids = torch.ones(args.micro_batch_size, seq_len, dtype=torch.long, device=self.local_rank) * 3545
        fake_mask = torch.ones(args.micro_batch_size, seq_len, dtype=torch.long, device=self.local_rank)
        batch = {"input_ids": fake_ids, "loss_mask": fake_mask}
        self.optimizer.zero_grad()
        self._forward_backward(batch)
        self.optimizer.step()

    def _check_early_stop(self, loss_val: float) -> bool:
        args = self.args
        if args.val_loss_target is None:
            return False
        if self._val_loss_ema is None:
            self._val_loss_ema = loss_val
        else:
            self._val_loss_ema = 0.9 * self._val_loss_ema + 0.1 * loss_val
        if self._val_loss_ema < args.val_loss_target:
            _rank0_print(
                f"> [Early Stop] Loss EMA ({self._val_loss_ema:.4f}) < "
                f"target ({args.val_loss_target:.4f})"
            )
            return True
        return False

    def train(self):
        """Run the full training loop."""
        args = self.args

        if args.warmup_steps > 0:
            _rank0_print(f"> Running {args.warmup_steps} synthetic warmup steps ...")
            for _ in range(args.warmup_steps):
                self._synthetic_warmup_step()
            if args.fp8_training:
                reset_fp8_state(self.model)
            if dist.is_initialized():
                dist.barrier()
            _rank0_print("> Warmup complete, starting real training.")

        self.model.train()
        grad_accum = args.gradient_accumulation_steps
        global_step = 0

        data_iter = iter(self.train_loader)
        while global_step < args.max_steps:
            self.optimizer.zero_grad()
            accum_loss = 0.0

            for micro_step in range(grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                loss = self._forward_backward(batch)
                accum_loss += loss.item()

            if args.max_grad_norm > 0:
                self.model.clip_grad_norm_(args.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()
            global_step += 1

            avg_loss = accum_loss / grad_accum
            if global_step % args.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                _rank0_print(
                    f"  step {global_step}/{args.max_steps} | "
                    f"loss {avg_loss:.4f} | lr {lr:.2e}"
                )

            if self._check_early_stop(avg_loss):
                break

            if (args.save_interval > 0 and global_step % args.save_interval == 0
                    and self.global_rank == 0):
                save_path = os.path.join(args.save_dir, f"step_{global_step}")
                self._save_checkpoint(save_path)

        _rank0_print(f"> Training complete after {global_step} steps.")

    def _save_checkpoint(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        unwrapped = self.model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        if hasattr(unwrapped, "save_pretrained"):
            unwrapped.save_pretrained(path)
        else:
            torch.save(unwrapped.state_dict(), os.path.join(path, "model.pt"))
        _rank0_print(f"> Checkpoint saved to {path}")


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    """Parse command-line arguments for FSDP-based LLaMA2 SFT."""
    parser = argparse.ArgumentParser(description="LLaMA2 SFT with FSDP + Transformer Light")

    parser.add_argument("--backend", type=str, default="fsdp",
                        choices=["megatron", "fsdp"], help="Training backend.")

    # -- Model --
    m = parser.add_argument_group("model")
    m.add_argument("--model-name-or-path", type=str, default=None,
                    help="HuggingFace model name or local path (e.g. meta-llama/Llama-2-7b-hf).")
    m.add_argument("--hidden-size", type=int, default=4096)
    m.add_argument("--intermediate-size", type=int, default=11008)
    m.add_argument("--num-layers", type=int, default=32)
    m.add_argument("--num-attention-heads", type=int, default=32)
    m.add_argument("--num-key-value-heads", type=int, default=None)
    m.add_argument("--vocab-size", type=int, default=32000)
    m.add_argument("--seq-length", type=int, default=4096)
    m.add_argument("--tokenizer-name-or-path", type=str, default="meta-llama/Llama-2-7b-hf")

    # -- Training --
    t = parser.add_argument_group("training")
    t.add_argument("--micro-batch-size", type=int, default=1)
    t.add_argument("--gradient-accumulation-steps", type=int, default=8)
    t.add_argument("--max-steps", type=int, default=800)
    t.add_argument("--lr", type=float, default=4e-4)
    t.add_argument("--min-lr", type=float, default=0.0)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--max-grad-norm", type=float, default=1.0)
    t.add_argument("--log-interval", type=int, default=10)
    t.add_argument("--save-interval", type=int, default=0,
                    help="Save checkpoint every N steps. 0 = disabled.")
    t.add_argument("--save-dir", type=str, default="./checkpoints")
    t.add_argument("--num-workers", type=int, default=4)
    t.add_argument("--gradient-checkpointing", action="store_true", default=True,
                    help="Enable gradient/activation checkpointing (default: on).")
    t.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                    action="store_false")

    # -- Data --
    d = parser.add_argument_group("data")
    d.add_argument("--train-data-path", type=str, default=None)
    d.add_argument("--val-data-path", type=str, default=None)
    d.add_argument("--train-samples", type=int, default=10000)
    d.add_argument("--val-samples", type=int, default=500)

    # -- FSDP --
    f = parser.add_argument_group("fsdp")
    f.add_argument("--sharding-strategy", type=str, default="full_shard",
                    choices=["full_shard", "shard_grad_op", "no_shard"])

    # -- LoRA --
    lora = parser.add_argument_group("lora")
    lora.add_argument("--lora-rank", type=int, default=0,
                       help="LoRA rank. 0 = disabled (full fine-tuning).")
    lora.add_argument("--lora-alpha", type=float, default=32.0)
    lora.add_argument("--lora-dropout", type=float, default=0.1)

    # -- FP8 training --
    fp8 = parser.add_argument_group("fp8-training")
    fp8.add_argument("--fp8-training", action="store_true", default=False)
    fp8.add_argument("--fp8-format", type=str, default="fp8_e4m3",
                      choices=["fp8_e4m3", "fp8_e5m2", "hybrid", "mxfp8"])
    fp8.add_argument("--fp8-scaling", type=str, default="delayed",
                      choices=["dynamic", "delayed", "blockwise"])
    fp8.add_argument("--fp8-block-size", type=int, default=128)
    fp8.add_argument("--fp8-amax-algo", type=str, default="max",
                      choices=["max", "most_recent"])
    fp8.add_argument("--fp8-reduce-amax", action="store_true", default=False)
    fp8.add_argument("--fp8-amax-history", type=int, default=16)
    fp8.add_argument("--fp8-margin", type=int, default=0,
                      help="Margin for FP8 scaling factor computation (TE-compatible).")
    fp8.add_argument("--fp8-activation", action="store_true", default=True)
    fp8.add_argument("--no-fp8-activation", dest="fp8_activation",
                      action="store_false")
    fp8.add_argument("--grad-quant-type", type=str, default=None,
                      choices=["fp8", "mxfp8", "fp4"],
                      help="Gradient quantization type (None=disabled).")

    # -- Warmup + Early stopping --
    sft = parser.add_argument_group("sft")
    sft.add_argument("--warmup-steps", type=int, default=0)
    sft.add_argument("--val-loss-target", type=float, default=None)

    return parser.parse_args()
