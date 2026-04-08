###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""LLaMA2 Supervised Fine-Tuning with PyTorch FSDP.

This module provides a complete SFT trainer using:

- **HuggingFace Transformers** for the LLaMA2 model
- **PyTorch FSDP** for distributed sharding (no Megatron dependency)
- **Lumen** for FP8 quantised training and attention

Features (parity with the Megatron backend):
    - Full fine-tuning or LoRA (via HuggingFace PEFT)
    - FP8 quantised training (weight/activation via ``quant.enable``)
    - Packed sequences with answer-only loss masking
    - Synthetic warmup with FP8 state reset
    - Early stopping based on validation loss target
    - Gradient accumulation + gradient clipping

Example::

    from lumen.models.llama2.fsdp import FSDPTrainer, get_args

    args = get_args()
    trainer = FSDPTrainer(args)
    trainer.train()
"""

import argparse
import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler

from lumen.models.experiment_ops import (
    ExperimentTracker,
    get_effective_stop_step,
)

# Re-export shared FSDP helpers so existing callers are not broken.
from lumen.models.fsdp import (  # noqa: F401
    _rank0_print,
    add_common_fsdp_args,
    apply_fp8_training,
    apply_lora,
    build_cosine_warmup_scheduler,
    reset_fp8_state,
    save_fsdp_checkpoint,
    should_run_eval_step,
    sync_scheduler_to_ckpt_step,
)
from lumen.models.llama2.dataset import LLaMA2SFTDataset

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
    """Load a HuggingFace LlamaForCausalLM and return it (unwrapped).

    FSDP wrapping is done separately by :class:`FSDPTrainer`.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

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

        if args.use_ckpt:
            resume_path = args.initial_ckpt_path or args.continual_ckpt_path
            if resume_path:
                args.model_name_or_path = resume_path
                source_kind = "HF weights" if args.resume_from_hf else "checkpoint weights"
                _rank0_print(f"> Resuming {source_kind} from {resume_path}")

        self.model = self._build_and_wrap_model()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        sync_scheduler_to_ckpt_step(self.scheduler, args)
        self.train_loader = self._build_dataloader(args.train_data_path, args.train_samples)
        self.val_loader = self._build_dataloader(args.val_data_path, args.val_samples) if args.val_data_path else None
        self.tracker = ExperimentTracker(
            args=args,
            global_batch_size=args.micro_batch_size * args.gradient_accumulation_steps * self.world_size,
            seq_length=args.seq_length,
            backend="fsdp",
            task_name="llama2-sft",
            rank=self.global_rank,
        )

    def _build_and_wrap_model(self) -> nn.Module:
        args = self.args
        model = build_model(args)

        if getattr(args, "gradient_checkpointing", True):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            _rank0_print("> Gradient checkpointing enabled")

        from lumen.config import LumenConfig
        cfg = LumenConfig.from_args(args)
        _manager, model = cfg.enable(model)

        if getattr(args, "fsdp_version", 1) == 2:
            from lumen.models.fsdp import apply_fsdp2

            apply_fsdp2(model, args)
        else:
            from functools import partial

            from transformers.models.llama.modeling_llama import LlamaDecoderLayer

            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={LlamaDecoderLayer},
            )
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32 if args.linear_fp8 else torch.bfloat16,
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

        _rank0_print(
            f"> FSDP model ready (version={getattr(args, 'fsdp_version', 1)}, "
            f"sharding={args.sharding_strategy}, world_size={self.world_size})"
        )
        return model

    def _build_optimizer(self) -> torch.optim.Optimizer:
        args = self.args
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=1e-5,
            weight_decay=args.weight_decay,
        )

    def _build_scheduler(self):
        return build_cosine_warmup_scheduler(self.optimizer, self.args)

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

        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=True,
            )
            if self.world_size > 1
            else None
        )

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
            shift_logits,
            shift_labels,
            reduction="none",
        )
        masked_loss = (per_token_loss * loss_mask.reshape(-1)).sum()
        num_tokens = loss_mask.sum()
        loss = masked_loss / num_tokens.clamp(min=1)

        (loss / self.args.gradient_accumulation_steps).backward()
        return loss.detach()

    def _synthetic_warmup_step(self):
        """Run one warmup step with synthetic data."""
        args = self.args
        seq_len = args.seq_length + 1
        fake_ids = torch.ones(args.micro_batch_size, seq_len, dtype=torch.long, device=self.local_rank) * 3545
        fake_mask = torch.ones(args.micro_batch_size, seq_len, dtype=torch.long, device=self.local_rank)

        self.optimizer.zero_grad()
        outputs = self.model(input_ids=fake_ids[:, :-1])
        logits = outputs.logits
        shift_logits = logits.view(-1, logits.size(-1))
        shift_labels = fake_ids[:, 1:].reshape(-1)
        per_token_loss = nn.functional.cross_entropy(
            shift_logits,
            shift_labels,
            reduction="none",
        )
        loss_mask = fake_mask[:, 1:].reshape(-1).float()
        masked_loss = (per_token_loss * loss_mask).sum()
        loss = masked_loss / loss_mask.sum().clamp(min=1)
        loss.backward()
        self.optimizer.step()

    def _validate(self) -> float:
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"][:, :-1].to(self.local_rank)
                labels = batch["input_ids"][:, 1:].to(self.local_rank)
                loss_mask = batch["loss_mask"][:, 1:].to(self.local_rank).float()

                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits
                shift_logits = logits.view(-1, logits.size(-1))
                shift_labels = labels.reshape(-1)
                per_token_loss = nn.functional.cross_entropy(
                    shift_logits,
                    shift_labels,
                    reduction="none",
                )
                masked_loss = (per_token_loss * loss_mask.reshape(-1)).sum()
                num_tokens = loss_mask.sum()
                total_loss += (masked_loss / num_tokens.clamp(min=1)).item()
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

    def train(self):
        """Run the full training loop."""
        args = self.args

        if args.warmup_steps > 0:
            _rank0_print(f"> Running {args.warmup_steps} synthetic warmup steps ...")
            for _ in range(args.warmup_steps):
                self._synthetic_warmup_step()
            if args.linear_fp8:
                reset_fp8_state(self.model)
            if dist.is_initialized():
                dist.barrier()
            _rank0_print("> Warmup complete, starting real training.")

        self.model.train()
        grad_accum = args.gradient_accumulation_steps
        global_step = args.ckpt_start_step
        effective_stop_step = get_effective_stop_step(args)
        self.tracker.on_train_start(
            configs={
                "global_batch_size": self.tracker.global_batch_size,
                "max_sequence_length": args.seq_length,
                "max_steps": args.max_steps,
                "effective_stop_step": effective_stop_step,
            }
        )

        data_iter = iter(self.train_loader)
        try:
            while global_step < effective_stop_step:
                self.tracker.record_train_step_start(global_step)
                self.optimizer.zero_grad()
                accum_loss = 0.0

                for _micro_step in range(grad_accum):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(self.train_loader)
                        batch = next(data_iter)

                    loss = self._forward_backward(batch)
                    accum_loss += loss.item()

                if args.max_grad_norm > 0:
                    if getattr(args, "fsdp_version", 1) == 2:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                    else:
                        self.model.clip_grad_norm_(args.max_grad_norm)

                self.optimizer.step()
                self.scheduler.step()
                global_step += 1

                avg_loss = accum_loss / grad_accum
                lr = self.scheduler.get_last_lr()[0]
                step_time_ms = self.tracker.record_train_step_end(
                    global_step=global_step,
                    train_loss=avg_loss,
                    lr=lr,
                )

                if global_step % args.log_interval == 0:
                    _rank0_print(
                        f"  step {global_step}/{effective_stop_step} | "
                        f"loss {avg_loss:.4f} | lr {lr:.2e} | step_time_ms {step_time_ms:.1f}"
                    )

                if self.tracker.should_preempt(global_step=global_step):
                    _rank0_print(f"> Preemptive stop triggered at step {global_step}")
                    break

                if self.val_loader and should_run_eval_step(global_step, args):
                    self.tracker.log_validation_start(global_step)
                    val_loss = self._validate()
                    _rank0_print(f"  step {global_step}/{effective_stop_step} | val_loss {val_loss:.4f}")
                    if self.tracker.should_stop_on_validation(global_step=global_step, val_loss=val_loss):
                        break

                if args.save_interval > 0 and global_step % args.save_interval == 0 and self.global_rank == 0:
                    save_path = os.path.join(args.save_dir, f"step_{global_step}")
                    save_fsdp_checkpoint(self.model, save_path)
                    _rank0_print(f"> Checkpoint saved to {save_path}")

            if args.save_ckpt and self.global_rank == 0:
                final_path = args.continual_ckpt_path or os.path.join(args.save_dir, "final")
                save_fsdp_checkpoint(self.model, final_path)
                _rank0_print(f"> Final checkpoint saved to {final_path}")
        finally:
            self.tracker.finish_run(global_step=global_step)

        _rank0_print(f"> Training complete after {global_step} steps.")


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for FSDP-based LLaMA2 SFT."""
    parser = argparse.ArgumentParser(description="LLaMA2 SFT with FSDP + Lumen")

    # -- Model --
    m = parser.add_argument_group("model")
    m.add_argument(
        "--model-name-or-path",
        type=str,
        default=None,
        help="HuggingFace model name or local path (e.g. meta-llama/Llama-2-7b-hf).",
    )
    m.add_argument("--hidden-size", type=int, default=4096)
    m.add_argument("--intermediate-size", type=int, default=11008)
    m.add_argument("--num-layers", type=int, default=32)
    m.add_argument("--num-attention-heads", type=int, default=32)
    m.add_argument("--num-key-value-heads", type=int, default=None)
    m.add_argument("--vocab-size", type=int, default=32000)
    m.add_argument("--seq-length", type=int, default=4096)
    m.add_argument("--tokenizer-name-or-path", type=str, default="meta-llama/Llama-2-7b-hf")
    add_common_fsdp_args(parser)
    return parser.parse_args()
