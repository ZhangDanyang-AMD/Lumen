"""Qwen2.5-Coder-32B SFT with PyTorch FSDP2 + LoRA + Lumen.

Supervised fine-tuning on FlyDSL instruction-response pairs. Only assistant
tokens contribute to the loss (system + user tokens are masked).

Usage::

    torchrun --nproc_per_node=8 train_sft.py \\
        --backend fsdp --fsdp-version 2 \\
        --model-name-or-path /dev/shm/qwen2.5-coder-32b \\
        --train-data-path /data/sft/train.jsonl \\
        --val-data-path /data/sft/val.jsonl \\
        --lora-rank 32 --lora-alpha 64 \\
        --seq-length 8192 --max-steps 264 --lr 1e-5
"""

import argparse
import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from lumen.models.fsdp import (
    _rank0_print,
    add_common_fsdp_args,
    build_cosine_warmup_scheduler,
)

from dataset import SFTDataset, collate_fn

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def build_model(args) -> nn.Module:
    from transformers import AutoModelForCausalLM
    _rank0_print(f"> Loading model from {args.model_name_or_path} ...")
    return AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )


def _save_dcp(model, path, rank):
    import torch.distributed.checkpoint as dcp
    os.makedirs(path, exist_ok=True)
    dcp.save(model.state_dict(), checkpoint_id=path)
    if rank == 0:
        _rank0_print(f"> DCP checkpoint saved to {path}")


class SFTTrainer:
    def __init__(self, args):
        self.args = args
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        if not dist.is_initialized():
            dist.init_process_group("nccl")
        torch.cuda.set_device(self.local_rank)

        self.model = self._build_and_wrap_model()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr, betas=(0.9, 0.95), eps=1e-5,
            weight_decay=args.weight_decay,
        )
        self.scheduler = build_cosine_warmup_scheduler(self.optimizer, args)
        self.train_loader = self._build_dataloader(args.train_data_path, shuffle=True)
        self.val_loader = (
            self._build_dataloader(args.val_data_path, shuffle=False)
            if args.val_data_path else None
        )

    def _build_and_wrap_model(self):
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

        for name, p in model.named_parameters():
            if "lora_" in name and p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)

        from lumen.models.fsdp import apply_fsdp2
        apply_fsdp2(model, args)
        _rank0_print(f"> FSDP2 ready (world_size={self.world_size})")
        return model

    def _build_dataloader(self, data_path, shuffle=True):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.tokenizer_name_or_path or self.args.model_name_or_path,
            trust_remote_code=True,
        )
        dataset = SFTDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_length=self.args.seq_length,
        )
        sampler = DistributedSampler(
            dataset, num_replicas=self.world_size,
            rank=self.global_rank, shuffle=shuffle,
        )
        return DataLoader(
            dataset, batch_size=self.args.micro_batch_size,
            sampler=sampler, collate_fn=collate_fn,
            num_workers=getattr(self.args, "num_workers", 4),
            pin_memory=True, drop_last=True,
        )

    def _forward_backward(self, batch):
        input_ids = batch["input_ids"].cuda(non_blocking=True)
        labels = batch["labels"].cuda(non_blocking=True)
        loss_mask = batch["loss_mask"].cuda(non_blocking=True)

        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits

        shift_logits = logits.view(-1, logits.size(-1))
        shift_labels = labels.view(-1)
        per_token_loss = nn.functional.cross_entropy(
            shift_logits, shift_labels, reduction="none",
        )
        masked_loss = (per_token_loss * loss_mask.view(-1)).sum()
        num_tokens = loss_mask.sum().clamp(min=1)
        loss = masked_loss / num_tokens
        return loss

    def _validate(self):
        self.model.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                loss = self._forward_backward(batch)
                total_loss += loss.item()
                n += 1
                if n >= 20:
                    break
        self.model.train()
        avg = total_loss / max(n, 1)
        if dist.is_initialized():
            t = torch.tensor([avg], device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            avg = t.item()
        return avg

    def train(self):
        args = self.args
        self.model.train()
        grad_accum = args.gradient_accumulation_steps
        global_step = 0
        gbs = args.micro_batch_size * grad_accum * self.world_size

        _rank0_print(
            f"> Training: max_steps={args.max_steps}, GBS={gbs}, "
            f"seq_len={args.seq_length}, lr={args.lr}, "
            f"lora_rank={args.lora_rank}"
        )

        data_iter = iter(self.train_loader)

        while global_step < args.max_steps:
            self.optimizer.zero_grad()
            accum_loss = 0.0

            for _ in range(grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                loss = self._forward_backward(batch)
                (loss / grad_accum).backward()
                accum_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), args.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            global_step += 1

            lr = self.scheduler.get_last_lr()[0]
            avg_loss = accum_loss / grad_accum

            if global_step % args.log_interval == 0:
                _rank0_print(
                    f"  step {global_step}/{args.max_steps} | "
                    f"loss {avg_loss:.4f} | lr {lr:.2e}"
                )

            # Validation
            eval_interval = getattr(args, "eval_interval", 0)
            if eval_interval > 0 and global_step % eval_interval == 0 and self.val_loader:
                val_loss = self._validate()
                _rank0_print(
                    f"  step {global_step}/{args.max_steps} | "
                    f"val_loss {val_loss:.4f}"
                )

            # Mid-train checkpoint
            if args.save_interval > 0 and global_step % args.save_interval == 0:
                _save_dcp(self.model,
                          os.path.join(args.save_dir, f"step_{global_step}"),
                          self.global_rank)

        # Final checkpoint
        _save_dcp(self.model,
                  os.path.join(args.save_dir, "final"),
                  self.global_rank)
        results_dir = getattr(args, "results_dir", None)
        if results_dir:
            _save_dcp(self.model,
                      os.path.join(results_dir, "final"),
                      self.global_rank)

        _rank0_print(f"> SFT training complete after {global_step} steps.")


def get_args():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-Coder-32B SFT with FSDP2 + LoRA + Lumen")

    m = parser.add_argument_group("model")
    m.add_argument("--model-name-or-path", type=str, required=True)
    m.add_argument("--tokenizer-name-or-path", type=str, default=None)
    m.add_argument("--seq-length", type=int, default=8192)

    sft = parser.add_argument_group("sft")
    sft.add_argument("--results-dir", type=str, default=None)

    add_common_fsdp_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    trainer = SFTTrainer(args)
    trainer.train()
