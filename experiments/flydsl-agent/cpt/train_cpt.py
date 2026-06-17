"""Qwen2.5-Coder-32B Continued Pre-Training with PyTorch FSDP2 + LoRA + Lumen.

Trains a LoRA adapter on top of Qwen2.5-Coder-32B using next-token prediction
on the FlyDSL/aiter/gpu-docs domain corpus.  Uses Lumen's FSDP2 infrastructure
with AITER-patched operators via ``LumenConfig.enable()``.

Usage::

    torchrun --nproc_per_node=8 train_cpt.py \\
        --backend fsdp --fsdp-version 2 \\
        --model-name-or-path /dev/shm/qwen2.5-coder-32b \\
        --train-data-path /data/cpt/train-00000-of-00001.jsonl \\
        --lora-rank 64 --lora-alpha 128 \\
        --seq-length 8192 --max-steps 125 --lr 2e-5
"""

import argparse
import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from lumen.models.fsdp import (
    _rank0_print,
    add_common_fsdp_args,
    build_cosine_warmup_scheduler,
)

from dataset import CPTDataset

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def build_model(args) -> nn.Module:
    """Load Qwen2.5-Coder-32B from HuggingFace."""
    from transformers import AutoModelForCausalLM

    _rank0_print(f"> Loading model from {args.model_name_or_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    return model


def _save_distributed_checkpoint(model, path: str, rank: int) -> None:
    """Save checkpoint using ``torch.distributed.checkpoint`` (DCP).

    DCP is the standard PyTorch API for saving FSDP2 (``fully_shard``)
    models.  Each rank writes its own shard in parallel — no all-gather,
    no NCCL timeout risk, and the save is fast regardless of model size.

    The saved directory can be loaded with ``dcp.load`` or converted back
    to a single-file checkpoint with ``torch.distributed.checkpoint.format_utils``.
    """
    import torch.distributed.checkpoint as dcp

    os.makedirs(path, exist_ok=True)
    dcp.save(model.state_dict(), checkpoint_id=path)
    if rank == 0:
        _rank0_print(f"> DCP checkpoint saved to {path}")


def _save_lora_adapter(model, path: str, rank: int) -> None:
    """Save only the LoRA adapter weights (small, rank-0 only).

    Unwraps FSDP / PEFT wrappers and calls ``save_pretrained`` which
    writes only the adapter delta (~1 GB for r=64 on 32B model).
    For FSDP2 the model parameters are auto-unsharded on access.
    """
    if rank != 0:
        return
    os.makedirs(path, exist_ok=True)

    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module

    if hasattr(unwrapped, "save_pretrained"):
        unwrapped.save_pretrained(path)
        _rank0_print(f"> LoRA adapter saved to {path}")
    else:
        torch.save(unwrapped.state_dict(), os.path.join(path, "model.pt"))
        _rank0_print(f"> State dict saved to {path}/model.pt")


class CPTTrainer:
    """FSDP2 training loop for continued pre-training with Lumen optimizations."""

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
        self.scheduler = build_cosine_warmup_scheduler(self.optimizer, args)
        self.train_loader = self._build_dataloader()

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

        # LoRA adapters may be fp32 from PEFT init — cast to bf16 for FSDP.
        for name, p in model.named_parameters():
            if "lora_" in name and p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)

        from lumen.models.fsdp import apply_fsdp2
        apply_fsdp2(model, args)

        _rank0_print(
            f"> FSDP2 model ready (sharding={args.sharding_strategy}, "
            f"world_size={self.world_size})"
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

    def _build_dataloader(self) -> DataLoader:
        args = self.args
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path or args.model_name_or_path,
            trust_remote_code=True,
        )

        dataset = CPTDataset(
            data_path=args.train_data_path,
            seq_length=args.seq_length,
            tokenizer=tokenizer,
            max_samples=args.train_samples if args.train_samples > 0 else None,
        )

        total_tokens = dataset.token_count()
        _rank0_print(f"> Dataset: {len(dataset)} chunks, ~{total_tokens:,} tokens")

        epochs = getattr(args, "epochs", 3)
        if dataset.weights:
            total_samples = len(dataset) * epochs
            sampler = WeightedRandomSampler(
                weights=dataset.weights,
                num_samples=total_samples,
                replacement=True,
            )
        else:
            from torch.utils.data import DistributedSampler
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=True,
            )

        return DataLoader(
            dataset,
            batch_size=args.micro_batch_size,
            sampler=sampler,
            num_workers=getattr(args, "num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )

    def _save(self, path: str) -> None:
        """Save checkpoint — all ranks participate (DCP is parallel)."""
        _save_distributed_checkpoint(self.model, path, self.global_rank)

    def train(self):
        args = self.args
        self.model.train()
        grad_accum = args.gradient_accumulation_steps
        global_step = 0
        gbs = args.micro_batch_size * grad_accum * self.world_size

        _rank0_print(
            f"> Training config: max_steps={args.max_steps}, "
            f"GBS={gbs}, seq_len={args.seq_length}, lr={args.lr}"
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

                input_ids = batch["input_ids"].cuda(non_blocking=True)
                labels = batch["labels"].cuda(non_blocking=True)

                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / grad_accum
                loss.backward()
                accum_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()
            global_step += 1

            lr = self.scheduler.get_last_lr()[0]

            if global_step % args.log_interval == 0:
                _rank0_print(
                    f"  step {global_step}/{args.max_steps} | "
                    f"loss {accum_loss:.4f} | lr {lr:.2e}"
                )

            if args.save_interval > 0 and global_step % args.save_interval == 0:
                self._save(os.path.join(args.save_dir, f"step_{global_step}"))

        # Final checkpoint: save to save_dir (fast /dev/shm) and also
        # copy to results_dir (persistent storage) for downstream use.
        self._save(os.path.join(args.save_dir, "final"))
        results_dir = getattr(args, "results_dir", None)
        if results_dir:
            self._save(os.path.join(results_dir, "final"))

        _rank0_print(f"> CPT training complete after {global_step} steps.")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5-Coder-32B CPT with FSDP2 + LoRA + Lumen")

    m = parser.add_argument_group("model")
    m.add_argument("--model-name-or-path", type=str, required=True)
    m.add_argument("--tokenizer-name-or-path", type=str, default=None)
    m.add_argument("--seq-length", type=int, default=8192)

    cpt = parser.add_argument_group("cpt")
    cpt.add_argument("--epochs", type=int, default=3)
    cpt.add_argument("--results-dir", type=str, default=None,
                     help="Persistent dir for final checkpoint (save-dir is for fast mid-train saves).")

    add_common_fsdp_args(parser)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    trainer = CPTTrainer(args)
    trainer.train()
