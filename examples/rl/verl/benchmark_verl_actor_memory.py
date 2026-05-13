#!/usr/bin/env python3
"""VERL actor memory benchmark — measures FSDP2 training memory with different
Lumen optimization configurations.

Simulates VERL's actor training path:
  - Load model
  - Optionally apply FP8ParamManager + LoRA via LumenConfig.enable()
  - Wrap with FSDP2 (or DDP for FP8ParamManager)
  - Run forward + backward + optimizer step
  - Report peak GPU memory per rank

Usage (multi-GPU):
    torchrun --nproc_per_node=8 benchmark_verl_actor_memory.py --config bf16
    torchrun --nproc_per_node=8 benchmark_verl_actor_memory.py --config lora
    torchrun --nproc_per_node=8 benchmark_verl_actor_memory.py --config fp8pm_lora
    torchrun --nproc_per_node=8 benchmark_verl_actor_memory.py --config fp8_linear
"""
import argparse
import gc
import json
import os
import time

import torch
import torch.distributed as dist


CONFIGS = {
    "bf16": {
        "desc": "BF16 baseline (full finetune, FSDP2)",
        "fp8_param_manager": False,
        "lora_rank": 0,
        "linear_fp8": False,
        "use_8bit_adam": False,
        "strategy": "fsdp2",
    },
    "lora": {
        "desc": "BF16 + LoRA r=16 (FSDP2)",
        "fp8_param_manager": False,
        "lora_rank": 16,
        "linear_fp8": False,
        "use_8bit_adam": False,
        "strategy": "fsdp2",
    },
    "fp8pm_lora": {
        "desc": "FP8ParamManager + LoRA r=16 + 8-bit Adam (DDP)",
        "fp8_param_manager": True,
        "lora_rank": 16,
        "linear_fp8": False,
        "use_8bit_adam": True,
        "strategy": "ddp",
    },
    "fp8_linear": {
        "desc": "FP8 Linear (FSDP2)",
        "fp8_param_manager": False,
        "lora_rank": 0,
        "linear_fp8": True,
        "use_8bit_adam": False,
        "strategy": "fsdp2",
    },
    "fp8_linear_lora": {
        "desc": "FP8 Linear + LoRA r=16 (FSDP2)",
        "fp8_param_manager": False,
        "lora_rank": 16,
        "linear_fp8": True,
        "use_8bit_adam": False,
        "strategy": "fsdp2",
    },
    "bf16_ddp": {
        "desc": "BF16 baseline (full finetune, DDP)",
        "fp8_param_manager": False,
        "lora_rank": 0,
        "linear_fp8": False,
        "use_8bit_adam": False,
        "strategy": "ddp",
    },
    "lora_ddp": {
        "desc": "BF16 + LoRA r=16 (DDP)",
        "fp8_param_manager": False,
        "lora_rank": 16,
        "linear_fp8": False,
        "use_8bit_adam": False,
        "strategy": "ddp",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS.keys()), required=True)
    parser.add_argument("--model", default="/dev/shm/model/llama-3.1-8b")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--micro_bs", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-7)
    return parser.parse_args()


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def load_model(model_path, rank):
    from transformers import AutoConfig, AutoModelForCausalLM

    if rank == 0:
        print(f"Loading model from {model_path}...")
    config = AutoConfig.from_pretrained(model_path)
    config.attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()
    return model


def apply_lumen_config(model, cfg_dict, rank):
    from lumen.config import LumenConfig

    lumen_cfg = LumenConfig(
        fp8_param_manager=cfg_dict["fp8_param_manager"],
        lora_rank=cfg_dict["lora_rank"],
        scaling="delayed" if cfg_dict["linear_fp8"] else "none",
        use_8bit_adam=cfg_dict["use_8bit_adam"],
    )
    if lumen_cfg.has_any_features:
        if rank == 0:
            print(f"Applying LumenConfig: {cfg_dict}")
        _, model = lumen_cfg.enable(model)
    return model, lumen_cfg


def wrap_fsdp2(model, rank):
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)

    for module in model.modules():
        if hasattr(module, "self_attn") or hasattr(module, "mlp"):
            for child_name in ("self_attn", "mlp"):
                child = getattr(module, child_name, None)
                if child is not None:
                    fully_shard(child, mp_policy=mp_policy)

    fully_shard(model, mp_policy=mp_policy)
    if rank == 0:
        print("Model wrapped with FSDP2")
    return model


def wrap_ddp(model, local_rank, rank):
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank],
        find_unused_parameters=False,
    )
    if rank == 0:
        print("Model wrapped with DDP")
    return model


def create_optimizer(model, lr, use_8bit_adam, rank):
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")

    if use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(trainable, lr=lr)
        if rank == 0:
            print("Using 8-bit Adam optimizer")
    else:
        optimizer = torch.optim.AdamW(trainable, lr=lr)
        if rank == 0:
            print("Using AdamW optimizer")
    return optimizer


def run_training_steps(model, optimizer, steps, micro_bs, seq_len, rank, local_rank):
    torch.cuda.reset_peak_memory_stats(local_rank)
    torch.cuda.synchronize()

    for step in range(steps):
        input_ids = torch.randint(0, 32000, (micro_bs, seq_len), device=f"cuda:{local_rank}")
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()

        if rank == 0:
            peak = torch.cuda.max_memory_allocated(local_rank) / (1024**3)
            current = torch.cuda.memory_allocated(local_rank) / (1024**3)
            print(f"  Step {step+1}/{steps}: loss={loss.item():.4f}, "
                  f"peak={peak:.2f} GB, current={current:.2f} GB")

    return torch.cuda.max_memory_allocated(local_rank) / (1024**3)


def main():
    args = parse_args()
    cfg_dict = CONFIGS[args.config]
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"VERL Actor Memory Benchmark")
        print(f"Config:     {args.config} — {cfg_dict['desc']}")
        print(f"Model:      {args.model}")
        print(f"GPUs:       {world_size}")
        print(f"Steps:      {args.steps}")
        print(f"Micro BS:   {args.micro_bs}")
        print(f"Seq Len:    {args.seq_len}")
        print(f"Strategy:   {cfg_dict['strategy']}")
        print(f"{'='*60}\n")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(local_rank)

    model = load_model(args.model, rank)
    model, lumen_cfg = apply_lumen_config(model, cfg_dict, rank)

    if cfg_dict["strategy"] == "fsdp2":
        model = wrap_fsdp2(model, rank)
    else:
        model = wrap_ddp(model, local_rank, rank)

    optimizer = create_optimizer(model, args.lr, cfg_dict["use_8bit_adam"], rank)

    if rank == 0:
        print(f"\nRunning {args.steps} training steps...")

    peak_gb = run_training_steps(
        model, optimizer, args.steps, args.micro_bs, args.seq_len, rank, local_rank,
    )

    all_peaks = [None] * world_size
    dist.all_gather_object(all_peaks, peak_gb)

    if rank == 0:
        avg_peak = sum(all_peaks) / len(all_peaks)
        max_peak = max(all_peaks)
        print(f"\n{'='*60}")
        print(f"RESULTS: {args.config} — {cfg_dict['desc']}")
        print(f"  Avg Peak Memory/GPU: {avg_peak:.2f} GB")
        print(f"  Max Peak Memory/GPU: {max_peak:.2f} GB")
        print(f"  All GPUs: {[f'{p:.2f}' for p in all_peaks]}")
        print(f"{'='*60}\n")

        result = {
            "config": args.config,
            "desc": cfg_dict["desc"],
            "model": args.model,
            "world_size": world_size,
            "steps": args.steps,
            "micro_bs": args.micro_bs,
            "seq_len": args.seq_len,
            "avg_peak_gb": round(avg_peak, 2),
            "max_peak_gb": round(max_peak, 2),
            "all_peaks_gb": [round(p, 2) for p in all_peaks],
        }
        results_file = f"/workspace/Lumen/outputs/benchmark/verl_actor_{args.config}.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {results_file}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
