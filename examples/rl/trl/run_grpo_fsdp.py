###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Accelerate entrypoint for TRL + Lumen GRPO runs."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset

from lumen.rl.trl.args import TrlLumenArgs
from lumen.rl.trl.runner import run_grpo


def reward_fn(prompts, completions, **kwargs):
    """Reward concise, substantive completions to create a real training signal.

    GRPO requires reward variance across the group of completions for each
    prompt.  A constant reward yields zero advantages and therefore zero
    policy gradient.  This reward function penalises both trivially short
    and excessively long responses, creating a smooth gradient that drives
    the policy toward a preferred response length.

    TRL passes completions in chat-message format (list of dicts) when the
    dataset uses chat templates.  Extract the text content transparently.
    """
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            text = " ".join(m.get("content", "") for m in completion if isinstance(m, dict))
        else:
            text = completion
        n_words = len(text.split())
        if n_words < 5:
            r = 0.1
        elif n_words <= 60:
            r = min(1.0, 0.3 + 0.7 * n_words / 60)
        else:
            r = max(0.0, 1.0 - (n_words - 60) / 120)
        rewards.append(round(r, 4))
    return rewards


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRPO with TRL + Lumen FSDP/FSDP2.")
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--dataset-name", type=str, default=None)
    data.add_argument("--train-data-path", type=str, default=None)
    parser.add_argument("--dataset-config-name", type=str, default=None)
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--tokenizer-name-or-path", type=str, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fsdp-version", type=int, default=1, choices=[1, 2])
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Synthetic warmup steps. Defaults to 0 until the target FSDP stack is validated.",
    )
    parser.add_argument("--linear-fp8", action="store_true", default=False)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing for the actor build path.",
    )
    parser.set_defaults(gradient_checkpointing=True)

    parser.add_argument("--lumen-norm", action="store_true", default=False)
    parser.add_argument("--lumen-fp8-attn", type=str, default="none",
                        choices=["none", "dpa", "mha"])
    parser.add_argument("--lumen-fp8-quant-type", type=str, default="blockwise",
                        choices=["dynamic", "delayed", "blockwise", "blockwise2d",
                                 "per_token", "none", "mxfp8"])
    parser.add_argument("--lumen-attn-backend", type=str, default="auto",
                        choices=["auto", "triton", "csrc", "asm"])
    parser.add_argument("--lumen-fp8-activation-store", action="store_true", default=False)
    parser.add_argument("--lumen-fp8-param-gather", action="store_true", default=False)
    parser.add_argument("--lumen-fused-mlp", action="store_true", default=False)
    parser.add_argument("--lumen-cpu-offload", action="store_true", default=False)
    parser.add_argument("--lumen-delay-wgrad", action="store_true", default=False)
    parser.add_argument("--lumen-gradient-accumulation-fusion", action="store_true", default=False)
    parser.add_argument("--lumen-fused-rope", action="store_true", default=False)
    parser.add_argument("--lumen-hip-graphs", action="store_true", default=False)
    parser.add_argument("--lumen-fp8-checkpoint", action="store_true", default=False)
    parser.add_argument("--fp8-param-manager", action="store_true", default=False,
                        help="Replace nn.Linear weights with FP8 via FP8ParamManager (true memory savings).")
    parser.add_argument("--use-8bit-adam", action="store_true", default=False,
                        help="Use bitsandbytes Adam8bit optimizer for smaller optimizer states.")

    return parser.parse_args()


def _ensure_prompt_column(dataset):
    """Ensure the dataset has a ``prompt`` column required by GRPOTrainer.

    Datasets in chat-messages format (e.g. ``trl-lib/Capybara``) store
    conversations under a ``messages`` column.  Extract the first user
    message content as the prompt so GRPO can generate completions from it.
    """
    if "prompt" in dataset.column_names:
        return dataset
    if "messages" not in dataset.column_names:
        raise ValueError("Dataset must have either a 'prompt' or 'messages' column.")

    def _extract(example):
        for msg in example["messages"]:
            if msg["role"] == "user":
                return {"prompt": msg["content"]}
        return {"prompt": ""}

    return dataset.map(_extract)


def _load_train_dataset(raw):
    if raw.train_data_path:
        ds = load_dataset("json", data_files=raw.train_data_path, split="train")
    else:
        ds = load_dataset(
            raw.dataset_name,
            raw.dataset_config_name,
            split=raw.dataset_split,
        )
    return _ensure_prompt_column(ds)


def main():
    raw = parse_args()
    seq_length = raw.seq_length
    if seq_length is None:
        seq_length = raw.max_prompt_length + raw.max_completion_length

    args = TrlLumenArgs(
        model_name_or_path=raw.model_name_or_path,
        dataset_name=raw.dataset_name,
        dataset_config_name=raw.dataset_config_name,
        dataset_split=raw.dataset_split,
        output_dir=raw.output_dir,
        tokenizer_name_or_path=raw.tokenizer_name_or_path,
        fsdp_version=raw.fsdp_version,
        micro_batch_size=raw.micro_batch_size,
        gradient_accumulation_steps=raw.gradient_accumulation_steps,
        max_steps=raw.max_steps,
        lr=raw.lr,
        lr_warmup_steps=raw.lr_warmup_steps,
        log_interval=raw.log_interval,
        save_interval=raw.save_interval,
        seq_length=seq_length,
        max_prompt_length=raw.max_prompt_length,
        max_completion_length=raw.max_completion_length,
        num_generations=raw.num_generations,
        warmup_steps=raw.warmup_steps,
        linear_fp8=raw.linear_fp8,
        lora_rank=raw.lora_rank,
        lora_alpha=raw.lora_alpha,
        lora_dropout=raw.lora_dropout,
        gradient_checkpointing=raw.gradient_checkpointing,
        lumen_norm=raw.lumen_norm,
        lumen_fp8_attn=raw.lumen_fp8_attn,
        lumen_fp8_quant_type=raw.lumen_fp8_quant_type,
        lumen_attn_backend=raw.lumen_attn_backend,
        lumen_fp8_activation_store=raw.lumen_fp8_activation_store,
        lumen_fp8_param_gather=raw.lumen_fp8_param_gather,
        lumen_fused_mlp=raw.lumen_fused_mlp,
        lumen_cpu_offload=raw.lumen_cpu_offload,
        lumen_delay_wgrad=raw.lumen_delay_wgrad,
        lumen_gradient_accumulation_fusion=raw.lumen_gradient_accumulation_fusion,
        lumen_fused_rope=raw.lumen_fused_rope,
        lumen_hip_graphs=raw.lumen_hip_graphs,
        lumen_fp8_checkpoint=raw.lumen_fp8_checkpoint,
        fp8_param_manager=raw.fp8_param_manager,
        use_8bit_adam=raw.use_8bit_adam,
        train_dataset=_load_train_dataset(raw),
    )

    rank = int(os.environ.get("RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    trainer = run_grpo(args, reward_fn=reward_fn)
    elapsed = time.time() - t0

    if rank == 0:
        lumen_opts = {k: v for k, v in {
            "lumen_norm": raw.lumen_norm,
            "lumen_fp8_attn": raw.lumen_fp8_attn,
            "lumen_fp8_activation_store": raw.lumen_fp8_activation_store,
            "lumen_fp8_param_gather": raw.lumen_fp8_param_gather,
            "lumen_fused_mlp": raw.lumen_fused_mlp,
        }.items() if v and v != "none"}
        summary = {
            "model": raw.model_name_or_path,
            "linear_fp8": raw.linear_fp8,
            "fp8_param_manager": raw.fp8_param_manager,
            "use_8bit_adam": raw.use_8bit_adam,
            "lora_rank": raw.lora_rank,
            "max_steps": raw.max_steps,
            "elapsed_seconds": round(elapsed, 2),
            **lumen_opts,
        }
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            free, total = torch.cuda.mem_get_info()
            summary["peak_memory_gb"] = round(peak_gb, 2)
            summary["free_gpu_gb"] = round(free / 1e9, 2)
            summary["total_gpu_gb"] = round(total / 1e9, 2)

        print(f"\n{'='*60}", flush=True)
        print(f"BENCHMARK SUMMARY", flush=True)
        print(f"{'='*60}", flush=True)
        for k, v in summary.items():
            print(f"  {k}: {v}", flush=True)
        print(f"{'='*60}", flush=True)

        summary_path = Path(raw.output_dir) / "benchmark_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
