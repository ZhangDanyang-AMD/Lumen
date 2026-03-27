###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Accelerate entrypoint for TRL + Lumen GRPO runs."""

import argparse

from datasets import load_dataset

from lumen.rl.trl.args import TrlLumenArgs
from lumen.rl.trl.runner import run_grpo


def reward_fn(prompts, completions, **kwargs):
    return [1.0 for _ in completions]


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
    return parser.parse_args()


def _load_train_dataset(raw):
    if raw.train_data_path:
        return load_dataset("json", data_files=raw.train_data_path, split="train")
    return load_dataset(
        raw.dataset_name,
        raw.dataset_config_name,
        split=raw.dataset_split,
    )


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
        train_dataset=_load_train_dataset(raw),
    )
    run_grpo(args, reward_fn=reward_fn)


if __name__ == "__main__":
    main()
