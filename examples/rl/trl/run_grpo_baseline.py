###############################################################################
# Baseline GRPO run — pure TRL, zero Lumen imports.
#
# This script mirrors run_grpo_fsdp.py but lets TRL handle model construction
# internally (model name string, not a pre-built model object).  Everything
# else — reward function, dataset pipeline, GRPOConfig kwargs, seed — is kept
# identical so the two runs can be compared side-by-side.
###############################################################################

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer


# ---------------------------------------------------------------------------
# Eval callback (inlined — no Lumen dependency)
# ---------------------------------------------------------------------------

_TRACKED_KEYS = {
    "loss",
    "grad_norm",
    "learning_rate",
    "reward",
    "reward_std",
    "entropy",
    "completions/mean_length",
    "completions/min_length",
    "completions/max_length",
    "completions/clipped_ratio",
    "completions/mean_terminated_length",
    "clip_ratio/low_mean",
    "clip_ratio/high_mean",
    "clip_ratio/region_mean",
    "num_tokens",
    "step_time",
    "epoch",
    "rewards/reward_fn/mean",
    "rewards/reward_fn/std",
    "frac_reward_zero_std",
}


class BaselineEvalCallback(TrainerCallback):
    _WIN_RATE_WARMUP = 3
    _WIN_RATE_WINDOW = 5

    def __init__(self, output_dir: str):
        super().__init__()
        self._log_path = Path(output_dir) / "grpo_eval_log.jsonl"
        self._history: list[dict] = []
        self._reward_history: list[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return

        step = state.global_step
        record = {"step": step}

        for key in _TRACKED_KEYS:
            if key in logs:
                record[key] = logs[key]

        reward_mean = logs["reward"] if "reward" in logs else logs.get("rewards/reward_fn/mean")
        if reward_mean is not None:
            reward_mean = float(reward_mean)
            self._reward_history.append(reward_mean)

            n = len(self._reward_history)
            warmup = self._WIN_RATE_WARMUP
            window = self._WIN_RATE_WINDOW

            if n <= warmup:
                record["win_rate"] = 0.0
            else:
                baseline = sum(self._reward_history[:warmup]) / warmup
                recent = self._reward_history[-window:]
                record["win_rate"] = sum(1 for r in recent if r > baseline) / len(recent)

        self._history.append(record)
        self._flush()

    def _flush(self):
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "w") as fh:
            for entry in self._history:
                fh.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Reward function (identical to run_grpo_fsdp.py)
# ---------------------------------------------------------------------------

def reward_fn(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        n_words = len(completion.split())
        if n_words < 5:
            r = 0.1
        elif n_words <= 60:
            r = min(1.0, 0.3 + 0.7 * n_words / 60)
        else:
            r = max(0.0, 1.0 - (n_words - 60) / 120)
        rewards.append(round(r, 4))
    return rewards


# ---------------------------------------------------------------------------
# Dataset helpers (identical to run_grpo_fsdp.py)
# ---------------------------------------------------------------------------

def _ensure_prompt_column(dataset):
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Baseline GRPO run (pure TRL, no Lumen).")
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--dataset-name", type=str, default=None)
    data.add_argument("--train-data-path", type=str, default=None)
    parser.add_argument("--dataset-config-name", type=str, default=None)
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--tokenizer-name-or-path", type=str, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    raw = parse_args()

    train_dataset = _load_train_dataset(raw)

    tokenizer = AutoTokenizer.from_pretrained(
        raw.tokenizer_name_or_path or raw.model_name_or_path,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config = GRPOConfig(
        output_dir=raw.output_dir,
        learning_rate=raw.lr,
        weight_decay=0.01,
        max_grad_norm=1.0,
        per_device_train_batch_size=raw.micro_batch_size,
        gradient_accumulation_steps=raw.gradient_accumulation_steps,
        max_steps=raw.max_steps,
        warmup_steps=raw.lr_warmup_steps,
        logging_steps=raw.log_interval,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        max_completion_length=raw.max_completion_length,
        num_generations=raw.num_generations,
        beta=0.0,
        temperature=1.0,
        top_p=1.0,
        report_to="none",
        remove_unused_columns=False,
        seed=raw.seed,
        use_vllm=False,
    )

    eval_cb = BaselineEvalCallback(output_dir=raw.output_dir)

    trainer = GRPOTrainer(
        model=raw.model_name_or_path,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[eval_cb],
    )
    trainer.train()


if __name__ == "__main__":
    main()
