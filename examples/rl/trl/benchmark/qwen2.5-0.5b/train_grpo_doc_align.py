"""Reproduce the TRL GRPO Trainer documentation reference curves.

Reference: https://huggingface.co/docs/trl/en/grpo_trainer (grpo_curves.png)
  - Uploaded Jan 18, 2025 with TRL ~v0.14.0 defaults
  - Shows train/reward rising from 0 to ~1.0 over ~2700 steps
  - Shows train/reward_std rising from ~0.69 to ~0.93

Environment: TRL v1.0.0

The doc script uses pure defaults with no GRPOConfig. The curves were generated
with v0.14.0 which had different defaults than v1.0.0. To reproduce on v1.0.0
we must explicitly set the v0.14.0 defaults:

  v0.14.0 default -> v1.0.0 default (what we override)
  beta=0.04       -> 0.0    (CRITICAL: enables KL; reference shows train/kl)
  temperature=0.9 -> 1.0
  loss_type=grpo  -> dapo   (only loss type in v0.14.0)

The v0.14.0 reference used 8 GPUs with per_device_train_batch_size=1 and
gradient_accumulation_steps=8, giving an effective batch of 64 prompts.
We keep the same effective batch semantics regardless of GPU count:
  8 GPU: per_device=1, grad_accum=8 -> 8*1*8 = 64 prompts/step
  1 GPU: per_device=8, grad_accum=8 -> 1*8*8 = 64 prompts/step

Launch:
  # Single GPU
  accelerate launch --config_file single_gpu.yaml train_grpo_doc_align.py

  # 8 GPU (matches reference setup)
  accelerate launch --config_file ddp_8gpu.yaml train_grpo_doc_align.py
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "/workspace/Lumen/outputs/benchmark/qwen2-0.5b-exact/doc_align",
)
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/dev/shm/model/qwen2-0.5b-instruct",
)


class _PerfLogCallback(TrainerCallback):
    def __init__(self, log_dir):
        self._log_path = Path(log_dir) / "grpo_perf_log.jsonl"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = 0.0
        self._peak_mem_logged = False

    def on_step_begin(self, args, state, control, **kwargs):
        self._t0 = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return
        record = {"step": state.global_step, "step_time": round(time.perf_counter() - self._t0, 2)}
        if not self._peak_mem_logged and torch.cuda.is_available():
            record["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            self._peak_mem_logged = True
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                record[k] = v
        with open(self._log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=-1)
    cli_args, _ = parser.parse_known_args()

    num_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))

    # Keep effective batch = 64 prompts/step regardless of GPU count.
    # generation_batch must be divisible by num_generations=8.
    if num_gpus >= 8:
        per_device_bs = 1
        grad_accum = 8
    else:
        per_device_bs = max(8 // num_gpus, 1)
        grad_accum = 8

    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        # v0.14.0 defaults that differ from v1.0.0
        beta=0.04,
        temperature=0.9,
        loss_type="grpo",
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        # Same in both versions
        num_generations=8,
        max_completion_length=256,
        learning_rate=1e-6,
        # BF16
        bf16=True,
        gradient_checkpointing=True,
        # Logging / saving
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_steps=cli_args.max_steps,
    )

    print(f"[doc_align] GPUs={num_gpus}  per_device_bs={per_device_bs}  "
          f"grad_accum={grad_accum}  effective_batch={num_gpus * per_device_bs * grad_accum}")

    trainer = GRPOTrainer(
        model=MODEL_PATH,
        reward_funcs=accuracy_reward,
        args=training_args,
        train_dataset=dataset,
        callbacks=[_PerfLogCallback(OUTPUT_DIR)],
    )
    trainer.train()

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[doc_align] Peak GPU memory: {peak_gb:.2f} GB")
    print("Doc-aligned BF16 training done!")


if __name__ == "__main__":
    main()
