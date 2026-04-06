"""Step 2: BF16 with Lumen-style model path on DeepMath-103K.

Identical to Step 1 (train_grpo_doc_align.py) but passes the model as a
string path, same as Step 1. The purpose is to validate that the Lumen
Docker environment + dependencies produce the same training dynamics as
Step 1. Step 3 then applies FP8 to isolate Lumen's quantization effect.

Launch:
  accelerate launch --config_file ddp_8gpu.yaml \
    examples/rl/trl/benchmark/qwen2.5-0.5b/train_grpo_lumen_bf16.py --max-steps 5000
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
    "/workspace/Lumen/outputs/benchmark/qwen2-0.5b-exact/lumen_bf16",
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

    if num_gpus >= 8:
        per_device_bs = 1
        grad_accum = 8
    else:
        per_device_bs = max(8 // num_gpus, 1)
        grad_accum = 8

    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        beta=0.04,
        temperature=0.9,
        loss_type="grpo",
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        num_generations=8,
        max_completion_length=256,
        learning_rate=1e-6,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_steps=cli_args.max_steps,
    )

    print(f"[lumen_bf16] GPUs={num_gpus}  per_device_bs={per_device_bs}  "
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
        print(f"[lumen_bf16] Peak GPU memory: {peak_gb:.2f} GB")
    print("Lumen BF16 training done!")


if __name__ == "__main__":
    main()
