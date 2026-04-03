"""TRL GRPO reference — Qwen2-0.5B-Instruct on DeepMath-103K.

Reproduces the TRL docs quick-start exactly:
  https://huggingface.co/docs/trl/en/grpo_trainer#quick-start

All GRPOConfig parameters are left at their v1.0.0 defaults
(loss_type=dapo, beta=0.0, temperature=1.0, etc.) except
logging_steps, save_strategy, and report_to which don't affect training.
"""
import json
import time
from pathlib import Path

from datasets import load_dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward

OUTPUT_DIR = "/workspace/Lumen/outputs/benchmark/qwen2-0.5b-exact/R1"


class _PerfLogCallback(TrainerCallback):
    def __init__(self):
        self._log_path = Path(OUTPUT_DIR) / "grpo_perf_log.jsonl"
        self._history = []
        self._t0 = 0.0

    def on_step_begin(self, args, state, control, **kwargs):
        self._t0 = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return
        step = state.global_step
        record = {"step": step, "step_time": round(time.perf_counter() - self._t0, 2)}
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                record[k] = v
        self._history.append(record)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "w") as fh:
            for e in self._history:
                fh.write(json.dumps(e) + "\n")


dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
)

trainer = GRPOTrainer(
    model="/dev/shm/model/qwen2-0.5b-instruct",
    reward_funcs=accuracy_reward,
    args=training_args,
    train_dataset=dataset,
    callbacks=[_PerfLogCallback()],
)
trainer.train()
print("Done!")
