"""Phase B: Tuned BF16 baseline on DeepMath-103K with TRL v1.0.0.

Tuned config to address sparse reward problem:
- loss_type="bnpo": per-token normalization, better for sparse rewards
- beta=0.04: KL penalty prevents mode collapse
- mask_truncated_completions=True: prevents length exploitation
- max_completion_length=512: more room for math reasoning
- learning_rate=5e-6: higher LR to overcome sparse rewards
- lr_scheduler_type="cosine": better for long training
"""
import json
import time
from pathlib import Path

from datasets import load_dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward

OUTPUT_DIR = "/workspace/Lumen/outputs/benchmark/qwen2-0.5b-exact/bf16_baseline"


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
    max_steps=2500,
    loss_type="bnpo",
    beta=0.04,
    mask_truncated_completions=True,
    max_completion_length=512,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    temperature=1.0,
    num_generations=8,
    per_device_train_batch_size=8,
)

trainer = GRPOTrainer(
    model="/dev/shm/model/qwen2-0.5b-instruct",
    reward_funcs=accuracy_reward,
    args=training_args,
    train_dataset=dataset,
    callbacks=[_PerfLogCallback()],
)
trainer.train()
print("Phase B BF16 baseline training done!")
