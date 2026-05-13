"""Phase C: FP8 benchmark on DeepMath-103K with TRL v1.0.0 + Lumen FP8.

Identical config to Phase B BF16 baseline, with Lumen FP8 enabled.
"""
import json
import time
import torch
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward
from lumen.config import LumenConfig

OUTPUT_DIR = "/workspace/Lumen/outputs/benchmark/qwen2-0.5b-exact/fp8_run"
MODEL_PATH = "/dev/shm/model/qwen2-0.5b-instruct"


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

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

cfg = LumenConfig(format="fp8_e4m3", scaling="dynamic")
_, model = cfg.enable(model)
print("Lumen FP8 enabled on model")

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
    model=model,
    reward_funcs=accuracy_reward,
    args=training_args,
    train_dataset=dataset,
    callbacks=[_PerfLogCallback()],
)
trainer.train()
print("Phase C FP8 training done!")
