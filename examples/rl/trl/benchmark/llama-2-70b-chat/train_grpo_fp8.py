"""FP8 Linear (dynamic scaling) for Llama-2-70B-chat (FSDP sharded).

Applies Lumen FP8 quantization via LumenConfig to the actor model after
GRPOTrainer init (post-FSDP wrapping). Uses dynamic scaling which computes
per-tensor amax on every forward pass.

Launch:
  accelerate launch --config_file fsdp_8gpu.yaml \
    --num_processes 8 --main_process_port 29500 \
    examples/rl/trl/benchmark/llama-2-70b-chat/train_grpo_fp8.py --max-steps 20
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward

from lumen.config import LumenConfig

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "/workspace/Lumen/outputs/benchmark/llama-2-70b-chat/lumen_fp8_dynamic",
)
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/dev/shm/model/llama-2-70b-chat",
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
    parser.add_argument("--max-steps", type=int, default=20)
    cli_args, _ = parser.parse_known_args()

    num_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))

    per_device_bs = 1
    grad_accum = 4

    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        beta=0.0,
        temperature=0.9,
        loss_type="grpo",
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        num_generations=2,
        max_completion_length=64,
        learning_rate=1e-6,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_steps=cli_args.max_steps,
        model_init_kwargs={"torch_dtype": "bfloat16", "low_cpu_mem_usage": True},
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}{{ message['content'] }}{% endif %}"
            "{% endfor %}"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[fp8_70b] Model={MODEL_PATH}  GPUs={num_gpus}  per_device_bs={per_device_bs}  "
          f"grad_accum={grad_accum}  effective_batch={num_gpus * per_device_bs * grad_accum}")

    trainer = GRPOTrainer(
        model=MODEL_PATH,
        reward_funcs=accuracy_reward,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[_PerfLogCallback(OUTPUT_DIR)],
    )

    print("[fp8_70b] Applying Lumen FP8 Linear (dynamic scaling) to actor model...")
    cfg = LumenConfig(format="fp8_e4m3", scaling="dynamic")
    _, trainer.model = cfg.enable(trainer.model)
    print("[fp8_70b] LumenConfig FP8 Linear applied!")

    trainer.train()

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[fp8_70b] Peak GPU memory: {peak_gb:.2f} GB")
    print("[fp8_70b] FP8 Linear training done!")


if __name__ == "__main__":
    main()
