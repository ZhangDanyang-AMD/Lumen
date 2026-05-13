"""FP8 Linear optimized training for Llama-2-70B-chat (FSDP sharded).

Same approach as train_grpo_fp8.py with aggressive memory management:
- gc.collect() + empty_cache() between generation and training
- Explicit memory tracking at each phase
- FP8 Linear (dynamic scaling) via LumenConfig applied post-FSDP

Note: LumenConfig FP8 Linear only quantizes the GEMM compute path —
weights stay BF16. True memory savings require FP8ParamManager.

Launch:
  accelerate launch --config_file fsdp_8gpu.yaml \
    --num_processes 8 --main_process_port 29502 \
    examples/rl/trl/benchmark/llama-2-70b-chat/train_grpo_fp8_optimized.py --max-steps 10
"""

import argparse
import gc
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
    "/workspace/Lumen/outputs/benchmark/llama-2-70b-chat/lumen_fp8_optimized",
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
        self._step_times = []

    def on_step_begin(self, args, state, control, **kwargs):
        self._t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return
        step_time = round(time.perf_counter() - self._t0, 2)
        self._step_times.append(step_time)
        record = {"step": state.global_step, "step_time": step_time}
        if torch.cuda.is_available():
            record["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            record["current_mem_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
            record["reserved_mem_gb"] = round(torch.cuda.memory_reserved() / 1e9, 2)
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                record[k] = v
        with open(self._log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"[fp8_70b] Step {state.global_step}: "
              f"time={step_time}s  peak_mem={record.get('peak_mem_gb', '?')} GB  "
              f"current={record.get('current_mem_gb', '?')} GB  "
              f"reserved={record.get('reserved_mem_gb', '?')} GB",
              flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self._step_times:
            avg = sum(self._step_times) / len(self._step_times)
            print(f"[fp8_70b] Avg step time: {avg:.2f}s over {len(self._step_times)} steps",
                  flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=10)
    cli_args, _ = parser.parse_known_args()

    num_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    rank = int(os.environ.get("RANK", 0))

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

    print(f"[fp8_70b] Rank={rank}  Model={MODEL_PATH}  GPUs={num_gpus}  "
          f"per_device_bs={per_device_bs}  grad_accum={grad_accum}  "
          f"effective_batch={num_gpus * per_device_bs * grad_accum}", flush=True)

    trainer = GRPOTrainer(
        model=MODEL_PATH,
        reward_funcs=accuracy_reward,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[_PerfLogCallback(OUTPUT_DIR)],
    )

    print(f"[fp8_70b] Rank={rank} Applying LumenConfig FP8 Linear (dynamic scaling)...", flush=True)
    cfg = LumenConfig(format="fp8_e4m3", scaling="dynamic")
    _, trainer.model = cfg.enable(trainer.model)
    print(f"[fp8_70b] Rank={rank} LumenConfig FP8 Linear applied!", flush=True)

    gc.collect()
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        pre_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"[fp8_70b] Rank={rank} Pre-training peak GPU memory: {pre_mem:.2f} GB", flush=True)

    trainer.train()

    if torch.cuda.is_available() and rank == 0:
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[fp8_70b] Final peak GPU memory: {peak_gb:.2f} GB", flush=True)
    if rank == 0:
        print("[fp8_70b] FP8 optimized training COMPLETE!", flush=True)


if __name__ == "__main__":
    main()
