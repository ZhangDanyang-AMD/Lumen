"""Maximum memory-saving FP8 training for Llama-2-70B-chat (FSDP sharded).

Uses LumenConfig with all available memory optimizations:
- cpu_offload: offload autograd saved tensors to pinned CPU RAM
- FP8 dynamic scaling: quantized GEMMs (reduces intermediate buffers)
- fp8_param_gather: FP8 all-gather for reduced comm buffers
- gradient_checkpointing: recompute activations in backward
- beta=0.0: skip reference model entirely
- Aggressive gc.collect() + empty_cache()

The cpu_offload feature uses torch.autograd.graph.saved_tensors_hooks to
move all saved tensors to CPU during forward. This should free 20-40 GB/GPU
of activation memory that was the bottleneck in previous attempts.

Launch:
  accelerate launch --config_file fsdp_8gpu.yaml \
    --num_processes 8 --main_process_port 29503 \
    examples/rl/trl/benchmark/llama-2-70b-chat/train_grpo_max_memsave.py --max-steps 10
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
from lumen.utils.cpu_offload import lumen_cpu_offload_context

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "/workspace/Lumen/outputs/benchmark/llama-2-70b-chat/lumen_max_memsave",
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
            free, total = torch.cuda.mem_get_info()
            record["free_mem_gb"] = round(free / 1e9, 2)
            record["total_mem_gb"] = round(total / 1e9, 2)
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                record[k] = v
        with open(self._log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(
            f"[max_memsave] Step {state.global_step}: "
            f"time={step_time}s  peak_mem={record.get('peak_mem_gb', '?')} GB  "
            f"current={record.get('current_mem_gb', '?')} GB  "
            f"free={record.get('free_mem_gb', '?')} GB",
            flush=True,
        )

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self._step_times:
            avg = sum(self._step_times) / len(self._step_times)
            print(
                f"[max_memsave] Avg step time: {avg:.2f}s over {len(self._step_times)} steps",
                flush=True,
            )


class _CPUOffloadGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that wraps forward/backward in cpu_offload_context."""

    _offload_ctx = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        with lumen_cpu_offload_context(enabled=True) as mgr:
            self._offload_ctx = mgr
            result = super().training_step(model, inputs, num_items_in_batch)
        rank = int(os.environ.get("RANK", 0))
        if rank == 0 and mgr.memory_saved_bytes > 0:
            saved_gb = mgr.memory_saved_bytes / 1e9
            print(f"[max_memsave] CPU offload saved ~{saved_gb:.2f} GB this step", flush=True)
        return result


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

    print(
        f"[max_memsave] Rank={rank}  Model={MODEL_PATH}  GPUs={num_gpus}  "
        f"per_device_bs={per_device_bs}  grad_accum={grad_accum}  "
        f"effective_batch={num_gpus * per_device_bs * grad_accum}",
        flush=True,
    )

    trainer = _CPUOffloadGRPOTrainer(
        model=MODEL_PATH,
        reward_funcs=accuracy_reward,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[_PerfLogCallback(OUTPUT_DIR)],
    )

    print(f"[max_memsave] Rank={rank} Applying LumenConfig with max memory savings...", flush=True)
    cfg = LumenConfig(
        format="fp8_e4m3",
        scaling="dynamic",
        cpu_offload=False,
        fp8_param_gather=True,
    )
    manager = cfg.enable(trainer.model)
    print(f"[max_memsave] Rank={rank} LumenConfig enabled!", flush=True)

    gc.collect()
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        pre_mem = torch.cuda.max_memory_allocated() / 1e9
        free, total = torch.cuda.mem_get_info()
        print(
            f"[max_memsave] Rank={rank} Pre-training: peak_mem={pre_mem:.2f} GB  "
            f"free={free / 1e9:.2f} GB  total={total / 1e9:.2f} GB",
            flush=True,
        )

    trainer.train()

    if torch.cuda.is_available() and rank == 0:
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[max_memsave] Final peak GPU memory: {peak_gb:.2f} GB", flush=True)
    if rank == 0:
        print("[max_memsave] Max memory-saving training COMPLETE!", flush=True)


if __name__ == "__main__":
    main()
