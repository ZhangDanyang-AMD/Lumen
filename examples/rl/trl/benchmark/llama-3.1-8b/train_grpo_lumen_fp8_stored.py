"""FP8-cached-weights run for Llama-3.1-8B (dynamic scaling + FP8 weight cache).

Pre-quantizes nn.Linear weights to FP8 and caches them.  During forward,
the cached FP8 weights are fed directly to the GEMM, skipping per-forward
weight quantization.  After each optimizer step, a post-hook re-quantizes
the updated BF16 master weights into the FP8 cache.

The BF16 nn.Parameter is kept for DDP gradient sync and the optimizer.
Memory cost: +~7.5 GB FP8 cache on top of 16 GB BF16 master.
Benefit: eliminates per-forward weight quantization latency (~35% faster
than FP8 dynamic).

Launch:
  accelerate launch --config_file ddp_8gpu.yaml \\
    --num_processes 8 --main_process_port 29500 \\
    examples/rl/trl/benchmark/llama-3.1-8b/train_grpo_lumen_fp8_stored.py --max-steps 1000
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward

from lumen.config import LumenConfig
import lumen.quantize as quant

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "/workspace/Lumen/outputs/benchmark/llama-3.1-8b/lumen_fp8_stored",
)
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/dev/shm/model/llama-3.1-8b",
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


class _FP8OptimizerHookCallback(TrainerCallback):
    """Register FP8 weight cache refresh hook once optimizer is available."""

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        optimizer = kwargs.get("optimizer")
        if optimizer is None or model is None:
            return
        inner_opt = getattr(optimizer, "optimizer", optimizer)
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        quant.register_fp8_weight_optimizer_hooks(unwrapped, inner_opt)
        if state.is_world_process_zero:
            print(f"[fp8_stored] Registered FP8 weight optimizer hooks on {type(inner_opt).__name__}")


def _build_fp8_stored_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    cfg = LumenConfig(format="fp8_e4m3", scaling="dynamic")
    _, model = cfg.enable(model)
    n = quant.store_weights_fp8(model, fp8_dtype=torch.float8_e4m3fn)
    print(f"[fp8_stored] Cached {n} linear weights in FP8")

    bf16_bytes = sum(p.numel() * 2 for p in model.parameters() if p.dtype == torch.bfloat16)
    cache_bytes = sum(
        m._fp8_weight_data.numel()
        for m in model.modules()
        if hasattr(m, "_fp8_weight_data")
    )
    print(f"[fp8_stored] Param memory: BF16={bf16_bytes/1e9:.2f} GB, "
          f"FP8 cache={cache_bytes/1e9:.2f} GB, "
          f"total={(bf16_bytes + cache_bytes)/1e9:.2f} GB")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=-1)
    cli_args, _ = parser.parse_known_args()

    num_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))

    per_device_bs = 1
    grad_accum = 8

    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")
    model = _build_fp8_stored_model()

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
        model_init_kwargs={"torch_dtype": "bfloat16"},
    )

    print(f"[fp8_stored] Model={MODEL_PATH}  GPUs={num_gpus}  per_device_bs={per_device_bs}  "
          f"grad_accum={grad_accum}  effective_batch={num_gpus * per_device_bs * grad_accum}")
    print("[fp8_stored] FP8 training: dynamic scaling + FP8 weight cache")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=accuracy_reward,
        args=training_args,
        train_dataset=dataset,
        callbacks=[_PerfLogCallback(OUTPUT_DIR), _FP8OptimizerHookCallback()],
    )
    trainer.train()

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[fp8_stored] Peak GPU memory: {peak_gb:.2f} GB")
    print("FP8 stored-weights training done!")


if __name__ == "__main__":
    main()
