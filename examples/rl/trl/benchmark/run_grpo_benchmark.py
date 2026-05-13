###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Benchmark runner for Lumen FP8 GRPO training.

Supports all run configurations from the benchmark spec:
  R1: BF16 baseline (no Lumen)
  R2: Lumen BF16 (actor build + FSDP2)
  R3: Lumen FP8 Linear only
  R4: Lumen FP8 Full suite
  R5: Lumen FP8 Full + LoRA

Usage:
  # R1 — pure TRL baseline
  python run_grpo_benchmark.py --run-id R1 --model-name-or-path /dev/shm/model/llama-3.1-8b ...

  # R4 — full Lumen FP8
  python run_grpo_benchmark.py --run-id R4 --model-name-or-path /dev/shm/model/llama-3.1-8b ...
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def _extract_boxed(text: str) -> str | None:
    """Extract the last \\boxed{...} content from model output."""
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def _normalize_answer(s: str) -> str:
    """Normalize a math answer string for comparison."""
    s = s.strip()
    s = s.replace(" ", "")
    s = s.replace("\\,", "")
    s = s.replace("{", "").replace("}", "")
    s = s.rstrip(".")
    try:
        return str(float(s))
    except ValueError:
        return s.lower()


def reward_fn(prompts, completions, expected_answer=None, **kwargs):
    """Math correctness reward: 1.0 if boxed answer matches expected, 0.0 otherwise."""
    rewards = []
    for i, completion in enumerate(completions):
        if expected_answer is None:
            rewards.append(0.0)
            continue
        gold = expected_answer[i] if isinstance(expected_answer, list) else expected_answer
        predicted = _extract_boxed(completion)
        if predicted is None:
            rewards.append(0.0)
        elif _normalize_answer(predicted) == _normalize_answer(str(gold)):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def _get_reward_func(dataset):
    """Select reward function based on dataset columns."""
    cols = dataset.column_names
    if "solution" in cols:
        return accuracy_reward
    if "expected_answer" in cols:
        return reward_fn
    return reward_fn


def _ensure_prompt_column(dataset):
    if "prompt" in dataset.column_names:
        return dataset
    if "problem" in dataset.column_names:
        dataset = dataset.rename_column("problem", "prompt")
        drop_cols = [c for c in ["generated_solution", "problem_source"] if c in dataset.column_names]
        if drop_cols:
            dataset = dataset.remove_columns(drop_cols)
        return dataset
    if "messages" in dataset.column_names:
        def _extract(example):
            for msg in example["messages"]:
                if msg["role"] == "user":
                    return {"prompt": msg["content"]}
            return {"prompt": ""}
        return dataset.map(_extract)
    raise ValueError(
        f"Dataset must have a 'prompt', 'problem', or 'messages' column. "
        f"Found: {dataset.column_names}"
    )


def _load_dataset(args):
    if args.train_data_path:
        ds = load_dataset("json", data_files=args.train_data_path, split="train")
    else:
        ds = load_dataset(args.dataset_name, args.dataset_config_name, split=args.dataset_split)
    return _ensure_prompt_column(ds)


def _collect_env_info(args) -> dict:
    """Collect software versions for reproducibility (spec §10)."""
    info = {
        "run_id": args.run_id,
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "torch_git": getattr(torch.version, "git_version", "unknown"),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)

    try:
        import trl
        info["trl_version"] = trl.__version__
    except Exception:
        pass
    try:
        import accelerate
        info["accelerate_version"] = accelerate.__version__
    except Exception:
        pass
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parents[4],
        )
        if result.returncode == 0:
            info["lumen_commit"] = result.stdout.strip()
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "ROCm" in line and "Version" in line:
                    info["rocm_version"] = line.strip()
                    break
    except Exception:
        pass

    if hasattr(torch.version, "hip"):
        info["hip_version"] = torch.version.hip or "unknown"

    return info


# =========================================================================
# R1: Pure TRL baseline (no Lumen imports)
# =========================================================================

class _ConvergenceChecker:
    """Convergence-based early stopping per rl-training-evaluation-guide.md section 3.

    Stops training when ANY of these fire:
      - Reward plateau: EMA reward improves < `reward_tol` over `patience` steps
      - KL explosion: KL divergence > `kl_threshold`
      - Entropy collapse: entropy < 10% of initial mean
      - Length collapse: mean_length < 50 tokens
    """

    def __init__(
        self,
        patience: int = 15,
        reward_tol: float = 0.02,
        kl_threshold: float = 15.0,
        ema_alpha: float = 0.3,
        min_steps: int = 20,
    ):
        self.patience = patience
        self.reward_tol = reward_tol
        self.kl_threshold = kl_threshold
        self.ema_alpha = ema_alpha
        self.min_steps = min_steps

        self._ema: float | None = None
        self._best_ema: float = -float("inf")
        self._steps_since_improve: int = 0
        self._initial_entropies: list[float] = []
        self._initial_entropy_mean: float | None = None
        self.stop_reason: str | None = None

    def update(self, step: int, reward: float | None, kl: float | None,
               entropy: float | None, mean_length: float | None) -> bool:
        """Return True if training should stop."""
        if step < self.min_steps:
            if entropy is not None and len(self._initial_entropies) < 5:
                self._initial_entropies.append(entropy)
                if len(self._initial_entropies) == 5:
                    self._initial_entropy_mean = sum(self._initial_entropies) / 5
            return False

        if reward is not None:
            if self._ema is None:
                self._ema = reward
            else:
                self._ema = self.ema_alpha * reward + (1 - self.ema_alpha) * self._ema

            if self._ema > self._best_ema + self.reward_tol:
                self._best_ema = self._ema
                self._steps_since_improve = 0
            else:
                self._steps_since_improve += 1

            if self._steps_since_improve >= self.patience:
                self.stop_reason = (
                    f"Reward plateau: EMA={self._ema:.4f}, best={self._best_ema:.4f}, "
                    f"no improvement for {self._steps_since_improve} steps"
                )
                return True

        if kl is not None and kl > self.kl_threshold:
            self.stop_reason = f"KL explosion: {kl:.2f} > threshold {self.kl_threshold}"
            return True

        if (entropy is not None and self._initial_entropy_mean is not None
                and self._initial_entropy_mean > 0
                and entropy < 0.1 * self._initial_entropy_mean):
            self.stop_reason = (
                f"Entropy collapse: {entropy:.3f} < 10% of initial {self._initial_entropy_mean:.3f}"
            )
            return True

        if mean_length is not None and mean_length < 50:
            self.stop_reason = f"Length collapse: mean_length={mean_length:.0f} < 50"
            return True

        return False


class _PeriodicEvalCallback(TrainerCallback):
    """Periodically evaluate the model on GSM8K during training.

    Produces an accuracy-vs-step curve (ref: arXiv:2512.07611 style).
    Eval runs on rank 0 only, using greedy decoding on a small GSM8K subset.
    """

    def __init__(self, output_dir: str, eval_every: int = 25,
                 eval_samples: int = 50, tokenizer=None):
        super().__init__()
        self._output_dir = Path(output_dir)
        self._eval_every = eval_every
        self._eval_samples = eval_samples
        self._tokenizer = tokenizer
        self._eval_log_path = self._output_dir / "gsm8k_accuracy_curve.jsonl"
        self._eval_ds = None
        self._results: list[dict] = []

    def _load_eval_ds(self):
        if self._eval_ds is not None:
            return
        try:
            ds = load_dataset("openai/gsm8k", "main", split="test")
            if len(ds) > self._eval_samples:
                ds = ds.select(range(self._eval_samples))
            self._eval_ds = ds
        except Exception as e:
            print(f"  [PeriodicEval] Could not load GSM8K: {e}")

    def _run_eval(self, model, step: int):
        self._load_eval_ds()
        if self._eval_ds is None:
            return
        tok = self._tokenizer
        was_training = model.training
        model.eval()

        correct = 0
        total = 0
        for ex in self._eval_ds:
            q = ex["question"]
            gold_raw = ex["answer"]
            gold = gold_raw.split("####")[-1].strip() if "####" in gold_raw else gold_raw.strip()
            prompt = f"Solve the following math problem step by step. Put your final answer in \\boxed{{}}.\n\n{q}\n\n"
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            comp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            predicted = _extract_boxed(comp)
            if predicted is not None and _normalize_answer(predicted) == _normalize_answer(gold):
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0.0
        record = {"step": step, "gsm8k_accuracy": acc, "correct": correct, "total": total}
        self._results.append(record)

        self._eval_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._eval_log_path, "w") as fh:
            for entry in self._results:
                fh.write(json.dumps(entry) + "\n")
        print(f"  [GSM8K Eval] Step {step}: {correct}/{total} = {acc:.1%}")

        if was_training:
            model.train()

    def on_log(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        step = state.global_step
        if step > 0 and step % self._eval_every == 0:
            if model is None:
                return
            actual_model = model
            if hasattr(actual_model, "module"):
                actual_model = actual_model.module
            self._run_eval(actual_model, step)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        step = state.global_step
        if not self._results or self._results[-1]["step"] != step:
            if model is None:
                return
            actual_model = model
            if hasattr(actual_model, "module"):
                actual_model = actual_model.module
            self._run_eval(actual_model, step)


class _BaselinePerfCallback(TrainerCallback):
    """Lightweight perf callback for R1 baseline — no Lumen dependency.

    Includes convergence-based early stopping when `convergence_checker` is provided.
    """

    def __init__(self, output_dir: str, warmup_steps: int = 10,
                 convergence_checker: _ConvergenceChecker | None = None):
        super().__init__()
        self._log_path = Path(output_dir) / "grpo_perf_log.jsonl"
        self._warmup_steps = warmup_steps
        self._history: list[dict] = []
        self._reward_history: list[float] = []
        self._step_start: float = 0.0
        self._convergence = convergence_checker

    def on_step_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._step_start = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_end = time.perf_counter()

        step = state.global_step
        record = {"step": step}

        if self._step_start > 0:
            record["step_time_total"] = round(step_end - self._step_start, 4)

        if torch.cuda.is_available():
            record["peak_memory_allocated_gb"] = round(
                torch.cuda.max_memory_allocated() / (1024**3), 3
            )
            record["peak_memory_reserved_gb"] = round(
                torch.cuda.max_memory_reserved() / (1024**3), 3
            )

        record["tokens_processed"] = logs.get("num_tokens", 0)

        _SKIP = {"total_flos", "train_runtime", "train_samples_per_second",
                 "train_steps_per_second", "train_loss"}
        for key, val in logs.items():
            if key in _SKIP or key in record:
                continue
            if isinstance(val, (int, float)):
                record[key] = val

        reward_mean = logs.get("reward") if "reward" in logs else logs.get("rewards/reward_fn/mean")
        if reward_mean is not None:
            reward_mean = float(reward_mean)
            self._reward_history.append(reward_mean)
            n = len(self._reward_history)
            if n <= 3:
                record["win_rate"] = 0.0
            else:
                baseline = sum(self._reward_history[:3]) / 3
                recent = self._reward_history[-5:]
                record["win_rate"] = sum(1 for r in recent if r > baseline) / len(recent)

        record["is_warmup"] = step <= self._warmup_steps

        self._history.append(record)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "w") as fh:
            for entry in self._history:
                fh.write(json.dumps(entry) + "\n")

        if self._convergence is not None:
            should_stop = self._convergence.update(
                step=step,
                reward=reward_mean if reward_mean is not None else None,
                kl=logs.get("kl"),
                entropy=logs.get("entropy"),
                mean_length=logs.get("completions/mean_length"),
            )
            if should_stop:
                reason = self._convergence.stop_reason
                record["early_stop_reason"] = reason
                with open(self._log_path, "w") as fh:
                    for entry in self._history:
                        fh.write(json.dumps(entry) + "\n")
                print(f"\n  [Early Stop] Step {step}: {reason}")
                control.should_training_stop = True


def run_baseline(args, train_dataset):
    """R1: Pure TRL baseline — no Lumen imports at all."""
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path or args.model_name_or_path
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    save_strat = "no"
    save_steps_val = 500
    if args.periodic_eval:
        save_strat = "steps"
        save_steps_val = args.periodic_eval_every

    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        warmup_steps=args.lr_warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.log_interval,
        save_strategy=save_strat,
        save_steps=save_steps_val,
        save_total_limit=10,
        bf16=True,
        gradient_checkpointing=True,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        beta=args.beta,
        temperature=1.0,
        top_p=1.0,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        use_vllm=False,
    )

    conv_checker = None
    if args.early_stop:
        conv_checker = _ConvergenceChecker(
            patience=args.early_stop_patience,
            reward_tol=0.02,
            min_steps=max(args.perf_warmup_steps + 5, 20),
        )

    perf_cb = _BaselinePerfCallback(
        output_dir=args.output_dir,
        warmup_steps=args.perf_warmup_steps,
        convergence_checker=conv_checker,
    )

    chosen_reward = _get_reward_func(train_dataset)

    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        reward_funcs=chosen_reward,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[perf_cb],
    )
    trainer.train()
    return perf_cb, trainer


# =========================================================================
# R2-R5: Lumen path (imports Lumen only here)
# =========================================================================

def run_lumen(args, train_dataset):
    """R2-R5: Lumen actor build path with optional FP8/LoRA."""
    from lumen.rl.trl.args import TrlLumenArgs
    from lumen.rl.trl.modeling import build_actor_model
    from lumen.rl.trl.perf_callback import GRPOPerfCallback
    from lumen.rl.trl.warmup import maybe_run_synthetic_warmup

    seq_length = max(args.max_prompt_length + args.max_completion_length, 4096)

    lumen_args = TrlLumenArgs(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_split=args.dataset_split,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        fsdp_version=args.fsdp_version,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        lr=args.lr,
        lr_warmup_steps=args.lr_warmup_steps,
        log_interval=args.log_interval,
        save_interval=0,
        seq_length=seq_length,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        warmup_steps=args.lumen_warmup_steps,
        linear_fp8=args.linear_fp8,
        linear_fp8_activation=args.fp8_activation,
        linear_fp8_wgrad=args.fp8_wgrad,
        linear_fp8_reduce_amax=args.fp8_reduce_amax,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        gradient_checkpointing=True,
        seed=args.seed,
        train_dataset=train_dataset,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path or args.model_name_or_path
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = build_actor_model(lumen_args)

    from lumen.rl.trl.args import build_grpo_config_kwargs
    config_kwargs = build_grpo_config_kwargs(lumen_args)
    config = GRPOConfig(**config_kwargs)

    perf_cb = GRPOPerfCallback(
        output_dir=args.output_dir,
        warmup_steps=args.perf_warmup_steps,
        memory_snapshot_step=args.perf_warmup_steps + 10,
    )

    callbacks = [perf_cb]
    if args.early_stop:
        conv_checker = _ConvergenceChecker(
            patience=args.early_stop_patience,
            reward_tol=0.02,
            min_steps=max(args.perf_warmup_steps + 5, 20),
        )
        conv_cb = _BaselinePerfCallback(
            output_dir=args.output_dir + "/_convergence_tmp",
            warmup_steps=args.perf_warmup_steps,
            convergence_checker=conv_checker,
        )
        callbacks.append(conv_cb)

    if args.periodic_eval:
        eval_cb = _PeriodicEvalCallback(
            output_dir=args.output_dir,
            eval_every=args.periodic_eval_every,
            eval_samples=args.periodic_eval_samples,
            tokenizer=tokenizer,
        )
        callbacks.append(eval_cb)

    from lumen.rl.trl.patched_trainer import PatchedGRPOTrainer

    trainer = PatchedGRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    maybe_run_synthetic_warmup(getattr(trainer, "model", model), lumen_args, device=device)

    trainer.train()
    return perf_cb, trainer


# =========================================================================
# CLI
# =========================================================================

_RUN_CONFIGS = {
    "R1": {"linear_fp8": False, "lora_rank": 0, "use_lumen": False,
            "fp8_activation": False, "fp8_wgrad": False, "fp8_reduce_amax": False},
    "R2": {"linear_fp8": False, "lora_rank": 0, "use_lumen": True,
            "fp8_activation": False, "fp8_wgrad": False, "fp8_reduce_amax": False},
    "R3": {"linear_fp8": True, "lora_rank": 0, "use_lumen": True,
            "fp8_activation": False, "fp8_wgrad": False, "fp8_reduce_amax": False},
    "R4": {"linear_fp8": True, "lora_rank": 0, "use_lumen": True,
            "fp8_activation": True, "fp8_wgrad": True, "fp8_reduce_amax": True},
    "R5": {"linear_fp8": True, "lora_rank": 32, "use_lumen": True,
            "fp8_activation": True, "fp8_wgrad": True, "fp8_reduce_amax": True},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lumen FP8 GRPO Benchmark Runner (spec: lumen-fp8-grpo-benchmark-spec.md)"
    )
    parser.add_argument("--run-id", required=True, choices=list(_RUN_CONFIGS.keys()),
                        help="Run configuration ID from benchmark spec")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--tokenizer-name-or-path", type=str, default=None)
    parser.add_argument("--output-dir", required=True)

    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--dataset-name", type=str, default=None)
    data.add_argument("--train-data-path", type=str, default=None)
    parser.add_argument("--dataset-config-name", type=str, default=None)
    parser.add_argument("--dataset-split", type=str, default="train")

    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear",
                        help="LR scheduler: linear (TRL default), cosine, or constant_with_warmup")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.0,
                        help="KL penalty coefficient (0.0 = no KL, per TRL/DAPO best practice)")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fsdp-version", type=int, default=2, choices=[1, 2])

    parser.add_argument("--perf-warmup-steps", type=int, default=10,
                        help="Steps excluded from perf measurement")
    parser.add_argument("--lumen-warmup-steps", type=int, default=0,
                        help="Lumen synthetic FP8 warmup steps")

    parser.add_argument("--linear-fp8", action="store_true", default=None,
                        help="Override: enable FP8 linear (auto-set by run-id)")
    parser.add_argument("--lora-rank", type=int, default=None,
                        help="Override: LoRA rank (auto-set by run-id)")
    parser.add_argument("--lora-alpha", type=float, default=32.0)

    parser.add_argument("--eval-after-training", action="store_true", default=False,
                        help="Run GSM8K eval after training (rank 0 only)")
    parser.add_argument("--eval-samples", type=int, default=200,
                        help="Number of GSM8K test samples to evaluate")

    parser.add_argument("--early-stop", action="store_true", default=False,
                        help="Enable convergence-based early stopping")
    parser.add_argument("--early-stop-patience", type=int, default=15,
                        help="Steps without EMA reward improvement before stopping")

    parser.add_argument("--periodic-eval", action="store_true", default=False,
                        help="Run GSM8K eval periodically during training (rank 0)")
    parser.add_argument("--periodic-eval-every", type=int, default=25,
                        help="Eval every N steps")
    parser.add_argument("--periodic-eval-samples", type=int, default=50,
                        help="Number of GSM8K samples per periodic eval")

    return parser.parse_args()


def main():
    args = parse_args()

    run_cfg = _RUN_CONFIGS[args.run_id]
    if args.linear_fp8 is None:
        args.linear_fp8 = run_cfg["linear_fp8"]
    if args.lora_rank is None:
        args.lora_rank = run_cfg["lora_rank"]
    args.fp8_activation = run_cfg["fp8_activation"]
    args.fp8_wgrad = run_cfg["fp8_wgrad"]
    args.fp8_reduce_amax = run_cfg["fp8_reduce_amax"]
    use_lumen = run_cfg["use_lumen"]

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(f"\n{'='*60}")
        print(f"  Lumen FP8 GRPO Benchmark — Run {args.run_id}")
        print(f"  Model: {args.model_name_or_path}")
        fp8_desc = "off"
        if args.linear_fp8:
            fp8_desc = "linear" if not args.fp8_activation else "full (act+wgrad+reduce)"
        print(f"  FP8: {fp8_desc} | LoRA: r={args.lora_rank} | Lumen: {use_lumen}")
        print(f"  Steps: {args.max_steps} (perf warmup: {args.perf_warmup_steps})")
        print(f"{'='*60}\n")

    train_dataset = _load_dataset(args)

    env_info = _collect_env_info(args)
    if local_rank == 0:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "env_info.json", "w") as f:
            json.dump(env_info, f, indent=2)

    if use_lumen:
        perf_cb, trainer = run_lumen(args, train_dataset)
    else:
        perf_cb, trainer = run_baseline(args, train_dataset)

    if local_rank == 0 and hasattr(perf_cb, "get_measurement_stats"):
        stats = perf_cb.get_measurement_stats()
        if stats:
            print(f"\n{'='*60}")
            print(f"  Performance Summary — Run {args.run_id}")
            print(f"{'='*60}")
            for key, val in stats.items():
                print(f"  {key}: {val['mean']:.3f}s ± {val['std']:.3f}s")
            print(f"{'='*60}\n")

            with open(Path(args.output_dir) / "perf_summary.json", "w") as f:
                json.dump({"run_id": args.run_id, "stats": stats}, f, indent=2)

    if local_rank == 0 and args.eval_after_training:
        run_post_eval(args, trainer)


def run_post_eval(args, trainer=None):
    """Evaluate trained model on GSM8K test set (rank 0 only).

    Uses the in-memory trainer model when available, avoiding a separate model load.
    """
    print(f"\n{'='*60}")
    print(f"  Post-Training Eval — Run {args.run_id}")
    print(f"  Dataset: openai/gsm8k (test split, first {args.eval_samples} samples)")
    print(f"{'='*60}\n")

    try:
        eval_ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:
        print(f"  [WARN] Could not load GSM8K: {e}. Skipping eval.")
        return

    if len(eval_ds) > args.eval_samples:
        eval_ds = eval_ds.select(range(args.eval_samples))

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path or args.model_name_or_path
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if trainer is not None:
        model = trainer.model
        if hasattr(model, "module"):
            model = model.module
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=torch.bfloat16
        )
    model.eval()
    model.to("cuda:0")

    correct = 0
    total = 0
    results = []

    for example in eval_ds:
        question = example["question"]
        gold_raw = example["answer"]
        gold = gold_raw.split("####")[-1].strip() if "####" in gold_raw else gold_raw.strip()

        prompt = f"Solve the following math problem step by step. Put your final answer in \\boxed{{}}.\n\n{question}\n\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda:0")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
            )

        completion = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted = _extract_boxed(completion)

        is_correct = False
        if predicted is not None:
            is_correct = _normalize_answer(predicted) == _normalize_answer(gold)

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question[:100],
            "gold": gold,
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0.0
    print(f"\n  GSM8K Eval: {correct}/{total} = {accuracy:.1%}")

    eval_path = Path(args.output_dir) / "gsm8k_eval.json"
    with open(eval_path, "w") as f:
        json.dump({
            "run_id": args.run_id,
            "dataset": "gsm8k",
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "results": results,
        }, f, indent=2)
    print(f"  Results saved to {eval_path}\n")

    if trainer is None:
        del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
