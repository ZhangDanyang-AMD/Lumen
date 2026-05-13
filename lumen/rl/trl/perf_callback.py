###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Performance-instrumented callback for GRPO benchmark runs.

Captures per-step stage-level timings (generation, forward, backward,
optimizer+comm) alongside the standard training quality metrics. All
timings use torch.cuda.synchronize() + time.perf_counter() to ensure
accurate wall-clock measurements on GPU.

Designed for the Lumen FP8 GRPO benchmark spec. See
lumen-fp8-grpo-benchmark-spec.md for metric definitions.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import TrainerCallback

__all__ = ["GRPOPerfCallback"]

_QUALITY_KEYS = {
    "loss",
    "grad_norm",
    "learning_rate",
    "reward",
    "reward_std",
    "entropy",
    "completions/mean_length",
    "completions/min_length",
    "completions/max_length",
    "completions/clipped_ratio",
    "completions/mean_terminated_length",
    "num_tokens",
    "epoch",
    "rewards/reward_fn/mean",
    "rewards/reward_fn/std",
    "frac_reward_zero_std",
}


def _sync_time() -> float:
    """GPU-synchronized wall clock timestamp."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class GRPOPerfCallback(TrainerCallback):
    """Captures stage-level timings and quality metrics per GRPO step.

    Timing is split into:
      - generation: model.generate() rollout
      - forward: actor + reference log-prob computation
      - backward: loss.backward()
      - optim_comm: optimizer.step() + FSDP communication residual

    The forward/backward/optim_comm split is approximate because FSDP2
    overlaps communication with backward computation. The training
    subtotal (= step_total - generation) is exact.

    Memory is captured every step (debug) with a headline snapshot at
    the configured steady-state step.
    """

    def __init__(
        self,
        output_dir: str,
        warmup_steps: int = 10,
        memory_snapshot_step: int = 20,
    ):
        super().__init__()
        self._log_path = Path(output_dir) / "grpo_perf_log.jsonl"
        self._warmup_steps = warmup_steps
        self._memory_snapshot_step = memory_snapshot_step
        self._history: list[dict] = []
        self._reward_history: list[float] = []

        self._step_start: float = 0.0
        self._gen_end: float = 0.0
        self._fwd_end: float = 0.0
        self._bwd_end: float = 0.0

    # ------------------------------------------------------------------
    # Timing marks — called by the benchmark runner at known boundaries
    # ------------------------------------------------------------------

    def mark_step_start(self) -> None:
        self._step_start = _sync_time()

    def mark_generation_end(self) -> None:
        self._gen_end = _sync_time()

    def mark_forward_end(self) -> None:
        self._fwd_end = _sync_time()

    def mark_backward_end(self) -> None:
        self._bwd_end = _sync_time()

    def _compute_timings(self, step_end: float) -> dict:
        total = step_end - self._step_start
        gen = self._gen_end - self._step_start
        fwd = self._fwd_end - self._gen_end
        bwd = self._bwd_end - self._fwd_end
        optim_comm = step_end - self._bwd_end
        training = step_end - self._gen_end

        if self._step_start == 0.0:
            return {}

        return {
            "step_time_total": round(total, 4),
            "step_time_generation": round(gen, 4),
            "step_time_training": round(training, 4),
            "step_time_forward": round(fwd, 4),
            "step_time_backward": round(bwd, 4),
            "step_time_optim_comm": round(optim_comm, 4),
        }

    # ------------------------------------------------------------------
    # Memory capture
    # ------------------------------------------------------------------

    def _capture_memory(self, step: int) -> dict:
        if not torch.cuda.is_available():
            return {}

        if step == self._memory_snapshot_step - 1:
            torch.cuda.reset_peak_memory_stats()

        alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

        result = {
            "peak_memory_allocated_gb": round(alloc, 3),
            "peak_memory_reserved_gb": round(reserved, 3),
        }

        if step == self._memory_snapshot_step:
            result["is_headline_memory"] = True

        return result

    # ------------------------------------------------------------------
    # TrainerCallback hooks
    # ------------------------------------------------------------------

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return

        step = state.global_step
        step_end = _sync_time()

        record = {"step": step}

        timings = self._compute_timings(step_end)
        record.update(timings)

        record.update(self._capture_memory(step))

        if timings.get("step_time_total"):
            tokens = logs.get("num_tokens", 0)
            record["tokens_processed"] = tokens

        _SKIP = {"total_flos", "train_runtime", "train_samples_per_second",
                 "train_steps_per_second", "train_loss"}
        for key, val in logs.items():
            if key in _SKIP or key in record:
                continue
            if isinstance(val, (int, float)):
                if key == "entropy" and val < -100:
                    record["entropy_raw"] = val
                    record[key] = None
                else:
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
        self._flush()

    def _flush(self):
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "w") as fh:
            for entry in self._history:
                fh.write(json.dumps(entry) + "\n")

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def get_measurement_stats(self) -> dict:
        """Compute mean/std of timing metrics over non-warmup steps."""
        measured = [e for e in self._history if not e.get("is_warmup", True)]
        if not measured:
            return {}

        stats = {}
        for key in [
            "step_time_total",
            "step_time_generation",
            "step_time_training",
            "step_time_forward",
            "step_time_backward",
            "step_time_optim_comm",
        ]:
            vals = [e[key] for e in measured if key in e and e[key] is not None]
            if vals:
                arr = np.array(vals)
                stats[key] = {"mean": round(float(arr.mean()), 4), "std": round(float(arr.std()), 4)}
        return stats
