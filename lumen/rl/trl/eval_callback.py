###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Evaluation callback for the TRL + Lumen GRPO integration.

Captures per-step training metrics from GRPOTrainer and writes them to a
JSONL file for downstream plotting.  Runs only on the main process to
avoid duplicate writes in distributed settings.
"""

import json
from pathlib import Path

from transformers import TrainerCallback

__all__ = ["GRPOEvalCallback"]

_TRACKED_KEYS = {
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
    "clip_ratio/low_mean",
    "clip_ratio/high_mean",
    "clip_ratio/region_mean",
    "num_tokens",
    "step_time",
    "epoch",
    "rewards/reward_fn/mean",
    "rewards/reward_fn/std",
    "frac_reward_zero_std",
}


class GRPOEvalCallback(TrainerCallback):
    """Captures GRPO training metrics and writes them to a JSONL log.

    Metrics collected per step:
    - **Reward curve**: reward mean/std per step
    - **Response length**: mean/min/max completion lengths, clipped ratio
    - **Policy entropy**: proxy for KL divergence (entropy decreases as
      policy diverges from a uniform prior; with ``beta=0`` the runner
      does not maintain an explicit reference model so direct KL is not
      available without an extra forward pass)
    - **Win-rate proxy**: fraction of recent steps where mean reward
      exceeds the baseline (mean of first 3 steps), computed over a
      rolling window of 5 steps.
    """

    _WIN_RATE_WARMUP = 3
    _WIN_RATE_WINDOW = 5

    def __init__(self, output_dir: str):
        super().__init__()
        self._log_path = Path(output_dir) / "grpo_eval_log.jsonl"
        self._history: list[dict] = []
        self._reward_history: list[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return

        step = state.global_step
        record = {"step": step}

        for key in _TRACKED_KEYS:
            if key in logs:
                val = logs[key]
                if key == "entropy" and isinstance(val, (int, float)) and val < -100:
                    record["entropy_raw"] = val
                    record[key] = None
                else:
                    record[key] = val

        reward_mean = logs["reward"] if "reward" in logs else logs.get("rewards/reward_fn/mean")
        if reward_mean is not None:
            reward_mean = float(reward_mean)
            self._reward_history.append(reward_mean)

            n = len(self._reward_history)
            warmup = self._WIN_RATE_WARMUP
            window = self._WIN_RATE_WINDOW

            if n <= warmup:
                record["win_rate"] = 0.0
            else:
                baseline = sum(self._reward_history[:warmup]) / warmup
                recent = self._reward_history[-window:]
                record["win_rate"] = sum(1 for r in recent if r > baseline) / len(recent)

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
