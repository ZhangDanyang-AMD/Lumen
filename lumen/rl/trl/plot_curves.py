#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plot GRPO training curves from the JSONL eval log.

Usage:
    python -m lumen.rl.trl.plot_curves /path/to/output_dir

Reads ``grpo_eval_log.jsonl`` produced by :class:`GRPOEvalCallback` and
writes ``grpo_curves.png`` to the same directory.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_log(log_path: Path) -> list[dict]:
    entries = []
    with open(log_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _extract(entries: list[dict], key: str, *, filter_outliers: bool = False) -> tuple[list, list]:
    steps, vals = [], []
    for e in entries:
        if key in e and e[key] is not None:
            steps.append(e["step"])
            vals.append(float(e[key]))
    if filter_outliers and len(vals) >= 3:
        import numpy as _np
        arr = _np.array(vals)
        med = _np.median(arr)
        mad = _np.median(_np.abs(arr - med))
        if mad > 0:
            threshold = 10 * mad
            mask = _np.abs(arr - med) <= threshold
            steps = [s for s, m in zip(steps, mask) if m]
            vals = [v for v, m in zip(vals, mask) if m]
    return steps, vals


def plot(entries: list[dict], out_path: Path):
    panels = [
        ("reward", "Reward vs. Step", "Reward"),
        ("entropy", "Entropy vs. Step (KL proxy)", "Entropy"),
        ("completions/mean_length", "Response Length vs. Step", "Mean Length"),
        ("win_rate", "Win Rate vs. Step (over baseline)", "Win Rate"),
        ("loss", "Loss vs. Step", "Loss"),
        ("grad_norm", "Gradient Norm vs. Step", "Grad Norm"),
    ]

    _OUTLIER_KEYS = {"entropy", "grad_norm"}

    active = [(key, title, ylabel) for key, title, ylabel in panels if _extract(entries, key)[0]]
    if not active:
        print("No plottable data found in log.", file=sys.stderr)
        return

    n = len(active)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), squeeze=False)

    for idx, (key, title, ylabel) in enumerate(active):
        ax = axes[idx, 0]
        steps, vals = _extract(entries, key, filter_outliers=(key in _OUTLIER_KEYS))
        ax.plot(steps, vals, marker="o", markersize=4, linewidth=1.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved curves to {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot GRPO training curves.")
    parser.add_argument("output_dir", type=str, help="Directory containing grpo_eval_log.jsonl")
    args = parser.parse_args()

    log_path = Path(args.output_dir) / "grpo_eval_log.jsonl"
    if not log_path.exists():
        print(f"Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    entries = load_log(log_path)
    out_path = Path(args.output_dir) / "grpo_curves.png"
    plot(entries, out_path)


if __name__ == "__main__":
    main()
