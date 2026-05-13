#!/usr/bin/env python3
###############################################################################
# Compare Lumen vs. baseline GRPO training runs.
#
# Usage:
#     python examples/rl/trl/compare_runs.py \
#         outputs/trl-grpo-70b outputs/trl-grpo-70b-baseline
###############################################################################

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
        arr = np.array(vals)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad > 0:
            threshold = 10 * mad
            mask = np.abs(arr - med) <= threshold
            steps = [s for s, m in zip(steps, mask) if m]
            vals = [v for v, m in zip(vals, mask) if m]
    return steps, vals


def pearson_r(x, y):
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    n = min(len(x), len(y))
    x, y = np.array(x[:n], dtype=float), np.array(y[:n], dtype=float)
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    denom = np.sqrt((dx ** 2).sum() * (dy ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((dx * dy).sum() / denom)


def plot_comparison(lumen_entries, baseline_entries, out_path: Path,
                    label_a: str = "Run A", label_b: str = "Run B"):
    panels = [
        ("reward", "Reward vs. Step"),
        ("entropy", "Entropy vs. Step"),
        ("completions/mean_length", "Response Length vs. Step"),
        ("win_rate", "Win Rate vs. Step"),
        ("loss", "Loss vs. Step"),
        ("grad_norm", "Gradient Norm vs. Step"),
    ]

    _OUTLIER_KEYS = {"entropy", "grad_norm"}

    active = []
    for key, title in panels:
        ls, _ = _extract(lumen_entries, key)
        bs, _ = _extract(baseline_entries, key)
        if ls or bs:
            active.append((key, title))

    if not active:
        print("No plottable data found.", file=sys.stderr)
        return

    n = len(active)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), squeeze=False)

    for idx, (key, title) in enumerate(active):
        ax = axes[idx, 0]
        filt = key in _OUTLIER_KEYS
        ls, lv = _extract(lumen_entries, key, filter_outliers=filt)
        bs, bv = _extract(baseline_entries, key, filter_outliers=filt)

        if ls:
            ax.plot(ls, lv, marker="o", markersize=3, linewidth=1.5,
                    label=label_a, color="#2563eb", alpha=0.85)
        if bs:
            ax.plot(bs, bv, marker="s", markersize=3, linewidth=1.5,
                    label=label_b, color="#dc2626", alpha=0.85)

        r = pearson_r(lv, bv) if lv and bv else float("nan")
        ax.set_title(f"{title}  (r={r:.3f})" if not np.isnan(r) else title,
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel(key.split("/")[-1].replace("_", " ").title())
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot to {out_path}")
    plt.close(fig)


def build_stats_table(lumen_entries, baseline_entries,
                      label_a: str = "Run A", label_b: str = "Run B") -> str:
    keys = [
        ("reward", "Mean Reward"),
        ("completions/mean_length", "Mean Length"),
        ("entropy", "Entropy"),
        ("loss", "Loss"),
        ("grad_norm", "Grad Norm"),
        ("win_rate", "Win Rate"),
    ]

    rows = []
    rows.append(f"| Metric | {label_a} Mean | {label_a} Std | {label_b} Mean | {label_b} Std | Pearson r |")
    rows.append("|---|---|---|---|---|---|")

    for key, label in keys:
        filt = key in ("entropy", "grad_norm")
        _, lv = _extract(lumen_entries, key, filter_outliers=filt)
        _, bv = _extract(baseline_entries, key, filter_outliers=filt)
        la = np.array(lv) if lv else np.array([])
        ba = np.array(bv) if bv else np.array([])
        r = pearson_r(lv, bv) if lv and bv else float("nan")
        lm = f"{la.mean():.4f}" if len(la) else "N/A"
        ls = f"{la.std():.4f}" if len(la) else "N/A"
        bm = f"{ba.mean():.4f}" if len(ba) else "N/A"
        bs = f"{ba.std():.4f}" if len(ba) else "N/A"
        rs = f"{r:.3f}" if not np.isnan(r) else "N/A"
        rows.append(f"| {label} | {lm} | {ls} | {bm} | {bs} | {rs} |")

    return "\n".join(rows)


def write_comparison_md(dir_a: Path, dir_b: Path,
                        entries_a, entries_b,
                        label_a: str = "Run A", label_b: str = "Run B"):
    stats = build_stats_table(entries_a, entries_b, label_a=label_a, label_b=label_b)

    _, ra = _extract(entries_a, "reward")
    _, rb = _extract(entries_b, "reward")
    _, la = _extract(entries_a, "completions/mean_length")
    _, lb = _extract(entries_b, "completions/mean_length")

    reward_r = pearson_r(ra, rb)
    length_r = pearson_r(la, lb)

    md = f"""# {label_a} vs. {label_b} Comparison

## Results

![Comparison Curves](compare_curves.png)

{stats}

## Interpretation

- **Reward Pearson r = {reward_r:.3f}**: {"Strong" if reward_r > 0.7 else "Moderate" if reward_r > 0.4 else "Weak"} correlation — {"the two runs show the same reward learning trajectory." if reward_r > 0.7 else "some divergence in reward dynamics, likely due to stochastic generation." if reward_r > 0.4 else "significant divergence — investigate configuration differences."}
- **Length Pearson r = {length_r:.3f}**: {"Strong" if length_r > 0.7 else "Moderate" if length_r > 0.4 else "Weak"} correlation — {"both runs show the same behavioral adaptation." if length_r > 0.7 else "some divergence in length dynamics." if length_r > 0.4 else "significant divergence."}
- {label_a} final reward: {ra[-1]:.4f}, {label_b} final reward: {rb[-1]:.4f}
- {label_a} final length: {la[-1]:.1f}, {label_b} final length: {lb[-1]:.1f}

## Conclusion

{"Both runs produce equivalent training behavior, confirming consistent GRPO training dynamics." if reward_r > 0.4 and length_r > 0.4 else "There are notable differences between the two runs that merit further investigation."}
"""

    out_path = dir_b / "COMPARISON.md"
    out_path.write_text(md)
    print(f"Saved comparison report to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare two GRPO training runs.")
    parser.add_argument("dir_a", type=str, help="First run output directory")
    parser.add_argument("dir_b", type=str, help="Second run output directory")
    parser.add_argument("--label-a", type=str, default=None,
                        help="Legend label for first run (default: directory name)")
    parser.add_argument("--label-b", type=str, default=None,
                        help="Legend label for second run (default: directory name)")
    args = parser.parse_args()

    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)
    label_a = args.label_a or dir_a.name
    label_b = args.label_b or dir_b.name

    log_a = dir_a / "grpo_eval_log.jsonl"
    log_b = dir_b / "grpo_eval_log.jsonl"

    if not log_a.exists():
        print(f"Log not found: {log_a}", file=sys.stderr)
        sys.exit(1)
    if not log_b.exists():
        print(f"Log not found: {log_b}", file=sys.stderr)
        sys.exit(1)

    entries_a = load_log(log_a)
    entries_b = load_log(log_b)

    plot_path = dir_b / "compare_curves.png"
    plot_comparison(entries_a, entries_b, plot_path, label_a=label_a, label_b=label_b)
    write_comparison_md(dir_a, dir_b, entries_a, entries_b,
                        label_a=label_a, label_b=label_b)


if __name__ == "__main__":
    main()
