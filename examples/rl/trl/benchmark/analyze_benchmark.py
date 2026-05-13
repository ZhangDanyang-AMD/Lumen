#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Post-hoc analysis for Lumen FP8 GRPO benchmark results.

Reads grpo_perf_log.jsonl from each run directory and produces:
  - Table 1: Headline throughput (spec §7, Table 1)
  - Table 2: Stage breakdown (spec §7, Table 2)
  - Table 3: Memory summary (spec §7, Table 3)
  - Figure 1: Reward curve overlay (spec §7, Figure 1)
  - Figure 2: Stage time stacked bar (spec §7, Figure 2)
  - Success criteria check (spec §8)

Usage:
  python analyze_benchmark.py outputs/benchmark
  python analyze_benchmark.py outputs/benchmark --num-gpus 8
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_RUN_ORDER = ["R1", "R2", "R3", "R4", "R5"]
_RUN_LABELS = {
    "R1": "BF16 Baseline",
    "R2": "Lumen BF16",
    "R3": "FP8 Linear",
    "R4": "FP8 Full",
    "R5": "FP8 + LoRA",
}
_RUN_COLORS = {
    "R1": "#6b7280",
    "R2": "#2563eb",
    "R3": "#7c3aed",
    "R4": "#059669",
    "R5": "#d97706",
}

_NEMO_REF = {
    "bf16_off_policy_tok_per_s": 2478,
    "fp8_off_policy_tok_per_s": 3052,
    "fp8_bf16_ratio": 3052 / 2478,
}


def load_log(log_path: Path) -> list[dict]:
    entries = []
    with open(log_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _measured_entries(entries: list[dict]) -> list[dict]:
    """Return only non-warmup entries."""
    return [e for e in entries if not e.get("is_warmup", True)]


def _stats(entries: list[dict], key: str) -> dict:
    vals = [e[key] for e in entries if key in e and e[key] is not None]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    arr = np.array(vals)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(vals)}


def _headline_memory(entries: list[dict], key: str) -> float:
    """Extract headline memory from the step marked with is_headline_memory."""
    for e in entries:
        if e.get("is_headline_memory") and key in e and e[key] is not None:
            return float(e[key])
    measured = _measured_entries(entries)
    s = _stats(measured, key)
    return s["mean"]


def pearson_r(x: list, y: list) -> float:
    """Pearson correlation on position-aligned pairs."""
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    n = min(len(x), len(y))
    xa, ya = np.array(x[:n], dtype=float), np.array(y[:n], dtype=float)
    mx, my = xa.mean(), ya.mean()
    dx, dy = xa - mx, ya - my
    denom = np.sqrt((dx**2).sum() * (dy**2).sum())
    if denom == 0:
        return float("nan")
    return float((dx * dy).sum() / denom)


def _extract_reward_by_step(entries: list[dict]) -> dict[int, float]:
    """Extract step -> reward mapping for step-aligned comparison."""
    result = {}
    for e in entries:
        reward = e.get("reward")
        if reward is None:
            reward = e.get("rewards/reward_fn/mean")
        if reward is not None and "step" in e:
            result[e["step"]] = float(reward)
    return result


def pearson_r_by_step(entries_a: list[dict], entries_b: list[dict]) -> float:
    """Pearson r aligned by step number, not position."""
    map_a = _extract_reward_by_step(entries_a)
    map_b = _extract_reward_by_step(entries_b)
    common_steps = sorted(set(map_a.keys()) & set(map_b.keys()))
    if len(common_steps) < 2:
        return float("nan")
    vals_a = [map_a[s] for s in common_steps]
    vals_b = [map_b[s] for s in common_steps]
    return pearson_r(vals_a, vals_b)


# =========================================================================
# Tables
# =========================================================================

def build_headline_table(run_data: dict, num_gpus: int) -> str:
    """Spec Table 1: Headline throughput & memory."""
    r2 = run_data.get("R2")
    r4 = run_data.get("R4")
    if not r2 or not r4:
        return "**Table 1**: Missing R2 or R4 data.\n"

    m2 = _measured_entries(r2)
    m4 = _measured_entries(r4)

    t2 = _stats(m2, "step_time_training")
    t4 = _stats(m4, "step_time_training")
    b2 = _stats(m2, "step_time_backward")
    b4 = _stats(m4, "step_time_backward")

    mem2_val = _headline_memory(r2, "peak_memory_allocated_gb")
    mem4_val = _headline_memory(r4, "peak_memory_allocated_gb")

    tok2 = sum(e.get("tokens_processed", 0) for e in m2)
    tok4 = sum(e.get("tokens_processed", 0) for e in m4)
    total2 = sum(e.get("step_time_training", 0) for e in m2)
    total4 = sum(e.get("step_time_training", 0) for e in m4)

    tps2 = tok2 / total2 / num_gpus if total2 > 0 else 0
    tps4 = tok4 / total4 / num_gpus if total4 > 0 else 0

    def _pct(a: float, b: float) -> str:
        if a == 0 or np.isnan(a) or np.isnan(b):
            return "N/A"
        return f"{((a - b) / a) * 100:+.1f}%"

    lines = [
        "## Table 1 — Headline (R4 vs R2)",
        "",
        "| Metric | Lumen BF16 (R2) | Lumen FP8 Full (R4) | FP8 Gain | NeMo RL Ref |",
        "|---|---|---|---|---|",
        f"| Training time/step | {t2['mean']:.2f}s ± {t2['std']:.2f}s "
        f"| {t4['mean']:.2f}s ± {t4['std']:.2f}s | **{_pct(t2['mean'], t4['mean'])}** | ~23% |",
        f"| Backward time/step | {b2['mean']:.2f}s ± {b2['std']:.2f}s "
        f"| {b4['mean']:.2f}s ± {b4['std']:.2f}s | **{_pct(b2['mean'], b4['mean'])}** | — |",
        f"| Peak memory/GPU | {mem2_val:.1f} GB | {mem4_val:.1f} GB "
        f"| **{mem2_val - mem4_val:.1f} GB** | — |",
        f"| tok/s/GPU (training) | {tps2:.0f} | {tps4:.0f} "
        f"| **{_pct(tps4, tps2) if tps2 > 0 else 'N/A'}** | — |",
        f"| tok/s/GPU (total, NeMo ref) | — | — | — | 2478→3052 |",
        "",
    ]
    return "\n".join(lines)


def build_stage_table(run_data: dict) -> str:
    """Spec Table 2: Stage breakdown."""
    stage_keys = [
        ("step_time_generation", "Generation"),
        ("step_time_forward", "Forward (training)"),
        ("step_time_backward", "Backward"),
        ("step_time_optim_comm", "Optimizer + Comm"),
        ("step_time_training", "**Training subtotal**"),
        ("step_time_total", "**Total step**"),
    ]

    header = "| Stage |"
    sep = "|---|"
    for rid in _RUN_ORDER:
        if rid in run_data:
            header += f" {rid} {_RUN_LABELS.get(rid, '')} |"
            sep += "---|"

    r2_data = run_data.get("R2")
    if r2_data and "R4" in run_data:
        header += " R4 vs R2 |"
        sep += "---|"

    lines = [
        "## Table 2 — Stage Breakdown",
        "",
        header,
        sep,
    ]

    for key, label in stage_keys:
        row = f"| {label} |"
        r2_val = None
        r4_val = None
        for rid in _RUN_ORDER:
            if rid not in run_data:
                continue
            measured = _measured_entries(run_data[rid])
            s = _stats(measured, key)
            row += f" {s['mean']:.2f}s |" if not np.isnan(s["mean"]) else " — |"
            if rid == "R2":
                r2_val = s["mean"]
            if rid == "R4":
                r4_val = s["mean"]

        if r2_data and "R4" in run_data:
            if r2_val and r4_val and not np.isnan(r2_val) and not np.isnan(r4_val) and r2_val > 0:
                pct = ((r2_val - r4_val) / r2_val) * 100
                row += f" **{pct:+.1f}%** |"
            else:
                row += " — |"

        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def build_memory_table(run_data: dict) -> str:
    """Spec Table 3: Memory."""
    header = "| Metric |"
    sep = "|---|"
    for rid in _RUN_ORDER:
        if rid in run_data:
            header += f" {rid} |"
            sep += "---|"

    lines = [
        "## Table 3 — Memory",
        "",
        header,
        sep,
    ]

    for key, label in [
        ("peak_memory_allocated_gb", "Peak allocated (GB)"),
        ("peak_memory_reserved_gb", "Peak reserved (GB)"),
    ]:
        row = f"| {label} |"
        for rid in _RUN_ORDER:
            if rid not in run_data:
                continue
            val = _headline_memory(run_data[rid], key)
            row += f" {val:.1f} |" if not np.isnan(val) else " — |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


# =========================================================================
# Figures
# =========================================================================

def plot_reward_overlay(run_data: dict, out_path: Path):
    """Spec Figure 1: Reward curve overlay."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for rid in _RUN_ORDER:
        if rid not in run_data:
            continue
        entries = run_data[rid]
        steps = [e["step"] for e in entries if "reward" in e or "rewards/reward_fn/mean" in e]
        rewards = [
            float(e.get("reward") or e.get("rewards/reward_fn/mean", 0))
            for e in entries
            if "reward" in e or "rewards/reward_fn/mean" in e
        ]
        if steps:
            ax.plot(
                steps, rewards,
                marker="o", markersize=3, linewidth=1.5,
                label=f"{rid}: {_RUN_LABELS.get(rid, rid)}",
                color=_RUN_COLORS.get(rid, "#000"),
                alpha=0.85,
            )

    ax.set_title("Reward Curve Overlay (All Runs)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)

    if "R2" in run_data and "R4" in run_data:
        r = pearson_r_by_step(run_data["R2"], run_data["R4"])
        print(f"  Reward Pearson r (R2 vs R4): {r:.4f}")
        return r
    return float("nan")


_METRIC_GROUPS = [
    ("Reward & Quality", [
        ("reward", "Reward (mean)"),
        ("reward_std", "Reward (std)"),
        ("loss", "Loss"),
        ("entropy", "Entropy"),
        ("win_rate", "Win Rate"),
    ]),
    ("Policy Optimization", [
        ("kl", "KL Divergence"),
        ("clip_ratio/region_mean", "Clip Ratio (region)"),
        ("clip_ratio/low_mean", "Clip Ratio (low)"),
        ("clip_ratio/high_mean", "Clip Ratio (high)"),
    ]),
    ("Training Dynamics", [
        ("grad_norm", "Grad Norm"),
        ("learning_rate", "Learning Rate"),
        ("frac_reward_zero_std", "Frac Reward Zero Std"),
    ]),
    ("Completions", [
        ("completions/mean_length", "Mean Length"),
        ("completions/min_length", "Min Length"),
        ("completions/max_length", "Max Length"),
        ("completions/clipped_ratio", "Clipped Ratio"),
    ]),
    ("Performance", [
        ("step_time_total", "Total Step Time (s)"),
        ("peak_memory_allocated_gb", "Peak Memory (GB)"),
    ]),
]


def _collect_available_metrics(run_data: dict) -> list[tuple[str, str]]:
    """Return (key, label) pairs for metrics present in at least one run."""
    available = []
    seen = set()
    for group_name, metrics in _METRIC_GROUPS:
        for key, label in metrics:
            if key in seen:
                continue
            for entries in run_data.values():
                if any(key in e and e[key] is not None for e in entries):
                    available.append((key, label))
                    seen.add(key)
                    break
    return available


def plot_training_curves(run_data: dict, out_path: Path):
    """Multi-panel training convergence curves (veRL/WandB style)."""
    metrics = _collect_available_metrics(run_data)
    if not metrics:
        print("  No metrics found for training curves.")
        return

    n = len(metrics)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, (key, label) in enumerate(metrics):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        for rid in _RUN_ORDER:
            if rid not in run_data:
                continue
            entries = run_data[rid]
            steps = [e["step"] for e in entries if key in e and e[key] is not None]
            vals = [float(e[key]) for e in entries if key in e and e[key] is not None]
            if steps:
                ax.plot(
                    steps, vals,
                    linewidth=1.4, alpha=0.85, markersize=2, marker=".",
                    label=f"{rid}",
                    color=_RUN_COLORS.get(rid, "#000"),
                )

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Step", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)

    for idx in range(len(metrics), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center",
            ncol=len(labels), fontsize=9,
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.suptitle(
        "GRPO Training Convergence Curves",
        fontsize=14, fontweight="bold", y=1.05,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_stage_bars(run_data: dict, out_path: Path):
    """Spec Figure 2: Stage time stacked bar."""
    stage_keys = [
        ("step_time_generation", "Generation"),
        ("step_time_forward", "Forward"),
        ("step_time_backward", "Backward"),
        ("step_time_optim_comm", "Optim+Comm"),
    ]
    stage_colors = ["#94a3b8", "#3b82f6", "#ef4444", "#f59e0b"]

    runs_present = [rid for rid in _RUN_ORDER if rid in run_data]
    if not runs_present:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(runs_present))
    width = 0.6
    bottoms = np.zeros(len(runs_present))

    for (key, label), color in zip(stage_keys, stage_colors):
        vals = []
        for rid in runs_present:
            measured = _measured_entries(run_data[rid])
            s = _stats(measured, key)
            vals.append(s["mean"] if not np.isnan(s["mean"]) else 0)
        vals_arr = np.array(vals)
        ax.bar(x, vals_arr, width, bottom=bottoms, label=label, color=color, alpha=0.85)
        bottoms += vals_arr

    ax.set_xticks(x)
    ax.set_xticklabels([f"{rid}\n{_RUN_LABELS.get(rid, '')}" for rid in runs_present], fontsize=10)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("GRPO Step Time Breakdown by Stage", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# =========================================================================
# Training Health Diagnostics (Guide §1.3-1.5, §4.2)
# =========================================================================

def _ema(values: list[float], alpha: float = 0.3) -> list[float]:
    """Exponential moving average."""
    if not values:
        return []
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


def diagnose_training_health(run_data: dict) -> str:
    """Detect anomalies per run: grad_norm spikes, entropy collapse,
    length explosion/collapse, and reward trend."""
    lines = ["## Training Health Diagnostics", ""]

    for rid in _RUN_ORDER:
        if rid not in run_data:
            continue
        entries = run_data[rid]
        if len(entries) < 3:
            continue

        alerts: list[str] = []

        grad_norms = [(e["step"], e["grad_norm"]) for e in entries
                      if "grad_norm" in e and e["grad_norm"] is not None]
        if len(grad_norms) >= 5:
            gn_vals = [g for _, g in grad_norms]
            gn_median = float(np.median(gn_vals))
            for step, gn in grad_norms:
                if gn_median > 0 and gn > 10 * gn_median:
                    alerts.append(f"  - grad_norm spike at step {step}: "
                                  f"{gn:.4f} (median={gn_median:.4f}, {gn/gn_median:.0f}x)")

        entropies = [(e["step"], e["entropy"]) for e in entries
                     if "entropy" in e and e["entropy"] is not None]
        if len(entropies) >= 3:
            for step, ent in entropies:
                if ent < 0:
                    alerts.append(f"  - entropy negative at step {step}: {ent:.2f} "
                                  f"(likely numerical overflow)")
            ent_vals = [v for _, v in entropies]
            if len(ent_vals) >= 5:
                early_mean = np.mean(ent_vals[:3])
                late_mean = np.mean(ent_vals[-3:])
                if early_mean > 0 and late_mean / early_mean < 0.1:
                    alerts.append(f"  - entropy collapse: early={early_mean:.2f} -> "
                                  f"late={late_mean:.2f} (possible mode collapse)")

        lengths = [(e["step"], e["completions/mean_length"]) for e in entries
                   if "completions/mean_length" in e and e["completions/mean_length"] is not None]
        if len(lengths) >= 5:
            len_vals = [v for _, v in lengths]
            len_median = float(np.median(len_vals))
            for step, ml in lengths:
                if len_median > 10 and ml < len_median * 0.05:
                    alerts.append(f"  - length collapse at step {step}: "
                                  f"{ml:.0f} tokens (median={len_median:.0f})")

        rewards = [(e["step"], float(e.get("reward") or e.get("rewards/reward_fn/mean", 0)))
                   for e in entries
                   if ("reward" in e or "rewards/reward_fn/mean" in e)]
        if len(rewards) >= 6:
            rw_vals = [r for _, r in rewards]
            first_half = np.mean(rw_vals[:len(rw_vals)//2])
            second_half = np.mean(rw_vals[len(rw_vals)//2:])
            ema_vals = _ema(rw_vals)
            trend = "upward" if second_half > first_half * 1.05 else (
                "plateau" if abs(second_half - first_half) <= first_half * 0.05 else "declining")
            lines.append(f"**{rid}** ({_RUN_LABELS.get(rid, rid)}):")
            lines.append(f"  - Reward trend: **{trend}** "
                         f"(first half mean={first_half:.3f}, second half={second_half:.3f}, "
                         f"final EMA={ema_vals[-1]:.3f})")
        else:
            lines.append(f"**{rid}** ({_RUN_LABELS.get(rid, rid)}):")

        if alerts:
            for a in alerts:
                lines.append(a)
        else:
            lines.append("  - No anomalies detected")

        lines.append("")

    return "\n".join(lines)


def plot_reward_with_ema(run_data: dict, out_path: Path):
    """Reward curve with EMA smoothing overlay for convergence clarity."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for rid in _RUN_ORDER:
        if rid not in run_data:
            continue
        entries = run_data[rid]
        steps = [e["step"] for e in entries
                 if "reward" in e or "rewards/reward_fn/mean" in e]
        rewards = [
            float(e.get("reward") or e.get("rewards/reward_fn/mean", 0))
            for e in entries
            if "reward" in e or "rewards/reward_fn/mean" in e
        ]
        if not steps:
            continue
        color = _RUN_COLORS.get(rid, "#000")
        ax.plot(steps, rewards, linewidth=0.8, alpha=0.35, color=color)
        ema_vals = _ema(rewards, alpha=0.3)
        ax.plot(steps, ema_vals, linewidth=2.2, alpha=0.9, color=color,
                label=f"{rid}: {_RUN_LABELS.get(rid, rid)} (EMA)")

    ax.set_title("Reward Convergence (with EMA smoothing)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# =========================================================================
# Success Criteria Check
# =========================================================================

def check_success_criteria(run_data: dict, reward_r: float) -> str:
    """Spec §8: Check all P0 and P1 criteria."""
    lines = ["## Success Criteria Check", ""]

    r2 = run_data.get("R2")
    r4 = run_data.get("R4")
    r1 = run_data.get("R1")

    results = []

    if r2 and r4:
        m2 = _measured_entries(r2)
        m4 = _measured_entries(r4)

        t2 = _stats(m2, "step_time_training")["mean"]
        t4 = _stats(m4, "step_time_training")["mean"]
        if t2 > 0 and not np.isnan(t2) and not np.isnan(t4):
            speedup = (t2 - t4) / t2 * 100
            ok = speedup >= 15
            results.append(
                f"| FP8 training speedup (R4 vs R2) | >= 15% | {speedup:.1f}% | "
                f"{'PASS' if ok else 'FAIL'} | P0 |"
            )

        b2 = _stats(m2, "step_time_backward")["mean"]
        b4 = _stats(m4, "step_time_backward")["mean"]
        if b2 > 0 and not np.isnan(b2) and not np.isnan(b4):
            bwd_speedup = (b2 - b4) / b2 * 100
            ok = bwd_speedup >= 20
            results.append(
                f"| FP8 backward speedup (R4 vs R2) | >= 20% | {bwd_speedup:.1f}% | "
                f"{'PASS' if ok else 'FAIL'} | P0 |"
            )

        mem2 = _headline_memory(r2, "peak_memory_allocated_gb")
        mem4 = _headline_memory(r4, "peak_memory_allocated_gb")
        if mem2 > 0 and not np.isnan(mem2) and not np.isnan(mem4):
            mem_save = (mem2 - mem4) / mem2 * 100
            ok = mem_save >= 10
            results.append(
                f"| FP8 memory saving (R4 vs R2) | >= 10% | {mem_save:.1f}% | "
                f"{'PASS' if ok else 'FAIL'} | P0 |"
            )

        if t2 > 0 and not np.isnan(t2) and not np.isnan(t4):
            our_ratio = (t2 - t4) / t2 * 100
            nemo_ratio = 23.0
            within = abs(our_ratio - nemo_ratio) <= nemo_ratio * 0.10
            results.append(
                f"| FP8/BF16 ratio vs NeMo ~23% | ±10% of 23% | {our_ratio:.1f}% | "
                f"{'PASS' if within else 'NOTE'} | P1 |"
            )

    if r1 and r2:
        m1 = _measured_entries(r1)
        m2 = _measured_entries(r2)
        t1 = _stats(m1, "step_time_total")["mean"]
        t2 = _stats(m2, "step_time_total")["mean"]
        if t1 > 0 and not np.isnan(t1) and not np.isnan(t2):
            overhead = (t2 - t1) / t1 * 100
            ok = overhead <= 5
            results.append(
                f"| Lumen overhead (R2 vs R1) | <= 5% | {overhead:+.1f}% | "
                f"{'PASS' if ok else 'NOTE'} | P1 |"
            )

    if not np.isnan(reward_r):
        ok = reward_r >= 0.9
        results.append(
            f"| Reward Pearson r (R2 vs R4) | >= 0.9 | {reward_r:.4f} | "
            f"{'PASS' if ok else 'FAIL'} | P0 |"
        )

    for rid in ["R3", "R4"]:
        if rid in run_data:
            has_nan = any(
                e.get("grad_norm") is not None and (np.isnan(e["grad_norm"]) or np.isinf(e["grad_norm"]))
                for e in run_data[rid]
            )
            results.append(
                f"| No NaN/Inf in {rid} | 0 occurrences | {'FAIL' if has_nan else '0'} | "
                f"{'FAIL' if has_nan else 'PASS'} | P0 |"
            )

    for rid in _RUN_ORDER:
        if rid in run_data:
            n = len(run_data[rid])
            ok = n >= 50
            results.append(
                f"| {rid} completion | 50 steps | {n} steps | "
                f"{'PASS' if ok else 'FAIL'} | P0 |"
            )

    lines.append("| Criterion | Threshold | Actual | Status | Priority |")
    lines.append("|---|---|---|---|---|")
    lines.extend(results)
    lines.append("")

    return "\n".join(lines)


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze Lumen FP8 GRPO benchmark results.")
    parser.add_argument("benchmark_dir", type=str, help="Directory containing R1/, R2/, ... subdirs")
    parser.add_argument("--num-gpus", type=int, default=8)
    args = parser.parse_args()

    base = Path(args.benchmark_dir)
    if not base.exists():
        print(f"Directory not found: {base}", file=sys.stderr)
        sys.exit(1)

    run_data: dict[str, list[dict]] = {}
    for rid in _RUN_ORDER:
        log_path = base / rid / "grpo_perf_log.jsonl"
        if log_path.exists():
            run_data[rid] = load_log(log_path)
            print(f"  Loaded {rid}: {len(run_data[rid])} entries")

    if not run_data:
        print("No run data found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nRuns found: {', '.join(run_data.keys())}")
    print(f"{'='*60}\n")

    report_parts = [
        "# Lumen FP8 GRPO Benchmark Results",
        "",
        f"**Generated**: auto-generated from benchmark data",
        f"**Runs**: {', '.join(run_data.keys())}",
        f"**GPUs**: {args.num_gpus}",
        "",
        "---",
        "",
    ]

    report_parts.append(build_headline_table(run_data, args.num_gpus))
    report_parts.append(build_stage_table(run_data))
    report_parts.append(build_memory_table(run_data))

    report_parts.append(diagnose_training_health(run_data))

    print("Generating figures...")
    reward_r = plot_reward_overlay(run_data, base / "reward_overlay.png")
    plot_reward_with_ema(run_data, base / "reward_ema.png")
    plot_stage_bars(run_data, base / "stage_breakdown.png")
    plot_training_curves(run_data, base / "training_curves.png")

    report_parts.extend([
        "## Figures",
        "",
        "![Reward Curve Overlay](reward_overlay.png)",
        "",
        "![Reward Convergence with EMA](reward_ema.png)",
        "",
        "![Stage Time Breakdown](stage_breakdown.png)",
        "",
        "![Training Convergence Curves](training_curves.png)",
        "",
    ])

    report_parts.append(check_success_criteria(run_data, reward_r))

    report_parts.extend([
        "---",
        "",
        "## NeMo RL Reference",
        "",
        f"NeMo RL (H100 16 GPUs, 1-step off-policy): BF16 {_NEMO_REF['bf16_off_policy_tok_per_s']} "
        f"→ FP8 {_NEMO_REF['fp8_off_policy_tok_per_s']} tok/s/GPU "
        f"(**{(_NEMO_REF['fp8_bf16_ratio'] - 1) * 100:.0f}% speedup**)",
        "",
        "Our training-subtotal FP8/BF16 ratio is expected to differ from NeMo's end-to-end ratio",
        "because NeMo's number includes vLLM generation time (which is not affected by FP8 training).",
        "See spec §8 footnote 2 for details.",
        "",
    ])

    report_path = base / "lumen-fp8-grpo-benchmark-results.md"
    report_path.write_text("\n".join(report_parts))
    print(f"\n  Saved report: {report_path}")
    print(f"\n{'='*60}")
    print("  Analysis complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
