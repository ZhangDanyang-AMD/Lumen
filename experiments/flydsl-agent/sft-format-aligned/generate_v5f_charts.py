#!/usr/bin/env python3
"""Generate v5f benchmark and loss curve charts."""

import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def make_benchmark_chart():
    with open(RESULTS / "benchmark_v5f.json") as f:
        data = json.load(f)

    def per_level(results):
        levels = {}
        for r in results:
            levels.setdefault(r["level"], []).append(r["api_score"])
        out = {}
        for lv in sorted(levels):
            out[lv] = np.mean(levels[lv]) * 100
        out["overall"] = np.mean([r["api_score"] for r in results]) * 100
        return out

    v5f = per_level(data["format_aligned_results"])
    v5e = per_level(data["v5e_baseline_results"])

    # Format compliance
    fmt_checks = data["format_checks"]
    compliant = sum(1 for f in fmt_checks if f["compliant"])
    fmt_rate = compliant / len(fmt_checks) * 100

    # Sandbox
    sandbox = data["sandbox_results"]
    sb_pass = sum(1 for r in sandbox if r["pass"]) / len(sandbox) * 100 if sandbox else 0

    labels = ["Level 1\n(Basic)", "Level 2\n(Elementary)", "Level 3\n(Intermediate)",
              "Level 4\n(Advanced)", "Level 5\n(Expert)", "Overall"]
    keys = [1, 2, 3, 4, 5, "overall"]

    v5f_vals = [v5f[k] for k in keys]
    v5e_vals = [v5e[k] for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6),
                                     gridspec_kw={"width_ratios": [3, 2]})

    x = np.arange(len(labels))
    w = 0.3

    bars_v5e = ax1.bar(x - w/2, v5e_vals, w, label="SFT v5e (baseline)",
                       color="#6BAED6", alpha=0.85)
    bars_v5f = ax1.bar(x + w/2, v5f_vals, w, label="SFT v5f (format-aligned)",
                       color="#E6550D", alpha=0.85)

    for bars in [bars_v5e, bars_v5f]:
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f"{h:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_ylabel("API Score (%)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(0, 115)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.set_title("SFT v5f Benchmark — Per-Level API Score\n(v5f vs v5e baseline)",
                  fontsize=13, fontweight="bold")

    # Right panel: format + sandbox metrics
    metrics = ["API Score\nOverall", "Format\nCompliance", "Sandbox\nCompilation",
               "<code> tag\nExtraction"]
    values = [v5f["overall"], fmt_rate, sb_pass,
              sum(1 for r in sandbox if r["prompt_type"] == "code_tag" and r["pass"])
              / max(1, sum(1 for r in sandbox if r["prompt_type"] == "code_tag")) * 100]
    targets = [74, 90, 80, 80]
    colors = ["#2ca02c" if v >= t else "#d62728" for v, t in zip(values, targets)]

    y_pos = np.arange(len(metrics))
    bars = ax2.barh(y_pos, values, 0.5, color=colors, alpha=0.8)
    for i, (v, t) in enumerate(zip(values, targets)):
        ax2.axvline(t, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax2.annotate(f"{v:.0f}%", xy=(v + 1, i), va="center", fontsize=10, fontweight="bold")
        ax2.annotate(f"target {t}%", xy=(t - 1, i + 0.3), va="center",
                    fontsize=7, color="gray", ha="right")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(metrics, fontsize=10)
    ax2.set_xlabel("Rate (%)", fontsize=11)
    ax2.set_xlim(0, 115)
    ax2.set_title("v5f Quality Metrics", fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(HERE / "sft_benchmark.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {HERE / 'sft_benchmark.png'}")


def make_loss_chart():
    log_path = RESULTS / "v5f_run.log"
    if not log_path.exists():
        # Try auto_pipeline log
        log_path = RESULTS / "auto_pipeline-v5f.log"
    log = log_path.read_text()

    train_steps, train_losses = [], []
    val_steps, val_losses = [], []
    total_steps = 0
    for line in log.splitlines():
        m = re.search(r"step (\d+)/(\d+) \| loss ([\d.]+)", line)
        if m:
            train_steps.append(int(m.group(1)))
            train_losses.append(float(m.group(3)))
            total_steps = int(m.group(2))
        m = re.search(r"step (\d+)/(\d+) \| val_loss ([\d.]+)", line)
        if m:
            val_steps.append(int(m.group(1)))
            val_losses.append(float(m.group(3)))
            total_steps = int(m.group(2))

    train_steps = np.array(train_steps)
    train_losses = np.array(train_losses)

    ma_window = 5
    ma_losses = np.convolve(train_losses, np.ones(ma_window) / ma_window, mode="valid")
    ma_steps = train_steps[ma_window - 1:]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.scatter(train_steps, train_losses, s=6, alpha=0.25, color="#4292C6",
               label="Train loss (per step)", zorder=1)
    ax.plot(ma_steps, ma_losses, color="#2171B5", linewidth=1.8,
            label=f"Train loss (MA-{ma_window})", zorder=2)
    ax.plot(val_steps, val_losses, "o-", color="#E6550D", linewidth=2.2,
            markersize=5, label="Val loss (every 50 steps)", zorder=3)

    # Annotate key val_loss points (every 10th to avoid clutter)
    for i, (s, v) in enumerate(zip(val_steps, val_losses)):
        if i % 10 == 0 or i == len(val_steps) - 1:
            ax.annotate(f"{v:.3f}", xy=(s, v), xytext=(0, 10),
                        textcoords="offset points", fontsize=6,
                        ha="center", color="#E6550D")

    # Epoch boundaries (3 epochs)
    epoch_size = total_steps / 3
    for i in range(1, 3):
        boundary = int(epoch_size * i)
        ax.axvline(boundary, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.text(boundary, 3.8, f"Epoch {i}", ha="center", fontsize=8, color="gray")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss (Cross Entropy)", fontsize=12)
    ax.set_title("SFT v5f Loss Curves — Qwen2.5-Coder-32B on FlyDSL (format-aligned)\n"
                 f"(LoRA r=64, 3 Epochs, 8xMI350X, seq=32768, {total_steps} steps)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 4.5)
    ax.set_xlim(0, max(train_steps) + 20)

    plt.tight_layout()
    fig.savefig(HERE / "sft_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {HERE / 'sft_loss_curves.png'}")


if __name__ == "__main__":
    make_benchmark_chart()
    make_loss_chart()
