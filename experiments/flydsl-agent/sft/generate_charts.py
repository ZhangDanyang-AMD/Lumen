#!/usr/bin/env python3
"""Generate SFT benchmark and loss curve charts with v5e results."""

import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"


# ── 1. Benchmark chart ──────────────────────────────────────────────────────

def make_benchmark_chart():
    with open(RESULTS / "benchmark_v5e.json") as f:
        data = json.load(f)

    def per_level_scores(group):
        levels = {}
        for item in data[group]:
            lv = item["level"]
            levels.setdefault(lv, []).append(item["api_score"])
        result = {}
        for lv in sorted(levels):
            result[lv] = np.mean(levels[lv]) * 100
        result["overall"] = np.mean([it["api_score"] for it in data[group]]) * 100
        return result

    base = per_level_scores("base")
    v5e = per_level_scores("sft")
    target = {1: 90, 2: 85, 3: 70, 4: 50, 5: 20, "overall": 60}

    labels = ["Level 1\n(Basic)", "Level 2\n(Elementary)", "Level 3\n(Intermediate)",
              "Level 4\n(Advanced)", "Level 5\n(Expert)", "Overall"]
    keys = [1, 2, 3, 4, 5, "overall"]

    base_vals = [base[k] for k in keys]
    v5e_vals = [v5e[k] for k in keys]
    target_vals = [target[k] for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6),
                                     gridspec_kw={"width_ratios": [3, 2]})

    x = np.arange(len(labels))
    w = 0.22

    bars_base = ax1.bar(x - w, base_vals, w, label="Base (Qwen2.5-Coder-32B)",
                        color="#6BAED6", alpha=0.85)
    bars_v5e = ax1.bar(x, v5e_vals, w, label="SFT v5e (LoRA r=64, 3ep, 3889 samples)",
                       color="#E6550D", alpha=0.85)
    bars_target = ax1.bar(x + w, target_vals, w, label="Target (plan.md 8.2)",
                          color="#B2DF8A", alpha=0.6)

    for bars in [bars_base, bars_v5e, bars_target]:
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f"{h:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax1.set_ylabel("API Score (%)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(0, 115)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_title("SFT Benchmark — Per-Level API Score\n(Base vs SFT v5e vs Target)",
                  fontsize=13, fontweight="bold")

    # ── Right panel: pattern usage comparison (v5e vs base) ──
    pattern_labels = {
        "flyc_kernel": "@flyc.kernel", "fx_api": "fx.* API",
        "import_flyc": "import flydsl", "smem_alloc": "SmemAlloc",
        "mfma": "MFMA", "pipeline": "Pipeline",
        "swizzle": "Swizzle", "fx_layout": "fx.layout",
        "buffer_ops": "BufferOps", "syncthreads": "Syncthreads",
    }

    def pattern_rates(group):
        rates = {}
        items = data[group]
        n = len(items)
        for key in pattern_labels:
            count = sum(1 for it in items if it["analysis"].get(key, False))
            rates[key] = count / n * 100
        return rates

    base_p = pattern_rates("base")
    v5e_p = pattern_rates("sft")

    pkeys = list(pattern_labels.keys())
    plabels = [pattern_labels[k] for k in pkeys]
    base_pvals = [base_p[k] for k in pkeys]
    v5e_pvals = [v5e_p[k] for k in pkeys]

    y_pos = np.arange(len(plabels))
    ax2.barh(y_pos + 0.15, v5e_pvals, 0.3, label="SFT v5e", color="#E6550D", alpha=0.85)
    ax2.barh(y_pos - 0.15, base_pvals, 0.3, label="Base", color="#6BAED6", alpha=0.85)

    for i, (b, v) in enumerate(zip(base_pvals, v5e_pvals)):
        diff = v - b
        if abs(diff) > 0.5:
            color = "#2ca02c" if diff > 0 else "#d62728"
            ax2.annotate(f"{diff:+.0f}%", xy=(max(b, v) + 2, i),
                        fontsize=8, color=color, fontweight="bold", va="center")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(plabels, fontsize=9)
    ax2.set_xlabel("Usage Rate (%)", fontsize=11)
    ax2.set_xlim(0, 120)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_title("FlyDSL Pattern Usage", fontsize=13, fontweight="bold")

    plt.tight_layout()
    out = HERE / "sft_benchmark.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── 2. Loss curves chart ────────────────────────────────────────────────────

def make_loss_chart():
    log = (RESULTS / "auto_pipeline-v5e.log").read_text()

    train_steps, train_losses = [], []
    val_steps, val_losses = [], []
    for line in log.splitlines():
        m = re.search(r"step (\d+)/1459 \| loss ([\d.]+)", line)
        if m:
            train_steps.append(int(m.group(1)))
            train_losses.append(float(m.group(2)))
        m = re.search(r"step (\d+)/1459 \| val_loss ([\d.]+)", line)
        if m:
            val_steps.append(int(m.group(1)))
            val_losses.append(float(m.group(2)))

    train_steps = np.array(train_steps)
    train_losses = np.array(train_losses)

    # Moving average (window=5)
    ma_window = 5
    ma_losses = np.convolve(train_losses, np.ones(ma_window) / ma_window, mode="valid")
    ma_steps = train_steps[ma_window - 1:]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.scatter(train_steps, train_losses, s=8, alpha=0.3, color="#4292C6",
               label="Train loss (per step)", zorder=1)
    ax.plot(ma_steps, ma_losses, color="#2171B5", linewidth=1.8,
            label=f"Train loss (MA-{ma_window})", zorder=2)
    ax.plot(val_steps, val_losses, "o-", color="#E6550D", linewidth=2.2,
            markersize=6, label="Val loss (every 50 steps)", zorder=3)

    for s, v in zip(val_steps, val_losses):
        ax.annotate(f"{v:.3f}", xy=(s, v), xytext=(0, 10),
                    textcoords="offset points", fontsize=6.5,
                    ha="center", color="#E6550D")

    # Epoch boundaries
    epoch_size = 1459 / 3
    for i in range(1, 3):
        boundary = int(epoch_size * i)
        ax.axvline(boundary, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.text(boundary, ax.get_ylim()[1] * 0.95, f"Epoch {i}",
                ha="center", fontsize=8, color="gray")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss (Cross Entropy)", fontsize=12)
    ax.set_title("SFT v5e Loss Curves — Qwen2.5-Coder-32B on FlyDSL Instructions\n"
                 "(LoRA r=64, 3 Epochs, 8xMI350X, seq=16384, 3889 samples)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 4.5)
    ax.set_xlim(0, max(train_steps) + 20)

    plt.tight_layout()
    out = HERE / "sft_loss_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_benchmark_chart()
    make_loss_chart()
