#!/usr/bin/env python3
"""Full analysis of R1 (200 steps) training data with publication-quality charts.

Reference style: arXiv:2512.07611 (PPO/GRPO/DAPO comparative analysis)
 - Smoothed reward curve (Savitzky-Golay + EMA)
 - Separate accuracy panel (pass@1 rolling window)
 - Multi-panel: reward, accuracy, KL, entropy, loss, completion length
"""
import json
import os
import numpy as np

LOG_PATH = os.path.join(os.path.dirname(__file__), "grpo_perf_log.jsonl")
OUT_DIR = os.path.dirname(__file__)

with open(LOG_PATH) as f:
    records = [json.loads(line) for line in f]

records = [r for r in records if r.get("reward") is not None or r.get("rewards/reward_fn/mean") is not None]

steps = np.array([r["step"] for r in records])
rewards = np.array([r.get("reward", r.get("rewards/reward_fn/mean", 0)) for r in records])
kl_vals = np.array([r.get("kl", 0) for r in records])
entropy_vals = np.array([r.get("entropy", 0) for r in records])
loss_vals = np.array([r.get("loss", 0) for r in records])
frac_zero = np.array([r.get("frac_reward_zero_std", 0) for r in records])
grad_norm = np.array([r.get("grad_norm", 0) for r in records])
step_times = np.array([r.get("step_time_total", 0) for r in records])
mean_len = np.array([r.get("completions/mean_length", 0) for r in records])
clip_ratio = np.array([r.get("completions/clipped_ratio", 0) for r in records])

n = len(records)


def ema(values, alpha=0.3):
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def savgol_smooth(values, window=21, polyorder=3):
    """Savitzky-Golay filter, falls back to EMA if scipy unavailable."""
    try:
        from scipy.signal import savgol_filter
        w = min(window, len(values))
        if w % 2 == 0:
            w -= 1
        if w < polyorder + 2:
            return ema(values, 0.15)
        return savgol_filter(values, w, polyorder)
    except ImportError:
        return ema(values, 0.15)


def rolling_mean(values, window=20):
    """Centered rolling mean with edge handling."""
    out = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        out[i] = values[lo:hi].mean()
    return out


emas_reward = ema(rewards, 0.3)
smooth_reward = savgol_smooth(rewards, window=31, polyorder=2)

# "Accuracy" = rolling pass@1 (reward is binary 0/1 per completion, mean over group)
# rolling_accuracy over a 20-step window gives a proxy pass@1
rolling_acc = rolling_mean(rewards, window=20)

# Quartile analysis on EMA
q_size = max(1, n // 4)
q1 = emas_reward[:q_size].mean()
q2 = emas_reward[q_size:2*q_size].mean()
q3 = emas_reward[2*q_size:3*q_size].mean()
q4 = emas_reward[3*q_size:].mean()

best_ema_idx = int(emas_reward.argmax())
best_ema_step = steps[best_ema_idx]
best_ema_val = emas_reward[best_ema_idx]

# Smoothed KL/entropy
smooth_kl = savgol_smooth(kl_vals, window=31, polyorder=2)
smooth_entropy = savgol_smooth(entropy_vals, window=31, polyorder=2)
smooth_loss = savgol_smooth(loss_vals, window=31, polyorder=2)

# Print report
print(f"=" * 70)
print(f"  R1 BF16 Baseline — GRPO Training Report ({n} steps)")
print(f"=" * 70)
print(f"  Model:     LLaMA 3.1 8B-Instruct")
print(f"  Config:    LR=5e-7, constant_with_warmup, WD=0.0, Beta=0.01")
print(f"  Gen:       num_generations=8, max_completion=2048")
print(f"  Steps:     {n} (no early stop)")
print()
print(f"  --- Reward / Accuracy ---")
print(f"  First 10 avg:  {rewards[:10].mean():.4f}")
print(f"  Last 10 avg:   {rewards[-10:].mean():.4f}")
print(f"  Last 20 avg (pass@1): {rewards[-20:].mean():.4f}")
print(f"  Max:            {rewards.max():.4f} (step {steps[rewards.argmax()]})")
print(f"  Overall avg:    {rewards.mean():.4f}")
print()
print(f"  --- EMA(0.3) Quartile Trend ---")
print(f"  Q1 (1-{q_size}):      {q1:.4f}")
print(f"  Q2 ({q_size+1}-{2*q_size}):   {q2:.4f}  ({(q2-q1)/max(q1,1e-6)*100:+.1f}%)")
print(f"  Q3 ({2*q_size+1}-{3*q_size}):  {q3:.4f}  ({(q3-q1)/max(q1,1e-6)*100:+.1f}%)")
print(f"  Q4 ({3*q_size+1}-{n}): {q4:.4f}  ({(q4-q1)/max(q1,1e-6)*100:+.1f}%)")
print(f"  Best EMA:       {best_ema_val:.4f} (step {best_ema_step})")
print(f"  Final EMA:      {emas_reward[-1]:.4f}")
print()
print(f"  --- Training Health ---")
print(f"  KL:       {kl_vals[0]:.2f} → {kl_vals[-1]:.2f} ({'↓ stable' if kl_vals[-1] < kl_vals[0] else '↑'})")
print(f"  Entropy:  {entropy_vals[0]:.2f} → {entropy_vals[-1]:.2f} ({'↑ normal' if entropy_vals[-1] > entropy_vals[0] else '↓'})")
print(f"  Loss:     {loss_vals[0]:.4f} → {loss_vals[-1]:.4f}")
fz_first = frac_zero[:10].mean()
fz_last = frac_zero[-10:].mean()
print(f"  frac_zero_std: {fz_first:.3f} → {fz_last:.3f}")
high_fz = (frac_zero >= 0.5).sum()
print(f"  Steps with >=50% zero-std: {high_fz}/{n} ({high_fz/n*100:.0f}%)")
print()
print(f"  --- Performance ---")
valid_times = step_times[step_times > 0]
print(f"  Avg step time: {valid_times.mean():.1f}s")
print(f"  Peak memory:   {records[-1].get('peak_memory_allocated_gb', 0):.1f} GB")
print()

verdict = "RISING" if q4 > q1 + 0.02 else ("STABLE" if abs(q4 - q3) < 0.03 else "MIXED")
print(f"  === VERDICT: {verdict} ===")
if verdict == "RISING":
    print(f"  Reward trend is positive. Q1→Q4: {q1:.3f} → {q4:.3f} (+{(q4-q1)/max(q1,1e-6)*100:.0f}%)")
print(f"=" * 70)


# ─── Charts (publication style, ref arXiv:2512.07611) ───────────────

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })

    C = {
        "bg":       "#0c0e13",
        "card":     "#14161e",
        "raw":      "#4dd0e1",
        "ema":      "#ab47bc",
        "smooth":   "#ff7043",
        "acc":      "#66bb6a",
        "kl":       "#ffa726",
        "entropy":  "#26c6da",
        "loss":     "#42a5f5",
        "frac":     "#ef5350",
        "length":   "#78909c",
        "text":     "#eceff1",
        "muted":    "#546e7a",
        "grid":     "#1e272e",
    }

    def style(ax, title, ylabel=""):
        ax.set_facecolor(C["card"])
        ax.set_title(title, color=C["text"], fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, color=C["muted"], fontsize=9)
        ax.tick_params(colors=C["muted"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(C["grid"])
        ax.grid(True, alpha=0.12, color=C["grid"])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    fig, axes = plt.subplots(3, 2, figsize=(16, 13), facecolor=C["bg"])

    # ── Panel 1: Reward with EMA + Savitzky-Golay smooth ──
    ax = axes[0, 0]
    style(ax, "Reward (Math Correctness)", "Reward")
    ax.scatter(steps, rewards, c=C["raw"], alpha=0.2, s=8, zorder=1, label="Raw")
    ax.plot(steps, emas_reward, color=C["ema"], lw=1.8, zorder=2,
            label=f"EMA α=0.3 (final {emas_reward[-1]:.3f})")
    ax.plot(steps, smooth_reward, color=C["smooth"], lw=2.2, zorder=3,
            label=f"Smoothed (Savitzky-Golay)")
    ax.axhline(q1, color=C["muted"], ls="--", alpha=0.4, lw=0.8)
    ax.axhline(q4, color=C["ema"], ls="--", alpha=0.4, lw=0.8)
    ax.text(steps[-1]+2, q1, f"Q1={q1:.2f}", color=C["muted"], fontsize=7, va="center")
    ax.text(steps[-1]+2, q4, f"Q4={q4:.2f}", color=C["ema"], fontsize=7, va="center")
    ax.set_xlabel("Step", color=C["muted"])
    ax.legend(fontsize=8, facecolor=C["card"], edgecolor=C["grid"], labelcolor=C["text"], loc="upper left")

    # ── Panel 2: Accuracy (GSM8K periodic + rolling training pass@1) ──
    ax = axes[0, 1]
    gsm8k_curve_path = os.path.join(OUT_DIR, "gsm8k_accuracy_curve.jsonl")
    has_gsm8k = os.path.exists(gsm8k_curve_path)
    if has_gsm8k:
        with open(gsm8k_curve_path) as gf:
            gsm8k_records = [json.loads(line) for line in gf]
        gsm8k_steps = np.array([r["step"] for r in gsm8k_records])
        gsm8k_acc = np.array([r["gsm8k_accuracy"] for r in gsm8k_records])
        style(ax, "Model Accuracy vs Step", "Accuracy")
        ax.plot(gsm8k_steps, gsm8k_acc, "o-", color="#e040fb", lw=2, markersize=5,
                label=f"GSM8K pass@1 (N={gsm8k_records[0].get('total', '?')})")
        ax.plot(steps, rolling_acc, color=C["acc"], lw=1.5, alpha=0.6,
                label=f"Train reward (rolling-20)")
        ax.set_xlabel("Step", color=C["muted"])
        ax.set_ylim(0, min(1.0, max(gsm8k_acc.max(), rolling_acc.max()) * 1.3))
        ax.legend(fontsize=8, facecolor=C["card"], edgecolor=C["grid"], labelcolor=C["text"])
    else:
        style(ax, "Accuracy (Rolling pass@1, window=20)", "Accuracy")
        ax.fill_between(steps, rolling_acc, alpha=0.15, color=C["acc"])
        ax.plot(steps, rolling_acc, color=C["acc"], lw=2, label=f"pass@1 (final {rolling_acc[-1]:.3f})")
        ax.axhline(rewards[:10].mean(), color=C["muted"], ls=":", alpha=0.5, lw=0.8)
        ax.text(5, rewards[:10].mean() + 0.01, f"baseline={rewards[:10].mean():.2f}",
                color=C["muted"], fontsize=7)
        ax.set_xlabel("Step", color=C["muted"])
        ax.set_ylim(0, min(1.0, rolling_acc.max() * 1.3))
        ax.legend(fontsize=8, facecolor=C["card"], edgecolor=C["grid"], labelcolor=C["text"])

    # ── Panel 3: KL Divergence ──
    ax = axes[1, 0]
    style(ax, "KL Divergence", "KL")
    ax.plot(steps, kl_vals, color=C["kl"], alpha=0.35, lw=0.8)
    ax.plot(steps, smooth_kl, color=C["kl"], lw=2, label=f"Smoothed ({kl_vals[-1]:.2f})")
    ax.fill_between(steps, smooth_kl, alpha=0.08, color=C["kl"])
    ax.set_xlabel("Step", color=C["muted"])
    ax.legend(fontsize=8, facecolor=C["card"], edgecolor=C["grid"], labelcolor=C["text"])

    # ── Panel 4: Entropy ──
    ax = axes[1, 1]
    style(ax, "Entropy", "Entropy")
    ax.plot(steps, entropy_vals, color=C["entropy"], alpha=0.35, lw=0.8)
    ax.plot(steps, smooth_entropy, color=C["entropy"], lw=2, label=f"Smoothed ({entropy_vals[-1]:.2f})")
    ax.fill_between(steps, smooth_entropy, alpha=0.08, color=C["entropy"])
    ax.set_xlabel("Step", color=C["muted"])
    ax.legend(fontsize=8, facecolor=C["card"], edgecolor=C["grid"], labelcolor=C["text"])

    # ── Panel 5: Loss ──
    ax = axes[2, 0]
    style(ax, "Policy Loss", "Loss")
    ax.plot(steps, loss_vals, color=C["loss"], alpha=0.35, lw=0.8)
    ax.plot(steps, smooth_loss, color=C["loss"], lw=2, label=f"Smoothed ({loss_vals[-1]:.4f})")
    ax.fill_between(steps, smooth_loss, alpha=0.08, color=C["loss"])
    ax.set_xlabel("Step", color=C["muted"])
    ax.legend(fontsize=8, facecolor=C["card"], edgecolor=C["grid"], labelcolor=C["text"])

    # ── Panel 6: Completion Length ──
    ax = axes[2, 1]
    style(ax, "Mean Completion Length", "Tokens")
    smooth_len = savgol_smooth(mean_len, window=31, polyorder=2)
    ax.plot(steps, mean_len, color=C["length"], alpha=0.35, lw=0.8)
    ax.plot(steps, smooth_len, color=C["length"], lw=2, label=f"Smoothed ({mean_len[-1]:.0f})")
    ax.fill_between(steps, smooth_len, alpha=0.08, color=C["length"])
    ax2 = ax.twinx()
    ax2.plot(steps, clip_ratio * 100, color=C["frac"], alpha=0.4, lw=1, label="Clipped %")
    ax2.set_ylabel("Clipped %", color=C["frac"], fontsize=8)
    ax2.tick_params(colors=C["frac"], labelsize=7)
    ax2.spines["right"].set_color(C["frac"])
    ax.set_xlabel("Step", color=C["muted"])
    ax.legend(fontsize=8, facecolor=C["card"], edgecolor=C["grid"], labelcolor=C["text"], loc="upper left")
    ax2.legend(fontsize=7, facecolor=C["card"], edgecolor=C["grid"], labelcolor=C["frac"], loc="upper right")

    fig.suptitle(
        f"R1 BF16 Baseline — GRPO Training Analysis ({n} steps)  ·  Verdict: {verdict}",
        color=C["text"], fontsize=14, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(OUT_DIR, "r1_training_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"\nChart saved to: {out_path}")

except ImportError as e:
    print(f"\nmatplotlib not available ({e}), skipping chart generation")
