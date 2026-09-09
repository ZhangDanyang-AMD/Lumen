#!/usr/bin/env python3
"""Compare the baseline and FlyDSL runs of this example.

    PYTHONPATH=$LUMEN_PYTHONPATH python3 $EXAMPLE_DIR/compare_runs.py baseline flydsl

The first name is the control; the rest are diffed against it. Reads the offline
wandb files rather than the log: the tqdm postfix rounds every metric to two
decimals, which is useless for a loss of order 1e-2, and wandb records a
``_timestamp`` per step, giving per-step wall time without the progress bar's
1-second quantisation.

Pass every repeat, not one run per mode:

    compare_runs.py wan-baseline-r1 wan-baseline-r2 wan-baseline-r3 \
                    wan-flydsl-r1 wan-flydsl-r2 wan-flydsl-r3

Runs whose names differ only by a trailing -repN or -rN are grouped, and the
spread within a mode is printed next to the gap between modes. That comparison
is the whole point: with one run per mode the step time is not interpretable --
see the README section "What a single run can and cannot tell you".
"""

import glob
import os
import re
import sys

LOG_DIR = os.environ.get("LOG_DIR", "/work/logs")
OUT_DIR = os.environ.get("OUT_DIR", "/work/outputs")


def read_run(mode):
    """Pull per-step history out of an offline wandb run directory."""
    root = os.path.join(OUT_DIR, f"wandb-{mode}")
    paths = sorted(glob.glob(os.path.join(root, "wandb", "offline-run-*", "run-*.wandb")))
    if not paths:
        sys.exit(f"no wandb run file under {root} -- did '{mode}' finish?")
    path = paths[-1]

    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    ds = DataStore()
    ds.open_for_scan(path)
    rows = {}
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break
        if data is None:
            break
        pb = wandb_internal_pb2.Record()
        try:
            pb.ParseFromString(data)
        except Exception:
            continue
        if pb.WhichOneof("record_type") != "history":
            continue
        item = {}
        for h in pb.history.item:
            key = h.nested_key[0] if h.nested_key else h.key
            try:
                item[key] = float(h.value_json)
            except (TypeError, ValueError):
                pass
        if item.get("_step") is not None:
            rows[int(item["_step"])] = item
    return os.path.basename(path), rows


def meta(mode, field):
    path = os.path.join(LOG_DIR, f"{mode}.meta.txt")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        for line in fh:
            if line.startswith(field) and ":" in line:
                return line.split(":", 1)[1].strip()
    return None


def backend_line(mode):
    """What the patch reported about the backend it actually used."""
    path = os.path.join(LOG_DIR, f"{mode}.log")
    if not os.path.exists(path):
        return None
    hits = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            if "[lumen]" in line and ("backend for" in line or "patched" in line):
                hits.append(line.split("[lumen]", 1)[1].strip())
    return hits


def step_times(rows):
    """(step 1 seconds, mean of the rest) from wandb timestamps."""
    steps = sorted(rows)
    if len(steps) < 3:
        return None, None
    ts = [rows[s].get("_timestamp") for s in steps]
    if any(t is None for t in ts):
        return None, None
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    return rows[steps[0]].get("_runtime"), sum(gaps[1:]) / len(gaps[1:])


def _group_repeats(names):
    """{mode: [run names]}, stripping a trailing -rep1 / -r2 style suffix."""
    groups = {}
    for n in names:
        base = re.sub(r"-(rep|r)\d+$", "", n)
        groups.setdefault(base, []).append(n)
    return groups


def main():
    modes = sys.argv[1:] or ["baseline", "flydsl"]
    data = {}
    for m in modes:
        fname, rows = read_run(m)
        data[m] = rows
        print(f"{m:12} {fname}  {len(rows)} steps  exit={meta(m, 'EXIT_CODE')}")

    print("\n=== did the patch actually run? ===")
    for m in modes:
        lines = backend_line(m) or []
        if not lines:
            print(f"  {m:12} no [lumen] patch lines (this is expected for the baseline)")
        for ln in lines:
            print(f"  {m:12} {ln}")
    print("  'backend for conv2d: FLYDSL' is the one line that proves the kernel ran.")
    print("  Its absence in a patched run means Lumen demoted to torch.")

    control, *others = modes
    base = data[control]
    steps = sorted(set(base).intersection(*(set(data[m]) for m in others)))
    if not steps:
        sys.exit("no overlapping steps")

    print("\n=== total_loss per step ===")
    print(f"{'step':>5}" + "".join(f"{m:>20}" for m in modes))
    for s in steps:
        print(f"{s:>5}" + "".join(f"{data[m][s].get('training/total_loss', float('nan')):>20.9f}" for m in modes))

    print("\n=== loss deviation from the control ===")
    for m in others:
        devs = [
            abs(data[m][s]["training/total_loss"] - base[s]["training/total_loss"])
            / abs(base[s]["training/total_loss"])
            for s in steps
            if base[s].get("training/total_loss")
        ]
        if not devs:
            # offline_embedding computes no loss: it returns before the DiT runs.
            print(f"  {m:12} no loss to compare (control's total_loss is 0 at every step)")
            continue
        print(f"  {m:12} max {max(devs):.3%}   mean {sum(devs) / len(devs):.3%}")
    print("  Read these against the control's own repeats, which are the noise floor")
    print("  (enable_full_determinism is false). If a repeat of the control deviates")
    print("  as much as a patched run does, the patch did not change the arithmetic")
    print("  meaningfully. On Qwen-Image that floor measured 0.28% mean / 0.86% max;")
    print("  do not carry that number to another model, measure it again.")

    print("\n=== grad_norm, first and last step ===")
    for s in (steps[0], steps[-1]):
        print(f"  step {s:<3}" + "".join(f"{data[m][s].get('training/grad_norm', float('nan')):>16.6f}" for m in modes))

    print("\n=== timing ===")
    print(f"  {'mode':12}{'step 1':>10}{'steady':>10}{'wall':>9}")
    for m in modes:
        first, steady = step_times(data[m])
        print(f"  {m:12}{first:>9.1f}s{steady:>9.2f}s{meta(m, 'WALL_SECONDS') or '?':>8}s")
    reps = _group_repeats(modes)
    if any(len(v) > 1 for v in reps.values()):
        print("\n=== steady step time grouped by mode, over repeats ===")
        for mode, names in sorted(reps.items()):
            vals = [step_times(data[n])[1] for n in names if step_times(data[n])[1] is not None]
            if not vals:
                continue
            spread = (max(vals) - min(vals)) / (sum(vals) / len(vals)) * 100
            print(
                f"  {mode:14} n={len(vals)}  mean {sum(vals) / len(vals):.2f}s  "
                f"range {min(vals):.2f}-{max(vals):.2f}s  spread {spread:.1f}%"
            )
        print("  Compare the gap between modes against the spread within a mode.")
        print("  A gap smaller than the spread is not a result.")
    else:
        print()
        print("  !! Do not read a speedup out of the steady column with one run per mode.")
        print("     Repeat runs of an identical configuration vary by about +/-6% here.")
        print("     Pass several -repN runs of each mode to get the spread printed.")
    print("\n  How much of a step the convolutions are is model-dependent and has to be")
    print("  measured, not assumed: 0.07% of a Qwen-Image step at 256, and a different")
    print("  order of magnitude on a video clip. LUMEN_TIME_VAE=1 measures it directly;")
    print("  verify_vae_patch.py and verify_video_vae_patch.py time the encode itself.")
    print("  Step 1 carries kernel autotune plus, for a FlyDSL run, JIT compilation --")
    print("  ~11 s on Qwen-Image's 11 shapes, minutes on a video VAE's 66. One-time,")
    print("  and cached on disk, so only the first run of a shape set pays it.")

    print("\n=== peak memory (torch max_memory_allocated) ===")
    for m in modes:
        print(f"  {m:12}{max(data[m][s].get('max_memory_allocated(GB)', 0) for s in steps):>8.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
