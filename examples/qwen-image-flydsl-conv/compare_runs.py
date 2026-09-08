#!/usr/bin/env python3
"""Compare the baseline and FlyDSL runs of this example.

    PYTHONPATH=$LUMEN_PYTHONPATH python3 $EXAMPLE_DIR/compare_runs.py baseline flydsl

The first name is the control; the rest are diffed against it. Reads the offline
wandb files rather than the log: the tqdm postfix rounds every metric to two
decimals, which is useless for a loss of order 1e-2, and wandb records a
``_timestamp`` per step, giving per-step wall time without the progress bar's
1-second quantisation.

Read the loss columns, not the step time. With one run per mode the step time is
not interpretable -- see the warning this prints and the README section
"What a single run can and cannot tell you".
"""

import glob
import os
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
        print(f"  {m:12} max {max(devs):.3%}   mean {sum(devs) / len(devs):.3%}")
    print("  For scale: two runs of the *same* mode differ by ~0.28% mean / ~0.86% max")
    print("  on this setup (enable_full_determinism is false). Deviations of that")
    print("  order mean the patch did not change the arithmetic meaningfully.")

    print("\n=== grad_norm, first and last step ===")
    for s in (steps[0], steps[-1]):
        print(f"  step {s:<3}" + "".join(f"{data[m][s].get('training/grad_norm', float('nan')):>16.6f}" for m in modes))

    print("\n=== timing ===")
    print(f"  {'mode':12}{'step 1':>10}{'steady':>10}{'wall':>9}")
    for m in modes:
        first, steady = step_times(data[m])
        print(f"  {m:12}{first:>9.1f}s{steady:>9.2f}s{meta(m, 'WALL_SECONDS') or '?':>8}s")
    print()
    print("  !! Do not read a speedup out of the steady column with one run per mode.")
    print("     Repeat runs of an identical configuration vary by about +/-6% here,")
    print("     while the VAE convolutions are 0.56% of a step at 1024 (0.07% at 256),")
    print("     so the noise is an order of magnitude larger than anything this patch")
    print("     can move. The measurement that does resolve the kernel is")
    print("     verify_vae_patch.py, which times the encode itself with internal repeats.")
    print("  Step 1 carries kernel autotune plus, for the FlyDSL run, JIT compilation.")
    print("  That cost is one-time, and small enough that it is not always visible.")

    print("\n=== peak memory (torch max_memory_allocated) ===")
    for m in modes:
        print(f"  {m:12}{max(data[m][s].get('max_memory_allocated(GB)', 0) for s in steps):>8.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
