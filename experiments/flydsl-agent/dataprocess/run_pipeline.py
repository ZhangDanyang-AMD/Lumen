#!/usr/bin/env python3
"""One-click data processing pipeline.

Orchestrates the full data processing flow:
  1. Generate manifest (Layer 1 auto-annotation)
  2. Extract CPT data
  3. Generate SFT data
  4. Prepare RL specs
  5. Validate all datasets
  6. Export to training formats

Usage:
  python run_pipeline.py --repo /workspace/aiter
  python run_pipeline.py --repo /workspace/aiter --steps manifest,cpt,sft
"""

import argparse
import os
import subprocess
import sys
import time


STEPS = [
    "manifest",
    "cpt",
    "sft",
    "rl",
    "validate",
    "export",
]


def run_step(name: str, cmd: list, cwd: str):
    print(f"\n{'=' * 60}")
    print(f"STEP: {name}")
    print(f"CMD:  {' '.join(cmd)}")
    print(f"{'=' * 60}")
    start = time.time()

    result = subprocess.run(cmd, cwd=cwd)

    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"\n[{status}] {name} completed in {elapsed:.1f}s (exit code: {result.returncode})")

    if result.returncode != 0:
        print(f"WARNING: Step '{name}' failed. Continuing...")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run full data processing pipeline")
    parser.add_argument("--repo", default="/workspace/aiter",
                        help="Path to aiter repo root")
    parser.add_argument("--output-dir", default="/workspace/dataprocess/output",
                        help="Output directory for processed data")
    parser.add_argument("--script-dir", default="/workspace/dataprocess",
                        help="Directory containing processing scripts")
    parser.add_argument("--steps", default=",".join(STEPS),
                        help=f"Comma-separated steps to run: {','.join(STEPS)}")
    parser.add_argument("--skip-git", action="store_true",
                        help="Skip git history extraction in SFT generation")
    args = parser.parse_args()

    steps_to_run = [s.strip() for s in args.steps.split(",")]
    for s in steps_to_run:
        if s not in STEPS:
            print(f"ERROR: Unknown step '{s}'. Valid: {STEPS}")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    sd = args.script_dir
    od = args.output_dir
    repo = args.repo

    manifest_path = os.path.join(od, "manifest.json")
    taxonomy_path = os.path.join(sd, "taxonomy.yaml")

    results = {}
    total_start = time.time()

    # Step 1: Generate manifest
    if "manifest" in steps_to_run:
        rc = run_step("Generate Manifest", [
            sys.executable, os.path.join(sd, "generate_manifest.py"),
            "--repo", repo,
            "--taxonomy", taxonomy_path,
            "--output", manifest_path,
        ], cwd=sd)
        results["manifest"] = rc

    # Step 2: Extract CPT data
    if "cpt" in steps_to_run:
        rc = run_step("Extract CPT Data", [
            sys.executable, os.path.join(sd, "extract_cpt_data.py"),
            "--manifest", manifest_path,
            "--repo", repo,
            "--output", os.path.join(od, "cpt_data.jsonl"),
        ], cwd=sd)
        results["cpt"] = rc

    # Step 3: Generate SFT data
    if "sft" in steps_to_run:
        sft_cmd = [
            sys.executable, os.path.join(sd, "generate_sft_data.py"),
            "--manifest", manifest_path,
            "--repo", repo,
            "--output", os.path.join(od, "sft_data.jsonl"),
        ]
        if args.skip_git:
            sft_cmd.append("--skip-git")
        rc = run_step("Generate SFT Data", sft_cmd, cwd=sd)
        results["sft"] = rc

    # Step 4: Prepare RL specs
    if "rl" in steps_to_run:
        rc = run_step("Prepare RL Specs", [
            sys.executable, os.path.join(sd, "prepare_rl_specs.py"),
            "--manifest", manifest_path,
            "--repo", repo,
            "--output", os.path.join(od, "rl_specs.json"),
        ], cwd=sd)
        results["rl"] = rc

    # Step 5: Validate datasets
    if "validate" in steps_to_run:
        rc = run_step("Validate Datasets", [
            sys.executable, os.path.join(sd, "validate_dataset.py"),
            "--output-dir", od,
        ], cwd=sd)
        results["validate"] = rc

    # Step 6: Export training formats
    if "export" in steps_to_run:
        rc = run_step("Export Training Data", [
            sys.executable, os.path.join(sd, "export_for_training.py"),
            "--output-dir", od,
        ], cwd=sd)
        results["export"] = rc

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 60}")
    print("PIPELINE SUMMARY")
    print(f"{'=' * 60}")
    for step, rc in results.items():
        status = "PASS" if rc == 0 else "FAIL"
        print(f"  {step:20s}: {status}")
    print(f"\nTotal time: {total_elapsed:.1f}s")

    # List output files
    print(f"\nOutput files:")
    if os.path.isdir(od):
        for fname in sorted(os.listdir(od)):
            fpath = os.path.join(od, fname)
            if os.path.isfile(fpath):
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                print(f"  {fname:40s}: {size_mb:.2f} MB")

    failed = sum(1 for rc in results.values() if rc != 0)
    if failed:
        print(f"\nWARNING: {failed} step(s) had issues")
        sys.exit(1)
    else:
        print("\nAll steps completed successfully!")


if __name__ == "__main__":
    main()
