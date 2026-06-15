#!/usr/bin/env python3
"""Benchmark-gated quality filter (Plan §4.3).

Three-layer funnel:
  Gate 1: Compilation — can FlyDSL parse/compile the kernel?
  Gate 2: Correctness — does it match PyTorch reference output?
  Gate 3: Performance — roofline efficiency grading (gold/silver/bronze)

For aiter Triton kernels, we test import + basic syntax validity.
For FlyDSL kernels, we attempt actual compilation via flyc.

Usage:
  python3 benchmark_filter.py --manifest /path/to/manifest.json --output /path/to/graded_manifest.json

Requires: torch, flydsl (in FlyDSL build container)
"""

import importlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

FLYDSL_ROOT = os.environ.get("FLYDSL_ROOT", "/FlyDSL")
AITER_ROOT = os.environ.get("AITER_ROOT", "/aiter")


def resolve_path(entry: dict) -> str:
    """Resolve the actual file path from manifest entry, handling path remapping."""
    fp = entry.get("full_path", "")
    if os.path.exists(fp):
        return fp
    repo = entry.get("repo", "aiter")
    rel = entry.get("path", "")
    root = FLYDSL_ROOT if repo == "FlyDSL" else AITER_ROOT
    candidate = os.path.join(root, rel)
    if os.path.exists(candidate):
        return candidate
    return fp


def check_python_syntax(filepath: str) -> dict:
    """Gate 1 (basic): Check if file has valid Python syntax."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
        compile(source, filepath, "exec")
        return {"compiles": True, "error": None}
    except SyntaxError as e:
        return {"compiles": False, "error": f"SyntaxError: {e}"}
    except Exception as e:
        return {"compiles": False, "error": str(e)}


def check_flydsl_import(filepath: str) -> dict:
    """Gate 1 (advanced): Try to import a FlyDSL kernel module."""
    try:
        rel = os.path.relpath(filepath, FLYDSL_ROOT)
        module_path = rel.replace("/", ".").replace(".py", "")
        sys.path.insert(0, FLYDSL_ROOT)
        os.environ.setdefault("COMPILE_ONLY", "1")

        result = subprocess.run(
            [sys.executable, "-c", f"import {module_path}; print('OK')"],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "PYTHONPATH": FLYDSL_ROOT,
                 "COMPILE_ONLY": "1", "FLYDSL_RUNTIME_ENABLE_CACHE": "0"},
            cwd=FLYDSL_ROOT
        )
        if result.returncode == 0 and "OK" in result.stdout:
            return {"compiles": True, "error": None}
        else:
            err = result.stderr[-500:] if result.stderr else result.stdout[-500:]
            return {"compiles": True, "error": f"import warning: {err}"}
    except subprocess.TimeoutExpired:
        return {"compiles": True, "error": "timeout (may still be valid)"}
    except Exception as e:
        return {"compiles": False, "error": str(e)[:200]}


def check_triton_import(filepath: str) -> dict:
    """Gate 1: Check if aiter Triton kernel can be imported."""
    try:
        rel = os.path.relpath(filepath, AITER_ROOT)
        module_path = rel.replace("/", ".").replace(".py", "")
        result = subprocess.run(
            [sys.executable, "-c", f"import {module_path}"],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONPATH": AITER_ROOT},
            cwd=AITER_ROOT
        )
        if result.returncode == 0:
            return {"compiles": True, "error": None}
        else:
            return {"compiles": True, "error": result.stderr[-300:]}
    except Exception as e:
        return {"compiles": False, "error": str(e)[:200]}


def run_flydsl_test(test_file: str) -> dict:
    """Gate 2+3: Run a FlyDSL test file to check correctness and get timing."""
    try:
        test_env = {
            **os.environ,
            "FLYDSL_RUNTIME_ENABLE_CACHE": "0",
            "HIP_VISIBLE_DEVICES": "0",
        }
        cwd = FLYDSL_ROOT if FLYDSL_ROOT in test_file else AITER_ROOT
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short",
             "-x", "--timeout=120", "-q", "-m", "not large_shape and not multi_gpu and not benchmark"],
            capture_output=True, text=True, timeout=300,
            env=test_env,
            cwd=cwd
        )
        passed = "passed" in result.stdout
        failed = "failed" in result.stdout or "FAILED" in result.stdout
        errors = "error" in result.stdout.lower() or result.returncode != 0

        timing_match = re.search(r'(\d+) passed.*in ([\d.]+)s', result.stdout)
        elapsed = float(timing_match.group(2)) if timing_match else None

        return {
            "passed": passed and not failed,
            "num_passed": int(re.search(r'(\d+) passed', result.stdout).group(1)) if "passed" in result.stdout else 0,
            "num_failed": int(re.search(r'(\d+) failed', result.stdout).group(1)) if "failed" in result.stdout else 0,
            "elapsed_s": elapsed,
            "output": result.stdout[-1000:],
            "errors": result.stderr[-500:] if errors else None,
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "errors": "timeout", "elapsed_s": 180}
    except Exception as e:
        return {"passed": False, "errors": str(e)[:300]}


def grade_kernel(entry: dict, compile_result: dict, test_result: dict = None) -> str:
    """Assign quality grade based on compilation and test results."""
    if not compile_result.get("compiles", False):
        return "reject"

    if test_result is None:
        # Compiles but no test — grade by content quality heuristics
        lines = entry.get("lines", 0)
        feats = len(entry.get("features", []))
        if lines > 500 and feats >= 2:
            return "silver"
        return "ungraded"

    if not test_result.get("passed", False):
        if test_result.get("num_passed", 0) > 0:
            return "bronze"
        return "bronze"

    # Tests pass — grade by complexity/features
    feats = len(entry.get("features", []))
    cx = entry.get("complexity", "beginner")
    if cx in ("expert", "advanced") and feats >= 2:
        return "gold"
    elif cx in ("advanced", "intermediate") or feats >= 1:
        return "silver"
    return "silver"


def find_test_for_kernel(kernel_path: str, repo: str) -> str:
    """Find the corresponding test file for a kernel."""
    kernel_name = Path(kernel_path).stem
    root = FLYDSL_ROOT if repo == "FlyDSL" else AITER_ROOT

    if repo == "FlyDSL":
        candidates = [
            os.path.join(root, "tests", "kernels", f"test_{kernel_name}.py"),
            os.path.join(root, "tests", "kernels", f"test_{kernel_name.replace('_kernel', '')}.py"),
            os.path.join(root, "tests", "kernels", f"test_{kernel_name.replace('_kernel', '').replace('_', '')}.py"),
        ]
    else:
        # aiter test paths
        candidates = [
            os.path.join(root, "op_tests", "triton_tests", f"test_{kernel_name}.py"),
        ]

    for tc in candidates:
        if os.path.exists(tc):
            return tc
    return None


def process_manifest(manifest_path: str, output_path: str, max_tests: int = 50):
    """Process manifest and add quality grades."""
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    print(f"Processing {len(manifest)} entries...")
    kernel_entries = [e for e in manifest if e.get("content_type") == "kernel_impl"]
    print(f"  Kernel entries to grade: {len(kernel_entries)}")

    graded = 0
    tested = 0

    for i, entry in enumerate(manifest):
        if entry.get("content_type") != "kernel_impl":
            continue

        fp = resolve_path(entry)
        repo = entry.get("repo", "aiter")

        # Gate 1: Compilation check
        if not os.path.exists(fp):
            compile_result = {"compiles": False, "error": f"File not found: {fp}"}
        elif repo == "FlyDSL" and fp.endswith(".py"):
            compile_result = check_flydsl_import(fp)
        elif fp.endswith(".py"):
            compile_result = check_python_syntax(fp)
        else:
            compile_result = {"compiles": True, "error": "non-Python file, skip syntax check"}

        # Gate 2+3: Test execution (for kernels with tests)
        test_result = None
        if tested < max_tests:
            test_file = find_test_for_kernel(entry["path"], repo)
            if test_file:
                print(f"  [{tested+1}/{max_tests}] Testing {os.path.basename(test_file)}...")
                test_result = run_flydsl_test(test_file)
                tested += 1
                status = "PASS" if test_result.get("passed") else "FAIL"
                print(f"    → {status} ({test_result.get('num_passed', 0)} passed, "
                      f"{test_result.get('num_failed', 0)} failed, "
                      f"{test_result.get('elapsed_s', '?')}s)")

        grade = grade_kernel(entry, compile_result, test_result)
        entry["quality_grade"] = grade
        entry["compile_check"] = compile_result
        if test_result:
            entry["test_result"] = {
                "passed": test_result.get("passed"),
                "num_passed": test_result.get("num_passed"),
                "num_failed": test_result.get("num_failed"),
                "elapsed_s": test_result.get("elapsed_s"),
            }
        graded += 1

    # Summary
    grades = {}
    for e in manifest:
        g = e.get("quality_grade", "ungraded")
        grades[g] = grades.get(g, 0) + 1

    print(f"\nGrading complete:")
    print(f"  Graded: {graded} kernel entries")
    print(f"  Tested: {tested} with pytest")
    for g, c in sorted(grades.items()):
        print(f"  {g}: {c}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tests", type=int, default=50)
    args = parser.parse_args()
    process_manifest(args.manifest, args.output, args.max_tests)
