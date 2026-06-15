#!/usr/bin/env python3
"""Performance benchmark for quality grading (Plan §4.3 Gate 3).

Runs actual GPU benchmarks on FlyDSL kernels and computes roofline efficiency
to produce quantitative performance labels: latency_us, throughput_tflops,
roofline_efficiency, and updated quality_grade (gold/silver/bronze).

This script MUST run inside the FlyDSL Docker container with GPU access.

Usage:
  python3 perf_benchmark.py \
      --manifest /path/to/graded_manifest.json \
      --output /path/to/perf_manifest.json \
      --perf-data /path/to/perf_data.jsonl

MI350X (gfx950) theoretical peaks:
  - FP16 MFMA: ~1300 TFLOPS
  - FP8 MFMA:  ~2600 TFLOPS
  - BF16 MFMA: ~1300 TFLOPS
  - Memory BW:  ~5.3 TB/s (HBM3e)
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

FLYDSL_ROOT = os.environ.get("FLYDSL_ROOT", "/FlyDSL")
AITER_ROOT = os.environ.get("AITER_ROOT", "/aiter")

# MI350X theoretical peak performance
HW_PEAKS = {
    "gfx950": {
        "fp16_tflops": 1300.0,
        "bf16_tflops": 1300.0,
        "fp8_tflops": 2600.0,
        "int8_tflops": 2600.0,
        "mem_bw_tbps": 5.3,
    },
    "gfx942": {
        "fp16_tflops": 653.0,
        "bf16_tflops": 653.0,
        "fp8_tflops": 1307.0,
        "int8_tflops": 1307.0,
        "mem_bw_tbps": 5.3,
    },
}

# Benchmark configs per operator
BENCH_CONFIGS = {
    "gemm": [
        {"M": 1, "N": 4096, "K": 4096, "dtype": "fp16"},
        {"M": 32, "N": 4096, "K": 4096, "dtype": "fp16"},
        {"M": 128, "N": 4096, "K": 14336, "dtype": "fp16"},
        {"M": 4096, "N": 4096, "K": 4096, "dtype": "fp16"},
        {"M": 4096, "N": 4096, "K": 4096, "dtype": "bf16"},
        {"M": 32, "N": 8192, "K": 8192, "dtype": "fp8"},
    ],
    "softmax": [
        {"M": 1024, "N": 8192, "dtype": "f32"},
        {"M": 32768, "N": 8192, "dtype": "bf16"},
    ],
    "rmsnorm": [
        {"M": 1024, "N": 8192, "dtype": "bf16"},
        {"M": 32768, "N": 8192, "dtype": "bf16"},
    ],
    "layernorm": [
        {"M": 1024, "N": 8192, "dtype": "bf16"},
    ],
}


def compute_gemm_flops(M, N, K):
    return 2 * M * N * K


def compute_elementwise_bytes(M, N, dtype):
    bytes_per = {"fp16": 2, "bf16": 2, "f16": 2, "f32": 4, "fp8": 1, "int8": 1}.get(dtype, 2)
    return M * N * bytes_per * 3  # read input + write output + read weights


def compute_roofline_efficiency(op, config, latency_us, hw="gfx950"):
    """Compute roofline efficiency based on operator type and hardware peak."""
    peaks = HW_PEAKS.get(hw, HW_PEAKS["gfx950"])
    latency_s = latency_us * 1e-6

    if op in ("gemm", "moe"):
        M = config.get("M", 1)
        N = config.get("N", 4096)
        K = config.get("K", 4096)
        flops = compute_gemm_flops(M, N, K)
        dtype = config.get("dtype", "fp16")
        peak_key = f"{dtype}_tflops" if f"{dtype}_tflops" in peaks else "fp16_tflops"
        peak_flops = peaks.get(peak_key, peaks["fp16_tflops"]) * 1e12
        achieved_tflops = flops / latency_s / 1e12
        return min(achieved_tflops / (peak_flops / 1e12), 1.0), achieved_tflops

    elif op in ("softmax", "rmsnorm", "layernorm", "rope"):
        M = config.get("M", 1024)
        N = config.get("N", 8192)
        dtype = config.get("dtype", "bf16")
        total_bytes = compute_elementwise_bytes(M, N, dtype)
        achieved_bw = total_bytes / latency_s / 1e12
        peak_bw = peaks["mem_bw_tbps"]
        return min(achieved_bw / peak_bw, 1.0), achieved_bw

    return 0.0, 0.0


def run_benchmark_pytest(test_file, marker="benchmark"):
    """Run benchmark tests and extract timing data."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short",
             "-m", marker, "--timeout=300", "-q"],
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "FLYDSL_RUNTIME_ENABLE_CACHE": "0",
                 "HIP_VISIBLE_DEVICES": "0"},
            cwd=FLYDSL_ROOT,
        )
        timings = []
        for line in result.stdout.split("\n"):
            time_match = re.search(r'(\d+\.?\d*)\s*(us|ms|s)\b', line)
            if time_match:
                val = float(time_match.group(1))
                unit = time_match.group(2)
                if unit == "ms":
                    val *= 1000
                elif unit == "s":
                    val *= 1e6
                timings.append(val)
        return {
            "success": result.returncode == 0,
            "timings_us": timings,
            "median_us": sorted(timings)[len(timings)//2] if timings else None,
            "output": result.stdout[-2000:],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_benchmark_common(op, configs, warmup=10, iters=50):
    """Run FlyDSL benchmark_common framework for elementwise ops."""
    script = f"""
import sys, os, json
sys.path.insert(0, '{FLYDSL_ROOT}')
os.environ['FLYDSL_RUNTIME_ENABLE_CACHE'] = '0'
from tests.kernels.benchmark_common import _bench_flydsl_torch
results = []
for cfg in {json.dumps(configs)}:
    try:
        us = _bench_flydsl_torch(op='{op}', M=cfg['M'], N=cfg['N'], dtype=cfg['dtype'],
                                  warmup={warmup}, iters={iters})
        results.append({{'config': cfg, 'latency_us': us, 'success': us is not None}})
    except Exception as e:
        results.append({{'config': cfg, 'latency_us': None, 'success': False, 'error': str(e)[:200]}})
print(json.dumps(results))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, "HIP_VISIBLE_DEVICES": "0"},
            cwd=FLYDSL_ROOT,
        )
        if result.returncode == 0:
            return json.loads(result.stdout.strip().split("\n")[-1])
        return [{"success": False, "error": result.stderr[-500:]}]
    except Exception as e:
        return [{"success": False, "error": str(e)}]


def run_gemm_benchmark(kernel_name, configs, warmup=10, iters=50):
    """Run GEMM-specific benchmark using test infrastructure."""
    test_file = os.path.join(FLYDSL_ROOT, "tests", "kernels", f"test_{kernel_name}.py")
    if not os.path.exists(test_file):
        test_file = os.path.join(FLYDSL_ROOT, "tests", "kernels",
                                  f"bench_{kernel_name}.py")
    if not os.path.exists(test_file):
        return [{"success": False, "error": f"no test file for {kernel_name}"}]

    return run_benchmark_pytest(test_file)


def grade_by_performance(efficiency):
    """Assign quality grade based on roofline efficiency (Plan §4.3)."""
    if efficiency >= 0.70:
        return "gold"
    elif efficiency >= 0.40:
        return "silver"
    else:
        return "bronze"


def process_manifest(manifest_path, output_path, perf_data_path):
    """Process manifest and add performance labels."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    kernel_entries = [e for e in manifest if e.get("content_type") == "kernel_impl"]
    print(f"Total entries: {len(manifest)}, kernels to benchmark: {len(kernel_entries)}")

    perf_results = []
    benchmarked = 0
    grades = {"gold": 0, "silver": 0, "bronze": 0, "reject": 0, "ungraded": 0}

    for entry in kernel_entries:
        op = entry.get("operator", "custom")
        kernel_name = Path(entry.get("path", "")).stem
        repo = entry.get("repo", "aiter")

        configs = BENCH_CONFIGS.get(op, [])
        if not configs or repo != "FlyDSL":
            entry.setdefault("quality_grade", "ungraded")
            grades[entry.get("quality_grade", "ungraded")] += 1
            continue

        print(f"  Benchmarking {kernel_name} ({op})...")

        if op in ("softmax", "rmsnorm", "layernorm"):
            results = run_benchmark_common(op, configs)
        else:
            results = run_gemm_benchmark(kernel_name, configs)

        if isinstance(results, dict):
            results = [results]

        best_efficiency = 0.0
        best_throughput = 0.0
        latencies = []

        for r in results:
            if not r.get("success"):
                continue
            lat = r.get("latency_us") or r.get("median_us")
            if lat and lat > 0:
                latencies.append(lat)
                cfg = r.get("config", configs[0] if configs else {})
                eff, throughput = compute_roofline_efficiency(op, cfg, lat)
                if eff > best_efficiency:
                    best_efficiency = eff
                    best_throughput = throughput

        if latencies:
            benchmarked += 1
            median_lat = sorted(latencies)[len(latencies) // 2]
            perf_grade = grade_by_performance(best_efficiency)

            entry["perf_latency_us"] = round(median_lat, 2)
            entry["perf_throughput"] = round(best_throughput, 4)
            entry["perf_roofline_efficiency"] = round(best_efficiency, 4)
            entry["perf_grade"] = perf_grade
            entry["quality_grade"] = perf_grade
            entry["perf_benchmarked"] = True

            perf_results.append({
                "path": entry["path"],
                "operator": op,
                "kernel": kernel_name,
                "latency_us": round(median_lat, 2),
                "throughput": round(best_throughput, 4),
                "roofline_efficiency": round(best_efficiency, 4),
                "grade": perf_grade,
                "n_configs": len(latencies),
            })
            grades[perf_grade] += 1
            print(f"    -> {perf_grade} (efficiency={best_efficiency:.1%}, latency={median_lat:.1f}us)")
        else:
            entry.setdefault("quality_grade", "ungraded")
            grades[entry.get("quality_grade", "ungraded")] += 1

    # Save results
    with open(output_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    with open(perf_data_path, "w") as f:
        for r in perf_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nBenchmark results:")
    print(f"  Benchmarked: {benchmarked}/{len(kernel_entries)} kernels")
    print(f"  Grades: {grades}")
    print(f"  Saved manifest: {output_path}")
    print(f"  Saved perf data: {perf_data_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/home/danyzhan/flydsl-agent-dataset/metadata/graded_manifest.json")
    parser.add_argument("--output", default="/home/danyzhan/flydsl-agent-dataset/metadata/perf_manifest.json")
    parser.add_argument("--perf-data", default="/home/danyzhan/flydsl-agent-dataset/metadata/perf_data.jsonl")
    args = parser.parse_args()

    process_manifest(args.manifest, args.output, args.perf_data)
