#!/usr/bin/env python3
"""Update graded_manifest.json with actual GPU benchmark performance data.

Reads perf_detailed_results.json from the parallel benchmark run and merges
real latency / TFLOPS / roofline efficiency into each kernel's manifest entry.
Assigns performance grades (gold/silver/bronze) based on measured efficiency.
"""

import json
import os
import re
import sys

MI350X_PEAK = {
    "fp8": 2600.0,
    "fp16": 1300.0,
    "bf16": 1300.0,
    "int8": 2600.0,
}

GOLD_THRESH = 0.70
SILVER_THRESH = 0.40
BRONZE_THRESH = 0.10


def classify_grade(efficiency):
    if efficiency >= GOLD_THRESH:
        return "gold"
    elif efficiency >= SILVER_THRESH:
        return "silver"
    elif efficiency >= BRONZE_THRESH:
        return "bronze"
    else:
        return "needs_optimization"


def main():
    perf_path = "/tmp/perf_detailed_results.json"
    manifest_path = os.path.expanduser("~/flydsl-agent-dataset/metadata/graded_manifest.json")
    output_path = manifest_path

    with open(perf_path) as f:
        perf_data = json.load(f)

    with open(manifest_path) as f:
        manifest = json.load(f)

    kernel_perf_map = {}
    for kname, perfs in perf_data.get("kernels", {}).items():
        if not perfs:
            continue

        peak_tflops = max(p["tflops"] for p in perfs)
        avg_tflops = sum(p["tflops"] for p in perfs) / len(perfs)
        min_latency = min(p["latency_us"] for p in perfs)

        is_fp8 = "fp8" in kname.lower()
        roofline_peak = MI350X_PEAK["fp8"] if is_fp8 else MI350X_PEAK["bf16"]
        peak_eff = peak_tflops / roofline_peak
        avg_eff = avg_tflops / roofline_peak

        kernel_perf_map[kname] = {
            "peak_tflops": round(peak_tflops, 2),
            "avg_tflops": round(avg_tflops, 2),
            "min_latency_us": round(min_latency, 2),
            "roofline_efficiency": round(peak_eff, 4),
            "avg_efficiency": round(avg_eff, 4),
            "num_configs_tested": len(perfs),
            "perf_grade": classify_grade(peak_eff),
            "perf_details": perfs[:10],
        }

    name_to_perf_key = {
        "preshuffle_gemm": ["preshuffle_gemm_fp8", "preshuffle_gemm_bf16"],
        "blockscale_preshuffle_gemm": ["blockscale_preshuffle_gemm"],
        "hgemm_splitk": ["hgemm_splitk"],
        "splitk_hgemm": ["hgemm_splitk"],
        "moe_gemm": ["moe_gemm"],
        "moe_gemm_2stage": ["moe_gemm"],
        "fp8_gemm_4wave": ["fp8_gemm_rowscale"],
        "fp8_gemm_8wave": ["fp8_gemm_rowscale"],
        "fp8_gemm_rowscale": ["fp8_gemm_rowscale"],
        "small_m_hgemm": ["hgemm_splitk"],
        "flash_attn": ["elementwise"],
        "softmax": ["elementwise"],
        "rmsnorm": ["elementwise"],
        "layernorm": ["elementwise"],
        "fused_rope": ["fused_rope_cache"],
        "topk_gating_softmax": ["elementwise"],
    }

    updated = 0
    entries = manifest if isinstance(manifest, list) else manifest.get("entries", manifest.get("files", []))

    for entry in entries:
        fp = entry.get("full_path", "") or entry.get("rel_path", "")
        fn = os.path.basename(fp).lower()

        matched_perf = None
        for kernel_name, perf_keys in name_to_perf_key.items():
            if kernel_name in fn:
                for pk in perf_keys:
                    if pk in kernel_perf_map:
                        matched_perf = kernel_perf_map[pk]
                        break
                if matched_perf:
                    break

        if matched_perf:
            entry["gpu_benchmark"] = {
                "device": "AMD MI350X (gfx950)",
                "peak_tflops": matched_perf["peak_tflops"],
                "avg_tflops": matched_perf["avg_tflops"],
                "min_latency_us": matched_perf["min_latency_us"],
                "roofline_efficiency": matched_perf["roofline_efficiency"],
                "num_configs_tested": matched_perf["num_configs_tested"],
            }
            entry["perf_grade"] = matched_perf["perf_grade"]

            old_grade = entry.get("training_grade", entry.get("quality_grade", ""))
            if matched_perf["perf_grade"] == "gold" and old_grade not in ("gold",):
                entry["training_grade"] = "gold"
            elif matched_perf["perf_grade"] == "silver" and old_grade in ("bronze", "needs_optimization", ""):
                entry["training_grade"] = "silver"

            updated += 1

    for entry in entries:
        if "gpu_benchmark" not in entry:
            fp = entry.get("full_path", "") or entry.get("rel_path", "")
            if entry.get("compile_ok"):
                entry["perf_grade"] = "compile_only"
            elif "kernel" in fp.lower() or "gemm" in fp.lower():
                entry["perf_grade"] = "untested"

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Updated {updated} manifest entries with GPU benchmark data")
    print(f"Manifest saved to {output_path}")

    print("\nPerformance grade distribution:")
    grades = {}
    for entry in entries:
        g = entry.get("perf_grade", "none")
        grades[g] = grades.get(g, 0) + 1
    for g, c in sorted(grades.items(), key=lambda x: -x[1]):
        print(f"  {g}: {c}")

    perf_summary_path = os.path.expanduser("~/flydsl-agent-dataset/metadata/perf_benchmark_summary.json")
    with open(perf_summary_path, "w") as f:
        json.dump({
            "device": "AMD MI350X (gfx950)",
            "gpu_count": 8,
            "total_benchmark_time_s": perf_data.get("total_time_s"),
            "kernels": kernel_perf_map,
            "grade_distribution": grades,
            "thresholds": {
                "gold": f">={GOLD_THRESH*100:.0f}% roofline",
                "silver": f">={SILVER_THRESH*100:.0f}% roofline",
                "bronze": f">={BRONZE_THRESH*100:.0f}% roofline",
            },
        }, f, indent=2)
    print(f"Performance summary saved to {perf_summary_path}")


if __name__ == "__main__":
    main()
