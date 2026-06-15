#!/usr/bin/env python3
"""RL Task Specification Generator.

Builds a library of task specifications for reinforcement learning training.
Each spec defines an operator, target hardware, parameter ranges, and
baseline performance. Specs are derived from:
  1. Manifest entries with gold/silver quality grades
  2. Tuned config CSVs (aiter/configs/)
  3. Exploration specs beyond existing kernel coverage
"""

import argparse
import csv
import json
import os
from pathlib import Path


# Standard shape ranges for RL exploration
GEMM_SHAPES = [
    {"M": 4096, "N": 4096, "K": 2048},
    {"M": 2048, "N": 4096, "K": 1024},
    {"M": 8192, "N": 4096, "K": 2048},
    {"M": 512, "N": 4096, "K": 2048},
    {"M": 128, "N": 4096, "K": 2048},
    {"M": 256, "N": 8192, "K": 4096},
    {"M": 1024, "N": 4096, "K": 4096},
    {"M": 4096, "N": 8192, "K": 1024},
]

ATTENTION_SHAPES = [
    {"batch": 1, "heads": 32, "seq_len": 2048, "head_dim": 128},
    {"batch": 4, "heads": 32, "seq_len": 4096, "head_dim": 128},
    {"batch": 8, "heads": 64, "seq_len": 2048, "head_dim": 64},
    {"batch": 1, "heads": 128, "seq_len": 8192, "head_dim": 128},
    {"batch": 2, "heads": 32, "seq_len": 16384, "head_dim": 128},
]

MOE_SHAPES = [
    {"M": 256, "N": 2048, "K": 1024, "experts": 8, "topk": 2},
    {"M": 512, "N": 4096, "K": 2048, "experts": 16, "topk": 4},
    {"M": 1024, "N": 4096, "K": 2048, "experts": 64, "topk": 8},
    {"M": 256, "N": 11008, "K": 4096, "experts": 8, "topk": 2},
]

NORM_SHAPES = [
    {"batch": 1, "hidden_dim": 4096},
    {"batch": 8, "hidden_dim": 4096},
    {"batch": 32, "hidden_dim": 8192},
    {"batch": 1, "hidden_dim": 14336},
]


def specs_from_manifest(manifest: list) -> list:
    """Generate RL specs from gold/silver manifest entries."""
    specs = []
    for entry in manifest:
        if entry.get("content_type") != "kernel_impl":
            continue
        grade = entry.get("quality_grade", "ungraded")
        if grade not in ("gold", "silver", "ungraded"):
            continue

        op = entry.get("operator", "custom")
        if op == "custom":
            continue

        for hw in entry.get("hardware", ["generic"]):
            if hw == "generic":
                hw = "gfx942"

            specs.append({
                "id": f"manifest_{op}_{hw}_{len(specs)}",
                "operator": op,
                "hardware": hw,
                "source": "manifest",
                "source_path": entry["path"],
                "features": entry.get("features", []),
                "baseline_grade": grade,
            })

    return specs


def specs_from_tuned_configs(repo_root: str) -> list:
    """Generate RL specs from aiter tuned config CSVs."""
    specs = []
    configs_dir = os.path.join(repo_root, "aiter", "configs")
    if not os.path.isdir(configs_dir):
        return specs

    for csv_file in sorted(Path(configs_dir).glob("*tuned*.csv")):
        if "untuned" in csv_file.name:
            continue

        op_name = csv_file.stem
        operator = "gemm"
        if "moe" in op_name or "fmoe" in op_name:
            operator = "moe"
        elif "batched" in op_name:
            operator = "gemm"

        dtypes = []
        if "a8w8" in op_name:
            dtypes = ["int8"]
        elif "a4w4" in op_name:
            dtypes = ["int4"]
        elif "bf16" in op_name:
            dtypes = ["bf16"]
        elif "fp8" in op_name:
            dtypes = ["fp8"]

        try:
            with open(csv_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception:
            continue

        for i, row in enumerate(rows[:20]):
            params = {}
            for k, v in row.items():
                try:
                    params[k] = int(v)
                except (ValueError, TypeError):
                    try:
                        params[k] = float(v)
                    except (ValueError, TypeError):
                        params[k] = v

            specs.append({
                "id": f"tuned_{op_name}_{i}",
                "operator": operator,
                "hardware": "gfx942",
                "params": params,
                "dtypes": dtypes,
                "source": "tuned_config",
                "source_file": str(csv_file.name),
            })

    return specs


def generate_exploration_specs() -> list:
    """Generate exploration specs beyond existing kernel coverage."""
    specs = []
    spec_id = 0

    for hw in ["gfx942", "gfx950"]:
        for dtype in ["fp8", "bf16"]:
            for shape in GEMM_SHAPES:
                specs.append({
                    "id": f"explore_gemm_{spec_id}",
                    "operator": "gemm",
                    "hardware": hw,
                    "params": {**shape, "in_dtype": dtype, "out_dtype": "bf16"},
                    "source": "exploration",
                })
                spec_id += 1

        for shape in ATTENTION_SHAPES:
            specs.append({
                "id": f"explore_attn_{spec_id}",
                "operator": "flash_attn",
                "hardware": hw,
                "params": shape,
                "source": "exploration",
            })
            spec_id += 1

        for shape in MOE_SHAPES:
            for dtype in ["fp8", "bf16"]:
                specs.append({
                    "id": f"explore_moe_{spec_id}",
                    "operator": "moe",
                    "hardware": hw,
                    "params": {**shape, "in_dtype": dtype},
                    "source": "exploration",
                })
                spec_id += 1

        for shape in NORM_SHAPES:
            for op in ["rmsnorm", "softmax", "layernorm"]:
                specs.append({
                    "id": f"explore_{op}_{spec_id}",
                    "operator": op,
                    "hardware": hw,
                    "params": shape,
                    "source": "exploration",
                })
                spec_id += 1

    return specs


def main():
    parser = argparse.ArgumentParser(description="Generate RL task specifications")
    parser.add_argument("--manifest", default="/workspace/dataprocess/output/manifest.json")
    parser.add_argument("--repo", default="/workspace/aiter")
    parser.add_argument("--output", default="/workspace/dataprocess/output/rl_specs.json")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    manifest_specs = specs_from_manifest(manifest)
    tuned_specs = specs_from_tuned_configs(args.repo)
    explore_specs = generate_exploration_specs()

    all_specs = manifest_specs + tuned_specs + explore_specs

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_specs, f, indent=2, ensure_ascii=False)

    print(f"\nRL Spec Generation Summary")
    print("=" * 50)
    print(f"From manifest:       {len(manifest_specs)}")
    print(f"From tuned configs:  {len(tuned_specs)}")
    print(f"Exploration specs:   {len(explore_specs)}")
    print(f"Total specs:         {len(all_specs)}")

    by_op = {}
    for s in all_specs:
        op = s.get("operator", "unknown")
        by_op[op] = by_op.get(op, 0) + 1
    print(f"\nBy operator:")
    for k, v in sorted(by_op.items(), key=lambda x: -x[1]):
        print(f"  {k:15s}: {v}")

    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
