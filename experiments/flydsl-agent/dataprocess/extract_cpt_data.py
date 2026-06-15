#!/usr/bin/env python3
"""CPT (Continued Pre-Training) data extraction and weighting.

Reads manifest.json, reads source files, wraps each in a structured document
format with metadata headers, and assigns sampling weights based on taxonomy
labels. Outputs a JSONL file where each line is a weighted training document.
"""

import argparse
import json
import os
from pathlib import Path


# Sampling weight multipliers per attribute (from plan §4.5)
PRIORITY_WEIGHT = {"P0": 3.0, "P1": 1.5, "P2": 0.5}

GRADE_WEIGHT = {
    "gold": 1.5,
    "silver": 1.0,
    "bronze": 0.7,
    "ungraded": 1.0,
    "reject": 0.0,
}

CONTENT_TYPE_WEIGHT = {
    "expert_skill": 2.5,
    "repo_guide": 2.0,
    "kernel_impl": 1.2,
    "framework_api": 1.0,
    "doc_guide": 1.0,
    "test": 0.8,
    "tuned_config": 0.6,
    "hardware_spec": 0.3,
    "build_script": 0.2,
}


def compute_weight(entry: dict) -> float:
    base = PRIORITY_WEIGHT.get(entry.get("priority", "P2"), 0.5)
    grade_mult = GRADE_WEIGHT.get(entry.get("quality_grade", "ungraded"), 1.0)
    type_mult = CONTENT_TYPE_WEIGHT.get(entry.get("content_type", "framework_api"), 1.0)
    return round(base * grade_mult * type_mult, 3)


def format_cpt_document(entry: dict, content: str) -> str:
    """Wrap source file in structured CPT document format."""
    meta_lines = [
        f"[file: {entry['path']}]",
        f"[type: {entry.get('content_type', 'unknown')}]",
        f"[operator: {entry.get('operator', 'unknown')}]",
        f"[hardware: {', '.join(entry.get('hardware', ['generic']))}]",
        f"[complexity: {entry.get('complexity', 'unknown')}]",
    ]
    grade = entry.get("quality_grade", "ungraded")
    if grade != "ungraded":
        meta_lines.append(f"[grade: {grade}]")

    header = "\n".join(meta_lines)
    return f"<|doc_start|>\n{header}\n\n{content}\n<|doc_end|>"


def extract_cpt_data(manifest_path: str, repo_root: str, output_path: str,
                     min_lines: int = 5, max_tokens: int = 50000):
    with open(manifest_path) as f:
        manifest = json.load(f)

    stats = {"total": 0, "included": 0, "skipped_short": 0, "skipped_reject": 0,
             "skipped_too_long": 0, "total_tokens": 0, "by_type": {}, "by_priority": {}}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for entry in manifest:
            stats["total"] += 1

            if entry.get("quality_grade") == "reject":
                stats["skipped_reject"] += 1
                continue

            if entry.get("lines", 0) < min_lines:
                stats["skipped_short"] += 1
                continue

            filepath = os.path.join(repo_root, entry["path"])
            if not os.path.exists(filepath):
                continue

            try:
                with open(filepath, encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception:
                continue

            tokens = entry.get("tokens_approx", len(content) // 4)
            if tokens > max_tokens:
                stats["skipped_too_long"] += 1
                continue

            weight = compute_weight(entry)
            if weight <= 0:
                stats["skipped_reject"] += 1
                continue

            document = format_cpt_document(entry, content)

            record = {
                "text": document,
                "weight": weight,
                "path": entry["path"],
                "content_type": entry.get("content_type"),
                "operator": entry.get("operator"),
                "priority": entry.get("priority"),
                "tokens_approx": tokens,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            stats["included"] += 1
            stats["total_tokens"] += tokens
            ct = entry.get("content_type", "unknown")
            stats["by_type"][ct] = stats["by_type"].get(ct, 0) + 1
            pr = entry.get("priority", "unknown")
            stats["by_priority"][pr] = stats["by_priority"].get(pr, 0) + 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Extract CPT training data from aiter")
    parser.add_argument("--manifest", default="/workspace/dataprocess/output/manifest.json")
    parser.add_argument("--repo", default="/workspace/aiter")
    parser.add_argument("--output", default="/workspace/dataprocess/output/cpt_data.jsonl")
    parser.add_argument("--min-lines", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=50000)
    args = parser.parse_args()

    stats = extract_cpt_data(args.manifest, args.repo, args.output,
                             args.min_lines, args.max_tokens)

    print("\nCPT Data Extraction Summary")
    print("=" * 50)
    print(f"Total files scanned:  {stats['total']}")
    print(f"Included:             {stats['included']}")
    print(f"Skipped (short):      {stats['skipped_short']}")
    print(f"Skipped (reject):     {stats['skipped_reject']}")
    print(f"Skipped (too long):   {stats['skipped_too_long']}")
    print(f"Total approx tokens:  {stats['total_tokens']:,}")
    print(f"\nBy content type:")
    for k, v in sorted(stats["by_type"].items(), key=lambda x: -x[1]):
        print(f"  {k:20s}: {v}")
    print(f"\nBy priority:")
    for k, v in sorted(stats["by_priority"].items(), key=lambda x: -x[1]):
        print(f"  {k:5s}: {v}")

    stats_path = args.output.replace(".jsonl", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")


if __name__ == "__main__":
    main()
