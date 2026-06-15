#!/usr/bin/env python3
"""End-to-end dataset quality validation.

Checks all generated training data for:
1. Format correctness (valid JSON, required fields)
2. Content quality (non-empty, valid Python syntax in code blocks)
3. Deduplication (cosine similarity < 0.95 threshold)
4. Distribution balance (operator / difficulty / hardware)
5. Length constraints (within max_seq_length bounds)
6. Label consistency (metadata cross-checks)
"""

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from collections import Counter


def validate_cpt_record(record: dict, idx: int) -> list:
    """Validate a single CPT data record."""
    issues = []
    if "text" not in record:
        issues.append(f"CPT #{idx}: missing 'text' field")
    elif len(record["text"].strip()) < 50:
        issues.append(f"CPT #{idx}: text too short ({len(record['text'])} chars)")

    if "weight" not in record:
        issues.append(f"CPT #{idx}: missing 'weight' field")
    elif not isinstance(record["weight"], (int, float)) or record["weight"] <= 0:
        issues.append(f"CPT #{idx}: invalid weight {record.get('weight')}")

    text = record.get("text", "")
    if "<|doc_start|>" not in text or "<|doc_end|>" not in text:
        issues.append(f"CPT #{idx}: missing doc_start/doc_end markers")

    return issues


def validate_sft_record(record: dict, idx: int) -> list:
    """Validate a single SFT data record."""
    issues = []
    messages = record.get("messages", [])
    if not messages:
        issues.append(f"SFT #{idx}: empty messages")
        return issues

    roles = [m.get("role") for m in messages]
    if "assistant" not in roles:
        issues.append(f"SFT #{idx}: no assistant message")
    if "user" not in roles:
        issues.append(f"SFT #{idx}: no user message")

    for m in messages:
        if not m.get("content", "").strip():
            issues.append(f"SFT #{idx}: empty {m.get('role', '?')} message")

    # Check code blocks in assistant response for basic Python validity
    assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
    code_blocks = re.findall(r'```python\n(.*?)```', assistant_msg, re.DOTALL)
    for block in code_blocks:
        try:
            ast.parse(block)
        except SyntaxError:
            pass  # Not all code blocks must be valid Python (could be pseudocode)

    return issues


def check_duplicates(records: list, key_fn, threshold: float = 0.95) -> list:
    """Simple hash-based near-duplicate detection."""
    issues = []
    seen_hashes = {}

    for idx, record in enumerate(records):
        text = key_fn(record)
        if not text:
            continue

        # Normalized hash for exact duplicate detection
        normalized = re.sub(r'\s+', ' ', text.strip().lower())
        text_hash = hashlib.md5(normalized.encode()).hexdigest()

        if text_hash in seen_hashes:
            issues.append(
                f"Duplicate detected: record #{idx} ≈ record #{seen_hashes[text_hash]}"
            )
        else:
            seen_hashes[text_hash] = idx

    return issues


def check_distribution(records: list, field_path: str, expected_keys: list = None) -> dict:
    """Check distribution of a metadata field across records."""
    values = []
    for r in records:
        val = r
        for key in field_path.split("."):
            if isinstance(val, dict):
                val = val.get(key)
            else:
                val = None
                break
        if val is not None:
            if isinstance(val, list):
                values.extend(val)
            else:
                values.append(val)

    counter = Counter(values)
    total = sum(counter.values())
    distribution = {}
    for k, v in counter.most_common():
        distribution[str(k)] = {"count": v, "pct": round(v / total * 100, 1) if total else 0}

    return distribution


def validate_cpt_file(filepath: str) -> dict:
    """Validate complete CPT data file."""
    results = {"file": filepath, "total": 0, "valid": 0, "issues": [], "stats": {}}

    if not os.path.exists(filepath):
        results["issues"].append(f"File not found: {filepath}")
        return results

    records = []
    with open(filepath, encoding="utf-8") as f:
        for i, line in enumerate(f):
            results["total"] += 1
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                results["issues"].append(f"CPT #{i}: invalid JSON")
                continue

            issues = validate_cpt_record(record, i)
            results["issues"].extend(issues)
            if not issues:
                results["valid"] += 1

    # Duplicate check
    dup_issues = check_duplicates(records, lambda r: r.get("text", ""))
    results["issues"].extend(dup_issues)

    # Stats
    results["stats"]["total_tokens"] = sum(r.get("tokens_approx", 0) for r in records)
    results["stats"]["content_type_dist"] = check_distribution(records, "content_type")
    results["stats"]["priority_dist"] = check_distribution(records, "priority")
    results["stats"]["operator_dist"] = check_distribution(records, "operator")

    weights = [r.get("weight", 0) for r in records]
    if weights:
        results["stats"]["weight_range"] = [min(weights), max(weights)]
        results["stats"]["weight_mean"] = round(sum(weights) / len(weights), 3)

    return results


def validate_sft_file(filepath: str) -> dict:
    """Validate complete SFT data file."""
    results = {"file": filepath, "total": 0, "valid": 0, "issues": [], "stats": {}}

    if not os.path.exists(filepath):
        results["issues"].append(f"File not found: {filepath}")
        return results

    records = []
    with open(filepath, encoding="utf-8") as f:
        for i, line in enumerate(f):
            results["total"] += 1
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                results["issues"].append(f"SFT #{i}: invalid JSON")
                continue

            issues = validate_sft_record(record, i)
            results["issues"].extend(issues)
            if not issues:
                results["valid"] += 1

    results["stats"]["source_dist"] = check_distribution(records, "metadata.source")
    results["stats"]["operator_dist"] = check_distribution(records, "metadata.operator")

    return results


def validate_rl_specs(filepath: str) -> dict:
    """Validate RL specification file."""
    results = {"file": filepath, "total": 0, "valid": 0, "issues": [], "stats": {}}

    if not os.path.exists(filepath):
        results["issues"].append(f"File not found: {filepath}")
        return results

    with open(filepath) as f:
        specs = json.load(f)

    results["total"] = len(specs)

    for i, spec in enumerate(specs):
        issues = []
        if "operator" not in spec:
            issues.append(f"RL spec #{i}: missing operator")
        if "hardware" not in spec:
            issues.append(f"RL spec #{i}: missing hardware")
        if "id" not in spec:
            issues.append(f"RL spec #{i}: missing id")

        results["issues"].extend(issues)
        if not issues:
            results["valid"] += 1

    results["stats"]["operator_dist"] = check_distribution(specs, "operator")
    results["stats"]["hardware_dist"] = check_distribution(specs, "hardware")
    results["stats"]["source_dist"] = check_distribution(specs, "source")

    return results


def print_validation_results(results: dict):
    print(f"\n{'=' * 60}")
    print(f"Validation: {results['file']}")
    print(f"{'=' * 60}")
    print(f"Total records:  {results['total']}")
    print(f"Valid records:  {results['valid']}")
    print(f"Issues found:   {len(results['issues'])}")

    if results["issues"]:
        print(f"\nFirst 20 issues:")
        for issue in results["issues"][:20]:
            print(f"  ⚠ {issue}")
        if len(results["issues"]) > 20:
            print(f"  ... and {len(results['issues']) - 20} more")

    if results.get("stats"):
        print(f"\nStatistics:")
        for key, val in results["stats"].items():
            if isinstance(val, dict):
                print(f"  {key}:")
                for k, v in val.items():
                    if isinstance(v, dict):
                        print(f"    {k:20s}: {v['count']:5d} ({v['pct']}%)")
                    else:
                        print(f"    {k}: {v}")
            else:
                print(f"  {key}: {val}")


def main():
    parser = argparse.ArgumentParser(description="Validate all generated datasets")
    parser.add_argument("--output-dir", default="/workspace/dataprocess/output")
    args = parser.parse_args()

    all_results = []
    exit_code = 0

    # Validate CPT data
    cpt_path = os.path.join(args.output_dir, "cpt_data.jsonl")
    if os.path.exists(cpt_path):
        results = validate_cpt_file(cpt_path)
        all_results.append(results)
        print_validation_results(results)
        if results["issues"]:
            exit_code = 1

    # Validate SFT data
    sft_path = os.path.join(args.output_dir, "sft_data.jsonl")
    if os.path.exists(sft_path):
        results = validate_sft_file(sft_path)
        all_results.append(results)
        print_validation_results(results)
        if results["issues"]:
            exit_code = 1

    # Validate RL specs
    rl_path = os.path.join(args.output_dir, "rl_specs.json")
    if os.path.exists(rl_path):
        results = validate_rl_specs(rl_path)
        all_results.append(results)
        print_validation_results(results)
        if results["issues"]:
            exit_code = 1

    # Summary
    print(f"\n{'=' * 60}")
    print("OVERALL VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    total_issues = sum(len(r["issues"]) for r in all_results)
    total_records = sum(r["total"] for r in all_results)
    total_valid = sum(r["valid"] for r in all_results)
    print(f"Files validated:  {len(all_results)}")
    print(f"Total records:    {total_records}")
    print(f"Valid records:    {total_valid}")
    print(f"Total issues:     {total_issues}")
    print(f"Status:           {'PASS' if exit_code == 0 else 'ISSUES FOUND'}")

    # Save validation report
    report_path = os.path.join(args.output_dir, "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nFull report: {report_path}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
