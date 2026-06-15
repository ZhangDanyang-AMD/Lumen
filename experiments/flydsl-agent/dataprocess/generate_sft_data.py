#!/usr/bin/env python3
"""SFT (Supervised Fine-Tuning) data generation.

Generates instruction-output pairs from aiter kernel code and documentation.
Sources:
  A. Kernel reverse-annotation: generate instructions from existing kernel code
  B. Test parameterization: extract pytest parameter combos → instruction pairs
  C. Documentation extraction: convert docs → QA pairs
  D. Cross-backend translation: Triton ↔ reference alignment pairs
  E. Git history: extract bugfix/optimization commit diffs
  F. Tuned config pairs: CSV configs → parameter recommendation instructions
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a FlyDSL GPU kernel programming expert. "
    "Generate compilable, correct, high-performance FlyDSL kernel code "
    "based on user requirements."
)


def generate_from_kernel_code(manifest: list, repo_root: str) -> list:
    """Source A: Generate SFT pairs by reverse-annotating kernel implementations."""
    pairs = []
    for entry in manifest:
        if entry.get("content_type") != "kernel_impl":
            continue
        if entry.get("priority") not in ("P0", "P1"):
            continue

        filepath = os.path.join(repo_root, entry["path"])
        if not os.path.exists(filepath):
            continue
        try:
            with open(filepath, encoding="utf-8", errors="replace") as f:
                code = f.read()
        except Exception:
            continue

        if len(code.strip()) < 100:
            continue

        op = entry.get("operator", "custom")
        hw = ", ".join(entry.get("hardware", ["generic"]))
        features = entry.get("features", [])
        complexity = entry.get("complexity", "intermediate")

        feature_str = ""
        if features:
            feature_str = f"\n- Features: {', '.join(features)}"

        # Style 1: precise technical specification
        inst_precise = (
            f"Implement a {op} kernel for AMD GPU ({hw}) using the aiter/Triton framework.\n"
            f"Requirements:\n"
            f"- Operator: {op}\n"
            f"- Target hardware: {hw}\n"
            f"- Complexity: {complexity}"
            f"{feature_str}\n"
            f"Provide complete, compilable code with all necessary imports."
        )

        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": inst_precise},
                {"role": "assistant", "content": f"```python\n{code}\n```"},
            ],
            "metadata": {
                "source": "kernel_reverse_annotation",
                "style": "precise_technical",
                "source_path": entry["path"],
                "operator": op,
                "complexity": complexity,
            },
        })

        # Style 2: natural language request
        inst_natural = (
            f"I need a high-performance {op} implementation for AMD {hw} GPUs. "
            f"The kernel should be written using Triton/aiter patterns"
            f"{' with ' + ', '.join(features) if features else ''}. "
            f"Please write the complete implementation."
        )

        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": inst_natural},
                {"role": "assistant", "content": f"```python\n{code}\n```"},
            ],
            "metadata": {
                "source": "kernel_reverse_annotation",
                "style": "natural_request",
                "source_path": entry["path"],
                "operator": op,
                "complexity": complexity,
            },
        })

    return pairs


def generate_from_tests(manifest: list, repo_root: str) -> list:
    """Source B: Extract test parameters to generate specification-based SFT pairs."""
    pairs = []
    for entry in manifest:
        if entry.get("content_type") != "test":
            continue

        filepath = os.path.join(repo_root, entry["path"])
        if not os.path.exists(filepath):
            continue
        try:
            with open(filepath, encoding="utf-8", errors="replace") as f:
                code = f.read()
        except Exception:
            continue

        op = entry.get("operator", "custom")
        if op == "custom":
            continue

        # Extract pytest.mark.parametrize blocks
        param_blocks = re.findall(
            r'@pytest\.mark\.parametrize\([^)]+\)', code, re.DOTALL
        )
        if not param_blocks:
            continue

        test_instruction = (
            f"Write a comprehensive test for the {op} operator on AMD GPU.\n"
            f"The test should cover multiple parameter combinations and verify correctness "
            f"against a PyTorch reference implementation.\n"
            f"Include parameterized test cases for different shapes, dtypes, and configurations."
        )

        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": test_instruction},
                {"role": "assistant", "content": f"```python\n{code}\n```"},
            ],
            "metadata": {
                "source": "test_parameterization",
                "source_path": entry["path"],
                "operator": op,
                "num_param_blocks": len(param_blocks),
            },
        })

    return pairs


def generate_from_docs(manifest: list, repo_root: str) -> list:
    """Source C: Convert documentation into QA-style SFT pairs."""
    pairs = []
    for entry in manifest:
        if entry.get("content_type") not in ("doc_guide", "expert_skill", "repo_guide"):
            continue

        filepath = os.path.join(repo_root, entry["path"])
        if not os.path.exists(filepath):
            continue
        try:
            with open(filepath, encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            continue

        if len(content.strip()) < 200:
            continue

        # Extract sections by markdown headers
        sections = re.split(r'\n##?\s+', content)
        for i, section in enumerate(sections):
            if len(section.strip()) < 100:
                continue

            lines = section.strip().split("\n")
            title = lines[0].strip().rstrip("#").strip() if lines else "Overview"
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else section.strip()

            if len(body) < 100:
                continue

            question = (
                f"Explain the following concept from the aiter/AITER documentation: {title}\n"
                f"Provide a detailed explanation with code examples where applicable."
            )

            pairs.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": body},
                ],
                "metadata": {
                    "source": "documentation_qa",
                    "source_path": entry["path"],
                    "section_title": title[:100],
                    "content_type": entry["content_type"],
                },
            })

    return pairs


def generate_from_tuned_configs(repo_root: str) -> list:
    """Source F: Generate parameter recommendation pairs from tuned CSVs."""
    pairs = []
    configs_dir = os.path.join(repo_root, "aiter", "configs")
    if not os.path.isdir(configs_dir):
        return pairs

    for csv_file in sorted(Path(configs_dir).glob("*.csv")):
        try:
            with open(csv_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception:
            continue

        if not rows:
            continue

        op_name = csv_file.stem.replace("_tuned_", " ").replace("_untuned_", " untuned ")
        op_name = op_name.replace("_", " ")

        # Summarize the config file
        columns = list(rows[0].keys()) if rows else []
        sample_rows = rows[:5]
        sample_text = ""
        for row in sample_rows:
            sample_text += "  " + ", ".join(f"{k}={v}" for k, v in row.items()) + "\n"

        instruction = (
            f"What are the tuned configurations for {op_name} on AMD GPU?\n"
            f"Show the optimal parameter settings from the aiter tuning database."
        )

        response = (
            f"Here are the tuned configurations for {op_name}:\n\n"
            f"Columns: {', '.join(columns)}\n"
            f"Total configurations: {len(rows)}\n\n"
            f"Sample entries:\n{sample_text}\n"
            f"These configurations were auto-tuned for optimal performance on AMD Instinct GPUs."
        )

        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "source": "tuned_config",
                "source_path": str(csv_file.relative_to(repo_root)),
                "num_configs": len(rows),
            },
        })

    return pairs


def generate_from_git_history(repo_root: str, max_commits: int = 200) -> list:
    """Source E: Extract meaningful commit diffs for bugfix/optimization SFT pairs."""
    import subprocess

    pairs = []

    try:
        result = subprocess.run(
            ["git", "log", f"--max-count={max_commits}", "--oneline",
             "--diff-filter=M", "--", "aiter/ops/", "op_tests/"],
            cwd=repo_root, capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return pairs

    if result.returncode != 0:
        return pairs

    commit_lines = result.stdout.strip().split("\n")
    keywords = ["fix", "bug", "optim", "perf", "improve", "refactor", "correct"]

    for line in commit_lines[:100]:
        if not line.strip():
            continue
        parts = line.split(" ", 1)
        if len(parts) < 2:
            continue
        sha, msg = parts
        msg_lower = msg.lower()

        if not any(kw in msg_lower for kw in keywords):
            continue

        try:
            diff_result = subprocess.run(
                ["git", "diff", f"{sha}~1", sha, "--stat"],
                cwd=repo_root, capture_output=True, text=True, timeout=10,
            )
        except Exception:
            continue

        if diff_result.returncode != 0:
            continue

        instruction = (
            f"The following commit was made to the aiter codebase:\n"
            f"Commit: {sha}\n"
            f"Message: {msg}\n\n"
            f"Explain what this change does and why it was necessary."
        )

        response = (
            f"This commit ({sha}) addresses: {msg}\n\n"
            f"Changed files:\n{diff_result.stdout[:2000]}\n\n"
            f"This change improves the aiter kernel implementation by addressing "
            f"the issue described in the commit message."
        )

        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "source": "git_history",
                "commit_sha": sha,
                "commit_msg": msg,
            },
        })

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate SFT training data from aiter")
    parser.add_argument("--manifest", default="/workspace/dataprocess/output/manifest.json")
    parser.add_argument("--repo", default="/workspace/aiter")
    parser.add_argument("--output", default="/workspace/dataprocess/output/sft_data.jsonl")
    parser.add_argument("--skip-git", action="store_true",
                        help="Skip git history extraction (if not in a git repo)")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    all_pairs = []

    print("Generating SFT pairs from kernel code...")
    pairs_a = generate_from_kernel_code(manifest, args.repo)
    all_pairs.extend(pairs_a)
    print(f"  Source A (kernel reverse-annotation): {len(pairs_a)} pairs")

    print("Generating SFT pairs from tests...")
    pairs_b = generate_from_tests(manifest, args.repo)
    all_pairs.extend(pairs_b)
    print(f"  Source B (test parameterization): {len(pairs_b)} pairs")

    print("Generating SFT pairs from documentation...")
    pairs_c = generate_from_docs(manifest, args.repo)
    all_pairs.extend(pairs_c)
    print(f"  Source C (documentation QA): {len(pairs_c)} pairs")

    print("Generating SFT pairs from tuned configs...")
    pairs_f = generate_from_tuned_configs(args.repo)
    all_pairs.extend(pairs_f)
    print(f"  Source F (tuned configs): {len(pairs_f)} pairs")

    if not args.skip_git:
        print("Generating SFT pairs from git history...")
        pairs_e = generate_from_git_history(args.repo)
        all_pairs.extend(pairs_e)
        print(f"  Source E (git history): {len(pairs_e)} pairs")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nTotal SFT pairs: {len(all_pairs)}")
    print(f"Output: {args.output}")

    # Summary by source
    by_source = {}
    for p in all_pairs:
        src = p.get("metadata", {}).get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
    print("\nBy source:")
    for k, v in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"  {k:30s}: {v}")


if __name__ == "__main__":
    main()
