#!/usr/bin/env python3
"""Layer 1: Deterministic auto-annotation via static analysis.

Scans the aiter repository and generates a manifest.json with per-file metadata.
Labels are inferred from file paths, content patterns, and code structure.
No AI models or GPU required — purely rule-based.
"""

import argparse
import json
import os
import re
from pathlib import Path

import yaml


def load_taxonomy(taxonomy_path: str) -> dict:
    with open(taxonomy_path) as f:
        return yaml.safe_load(f)


TAXONOMY = None


def infer_operator(filepath: str, content: str) -> str:
    fp_lower = filepath.lower()
    content_head = content[:2000].lower()

    op_patterns = {
        "gemm": r"gemm|matmul|batched_gemm",
        "flash_attn": r"flash_attn|fmha|flash_attention",
        "mla": r"\bmla\b|multi_latent|mla_decode|mla_fwd",
        "moe": r"\bmoe\b|fused_moe|fmoe|mixture.of.expert",
        "softmax": r"softmax",
        "rmsnorm": r"rmsnorm|rms_norm",
        "layernorm": r"layernorm|layer_norm",
        "rope": r"\brope\b|rotary",
        "topk": r"\btopk\b|top_k",
        "paged_attn": r"paged_attn|paged_attention|page_attention",
        "allreduce": r"allreduce|all_reduce|reduce_scatter|all_gather",
        "quant": r"\bquant\b|quantiz|dequant|fp8_cast|blockscale",
    }

    for op, pattern in op_patterns.items():
        if re.search(pattern, fp_lower) or re.search(pattern, content_head):
            return op
    return "custom"


def infer_hardware(content: str) -> list:
    hw = list(set(re.findall(r"gfx\d+", content)))
    return hw if hw else ["generic"]


def infer_features(content: str) -> list:
    feature_patterns = {
        "swizzle_xor16": r"swizzle_xor|xor16|swizzle.*16",
        "pipeline": r"lds_stage|pipeline|num_stages|software.pipelining",
        "async_copy": r"async_copy|cp\.async|async_load",
        "double_buffer": r"double_buf|lds_stage\s*=\s*2|ping.pong",
        "preshuffle": r"preshuffle|pre_shuffle|bpreshuffle",
        "blockscale": r"blockscale|block_scale|block.scaled",
        "split_k": r"split_k|splitk",
        "epilogue_fusion": r"epilogue|bias_add|activation_fuse",
        "multi_wave": r"waves_per_eu|num_waves|multi_wave",
        "sage_attention": r"sage.attention|sage_attn",
        "mxfp4": r"mxfp4|mx_fp4",
    }
    return [f for f, p in feature_patterns.items() if re.search(p, content, re.IGNORECASE)]


def infer_complexity(content: str, features: list) -> str:
    lines = content.count("\n") + 1
    n_feat = len(features)
    if lines > 1500 or n_feat >= 3:
        return "expert"
    elif lines > 800 or n_feat >= 2:
        return "advanced"
    elif lines > 300:
        return "intermediate"
    return "beginner"


def infer_content_type_and_priority(filepath: str) -> tuple:
    fp = filepath.lower()

    if ".claude/skills" in fp:
        return "expert_skill", "P0"
    if fp.endswith("claude.md") or (fp.endswith("readme.md") and "/docs" not in fp):
        return "repo_guide", "P0"
    if "/configs/" in fp and fp.endswith(".csv"):
        return "tuned_config", "P1"

    if "/ops/triton/" in fp or "/ops/" in fp:
        if "test" in fp:
            return "test", "P1"
        return "kernel_impl", "P0"
    if "/csrc/" in fp or "/hsa/" in fp:
        return "kernel_impl", "P1"
    if "/op_tests/" in fp or "/tests/" in fp or "test_" in os.path.basename(fp):
        return "test", "P1"
    if fp.endswith(".md") or fp.endswith(".rst"):
        return "doc_guide", "P0"
    if "setup.py" in fp or "pyproject.toml" in fp or "/scripts/" in fp:
        return "build_script", "P2"
    if "/aiter/" in fp and fp.endswith(".py"):
        return "framework_api", "P0"

    return "framework_api", "P2"


def count_tokens_approx(text: str) -> int:
    """Rough token count: ~4 chars per token for code."""
    return len(text) // 4


def process_file(filepath: str, repo_root: str) -> dict:
    rel_path = os.path.relpath(filepath, repo_root)
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return {"path": rel_path, "error": str(e)}

    operator = infer_operator(rel_path, content)
    hardware = infer_hardware(content)
    features = infer_features(content)
    complexity = infer_complexity(content, features)
    content_type, priority = infer_content_type_and_priority(rel_path)

    return {
        "path": rel_path,
        "operator": operator,
        "hardware": hardware,
        "features": features,
        "complexity": complexity,
        "content_type": content_type,
        "priority": priority,
        "quality_grade": "ungraded",
        "lines": content.count("\n") + 1,
        "tokens_approx": count_tokens_approx(content),
        "needs_human_review": [],
    }


SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".tox", "build", "dist",
    "*.egg-info", "3rdparty",
}
SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".co", ".bin", ".png", ".jpg",
    ".gif", ".ico", ".whl", ".tar", ".gz", ".zip",
}
INCLUDE_EXTENSIONS = {
    ".py", ".md", ".rst", ".yaml", ".yml", ".csv", ".toml", ".cfg",
    ".cpp", ".hpp", ".h", ".hip", ".cu", ".cuh", ".s", ".asm", ".sh",
}


def should_skip(path: Path) -> bool:
    for part in path.parts:
        if part in SKIP_DIRS:
            return True
        if part.endswith(".egg-info"):
            return True
    return path.suffix in SKIP_EXTENSIONS


def scan_repo(repo_root: str) -> list:
    manifest = []
    root = Path(repo_root)
    for filepath in sorted(root.rglob("*")):
        if not filepath.is_file():
            continue
        if should_skip(filepath):
            continue
        if filepath.suffix not in INCLUDE_EXTENSIONS:
            continue
        entry = process_file(str(filepath), repo_root)
        manifest.append(entry)
    return manifest


def print_summary(manifest: list):
    total = len(manifest)
    print(f"\nManifest Summary: {total} files")
    print("-" * 50)

    by_type = {}
    by_op = {}
    by_priority = {}
    total_tokens = 0

    for e in manifest:
        ct = e.get("content_type", "unknown")
        by_type[ct] = by_type.get(ct, 0) + 1
        op = e.get("operator", "unknown")
        by_op[op] = by_op.get(op, 0) + 1
        pr = e.get("priority", "unknown")
        by_priority[pr] = by_priority.get(pr, 0) + 1
        total_tokens += e.get("tokens_approx", 0)

    print(f"\nBy content_type:")
    for k, v in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {k:20s}: {v}")

    print(f"\nBy operator:")
    for k, v in sorted(by_op.items(), key=lambda x: -x[1]):
        print(f"  {k:20s}: {v}")

    print(f"\nBy priority:")
    for k, v in sorted(by_priority.items(), key=lambda x: -x[1]):
        print(f"  {k:5s}: {v}")

    print(f"\nTotal approx tokens: {total_tokens:,}")


def main():
    parser = argparse.ArgumentParser(description="Generate manifest from aiter repo")
    parser.add_argument("--repo", default="/workspace/aiter", help="Path to aiter repo root")
    parser.add_argument("--taxonomy", default="/workspace/dataprocess/taxonomy.yaml")
    parser.add_argument("--output", default="/workspace/dataprocess/output/manifest.json")
    args = parser.parse_args()

    global TAXONOMY
    TAXONOMY = load_taxonomy(args.taxonomy)

    print(f"Scanning repo: {args.repo}")
    manifest = scan_repo(args.repo)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Manifest written to {args.output}")
    print_summary(manifest)


if __name__ == "__main__":
    main()
