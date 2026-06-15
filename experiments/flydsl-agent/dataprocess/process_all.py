#!/usr/bin/env python3
"""Self-contained data processing pipeline.

Processes the aiter repository and outputs 4 HuggingFace-standard JSON files:
  1. flydsl_cpt_data.json   — CPT corpus: [{"text": ..., "meta": {...}}, ...]
  2. flydsl_sft_data.json   — SFT chat data: [{"messages": [...], "source": ...}, ...]
  3. flydsl_rl_data.json    — RL task specs: [{"id": ..., "operator": ..., "params": ...}, ...]
  4. flydsl_val_data.json   — Validation split of SFT data (same format)

All outputs follow HuggingFace datasets conventions so they can be loaded with:
    from datasets import load_dataset
    ds = load_dataset("json", data_files="flydsl_sft_data.json")

No external dependencies beyond Python stdlib.
"""

import csv
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = "/home/danyzhan/aiter"
OUTPUT_DIR = "/home/danyzhan"

SYSTEM_PROMPT = (
    "You are a FlyDSL GPU kernel programming expert. "
    "Generate compilable, correct, high-performance FlyDSL kernel code "
    "based on user requirements."
)

# ---------------------------------------------------------------------------
# Taxonomy (inline, no YAML dependency)
# ---------------------------------------------------------------------------
OPERATOR_LIST = [
    "gemm", "flash_attn", "mla", "moe", "softmax", "rmsnorm", "layernorm",
    "rope", "topk", "paged_attn", "allreduce", "quant", "custom",
]

OP_PATTERNS = {
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

FEATURE_PATTERNS = {
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

PRIORITY_WEIGHT = {"P0": 3.0, "P1": 1.5, "P2": 0.5}
GRADE_WEIGHT = {"gold": 1.5, "silver": 1.0, "bronze": 0.7, "ungraded": 1.0, "reject": 0.0}
CONTENT_TYPE_WEIGHT = {
    "expert_skill": 2.5, "repo_guide": 2.0, "kernel_impl": 1.2, "framework_api": 1.0,
    "doc_guide": 1.0, "test": 0.8, "tuned_config": 0.6, "hardware_spec": 0.3, "build_script": 0.2,
}

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".tox", "build", "dist", "3rdparty"}
SKIP_EXT = {".pyc", ".pyo", ".so", ".o", ".a", ".co", ".bin", ".png", ".jpg", ".gif", ".ico", ".whl", ".tar", ".gz", ".zip"}
INCLUDE_EXT = {".py", ".md", ".rst", ".yaml", ".yml", ".csv", ".toml", ".cfg", ".cpp", ".hpp", ".h", ".hip", ".cu", ".cuh", ".s", ".asm", ".sh"}

# RL exploration shapes
GEMM_SHAPES = [
    {"M": 4096, "N": 4096, "K": 2048}, {"M": 2048, "N": 4096, "K": 1024},
    {"M": 8192, "N": 4096, "K": 2048}, {"M": 512, "N": 4096, "K": 2048},
    {"M": 128, "N": 4096, "K": 2048}, {"M": 256, "N": 8192, "K": 4096},
]
ATTN_SHAPES = [
    {"batch": 1, "heads": 32, "seq_len": 2048, "head_dim": 128},
    {"batch": 4, "heads": 32, "seq_len": 4096, "head_dim": 128},
    {"batch": 8, "heads": 64, "seq_len": 2048, "head_dim": 64},
]
MOE_SHAPES = [
    {"M": 256, "N": 2048, "K": 1024, "experts": 8, "topk": 2},
    {"M": 512, "N": 4096, "K": 2048, "experts": 16, "topk": 4},
    {"M": 1024, "N": 4096, "K": 2048, "experts": 64, "topk": 8},
]
NORM_SHAPES = [
    {"batch": 1, "hidden_dim": 4096}, {"batch": 8, "hidden_dim": 4096},
    {"batch": 32, "hidden_dim": 8192},
]


# ---------------------------------------------------------------------------
# Step 1: Scan repo → manifest
# ---------------------------------------------------------------------------
def infer_operator(fp: str, content: str) -> str:
    fp_low, head = fp.lower(), content[:2000].lower()
    for op, pat in OP_PATTERNS.items():
        if re.search(pat, fp_low) or re.search(pat, head):
            return op
    return "custom"

def infer_hardware(content: str) -> list:
    hw = list(set(re.findall(r"gfx\d+", content)))
    return hw if hw else ["generic"]

def infer_features(content: str) -> list:
    return [f for f, p in FEATURE_PATTERNS.items() if re.search(p, content, re.I)]

def infer_complexity(content: str, features: list) -> str:
    lines = content.count("\n") + 1
    nf = len(features)
    if lines > 1500 or nf >= 3: return "expert"
    if lines > 800 or nf >= 2: return "advanced"
    if lines > 300: return "intermediate"
    return "beginner"

def infer_type_priority(fp: str) -> tuple:
    f = fp.lower()
    if ".claude/skills" in f: return "expert_skill", "P0"
    if f.endswith("claude.md") or (f.endswith("readme.md") and "/docs" not in f): return "repo_guide", "P0"
    if "/configs/" in f and f.endswith(".csv"): return "tuned_config", "P1"
    if "/ops/triton/" in f or "/ops/" in f:
        return ("test", "P1") if "test" in f else ("kernel_impl", "P0")
    if "/csrc/" in f or "/hsa/" in f: return "kernel_impl", "P1"
    if "/op_tests/" in f or "/tests/" in f or "test_" in os.path.basename(f): return "test", "P1"
    if f.endswith(".md") or f.endswith(".rst"): return "doc_guide", "P0"
    if "setup.py" in f or "pyproject.toml" in f or "/scripts/" in f: return "build_script", "P2"
    if "/aiter/" in f and f.endswith(".py"): return "framework_api", "P0"
    return "framework_api", "P2"

def should_skip(p: Path) -> bool:
    for part in p.parts:
        if part in SKIP_DIRS or part.endswith(".egg-info"):
            return True
    return p.suffix in SKIP_EXT

def scan_repo(root: str) -> list:
    manifest = []
    for fp in sorted(Path(root).rglob("*")):
        if not fp.is_file() or should_skip(fp) or fp.suffix not in INCLUDE_EXT:
            continue
        rel = os.path.relpath(str(fp), root)
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        op = infer_operator(rel, content)
        hw = infer_hardware(content)
        feats = infer_features(content)
        ctype, prio = infer_type_priority(rel)
        manifest.append({
            "path": rel, "operator": op, "hardware": hw, "features": feats,
            "complexity": infer_complexity(content, feats),
            "content_type": ctype, "priority": prio, "quality_grade": "ungraded",
            "lines": content.count("\n") + 1, "tokens_approx": len(content) // 4,
        })
    return manifest

def compute_weight(e: dict) -> float:
    return round(
        PRIORITY_WEIGHT.get(e.get("priority", "P2"), 0.5)
        * GRADE_WEIGHT.get(e.get("quality_grade", "ungraded"), 1.0)
        * CONTENT_TYPE_WEIGHT.get(e.get("content_type", "framework_api"), 1.0), 3)


# ---------------------------------------------------------------------------
# Step 2: CPT data  (HF format: [{"text": ..., "meta": {...}}])
# ---------------------------------------------------------------------------
def build_cpt_dataset(manifest: list, root: str) -> list:
    dataset = []
    for e in manifest:
        if e.get("quality_grade") == "reject" or e.get("lines", 0) < 5:
            continue
        fp = os.path.join(root, e["path"])
        if not os.path.exists(fp):
            continue
        try:
            content = open(fp, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        tok = e.get("tokens_approx", len(content) // 4)
        if tok > 50000:
            continue
        w = compute_weight(e)
        if w <= 0:
            continue

        meta_hdr = (
            f"[file: {e['path']}]\n[type: {e.get('content_type','unknown')}]\n"
            f"[operator: {e.get('operator','unknown')}]\n"
            f"[hardware: {', '.join(e.get('hardware',['generic']))}]\n"
            f"[complexity: {e.get('complexity','unknown')}]"
        )
        text = f"<|doc_start|>\n{meta_hdr}\n\n{content}\n<|doc_end|>"

        dataset.append({
            "text": text,
            "meta": {
                "source_path": e["path"],
                "content_type": e.get("content_type"),
                "operator": e.get("operator"),
                "hardware": e.get("hardware", ["generic"]),
                "complexity": e.get("complexity"),
                "priority": e.get("priority"),
                "quality_grade": e.get("quality_grade"),
                "weight": w,
                "tokens_approx": tok,
            }
        })
    return dataset


# ---------------------------------------------------------------------------
# Step 3: SFT data  (HF format: [{"messages": [...], "source": ..., "metadata": {...}}])
# ---------------------------------------------------------------------------
def build_sft_dataset(manifest: list, root: str) -> list:
    pairs = []

    # Source A: kernel reverse-annotation
    for e in manifest:
        if e.get("content_type") != "kernel_impl" or e.get("priority") not in ("P0", "P1"):
            continue
        fp = os.path.join(root, e["path"])
        if not os.path.exists(fp):
            continue
        try:
            code = open(fp, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        if len(code.strip()) < 100:
            continue

        op = e.get("operator", "custom")
        hw = ", ".join(e.get("hardware", ["generic"]))
        feats = e.get("features", [])
        cx = e.get("complexity", "intermediate")
        feat_s = f"\n- Features: {', '.join(feats)}" if feats else ""

        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Implement a {op} kernel for AMD GPU ({hw}) using the aiter/Triton framework.\n"
                    f"Requirements:\n- Operator: {op}\n- Target hardware: {hw}\n- Complexity: {cx}"
                    f"{feat_s}\nProvide complete, compilable code with all necessary imports."
                )},
                {"role": "assistant", "content": f"```python\n{code}\n```"},
            ],
            "source": "kernel_reverse_annotation",
            "metadata": {"source_path": e["path"], "operator": op, "complexity": cx, "style": "precise"},
        })

        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"I need a high-performance {op} implementation for AMD {hw} GPUs. "
                    f"The kernel should be written using Triton/aiter patterns"
                    f"{' with ' + ', '.join(feats) if feats else ''}. "
                    f"Please write the complete implementation."
                )},
                {"role": "assistant", "content": f"```python\n{code}\n```"},
            ],
            "source": "kernel_reverse_annotation",
            "metadata": {"source_path": e["path"], "operator": op, "complexity": cx, "style": "natural"},
        })

    # Source B: tests with parametrize
    for e in manifest:
        if e.get("content_type") != "test":
            continue
        fp = os.path.join(root, e["path"])
        if not os.path.exists(fp):
            continue
        try:
            code = open(fp, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        op = e.get("operator", "custom")
        if op == "custom":
            continue
        if not re.findall(r'@pytest\.mark\.parametrize\([^)]+\)', code, re.DOTALL):
            continue
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Write a comprehensive test for the {op} operator on AMD GPU.\n"
                    f"The test should cover multiple parameter combinations and verify correctness "
                    f"against a PyTorch reference implementation.\n"
                    f"Include parameterized test cases for different shapes, dtypes, and configurations."
                )},
                {"role": "assistant", "content": f"```python\n{code}\n```"},
            ],
            "source": "test_parameterization",
            "metadata": {"source_path": e["path"], "operator": op},
        })

    # Source C: documentation QA
    for e in manifest:
        if e.get("content_type") not in ("doc_guide", "expert_skill", "repo_guide"):
            continue
        fp = os.path.join(root, e["path"])
        if not os.path.exists(fp):
            continue
        try:
            content = open(fp, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        if len(content.strip()) < 200:
            continue
        for section in re.split(r'\n##?\s+', content):
            if len(section.strip()) < 100:
                continue
            lines = section.strip().split("\n")
            title = lines[0].strip().rstrip("#").strip() if lines else "Overview"
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else section.strip()
            if len(body) < 100:
                continue
            pairs.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Explain the following concept from the aiter/AITER documentation: {title}\n"
                        f"Provide a detailed explanation with code examples where applicable."
                    )},
                    {"role": "assistant", "content": body},
                ],
                "source": "documentation_qa",
                "metadata": {"source_path": e["path"], "section_title": title[:100], "content_type": e["content_type"]},
            })

    # Source D: tuned configs
    configs_dir = os.path.join(root, "aiter", "configs")
    if os.path.isdir(configs_dir):
        for csv_file in sorted(Path(configs_dir).glob("*.csv")):
            try:
                rows = list(csv.DictReader(open(csv_file, newline="", encoding="utf-8")))
            except Exception:
                continue
            if not rows:
                continue
            op_name = csv_file.stem.replace("_tuned_", " ").replace("_untuned_", " untuned ").replace("_", " ")
            cols = list(rows[0].keys())
            sample = "\n".join("  " + ", ".join(f"{k}={v}" for k, v in r.items()) for r in rows[:5])
            pairs.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"What are the tuned configurations for {op_name} on AMD GPU?\n"
                        f"Show the optimal parameter settings from the aiter tuning database."
                    )},
                    {"role": "assistant", "content": (
                        f"Here are the tuned configurations for {op_name}:\n\n"
                        f"Columns: {', '.join(cols)}\nTotal configurations: {len(rows)}\n\n"
                        f"Sample entries:\n{sample}\n\n"
                        f"These configurations were auto-tuned for optimal performance on AMD Instinct GPUs."
                    )},
                ],
                "source": "tuned_config",
                "metadata": {"source_path": str(csv_file.relative_to(root)), "num_configs": len(rows)},
            })

    # Source E: git history
    try:
        r = subprocess.run(
            ["git", "log", "--max-count=200", "--oneline", "--diff-filter=M", "--", "aiter/ops/", "op_tests/"],
            cwd=root, capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            kw = ["fix", "bug", "optim", "perf", "improve", "refactor", "correct"]
            for line in r.stdout.strip().split("\n")[:100]:
                if not line.strip():
                    continue
                parts = line.split(" ", 1)
                if len(parts) < 2:
                    continue
                sha, msg = parts
                if not any(k in msg.lower() for k in kw):
                    continue
                try:
                    dr = subprocess.run(["git", "diff", f"{sha}~1", sha, "--stat"],
                                        cwd=root, capture_output=True, text=True, timeout=10)
                except Exception:
                    continue
                if dr.returncode != 0:
                    continue
                pairs.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"The following commit was made to the aiter codebase:\n"
                            f"Commit: {sha}\nMessage: {msg}\n\n"
                            f"Explain what this change does and why it was necessary."
                        )},
                        {"role": "assistant", "content": (
                            f"This commit ({sha}) addresses: {msg}\n\n"
                            f"Changed files:\n{dr.stdout[:2000]}\n\n"
                            f"This change improves the aiter kernel implementation by addressing "
                            f"the issue described in the commit message."
                        )},
                    ],
                    "source": "git_history",
                    "metadata": {"commit_sha": sha, "commit_msg": msg},
                })
    except Exception:
        pass

    return pairs


# ---------------------------------------------------------------------------
# Step 4: RL specs  (HF format: [{"id": ..., "operator": ..., ...}])
# ---------------------------------------------------------------------------
def build_rl_dataset(manifest: list, root: str) -> list:
    specs = []

    # From manifest
    for e in manifest:
        if e.get("content_type") != "kernel_impl":
            continue
        if e.get("quality_grade") not in ("gold", "silver", "ungraded"):
            continue
        op = e.get("operator", "custom")
        if op == "custom":
            continue
        for hw in e.get("hardware", ["generic"]):
            if hw == "generic":
                hw = "gfx942"
            specs.append({
                "id": f"manifest_{op}_{hw}_{len(specs)}",
                "operator": op, "hardware": hw, "source": "manifest",
                "source_path": e["path"], "features": e.get("features", []),
                "baseline_grade": e.get("quality_grade", "ungraded"),
            })

    # From tuned configs
    configs_dir = os.path.join(root, "aiter", "configs")
    if os.path.isdir(configs_dir):
        for csv_file in sorted(Path(configs_dir).glob("*tuned*.csv")):
            if "untuned" in csv_file.name:
                continue
            op_name = csv_file.stem
            operator = "moe" if ("moe" in op_name or "fmoe" in op_name) else "gemm"
            dtypes = []
            if "a8w8" in op_name: dtypes = ["int8"]
            elif "a4w4" in op_name: dtypes = ["int4"]
            elif "bf16" in op_name: dtypes = ["bf16"]
            elif "fp8" in op_name: dtypes = ["fp8"]
            try:
                rows = list(csv.DictReader(open(csv_file, newline="", encoding="utf-8")))
            except Exception:
                continue
            for i, row in enumerate(rows[:20]):
                params = {}
                for k, v in row.items():
                    try: params[k] = int(v)
                    except (ValueError, TypeError):
                        try: params[k] = float(v)
                        except (ValueError, TypeError): params[k] = v
                specs.append({
                    "id": f"tuned_{op_name}_{i}", "operator": operator, "hardware": "gfx942",
                    "params": params, "dtypes": dtypes, "source": "tuned_config",
                    "source_file": csv_file.name,
                })

    # Exploration specs
    sid = 0
    for hw in ["gfx942", "gfx950"]:
        for dt in ["fp8", "bf16"]:
            for s in GEMM_SHAPES:
                specs.append({"id": f"explore_gemm_{sid}", "operator": "gemm", "hardware": hw,
                              "params": {**s, "in_dtype": dt, "out_dtype": "bf16"}, "source": "exploration"})
                sid += 1
        for s in ATTN_SHAPES:
            specs.append({"id": f"explore_attn_{sid}", "operator": "flash_attn", "hardware": hw,
                          "params": s, "source": "exploration"})
            sid += 1
        for s in MOE_SHAPES:
            for dt in ["fp8", "bf16"]:
                specs.append({"id": f"explore_moe_{sid}", "operator": "moe", "hardware": hw,
                              "params": {**s, "in_dtype": dt}, "source": "exploration"})
                sid += 1
        for s in NORM_SHAPES:
            for op in ["rmsnorm", "softmax", "layernorm"]:
                specs.append({"id": f"explore_{op}_{sid}", "operator": op, "hardware": hw,
                              "params": s, "source": "exploration"})
                sid += 1
    return specs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    random.seed(42)
    repo = REPO_ROOT
    out = OUTPUT_DIR

    print(f"Scanning repo: {repo}")
    manifest = scan_repo(repo)
    print(f"  Found {len(manifest)} files")

    # --- CPT ---
    print("\nBuilding CPT dataset...")
    cpt = build_cpt_dataset(manifest, repo)
    cpt_path = os.path.join(out, "flydsl_cpt_data.json")
    with open(cpt_path, "w", encoding="utf-8") as f:
        json.dump(cpt, f, ensure_ascii=False, indent=1)
    total_tok = sum(d["meta"]["tokens_approx"] for d in cpt)
    print(f"  CPT: {len(cpt)} documents, ~{total_tok:,} tokens → {cpt_path}")

    # --- SFT ---
    print("\nBuilding SFT dataset...")
    sft_all = build_sft_dataset(manifest, repo)
    random.shuffle(sft_all)

    # Stratified train/val split (10% val)
    groups = defaultdict(list)
    for r in sft_all:
        groups[r.get("source", "unknown")].append(r)
    train, val = [], []
    for gk, gr in groups.items():
        random.shuffle(gr)
        nv = max(1, int(len(gr) * 0.1))
        val.extend(gr[:nv])
        train.extend(gr[nv:])
    random.shuffle(train)
    random.shuffle(val)

    sft_path = os.path.join(out, "flydsl_sft_data.json")
    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=1)
    print(f"  SFT train: {len(train)} pairs → {sft_path}")

    val_path = os.path.join(out, "flydsl_val_data.json")
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=1)
    print(f"  SFT val:   {len(val)} pairs → {val_path}")

    by_src = Counter(r.get("source") for r in sft_all)
    print(f"  SFT total: {len(sft_all)} pairs")
    for k, v in by_src.most_common():
        print(f"    {k:30s}: {v}")

    # --- RL ---
    print("\nBuilding RL dataset...")
    rl = build_rl_dataset(manifest, repo)
    rl_path = os.path.join(out, "flydsl_rl_data.json")
    with open(rl_path, "w", encoding="utf-8") as f:
        json.dump(rl, f, ensure_ascii=False, indent=1)
    by_op = Counter(s["operator"] for s in rl)
    print(f"  RL: {len(rl)} specs → {rl_path}")
    for k, v in by_op.most_common():
        print(f"    {k:15s}: {v}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("OUTPUT FILES (HuggingFace standard JSON)")
    print(f"{'='*60}")
    for p in [cpt_path, sft_path, rl_path, val_path]:
        sz = os.path.getsize(p) / (1024 * 1024)
        print(f"  {os.path.basename(p):30s}  {sz:8.2f} MB")
    print(f"\nLoad with:  from datasets import load_dataset")
    print(f'  ds = load_dataset("json", data_files="flydsl_sft_data.json")')
    print("\nDone!")


if __name__ == "__main__":
    main()
