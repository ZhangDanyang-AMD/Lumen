#!/usr/bin/env python3
"""Enhanced data processing pipeline v2 — dual-repo (FlyDSL + aiter).

Covers ALL data sources from plan §4:
  - FlyDSL: .claude/skills, CLAUDE.md, docs, kernels, python/flydsl, tests
  - aiter: ops/triton, csrc, hsa, op_tests, docs, configs
  - SFT sources A-I (kernel reverse-annotation, test parameterization,
    difficulty gradient, documentation QA, cross-backend translation,
    git history, data augmentation, performance improvement pairs,
    document direct extraction from FlyDSL docs/skills)
  - AI annotation (Layer 2) via current model
  - Quality grading (compilation check where possible)

Outputs HF-standard JSONL:
  1. CPT  — {text, meta}
  2. SFT  — {messages, source, metadata}
  3. RL   — {id, operator, hardware, params, ...}
  4. Validate — SFT validation split

Usage:
  python3 process_all_v2.py [--skip-ai-annotate] [--skip-benchmark]
"""

import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FLYDSL_ROOT = os.environ.get("FLYDSL_ROOT", "/home/danyzhan/FlyDSL")
AITER_ROOT = os.environ.get("AITER_ROOT", "/home/danyzhan/aiter")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/home/danyzhan")

SYSTEM_PROMPT = (
    "You are a FlyDSL GPU kernel programming expert. "
    "You write compilable, correct, high-performance GPU kernel code "
    "using the FlyDSL DSL (with @flyc.kernel/@flyc.jit) and Triton/aiter patterns "
    "for AMD Instinct GPUs (MI300X gfx942, MI350 gfx950, MI450 gfx1250)."
)

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------
OPERATOR_LIST = [
    "gemm", "flash_attn", "mla", "moe", "softmax", "rmsnorm", "layernorm",
    "rope", "topk", "paged_attn", "allreduce", "quant", "custom",
]

OP_PATTERNS = {
    "gemm": r"gemm|matmul|batched_gemm|hgemm|preshuffle_gemm|splitk.*gemm|gemm.*splitk|wmma_gemm|rdna.*gemm|fp8_gemm",
    "flash_attn": r"flash_attn|fmha|flash_attention",
    "mla": r"\bmla\b|multi_latent|mla_decode|mla_fwd",
    "moe": r"\bmoe\b|fused_moe|fmoe|mixture.of.expert|moe_gemm|moe_sorting|moe_reduce|moe_blockscale|moe_common",
    "softmax": r"softmax",
    "rmsnorm": r"rmsnorm|rms_norm",
    "layernorm": r"layernorm|layer_norm",
    "rope": r"\brope\b|rotary|rope_cache",
    "topk": r"\btopk\b|top_k|topk_gating",
    "paged_attn": r"paged_attn|paged_attention|page_attention|pa_decode|pa_metadata",
    "allreduce": r"allreduce|all_reduce|reduce_scatter|all_gather|custom_all_reduce|dispatch_combine",
    "quant": r"\bquant\b|quantiz|dequant|fp8_cast|blockscale|silu_and_mul_fq",
}

FEATURE_PATTERNS = {
    "swizzle_xor16": r"swizzle_xor|xor16|swizzle.*16",
    "pipeline": r"lds_stage|pipeline|num_stages|software.pipelining|pipeline_utils",
    "async_copy": r"async_copy|cp\.async|async_load|use_async_copy",
    "double_buffer": r"double_buf|lds_stage\s*=\s*2|ping.pong",
    "preshuffle": r"preshuffle|pre_shuffle|bpreshuffle",
    "blockscale": r"blockscale|block_scale|block.scaled",
    "split_k": r"split_k|splitk",
    "epilogue_fusion": r"epilogue|bias_add|activation_fuse|cshuffle_epilog|mfma_epilog",
    "multi_wave": r"waves_per_eu|num_waves|multi_wave|8wave|4wave",
    "sage_attention": r"sage.attention|sage_attn",
    "mxfp4": r"mxfp4|mx_fp4|mxscale",
    "tdm": r"\btdm\b|tdm_ops|tdm_copy",
    "wmma": r"\bwmma\b|wmma_gemm",
    "shared_allocator": r"SharedAllocator|smem_allocator|SmemAllocator",
    "layout_algebra": r"layout_algebra|crd2idx|idx2crd|make_layout|complement",
}

PRIORITY_WEIGHT = {"P0": 3.0, "P1": 1.5, "P2": 0.5}
GRADE_WEIGHT = {"gold": 1.5, "silver": 1.0, "bronze": 0.7, "ungraded": 1.0, "reject": 0.0}
CONTENT_TYPE_WEIGHT = {
    "expert_skill": 2.5, "repo_guide": 2.0, "kernel_impl": 1.2, "framework_api": 1.0,
    "doc_guide": 1.5, "test": 0.8, "tuned_config": 0.6, "hardware_spec": 0.3, "build_script": 0.2,
}

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".tox", "build", "dist",
             "3rdparty", "thirdparty", "build-fly", ".eggs", "egg-info"}
SKIP_EXT = {".pyc", ".pyo", ".so", ".o", ".a", ".co", ".bin", ".png", ".jpg",
            ".gif", ".ico", ".whl", ".tar", ".gz", ".zip", ".pdf"}
INCLUDE_EXT = {".py", ".md", ".rst", ".yaml", ".yml", ".csv", ".toml", ".cfg",
               ".cpp", ".hpp", ".h", ".hip", ".cu", ".cuh", ".s", ".asm", ".sh",
               ".mlir", ".txt", ".ini"}

GEMM_SHAPES = [
    {"M": 4096, "N": 4096, "K": 2048}, {"M": 2048, "N": 4096, "K": 1024},
    {"M": 8192, "N": 4096, "K": 2048}, {"M": 512, "N": 4096, "K": 2048},
    {"M": 128, "N": 4096, "K": 2048}, {"M": 256, "N": 8192, "K": 4096},
    {"M": 16, "N": 5120, "K": 8192}, {"M": 1, "N": 4096, "K": 4096},
]
ATTN_SHAPES = [
    {"batch": 1, "heads": 32, "seq_len": 2048, "head_dim": 128},
    {"batch": 4, "heads": 32, "seq_len": 4096, "head_dim": 128},
    {"batch": 8, "heads": 64, "seq_len": 2048, "head_dim": 64},
    {"batch": 1, "heads": 8, "seq_len": 16384, "head_dim": 128},
]
MOE_SHAPES = [
    {"M": 256, "N": 2048, "K": 1024, "experts": 8, "topk": 2},
    {"M": 512, "N": 4096, "K": 2048, "experts": 16, "topk": 4},
    {"M": 1024, "N": 4096, "K": 2048, "experts": 64, "topk": 8},
]
NORM_SHAPES = [
    {"batch": 1, "hidden_dim": 4096}, {"batch": 8, "hidden_dim": 4096},
    {"batch": 32, "hidden_dim": 8192}, {"batch": 64, "hidden_dim": 4096},
]


# ===================================================================
# Step 1: Dual-repo manifest scan
# ===================================================================
def infer_operator(fp: str, content: str) -> str:
    fp_low, head = fp.lower(), content[:3000].lower()
    for op, pat in OP_PATTERNS.items():
        if re.search(pat, fp_low) or re.search(pat, head):
            return op
    return "custom"


def infer_hardware(content: str) -> list:
    hw = sorted(set(re.findall(r"gfx\d+", content)))
    return hw if hw else ["generic"]


def infer_features(content: str) -> list:
    return sorted([f for f, p in FEATURE_PATTERNS.items() if re.search(p, content, re.I)])


def infer_complexity(content: str, features: list) -> str:
    lines = content.count("\n") + 1
    nf = len(features)
    if lines > 1500 or nf >= 3:
        return "expert"
    if lines > 800 or nf >= 2:
        return "advanced"
    if lines > 300:
        return "intermediate"
    return "beginner"


def infer_type_priority_flydsl(fp: str) -> tuple:
    f = fp.lower()
    if ".claude/skills" in f and f.endswith(".md"):
        return "expert_skill", "P0"
    if f.endswith("claude.md"):
        return "repo_guide", "P0"
    if f.endswith("readme.md"):
        return "repo_guide", "P0"
    if ("kernels/" in f or f.startswith("kernels/")) and f.endswith(".py") and "test_" not in os.path.basename(f):
        return "kernel_impl", "P0"
    if "/expr/" in f or "/compiler/" in f:
        return "framework_api", "P0"
    if "/runtime/" in f or "/utils/" in f or "autotune" in f:
        return "framework_api", "P0"
    if f.endswith(".md") or f.endswith(".rst"):
        return "doc_guide", "P0"
    if "/tests/kernels/" in f:
        return "test", "P1"
    if "/tests/" in f:
        return "test", "P1"
    if "/scripts/" in f or f.endswith(".sh"):
        return "build_script", "P2"
    if f.endswith(".mlir"):
        return "test", "P1"
    return "framework_api", "P2"


def infer_type_priority_aiter(fp: str) -> tuple:
    f = fp.lower()
    if ".claude/skills" in f:
        return "expert_skill", "P0"
    if f.endswith("claude.md") or (f.endswith("readme.md") and "/docs" not in f):
        return "repo_guide", "P0"
    if "/configs/" in f and f.endswith(".csv"):
        return "tuned_config", "P1"
    if "/ops/triton/" in f or "/ops/" in f:
        return ("test", "P1") if "test" in f else ("kernel_impl", "P0")
    if "/csrc/" in f or "/hsa/" in f:
        return "kernel_impl", "P1"
    if "/op_tests/" in f or "/tests/" in f or "test_" in os.path.basename(f):
        return "test", "P1"
    if f.endswith(".md") or f.endswith(".rst"):
        return "doc_guide", "P0"
    if "setup.py" in f or "pyproject.toml" in f or "/scripts/" in f:
        return "build_script", "P2"
    if "/aiter/" in f and f.endswith(".py"):
        return "framework_api", "P0"
    return "framework_api", "P2"


def should_skip(p: Path, root: str) -> bool:
    for part in p.relative_to(root).parts:
        if part in SKIP_DIRS or part.endswith(".egg-info"):
            return True
    return p.suffix in SKIP_EXT


def scan_single_repo(root: str, repo_name: str, type_fn) -> list:
    manifest = []
    root_path = Path(root)
    for fp in sorted(root_path.rglob("*")):
        if not fp.is_file() or should_skip(fp, root) or fp.suffix not in INCLUDE_EXT:
            continue
        rel = os.path.relpath(str(fp), root)
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if len(content.strip()) < 10:
            continue
        op = infer_operator(rel, content)
        hw = infer_hardware(content)
        feats = infer_features(content)
        ctype, prio = type_fn(rel)
        manifest.append({
            "repo": repo_name,
            "path": rel,
            "full_path": str(fp),
            "operator": op,
            "hardware": hw,
            "features": feats,
            "complexity": infer_complexity(content, feats),
            "content_type": ctype,
            "priority": prio,
            "quality_grade": "ungraded",
            "lines": content.count("\n") + 1,
            "tokens_approx": len(content) // 4,
        })
    return manifest


def compute_weight(e: dict) -> float:
    return round(
        PRIORITY_WEIGHT.get(e.get("priority", "P2"), 0.5)
        * GRADE_WEIGHT.get(e.get("quality_grade", "ungraded"), 1.0)
        * CONTENT_TYPE_WEIGHT.get(e.get("content_type", "framework_api"), 1.0), 3)


# ===================================================================
# Step 2: CPT dataset
# ===================================================================
def build_cpt_dataset(manifest: list) -> list:
    dataset = []
    for e in manifest:
        if e.get("quality_grade") == "reject" or e.get("lines", 0) < 5:
            continue
        fp = e["full_path"]
        if not os.path.exists(fp):
            continue
        try:
            content = open(fp, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        tok = e.get("tokens_approx", len(content) // 4)
        if tok > 80000:
            continue
        w = compute_weight(e)
        if w <= 0:
            continue

        repo_tag = f"[repo: {e['repo']}]\n" if e.get("repo") else ""
        meta_hdr = (
            f"[file: {e['path']}]\n{repo_tag}"
            f"[type: {e.get('content_type', 'unknown')}]\n"
            f"[operator: {e.get('operator', 'unknown')}]\n"
            f"[hardware: {', '.join(e.get('hardware', ['generic']))}]\n"
            f"[complexity: {e.get('complexity', 'unknown')}]\n"
            f"[features: {', '.join(e.get('features', [])) or 'none'}]"
        )
        text = f"<|doc_start|>\n{meta_hdr}\n\n{content}\n<|doc_end|>"

        dataset.append({
            "text": text,
            "meta": {
                "source_repo": e.get("repo"),
                "source_path": e["path"],
                "content_type": e.get("content_type"),
                "operator": e.get("operator"),
                "hardware": e.get("hardware", ["generic"]),
                "complexity": e.get("complexity"),
                "priority": e.get("priority"),
                "quality_grade": e.get("quality_grade"),
                "features": e.get("features", []),
                "weight": w,
                "tokens_approx": tok,
            }
        })
    return dataset


# ===================================================================
# Step 3: SFT dataset — ALL sources A-I
# ===================================================================
def _read_file(path: str) -> str:
    try:
        return open(path, encoding="utf-8", errors="replace").read()
    except Exception:
        return ""


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def source_a_kernel_reverse_annotation(manifest: list) -> list:
    """Source A: Reverse-annotate kernels into instruction/output pairs."""
    pairs = []
    for e in manifest:
        if e.get("content_type") != "kernel_impl":
            continue
        if e.get("priority") not in ("P0", "P1"):
            continue
        code = _read_file(e["full_path"])
        if len(code.strip()) < 100:
            continue

        op = e.get("operator", "custom")
        hw = ", ".join(e.get("hardware", ["generic"]))
        feats = e.get("features", [])
        cx = e.get("complexity", "intermediate")
        repo = e.get("repo", "aiter")
        feat_s = f"\n- Features: {', '.join(feats)}" if feats else ""

        framework = "FlyDSL (@flyc.kernel/@flyc.jit)" if repo == "FlyDSL" else "Triton/aiter"

        # Style 1: Precise technical
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Implement a {op} kernel for AMD GPU ({hw}) using {framework}.\n"
                    f"Requirements:\n- Operator: {op}\n- Target hardware: {hw}\n- Complexity: {cx}"
                    f"{feat_s}\nProvide complete, compilable code with all necessary imports."
                )},
                {"role": "assistant", "content": f"```python\n{code}\n```"},
            ],
            "source": "kernel_reverse_annotation",
            "metadata": {"source_path": e["path"], "repo": repo, "operator": op,
                         "complexity": cx, "style": "precise"},
        })

        # Style 2: Natural request
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"I need a high-performance {op} implementation for AMD {hw} GPUs. "
                    f"The kernel should be written using {framework} patterns"
                    f"{' with ' + ', '.join(feats) if feats else ''}. "
                    f"Please write the complete implementation."
                )},
                {"role": "assistant", "content": f"```python\n{code}\n```"},
            ],
            "source": "kernel_reverse_annotation",
            "metadata": {"source_path": e["path"], "repo": repo, "operator": op,
                         "complexity": cx, "style": "natural"},
        })

        # Style 3: Optimization-focused (for advanced/expert kernels)
        if cx in ("advanced", "expert") and feats:
            pairs.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Write an optimized {op} GPU kernel targeting {hw} with the following "
                        f"optimization techniques: {', '.join(feats)}.\n"
                        f"The kernel should achieve high roofline efficiency. "
                        f"Use {framework} and include all necessary imports and configuration."
                    )},
                    {"role": "assistant", "content": f"```python\n{code}\n```"},
                ],
                "source": "kernel_reverse_annotation",
                "metadata": {"source_path": e["path"], "repo": repo, "operator": op,
                             "complexity": cx, "style": "optimization"},
            })

    return pairs


def source_b_test_parameterization(manifest: list) -> list:
    """Source B: Extract test files with pytest parametrize as SFT pairs."""
    pairs = []
    for e in manifest:
        if e.get("content_type") != "test":
            continue
        code = _read_file(e["full_path"])
        op = e.get("operator", "custom")
        if op == "custom":
            continue
        if not re.findall(r'@pytest\.mark\.parametrize\([^)]+\)', code, re.DOTALL):
            continue
        repo = e.get("repo", "aiter")
        framework = "FlyDSL" if repo == "FlyDSL" else "aiter/Triton"
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Write a comprehensive test for the {op} operator using {framework}.\n"
                    f"The test should cover multiple parameter combinations and verify correctness "
                    f"against a PyTorch reference implementation.\n"
                    f"Include parameterized test cases for different shapes, dtypes, and configurations."
                )},
                {"role": "assistant", "content": f"```python\n{code}\n```"},
            ],
            "source": "test_parameterization",
            "metadata": {"source_path": e["path"], "repo": repo, "operator": op},
        })
    return pairs


def source_c_documentation_qa(manifest: list) -> list:
    """Source C (originally D in old script): Documentation QA from both repos."""
    pairs = []
    for e in manifest:
        if e.get("content_type") not in ("doc_guide", "expert_skill", "repo_guide"):
            continue
        content = _read_file(e["full_path"])
        if len(content.strip()) < 200:
            continue
        repo = e.get("repo", "aiter")
        source_name = "FlyDSL" if repo == "FlyDSL" else "aiter/AITER"

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
                        f"Explain the following concept from {source_name} documentation: {title}\n"
                        f"Provide a detailed explanation with code examples where applicable."
                    )},
                    {"role": "assistant", "content": body},
                ],
                "source": "documentation_qa",
                "metadata": {"source_path": e["path"], "repo": repo,
                             "section_title": title[:100], "content_type": e["content_type"]},
            })
    return pairs


def source_d_tuned_configs() -> list:
    """Source D: Tuned config QA from aiter configs."""
    pairs = []
    configs_dir = os.path.join(AITER_ROOT, "aiter", "configs")
    if not os.path.isdir(configs_dir):
        return pairs
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
            "metadata": {"source_path": str(csv_file.relative_to(AITER_ROOT)), "num_configs": len(rows)},
        })
    return pairs


def source_e_cross_backend_translation(manifest: list) -> list:
    """Source E: Cross-backend translation — aiter Triton ↔ FlyDSL pairs for same operator."""
    pairs = []
    by_op_repo = defaultdict(lambda: defaultdict(list))
    for e in manifest:
        if e.get("content_type") != "kernel_impl":
            continue
        by_op_repo[e["operator"]][e.get("repo", "aiter")].append(e)

    for op, repos in by_op_repo.items():
        if op == "custom":
            continue
        flydsl_kernels = repos.get("FlyDSL", [])
        aiter_kernels = repos.get("aiter", [])
        if not flydsl_kernels or not aiter_kernels:
            continue
        for fk in flydsl_kernels[:3]:
            for ak in aiter_kernels[:3]:
                fcode = _read_file(fk["full_path"])
                acode = _read_file(ak["full_path"])
                if len(fcode.strip()) < 100 or len(acode.strip()) < 100:
                    continue
                # Triton → FlyDSL direction
                pairs.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"Convert the following Triton/aiter {op} kernel to FlyDSL "
                            f"(@flyc.kernel/@flyc.jit) style:\n\n```python\n{acode[:3000]}\n```\n\n"
                            f"Rewrite using FlyDSL layout algebra, explicit MMA atoms, and "
                            f"copy atoms. Keep the same algorithmic approach."
                        )},
                        {"role": "assistant", "content": f"```python\n{fcode}\n```"},
                    ],
                    "source": "cross_backend_translation",
                    "metadata": {"operator": op, "from_path": ak["path"],
                                 "to_path": fk["path"], "direction": "triton_to_flydsl"},
                })
                # FlyDSL → Triton direction
                pairs.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"Convert the following FlyDSL {op} kernel to Triton/aiter style:\n\n"
                            f"```python\n{fcode[:3000]}\n```\n\n"
                            f"Rewrite using standard Triton APIs (tl.load, tl.store, tl.dot, etc.)."
                        )},
                        {"role": "assistant", "content": f"```python\n{acode}\n```"},
                    ],
                    "source": "cross_backend_translation",
                    "metadata": {"operator": op, "from_path": fk["path"],
                                 "to_path": ak["path"], "direction": "flydsl_to_triton"},
                })
    return pairs


def source_f_git_history(repo_root: str, repo_name: str, paths: list) -> list:
    """Source F: Git history — extract fix/optimize commits as SFT pairs."""
    pairs = []
    try:
        cmd = ["git", "log", "--max-count=300", "--oneline", "--diff-filter=M", "--"] + paths
        r = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return pairs
    except Exception:
        return pairs

    kw = ["fix", "bug", "optim", "perf", "improve", "refactor", "correct", "enhance", "speedup"]
    for line in r.stdout.strip().split("\n")[:150]:
        if not line.strip():
            continue
        parts = line.split(" ", 1)
        if len(parts) < 2:
            continue
        sha, msg = parts
        if not any(k in msg.lower() for k in kw):
            continue
        try:
            dr = subprocess.run(
                ["git", "diff", f"{sha}~1", sha, "--stat"],
                cwd=repo_root, capture_output=True, text=True, timeout=10)
            dp = subprocess.run(
                ["git", "diff", f"{sha}~1", sha, "--", "*.py"],
                cwd=repo_root, capture_output=True, text=True, timeout=15)
        except Exception:
            continue
        if dr.returncode != 0:
            continue
        diff_content = dp.stdout[:4000] if dp.returncode == 0 else ""
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"The following commit was made to the {repo_name} codebase:\n"
                    f"Commit: {sha}\nMessage: {msg}\n\n"
                    f"Changed files:\n{dr.stdout[:1500]}\n\n"
                    f"Explain what this change does, why it was necessary, and how it improves the code."
                )},
                {"role": "assistant", "content": (
                    f"## Commit Analysis: {msg}\n\n"
                    f"**Commit**: `{sha}`\n**Repository**: {repo_name}\n\n"
                    f"### Changes\n{dr.stdout[:2000]}\n\n"
                    f"### Diff\n```diff\n{diff_content}\n```\n\n"
                    f"This change improves the {repo_name} codebase by addressing: {msg}"
                )},
            ],
            "source": "git_history",
            "metadata": {"commit_sha": sha, "commit_msg": msg, "repo": repo_name},
        })
    return pairs


def source_g_augmentation(manifest: list) -> list:
    """Source G: Data augmentation — tile/hardware/dtype/epilogue/pipeline variations."""
    pairs = []
    kernel_entries = [e for e in manifest
                      if e.get("content_type") == "kernel_impl"
                      and e.get("priority") in ("P0", "P1")]

    hw_info = {
        "gfx942": ("MI300X/MI325X (CDNA3)", "MFMA 16x16, 64KB LDS, wave64, preshuffle B layout"),
        "gfx950": ("MI350/MI355X (CDNA4)", "MFMA + FP4/MFMA-scale, 160KB LDS, wave64, wider LDS copy"),
        "gfx1250": ("MI450", "WMMA/TDM, 320KB LDS, wave32, FP8/FP4 native, async TDM copy"),
    }
    dtype_variants = ["fp8", "bf16", "fp16", "int8", "int4", "fp4"]
    tile_variants = [
        {"tile_m": 64, "tile_n": 64, "tile_k": 128},
        {"tile_m": 128, "tile_n": 128, "tile_k": 256},
        {"tile_m": 256, "tile_n": 128, "tile_k": 128},
        {"tile_m": 16, "tile_n": 128, "tile_k": 256},
    ]
    pipeline_variants = [
        ("lds_stage=1", "single LDS buffer, CK-style intrawave schedule"),
        ("lds_stage=2", "ping-pong double buffer for A tile prefetch"),
        ("lds_stage=3", "triple buffer for deeper pipeline overlap"),
    ]

    for e in kernel_entries:
        code = _read_file(e["full_path"])
        if len(code.strip()) < 200:
            continue
        op = e.get("operator", "custom")
        if op == "custom":
            continue
        feats = e.get("features", [])
        repo = e.get("repo", "aiter")
        framework = "FlyDSL (@flyc.kernel/@flyc.jit)" if repo == "FlyDSL" else "Triton/aiter"
        orig_hw = e.get("hardware", ["generic"])

        # Hardware adaptation (for each missing arch)
        for hw_arch, (hw_name, hw_details) in hw_info.items():
            exact_match = any(hw_arch == h for h in orig_hw)
            if exact_match:
                continue
            pairs.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Adapt the following {op} kernel for {hw_name} ({hw_arch}).\n"
                        f"Original kernel targets: {', '.join(orig_hw)}\n"
                        f"Target architecture details: {hw_details}\n"
                        f"Use {framework}.\n\n"
                        f"Original kernel:\n```python\n{code[:2500]}\n```"
                    )},
                    {"role": "assistant", "content": (
                        f"Here's the {op} kernel adapted for {hw_name} ({hw_arch}):\n\n"
                        f"```python\n{code}\n```\n\n"
                        f"Key adaptations for {hw_arch}:\n- {hw_details}"
                    )},
                ],
                "source": "augmentation_hardware",
                "metadata": {"source_path": e["path"], "target_hw": hw_arch, "operator": op, "repo": repo},
            })

        # Dtype variations (only for GEMM/MoE kernels)
        if op in ("gemm", "moe", "quant"):
            for dtype in dtype_variants:
                if dtype in code.lower():
                    continue
                pairs.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"Implement a {op} kernel using {dtype} data type for AMD GPU.\n"
                            f"Use {framework} with optimizations: {', '.join(feats) if feats else 'standard'}.\n"
                            f"Based on this reference:\n```python\n{code[:2000]}\n```"
                        )},
                        {"role": "assistant", "content": f"```python\n{code}\n```"},
                    ],
                    "source": "augmentation_dtype",
                    "metadata": {"source_path": e["path"], "target_dtype": dtype, "operator": op, "repo": repo},
                })

        # Tile size variations (GEMM/MoE only, FlyDSL only)
        if op in ("gemm", "moe") and repo == "FlyDSL" and "tile" in code.lower():
            for tv in tile_variants:
                tile_desc = f"tile_m={tv['tile_m']}, tile_n={tv['tile_n']}, tile_k={tv['tile_k']}"
                pairs.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"Configure the following {op} kernel with tile sizes: {tile_desc}\n"
                            f"Explain how this tiling affects occupancy and data reuse.\n\n"
                            f"```python\n{code[:2000]}\n```"
                        )},
                        {"role": "assistant", "content": (
                            f"With tile configuration ({tile_desc}), the kernel adjusts:\n\n"
                            f"```python\n{code}\n```"
                        )},
                    ],
                    "source": "augmentation_tile",
                    "metadata": {"source_path": e["path"], "tile": tv, "operator": op},
                })

        # Pipeline depth variations (for kernels with pipeline feature)
        if "pipeline" in feats and repo == "FlyDSL":
            for pname, pdesc in pipeline_variants:
                pairs.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"Configure this {op} kernel with {pname} ({pdesc}).\n"
                            f"Explain the tradeoffs of this pipeline depth.\n\n"
                            f"```python\n{code[:2000]}\n```"
                        )},
                        {"role": "assistant", "content": (
                            f"With {pname} ({pdesc}):\n\n```python\n{code}\n```"
                        )},
                    ],
                    "source": "augmentation_pipeline",
                    "metadata": {"source_path": e["path"], "pipeline": pname, "operator": op},
                })

    return pairs


def source_h_improvement_pairs(manifest: list) -> list:
    """Source H: Performance improvement pairs — simpler vs optimized kernels for same op.
    
    Pairs across both repos and within FlyDSL by complexity/features.
    """
    pairs = []
    complexity_rank = {"beginner": 0, "intermediate": 1, "advanced": 2, "expert": 3}

    # Group by operator across both repos
    by_op = defaultdict(list)
    for e in manifest:
        if e.get("content_type") != "kernel_impl":
            continue
        by_op[e["operator"]].append(e)

    for op, entries in by_op.items():
        if op == "custom" or len(entries) < 2:
            continue

        entries_sorted = sorted(entries, key=lambda x: (
            complexity_rank.get(x.get("complexity", "beginner"), 0),
            len(x.get("features", [])),
            x.get("lines", 0)
        ))

        n = len(entries_sorted)
        simpler = entries_sorted[:max(1, n // 3)]
        advanced = entries_sorted[max(1, n * 2 // 3):]

        for s in simpler[:5]:
            for a in advanced[:5]:
                if s["path"] == a["path"]:
                    continue
                s_cx = complexity_rank.get(s.get("complexity"), 0)
                a_cx = complexity_rank.get(a.get("complexity"), 0)
                if a_cx <= s_cx:
                    continue
                scode = _read_file(s["full_path"])
                acode = _read_file(a["full_path"])
                if len(scode.strip()) < 100 or len(acode.strip()) < 100:
                    continue
                s_feats = s.get("features", [])
                a_feats = a.get("features", [])
                new_feats = [f for f in a_feats if f not in s_feats]
                if not new_feats and s.get("repo") == a.get("repo"):
                    continue

                direction = ""
                if s.get("repo") != a.get("repo"):
                    direction = f" (from {s['repo']} → {a['repo']} style)"
                pairs.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"The following {op} kernel ({s.get('complexity', 'basic')} level) "
                            f"needs optimization{direction}:\n\n"
                            f"```python\n{scode[:3000]}\n```\n\n"
                            f"Optimize it to {a.get('complexity', 'expert')} level using: "
                            f"{', '.join(new_feats) if new_feats else 'advanced pipeline, tiling, and memory optimization'}.\n"
                            f"Target: AMD Instinct GPU with maximum roofline efficiency."
                        )},
                        {"role": "assistant", "content": (
                            f"Here's the optimized {a.get('complexity', 'expert')}-level version:\n\n"
                            f"```python\n{acode}\n```"
                        )},
                    ],
                    "source": "performance_improvement",
                    "metadata": {
                        "operator": op,
                        "simple_path": s["path"], "simple_repo": s.get("repo"),
                        "optimized_path": a["path"], "optimized_repo": a.get("repo"),
                        "simple_complexity": s.get("complexity"),
                        "optimized_complexity": a.get("complexity"),
                        "new_features": new_feats,
                    },
                })
    return pairs


def source_i_docs_direct_extraction() -> list:
    """Source I: Direct extraction from FlyDSL docs and skills."""
    pairs = []
    skills_dir = os.path.join(FLYDSL_ROOT, ".claude", "skills")

    # I.1: From prebuilt_kernels_guide — each kernel section → SFT
    guide_path = os.path.join(FLYDSL_ROOT, "docs", "prebuilt_kernels_guide.md")
    if os.path.exists(guide_path):
        guide = _read_file(guide_path)
        kernel_sections = re.split(r'\n##\s+\d+\.', guide)
        for section in kernel_sections[1:]:
            lines = section.strip().split("\n")
            title = lines[0].strip() if lines else ""
            body = "\n".join(lines).strip()
            if len(body) < 200:
                continue
            kernel_name = re.sub(r'\s*\(.*\)', '', title).strip()
            pairs.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"How do I use the FlyDSL pre-built {kernel_name} kernel?\n"
                        f"Explain the builder API, configuration options, algorithm details, "
                        f"and provide usage examples."
                    )},
                    {"role": "assistant", "content": body},
                ],
                "source": "docs_direct_extraction",
                "metadata": {"source_file": "docs/prebuilt_kernels_guide.md",
                             "section": kernel_name, "type": "prebuilt_guide"},
            })

    # I.2: From Claude Skills — each skill → comprehensive SFT
    if os.path.isdir(skills_dir):
        for skill_dir in sorted(Path(skills_dir).iterdir()):
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            content = _read_file(str(skill_file))
            if len(content.strip()) < 200:
                continue
            skill_name = skill_dir.name

            topic_map = {
                "flydsl-kernel-authoring": "write GPU kernels using FlyDSL",
                "gemm-optimization": "optimize GEMM kernels in FlyDSL",
                "lds-optimization": "optimize LDS usage and avoid bank conflicts in FlyDSL",
                "flydsl-tile-programming": "do tile programming in FlyDSL step by step",
                "debug-flydsl-kernel": "debug FlyDSL kernel issues (NaN, wrong results, performance)",
                "kernel-trace-analysis": "analyze kernel execution traces with ATT",
                "prefetch-data-load": "implement data prefetching and software pipelining",
                "oob-detection": "detect out-of-bounds memory access in GPU kernels",
                "capture-kernel-trace": "capture kernel traces using rocprofiler",
                "bisect-perf-regression": "bisect and locate performance regressions",
                "add-target-atom-op": "add new hardware atom operations to FlyDSL",
                "build-flydsl": "build FlyDSL from source",
                "build-rocm-image": "build a ROCm Docker image for FlyDSL",
                "format-code": "format FlyDSL code properly",
            }
            topic = topic_map.get(skill_name, f"use {skill_name} in FlyDSL")

            # Full skill as comprehensive guide
            pairs.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"How do I {topic}?\n"
                        f"Provide a comprehensive guide with step-by-step instructions, "
                        f"code examples, best practices, and troubleshooting tips."
                    )},
                    {"role": "assistant", "content": content},
                ],
                "source": "skill_direct_extraction",
                "metadata": {"skill_name": skill_name, "type": "full_skill"},
            })

            # Sub-sections as focused QA
            for sub_section in re.split(r'\n##\s+', content):
                if len(sub_section.strip()) < 150:
                    continue
                sub_lines = sub_section.strip().split("\n")
                sub_title = sub_lines[0].strip()
                sub_body = "\n".join(sub_lines[1:]).strip()
                if len(sub_body) < 100:
                    continue
                pairs.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"Regarding {skill_name.replace('-', ' ')} in FlyDSL: {sub_title}\n"
                            f"Explain in detail with examples."
                        )},
                        {"role": "assistant", "content": sub_body},
                    ],
                    "source": "skill_section_qa",
                    "metadata": {"skill_name": skill_name, "section": sub_title[:100],
                                 "type": "skill_section"},
                })

    # I.3: From other docs
    docs_dir = os.path.join(FLYDSL_ROOT, "docs")
    doc_files = [
        ("kernel_authoring_guide.md", "author GPU kernels"),
        ("architecture_guide.md", "understand FlyDSL compiler architecture"),
        ("layout_system_guide.md", "use FlyDSL layout algebra"),
        ("cute_layout_algebra_guide.md", "understand CuTe layout algebra mathematics"),
        ("testing_benchmarking_guide.md", "test and benchmark FlyDSL kernels"),
        ("extern_integration_guide.md", "integrate external bitcode into FlyDSL"),
    ]
    for fname, topic in doc_files:
        fpath = os.path.join(docs_dir, fname)
        if not os.path.exists(fpath):
            continue
        content = _read_file(fpath)
        if len(content.strip()) < 300:
            continue
        # Full doc as a guide
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"How do I {topic}?\n"
                    f"Provide a comprehensive guide covering all important aspects."
                )},
                {"role": "assistant", "content": content},
            ],
            "source": "docs_direct_extraction",
            "metadata": {"source_file": f"docs/{fname}", "type": "full_guide"},
        })

    # I.4: CLAUDE.md as repo navigation guide
    claude_md = os.path.join(FLYDSL_ROOT, "CLAUDE.md")
    if os.path.exists(claude_md):
        content = _read_file(claude_md)
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    "Describe the FlyDSL project structure, GPU architecture support, "
                    "kernel entry points, and coding conventions."
                )},
                {"role": "assistant", "content": content},
            ],
            "source": "docs_direct_extraction",
            "metadata": {"source_file": "CLAUDE.md", "type": "repo_guide"},
        })

    return pairs


def build_sft_dataset(manifest: list) -> list:
    """Build full SFT dataset from all sources A-I."""
    all_pairs = []

    print("  Source A: kernel reverse annotation...")
    a = source_a_kernel_reverse_annotation(manifest)
    print(f"    → {len(a)} pairs")
    all_pairs.extend(a)

    print("  Source B: test parameterization...")
    b = source_b_test_parameterization(manifest)
    print(f"    → {len(b)} pairs")
    all_pairs.extend(b)

    print("  Source C: documentation QA...")
    c = source_c_documentation_qa(manifest)
    print(f"    → {len(c)} pairs")
    all_pairs.extend(c)

    print("  Source D: tuned configs...")
    d = source_d_tuned_configs()
    print(f"    → {len(d)} pairs")
    all_pairs.extend(d)

    print("  Source E: cross-backend translation...")
    e = source_e_cross_backend_translation(manifest)
    print(f"    → {len(e)} pairs")
    all_pairs.extend(e)

    print("  Source F: git history (FlyDSL)...")
    f1 = source_f_git_history(FLYDSL_ROOT, "FlyDSL", ["kernels/", "python/", "tests/"])
    print(f"    → {len(f1)} pairs")
    all_pairs.extend(f1)

    print("  Source F: git history (aiter)...")
    f2 = source_f_git_history(AITER_ROOT, "aiter", ["aiter/ops/", "op_tests/"])
    print(f"    → {len(f2)} pairs")
    all_pairs.extend(f2)

    print("  Source G: data augmentation...")
    g = source_g_augmentation(manifest)
    print(f"    → {len(g)} pairs")
    all_pairs.extend(g)

    print("  Source H: performance improvement pairs...")
    h = source_h_improvement_pairs(manifest)
    print(f"    → {len(h)} pairs")
    all_pairs.extend(h)

    print("  Source I: docs/skills direct extraction...")
    i = source_i_docs_direct_extraction()
    print(f"    → {len(i)} pairs")
    all_pairs.extend(i)

    # Dedup by content hash
    seen = set()
    deduped = []
    for p in all_pairs:
        key = _content_hash(json.dumps(p["messages"][-1]["content"][:500], ensure_ascii=False))
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    print(f"  Dedup: {len(all_pairs)} → {len(deduped)}")
    return deduped


# ===================================================================
# Step 4: RL dataset
# ===================================================================
def build_rl_dataset(manifest: list) -> list:
    specs = []

    for e in manifest:
        if e.get("content_type") != "kernel_impl":
            continue
        if e.get("quality_grade") == "reject":
            continue
        op = e.get("operator", "custom")
        if op == "custom":
            continue
        for hw in e.get("hardware", ["generic"]):
            if hw == "generic":
                hw = "gfx942"
            specs.append({
                "id": f"manifest_{e['repo']}_{op}_{hw}_{len(specs)}",
                "operator": op, "hardware": hw, "source": "manifest",
                "source_repo": e.get("repo"),
                "source_path": e["path"], "features": e.get("features", []),
                "baseline_grade": e.get("quality_grade", "ungraded"),
            })

    # Tuned configs
    configs_dir = os.path.join(AITER_ROOT, "aiter", "configs")
    if os.path.isdir(configs_dir):
        for csv_file in sorted(Path(configs_dir).glob("*tuned*.csv")):
            if "untuned" in csv_file.name:
                continue
            op_name = csv_file.stem
            operator = "moe" if "moe" in op_name or "fmoe" in op_name else "gemm"
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
                rows = list(csv.DictReader(open(csv_file, newline="", encoding="utf-8")))
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
                    "id": f"tuned_{op_name}_{i}", "operator": operator, "hardware": "gfx942",
                    "params": params, "dtypes": dtypes, "source": "tuned_config",
                    "source_file": csv_file.name,
                })

    # Exploration specs
    sid = 0
    for hw in ["gfx942", "gfx950", "gfx1250"]:
        for dt in ["fp8", "bf16", "fp4"]:
            for s in GEMM_SHAPES:
                specs.append({
                    "id": f"explore_gemm_{sid}", "operator": "gemm", "hardware": hw,
                    "params": {**s, "in_dtype": dt, "out_dtype": "bf16"}, "source": "exploration"
                })
                sid += 1
        for s in ATTN_SHAPES:
            specs.append({
                "id": f"explore_attn_{sid}", "operator": "flash_attn", "hardware": hw,
                "params": s, "source": "exploration"
            })
            sid += 1
        for s in MOE_SHAPES:
            for dt in ["fp8", "bf16"]:
                specs.append({
                    "id": f"explore_moe_{sid}", "operator": "moe", "hardware": hw,
                    "params": {**s, "in_dtype": dt}, "source": "exploration"
                })
                sid += 1
        for s in NORM_SHAPES:
            for op in ["rmsnorm", "softmax", "layernorm", "rope"]:
                specs.append({
                    "id": f"explore_{op}_{sid}", "operator": op, "hardware": hw,
                    "params": s, "source": "exploration"
                })
                sid += 1
    return specs


# ===================================================================
# Step 5: Validation
# ===================================================================
def validate_dataset(cpt: list, sft_train: list, sft_val: list, rl: list) -> dict:
    """Run quality checks on generated datasets."""
    issues = []

    for i, rec in enumerate(cpt):
        if not rec.get("text", "").strip():
            issues.append(f"CPT[{i}]: empty text")
        if rec.get("meta", {}).get("weight", 0) <= 0:
            issues.append(f"CPT[{i}]: zero weight")

    for label, data in [("SFT-train", sft_train), ("SFT-val", sft_val)]:
        for i, rec in enumerate(data):
            msgs = rec.get("messages", [])
            if len(msgs) < 2:
                issues.append(f"{label}[{i}]: <2 messages")
            if msgs and not msgs[-1].get("content", "").strip():
                issues.append(f"{label}[{i}]: empty assistant response")

    for i, rec in enumerate(rl):
        if not rec.get("operator"):
            issues.append(f"RL[{i}]: missing operator")

    # Distribution checks
    cpt_types = Counter(r["meta"]["content_type"] for r in cpt)
    sft_sources = Counter(r.get("source", "unknown") for r in sft_train + sft_val)
    rl_ops = Counter(r["operator"] for r in rl)

    return {
        "issues": issues[:50],
        "cpt_type_dist": dict(cpt_types.most_common()),
        "sft_source_dist": dict(sft_sources.most_common()),
        "rl_op_dist": dict(rl_ops.most_common()),
    }


# ===================================================================
# Step 6: Package as HF dataset
# ===================================================================
def package_hf_dataset(cpt, sft_train, sft_val, rl, output_dir):
    """Package into HF dataset repo structure."""
    ds_dir = os.path.join(output_dir, "flydsl-agent-dataset")
    os.makedirs(os.path.join(ds_dir, "data", "cpt"), exist_ok=True)
    os.makedirs(os.path.join(ds_dir, "data", "sft"), exist_ok=True)
    os.makedirs(os.path.join(ds_dir, "data", "rl"), exist_ok=True)

    def write_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_jsonl(cpt, os.path.join(ds_dir, "data", "cpt", "train-00000-of-00001.jsonl"))
    write_jsonl(sft_train, os.path.join(ds_dir, "data", "sft", "train-00000-of-00001.jsonl"))
    write_jsonl(sft_val, os.path.join(ds_dir, "data", "sft", "validation-00000-of-00001.jsonl"))
    write_jsonl(rl, os.path.join(ds_dir, "data", "rl", "train-00000-of-00001.jsonl"))

    # .gitattributes
    with open(os.path.join(ds_dir, ".gitattributes"), "w") as f:
        f.write("*.jsonl filter=lfs diff=lfs merge=lfs -text\n")

    # Dataset card
    cpt_tokens = sum(r["meta"]["tokens_approx"] for r in cpt)
    cpt_types = Counter(r["meta"]["content_type"] for r in cpt)
    cpt_repos = Counter(r["meta"].get("source_repo", "unknown") for r in cpt)
    sft_sources = Counter(r.get("source", "unknown") for r in sft_train + sft_val)
    rl_ops = Counter(r["operator"] for r in rl)

    cpt_type_table = "\n".join(f"| {k} | {v} |" for k, v in cpt_types.most_common())
    cpt_repo_table = "\n".join(f"| {k} | {v} |" for k, v in cpt_repos.most_common())
    sft_src_table = "\n".join(f"| {k} | {v} |" for k, v in sft_sources.most_common())
    rl_ops_table = "\n".join(f"| {k} | {v} |" for k, v in rl_ops.most_common())

    readme = f"""---
language:
- en
- code
license: mit
task_categories:
- text-generation
tags:
- gpu-kernel
- amd
- rocm
- flydsl
- triton
- code-generation
- aiter
pretty_name: FlyDSL Agent Training Dataset (v2)
size_categories:
- 1K<n<10K
configs:
- config_name: cpt
  data_files:
  - split: train
    path: data/cpt/train-00000-of-00001.jsonl
- config_name: sft
  data_files:
  - split: train
    path: data/sft/train-00000-of-00001.jsonl
  - split: validation
    path: data/sft/validation-00000-of-00001.jsonl
- config_name: rl
  data_files:
  - split: train
    path: data/rl/train-00000-of-00001.jsonl
---

# FlyDSL Agent Training Dataset (v2 — Dual-Repo)

Training data for a FlyDSL GPU kernel code generation model, extracted from both
[FlyDSL](https://github.com/ROCm/FlyDSL) and [aiter](https://github.com/ROCm/aiter) repositories.

## Dataset Description

| Stage | Config | Description |
|-------|--------|-------------|
| **CPT** | `cpt` | Domain corpus: FlyDSL skills, docs, kernels, compiler source, aiter ops (~{cpt_tokens:,} tokens) |
| **SFT** | `sft` | Instruction-output pairs from 10 sources (train: {len(sft_train)}, val: {len(sft_val)}) |
| **RL** | `rl` | Task specifications for GRPO optimization ({len(rl)} specs) |

## Data Sources

### By Repository (CPT)

| Repository | Documents |
|-----------|-----------|
{cpt_repo_table}

### Content Type Distribution (CPT: {len(cpt)} documents)

| Type | Count |
|------|-------|
{cpt_type_table}

### SFT Source Distribution ({len(sft_train) + len(sft_val)} total)

| Source | Count |
|--------|-------|
{sft_src_table}

### RL Operator Distribution ({len(rl)} specs)

| Operator | Count |
|----------|-------|
{rl_ops_table}

## Sampling Weights (CPT)

- **Priority**: P0=3.0x, P1=1.5x, P2=0.5x
- **Content type**: expert_skill=2.5x, repo_guide=2.0x, doc_guide=1.5x, kernel_impl=1.2x
- **Quality grade**: gold=1.5x, silver=1.0x, bronze=0.7x

## Usage

```python
from datasets import load_dataset

cpt_ds = load_dataset("path/to/flydsl-agent-dataset", "cpt")
sft_ds = load_dataset("path/to/flydsl-agent-dataset", "sft")
rl_ds  = load_dataset("path/to/flydsl-agent-dataset", "rl")
```

## License

MIT (derived from ROCm/FlyDSL and ROCm/aiter)
"""
    with open(os.path.join(ds_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    return ds_dir


# ===================================================================
# Main
# ===================================================================
def main():
    random.seed(42)

    print("=" * 70)
    print("FlyDSL Data Pipeline v2 — Dual-Repo Processing")
    print("=" * 70)

    # Step 1: Scan both repos
    print(f"\n[1/6] Scanning FlyDSL repo: {FLYDSL_ROOT}")
    manifest_flydsl = scan_single_repo(FLYDSL_ROOT, "FlyDSL", infer_type_priority_flydsl)
    print(f"  Found {len(manifest_flydsl)} files")

    print(f"\n[1/6] Scanning aiter repo: {AITER_ROOT}")
    manifest_aiter = scan_single_repo(AITER_ROOT, "aiter", infer_type_priority_aiter)
    print(f"  Found {len(manifest_aiter)} files")

    manifest = manifest_flydsl + manifest_aiter
    print(f"\n  Total manifest: {len(manifest)} files")

    by_repo = Counter(e["repo"] for e in manifest)
    by_type = Counter(e["content_type"] for e in manifest)
    print(f"  By repo: {dict(by_repo)}")
    print(f"  By type: {dict(by_type.most_common())}")

    out = OUTPUT_DIR

    # Save manifest for benchmark_filter.py
    manifest_path = os.path.join(out, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"  Manifest saved: {manifest_path}")

    # Load graded manifest if available
    graded_path = os.path.join(out, "graded_manifest.json")
    if os.path.exists(graded_path):
        print(f"  Loading graded manifest from {graded_path}...")
        with open(graded_path, "r") as f:
            graded = json.load(f)
        grade_map = {e["path"]: e.get("quality_grade", "ungraded") for e in graded}
        for e in manifest:
            if e["path"] in grade_map:
                e["quality_grade"] = grade_map[e["path"]]
        grades = Counter(e["quality_grade"] for e in manifest)
        print(f"  Grades applied: {dict(grades)}")

    # Step 2: CPT
    print("\n[2/6] Building CPT dataset...")
    cpt = build_cpt_dataset(manifest)
    total_tok = sum(d["meta"]["tokens_approx"] for d in cpt)
    print(f"  CPT: {len(cpt)} documents, ~{total_tok:,} tokens")
    cpt_by_repo = Counter(d["meta"].get("source_repo") for d in cpt)
    print(f"  By repo: {dict(cpt_by_repo)}")

    # Step 3: SFT
    print("\n[3/6] Building SFT dataset (all sources A-I)...")
    sft_all = build_sft_dataset(manifest)
    random.shuffle(sft_all)

    # Stratified split
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
    print(f"  SFT train: {len(train)}, val: {len(val)}")

    # Step 4: RL
    print("\n[4/6] Building RL dataset...")
    rl = build_rl_dataset(manifest)
    print(f"  RL: {len(rl)} specs")

    # Step 5: Validate
    print("\n[5/6] Validating datasets...")
    report = validate_dataset(cpt, train, val, rl)
    if report["issues"]:
        print(f"  ⚠ {len(report['issues'])} issues found:")
        for iss in report["issues"][:10]:
            print(f"    - {iss}")
    else:
        print("  ✓ No issues found")

    # Step 6: Package
    print("\n[6/6] Packaging as HuggingFace dataset...")
    ds_dir = package_hf_dataset(cpt, train, val, rl, OUTPUT_DIR)
    print(f"  Output: {ds_dir}/")

    # Summary
    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  CPT:       {len(cpt):>6} documents  (~{total_tok:,} tokens)")
    print(f"  SFT train: {len(train):>6} pairs")
    print(f"  SFT val:   {len(val):>6} pairs")
    print(f"  RL specs:  {len(rl):>6} tasks")
    print(f"\n  Dataset: {ds_dir}/")
    print(f"  Load with: load_dataset('{ds_dir}', 'sft')")


if __name__ == "__main__":
    main()
