#!/usr/bin/env python3
"""Batch annotation using local analysis (no external API needed).

This script does intelligent rule-based + heuristic annotation that
approximates what an AI model would produce. It serves as one of the
"model votes" in the consensus pipeline, based on deep code analysis.

For each kernel, it:
1. Parses the code structure (imports, decorators, function signatures)
2. Analyzes algorithmic patterns
3. Detects optimization features
4. Estimates complexity
5. Generates description, sft_instruction, and typical_shapes

This is the "Model 0" (rule-based expert) in the consensus pipeline.
"""

import json
import os
import re
import sys
from pathlib import Path

TAXONOMY = {
    "operator": [
        "gemm", "flash_attn", "mla", "moe", "softmax", "rmsnorm", "layernorm",
        "rope", "topk", "paged_attn", "allreduce", "quant", "custom",
    ],
    "features": [
        "swizzle_xor16", "pipeline", "async_copy", "double_buffer", "preshuffle",
        "blockscale", "split_k", "epilogue_fusion", "multi_wave", "sage_attention",
        "mxfp4", "tdm", "wmma", "shared_allocator", "layout_algebra",
    ],
}

OPERATOR_PATTERNS = {
    "gemm": [r'\bgemm\b', r'\bmatmul\b', r'\bdot\s*product\b', r'BLOCK_M.*BLOCK_N.*BLOCK_K',
             r'\bmfma\b', r'\bwmma\b', r'tl\.dot\b'],
    "flash_attn": [r'flash.?attn', r'flash.?attention', r'\bfused.?attention\b',
                   r'softmax.*\bV\b.*attention', r'Q.*K.*V.*softmax'],
    "mla": [r'\bmla\b', r'multi.?latent.?attention', r'latent_attention'],
    "moe": [r'\bmoe\b', r'mixture.?of.?expert', r'expert.*routing', r'topk.*expert'],
    "softmax": [r'\bsoftmax\b', r'exp.*sum.*div'],
    "rmsnorm": [r'\brmsnorm\b', r'\brms.?norm\b', r'root.?mean.?square.?norm'],
    "layernorm": [r'\blayernorm\b', r'\blayer.?norm\b', r'mean.*var.*normalize'],
    "rope": [r'\brope\b', r'rotary.?pos', r'rotary.?embed'],
    "topk": [r'\btopk\b', r'\btop.?k\b', r'topk_softmax'],
    "paged_attn": [r'paged.?attn', r'paged.?attention', r'page.?table'],
    "allreduce": [r'\ballreduce\b', r'\ball.?reduce\b', r'nccl', r'rccl'],
    "quant": [r'\bquant\b', r'\bdequant\b', r'quantiz', r'fp8.*scale', r'int8.*scale',
              r'mxfp4', r'block.?scale'],
}

FEATURE_PATTERNS = {
    "swizzle_xor16": [r'swizzle_xor', r'xor16', r'swizzle.*16'],
    "pipeline": [r'lds_stage', r'pipeline', r'num_stages\s*[>=]\s*[23]', r'software.?pipeline'],
    "async_copy": [r'async_copy', r'cp\.async', r'tl\.async_copy'],
    "double_buffer": [r'double_buf', r'ping.?pong', r'lds_stage\s*=\s*2'],
    "preshuffle": [r'preshuffle', r'pre.?shuffle'],
    "blockscale": [r'blockscale', r'block_scale', r'mxfp'],
    "split_k": [r'split_k', r'splitk', r'SPLIT_K'],
    "epilogue_fusion": [r'epilogue', r'bias_add', r'fused.*activation', r'epilogue_fusion'],
    "multi_wave": [r'waves_per_eu', r'num_waves', r'WAVES_PER_EU'],
    "sage_attention": [r'sage.?attention', r'sage_attn'],
    "mxfp4": [r'mxfp4', r'mx_fp4', r'microscale'],
    "tdm": [r'\btdm\b', r'tensor_data_movement'],
    "wmma": [r'\bwmma\b', r'wave_matrix'],
    "shared_allocator": [r'SmemAllocator', r'shared_allocator', r'lds_allocat'],
    "layout_algebra": [r'Layout\(', r'Stride\(', r'Shape\(', r'make_layout', r'coalesce'],
}

TYPICAL_SHAPES = {
    "gemm": [
        {"M": 4096, "N": 4096, "K": 4096, "dtype": "fp16"},
        {"M": 1, "N": 4096, "K": 14336, "dtype": "fp8"},
        {"M": 8192, "N": 8192, "K": 4096, "dtype": "bf16"},
    ],
    "flash_attn": [
        {"batch": 32, "seq_len": 2048, "heads": 32, "head_dim": 128, "dtype": "fp16"},
        {"batch": 1, "seq_len": 32768, "heads": 64, "head_dim": 128, "dtype": "bf16"},
    ],
    "mla": [
        {"batch": 1, "seq_len": 4096, "num_heads": 128, "qk_nope_dim": 128, "v_head_dim": 128},
    ],
    "moe": [
        {"tokens": 4096, "experts": 8, "topk": 2, "hidden": 4096, "intermediate": 14336},
    ],
    "softmax": [
        {"batch": 32, "seq_len": 2048, "vocab": 32000},
        {"rows": 4096, "cols": 128256},
    ],
    "rmsnorm": [
        {"batch": 32, "seq_len": 2048, "hidden": 4096},
    ],
    "layernorm": [
        {"batch": 32, "seq_len": 2048, "hidden": 4096},
    ],
    "rope": [
        {"batch": 32, "seq_len": 2048, "heads": 32, "head_dim": 128},
    ],
    "quant": [
        {"rows": 4096, "cols": 4096, "group_size": 128, "bits": 8},
    ],
}


def detect_operator(path: str, code: str) -> str:
    fname = os.path.basename(path).lower()
    scores = {}
    for op, patterns in OPERATOR_PATTERNS.items():
        score = 0
        if op in fname:
            score += 5
        for pat in patterns:
            matches = re.findall(pat, code, re.IGNORECASE)
            score += len(matches)
        if score > 0:
            scores[op] = score

    if scores:
        return max(scores, key=scores.get)
    return "custom"


def detect_features(code: str) -> list:
    found = []
    for feat, patterns in FEATURE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, code, re.IGNORECASE):
                found.append(feat)
                break
    return sorted(set(found))


def detect_hardware(code: str) -> list:
    hw = set()
    for m in re.finditer(r'gfx(\d+)', code):
        arch = f"gfx{m.group(1)}"
        hw.add(arch)
    if not hw:
        hw.add("generic")
    return sorted(hw)


def estimate_complexity(code: str, features: list) -> str:
    lines = code.count('\n') + 1
    n_feat = len(features)
    n_funcs = len(re.findall(r'^\s*def\s+', code, re.MULTILINE))
    n_classes = len(re.findall(r'^\s*class\s+', code, re.MULTILINE))

    if lines > 1500 or n_feat >= 4 or (n_feat >= 3 and lines > 800):
        return "expert"
    elif lines > 800 or n_feat >= 2 or (n_funcs > 10 and lines > 500):
        return "advanced"
    elif lines > 300 or n_feat >= 1:
        return "intermediate"
    return "beginner"


def generate_description(path: str, op: str, features: list, hw: list, lines: int) -> str:
    fname = Path(path).stem
    feat_str = ", ".join(features[:3]) if features else "standard"
    hw_str = "/".join(hw[:2]) if hw and hw != ["generic"] else "AMD GPU"
    return (f"{fname}: {op.upper()} kernel implementation for {hw_str} "
            f"with {feat_str} optimizations ({lines} lines)")


def generate_sft_instruction(op: str, features: list, hw: list) -> str:
    hw_names = {"gfx942": "MI300X", "gfx950": "MI350X", "gfx1250": "MI450"}
    hw_str = hw_names.get(hw[0], hw[0]) if hw and hw[0] != "generic" else "AMD Instinct GPU"

    feat_str = ""
    if features:
        feat_str = " with " + ", ".join(features[:3])

    dtype_hint = ""
    if "preshuffle" in features or "mxfp4" in features:
        dtype_hint = " using FP8 inputs"
    elif "blockscale" in features:
        dtype_hint = " using MXFP4/block-scaled inputs"

    return f"Write a high-performance {op} kernel for {hw_str}{feat_str}{dtype_hint} using FlyDSL"


def annotate_entry(prompt_rec: dict) -> dict:
    """Analyze a kernel from its prompt record and produce annotation."""
    code = ""
    prompt_text = prompt_rec.get("prompt", "")
    code_match = re.search(r'```python\n(.*?)```', prompt_text, re.DOTALL)
    if code_match:
        code = code_match.group(1)

    path = prompt_rec.get("path", "")
    lines = code.count('\n') + 1

    op = detect_operator(path, code)
    features = detect_features(code)
    hardware = detect_hardware(code)
    complexity = estimate_complexity(code, features)

    return {
        "operator": op,
        "features": features,
        "hardware": hardware,
        "complexity": complexity,
        "description": generate_description(path, op, features, hardware, lines),
        "known_limitations": "none",
        "sft_instruction": generate_sft_instruction(op, features, hardware),
        "typical_shapes": TYPICAL_SHAPES.get(op, [{"note": "custom operator"}]),
    }


def main():
    prompts_path = sys.argv[1] if len(sys.argv) > 1 else "/home/danyzhan/flydsl-agent-dataset/metadata/annotation_prompts.jsonl"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "/home/danyzhan/flydsl-agent-dataset/metadata/responses_rulebased.jsonl"

    prompts = []
    with open(prompts_path) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    print(f"Annotating {len(prompts)} entries with rule-based expert system...")

    with open(output_path, "w", encoding="utf-8") as out:
        for i, p in enumerate(prompts):
            annotation = annotate_entry(p)
            record = {
                "id": p["id"],
                "path": p["path"],
                "repo": p.get("repo"),
                "model_name": "rulebased_v2",
                "model": "rule-based-expert-v2",
                "response": annotation,
                "raw_response": "",
                "current_labels": p.get("current_labels", {}),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(prompts)}] processed")

    print(f"Done: {len(prompts)} annotations saved to {output_path}")


if __name__ == "__main__":
    main()
