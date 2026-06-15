#!/usr/bin/env python3
"""AI-driven annotation (Plan §4.4 Layer 2).

Uses AI model to annotate kernel files with:
  - Refined operator classification
  - Feature detection beyond regex
  - Description (one-sentence summary)
  - SFT instruction generation
  - Known limitations
  - Typical shapes

This script generates annotation prompts that can be processed by any LLM.
It outputs a JSONL file of annotation requests, and can merge AI responses back
into the manifest.

Usage:
  # Step 1: Generate prompts
  python3 ai_annotate.py --manifest /path/to/manifest.json --output-prompts /path/to/prompts.jsonl

  # Step 2: (User feeds prompts to AI and collects responses)

  # Step 3: Merge responses
  python3 ai_annotate.py --manifest /path/to/manifest.json --responses /path/to/responses.jsonl --output /path/to/annotated_manifest.json
"""

import json
import os
import re
import sys
from pathlib import Path

FLYDSL_ROOT = os.environ.get("FLYDSL_ROOT", "/FlyDSL")
AITER_ROOT = os.environ.get("AITER_ROOT", "/aiter")

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
    "complexity": ["beginner", "intermediate", "advanced", "expert"],
    "hardware": ["gfx942", "gfx950", "gfx1250", "gfx11", "gfx120", "generic"],
}


def build_annotation_prompt(entry: dict, code: str, context_docs: list = None) -> str:
    """Build an annotation prompt for a kernel file."""
    context = ""
    if context_docs:
        context = "\n\n--- Reference Context ---\n" + "\n\n---\n\n".join(context_docs[:3])

    return f"""Analyze the following GPU kernel code and output a JSON annotation.

Code path: {entry.get('path', 'unknown')}
Repository: {entry.get('repo', 'unknown')}
Lines: {entry.get('lines', 0)}

```python
{code[:8000]}
```
{context}

Output a JSON object with these fields:
1. "operator": one of {json.dumps(TAXONOMY['operator'])}
2. "features": list from {json.dumps(TAXONOMY['features'])} (select all that apply)
3. "complexity": one of {json.dumps(TAXONOMY['complexity'])}
4. "hardware": list of target architectures from {json.dumps(TAXONOMY['hardware'])}
5. "description": One sentence describing the kernel's purpose and key optimization strategy
6. "known_limitations": Brief note on which configurations may have poor performance (or "none" if unknown)
7. "sft_instruction": A natural language instruction a user might give to request this kernel (e.g. "Write an FP8 preshuffle GEMM kernel for MI300X with ping-pong LDS pipeline")
8. "typical_shapes": List of 2-3 typical parameter dictionaries for this kernel in LLM inference scenarios

Output ONLY valid JSON, no markdown formatting."""


def get_context_docs(entry: dict) -> list:
    """Get relevant context documents for annotation."""
    docs = []
    op = entry.get("operator", "custom")

    # Always include GPU arch info from CLAUDE.md
    claude_md = os.path.join(FLYDSL_ROOT, "CLAUDE.md")
    if os.path.exists(claude_md):
        content = open(claude_md, encoding="utf-8", errors="replace").read()
        arch_section = re.search(r'## GPU Architecture Support.*?(?=\n## |\Z)', content, re.DOTALL)
        if arch_section:
            docs.append(f"GPU Architecture Reference:\n{arch_section.group()[:1500]}")

    # Operator-specific skill/doc context
    skill_map = {
        "gemm": "gemm-optimization",
        "flash_attn": "flydsl-kernel-authoring",
        "mla": "flydsl-kernel-authoring",
        "moe": "flydsl-kernel-authoring",
    }
    if op in skill_map:
        skill_path = os.path.join(FLYDSL_ROOT, ".claude", "skills", skill_map[op], "SKILL.md")
        if os.path.exists(skill_path):
            skill = open(skill_path, encoding="utf-8", errors="replace").read()
            docs.append(f"Expert Skill Reference ({skill_map[op]}):\n{skill[:2000]}")

    return docs


def generate_prompts(manifest_path: str, output_path: str):
    """Generate annotation prompts for kernel entries."""
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    kernel_entries = [e for e in manifest
                      if e.get("content_type") == "kernel_impl"
                      and e.get("priority") in ("P0", "P1")]

    print(f"Generating prompts for {len(kernel_entries)} kernel entries...")

    with open(output_path, "w", encoding="utf-8") as out:
        for i, entry in enumerate(kernel_entries):
            fp = entry.get("full_path", "")
            if not os.path.exists(fp):
                continue
            try:
                code = open(fp, encoding="utf-8", errors="replace").read()
            except Exception:
                continue

            context = get_context_docs(entry)
            prompt = build_annotation_prompt(entry, code, context)

            record = {
                "id": f"{entry.get('repo', 'unknown')}:{entry['path']}",
                "path": entry["path"],
                "repo": entry.get("repo"),
                "prompt": prompt,
                "current_labels": {
                    "operator": entry.get("operator"),
                    "features": entry.get("features"),
                    "complexity": entry.get("complexity"),
                    "hardware": entry.get("hardware"),
                },
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(kernel_entries)} prompts to {output_path}")


def merge_responses(manifest_path: str, responses_path: str, output_path: str):
    """Merge AI annotation responses back into manifest."""
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    responses = {}
    with open(responses_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = rec.get("id", "")
            resp = rec.get("response", {})
            if isinstance(resp, str):
                try:
                    resp = json.loads(resp)
                except Exception:
                    continue
            responses[rid] = resp

    print(f"Loaded {len(responses)} AI responses")

    merged = 0
    for entry in manifest:
        rid = f"{entry.get('repo', 'unknown')}:{entry['path']}"
        if rid in responses:
            resp = responses[rid]
            # Merge AI annotations (AI wins for text fields, consensus for enums)
            for field in ["description", "known_limitations", "sft_instruction", "typical_shapes"]:
                if field in resp:
                    entry[f"ai_{field}"] = resp[field]

            # For enum fields, use AI if script was "custom" or matches
            if resp.get("operator") and entry.get("operator") == "custom":
                entry["operator"] = resp["operator"]
            if resp.get("features"):
                existing = set(entry.get("features", []))
                ai_feats = set(resp["features"])
                entry["features"] = sorted(existing | ai_feats)
            if resp.get("complexity"):
                entry["ai_complexity"] = resp["complexity"]
            if resp.get("hardware"):
                existing = set(entry.get("hardware", []))
                ai_hw = set(resp["hardware"])
                entry["hardware"] = sorted(existing | ai_hw)
            merged += 1

    print(f"Merged {merged} annotations into manifest")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-prompts", help="Generate prompts JSONL")
    parser.add_argument("--responses", help="AI responses JSONL to merge")
    parser.add_argument("--output", help="Output annotated manifest")
    args = parser.parse_args()

    if args.output_prompts:
        generate_prompts(args.manifest, args.output_prompts)
    elif args.responses and args.output:
        merge_responses(args.manifest, args.responses, args.output)
    else:
        print("Specify --output-prompts to generate, or --responses + --output to merge")
