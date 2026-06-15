#!/usr/bin/env python3
"""Rebuild datasets using annotated manifest.

After multi-model consensus annotation, this script rebuilds the
CPT/SFT/RL datasets with enriched metadata from AI annotations.

Key improvements from annotations:
- More accurate operator labels → better SFT categorization
- Richer feature detection → better weighted sampling
- AI-generated descriptions → richer CPT metadata headers
- AI-generated sft_instructions → higher quality SFT pairs
- Typical shapes → more realistic RL task specs

Usage:
  python3 rebuild_with_annotations.py \
      --manifest /path/to/annotated_manifest.json \
      --output-dir /home/danyzhan/flydsl-agent-dataset/data
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def compute_weight(entry):
    """Compute sampling weight with AI annotation boost."""
    base = {"P0": 3.0, "P1": 1.5, "P2": 0.5}.get(entry.get("priority", "P2"), 0.5)
    grade_mult = {"gold": 1.5, "silver": 1.0, "bronze": 0.7}.get(
        entry.get("quality_grade", ""), 1.0)
    type_mult = {
        "expert_skill": 2.5, "repo_guide": 2.0, "doc_guide": 1.5,
        "kernel_impl": 1.2, "framework_api": 1.0, "test": 0.8,
        "tuned_config": 0.6, "build_script": 0.4, "hardware_spec": 0.3,
    }.get(entry.get("content_type", ""), 1.0)

    ai_boost = 1.0
    if entry.get("ai_annotated"):
        ai_boost = 1.1
        if not entry.get("needs_human_review"):
            ai_boost = 1.2

    return base * grade_mult * type_mult * ai_boost


def build_cpt_text(entry, code):
    """Build CPT document with enriched AI metadata."""
    lines = [
        "<|doc_start|>",
        f"[file: {entry['path']}]",
        f"[type: {entry.get('content_type', 'unknown')}]",
        f"[operator: {entry.get('operator', 'custom')}]",
    ]
    if entry.get("hardware"):
        lines.append(f"[hardware: {', '.join(entry['hardware'])}]")
    if entry.get("complexity"):
        lines.append(f"[complexity: {entry.get('complexity', 'unknown')}]")
    if entry.get("quality_grade"):
        lines.append(f"[grade: {entry['quality_grade']}]")
    if entry.get("features"):
        lines.append(f"[features: {', '.join(entry['features'])}]")
    if entry.get("ai_description"):
        lines.append(f"[description: {entry['ai_description']}]")
    lines.append("")
    lines.append(code)
    lines.append("<|doc_end|>")
    return "\n".join(lines)


def generate_sft_from_annotations(manifest):
    """Generate SFT pairs using AI-generated sft_instructions."""
    pairs = []
    for entry in manifest:
        if not entry.get("ai_sft_instruction"):
            continue
        fp = entry.get("full_path", "")
        if not os.path.exists(fp):
            continue
        try:
            code = open(fp, encoding="utf-8", errors="replace").read()
        except Exception:
            continue

        pair = {
            "messages": [
                {"role": "system", "content": "You are a FlyDSL GPU kernel expert. Write high-performance GPU kernels using the FlyDSL framework for AMD Instinct accelerators."},
                {"role": "user", "content": entry["ai_sft_instruction"]},
                {"role": "assistant", "content": code},
            ],
            "source": "ai_annotated_instruction",
            "metadata": {
                "operator": entry.get("operator"),
                "complexity": entry.get("ai_complexity", entry.get("complexity")),
                "hardware": entry.get("hardware", []),
                "features": entry.get("features", []),
                "quality_grade": entry.get("quality_grade", ""),
                "ai_models": entry.get("ai_models", []),
                "agreement": entry.get("ai_agreement", {}),
            },
        }
        pairs.append(pair)
    return pairs


def generate_rl_from_annotations(manifest):
    """Generate RL specs using AI-generated typical_shapes."""
    specs = []
    for entry in manifest:
        shapes = entry.get("ai_typical_shapes", [])
        if not shapes or not isinstance(shapes, list):
            continue
        op = entry.get("operator", "custom")
        hw_list = entry.get("hardware", ["generic"])

        for i, shape in enumerate(shapes[:3]):
            if not isinstance(shape, dict):
                continue
            for hw in hw_list:
                spec = {
                    "id": f"ai_{entry['path']}_{hw}_{i}",
                    "operator": op,
                    "hardware": hw,
                    "params": shape,
                    "source": "ai_annotation",
                    "quality_grade": entry.get("quality_grade", ""),
                    "features": entry.get("features", []),
                }
                specs.append(spec)
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--merge-existing", action="store_true",
                        help="Merge with existing datasets instead of replacing")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    annotated_count = sum(1 for e in manifest if e.get("ai_annotated"))
    print(f"Manifest: {len(manifest)} entries, {annotated_count} AI-annotated")

    # Generate AI-enhanced SFT pairs
    ai_sft = generate_sft_from_annotations(manifest)
    print(f"AI-annotated SFT pairs: {len(ai_sft)}")

    # Generate AI-enhanced RL specs
    ai_rl = generate_rl_from_annotations(manifest)
    print(f"AI-annotated RL specs: {len(ai_rl)}")

    # Merge with existing datasets if requested
    sft_path = os.path.join(args.output_dir, "sft", "train-00000-of-00001.jsonl")
    rl_path = os.path.join(args.output_dir, "rl", "train-00000-of-00001.jsonl")

    if args.merge_existing and os.path.exists(sft_path):
        existing_sft = []
        with open(sft_path) as f:
            for line in f:
                if line.strip():
                    existing_sft.append(json.loads(line))
        print(f"Existing SFT: {len(existing_sft)}")
        existing_ids = {json.dumps(e.get("messages", [{}])[1].get("content", "")[:100])
                        for e in existing_sft}
        new_sft = [p for p in ai_sft
                   if json.dumps(p["messages"][1]["content"][:100]) not in existing_ids]
        print(f"New unique AI SFT pairs: {len(new_sft)}")
        all_sft = existing_sft + new_sft
    else:
        all_sft = ai_sft

    if args.merge_existing and os.path.exists(rl_path):
        existing_rl = []
        with open(rl_path) as f:
            for line in f:
                if line.strip():
                    existing_rl.append(json.loads(line))
        print(f"Existing RL: {len(existing_rl)}")
        existing_rl_ids = {e.get("id", "") for e in existing_rl}
        new_rl = [s for s in ai_rl if s["id"] not in existing_rl_ids]
        print(f"New unique AI RL specs: {len(new_rl)}")
        all_rl = existing_rl + new_rl
    else:
        all_rl = ai_rl

    # Write outputs
    if all_sft:
        os.makedirs(os.path.dirname(sft_path), exist_ok=True)
        with open(sft_path, "w", encoding="utf-8") as f:
            for item in all_sft:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Wrote {len(all_sft)} SFT pairs to {sft_path}")

    if all_rl:
        os.makedirs(os.path.dirname(rl_path), exist_ok=True)
        with open(rl_path, "w", encoding="utf-8") as f:
            for item in all_rl:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Wrote {len(all_rl)} RL specs to {rl_path}")

    print("\nDone! Datasets enriched with AI annotations.")


if __name__ == "__main__":
    main()
