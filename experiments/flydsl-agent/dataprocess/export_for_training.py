#!/usr/bin/env python3
"""Export processed data into training-ready formats.

Supports:
  - ChatML format (for SFT with LLaMA-Factory)
  - Alpaca format (instruction/input/output)
  - Plain text (for CPT)
  - Train/val split with stratification
"""

import argparse
import json
import os
import random
from collections import defaultdict


def export_cpt_plain_text(input_path: str, output_path: str, weighted: bool = True):
    """Export CPT data as weighted plain text (one doc per line, repeated by weight)."""
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    documents = []
    for r in records:
        text = r.get("text", "")
        weight = r.get("weight", 1.0) if weighted else 1.0
        repeat = max(1, round(weight))
        for _ in range(repeat):
            documents.append(text)

    random.shuffle(documents)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc + "\n")

    print(f"CPT plain text: {len(documents)} documents → {output_path}")
    return len(documents)


def export_sft_chatml(input_path: str, output_path: str):
    """Export SFT data in ChatML format (for LLaMA-Factory)."""
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    chatml_data = []
    for r in records:
        messages = r.get("messages", [])
        if not messages:
            continue

        formatted = []
        for m in messages:
            formatted.append({
                "role": m["role"],
                "content": m["content"],
            })

        chatml_data.append({"conversations": formatted})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chatml_data, f, indent=2, ensure_ascii=False)

    print(f"SFT ChatML: {len(chatml_data)} conversations → {output_path}")
    return len(chatml_data)


def export_sft_alpaca(input_path: str, output_path: str):
    """Export SFT data in Alpaca format (instruction/input/output)."""
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    alpaca_data = []
    for r in records:
        messages = r.get("messages", [])
        system_msg = ""
        user_msg = ""
        assistant_msg = ""

        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            elif m["role"] == "user":
                user_msg = m["content"]
            elif m["role"] == "assistant":
                assistant_msg = m["content"]

        if not user_msg or not assistant_msg:
            continue

        alpaca_data.append({
            "instruction": user_msg,
            "input": system_msg,
            "output": assistant_msg,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)

    print(f"SFT Alpaca: {len(alpaca_data)} examples → {output_path}")
    return len(alpaca_data)


def train_val_split(input_path: str, train_path: str, val_path: str,
                    val_ratio: float = 0.1, stratify_key: str = "metadata.source"):
    """Split SFT data into train/val with stratification."""
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    # Group by stratification key
    groups = defaultdict(list)
    for r in records:
        val = r
        for key in stratify_key.split("."):
            if isinstance(val, dict):
                val = val.get(key, "unknown")
            else:
                val = "unknown"
                break
        groups[str(val)].append(r)

    train_records = []
    val_records = []

    for group_key, group_records in groups.items():
        random.shuffle(group_records)
        n_val = max(1, int(len(group_records) * val_ratio))
        val_records.extend(group_records[:n_val])
        train_records.extend(group_records[n_val:])

    random.shuffle(train_records)
    random.shuffle(val_records)

    for path, data in [(train_path, train_records), (val_path, val_records)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Train/Val split: {len(train_records)} train, {len(val_records)} val")
    return len(train_records), len(val_records)


def main():
    parser = argparse.ArgumentParser(description="Export training data")
    parser.add_argument("--output-dir", default="/workspace/dataprocess/output")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", choices=["all", "chatml", "alpaca", "plain"],
                        default="all")
    args = parser.parse_args()

    random.seed(args.seed)
    od = args.output_dir
    os.makedirs(od, exist_ok=True)

    cpt_input = os.path.join(od, "cpt_data.jsonl")
    sft_input = os.path.join(od, "sft_data.jsonl")

    print("Exporting training data...")
    print("=" * 50)

    # CPT exports
    if os.path.exists(cpt_input):
        if args.format in ("all", "plain"):
            export_cpt_plain_text(cpt_input, os.path.join(od, "cpt_train.txt"))

    # SFT exports
    if os.path.exists(sft_input):
        # Train/val split first
        train_path = os.path.join(od, "sft_train.jsonl")
        val_path = os.path.join(od, "sft_val.jsonl")
        train_val_split(sft_input, train_path, val_path, args.val_ratio)

        if args.format in ("all", "chatml"):
            export_sft_chatml(train_path, os.path.join(od, "sft_train_chatml.json"))
            export_sft_chatml(val_path, os.path.join(od, "sft_val_chatml.json"))

        if args.format in ("all", "alpaca"):
            export_sft_alpaca(train_path, os.path.join(od, "sft_train_alpaca.json"))
            export_sft_alpaca(val_path, os.path.join(od, "sft_val_alpaca.json"))

    print("\nExport complete.")


if __name__ == "__main__":
    main()
