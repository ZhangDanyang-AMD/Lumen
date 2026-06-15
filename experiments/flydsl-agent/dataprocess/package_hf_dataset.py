#!/usr/bin/env python3
"""Package processed data into a HuggingFace dataset repository structure.

Creates a dataset repo at /home/danyzhan/flydsl-agent-dataset/ with:
  - README.md dataset card (YAML frontmatter + description)
  - data/cpt/train-00000-of-00001.jsonl
  - data/sft/train-00000-of-00001.jsonl
  - data/sft/validation-00000-of-00001.jsonl
  - data/rl/train-00000-of-00001.jsonl

Compatible with:
    from datasets import load_dataset
    ds = load_dataset("path/to/flydsl-agent-dataset", "sft")
"""

import json
import os
import sys
from collections import Counter

SRC_DIR = "/home/danyzhan"
DEST_DIR = "/home/danyzhan/flydsl-agent-dataset"


def convert_json_to_jsonl(json_path: str, jsonl_path: str) -> int:
    """Convert a JSON array file to JSONL (one object per line)."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(data)


def gather_stats(data: list, config_name: str) -> dict:
    """Collect statistics for the dataset card."""
    stats = {"num_examples": len(data)}

    if config_name == "cpt":
        total_tok = sum(d.get("meta", {}).get("tokens_approx", 0) for d in data)
        stats["total_tokens_approx"] = total_tok
        ct = Counter(d.get("meta", {}).get("content_type") for d in data)
        stats["content_type_dist"] = dict(ct.most_common())
        op = Counter(d.get("meta", {}).get("operator") for d in data)
        stats["operator_dist"] = dict(op.most_common())

    elif config_name == "sft":
        src = Counter(d.get("source") for d in data)
        stats["source_dist"] = dict(src.most_common())
        op = Counter(d.get("metadata", {}).get("operator") for d in data if d.get("metadata", {}).get("operator"))
        stats["operator_dist"] = dict(op.most_common())

    elif config_name == "rl":
        op = Counter(d.get("operator") for d in data)
        stats["operator_dist"] = dict(op.most_common())
        hw = Counter(d.get("hardware") for d in data)
        stats["hardware_dist"] = dict(hw.most_common())
        src = Counter(d.get("source") for d in data)
        stats["source_dist"] = dict(src.most_common())

    return stats


def build_dataset_card(cpt_stats, sft_train_stats, sft_val_stats, rl_stats) -> str:
    return f"""---
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
pretty_name: FlyDSL Agent Training Dataset
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

# FlyDSL Agent Training Dataset

Training data for a FlyDSL GPU kernel code generation model, extracted and processed
from the [aiter](https://github.com/ROCm/aiter) repository (AMD's AI Tensor Engine for ROCm).

## Dataset Description

This dataset supports a three-stage post-training pipeline for building a specialized
GPU kernel generation model:

| Stage | Config | Description |
|-------|--------|-------------|
| **CPT** (Continued Pre-Training) | `cpt` | Domain-adapted corpus for next-token prediction — internalizes FlyDSL/Triton syntax, AMD GPU patterns, and kernel idioms |
| **SFT** (Supervised Fine-Tuning) | `sft` | Instruction-output pairs teaching the model to follow kernel generation instructions |
| **RL** (Reinforcement Learning) | `rl` | Task specifications defining operator types, hardware targets, and parameter ranges for GRPO-based optimization |

## Usage

```python
from datasets import load_dataset

# Load a specific config
cpt_ds = load_dataset("path/to/flydsl-agent-dataset", "cpt")
sft_ds = load_dataset("path/to/flydsl-agent-dataset", "sft")
rl_ds  = load_dataset("path/to/flydsl-agent-dataset", "rl")

# Access data
print(sft_ds["train"][0]["messages"])
print(rl_ds["train"][0]["operator"])
```

## Dataset Statistics

### CPT Config (Continued Pre-Training)

- **Examples**: {cpt_stats['num_examples']:,}
- **Approx. tokens**: {cpt_stats.get('total_tokens_approx', 0):,}

Each example is a structured document with metadata headers and full source file content:

```json
{{
  "text": "<|doc_start|>\\n[file: ...]\\n[type: ...]\\n...\\n<|doc_end|>",
  "meta": {{
    "source_path": "aiter/ops/triton/...",
    "content_type": "kernel_impl",
    "operator": "gemm",
    "hardware": ["gfx942"],
    "complexity": "advanced",
    "priority": "P0",
    "weight": 3.6,
    "tokens_approx": 5000
  }}
}}
```

**Content type distribution**:

| Type | Count |
|------|-------|
{_format_dist_table(cpt_stats.get('content_type_dist', {}))}

### SFT Config (Supervised Fine-Tuning)

- **Train examples**: {sft_train_stats['num_examples']:,}
- **Validation examples**: {sft_val_stats['num_examples']:,}

Each example follows the OpenAI chat format:

```json
{{
  "messages": [
    {{"role": "system", "content": "You are a FlyDSL GPU kernel programming expert..."}},
    {{"role": "user", "content": "Implement a gemm kernel for AMD GPU (gfx942)..."}},
    {{"role": "assistant", "content": "```python\\nimport triton\\n...```"}}
  ],
  "source": "kernel_reverse_annotation",
  "metadata": {{
    "source_path": "aiter/ops/triton/gemm.py",
    "operator": "gemm",
    "complexity": "advanced"
  }}
}}
```

**Source distribution**:

| Source | Count |
|--------|-------|
{_format_dist_table(sft_train_stats.get('source_dist', {}))}

### RL Config (Reinforcement Learning)

- **Task specifications**: {rl_stats['num_examples']:,}

Each example defines an RL task specification:

```json
{{
  "id": "explore_gemm_0",
  "operator": "gemm",
  "hardware": "gfx942",
  "params": {{"M": 4096, "N": 4096, "K": 2048, "in_dtype": "fp8", "out_dtype": "bf16"}},
  "source": "exploration"
}}
```

**Operator distribution**:

| Operator | Count |
|----------|-------|
{_format_dist_table(rl_stats.get('operator_dist', {}))}

## Data Sources

All data is extracted from [ROCm/aiter](https://github.com/ROCm/aiter):

| Source | Content | Files |
|--------|---------|-------|
| Triton kernels (`aiter/ops/triton/`) | Triton GPU kernel implementations | ~320 |
| Python ops (`aiter/ops/`) | Op wrappers and dispatch | ~415 |
| C++ kernels (`csrc/`) | HIP C++ kernel implementations | ~360 |
| Operator tests (`op_tests/`) | Parameterized unit tests | ~50 |
| Documentation (`docs/`) | ISA optimization guides, autotuning | ~15 |
| Tuned configs (`aiter/configs/`) | Auto-tuned optimal parameters | ~20 CSV |
| Claude skills (`.claude/skills/`) | Expert programming guides | varies |

## Supported Hardware

| GPU | Architecture | Status |
|-----|-------------|--------|
| AMD Instinct MI300X | gfx942 (CDNA3) | Primary |
| AMD Instinct MI325X | gfx942 (CDNA3) | Primary |
| AMD Instinct MI350 | gfx950 (CDNA4) | Supported |
| AMD Instinct MI355X | gfx950 (CDNA4) | Supported |

## Sampling Weights (CPT)

CPT documents have a `weight` field in `meta` for weighted sampling during training:

- **Priority**: P0=3.0x, P1=1.5x, P2=0.5x
- **Quality grade**: gold=1.5x, silver=1.0x, bronze=0.7x
- **Content type**: expert_skill=2.5x, repo_guide=2.0x, kernel_impl=1.2x, ...

Final weight = priority × grade × type (range: 0.1 to 7.5)

## License

This dataset is derived from [ROCm/aiter](https://github.com/ROCm/aiter) which is
released under the MIT License.
"""


def _format_dist_table(dist: dict) -> str:
    lines = []
    for k, v in dist.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def main():
    os.makedirs(DEST_DIR, exist_ok=True)

    # --- CPT ---
    print("Packaging CPT data...")
    cpt_src = os.path.join(SRC_DIR, "flydsl_cpt_data.json")
    cpt_dst = os.path.join(DEST_DIR, "data", "cpt", "train-00000-of-00001.jsonl")
    n_cpt = convert_json_to_jsonl(cpt_src, cpt_dst)
    with open(cpt_src) as f:
        cpt_data = json.load(f)
    cpt_stats = gather_stats(cpt_data, "cpt")
    print(f"  CPT: {n_cpt} examples → {cpt_dst}")

    # --- SFT train ---
    print("Packaging SFT train data...")
    sft_src = os.path.join(SRC_DIR, "flydsl_sft_data.json")
    sft_dst = os.path.join(DEST_DIR, "data", "sft", "train-00000-of-00001.jsonl")
    n_sft = convert_json_to_jsonl(sft_src, sft_dst)
    with open(sft_src) as f:
        sft_train_data = json.load(f)
    sft_train_stats = gather_stats(sft_train_data, "sft")
    print(f"  SFT train: {n_sft} examples → {sft_dst}")

    # --- SFT validation ---
    print("Packaging SFT validation data...")
    val_src = os.path.join(SRC_DIR, "flydsl_val_data.json")
    val_dst = os.path.join(DEST_DIR, "data", "sft", "validation-00000-of-00001.jsonl")
    n_val = convert_json_to_jsonl(val_src, val_dst)
    with open(val_src) as f:
        val_data = json.load(f)
    sft_val_stats = gather_stats(val_data, "sft")
    print(f"  SFT val: {n_val} examples → {val_dst}")

    # --- RL ---
    print("Packaging RL data...")
    rl_src = os.path.join(SRC_DIR, "flydsl_rl_data.json")
    rl_dst = os.path.join(DEST_DIR, "data", "rl", "train-00000-of-00001.jsonl")
    n_rl = convert_json_to_jsonl(rl_src, rl_dst)
    with open(rl_src) as f:
        rl_data = json.load(f)
    rl_stats = gather_stats(rl_data, "rl")
    print(f"  RL: {n_rl} examples → {rl_dst}")

    # --- Dataset Card ---
    print("Generating dataset card...")
    card = build_dataset_card(cpt_stats, sft_train_stats, sft_val_stats, rl_stats)
    card_path = os.path.join(DEST_DIR, "README.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card)
    print(f"  Dataset card → {card_path}")

    # --- .gitattributes for LFS ---
    gitattr_path = os.path.join(DEST_DIR, ".gitattributes")
    with open(gitattr_path, "w") as f:
        f.write("*.jsonl filter=lfs diff=lfs merge=lfs -text\n")
        f.write("*.json filter=lfs diff=lfs merge=lfs -text\n")
        f.write("*.parquet filter=lfs diff=lfs merge=lfs -text\n")
    print(f"  .gitattributes → {gitattr_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"HuggingFace Dataset Repository: {DEST_DIR}")
    print(f"{'='*60}")
    total_size = 0
    for root, dirs, files in os.walk(DEST_DIR):
        for fname in files:
            fpath = os.path.join(root, fname)
            sz = os.path.getsize(fpath)
            total_size += sz
            rel = os.path.relpath(fpath, DEST_DIR)
            print(f"  {rel:55s}  {sz/1024/1024:8.2f} MB")
    print(f"  {'TOTAL':55s}  {total_size/1024/1024:8.2f} MB")

    print(f"\nLoad with:")
    print(f'  from datasets import load_dataset')
    print(f'  ds = load_dataset("{DEST_DIR}", "sft")')
    print(f'  ds = load_dataset("{DEST_DIR}", "cpt")')
    print(f'  ds = load_dataset("{DEST_DIR}", "rl")')
    print(f"\nUpload to HF Hub:")
    print(f'  huggingface-cli upload <your-username>/flydsl-agent-dataset {DEST_DIR} .')
    print(f"\nDone!")


if __name__ == "__main__":
    main()
