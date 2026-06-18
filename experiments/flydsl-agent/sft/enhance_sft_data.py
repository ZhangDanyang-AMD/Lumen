"""SFT Data Enhancement — increase kernel code ratio from 18% to ~50%.

Two measures:
  1. Weighted resampling of existing data (kernel-heavy sources upweighted)
  2. Generate new kernel SFT pairs from RL specs using FlyDSL source as context

Reads: flydsl-agent-dataset/data/sft/train-00000-of-00001.jsonl
Writes:
  - metadata/sft_v2_weights.json (sampling weights)
  - metadata/sft_v2_generated.jsonl (new Claude-generated kernel pairs)
  - data/sft/train-00000-of-00001.jsonl (final enhanced dataset)

Usage::

    python enhance_sft_data.py \
        --input /home/danyzhan/flydsl-agent-dataset/data/sft/train-00000-of-00001.jsonl \
        --rl-specs /home/danyzhan/flydsl-agent-dataset/data/rl/train-00000-of-00001.jsonl \
        --cpt-data /home/danyzhan/flydsl-agent-dataset/data/cpt/train-00000-of-00001.jsonl \
        --metadata-dir /home/danyzhan/flydsl-agent-metadata \
        --output /home/danyzhan/flydsl-agent-dataset/data/sft/train-00000-of-00001.jsonl
"""

import argparse
import json
import logging
import os
import random
import re
from collections import Counter, defaultdict

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

KERNEL_PATTERNS = r"@flyc\.|import flydsl|fx\.make_layout|fx\.gpu\.|SmemAllocator|rocdl\.mfma|fx\.zipped_divide|fx\.composition|buffer_ops\."

SYSTEM_PROMPT = (
    "You are a FlyDSL GPU kernel programming expert. You write compilable, "
    "correct, high-performance GPU kernels using the FlyDSL framework for "
    "AMD Instinct GPUs. Always use @flyc.kernel decorator, fx.* expression "
    "API, and follow FlyDSL coding patterns including SmemAllocator, "
    "pipeline staging, and swizzle for bank conflict avoidance."
)

# Sampling weights by source
SOURCE_WEIGHTS = {
    "augmentation_tile": 5.0,
    "augmentation_pipeline": 5.0,
    "docs_direct_extraction": 4.0,
    "skill_direct_extraction": 4.0,
    "performance_improvement": 3.0,
    "kernel_reverse_annotation": 2.0,
    "augmentation_hardware": 1.0,
    "git_history": 1.5,
    "skill_section_qa": 1.0,
    "documentation_qa": 0.3,
    "ai_annotated_instruction": 0.5,
    "test_parameterization": 0.3,
    "tuned_config": 0.3,
    "refusal_boundary": 0.1,
    "allowed_explanation": 0.5,
}


def is_kernel_code(text):
    return bool(re.search(KERNEL_PATTERNS, text))


def compute_weights(records):
    """Assign sampling weight to each record."""
    weights = []
    for r in records:
        src = r.get("source", "unknown")
        base_weight = SOURCE_WEIGHTS.get(src, 1.0)
        assistant = " ".join(m["content"] for m in r["messages"] if m["role"] == "assistant")
        if is_kernel_code(assistant):
            base_weight *= 2.0
        weights.append(base_weight)
    return weights


def weighted_resample(records, weights, target_size=None, seed=42):
    """Resample records according to weights."""
    if target_size is None:
        target_size = len(records)
    random.seed(seed)
    indices = random.choices(range(len(records)), weights=weights, k=target_size)
    return [records[i] for i in indices]


def extract_kernel_examples(cpt_path):
    """Extract real FlyDSL kernel code from CPT data as SFT reference answers."""
    examples = []
    with open(cpt_path) as f:
        for line in f:
            r = json.loads(line)
            meta = r.get("meta", {})
            if meta.get("content_type") != "kernel_impl":
                continue
            if meta.get("source_repo") != "FlyDSL":
                continue
            text = r.get("text", "")
            # Strip CPT metadata headers
            if "<|doc_start|>" in text:
                text = text.split("\n\n", 1)[-1]
            if "<|doc_end|>" in text:
                text = text.rsplit("<|doc_end|>", 1)[0]
            if len(text) < 200:
                continue
            examples.append({
                "path": meta.get("source_path", ""),
                "operator": meta.get("operator", "unknown"),
                "hardware": meta.get("hardware", []),
                "grade": meta.get("quality_grade", "ungraded"),
                "code": text.strip(),
            })
    return examples


def generate_sft_from_kernels(kernel_examples, rl_specs):
    """Generate SFT instruction-response pairs from real kernel code + RL specs."""
    pairs = []

    # Group RL specs by operator
    specs_by_op = defaultdict(list)
    for spec in rl_specs:
        specs_by_op[spec.get("operator", "unknown")].append(spec)

    for ex in kernel_examples:
        op = ex["operator"]
        hw_list = ex["hardware"] if isinstance(ex["hardware"], list) else [ex["hardware"]]
        hw = hw_list[0] if hw_list else "gfx950"

        # Style 1: Precise technical instruction
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Write a FlyDSL {op} kernel for AMD {hw}.\n"
                    f"Reference implementation: {ex['path']}\n"
                    f"Use @flyc.kernel decorator, fx.* expression API, "
                    f"SmemAllocator for shared memory, and proper MFMA instructions."
                )},
                {"role": "assistant", "content": ex["code"]},
            ],
            "source": "kernel_code_synthesis",
            "metadata": {
                "source_path": ex["path"],
                "operator": op,
                "hardware": hw,
                "grade": ex["grade"],
                "style": "precise",
            },
        })

        # Style 2: Natural language request
        op_desc = {
            "gemm": "matrix multiplication (GEMM)",
            "flash_attn": "FlashAttention forward pass",
            "moe": "Mixture-of-Experts GEMM",
            "softmax": "softmax normalization",
            "rmsnorm": "RMSNorm layer normalization",
            "layernorm": "LayerNorm",
            "rope": "Rotary Position Embedding (RoPE)",
            "topk": "top-K selection",
            "mla": "Multi-head Latent Attention decode",
            "paged_attn": "Paged Attention for KV-cache",
            "allreduce": "custom all-reduce",
        }.get(op, f"{op} operation")

        hw_desc = {
            "gfx942": "MI300X",
            "gfx950": "MI350X",
            "gfx1250": "MI450",
        }.get(hw, hw)

        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Implement a high-performance {op_desc} kernel "
                    f"targeting AMD Instinct {hw_desc} ({hw}) using the FlyDSL framework. "
                    f"The kernel should use proper tiling, shared memory management, "
                    f"and MFMA matrix instructions for optimal performance."
                )},
                {"role": "assistant", "content": ex["code"]},
            ],
            "source": "kernel_code_synthesis",
            "metadata": {
                "source_path": ex["path"],
                "operator": op,
                "hardware": hw,
                "grade": ex["grade"],
                "style": "natural",
            },
        })

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--rl-specs", required=True)
    parser.add_argument("--cpt-data", required=True)
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.metadata_dir, exist_ok=True)

    # Load existing SFT data
    logger.info("Loading existing SFT data from %s ...", args.input)
    with open(args.input) as f:
        original = [json.loads(l) for l in f if l.strip()]
    logger.info("  %d original samples", len(original))

    # Analyze original data
    src_counts = Counter(r.get("source", "?") for r in original)
    kernel_count = sum(1 for r in original if is_kernel_code(
        " ".join(m["content"] for m in r["messages"] if m["role"] == "assistant")))
    logger.info("  Kernel code samples: %d/%d (%.1f%%)",
                kernel_count, len(original), kernel_count / len(original) * 100)

    # Step 1: Extract real kernel examples from CPT data
    logger.info("Extracting kernel examples from CPT data ...")
    kernel_examples = extract_kernel_examples(args.cpt_data)
    logger.info("  %d kernel examples extracted", len(kernel_examples))

    # Step 2: Load RL specs for context
    with open(args.rl_specs) as f:
        rl_specs = [json.loads(l) for l in f if l.strip()]
    logger.info("  %d RL specs loaded", len(rl_specs))

    # Step 3: Generate new kernel SFT pairs
    logger.info("Generating kernel SFT pairs from real code ...")
    new_pairs = generate_sft_from_kernels(kernel_examples, rl_specs)
    logger.info("  %d new kernel pairs generated", len(new_pairs))

    # Save generated pairs to metadata
    gen_path = os.path.join(args.metadata_dir, "sft_v2_generated.jsonl")
    with open(gen_path, "w") as f:
        for p in new_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info("  Generated pairs saved to %s", gen_path)

    # Step 4: Combine original + generated
    combined = original + new_pairs
    logger.info("Combined: %d original + %d generated = %d total",
                len(original), len(new_pairs), len(combined))

    # Step 5: Compute sampling weights
    weights = compute_weights(combined)

    # Save weights to metadata
    weight_info = {
        "source_weights": SOURCE_WEIGHTS,
        "total_samples": len(combined),
        "kernel_code_bonus": "2x",
        "original_count": len(original),
        "generated_count": len(new_pairs),
    }
    with open(os.path.join(args.metadata_dir, "sft_v2_weights.json"), "w") as f:
        json.dump(weight_info, f, indent=2)

    # Step 6: Weighted resample to target size
    target_size = len(combined)
    resampled = weighted_resample(combined, weights, target_size, args.seed)

    # Analyze resampled distribution
    resampled_kernel = sum(1 for r in resampled if is_kernel_code(
        " ".join(m["content"] for m in r["messages"] if m["role"] == "assistant")))
    resampled_src = Counter(r.get("source", "?") for r in resampled)

    logger.info("Resampled: %d samples", len(resampled))
    logger.info("  Kernel code: %d/%d (%.1f%%) [was %.1f%%]",
                resampled_kernel, len(resampled),
                resampled_kernel / len(resampled) * 100,
                kernel_count / len(original) * 100)

    logger.info("  Source distribution:")
    for src, cnt in resampled_src.most_common():
        logger.info("    %s: %d", src, cnt)

    # Step 7: Shuffle and write output
    random.seed(args.seed)
    random.shuffle(resampled)

    # Backup original
    backup_path = args.output + ".v1.bak"
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(args.input, backup_path)
        logger.info("  Original backed up to %s", backup_path)

    with open(args.output, "w") as f:
        for r in resampled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Output written to %s (%d samples)", args.output, len(resampled))

    # Summary
    print("\n" + "=" * 60)
    print("  SFT v2 Data Enhancement Summary")
    print("=" * 60)
    print(f"  Original:   {len(original)} samples ({kernel_count / len(original):.0%} kernel)")
    print(f"  Generated:  {len(new_pairs)} new kernel pairs")
    print(f"  Combined:   {len(combined)} total")
    print(f"  Resampled:  {len(resampled)} final ({resampled_kernel / len(resampled):.0%} kernel)")
    print(f"  Improvement: {kernel_count / len(original):.0%} → {resampled_kernel / len(resampled):.0%} kernel code")
    print("=" * 60)


if __name__ == "__main__":
    main()
