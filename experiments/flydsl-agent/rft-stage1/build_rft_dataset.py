"""RFT Step 4: Build RFT training dataset from verified candidates.

Converts verified candidates into ChatML SFT format and merges with
existing SFT data for short fine-tuning.

Usage::

    python build_rft_dataset.py \
        --verified /home/danyzhan/rft-results/verified.jsonl \
        --sft-data /home/danyzhan/flydsl-agent-dataset/data/sft/train-00000-of-00001.jsonl \
        --output /home/danyzhan/flydsl-agent-dataset/data/rft/train-00000-of-00001.jsonl
"""

import argparse
import json
import logging
import os
import random

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a FlyDSL GPU kernel programming expert. You write compilable, "
    "correct, high-performance GPU kernels using the FlyDSL framework for "
    "AMD Instinct GPUs. Always use @flyc.kernel decorator, fx.* expression "
    "API, and follow FlyDSL coding patterns."
)

OP_DESCRIPTIONS = {
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
    "quant": "quantization kernel",
    "custom": "custom GPU kernel",
}

HW_DESCRIPTIONS = {
    "gfx942": "MI300X", "gfx950": "MI350X", "generic": "AMD GPU",
}


def verified_to_sft(record):
    """Convert a verified record to ChatML SFT pairs (one per candidate)."""
    op = record["operator"]
    hw = record.get("hardware", "generic")
    features = ", ".join(record.get("features", [])) or "standard optimization"
    op_desc = OP_DESCRIPTIONS.get(op, op)
    hw_desc = HW_DESCRIPTIONS.get(hw, hw)

    pairs = []
    for cand in record.get("verified_candidates", []):
        code = cand["code"]
        style = cand.get("style", "precise")

        if style == "precise":
            user_msg = (
                f"Write a FlyDSL {op} kernel for AMD {hw}.\n"
                f"Features: {features}\n"
                f"Use @flyc.kernel decorator and fx.* expression API."
            )
        elif style == "natural":
            user_msg = (
                f"Implement a high-performance {op_desc} kernel "
                f"targeting AMD Instinct {hw_desc} ({hw}) using FlyDSL."
            )
        else:
            user_msg = (
                f"Write an optimized FlyDSL {op} kernel for {hw} "
                f"with features: {features}."
            )

        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": code},
            ],
            "source": "rft_verified",
            "metadata": {
                "spec_id": record.get("spec_id", ""),
                "operator": op,
                "hardware": hw,
                "style": style,
            },
        })
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified", required=True)
    parser.add_argument("--sft-data", default=None, help="Existing SFT data to merge with")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rft-repeat", type=int, default=2,
                        help="Repeat RFT data N times to boost weight")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load verified candidates
    with open(args.verified) as f:
        records = [json.loads(l) for l in f if l.strip()]
    logger.info("Loaded %d specs with verified candidates", len(records))

    # Convert to SFT format
    rft_pairs = []
    for rec in records:
        rft_pairs.extend(verified_to_sft(rec))
    logger.info("Generated %d RFT training pairs", len(rft_pairs))

    # Repeat RFT pairs to boost weight
    rft_boosted = rft_pairs * args.rft_repeat
    logger.info("RFT pairs after %dx repeat: %d", args.rft_repeat, len(rft_boosted))

    # Optionally merge with existing SFT data
    if args.sft_data:
        with open(args.sft_data) as f:
            sft_data = [json.loads(l) for l in f if l.strip()]
        logger.info("Loaded %d existing SFT samples", len(sft_data))
        combined = sft_data + rft_boosted
    else:
        combined = rft_boosted

    random.shuffle(combined)

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    src_dist = Counter(r.get("source", "?") for r in combined)
    logger.info("Output: %d samples → %s", len(combined), args.output)
    logger.info("  Source distribution:")
    for s, c in src_dist.most_common(10):
        logger.info("    %s: %d", s, c)


if __name__ == "__main__":
    main()
