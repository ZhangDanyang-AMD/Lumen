"""Clean hardware-feature mismatches from SFT and RL datasets.

Problem: 76% of gfx950 SFT samples mention features that only exist on
gfx1250 (wmma, tdm, mxfp4) or don't exist at all (sage_attention).
This teaches the model impossible code patterns.

Solution: For each sample mentioning a specific hardware target, remove or
replace references to features that hardware doesn't support.

Usage::

    python clean_hw_features.py \
        --sft /home/danyzhan/flydsl-agent-dataset/data/sft/train-00000-of-00001.jsonl \
        --rl /home/danyzhan/flydsl-agent-dataset/data/rl/train-00000-of-00001.jsonl \
        --output-sft /home/danyzhan/flydsl-agent-dataset/data/sft/train-00000-of-00001.jsonl \
        --output-rl /home/danyzhan/flydsl-agent-dataset/data/rl/train-00000-of-00001.jsonl \
        --metadata-dir /home/danyzhan/flydsl-agent-metadata
"""

import argparse
import json
import logging
import os
import re
import shutil
from collections import Counter

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Ground truth: which features are supported on which hardware
# Source: FlyDSL/lib/Dialect/FlyROCDL/{CDNA3,CDNA4,GFX11,GFX1250}/
HW_SUPPORTED_FEATURES = {
    "gfx942": {  # MI300X, CDNA3
        "mfma", "preshuffle", "pipeline", "swizzle_xor16", "shared_allocator",
        "double_buffer", "multi_wave", "split_k", "blockscale",
        "epilogue_fusion", "async_copy", "layout_algebra", "buffer_ops",
    },
    "gfx950": {  # MI350X, CDNA4
        "mfma", "preshuffle", "pipeline", "swizzle_xor16", "shared_allocator",
        "double_buffer", "multi_wave", "split_k", "blockscale",
        "epilogue_fusion", "async_copy", "layout_algebra", "buffer_ops",
        # gfx950 has wider LDS (160KB) and MFMA scale, but NOT wmma/tdm/mxfp4
    },
    "gfx1250": {  # MI450
        "wmma", "tdm", "mxfp4", "pipeline", "shared_allocator",
        "double_buffer", "split_k", "layout_algebra", "buffer_ops",
        # gfx1250 uses WMMA not MFMA, has TDM and FP4
    },
    "generic": {
        "pipeline", "shared_allocator", "double_buffer", "multi_wave",
        "split_k", "layout_algebra", "buffer_ops", "epilogue_fusion",
    },
}

# Features that should NEVER appear on certain hardware
HW_FORBIDDEN_FEATURES = {
    "gfx942": {"wmma", "tdm", "mxfp4", "sage_attention"},
    "gfx950": {"wmma", "tdm", "mxfp4", "sage_attention"},
    "gfx1250": {"mfma", "preshuffle", "swizzle_xor16"},  # gfx1250 doesn't use MFMA
    "generic": {"sage_attention"},
}

# Keywords in text that indicate specific features
FEATURE_KEYWORDS = {
    # Only match actual code usage, not comments/descriptions
    # Use tighter patterns to avoid false positives on hardware description text
    "wmma": [r"wmma_\w+\(", r"WmmaAtom", r"from.*wmma import", r"wmma_gemm"],
    "tdm": [r"tdm_ops\.", r"TdmCopy", r"from.*tdm import", r"tdm_load"],
    "mxfp4": [r"mxfp4_quant", r"mxfp4_dequant", r"fp4_gemm", r"wfp4", r"afp4"],
    "sage_attention": [r"sage_attention\(", r"SageAttention"],
    "mfma": [r"rocdl\.mfma", r"mfma_f32", r"MfmaAtom", r"mfma_epilog"],
    "preshuffle": [r"preshuffle_gemm\(", r"preshuffle_pipeline", r"PreshuffleConfig"],
}


def detect_hardware(text):
    """Detect which hardware a text refers to."""
    text_lower = text.lower()
    if "gfx950" in text_lower or "mi350" in text_lower or "mi355" in text_lower:
        return "gfx950"
    if "gfx942" in text_lower or "mi300" in text_lower or "mi308" in text_lower or "mi325" in text_lower:
        return "gfx942"
    if "gfx1250" in text_lower or "mi450" in text_lower:
        return "gfx1250"
    return None


def has_forbidden_feature(text, hw):
    """Check if text mentions features forbidden on given hardware."""
    forbidden = HW_FORBIDDEN_FEATURES.get(hw, set())
    found = set()
    for feat in forbidden:
        keywords = FEATURE_KEYWORDS.get(feat, [feat])
        for kw in keywords:
            if re.search(kw, text, re.IGNORECASE):
                found.add(feat)
                break
    return found


def clean_sft_record(record):
    """Clean a single SFT record. Returns (cleaned_record, action)."""
    text = " ".join(m["content"] for m in record["messages"])
    hw = detect_hardware(text)

    if hw is None:
        return record, "keep"

    forbidden = has_forbidden_feature(text, hw)
    if not forbidden:
        return record, "keep"

    # For assistant responses containing forbidden features:
    # Option 1: Drop the record entirely (safest)
    # Option 2: Keep if the forbidden feature is only in the user request
    #           (model needs to learn to NOT use these features)
    assistant_text = " ".join(m["content"] for m in record["messages"] if m["role"] == "assistant")
    assistant_forbidden = has_forbidden_feature(assistant_text, hw)

    if assistant_forbidden:
        return None, f"dropped:assistant_has_{','.join(assistant_forbidden)}_on_{hw}"
    else:
        return record, "keep"


def clean_rl_spec(spec):
    """Clean a single RL spec. Returns (cleaned_spec, action)."""
    hw = spec.get("hardware", "")
    forbidden = HW_FORBIDDEN_FEATURES.get(hw, set())
    features = set(spec.get("features", []))

    bad_features = features & forbidden
    if not bad_features:
        return spec, "keep"

    # Remove forbidden features from the spec
    cleaned = spec.copy()
    cleaned["features"] = [f for f in spec["features"] if f not in forbidden]
    if not cleaned["features"]:
        return None, f"dropped:all_features_forbidden_on_{hw}"

    return cleaned, f"cleaned:removed_{','.join(bad_features)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", required=True)
    parser.add_argument("--rl", required=True)
    parser.add_argument("--output-sft", required=True)
    parser.add_argument("--output-rl", required=True)
    parser.add_argument("--metadata-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.metadata_dir, exist_ok=True)

    # ---- Clean SFT data ----
    logger.info("Cleaning SFT data: %s", args.sft)
    with open(args.sft) as f:
        sft_records = [json.loads(l) for l in f if l.strip()]

    sft_clean = []
    sft_actions = Counter()
    for r in sft_records:
        cleaned, action = clean_sft_record(r)
        sft_actions[action] += 1
        if cleaned is not None:
            sft_clean.append(cleaned)

    logger.info("SFT: %d → %d (dropped %d)",
                len(sft_records), len(sft_clean), len(sft_records) - len(sft_clean))
    for action, cnt in sft_actions.most_common():
        logger.info("  %s: %d", action, cnt)

    # Backup and write
    backup = args.output_sft + ".v3.bak"
    if not os.path.exists(backup):
        shutil.copy2(args.sft, backup)
    with open(args.output_sft, "w") as f:
        for r in sft_clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- Clean RL data ----
    logger.info("Cleaning RL data: %s", args.rl)
    with open(args.rl) as f:
        rl_records = [json.loads(l) for l in f if l.strip()]

    rl_clean = []
    rl_actions = Counter()
    for r in rl_records:
        cleaned, action = clean_rl_spec(r)
        rl_actions[action] += 1
        if cleaned is not None:
            rl_clean.append(cleaned)

    logger.info("RL: %d → %d (dropped %d, cleaned %d)",
                len(rl_records), len(rl_clean),
                len(rl_records) - len(rl_clean),
                sum(v for k, v in rl_actions.items() if k.startswith("cleaned")))
    for action, cnt in rl_actions.most_common():
        logger.info("  %s: %d", action, cnt)

    # Backup and write
    rl_backup = args.output_rl + ".pre_clean.bak"
    if not os.path.exists(rl_backup):
        shutil.copy2(args.rl, rl_backup)
    with open(args.output_rl, "w") as f:
        for r in rl_clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save stats
    stats = {
        "sft_before": len(sft_records),
        "sft_after": len(sft_clean),
        "sft_dropped": len(sft_records) - len(sft_clean),
        "sft_actions": dict(sft_actions),
        "rl_before": len(rl_records),
        "rl_after": len(rl_clean),
        "rl_actions": dict(rl_actions),
    }
    with open(os.path.join(args.metadata_dir, "hw_feature_clean_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Stats saved to %s/hw_feature_clean_stats.json", args.metadata_dir)


if __name__ == "__main__":
    main()
