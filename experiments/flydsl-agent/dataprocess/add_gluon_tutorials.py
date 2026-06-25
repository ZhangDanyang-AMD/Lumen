"""Add gfx950 Gluon GEMM optimization tutorials to SFT dataset.

Generates 4 types of SFT pairs from the v0→v9 progressive tutorial:
  1. Full kernel with optimization explanation (kernel + README)
  2. Step-by-step optimization (vN → vN+1 upgrade explanation)
  3. Optimization Q&A (why did we make this change?)
  4. Performance analysis (what's the bottleneck, how to fix?)

Usage::
    python add_gluon_tutorials.py \
        --tutorial-dir /home/danyzhan/gfx950-gluon-tutorials/kernels/gemm/a16w16 \
        --sft-data /home/danyzhan/flydsl-agent-dataset/data/sft/train-00000-of-00001.jsonl \
        --output /home/danyzhan/flydsl-agent-dataset/data/sft/train-00000-of-00001.jsonl \
        --metadata-dir /home/danyzhan/flydsl-agent-metadata
"""

import argparse
import json
import logging
import os
import random
import shutil

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a GPU kernel optimization expert for AMD Instinct MI350X (gfx950). "
    "You explain kernel optimizations step by step, covering memory hierarchy, "
    "layout design, latency hiding, and instruction scheduling. "
    "Output raw code directly, never wrap in markdown code blocks."
)

VERSIONS = [
    ("v0_naive", "Naive baseline", "Explicit layouts, correctness-first MFMA kernel"),
    ("v1_buffer_load", "Buffer load", "Hardware OOB checking, branch elimination"),
    ("v2_async_copy", "Async copy", "Direct-to-LDS async copy, eliminates register staging"),
    ("v3_lds", "LDS layout", "LDS layout design: swizzling vs padding for bank conflict"),
    ("v4_global_prefetch", "Global prefetch", "2-stage pipeline, double buffering"),
    ("v5_local_prefetch", "Local prefetch", "3-stage pipeline with local prefetch"),
    ("v6_loop_unroll", "Loop unroll", "Loop unrolling to eliminate copy overhead"),
    ("v7_sliceN", "Slice N", "N-slicing for register pressure reduction"),
    ("v8_sliceMN", "Slice M+N", "M+N slicing, buffer load throughput analysis"),
    ("v9_beyond_hotloop", "Beyond hotloop", "L2 cache locality via XCD-aware PID remapping"),
]


def load_version(tutorial_dir, version_name):
    vdir = os.path.join(tutorial_dir, version_name)
    kernel_path = os.path.join(vdir, "matmul_kernel.py")
    readme_path = os.path.join(vdir, "README.md")
    kernel = open(kernel_path).read() if os.path.exists(kernel_path) else ""
    readme = open(readme_path).read() if os.path.exists(readme_path) else ""
    return kernel, readme


def generate_pairs(tutorial_dir):
    pairs = []
    main_readme = open(os.path.join(tutorial_dir, "README.md")).read()

    for i, (vname, short, desc) in enumerate(VERSIONS):
        kernel, readme = load_version(tutorial_dir, vname)
        if not kernel:
            continue

        # Type 1: Full kernel + explanation
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"Write a FP16 GEMM kernel for AMD MI350X (gfx950) using Gluon/FlyDSL.\n"
                    f"Optimization level: {short} ({desc}).\n"
                    f"Explain the key optimization decisions."},
                {"role": "assistant", "content":
                    f"{kernel}\n\n"
                    f"# Optimization Notes\n"
                    f"# {short}: {desc}\n"
                    f"# See detailed explanation below.\n\n"
                    f"{readme[:2000]}"},
            ],
            "source": "gluon_tutorial_kernel",
            "metadata": {"version": vname, "level": i, "optimization": short},
        })

        # Type 2: Step-by-step upgrade (vN → vN+1)
        if i > 0:
            prev_name, prev_short, prev_desc = VERSIONS[i - 1]
            prev_kernel, _ = load_version(tutorial_dir, prev_name)
            if prev_kernel:
                pairs.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content":
                            f"I have this GEMM kernel ({prev_short}):\n\n"
                            f"{prev_kernel[:1500]}\n\n"
                            f"How do I optimize it further? "
                            f"The next optimization target is: {desc}."},
                        {"role": "assistant", "content":
                            f"Here's the optimized version ({short}: {desc}):\n\n"
                            f"{kernel}\n\n"
                            f"Key changes:\n{readme[:1500]}"},
                    ],
                    "source": "gluon_tutorial_upgrade",
                    "metadata": {"from": prev_name, "to": vname, "optimization": short},
                })

        # Type 3: Optimization Q&A from README
        if readme:
            pairs.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content":
                        f"Explain the '{short}' optimization technique for "
                        f"GEMM kernels on AMD MI350X (gfx950). "
                        f"What problem does it solve and how?"},
                    {"role": "assistant", "content": readme[:3000]},
                ],
                "source": "gluon_tutorial_qa",
                "metadata": {"version": vname, "optimization": short},
            })

    # Type 4: Full optimization journey overview
    pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                "Describe the complete optimization journey for an FP16 GEMM kernel "
                "on AMD MI350X (gfx950), from naive baseline to near-optimal performance."},
            {"role": "assistant", "content": main_readme[:4000]},
        ],
        "source": "gluon_tutorial_overview",
        "metadata": {"type": "full_journey"},
    })

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tutorial-dir", required=True)
    parser.add_argument("--sft-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.metadata_dir, exist_ok=True)

    pairs = generate_pairs(args.tutorial_dir)
    logger.info("Generated %d tutorial SFT pairs", len(pairs))

    # Save to metadata
    gen_path = os.path.join(args.metadata_dir, "sft_v5_gluon_tutorials.jsonl")
    with open(gen_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info("Saved to %s", gen_path)

    # Merge with existing SFT data (tutorials repeated 2x)
    with open(args.sft_data) as f:
        existing = [json.loads(l) for l in f if l.strip()]

    combined = existing + pairs * 2
    random.shuffle(combined)

    backup = args.output + ".v5pre.bak"
    if not os.path.exists(backup):
        shutil.copy2(args.sft_data, backup)

    with open(args.output, "w") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    src = Counter(r.get("source", "?") for r in combined)
    logger.info("Output: %d samples", len(combined))
    tutorial_count = sum(c for s, c in src.items() if "gluon" in s)
    logger.info("  Tutorial data: %d (%.1f%%)", tutorial_count, tutorial_count / len(combined) * 100)


if __name__ == "__main__":
    main()
