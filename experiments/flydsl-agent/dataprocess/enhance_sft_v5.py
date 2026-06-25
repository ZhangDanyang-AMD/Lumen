"""SFT v5 Enhancement — fix 3 root causes from RFT failure analysis.

1. Add real gfx950 kernel code from FlyDSL repo (pipeline/swizzle/preshuffle)
2. Add "no markdown" emphasis data to stop ``` wrapping
3. Add correct API type system examples (fx.Tensor, fx.Constexpr, not fx.DeviceArray)

Usage::
    python enhance_sft_v5.py \
        --sft-data /home/danyzhan/flydsl-agent-dataset/data/sft/train-00000-of-00001.jsonl \
        --flydsl-dir /home/danyzhan/FlyDSL \
        --output /home/danyzhan/flydsl-agent-dataset/data/sft/train-00000-of-00001.jsonl \
        --metadata-dir /home/danyzhan/flydsl-agent-metadata
"""

import argparse
import json
import logging
import os
import random
import re
import shutil

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT_V5 = (
    "You are a FlyDSL GPU kernel programming expert for AMD Instinct GPUs. "
    "Always output raw Python code directly — never wrap in markdown code blocks. "
    "Use the standard FlyDSL imports:\n"
    "  import flydsl.compiler as flyc\n"
    "  import flydsl.expr as fx\n"
    "Use @flyc.kernel for device kernels, @flyc.jit for launch wrappers. "
    "Use fx.Tensor for tensor arguments, fx.Constexpr[int] for compile-time constants."
)

# gfx1250-only patterns — skip these kernels
GFX1250_ONLY = re.compile(r"gfx1250|wmma_gemm|moe_gemm_2stage_wmma|gemm_common_gfx1250|gemm_fp8fp4_gfx1250|rdna")

# Operator detection
OP_PATTERNS = {
    "gemm": r"gemm|matmul|preshuffle_gemm|splitk_hgemm|small_m_hgemm",
    "flash_attn": r"flash_attn",
    "moe": r"moe_gemm|moe_blockscale|mixed_moe|moe_sorting",
    "softmax": r"softmax",
    "rmsnorm": r"rmsnorm",
    "layernorm": r"layernorm",
    "rope": r"rope",
    "paged_attn": r"pa_decode",
    "mla": r"mla_fwd",
    "allreduce": r"all_reduce",
    "custom": r"dispatch_combine|custom",
}


def detect_op(filename):
    for op, pat in OP_PATTERNS.items():
        if re.search(pat, filename, re.IGNORECASE):
            return op
    return "custom"


def detect_features(code):
    feats = []
    if re.search(r"rocdl\.mfma|mfma_f32|MfmaAtom", code): feats.append("mfma")
    if "preshuffle" in code: feats.append("preshuffle")
    if re.search(r"lds_stage|pipeline|num_stages", code): feats.append("pipeline")
    if "swizzle_xor" in code: feats.append("swizzle")
    if re.search(r"SmemAllocator|SharedAllocator", code): feats.append("smem")
    if "split_k" in code: feats.append("split_k")
    if "blockscale" in code: feats.append("blockscale")
    return feats


def generate_kernel_sft_pairs(flydsl_dir):
    """Generate SFT pairs from real gfx950-compatible FlyDSL kernels."""
    kernel_dir = os.path.join(flydsl_dir, "kernels")
    pairs = []

    for fname in sorted(os.listdir(kernel_dir)):
        if not fname.endswith(".py") or fname.startswith("__"):
            continue
        if GFX1250_ONLY.search(fname):
            continue

        path = os.path.join(kernel_dir, fname)
        code = open(path).read()

        # Skip if no FlyDSL patterns
        if not re.search(r"@flyc\.(kernel|jit)|import flydsl", code):
            continue

        op = detect_op(fname)
        features = detect_features(code)
        if not features:
            continue

        feat_str = ", ".join(features)
        lines = len(code.split("\n"))

        # Truncate very long kernels to first meaningful section
        if lines > 500:
            # Keep imports + first kernel function + first jit function
            truncated_lines = []
            in_func = False
            func_count = 0
            for line in code.split("\n"):
                truncated_lines.append(line)
                if line.startswith("@flyc.") or line.startswith("def "):
                    in_func = True
                if in_func and line.strip() == "" and func_count > 0:
                    func_count += 1
                    if func_count >= 3:
                        truncated_lines.append("# ... (truncated for training)")
                        break
                if re.match(r"^(def |@flyc)", line):
                    func_count += 1
            code_for_sft = "\n".join(truncated_lines[:300])
        else:
            code_for_sft = code

        # Style 1: Precise technical
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_V5},
                {"role": "user", "content":
                    f"Write a FlyDSL {op} kernel for AMD MI350X (gfx950).\n"
                    f"Features: {feat_str}\n"
                    f"Use @flyc.kernel decorator, fx.Tensor for tensor args, "
                    f"and SmemAllocator for shared memory.\n"
                    f"Output raw Python code, no markdown."},
                {"role": "assistant", "content": code_for_sft},
            ],
            "source": "gfx950_kernel_real",
            "metadata": {"source_path": f"kernels/{fname}", "operator": op,
                         "hardware": "gfx950", "features": features},
        })

        # Style 2: Natural
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_V5},
                {"role": "user", "content":
                    f"Implement a high-performance {op} kernel targeting "
                    f"AMD Instinct MI350X (gfx950) using FlyDSL with {feat_str}."},
                {"role": "assistant", "content": code_for_sft},
            ],
            "source": "gfx950_kernel_real",
            "metadata": {"source_path": f"kernels/{fname}", "operator": op,
                         "hardware": "gfx950", "features": features},
        })

    return pairs


def generate_no_markdown_pairs():
    """Teach model to NOT wrap code in markdown blocks."""
    pairs = []
    ops = ["gemm", "softmax", "rmsnorm", "vec_add", "layernorm", "rope"]

    for op in ops:
        # "Don't use markdown" instruction
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_V5},
                {"role": "user", "content":
                    f"Write a FlyDSL {op} kernel for gfx950. "
                    f"Output the Python source code directly, without markdown formatting."},
                {"role": "assistant", "content":
                    f"import flydsl.compiler as flyc\n"
                    f"import flydsl.expr as fx\n\n\n"
                    f"@flyc.kernel\n"
                    f"def {op}_kernel(\n"
                    f"    A: fx.Tensor,\n"
                    f"    B: fx.Tensor,\n"
                    f"    C: fx.Tensor,\n"
                    f"    N: fx.Constexpr[int],\n"
                    f"):\n"
                    f"    bid = fx.block_idx.x\n"
                    f"    tid = fx.thread_idx.x\n"
                    f"    # TODO: implement {op} logic\n"
                    f"    pass\n\n\n"
                    f"@flyc.jit\n"
                    f"def {op}_launch(A, B, C, N):\n"
                    f"    block_dim = 256\n"
                    f"    grid = (N + block_dim - 1) // block_dim\n"
                    f"    {op}_kernel[(grid,), (block_dim,)](A, B, C, N)\n"},
            ],
            "source": "no_markdown_template",
            "metadata": {"operator": op},
        })

    return pairs


def generate_api_correction_pairs():
    """Teach correct FlyDSL type system — fx.Tensor not fx.DeviceArray."""
    corrections = [
        ("fx.DeviceArray(3, dtype=torch.float32)", "fx.Tensor"),
        ("fx.Arg('N')", "fx.Constexpr[int]"),
        ("fx.RuntimeArg(dtype=torch.int64)", "fx.Constexpr[int]"),
        ("flydsl.types.Tensor", "fx.Tensor"),
        ("ft.Tensor", "fx.Tensor"),
        ("fx.InputTensor", "fx.Tensor"),
        ("fx.OutputTensor", "fx.Tensor"),
        ("flyc.SmemAllocator", "from flydsl.utils.smem_allocator import SmemAllocator"),
        ("fx.shared_memory(size)", "SmemAllocator(total_bytes)"),
    ]

    pairs = []
    for wrong, correct in corrections:
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_V5},
                {"role": "user", "content":
                    f"I'm getting an error with `{wrong}` in my FlyDSL kernel. "
                    f"What's the correct type?"},
                {"role": "assistant", "content":
                    f"The correct FlyDSL type is `{correct}`.\n\n"
                    f"Standard kernel signature:\n"
                    f"import flydsl.compiler as flyc\n"
                    f"import flydsl.expr as fx\n\n"
                    f"@flyc.kernel\n"
                    f"def my_kernel(\n"
                    f"    A: fx.Tensor,          # tensor argument\n"
                    f"    B: fx.Tensor,\n"
                    f"    N: fx.Constexpr[int],   # compile-time constant\n"
                    f"):\n"
                    f"    pass"},
            ],
            "source": "api_type_correction",
            "metadata": {"wrong": wrong, "correct": correct},
        })

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-data", required=True)
    parser.add_argument("--flydsl-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.metadata_dir, exist_ok=True)

    # Load existing v4 data
    with open(args.sft_data) as f:
        existing = [json.loads(l) for l in f if l.strip()]
    logger.info("Existing SFT data: %d samples", len(existing))

    # Generate new data
    kernel_pairs = generate_kernel_sft_pairs(args.flydsl_dir)
    logger.info("Real gfx950 kernel pairs: %d", len(kernel_pairs))

    no_md_pairs = generate_no_markdown_pairs()
    logger.info("No-markdown pairs: %d", len(no_md_pairs))

    api_pairs = generate_api_correction_pairs()
    logger.info("API correction pairs: %d", len(api_pairs))

    # Save generated pairs
    all_new = kernel_pairs + no_md_pairs + api_pairs
    gen_path = os.path.join(args.metadata_dir, "sft_v5_generated.jsonl")
    with open(gen_path, "w") as f:
        for p in all_new:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info("Saved %d new pairs to %s", len(all_new), gen_path)

    # Combine: existing + new (kernel pairs repeated 3x for emphasis)
    combined = existing + kernel_pairs * 3 + no_md_pairs * 3 + api_pairs * 2
    random.shuffle(combined)

    # Backup
    backup = args.output + ".v4.bak"
    if not os.path.exists(backup):
        shutil.copy2(args.sft_data, backup)

    with open(args.output, "w") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    src = Counter(r.get("source", "?") for r in combined)
    logger.info("v5 output: %d samples → %s", len(combined), args.output)
    for s, c in src.most_common(10):
        logger.info("  %s: %d", s, c)

    # Check kernel feature coverage
    kernel_feat = Counter()
    for p in kernel_pairs:
        for f in p["metadata"].get("features", []):
            kernel_feat[f] += 1
    logger.info("New kernel feature coverage:")
    for f, c in kernel_feat.most_common():
        logger.info("  %s: %d pairs", f, c)


if __name__ == "__main__":
    main()
