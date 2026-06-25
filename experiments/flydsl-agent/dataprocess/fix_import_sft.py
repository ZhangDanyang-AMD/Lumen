"""Generate import-focused SFT data to fix FlyDSL API hallucination.

Problem: SFT v2 model generates wrong imports (81% hallucinated).
Solution: Create short, focused SFT pairs that drill the correct import pattern.

Three types of data:
  1. Import template pairs — "write the standard FlyDSL imports" → correct imports
  2. Import-prefixed kernel pairs — kernel code with emphasized correct imports
  3. Import correction pairs — wrong import → corrected import

Usage::

    python fix_import_sft.py \
        --sft-data /home/danyzhan/flydsl-agent-dataset/data/sft/train-00000-of-00001.jsonl \
        --output /home/danyzhan/flydsl-agent-dataset/data/sft/train-00000-of-00001.jsonl \
        --metadata-dir /home/danyzhan/flydsl-agent-metadata
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
    "AMD Instinct GPUs."
)

CORRECT_IMPORT_BLOCK = """import torch

import flydsl.compiler as flyc
import flydsl.expr as fx"""

CORRECT_IMPORT_EXTENDED = """import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr"""

WRONG_IMPORTS = [
    ("from flydsl.allocators import SharedAllocator", "from flydsl.utils.smem_allocator import SmemAllocator"),
    ("from flydsl.core.fx import fx_make_variable_value_type_impl", "import flydsl.expr as fx"),
    ("import flydsl as fx", "import flydsl.expr as fx"),
    ("from flydsl.gpu.wmma.mma.mma_config.mma_config import WmmaMmaConfig", "from flydsl.expr.rocdl import cluster"),
    ("import flydsl as flyc", "import flydsl.compiler as flyc"),
    ("from flydsl import fx", "import flydsl.expr as fx"),
    ("from flydsl import flyc", "import flydsl.compiler as flyc"),
    ("from flydsl.smem_allocator import SmemAllocator", "from flydsl.utils.smem_allocator import SmemAllocator"),
    ("from flydsl.ops import repeat as fx_repeat", "# No such module. Use fx.* APIs from flydsl.expr"),
    ("import flydsl.fx as fx", "import flydsl.expr as fx"),
    ("from flydsl.kernel import kernel", "import flydsl.compiler as flyc  # use @flyc.kernel"),
    ("from flydsl.layout import *", "from flydsl.expr import fx  # layouts via fx.make_layout()"),
    ("import flydsl.core as fx", "import flydsl.expr as fx"),
    ("import flydsl.core as flyc", "import flydsl.compiler as flyc"),
    ("from flydsl.ir import *", "from flydsl._mlir import ir  # only if needed for low-level IR"),
]

OPERATORS = ["gemm", "softmax", "rmsnorm", "layernorm", "rope", "flash_attn",
             "moe", "topk", "paged_attn", "mla", "allreduce", "vec_add"]
HARDWARE = ["gfx942", "gfx950"]


def generate_import_template_pairs():
    """Type 1: Direct import template questions."""
    pairs = []

    # Basic import question
    for op in OPERATORS:
        for hw in HARDWARE:
            pairs.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content":
                        f"What are the correct imports for writing a FlyDSL {op} kernel "
                        f"targeting {hw}? Show the standard import block."},
                    {"role": "assistant", "content":
                        f"Here are the standard FlyDSL imports for a {op} kernel on {hw}:\n\n"
                        f"```python\n{CORRECT_IMPORT_EXTENDED}\n```\n\n"
                        f"Key points:\n"
                        f"- `flyc` is the compiler module: use `@flyc.kernel` and `@flyc.jit`\n"
                        f"- `fx` is the expression API: use `fx.make_layout()`, `fx.block_idx`, etc.\n"
                        f"- SmemAllocator is in `flydsl.utils.smem_allocator`, not `flydsl.allocators`\n"
                        f"- ROCDL ops are in `flydsl.expr.rocdl`, not `flydsl.gpu.wmma`"},
                ],
                "source": "import_fix_template",
                "metadata": {"operator": op, "hardware": hw},
            })

    # Skeleton question
    for op in OPERATORS[:6]:
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"Write the skeleton of a FlyDSL {op} kernel with correct imports, "
                    f"@flyc.kernel decorator, and @flyc.jit launch wrapper. No implementation needed."},
                {"role": "assistant", "content":
                    f"```python\n{CORRECT_IMPORT_BLOCK}\n\n\n"
                    f"@flyc.kernel\n"
                    f"def {op}_kernel(\n"
                    f"    A: fx.Tensor,\n"
                    f"    B: fx.Tensor,\n"
                    f"    C: fx.Tensor,\n"
                    f"    block_dim: fx.Constexpr[int],\n"
                    f"):\n"
                    f"    # Kernel implementation here\n"
                    f"    bid = fx.block_idx.x\n"
                    f"    tid = fx.thread_idx.x\n"
                    f"    pass\n\n\n"
                    f"@flyc.jit\n"
                    f"def {op}_launch(A, B, C, N):\n"
                    f"    block_dim = 256\n"
                    f"    grid = (N + block_dim - 1) // block_dim\n"
                    f"    {op}_kernel[(grid,), (block_dim,)](A, B, C, block_dim)\n"
                    f"```"},
            ],
            "source": "import_fix_skeleton",
            "metadata": {"operator": op},
        })

    return pairs


def generate_import_correction_pairs():
    """Type 3: Wrong import → corrected import."""
    pairs = []
    for wrong, correct in WRONG_IMPORTS:
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"I'm getting an ImportError with this FlyDSL import:\n"
                    f"```python\n{wrong}\n```\n"
                    f"What's the correct import?"},
                {"role": "assistant", "content":
                    f"That import path doesn't exist in FlyDSL. The correct import is:\n\n"
                    f"```python\n{correct}\n```\n\n"
                    f"Standard FlyDSL imports:\n"
                    f"```python\n{CORRECT_IMPORT_BLOCK}\n```"},
            ],
            "source": "import_fix_correction",
            "metadata": {"wrong_import": wrong},
        })

        # Also add a variant without the user asking
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"Fix this FlyDSL code. The imports are wrong:\n"
                    f"```python\n{wrong}\n\n@flyc.kernel\ndef my_kernel(A: fx.Tensor):\n    pass\n```"},
                {"role": "assistant", "content":
                    f"The imports need to be corrected. Here's the fixed version:\n\n"
                    f"```python\n{CORRECT_IMPORT_BLOCK}\n\n\n"
                    f"@flyc.kernel\ndef my_kernel(A: fx.Tensor):\n    pass\n```"},
            ],
            "source": "import_fix_correction",
            "metadata": {"wrong_import": wrong},
        })

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.metadata_dir, exist_ok=True)

    # Generate import-focused data
    template_pairs = generate_import_template_pairs()
    correction_pairs = generate_import_correction_pairs()
    all_new = template_pairs + correction_pairs
    logger.info("Generated %d import-fix pairs: %d templates + %d corrections",
                len(all_new), len(template_pairs), len(correction_pairs))

    # Save to metadata
    gen_path = os.path.join(args.metadata_dir, "sft_v3_import_fix.jsonl")
    with open(gen_path, "w") as f:
        for p in all_new:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info("Saved to %s", gen_path)

    # Load existing SFT data and merge
    with open(args.sft_data) as f:
        existing = [json.loads(l) for l in f if l.strip()]
    logger.info("Existing SFT data: %d samples", len(existing))

    # High weight for import-fix data: repeat 3x
    combined = existing + all_new * 3
    random.shuffle(combined)

    # Backup original
    backup = args.output + ".v2.bak"
    if not os.path.exists(backup):
        import shutil
        shutil.copy2(args.sft_data, backup)
        logger.info("Backed up to %s", backup)

    with open(args.output, "w") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    src = Counter(r.get("source", "?") for r in combined)
    logger.info("Output: %d samples → %s", len(combined), args.output)
    for s, c in src.most_common(5):
        logger.info("  %s: %d", s, c)


if __name__ == "__main__":
    main()
