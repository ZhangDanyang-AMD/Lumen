"""Fix second-level API hallucination — teach model the exact FlyDSL API surface.

Problem: Model knows to use `import flydsl.expr as fx` but hallucinates
sub-imports like `from flydsl.expr import Expr, ArithOp, dtypes` (all nonexistent).

Solution: Generate SFT data that drills the exact API surface:
  1. API reference QA — "what's in fx.*?" → real list
  2. Valid vs invalid import corrections
  3. Real code snippets showing correct fx.* usage from FlyDSL kernels
  4. Negative examples — "this import doesn't exist, use this instead"

Usage::
    python fix_api_hallucination.py \
        --flydsl-dir /home/danyzhan/FlyDSL \
        --sft-data /path/to/sft/train.jsonl \
        --output /path/to/sft/train.jsonl \
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

SYSTEM_PROMPT = (
    "You are a FlyDSL GPU kernel programming expert for AMD Instinct GPUs. "
    "Always output raw Python code directly — never wrap in markdown code blocks. "
    "Use only APIs that actually exist in FlyDSL. "
    "The standard imports are:\n"
    "  import flydsl.compiler as flyc\n"
    "  import flydsl.expr as fx\n"
    "  from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr\n"
    "  from flydsl.expr import arith, buffer_ops, rocdl, gpu, range_constexpr"
)

# Real FlyDSL API — types available as fx.*
FX_TYPES = [
    "fx.Tensor", "fx.Constexpr", "fx.Layout", "fx.LayoutType",
    "fx.ComposedLayout", "fx.CopyAtom", "fx.MmaAtomType",
    "fx.Swizzle", "fx.SwizzleType", "fx.SharedAllocator",
    "fx.Stream", "fx.Index", "fx.PointerType", "fx.MemRefType",
]

# Real FlyDSL API — functions available as fx.*
FX_FUNCTIONS = [
    "fx.make_layout", "fx.make_shape", "fx.make_stride",
    "fx.logical_divide", "fx.zipped_divide", "fx.composition",
    "fx.complement", "fx.slice", "fx.copy_atom_call",
    "fx.make_copy_atom", "fx.make_rmem_tensor",
    "fx.memref_load_vec", "fx.memref_store_vec",
    "fx.block_idx", "fx.thread_idx", "fx.block_dim",
    "fx.syncthreads", "fx.printf",
]

# Real submodule imports
VALID_SUBMODULE_IMPORTS = [
    "from flydsl.expr import arith",
    "from flydsl.expr import buffer_ops",
    "from flydsl.expr import rocdl",
    "from flydsl.expr import gpu",
    "from flydsl.expr import const_expr",
    "from flydsl.expr import range_constexpr",
    "from flydsl.expr import vector",
    "from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector",
    "from flydsl.expr.typing import T",
    "from flydsl.expr.typing import Vector as Vec",
    "from flydsl.expr.rocdl import cluster",
    "from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr",
    "from flydsl.runtime.device import get_rocm_arch",
    "from flydsl.compiler.kernel_function import CompilationContext",
]

# Common hallucinated imports (from RFT failure analysis)
HALLUCINATED_IMPORTS = [
    ("from flydsl.expr import Expr", "No 'Expr' class in flydsl.expr. Use fx.Tensor for tensor types."),
    ("from flydsl.expr import ArithOp", "No 'ArithOp' in flydsl.expr. Use 'from flydsl.expr import arith' then arith.addf(), arith.mulf() etc."),
    ("from flydsl.expr import dtypes", "No 'dtypes' module. Use fx.Float32, fx.BFloat16, fx.Float8E4M3FNUZ etc. directly."),
    ("from flydsl.expr import astype", "No 'astype' function. Use arith type conversion functions."),
    ("from flydsl.utils import Layout", "No 'Layout' in flydsl.utils. Layout is fx.Layout (from flydsl.expr)."),
    ("from flydsl.utils import MultiWave", "No 'MultiWave' class. Multi-wave is a kernel launch config, not a utility class."),
    ("from flydsl.utils import SwizzleMode", "No 'SwizzleMode'. Use fx.Swizzle or swizzle_xor patterns."),
    ("from flydsl.utils import Tile", "No 'Tile' class. Tiling is done via fx.make_layout(shape, stride)."),
    ("from flydsl.utils import Warp", "No 'Warp' class. Warp info via fx.thread_idx, warp_size from kernels_common."),
    ("from flydsl.types import Tensor", "No 'flydsl.types' module. Use fx.Tensor."),
    ("import flydsl.types as ft", "No 'flydsl.types' module. Types are in fx: fx.Float32, fx.Tensor, fx.Constexpr."),
    ("from flydsl.expr import fx", "Don't import fx from flydsl.expr — use 'import flydsl.expr as fx' instead."),
    ("fx.DeviceArray", "No 'fx.DeviceArray'. Use fx.Tensor for kernel tensor arguments."),
    ("fx.Arg('N')", "No 'fx.Arg'. Use fx.Constexpr[int] for compile-time constants."),
    ("fx.RuntimeArg", "No 'fx.RuntimeArg'. Use fx.Constexpr[int] or plain int."),
    ("fx.InputTensor", "No 'fx.InputTensor'. Just use fx.Tensor for all tensor args."),
    ("fx.shared_memory(size)", "No 'fx.shared_memory'. Use SmemAllocator from flydsl.utils.smem_allocator."),
]


def generate_api_reference_pairs():
    """QA pairs about the FlyDSL API surface."""
    pairs = []

    # "What types are available in fx?"
    types_str = ", ".join(FX_TYPES)
    pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What types are available in FlyDSL's fx module (flydsl.expr)?"},
            {"role": "assistant", "content":
                f"The main types available as fx.* are:\n\n"
                f"Tensor types: fx.Tensor (kernel argument), fx.Constexpr[int] (compile-time constant)\n"
                f"Numeric types: fx.Float32, fx.BFloat16, fx.Float16, fx.Float8E4M3FNUZ, fx.Int32, fx.Int64\n"
                f"Layout types: fx.Layout, fx.ComposedLayout, fx.Swizzle\n"
                f"Copy/MMA atoms: fx.CopyAtom, fx.MmaAtomType\n"
                f"Memory: fx.SharedAllocator, fx.MemRefType, fx.PointerType\n\n"
                f"Important: There is NO fx.Expr, fx.DeviceArray, fx.Arg, fx.RuntimeArg, "
                f"fx.InputTensor, fx.OutputTensor, fx.dtypes, or fx.AsType. These do not exist."},
        ],
        "source": "api_reference",
        "metadata": {"topic": "fx_types"},
    })

    # "What functions are available in fx?"
    funcs_str = ", ".join(FX_FUNCTIONS)
    pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What functions can I call from fx.* in FlyDSL?"},
            {"role": "assistant", "content":
                f"Key functions in fx (flydsl.expr):\n\n"
                f"Layout algebra: fx.make_layout(), fx.make_shape(), fx.make_stride(), "
                f"fx.logical_divide(), fx.zipped_divide(), fx.composition(), fx.complement(), fx.slice()\n"
                f"Memory ops: fx.copy_atom_call(), fx.make_copy_atom(), fx.make_rmem_tensor(), "
                f"fx.memref_load_vec(), fx.memref_store_vec()\n"
                f"GPU indexing: fx.block_idx.x, fx.thread_idx.x, fx.block_dim.x\n"
                f"Synchronization: fx.syncthreads()\n"
                f"Debug: fx.printf()\n\n"
                f"For arithmetic: use 'from flydsl.expr import arith' then arith.addf(), arith.mulf()\n"
                f"For buffer ops: use 'from flydsl.expr import buffer_ops'\n"
                f"For ROCDL: use 'from flydsl.expr import rocdl' for MFMA, buffer loads etc."},
        ],
        "source": "api_reference",
        "metadata": {"topic": "fx_functions"},
    })

    # "What can I import from flydsl.expr?"
    imports_str = "\n".join(VALID_SUBMODULE_IMPORTS)
    pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What are all the valid imports from flydsl submodules?"},
            {"role": "assistant", "content":
                f"Valid FlyDSL imports:\n\n"
                f"{imports_str}\n\n"
                f"INVALID imports (these will cause ImportError):\n"
                f"  × from flydsl.expr import Expr, ArithOp, dtypes, astype\n"
                f"  × from flydsl.utils import Layout, MultiWave, SwizzleMode, Tile, Warp\n"
                f"  × from flydsl.types import Tensor\n"
                f"  × import flydsl.types as ft"},
        ],
        "source": "api_reference",
        "metadata": {"topic": "valid_imports"},
    })

    return pairs


def generate_hallucination_corrections():
    """Correction pairs for each hallucinated import."""
    pairs = []
    for wrong, explanation in HALLUCINATED_IMPORTS:
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"I'm getting ImportError with: {wrong}\nWhat's wrong?"},
                {"role": "assistant", "content":
                    f"{explanation}\n\n"
                    f"Standard FlyDSL imports:\n"
                    f"import flydsl.compiler as flyc\n"
                    f"import flydsl.expr as fx\n"
                    f"from flydsl.expr import arith, buffer_ops, rocdl, gpu, range_constexpr\n"
                    f"from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr"},
            ],
            "source": "api_hallucination_fix",
            "metadata": {"wrong_import": wrong},
        })
    return pairs


def generate_kernel_signature_examples(flydsl_dir):
    """Extract real kernel function signatures from FlyDSL repo to show correct fx.Tensor usage."""
    pairs = []
    kernel_dir = os.path.join(flydsl_dir, "kernels")

    for fname in sorted(os.listdir(kernel_dir)):
        if not fname.endswith(".py") or fname.startswith("__"):
            continue
        if "gfx1250" in fname or "rdna" in fname:
            continue

        code = open(os.path.join(kernel_dir, fname)).read()
        # Find @flyc.kernel function signatures
        for m in re.finditer(r'(@flyc\.kernel\s*\n\s*def \w+\([^)]+\))', code, re.DOTALL):
            sig = m.group(1)
            if "fx.Tensor" in sig or "fx.Constexpr" in sig:
                pairs.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content":
                            f"Show me a correct FlyDSL kernel function signature from {fname}."},
                        {"role": "assistant", "content":
                            f"import flydsl.compiler as flyc\nimport flydsl.expr as fx\n\n{sig}\n    ...\n\n"
                            f"Note: Use fx.Tensor for tensor arguments, fx.Constexpr[int] for compile-time constants. "
                            f"Never use fx.DeviceArray, fx.Arg, fx.RuntimeArg, or fx.InputTensor — they don't exist."},
                    ],
                    "source": "api_signature_example",
                    "metadata": {"source_file": fname},
                })
                break  # one per file

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flydsl-dir", required=True)
    parser.add_argument("--sft-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.metadata_dir, exist_ok=True)

    ref_pairs = generate_api_reference_pairs()
    logger.info("API reference pairs: %d", len(ref_pairs))

    correction_pairs = generate_hallucination_corrections()
    logger.info("Hallucination correction pairs: %d", len(correction_pairs))

    sig_pairs = generate_kernel_signature_examples(args.flydsl_dir)
    logger.info("Kernel signature example pairs: %d", len(sig_pairs))

    all_new = ref_pairs + correction_pairs + sig_pairs
    gen_path = os.path.join(args.metadata_dir, "sft_v5_api_fix.jsonl")
    with open(gen_path, "w") as f:
        for p in all_new:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info("Saved %d new pairs to %s", len(all_new), gen_path)

    # Merge: existing + new (repeated 3x for emphasis)
    with open(args.sft_data) as f:
        existing = [json.loads(l) for l in f if l.strip()]

    combined = existing + all_new * 3
    random.shuffle(combined)

    backup = args.output + ".v5.bak"
    if not os.path.exists(backup):
        shutil.copy2(args.sft_data, backup)

    with open(args.output, "w") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    src = Counter(r.get("source", "?") for r in combined)
    api_fix = sum(c for s, c in src.items() if "api" in s.lower() or "halluc" in s.lower() or "reference" in s.lower() or "signature" in s.lower())
    logger.info("Output: %d samples (api fix data: %d = %.1f%%)", len(combined), api_fix, api_fix / len(combined) * 100)


if __name__ == "__main__":
    main()
