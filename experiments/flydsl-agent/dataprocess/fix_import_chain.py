"""Fix API hallucination by teaching the complete FlyDSL import chain tree.

Core insight: model hallucinates because it doesn't know the MODULE STRUCTURE.
Instead of isolated corrections, teach the entire import tree so the model can
navigate: flydsl → compiler/expr/utils → submodules → specific names.

Generates:
  1. Import tree navigation QA — "I need X, where is it?"
  2. Module-level reference cards — complete listing of each module
  3. Chain reasoning examples — "want SmemAllocator → flydsl.utils → smem_allocator → SmemAllocator"
  4. Negative chain examples — "flydsl.expr does NOT have: Expr, dtypes, ArithOp..."
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
    "You are a FlyDSL GPU kernel programming expert. "
    "You know the exact FlyDSL module structure and never hallucinate imports. "
    "Output raw Python code, never markdown blocks."
)

# Load the extracted import tree
IMPORT_TREE = None

def load_tree():
    global IMPORT_TREE
    if IMPORT_TREE is None:
        IMPORT_TREE = json.load(open("/tmp/flydsl_import_tree.json"))
    return IMPORT_TREE


def generate_tree_overview():
    """Teach the full module tree as a single reference."""
    tree = load_tree()
    tree_text = """FlyDSL Module Structure (complete import chain):

flydsl/
├── compiler (alias: flyc)           # import flydsl.compiler as flyc
│   └── @flyc.kernel, @flyc.jit     # decorators for device kernel / launch wrapper
│
├── expr (alias: fx)                 # import flydsl.expr as fx
│   ├── Types: fx.Tensor, fx.Constexpr[int], fx.Layout, fx.ComposedLayout,
│   │         fx.CopyAtom, fx.MmaAtomType, fx.Swizzle, fx.SharedAllocator,
│   │         fx.Float32, fx.BFloat16, fx.Float8E4M3FNUZ, fx.Int32, fx.Int64...
│   │
│   ├── Functions: fx.make_layout(), fx.make_shape(), fx.make_stride(),
│   │             fx.logical_divide(), fx.zipped_divide(), fx.composition(),
│   │             fx.complement(), fx.slice(), fx.copy_atom_call(),
│   │             fx.block_idx.x, fx.thread_idx.x, fx.syncthreads()...
│   │
│   ├── arith                        # from flydsl.expr import arith
│   │   └── arith.addf(), arith.mulf(), arith.subf(), arith.divf()...
│   │
│   ├── buffer_ops                   # from flydsl.expr import buffer_ops
│   │   └── buffer_ops.buffer_load(), buffer_ops.buffer_store(),
│   │       buffer_ops.create_buffer_resource()...
│   │
│   ├── rocdl                        # from flydsl.expr import rocdl
│   │   └── rocdl.mfma_f32_32x32x16_bf16(), rocdl.make_buffer_tensor(),
│   │       rocdl.BufferCopy32b()...
│   │   └── rocdl.cluster            # from flydsl.expr.rocdl import cluster
│   │
│   ├── gpu                          # from flydsl.expr import gpu
│   │   └── gpu.SharedAllocator, gpu thread/block indexing helpers
│   │
│   ├── typing                       # from flydsl.expr.typing import T, Vector as Vec
│   ├── const_expr                   # from flydsl.expr import const_expr
│   ├── range_constexpr              # from flydsl.expr import range_constexpr
│   └── vector                       # from flydsl.expr import vector
│
├── utils/
│   └── smem_allocator               # from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
│
└── runtime/
    └── device                       # from flydsl.runtime.device import get_rocm_arch

⚠️ DOES NOT EXIST (common hallucinations):
  × flydsl.types             — use fx.Float32, fx.Tensor etc.
  × flydsl.allocators        — use flydsl.utils.smem_allocator
  × flydsl.ops               — use flydsl.expr submodules
  × flydsl.core              — does not exist
  × flydsl.expr.Expr         — not a class, use fx.Tensor
  × flydsl.expr.dtypes       — not a module, types are fx.*
  × flydsl.expr.ArithOp      — use from flydsl.expr import arith
  × flydsl.expr.astype       — not a function
  × flydsl.utils.Layout      — Layout is fx.Layout
  × flydsl.utils.Tile        — not a class, tiling via fx.make_layout
  × flydsl.utils.Warp        — not a class
  × flydsl.utils.MultiWave   — not a class
  × flydsl.utils.SwizzleMode — not a class, use fx.Swizzle
"""
    return tree_text


def generate_chain_navigation_pairs():
    """'I need X, where do I find it?' → chain reasoning."""
    pairs = []
    tree_text = generate_tree_overview()

    navigations = [
        ("SmemAllocator", "flydsl → utils → smem_allocator → SmemAllocator",
         "from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr"),
        ("MFMA instruction", "flydsl → expr → rocdl → mfma_*",
         "from flydsl.expr import rocdl\nrocdl.mfma_f32_32x32x16_bf16(...)"),
        ("tensor type for kernel args", "flydsl → expr → fx.Tensor",
         "import flydsl.expr as fx\n\n@flyc.kernel\ndef my_kernel(A: fx.Tensor, B: fx.Tensor): ..."),
        ("compile-time constant", "flydsl → expr → fx.Constexpr",
         "import flydsl.expr as fx\n\n@flyc.kernel\ndef my_kernel(N: fx.Constexpr[int]): ..."),
        ("arithmetic operations", "flydsl → expr → arith → addf/mulf/subf",
         "from flydsl.expr import arith\nresult = arith.addf(a, b)"),
        ("buffer load from global memory", "flydsl → expr → buffer_ops or rocdl.make_buffer_tensor",
         "from flydsl.expr import buffer_ops\n# or use rocdl.make_buffer_tensor(tensor)"),
        ("layout algebra", "flydsl → expr → fx.make_layout/fx.logical_divide/fx.zipped_divide",
         "import flydsl.expr as fx\nlayout = fx.make_layout(fx.make_shape(128, 64), fx.make_stride(64, 1))"),
        ("thread/block indexing", "flydsl → expr → fx.block_idx/fx.thread_idx",
         "import flydsl.expr as fx\nbid = fx.block_idx.x\ntid = fx.thread_idx.x"),
        ("GPU architecture detection", "flydsl → runtime → device → get_rocm_arch",
         "from flydsl.runtime.device import get_rocm_arch\narch = get_rocm_arch()  # 'gfx950'"),
        ("swizzle for bank conflict", "flydsl → expr → fx.Swizzle",
         "import flydsl.expr as fx\n# Swizzle is a type: fx.Swizzle\n# In practice, use swizzle_xor16 pattern in LDS layouts"),
    ]

    for need, chain, code in navigations:
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"I need {need} in FlyDSL. Where do I import it from?"},
                {"role": "assistant", "content":
                    f"Import chain: {chain}\n\n{code}\n\n"
                    f"Remember the FlyDSL module structure:\n"
                    f"  flydsl.compiler (flyc) — kernel/jit decorators\n"
                    f"  flydsl.expr (fx) — types, functions, layout algebra\n"
                    f"  flydsl.expr.{{arith,buffer_ops,rocdl,gpu}} — operation submodules\n"
                    f"  flydsl.utils.smem_allocator — shared memory\n"
                    f"  flydsl.runtime.device — GPU detection"},
            ],
            "source": "import_chain_navigation",
            "metadata": {"need": need},
        })

    return pairs


def generate_module_reference_cards():
    """Complete reference card for each module."""
    tree = load_tree()
    pairs = []

    tree_text = generate_tree_overview()

    # Full tree reference
    pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                "Show me the complete FlyDSL module structure with all valid import paths."},
            {"role": "assistant", "content": tree_text},
        ],
        "source": "import_chain_tree",
        "metadata": {"type": "full_tree"},
    })

    # Per-module cards
    for mod, info in tree.items():
        if mod == "flydsl":
            continue
        exports = info.get("exports", info.get("types", []))
        if not exports:
            continue
        alias = info.get("alias", "")
        imp = info.get("import", f"import {mod}" + (f" as {alias}" if alias else ""))

        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"What's available in {mod}?"},
                {"role": "assistant", "content":
                    f"Import: {imp}\n\n"
                    f"Available names ({len(exports)}):\n"
                    f"  {', '.join(exports[:30])}\n"
                    + (f"  ... and {len(exports)-30} more" if len(exports) > 30 else "")},
            ],
            "source": "import_chain_module_ref",
            "metadata": {"module": mod},
        })

    return pairs


def generate_negative_chain_examples():
    """Explicit 'this path does NOT exist' examples."""
    pairs = []

    wrong_chains = [
        ("from flydsl.expr import Expr",
         "flydsl.expr has NO class named 'Expr'. "
         "If you need a tensor type, use fx.Tensor. "
         "Chain: flydsl → expr → fx.Tensor (it's a direct attribute, not a sub-import)."),
        ("from flydsl.expr import dtypes",
         "flydsl.expr has NO submodule 'dtypes'. "
         "Data types are direct attributes of fx: fx.Float32, fx.BFloat16, fx.Int32, fx.Float8E4M3FNUZ. "
         "Chain: flydsl → expr → fx.Float32 (direct attribute)."),
        ("from flydsl.utils import Layout",
         "flydsl.utils has NO 'Layout'. Layout is in flydsl.expr: fx.Layout. "
         "Chain: flydsl → expr → fx.Layout (not utils!). "
         "flydsl.utils only has: smem_allocator, env, logger."),
        ("from flydsl.types import Tensor",
         "flydsl has NO 'types' module at all. The module structure is: "
         "flydsl.{compiler, expr, utils, runtime, _mlir}. "
         "Tensor type is fx.Tensor: flydsl → expr → fx.Tensor."),
        ("from flydsl.allocators import SharedAllocator",
         "flydsl has NO 'allocators' module. "
         "SharedAllocator is in two places: "
         "1) fx.SharedAllocator (from flydsl.expr) "
         "2) SmemAllocator from flydsl.utils.smem_allocator. "
         "Chain: flydsl → utils → smem_allocator → SmemAllocator."),
    ]

    for wrong, explanation in wrong_chains:
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"My code has `{wrong}` and it fails. Why?"},
                {"role": "assistant", "content": explanation},
            ],
            "source": "import_chain_negative",
            "metadata": {"wrong": wrong},
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

    nav_pairs = generate_chain_navigation_pairs()
    ref_pairs = generate_module_reference_cards()
    neg_pairs = generate_negative_chain_examples()
    all_new = nav_pairs + ref_pairs + neg_pairs

    logger.info("Chain navigation: %d, module refs: %d, negative examples: %d, total: %d",
                len(nav_pairs), len(ref_pairs), len(neg_pairs), len(all_new))

    gen_path = os.path.join(args.metadata_dir, "sft_v5_import_chains.jsonl")
    with open(gen_path, "w") as f:
        for p in all_new:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(args.sft_data) as f:
        existing = [json.loads(l) for l in f if l.strip()]

    # Repeat 4x — this is critical knowledge
    combined = existing + all_new * 4
    random.shuffle(combined)

    backup = args.output + ".v5pre_chain.bak"
    if not os.path.exists(backup):
        shutil.copy2(args.sft_data, backup)

    with open(args.output, "w") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    src = Counter(r.get("source", "?") for r in combined)
    chain_count = sum(c for s, c in src.items() if "chain" in s)
    logger.info("Output: %d samples (import chain data: %d = %.1f%%)",
                len(combined), chain_count, chain_count / len(combined) * 100)


if __name__ == "__main__":
    main()
