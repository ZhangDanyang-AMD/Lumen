"""Add FlyDSL module structure digest to SFT dataset (v5d).

Embeds the complete module tree + API surface into training data so the model
internalizes what exists and what doesn't. Five data types:
  1. System prompt with module digest — every kernel generation sees it
  2. "What does X have?" reference QA — exhaustive per-module listings
  3. "X does NOT exist" negative QA — top hallucinated paths explicitly denied
  4. Import pattern correction — "from X import Y" vs "import X as Y"
  5. flyc API boundary — only kernel/jit/compile exist
"""

import argparse
import json
import logging
import os
import random
import shutil

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

MODULE_DIGEST = """FlyDSL Complete Module Map (30 modules total — use ONLY these import paths):

flydsl/
├── autotune              # @autotune decorator, Config, do_bench
├── compiler/             # import flydsl.compiler as flyc
│   ├── __init__          #   flyc.kernel, flyc.jit, flyc.compile (ONLY these 3)
│   ├── ast_rewriter      #   ASTRewriter (internal)
│   ├── backends          #   RocmBackend, get_backend
│   ├── jit_argument      #   JitArgumentRegistry
│   ├── jit_executor      #   (internal)
│   ├── jit_function      #   (internal)
│   ├── kernel_function   #   CompilationContext
│   ├── llvm_options      #   (internal)
│   └── protocol          #   (internal)
├── expr/                 # import flydsl.expr as fx
│   ├── __init__          #   222 public names: fx.Tensor, fx.Constexpr, fx.Layout, ...
│   ├── arith             #   from flydsl.expr import arith → arith.addf/mulf/subf
│   ├── buffer_ops        #   from flydsl.expr import buffer_ops → buffer_load/store
│   ├── derived           #   (internal layout ops)
│   ├── gpu               #   from flydsl.expr import gpu → SharedAllocator
│   ├── math              #   from flydsl.expr import math → math ops
│   ├── meta              #   (internal)
│   ├── numeric           #   (internal numeric helpers)
│   ├── primitive         #   (internal primitive ops)
│   ├── rocdl/            #   from flydsl.expr import rocdl → mfma_*, make_buffer_tensor
│   │   └── cluster       #   from flydsl.expr.rocdl import cluster
│   ├── typing            #   from flydsl.expr.typing import T, Vector as Vec
│   ├── utils/            #   (internal expr utilities)
│   └── vector            #   from flydsl.expr import vector
├── runtime/
│   ├── device            #   from flydsl.runtime.device import get_rocm_arch, is_rdna_arch
│   └── device_runtime    #   DeviceRuntime (internal)
└── utils/
    ├── env               #   EnvManager (internal)
    ├── logger            #   log (internal)
    └── smem_allocator    #   from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

CRITICAL IMPORT RULES:
  ✓ import flydsl.compiler as flyc     (NOT: from flydsl.compiler import flyc)
  ✓ import flydsl.expr as fx           (NOT: from flydsl.expr import fx)
  ✓ from flydsl.expr import arith      (submodules use 'from' import)
  ✓ from flydsl.expr import rocdl      (submodules use 'from' import)

flyc has ONLY 3 attributes: flyc.kernel, flyc.jit, flyc.compile
  × flyc.kernel_context — does NOT exist
  × flyc.SmemAllocator — SmemAllocator is in flydsl.utils.smem_allocator
  × flyc.launch — does NOT exist
  × flyc.get_shared_memory — does NOT exist
  × flyc.build — does NOT exist
  × flyc.compile_launch_func — does NOT exist

Key fx.* attributes (import flydsl.expr as fx):
  Types: fx.Tensor, fx.Constexpr[int], fx.Layout, fx.ComposedLayout, fx.CopyAtom,
         fx.Swizzle, fx.SharedAllocator, fx.Stream, fx.Index, fx.PointerType, fx.MemRefType
  Numerics: fx.Float32, fx.BFloat16, fx.Float16, fx.Float64,
            fx.Float8E4M3FNUZ, fx.Float8E4M3FN, fx.Float8E5M2,
            fx.Int32, fx.Int64, fx.Int16, fx.Int8, fx.Int4, fx.Boolean
  Layout: fx.make_layout(), fx.make_shape(), fx.make_stride(),
          fx.logical_divide(), fx.zipped_divide(), fx.composition(), fx.complement(), fx.slice()
  Copy/MMA: fx.copy_atom_call(), fx.make_copy_atom(), fx.make_rmem_tensor(),
            fx.memref_load_vec(), fx.memref_store_vec()
  GPU: fx.block_idx.x, fx.thread_idx.x, fx.block_dim.x, fx.syncthreads(), fx.printf()

⚠️ DOES NOT EXIST — never import these:
  × flydsl.expr.types       — types are direct fx.* attributes (fx.Float32, fx.Tensor)
  × flydsl.expr.ops         — use flydsl.expr.arith / buffer_ops / rocdl
  × flydsl.expr.expr        — internal, never import directly
  × flydsl.expr._expr       — internal, never import directly
  × flydsl.expr.func        — does not exist
  × flydsl.expr.context     — does not exist
  × flydsl.expr.memory      — does not exist
  × flydsl.expr.nn          — does not exist
  × flydsl.expr.atomics     — does not exist
  × flydsl.expr.enums       — does not exist
  × flydsl.expr.dtypes      — does not exist (use fx.Float32, fx.BFloat16 directly)
  × flydsl.expr.type_traits — does not exist
  × flydsl.expr.ir          — does not exist
  × flydsl.expr.smem        — does not exist
  × flydsl.expr.f32         — does not exist (use fx.Float32)
  × flydsl.expr.smem_allocator — SmemAllocator is in flydsl.utils.smem_allocator
  × flydsl.types            — does not exist (use fx.Float32, fx.Tensor, etc.)
  × flydsl.core             — does not exist
  × flydsl.kernel           — does not exist (use flyc.kernel)
  × flydsl.layout           — does not exist (use fx.Layout, fx.make_layout)
  × flydsl.allocators       — does not exist (use flydsl.utils.smem_allocator)
  × flydsl.ops              — does not exist
  × flydsl.runtime.rocdl    — rocdl is in flydsl.expr, not runtime
  × flydsl.runtime.rocm     — does not exist
  × flydsl.runtime.smem_allocator — SmemAllocator is in flydsl.utils
  × flydsl.runtime.stream   — Stream is fx.Stream
  × flydsl.utils.gfx90a     — does not exist
  × flydsl.utils.Layout     — Layout is fx.Layout
  × flydsl.utils.div_up     — use (a + b - 1) // b
  × flydsl.utils.gemm_test_utils — does not exist
  × from flydsl.expr import fx — WRONG! Use: import flydsl.expr as fx
  × from flydsl.compiler import flyc — WRONG! Use: import flydsl.compiler as flyc
  × from flydsl import kernel — WRONG! Use: import flydsl.compiler as flyc
  × from flydsl import expr  — WRONG! Use: import flydsl.expr as fx
  × fx.Expr, fx.DeviceArray, fx.Arg, fx.RuntimeArg, fx.InputTensor, fx.dtypes — don't exist
  × fx.arange, fx.buffer, fx.constexpr, fx._expr, fx.FP8, fx.Buffer — don't exist"""

SYSTEM_WITH_DIGEST = (
    "You are a FlyDSL GPU kernel programming expert for AMD Instinct GPUs. "
    "Output raw Python code directly, never markdown blocks.\n\n"
    + MODULE_DIGEST
)

# Pairs for per-module reference
MODULE_QA = [
    ("What can I import from flydsl.expr?",
     "flydsl.expr (import as fx) provides:\n\n"
     "Direct attributes (222 total):\n"
     "  Types: fx.Tensor, fx.Constexpr, fx.Layout, fx.ComposedLayout, fx.CopyAtom, fx.Swizzle, fx.SharedAllocator\n"
     "  Numerics: fx.Float32, fx.BFloat16, fx.Float16, fx.Float8E4M3FNUZ, fx.Int32, fx.Int64\n"
     "  Functions: fx.make_layout(), fx.logical_divide(), fx.zipped_divide(), fx.composition(), fx.copy_atom_call()\n"
     "  GPU: fx.block_idx.x, fx.thread_idx.x, fx.syncthreads()\n\n"
     "Submodules (import separately):\n"
     "  from flydsl.expr import arith, buffer_ops, rocdl, gpu, const_expr, range_constexpr, vector\n\n"
     "NOT in flydsl.expr:\n"
     "  × types, ops, expr, _expr, func, context, memory, nn, atomics, enums, dtypes, type_traits, ir, smem, f32 — these DO NOT EXIST"),

    ("What submodules does flydsl.expr have?",
     "Real submodules of flydsl.expr:\n"
     "  arith — arithmetic: arith.addf(), arith.mulf(), arith.subf(), arith.divf()\n"
     "  buffer_ops — buffer: buffer_ops.buffer_load(), buffer_ops.buffer_store()\n"
     "  rocdl — ROCDL/MFMA: rocdl.mfma_f32_32x32x16_bf16(), rocdl.make_buffer_tensor()\n"
     "  gpu — GPU: gpu.SharedAllocator\n"
     "  const_expr, range_constexpr, vector, typing, numeric, primitive, derived, math, meta\n\n"
     "NOT real submodules (hallucinations):\n"
     "  × flydsl.expr.types — types are direct fx.* attributes\n"
     "  × flydsl.expr.ops — use arith/buffer_ops/rocdl instead\n"
     "  × flydsl.expr.expr, flydsl.expr._expr — internal, never import\n"
     "  × flydsl.expr.func — does not exist\n"
     "  × flydsl.expr.context, flydsl.expr.memory, flydsl.expr.nn — don't exist\n"
     "  × flydsl.expr.atomics, flydsl.expr.enums, flydsl.expr.dtypes — don't exist\n"
     "  × flydsl.expr.type_traits, flydsl.expr.ir, flydsl.expr.smem — don't exist"),

    ("Where is SmemAllocator in FlyDSL?",
     "SmemAllocator is in flydsl.utils.smem_allocator:\n"
     "  from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr\n\n"
     "NOT in:\n"
     "  × flydsl.allocators (doesn't exist)\n"
     "  × flydsl.expr.smem_allocator (doesn't exist)\n"
     "  × flydsl.runtime.smem_allocator (doesn't exist)\n"
     "  × flyc.SmemAllocator (not in compiler — flyc only has kernel/jit/compile)\n\n"
     "Alternative: fx.SharedAllocator (newer API, from flydsl.expr)"),

    ("Where are data types like Float16, BFloat16, Int32 in FlyDSL?",
     "Data types are DIRECT attributes of fx (flydsl.expr):\n"
     "  fx.Float32, fx.Float16, fx.BFloat16, fx.Float64\n"
     "  fx.Int32, fx.Int64, fx.Int16, fx.Int8, fx.Int4\n"
     "  fx.Float8E4M3FNUZ, fx.Float8E4M3FN, fx.Float8E5M2\n"
     "  fx.Boolean, fx.Index\n\n"
     "NOT in:\n"
     "  × flydsl.expr.types — this module doesn't exist\n"
     "  × flydsl.expr.dtypes — this module doesn't exist\n"
     "  × flydsl.types — this module doesn't exist\n"
     "  × fx.dtypes — not an attribute\n"
     "  × fx.FP8, fx.f32 — not attributes (use fx.Float8E4M3FNUZ, fx.Float32)"),

    ("Where are ROCDL/MFMA operations in FlyDSL?",
     "ROCDL operations are in flydsl.expr.rocdl:\n"
     "  from flydsl.expr import rocdl\n"
     "  rocdl.mfma_f32_32x32x16_bf16(...)\n"
     "  rocdl.make_buffer_tensor(tensor)\n"
     "  rocdl.BufferCopy32b()\n\n"
     "For cluster ops: from flydsl.expr.rocdl import cluster\n\n"
     "NOT in:\n"
     "  × flydsl.runtime.rocdl — rocdl is in expr, not runtime\n"
     "  × flydsl.runtime.rocm — doesn't exist\n"
     "  × flydsl.runtime.BlockIdx — use fx.block_idx instead"),

    ("What attributes does flyc (flydsl.compiler) have?",
     "flyc has ONLY 3 public attributes:\n"
     "  flyc.kernel  — decorator for GPU kernel functions\n"
     "  flyc.jit     — decorator for JIT-compiled host functions\n"
     "  flyc.compile — compile a kernel to native code\n\n"
     "Usage:\n"
     "  import flydsl.compiler as flyc  # MUST use this form\n\n"
     "  @flyc.kernel\n"
     "  def my_kernel(A: fx.Tensor, B: fx.Tensor): ...\n\n"
     "  @flyc.jit\n"
     "  def launch(A, B): ...\n\n"
     "NOT in flyc:\n"
     "  × flyc.kernel_context — does NOT exist\n"
     "  × flyc.SmemAllocator — SmemAllocator is in flydsl.utils.smem_allocator\n"
     "  × flyc.launch — does NOT exist\n"
     "  × flyc.get_shared_memory — does NOT exist\n"
     "  × flyc.build — does NOT exist\n"
     "  × flyc.compile_launch_func — does NOT exist"),

    ("How do I import flydsl.compiler and flydsl.expr?",
     "CORRECT import patterns:\n"
     "  import flydsl.compiler as flyc    # creates 'flyc' alias\n"
     "  import flydsl.expr as fx          # creates 'fx' alias\n\n"
     "WRONG patterns that cause ImportError:\n"
     "  × from flydsl.expr import fx      # fx is not an object inside flydsl.expr!\n"
     "  × from flydsl.compiler import flyc # flyc is not an object inside compiler!\n"
     "  × from flydsl import kernel       # flydsl.kernel doesn't exist\n"
     "  × from flydsl import expr as fx   # wrong syntax\n"
     "  × import flydsl as fx             # wrong — fx should be flydsl.expr\n\n"
     "For submodules, 'from' import is correct:\n"
     "  from flydsl.expr import arith          # ✓\n"
     "  from flydsl.expr import buffer_ops     # ✓\n"
     "  from flydsl.expr import rocdl          # ✓\n"
     "  from flydsl.utils.smem_allocator import SmemAllocator  # ✓"),
]

# Negative examples — high-frequency hallucinations from v5c analysis
NEGATIVE_QA = [
    # Import pattern errors (most frequent in v5c)
    "from flydsl.expr import fx",
    "from flydsl.compiler import flyc",
    "from flydsl import kernel",
    "from flydsl import expr as fx",
    "import flydsl as fx",
    # flydsl.expr.* hallucinations
    "from flydsl.expr.types import F16",
    "from flydsl.expr.types import int32",
    "from flydsl.expr.types import DType",
    "from flydsl.expr.types import ScaledFP8",
    "import flydsl.expr.ops as ops",
    "from flydsl.expr import ops",
    "from flydsl.expr import types",
    "from flydsl.expr import dtypes",
    "from flydsl.expr import expr",
    "from flydsl.expr import utils",
    "from flydsl.expr import memory",
    "from flydsl.expr import atomics",
    "from flydsl.expr import type_traits",
    "from flydsl.expr import enums",
    "from flydsl.expr import ir",
    "from flydsl.expr import f32",
    "from flydsl.expr import smem",
    "from flydsl.expr import func",
    "from flydsl.expr.context import smem",
    "from flydsl.expr.memory import make_tensor_pointer",
    "from flydsl.expr._expr import Expr, Op",
    "from flydsl.expr._expr import _make_call",
    "from flydsl.expr._expr import _TensorBase",
    "from flydsl.expr import fx",
    "from flydsl.expr.func import kernel_func",
    "import flydsl.expr.nn as fxnn",
    # flyc.* attribute hallucinations (new in v5c)
    "flyc.kernel_context",
    "flyc.SmemAllocator",
    "flyc.launch",
    "flyc.get_shared_memory",
    "flyc.build",
    "flyc.compile_launch_func",
    # flydsl.runtime.* hallucinations
    "from flydsl.runtime.rocdl import BlockIdx, ThreadIdx",
    "from flydsl.runtime.smem_allocator import SmemAllocator",
    "from flydsl.runtime.rocm import ROCm",
    "from flydsl.runtime import rocdl",
    # flydsl.utils.* hallucinations
    "from flydsl.utils.gfx90a import m0x8_mfma_f32_f32_f32",
    "from flydsl.utils.gemm_test_utils import run_gemm",
    "from flydsl.utils import div_up",
    # flydsl top-level hallucinations
    "import flydsl.layout as fl",
    "from flydsl.kernel import flyc",
    "import flydsl.types as ft",
    "from flydsl import expr as fx",
    "from flydsl.expr.utils import div_up",
    # fx.* attribute hallucinations
    "fx.Expr",
    "fx.DeviceArray",
    "fx.Arg",
    "fx.arange",
    "fx.buffer",
    "fx.constexpr",
    "fx.FP8",
    "fx.Buffer",
    "fx._expr",
]

# Correct import patterns — positive examples (kernel skeletons)
CORRECT_IMPORT_KERNELS = [
    {
        "op": "gemm",
        "code": (
            "import flydsl.compiler as flyc\n"
            "import flydsl.expr as fx\n"
            "from flydsl.expr import arith, buffer_ops, rocdl\n"
            "from flydsl.utils.smem_allocator import SmemAllocator\n\n\n"
            "@flyc.kernel\n"
            "def gemm_kernel(\n"
            "    A: fx.Tensor,\n"
            "    B: fx.Tensor,\n"
            "    C: fx.Tensor,\n"
            "    M: fx.Constexpr[int],\n"
            "    N: fx.Constexpr[int],\n"
            "    K: fx.Constexpr[int],\n"
            "):\n"
            "    bid = fx.block_idx.x\n"
            "    tid = fx.thread_idx.x\n"
            "    smem = SmemAllocator()\n"
            "    tA = smem.alloc((128, 64), fx.BFloat16)\n"
            "    tB = smem.alloc((64, 128), fx.BFloat16)\n"
            "    acc = fx.make_rmem_tensor((4, 1), fx.Float32)\n"
            "    mfma = rocdl.mfma_f32_32x32x16_bf16\n"
            "    for k in range(K // 64):\n"
            "        buffer_ops.buffer_load(tA, A, bid * 128, k * 64)\n"
            "        buffer_ops.buffer_load(tB, B, k * 64, bid * 128)\n"
            "        fx.syncthreads()\n"
            "        acc = mfma(tA, tB, acc)\n"
            "        fx.syncthreads()\n"
            "    buffer_ops.buffer_store(C, acc, bid * 128, bid * 128)\n\n\n"
            "@flyc.jit\n"
            "def gemm_launch(A, B, C, M, N, K):\n"
            "    grid = (M // 128 * N // 128,)\n"
            "    block = (256,)\n"
            "    gemm_kernel[grid, block](A, B, C, M, N, K)\n"
        ),
    },
    {
        "op": "softmax",
        "code": (
            "import flydsl.compiler as flyc\n"
            "import flydsl.expr as fx\n"
            "from flydsl.expr import arith\n\n\n"
            "@flyc.kernel\n"
            "def softmax_kernel(\n"
            "    input: fx.Tensor,\n"
            "    output: fx.Tensor,\n"
            "    N: fx.Constexpr[int],\n"
            "):\n"
            "    bid = fx.block_idx.x\n"
            "    tid = fx.thread_idx.x\n"
            "    row_max = fx.Float32(-1e30)\n"
            "    for i in range(N):\n"
            "        val = input[bid, i]\n"
            "        row_max = arith.maximumf(row_max, val)\n"
            "    row_sum = fx.Float32(0.0)\n"
            "    for i in range(N):\n"
            "        val = arith.subf(input[bid, i], row_max)\n"
            "        exp_val = fx.exp(val)\n"
            "        row_sum = arith.addf(row_sum, exp_val)\n"
            "    for i in range(N):\n"
            "        val = arith.subf(input[bid, i], row_max)\n"
            "        output[bid, i] = arith.divf(fx.exp(val), row_sum)\n\n\n"
            "@flyc.jit\n"
            "def softmax_launch(input, output, N):\n"
            "    M = input.shape[0]\n"
            "    softmax_kernel[(M,), (1,)](input, output, N)\n"
        ),
    },
    {
        "op": "rmsnorm",
        "code": (
            "import flydsl.compiler as flyc\n"
            "import flydsl.expr as fx\n"
            "from flydsl.expr import arith\n\n\n"
            "@flyc.kernel\n"
            "def rmsnorm_kernel(\n"
            "    x: fx.Tensor,\n"
            "    weight: fx.Tensor,\n"
            "    out: fx.Tensor,\n"
            "    N: fx.Constexpr[int],\n"
            "    eps: fx.Constexpr[float],\n"
            "):\n"
            "    bid = fx.block_idx.x\n"
            "    tid = fx.thread_idx.x\n"
            "    ss = fx.Float32(0.0)\n"
            "    for i in range(N):\n"
            "        val = x[bid, i]\n"
            "        ss = arith.addf(ss, arith.mulf(val, val))\n"
            "    ss = arith.divf(ss, fx.Float32(N))\n"
            "    ss = arith.addf(ss, fx.Float32(eps))\n"
            "    rms = fx.rsqrt(ss)\n"
            "    for i in range(N):\n"
            "        out[bid, i] = arith.mulf(arith.mulf(x[bid, i], rms), weight[i])\n\n\n"
            "@flyc.jit\n"
            "def rmsnorm_launch(x, weight, out, N, eps=1e-6):\n"
            "    M = x.shape[0]\n"
            "    rmsnorm_kernel[(M,), (1,)](x, weight, out, N, eps)\n"
        ),
    },
    {
        "op": "rope",
        "code": (
            "import flydsl.compiler as flyc\n"
            "import flydsl.expr as fx\n"
            "from flydsl.expr import arith\n\n\n"
            "@flyc.kernel\n"
            "def rope_kernel(\n"
            "    x: fx.Tensor,\n"
            "    cos_cache: fx.Tensor,\n"
            "    sin_cache: fx.Tensor,\n"
            "    out: fx.Tensor,\n"
            "    seq_len: fx.Constexpr[int],\n"
            "    head_dim: fx.Constexpr[int],\n"
            "):\n"
            "    bid = fx.block_idx.x\n"
            "    tid = fx.thread_idx.x\n"
            "    half_dim = head_dim // 2\n"
            "    for pos in range(seq_len):\n"
            "        cos_val = cos_cache[pos, tid]\n"
            "        sin_val = sin_cache[pos, tid]\n"
            "        x0 = x[bid, pos, tid]\n"
            "        x1 = x[bid, pos, tid + half_dim]\n"
            "        out[bid, pos, tid] = arith.subf(\n"
            "            arith.mulf(x0, cos_val), arith.mulf(x1, sin_val))\n"
            "        out[bid, pos, tid + half_dim] = arith.addf(\n"
            "            arith.mulf(x1, cos_val), arith.mulf(x0, sin_val))\n\n\n"
            "@flyc.jit\n"
            "def rope_launch(x, cos_cache, sin_cache, out, seq_len, head_dim):\n"
            "    batch = x.shape[0]\n"
            "    rope_kernel[(batch,), (head_dim // 2,)](x, cos_cache, sin_cache, out, seq_len, head_dim)\n"
        ),
    },
    {
        "op": "flash_attn",
        "code": (
            "import flydsl.compiler as flyc\n"
            "import flydsl.expr as fx\n"
            "from flydsl.expr import arith, buffer_ops, rocdl\n"
            "from flydsl.utils.smem_allocator import SmemAllocator\n\n\n"
            "@flyc.kernel\n"
            "def flash_attn_fwd(\n"
            "    Q: fx.Tensor,\n"
            "    K: fx.Tensor,\n"
            "    V: fx.Tensor,\n"
            "    O: fx.Tensor,\n"
            "    num_heads: fx.Constexpr[int],\n"
            "    head_dim: fx.Constexpr[int],\n"
            "    seq_len: fx.Constexpr[int],\n"
            "    scale: fx.Constexpr[float],\n"
            "):\n"
            "    bid = fx.block_idx.x\n"
            "    tid = fx.thread_idx.x\n"
            "    smem = SmemAllocator()\n"
            "    sQ = smem.alloc((64, head_dim), fx.BFloat16)\n"
            "    sK = smem.alloc((64, head_dim), fx.BFloat16)\n"
            "    sV = smem.alloc((64, head_dim), fx.BFloat16)\n"
            "    acc = fx.make_rmem_tensor((2, 1), fx.Float32)\n"
            "    m_prev = fx.Float32(-1e30)\n"
            "    l_prev = fx.Float32(0.0)\n"
            "    buffer_ops.buffer_load(sQ, Q, bid * 64, 0)\n"
            "    fx.syncthreads()\n"
            "    for block_k in range(seq_len // 64):\n"
            "        buffer_ops.buffer_load(sK, K, block_k * 64, 0)\n"
            "        buffer_ops.buffer_load(sV, V, block_k * 64, 0)\n"
            "        fx.syncthreads()\n"
            "        mfma = rocdl.mfma_f32_32x32x16_bf16\n"
            "        scores = mfma(sQ, sK, acc)\n"
            "        fx.syncthreads()\n"
            "    buffer_ops.buffer_store(O, acc, bid * 64, 0)\n\n\n"
            "@flyc.jit\n"
            "def flash_attn_launch(Q, K, V, O, num_heads, head_dim, seq_len):\n"
            "    scale = 1.0 / (head_dim ** 0.5)\n"
            "    grid = (seq_len // 64 * num_heads,)\n"
            "    block = (256,)\n"
            "    flash_attn_fwd[grid, block](Q, K, V, O, num_heads, head_dim, seq_len, scale)\n"
        ),
    },
    {
        "op": "topk",
        "code": (
            "import flydsl.compiler as flyc\n"
            "import flydsl.expr as fx\n"
            "from flydsl.expr import arith\n\n\n"
            "@flyc.kernel\n"
            "def topk_kernel(\n"
            "    input: fx.Tensor,\n"
            "    values: fx.Tensor,\n"
            "    indices: fx.Tensor,\n"
            "    N: fx.Constexpr[int],\n"
            "    K: fx.Constexpr[int],\n"
            "):\n"
            "    bid = fx.block_idx.x\n"
            "    tid = fx.thread_idx.x\n"
            "    for k in range(K):\n"
            "        best_val = fx.Float32(-1e30)\n"
            "        best_idx = fx.Int32(-1)\n"
            "        for i in range(N):\n"
            "            val = input[bid, i]\n"
            "            is_better = arith.cmpf(val, best_val, 'ogt')\n"
            "            if is_better:\n"
            "                best_val = val\n"
            "                best_idx = fx.Int32(i)\n"
            "        values[bid, k] = best_val\n"
            "        indices[bid, k] = best_idx\n\n\n"
            "@flyc.jit\n"
            "def topk_launch(input, values, indices, N, K):\n"
            "    batch = input.shape[0]\n"
            "    topk_kernel[(batch,), (1,)](input, values, indices, N, K)\n"
        ),
    },
]

# Import correction pairs — wrong → right
IMPORT_CORRECTIONS = [
    ("from flydsl.expr import fx",
     "WRONG: 'from flydsl.expr import fx' causes ImportError because 'fx' is not an object inside flydsl.expr.\n\n"
     "CORRECT:\n  import flydsl.expr as fx\n\n"
     "This creates an alias 'fx' for the flydsl.expr module. Then use fx.Tensor, fx.Float32, fx.make_layout(), etc."),
    ("from flydsl.compiler import flyc",
     "WRONG: 'from flydsl.compiler import flyc' causes ImportError because 'flyc' is not an object inside flydsl.compiler.\n\n"
     "CORRECT:\n  import flydsl.compiler as flyc\n\n"
     "This creates an alias 'flyc' for the flydsl.compiler module. Then use @flyc.kernel, @flyc.jit, flyc.compile()."),
    ("from flydsl import kernel",
     "WRONG: 'from flydsl import kernel' causes ImportError because flydsl.kernel doesn't exist.\n\n"
     "CORRECT:\n  import flydsl.compiler as flyc\n\n"
     "Then use @flyc.kernel as a decorator."),
    ("import flydsl as fx",
     "WRONG: 'import flydsl as fx' gives you the top-level flydsl package, not the expr module.\n\n"
     "CORRECT:\n  import flydsl.expr as fx\n\n"
     "fx should be an alias for flydsl.expr specifically."),
    ("flyc.kernel_context",
     "WRONG: flyc.kernel_context does NOT exist. flyc only has 3 attributes: kernel, jit, compile.\n\n"
     "If you need kernel context, use function parameters:\n"
     "  @flyc.kernel\n"
     "  def my_kernel(A: fx.Tensor, B: fx.Tensor, N: fx.Constexpr[int]):\n"
     "      bid = fx.block_idx.x  # block index from fx, not flyc\n"
     "      tid = fx.thread_idx.x  # thread index from fx, not flyc"),
    ("flyc.SmemAllocator",
     "WRONG: flyc.SmemAllocator does NOT exist. SmemAllocator is NOT in the compiler module.\n\n"
     "CORRECT:\n  from flydsl.utils.smem_allocator import SmemAllocator\n\n"
     "Usage:\n"
     "  smem = SmemAllocator()\n"
     "  tile_a = smem.alloc((128, 64), fx.BFloat16)"),
    ("import jax",
     "WRONG: FlyDSL does not use JAX. Remove 'import jax' and all jax references.\n\n"
     "FlyDSL imports:\n"
     "  import flydsl.compiler as flyc\n"
     "  import flydsl.expr as fx\n"
     "  from flydsl.expr import arith, buffer_ops, rocdl\n"
     "  import torch  # for tensor creation only"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.metadata_dir, exist_ok=True)

    pairs = []

    # Type 1: Module digest embedded in system prompt for kernel generation tasks (12 ops)
    ops = ["gemm", "softmax", "rmsnorm", "layernorm", "flash_attn", "moe",
           "rope", "topk", "paged_attn", "mla", "allreduce", "vec_add"]
    for op in ops:
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_WITH_DIGEST},
                {"role": "user", "content": f"Write a FlyDSL {op} kernel for gfx950."},
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
                    f"    pass\n\n\n"
                    f"@flyc.jit\n"
                    f"def {op}_launch(A, B, C, N):\n"
                    f"    block_dim = 256\n"
                    f"    grid = (N + block_dim - 1) // block_dim\n"
                    f"    {op}_kernel[(grid,), (block_dim,)](A, B, C, N)\n"},
            ],
            "source": "module_digest_kernel",
            "metadata": {"operator": op},
        })

    # Type 2: Correct import kernel skeletons (detailed, with real patterns)
    for kernel in CORRECT_IMPORT_KERNELS:
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_WITH_DIGEST},
                {"role": "user", "content": f"Write a FlyDSL {kernel['op']} kernel for AMD gfx950 using proper imports and patterns."},
                {"role": "assistant", "content": kernel["code"]},
            ],
            "source": "module_digest_correct_kernel",
            "metadata": {"operator": kernel["op"]},
        })

    # Type 3: Module reference QA
    for question, answer in MODULE_QA:
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_WITH_DIGEST},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "source": "module_digest_qa",
            "metadata": {"type": "reference"},
        })

    # Type 4: Negative — "this doesn't exist"
    for wrong in NEGATIVE_QA:
        if wrong.startswith("fx."):
            user_msg = f"Why does my code fail when I use {wrong}?"
        elif wrong.startswith("flyc."):
            user_msg = f"My code crashes with AttributeError: flyc has no attribute '{wrong.split('.')[1]}'"
        else:
            user_msg = f"My code fails with: {wrong}"
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_WITH_DIGEST},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content":
                    f"That import path or attribute does not exist in FlyDSL.\n\n"
                    + MODULE_DIGEST},
            ],
            "source": "module_digest_negative",
            "metadata": {"wrong": wrong},
        })

    # Type 5: Import correction pairs
    for wrong_pattern, correction in IMPORT_CORRECTIONS:
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_WITH_DIGEST},
                {"role": "user", "content": f"I get an error with: {wrong_pattern}"},
                {"role": "assistant", "content": correction},
            ],
            "source": "module_digest_import_fix",
            "metadata": {"wrong": wrong_pattern},
        })

    n_kernel = len(ops)
    n_correct = len(CORRECT_IMPORT_KERNELS)
    n_qa = len(MODULE_QA)
    n_negative = len(NEGATIVE_QA)
    n_import_fix = len(IMPORT_CORRECTIONS)
    logger.info("Generated %d module digest pairs: %d skeleton + %d correct_kernel + %d QA + %d negative + %d import_fix",
                len(pairs), n_kernel, n_correct, n_qa, n_negative, n_import_fix)

    # Save to metadata
    gen_path = os.path.join(args.metadata_dir, "sft_v5d_module_digest.jsonl")
    with open(gen_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Load existing and merge (repeat 3x for emphasis)
    with open(args.sft_data) as f:
        existing = [json.loads(l) for l in f if l.strip()]

    combined = existing + pairs * 3
    random.shuffle(combined)

    backup = args.output + ".v5c.bak"
    if not os.path.exists(backup):
        shutil.copy2(args.sft_data, backup)

    with open(args.output, "w") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    src = Counter(r.get("source", "?") for r in combined)
    digest_count = sum(c for s, c in src.items() if "digest" in s)
    logger.info("Output: %d samples (module digest: %d = %.1f%%)",
                len(combined), digest_count, digest_count / len(combined) * 100)
    for s, c in src.most_common():
        if "digest" in s:
            logger.info("  %s: %d", s, c)


if __name__ == "__main__":
    main()
