"""Boost correct FlyDSL kernel patterns in SFT dataset (v5e).

Strategy shift: instead of adding more negative examples (whack-a-mole),
massively increase the representation of CORRECT patterns so the model's
generation probability strongly favors the right import/API patterns.

Three approaches:
  1. Extract all unique correct kernels and repeat them with varied prompts
  2. Create import-first mini-kernels (short, focused on correct import block)
  3. Add flyc-boundary-aware kernel templates (flyc ONLY has kernel/jit/compile)

Target: correct kernel ratio 35% → 55%+, with correct import as dominant pattern.
"""

import argparse
import json
import hashlib
import logging
import os
import random
import re
import shutil

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a FlyDSL GPU kernel programming expert for AMD Instinct GPUs. "
    "Output raw Python code directly, never markdown blocks."
)

# Varied prompt templates for kernel generation
KERNEL_PROMPTS = [
    "Write a FlyDSL {op} kernel for AMD gfx950.",
    "Implement a {op} kernel using FlyDSL for MI350X (gfx950).",
    "Write a high-performance FlyDSL {op} kernel targeting AMD gfx950 GPU.",
    "Generate a complete FlyDSL {op} kernel with proper imports for gfx950.",
    "Create a FlyDSL {op} kernel. Use import flydsl.compiler as flyc and import flydsl.expr as fx.",
    "Write a {op} kernel in FlyDSL. Must use @flyc.kernel decorator and fx.Tensor parameters.",
    "Implement {op} using FlyDSL framework for AMD Instinct MI350X. Include @flyc.jit launch function.",
    "Write an optimized {op} kernel using FlyDSL with SmemAllocator and MFMA instructions for gfx950.",
]

# Operations to infer from code
OP_KEYWORDS = {
    "gemm": ["gemm", "matmul", "matrix_mul"],
    "softmax": ["softmax"],
    "rmsnorm": ["rmsnorm", "rms_norm"],
    "layernorm": ["layernorm", "layer_norm"],
    "flash_attn": ["flash_attn", "flash_attention", "fmha"],
    "moe": ["moe", "mixture_of_expert"],
    "rope": ["rope", "rotary"],
    "topk": ["topk", "top_k"],
    "paged_attn": ["paged_attn", "paged_attention"],
    "mla": ["mla", "multi_head_latent"],
    "allreduce": ["allreduce", "all_reduce"],
    "vec_add": ["vec_add", "vector_add", "elementwise"],
    "quant": ["quant", "quantiz"],
}

# Mini-kernel templates with CORRECT imports — short and focused
MINI_KERNELS = [
    {
        "prompt": "Write a minimal FlyDSL vector add kernel with correct imports.",
        "code": (
            "import flydsl.compiler as flyc\n"
            "import flydsl.expr as fx\n\n\n"
            "@flyc.kernel\n"
            "def vec_add(\n"
            "    A: fx.Tensor,\n"
            "    B: fx.Tensor,\n"
            "    C: fx.Tensor,\n"
            "    N: fx.Constexpr[int],\n"
            "):\n"
            "    tid = fx.thread_idx.x\n"
            "    bid = fx.block_idx.x\n"
            "    idx = bid * fx.block_dim.x + tid\n"
            "    if idx < N:\n"
            "        C[idx] = A[idx] + B[idx]\n\n\n"
            "@flyc.jit\n"
            "def vec_add_launch(A, B, C, N):\n"
            "    block = 256\n"
            "    grid = (N + block - 1) // block\n"
            "    vec_add[(grid,), (block,)](A, B, C, N)\n"
        ),
    },
    {
        "prompt": "Show the standard FlyDSL import block and kernel structure.",
        "code": (
            "import flydsl.compiler as flyc\n"
            "import flydsl.expr as fx\n"
            "from flydsl.expr import arith\n"
            "from flydsl.expr import buffer_ops\n"
            "from flydsl.expr import rocdl\n"
            "from flydsl.utils.smem_allocator import SmemAllocator\n\n\n"
            "@flyc.kernel\n"
            "def example_kernel(\n"
            "    input: fx.Tensor,\n"
            "    output: fx.Tensor,\n"
            "    N: fx.Constexpr[int],\n"
            "):\n"
            "    bid = fx.block_idx.x\n"
            "    tid = fx.thread_idx.x\n"
            "    smem = SmemAllocator()\n"
            "    tile = smem.alloc((64, 64), fx.BFloat16)\n"
            "    buffer_ops.buffer_load(tile, input, bid * 64, 0)\n"
            "    fx.syncthreads()\n"
            "    val = arith.mulf(tile[tid, 0], fx.Float32(2.0))\n"
            "    buffer_ops.buffer_store(output, val, bid * 64 + tid, 0)\n\n\n"
            "@flyc.jit\n"
            "def example_launch(input, output, N):\n"
            "    grid = (N // 64,)\n"
            "    block = (64,)\n"
            "    example_kernel[grid, block](input, output, N)\n"
        ),
    },
    {
        "prompt": "Write a FlyDSL kernel that uses MFMA with proper rocdl import.",
        "code": (
            "import flydsl.compiler as flyc\n"
            "import flydsl.expr as fx\n"
            "from flydsl.expr import arith, buffer_ops, rocdl\n"
            "from flydsl.utils.smem_allocator import SmemAllocator\n\n\n"
            "@flyc.kernel\n"
            "def mfma_kernel(\n"
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
            "    sA = smem.alloc((128, 64), fx.BFloat16)\n"
            "    sB = smem.alloc((64, 128), fx.BFloat16)\n"
            "    acc = fx.make_rmem_tensor((4, 1), fx.Float32)\n"
            "    mfma = rocdl.mfma_f32_32x32x16_bf16\n"
            "    for k_tile in range(K // 64):\n"
            "        buffer_ops.buffer_load(sA, A, bid * 128, k_tile * 64)\n"
            "        buffer_ops.buffer_load(sB, B, k_tile * 64, bid * 128)\n"
            "        fx.syncthreads()\n"
            "        acc = mfma(sA, sB, acc)\n"
            "        fx.syncthreads()\n"
            "    buffer_ops.buffer_store(C, acc, bid * 128, bid * 128)\n\n\n"
            "@flyc.jit\n"
            "def mfma_launch(A, B, C, M, N, K):\n"
            "    grid = (M // 128 * N // 128,)\n"
            "    block = (256,)\n"
            "    mfma_kernel[grid, block](A, B, C, M, N, K)\n"
        ),
    },
    {
        "prompt": "Write a simple FlyDSL reduce sum kernel.",
        "code": (
            "import flydsl.compiler as flyc\n"
            "import flydsl.expr as fx\n"
            "from flydsl.expr import arith\n\n\n"
            "@flyc.kernel\n"
            "def reduce_sum(\n"
            "    input: fx.Tensor,\n"
            "    output: fx.Tensor,\n"
            "    N: fx.Constexpr[int],\n"
            "):\n"
            "    bid = fx.block_idx.x\n"
            "    tid = fx.thread_idx.x\n"
            "    local_sum = fx.Float32(0.0)\n"
            "    for i in range(N):\n"
            "        local_sum = arith.addf(local_sum, input[bid, i])\n"
            "    output[bid] = local_sum\n\n\n"
            "@flyc.jit\n"
            "def reduce_sum_launch(input, output, N):\n"
            "    batch = input.shape[0]\n"
            "    reduce_sum[(batch,), (1,)](input, output, N)\n"
        ),
    },
    {
        "prompt": "Write a FlyDSL element-wise GELU activation kernel.",
        "code": (
            "import flydsl.compiler as flyc\n"
            "import flydsl.expr as fx\n"
            "from flydsl.expr import arith\n\n\n"
            "@flyc.kernel\n"
            "def gelu_kernel(\n"
            "    input: fx.Tensor,\n"
            "    output: fx.Tensor,\n"
            "    N: fx.Constexpr[int],\n"
            "):\n"
            "    tid = fx.thread_idx.x\n"
            "    bid = fx.block_idx.x\n"
            "    idx = bid * fx.block_dim.x + tid\n"
            "    if idx < N:\n"
            "        x = input[idx]\n"
            "        cdf = arith.mulf(fx.Float32(0.5), arith.addf(\n"
            "            fx.Float32(1.0), fx.tanh(arith.mulf(\n"
            "                fx.Float32(0.7978845608),\n"
            "                arith.addf(x, arith.mulf(\n"
            "                    fx.Float32(0.044715), arith.mulf(x, arith.mulf(x, x))))))))\n"
            "        output[idx] = arith.mulf(x, cdf)\n\n\n"
            "@flyc.jit\n"
            "def gelu_launch(input, output, N):\n"
            "    block = 256\n"
            "    grid = (N + block - 1) // block\n"
            "    gelu_kernel[(grid,), (block,)](input, output, N)\n"
        ),
    },
    {
        "prompt": "Write a FlyDSL SiLU (Swish) activation kernel.",
        "code": (
            "import flydsl.compiler as flyc\n"
            "import flydsl.expr as fx\n"
            "from flydsl.expr import arith\n\n\n"
            "@flyc.kernel\n"
            "def silu_kernel(\n"
            "    input: fx.Tensor,\n"
            "    output: fx.Tensor,\n"
            "    N: fx.Constexpr[int],\n"
            "):\n"
            "    tid = fx.thread_idx.x\n"
            "    bid = fx.block_idx.x\n"
            "    idx = bid * fx.block_dim.x + tid\n"
            "    if idx < N:\n"
            "        x = input[idx]\n"
            "        sigmoid_x = arith.divf(\n"
            "            fx.Float32(1.0),\n"
            "            arith.addf(fx.Float32(1.0), fx.exp(arith.negf(x))))\n"
            "        output[idx] = arith.mulf(x, sigmoid_x)\n\n\n"
            "@flyc.jit\n"
            "def silu_launch(input, output, N):\n"
            "    block = 256\n"
            "    grid = (N + block - 1) // block\n"
            "    silu_kernel[(grid,), (block,)](input, output, N)\n"
        ),
    },
    {
        "prompt": "Write a FlyDSL matrix transpose kernel.",
        "code": (
            "import flydsl.compiler as flyc\n"
            "import flydsl.expr as fx\n"
            "from flydsl.expr import buffer_ops\n"
            "from flydsl.utils.smem_allocator import SmemAllocator\n\n\n"
            "@flyc.kernel\n"
            "def transpose_kernel(\n"
            "    input: fx.Tensor,\n"
            "    output: fx.Tensor,\n"
            "    M: fx.Constexpr[int],\n"
            "    N: fx.Constexpr[int],\n"
            "):\n"
            "    bid_x = fx.block_idx.x\n"
            "    bid_y = fx.block_idx.y\n"
            "    tid = fx.thread_idx.x\n"
            "    smem = SmemAllocator()\n"
            "    tile = smem.alloc((32, 33), fx.Float32)\n"
            "    row = bid_y * 32 + tid // 32\n"
            "    col = bid_x * 32 + tid % 32\n"
            "    if row < M and col < N:\n"
            "        tile[tid // 32, tid % 32] = input[row, col]\n"
            "    fx.syncthreads()\n"
            "    out_row = bid_x * 32 + tid // 32\n"
            "    out_col = bid_y * 32 + tid % 32\n"
            "    if out_row < N and out_col < M:\n"
            "        output[out_row, out_col] = tile[tid % 32, tid // 32]\n\n\n"
            "@flyc.jit\n"
            "def transpose_launch(input, output, M, N):\n"
            "    grid = ((N + 31) // 32, (M + 31) // 32)\n"
            "    block = (32 * 32,)\n"
            "    transpose_kernel[grid, block](input, output, M, N)\n"
        ),
    },
    {
        "prompt": "Write a FlyDSL fused bias + ReLU kernel.",
        "code": (
            "import flydsl.compiler as flyc\n"
            "import flydsl.expr as fx\n"
            "from flydsl.expr import arith\n\n\n"
            "@flyc.kernel\n"
            "def bias_relu_kernel(\n"
            "    input: fx.Tensor,\n"
            "    bias: fx.Tensor,\n"
            "    output: fx.Tensor,\n"
            "    M: fx.Constexpr[int],\n"
            "    N: fx.Constexpr[int],\n"
            "):\n"
            "    bid = fx.block_idx.x\n"
            "    tid = fx.thread_idx.x\n"
            "    for j in range(N):\n"
            "        val = arith.addf(input[bid, j], bias[j])\n"
            "        output[bid, j] = arith.maximumf(val, fx.Float32(0.0))\n\n\n"
            "@flyc.jit\n"
            "def bias_relu_launch(input, bias, output, M, N):\n"
            "    bias_relu_kernel[(M,), (1,)](input, bias, output, M, N)\n"
        ),
    },
]


def infer_op(code):
    code_lower = code.lower()
    for op, keywords in OP_KEYWORDS.items():
        for kw in keywords:
            if kw in code_lower:
                return op
    return "custom"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-data", required=True, help="Base SFT dataset (v5b backup)")
    parser.add_argument("--output", required=True, help="Output dataset path")
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--boost-factor", type=int, default=3,
                        help="How many times to repeat each correct kernel with varied prompts")
    parser.add_argument("--mini-repeat", type=int, default=5,
                        help="How many times to repeat each mini-kernel")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.metadata_dir, exist_ok=True)

    # Load base data
    with open(args.sft_data) as f:
        base_data = [json.loads(l) for l in f if l.strip()]
    logger.info("Base data: %d samples", len(base_data))

    # Step 1: Extract unique correct kernels
    correct_kernels = []
    seen_hashes = set()

    for r in base_data:
        msgs = r.get("messages", [])
        assistant_code = None
        user_msg = None
        for m in msgs:
            if m["role"] == "assistant":
                assistant_code = m["content"]
            elif m["role"] == "user":
                user_msg = m["content"]

        if not assistant_code:
            continue

        has_flyc = "import flydsl.compiler as flyc" in assistant_code
        has_fx = "import flydsl.expr as fx" in assistant_code
        has_kernel = "@flyc.kernel" in assistant_code
        has_bad = bool(re.search(
            r"from flydsl\.expr import fx|from flydsl\.compiler import flyc|from flydsl import kernel",
            assistant_code))

        if has_flyc and has_fx and has_kernel and not has_bad:
            code_hash = hashlib.md5(assistant_code.encode()).hexdigest()
            if code_hash not in seen_hashes:
                seen_hashes.add(code_hash)
                op = infer_op(assistant_code)
                correct_kernels.append({
                    "code": assistant_code,
                    "user_msg": user_msg or f"Write a FlyDSL {op} kernel for gfx950.",
                    "op": op,
                })

    logger.info("Unique correct kernels: %d", len(correct_kernels))

    # Step 2: Create boosted pairs with varied prompts
    boosted = []
    for kernel in correct_kernels:
        op = kernel["op"]
        code = kernel["code"]

        for i in range(args.boost_factor):
            prompt_template = KERNEL_PROMPTS[i % len(KERNEL_PROMPTS)]
            prompt = prompt_template.format(op=op)
            boosted.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": code},
                ],
                "source": "boost_correct_kernel",
                "metadata": {"operator": op, "variant": i},
            })

    logger.info("Boosted kernel pairs: %d (from %d × %d)",
                len(boosted), len(correct_kernels), args.boost_factor)

    # Step 3: Mini-kernels (short, import-focused)
    mini_pairs = []
    for mk in MINI_KERNELS:
        for _ in range(args.mini_repeat):
            mini_pairs.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": mk["prompt"]},
                    {"role": "assistant", "content": mk["code"]},
                ],
                "source": "boost_mini_kernel",
            })

    logger.info("Mini kernel pairs: %d (%d templates × %d)",
                len(mini_pairs), len(MINI_KERNELS), args.mini_repeat)

    # Step 4: flyc boundary examples (flyc only has kernel/jit/compile)
    flyc_boundary = []
    flyc_prompts = [
        ("What decorators does flyc provide?",
         "flyc (flydsl.compiler) provides exactly 3 things:\n\n"
         "1. @flyc.kernel — decorate a GPU kernel function\n"
         "2. @flyc.jit — decorate a JIT-compiled host/launch function\n"
         "3. flyc.compile — compile a kernel to native code\n\n"
         "Example:\n"
         "  import flydsl.compiler as flyc\n"
         "  import flydsl.expr as fx\n\n"
         "  @flyc.kernel\n"
         "  def my_kernel(A: fx.Tensor, N: fx.Constexpr[int]):\n"
         "      tid = fx.thread_idx.x\n"
         "      pass\n\n"
         "  @flyc.jit\n"
         "  def launch(A, N):\n"
         "      my_kernel[(N // 256,), (256,)](A, N)\n\n"
         "There is NOTHING else in flyc. No flyc.load, flyc.launch, flyc.barrier, "
         "flyc.kernel_context, flyc.SmemAllocator, flyc.Config, flyc.pipeline, "
         "flyc.build, flyc.compile_kernel, flyc.get_shared_memory, flyc.smem_allocator."),
        ("How do I launch a FlyDSL kernel?",
         "Use @flyc.jit to create a launch function:\n\n"
         "  import flydsl.compiler as flyc\n"
         "  import flydsl.expr as fx\n\n"
         "  @flyc.kernel\n"
         "  def my_kernel(A: fx.Tensor, B: fx.Tensor, N: fx.Constexpr[int]):\n"
         "      tid = fx.thread_idx.x\n"
         "      bid = fx.block_idx.x\n"
         "      idx = bid * fx.block_dim.x + tid\n"
         "      if idx < N:\n"
         "          B[idx] = A[idx] * 2.0\n\n"
         "  @flyc.jit\n"
         "  def launch(A, B, N):\n"
         "      block = 256\n"
         "      grid = (N + block - 1) // block\n"
         "      my_kernel[(grid,), (block,)](A, B, N)\n\n"
         "Do NOT use flyc.launch() — it does not exist.\n"
         "Do NOT use flyc.kernel_context — it does not exist.\n"
         "The launch is done inside @flyc.jit via kernel[grid, block](args)."),
    ]
    for q, a in flyc_prompts:
        for _ in range(3):
            flyc_boundary.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
                "source": "boost_flyc_boundary",
            })

    logger.info("flyc boundary pairs: %d", len(flyc_boundary))

    # Save metadata
    all_new = boosted + mini_pairs + flyc_boundary
    gen_path = os.path.join(args.metadata_dir, "sft_v5e_boost.jsonl")
    with open(gen_path, "w") as f:
        for p in all_new:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info("Total new pairs: %d", len(all_new))

    # Combine: base + boosted + mini + flyc
    combined = base_data + all_new
    random.shuffle(combined)

    # Backup
    backup = args.output + ".v5d.bak"
    if not os.path.exists(backup) and os.path.exists(args.output):
        shutil.copy2(args.output, backup)

    with open(args.output, "w") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    from collections import Counter
    src = Counter(r.get("source", "?") for r in combined)
    total = len(combined)

    # Count correct kernel ratio
    correct_count = 0
    for r in combined:
        for m in r.get("messages", []):
            if m["role"] == "assistant":
                code = m["content"]
                if ("import flydsl.compiler as flyc" in code and
                    "import flydsl.expr as fx" in code and
                    "@flyc.kernel" in code and
                    not re.search(r"from flydsl\.expr import fx|from flydsl\.compiler import flyc", code)):
                    correct_count += 1
                    break

    logger.info("Output: %d samples", total)
    logger.info("  Correct kernel ratio: %d/%d = %.1f%%", correct_count, total, correct_count / total * 100)
    logger.info("  Boost sources:")
    for s in ["boost_correct_kernel", "boost_mini_kernel", "boost_flyc_boundary"]:
        logger.info("    %s: %d (%.1f%%)", s, src.get(s, 0), src.get(s, 0) / total * 100)


if __name__ == "__main__":
    main()
