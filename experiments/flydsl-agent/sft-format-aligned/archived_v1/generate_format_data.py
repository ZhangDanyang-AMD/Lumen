#!/usr/bin/env python3
"""Generate format-aligned SFT data: <plan>...</plan><code>...</code> dual-segment.

Three categories:
  Cat 1 (~55%): FlyDSL kernel samples from v5e, reverse-annotated with <plan> via Claude
  Cat 2 (~30%): General reasoning (Python algorithms, math) in plan+code format
  Cat 3 (~15%): Long CoT migration to dual-segment format

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python generate_format_data.py \
        --sft-data /home/danyzhan/flydsl-agent-dataset/data/sft/train-00000-of-00001.jsonl \
        --output /home/danyzhan/flydsl-agent-dataset/data/format_aligned/train.jsonl \
        --val-output /home/danyzhan/flydsl-agent-dataset/data/format_aligned/validation.jsonl \
        --max-cat1 1400 --max-cat2 800 --max-cat3 400

    # Or read key from file:
    python generate_format_data.py --api-key-file /path/to/key.txt ...
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import time
from pathlib import Path

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Model name: prefer the AMD proxy Sonnet alias if set, else public model id.
CLAUDE_MODEL = os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "claude-sonnet-4-20250514")


def parse_custom_headers():
    """Parse ANTHROPIC_CUSTOM_HEADERS (used by AMD's proxy for the subscription key).

    The Python SDK does not read this env var automatically (only Claude Code does),
    so we forward it as default_headers. Format: "Key: Value" pairs, one per line or
    separated by commas.
    """
    raw = os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "").strip()
    if not raw:
        return {}
    headers = {}
    for part in re.split(r"[\n,]", raw):
        part = part.strip()
        if not part or ":" not in part:
            continue
        key, _, value = part.partition(":")
        headers[key.strip()] = value.strip()
    return headers

# ── Constants ────────────────────────────────────────────────────────────────

FORMAT_SYSTEM_PROMPT = (
    "You are a FlyDSL GPU kernel programming expert for AMD Instinct GPUs. "
    "Always structure your response as:\n"
    "<plan>\n"
    "  1. Problem analysis and hardware constraints\n"
    "  2. Tiling decisions and why\n"
    "  3. Memory layout and pipeline strategy\n"
    "  4. Optimization choices (swizzle, etc.)\n"
    "</plan>\n"
    "<code>\n"
    "  Complete, compilable FlyDSL kernel code\n"
    "</code>\n\n"
    "The <plan> section should explain your reasoning in natural language. "
    "The <code> section should contain ONLY the Python code, no markdown blocks."
)

KERNEL_SOURCES = {
    "boost_correct_kernel", "gfx950_kernel_real", "augmentation_hardware",
    "augmentation_tile", "augmentation_pipeline", "kernel_reverse_annotation",
    "kernel_code_synthesis", "performance_improvement", "module_digest_correct_kernel",
    "gluon_tutorial_kernel",
}

CAT1_REVERSE_PROMPT = """\
You are analyzing a FlyDSL GPU kernel to extract the optimization reasoning.

Given:
- User specification: {user_prompt}
- Kernel code (already correct and compilable):
{kernel_code}

Write a <plan> section that reverse-engineers the key design decisions made in this kernel:
1. Problem decomposition: what the kernel computes, data types, target hardware
2. Tiling strategy: what tile sizes were chosen (extract from code constants) and why
3. Memory hierarchy: LDS usage, pipeline staging depth, preload strategy
4. Conflict avoidance: swizzle pattern if any, and why it helps

Rules:
- Be concise (4-8 sentences total). Focus on WHY, not WHAT.
- Reference actual constants from the code (tile sizes, stage count, etc.)
- Do NOT reproduce the code in the plan — only reference decisions
- Use vague language for hardware specs ("multi-bank architecture", "high-bandwidth LDS")
- Output ONLY the <plan>...</plan> content (no tags, no code)"""

CAT2_PROBLEMS = [
    # Python algorithms
    {"topic": "binary search", "prompt": "Implement binary search that finds the leftmost occurrence of a target in a sorted array."},
    {"topic": "merge sort", "prompt": "Implement merge sort for a list of integers with O(n log n) time complexity."},
    {"topic": "BFS shortest path", "prompt": "Implement BFS to find the shortest path in an unweighted graph represented as adjacency list."},
    {"topic": "dynamic programming", "prompt": "Solve the 0/1 knapsack problem using dynamic programming. Given weights, values, and capacity, return maximum value."},
    {"topic": "trie", "prompt": "Implement a Trie data structure with insert, search, and startsWith operations."},
    {"topic": "LRU cache", "prompt": "Implement an LRU cache with O(1) get and put operations using a dict and doubly-linked list."},
    {"topic": "topological sort", "prompt": "Implement topological sort for a DAG using DFS. Detect cycles."},
    {"topic": "union find", "prompt": "Implement Union-Find with path compression and union by rank for connected components."},
    {"topic": "segment tree", "prompt": "Implement a segment tree for range sum queries with point updates."},
    {"topic": "heap", "prompt": "Implement a min-heap with push, pop, and heapify operations from scratch."},
    {"topic": "sliding window max", "prompt": "Find the maximum in each sliding window of size k using a monotonic deque."},
    {"topic": "matrix chain", "prompt": "Solve matrix chain multiplication using dynamic programming to minimize scalar multiplications."},
    {"topic": "string matching", "prompt": "Implement KMP string matching algorithm with failure function computation."},
    {"topic": "interval merge", "prompt": "Merge overlapping intervals in a list of [start, end] pairs."},
    {"topic": "graph coloring", "prompt": "Implement greedy graph coloring to find a valid k-coloring of an undirected graph."},
    # Math / logic reasoning
    {"topic": "prime sieve", "prompt": "Implement the Sieve of Eratosthenes to find all primes up to N."},
    {"topic": "modular exponentiation", "prompt": "Implement fast modular exponentiation (a^b mod m) using binary method."},
    {"topic": "matrix multiply", "prompt": "Implement matrix multiplication for two NxN matrices using the naive O(n^3) algorithm."},
    {"topic": "Newton's method", "prompt": "Implement Newton's method to find the square root of a number to a given precision."},
    {"topic": "FFT", "prompt": "Implement the Cooley-Tukey radix-2 FFT algorithm for a power-of-2 length signal."},
]

CAT3_COT_PROMPT = """\
You are converting a complex programming problem into the <plan>+<code> dual-segment format.

Problem: {problem}

Generate a response in exactly this format:
<plan>
  Step-by-step reasoning about the solution approach, considering:
  - Problem constraints and edge cases
  - Algorithm selection and complexity analysis
  - Key implementation decisions
  (Natural language, 4-10 sentences, may include brief pseudocode)
</plan>
<code>
  Complete, correct Python implementation
</code>

Output the <plan> and <code> sections. No other text."""


# ── Category 1: FlyDSL kernel reverse annotation ────────────────────────────

def filter_kernel_samples(sft_data):
    """Select kernel-containing samples suitable for plan reverse-annotation."""
    candidates = []
    for sample in sft_data:
        source = sample.get("source", "")
        if source not in KERNEL_SOURCES:
            continue
        assistant = sample["messages"][-1]["content"]
        if len(assistant) < 200:
            continue
        if not re.search(r"@flyc\.|import flydsl|fx\.\w+", assistant):
            continue
        candidates.append(sample)
    return candidates


def is_complete_dual_segment(text):
    """Require a full, non-truncated <plan>...</plan><code>...</code> structure.

    Guards against responses truncated at max_tokens (which lack a closing
    </code> tag) — those must not enter the training set.
    """
    return bool(
        re.search(r"<plan>.*?</plan>", text, re.S)
        and re.search(r"<code>.*?</code>", text, re.S)
    )


def _extract_code_decisions(code_text):
    """Extract structural design decisions from kernel code.

    Covers GEMM-type kernels (tile sizes, pipeline, swizzle) AND
    element-wise/normalization kernels (BLOCK_THREADS, VEC_WIDTH, warp ops).
    """
    decisions = {}

    # Tile / block sizes — broad matching for both GEMM and non-GEMM kernels
    tile_pattern = (
        r"(?:BLOCK_SIZE_[MNK]|BLOCK_[MNK]|BLOCK_SIZE|BLOCK_THREADS|"
        r"tile_[mnk]|B[MNK]|TILE_[MNK]|"
        r"VEC_WIDTH|VEC|NUM_WARPS|WARP_SIZE)\s*[=:]\s*(\d+)"
    )
    for m in re.finditer(tile_pattern, code_text):
        key = m.group(0).split("=")[0].split(":")[0].strip().upper()
        decisions.setdefault("tiles", []).append((key, int(m.group(1))))

    # Pipeline / LDS stages
    if re.search(r"lds_stage|num_stages|NUM_STAGES|NUM_PREFETCH", code_text):
        for m in re.finditer(
            r"(?:lds_stage|num_stages|NUM_STAGES|NUM_PREFETCH_K)\s*[=:]\s*(\d+)", code_text
        ):
            decisions["pipeline_stages"] = int(m.group(1))

    # Swizzle pattern
    if re.search(r"swizzle|SWIZZLE", code_text, re.I):
        decisions["swizzle"] = True

    # SmemAllocator / shared memory / LDS
    if re.search(r"SmemAllocator|smem_alloc|flyc\.Stage|lds_stage", code_text):
        decisions["smem"] = True

    # MFMA instructions
    if re.search(r"rocdl\.mfma|mfma_|MFMA_LANE", code_text):
        decisions["mfma"] = True

    # Split-K
    if re.search(r"split.?k|SPLIT.?K|num_ksplit|NUM_KSPLIT", code_text, re.I):
        decisions["splitk"] = True

    # Vectorization (important for non-GEMM kernels)
    if re.search(r"VEC_WIDTH|VEC\s*=\s*\d|buffer_load.*x[248]|DMA_BYTES", code_text):
        decisions["vectorization"] = True

    # Warp-level ops (reduction kernels)
    if re.search(r"warp.*reduce|warp.*shuffle|DPP|dpp_|cross_lane|permute", code_text, re.I):
        decisions["warp_ops"] = True

    return decisions


def validate_plan_code_consistency(plan_text, code_text):
    """Verify plan segment references actual design decisions from the code.

    Three checks:
    1. Tile sizes mentioned in plan must appear in code
    2. If code uses pipeline/swizzle/splitk, plan must mention them
    3. Plan must not be hollow (>= 3 substantive sentences)
    """
    decisions = _extract_code_decisions(code_text)
    plan_lower = plan_text.lower()
    score = 0
    checks = 0

    # Check 1: tile sizes — plan numbers should match code constants
    if decisions.get("tiles"):
        checks += 1
        code_tile_values = {str(v) for _, v in decisions["tiles"]}
        plan_numbers = set(re.findall(r"\b(\d{2,4})\b", plan_text))
        if code_tile_values & plan_numbers:
            score += 1

    # Check 2: structural features — if code uses them, plan should mention them
    feature_keywords = {
        "pipeline_stages": ["pipeline", "stage", "double.?buffer", "ping.?pong", "prefetch"],
        "swizzle": ["swizzle", "xor", "bank.?conflict", "conflict.?free"],
        "smem": ["shared.?memory", "lds", "smem", "local.?data", "scratch"],
        "mfma": ["mfma", "matrix.?fma", "matrix.?multiply", "wmma", "matmul"],
        "splitk": ["split.?k", "k.?split", "k.?partition", "partial.?sum"],
        "vectorization": ["vector", "coalesce", "vec.*width", "wide.*load", "bulk.*load"],
        "warp_ops": ["warp", "wave", "shuffle", "cross.?lane", "reduction", "dpp"],
    }
    for feature, keywords in feature_keywords.items():
        if feature in decisions:
            checks += 1
            if any(re.search(kw, plan_lower) for kw in keywords):
                score += 1

    # Check 3: plan is substantive (not hollow template)
    checks += 1
    sentences = [s.strip() for s in re.split(r"[.。\n]", plan_text) if len(s.strip()) > 20]
    if len(sentences) >= 3:
        score += 1

    if checks == 0:
        return True
    return score / checks >= 0.6


# ── Category 2: General reasoning ───────────────────────────────────────────

CAT2_GEN_PROMPT = """\
Generate a programming problem and solution in <plan>+<code> dual-segment format.

Topic: {topic}
Task: {prompt}

Your response MUST be in exactly this format:
<plan>
  1. Problem analysis: what we need to solve and constraints
  2. Algorithm choice: which approach and why (complexity analysis)
  3. Key implementation details: edge cases, data structures
</plan>
<code>
  Complete, correct Python implementation (no markdown, just raw code)
</code>

Output ONLY the <plan> and <code> sections."""


# ── Category 3: Complex reasoning migration ─────────────────────────────────

CAT3_PROBLEMS = [
    "Implement a red-black tree with insert, delete, and search. Handle all rotation cases.",
    "Write a concurrent producer-consumer queue using threading locks and condition variables.",
    "Implement Dijkstra's shortest path with a Fibonacci heap for O(V log V + E) complexity.",
    "Build a simple regex engine supporting ., *, +, ?, and character classes [abc].",
    "Implement A* pathfinding on a 2D grid with obstacles and diagonal movement.",
    "Write a B-tree implementation with insert, search, and split operations for order m=5.",
    "Implement the Aho-Corasick algorithm for multi-pattern string matching.",
    "Build a skip list with probabilistic balancing for O(log n) search/insert/delete.",
    "Implement a persistent (immutable) balanced BST with path copying.",
    "Write a Bloom filter with configurable false-positive rate and optimal hash count.",
    "Implement the Bellman-Ford algorithm with negative cycle detection.",
    "Build a suffix array with the DC3/skew algorithm for O(n) construction.",
    "Implement a thread-safe memory pool allocator with free list management.",
    "Write a simple garbage collector using mark-and-sweep for a toy language runtime.",
    "Implement the Hungarian algorithm for optimal bipartite matching.",
    "Build an interval tree supporting insert, delete, and overlap queries.",
    "Implement Tarjan's algorithm for finding strongly connected components.",
    "Write an LZ77 compression/decompression pair with sliding window.",
    "Implement a lock-free stack using CAS (compare-and-swap) operations.",
    "Build a simple JIT compiler that converts arithmetic expressions to x86-64 machine code.",
]


# ── Claude API helpers ───────────────────────────────────────────────────────

def get_api_key(args):
    if args.api_key:
        return args.api_key
    if args.api_key_file and os.path.exists(args.api_key_file):
        return Path(args.api_key_file).read_text().strip()
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key and key != "dummy":
        return key
    # A proxy (e.g. AMD's llm-api) authenticates via custom header + base_url,
    # so a "dummy" placeholder key is acceptable in that case.
    if parse_custom_headers() or os.environ.get("ANTHROPIC_BASE_URL"):
        return key or "dummy"
    raise ValueError(
        "No API key found. Set ANTHROPIC_API_KEY env var, "
        "or pass --api-key or --api-key-file."
    )


async def call_claude(client, prompt, system="", max_tokens=4096, retries=3):
    """Call Claude API with retries and rate limiting."""
    import anthropic

    for attempt in range(retries):
        try:
            messages = [{"role": "user", "content": prompt}]
            kwargs = {
                "model": CLAUDE_MODEL,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            if system:
                kwargs["system"] = system
            response = await client.messages.create(**kwargs)
            return response.content[0].text
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 5
            logger.warning("Rate limited, waiting %ds...", wait)
            await asyncio.sleep(wait)
        except Exception as e:
            logger.error("API error (attempt %d): %s", attempt + 1, e)
            if attempt < retries - 1:
                await asyncio.sleep(2)
    return None


async def generate_cat1_batch(client, samples, semaphore, max_kernel_chars):
    """Generate <plan> sections for kernel samples."""
    results = []

    async def process_one(sample):
        async with semaphore:
            user_prompt = sample["messages"][1]["content"]
            kernel_code = sample["messages"][-1]["content"]

            # Keep kernels intact up to the training-window budget; only truncate
            # the ones too large to fit max_seq_length (they'd be cut at train time
            # regardless). max_kernel_chars<=0 disables truncation entirely.
            if 0 < max_kernel_chars < len(kernel_code):
                kernel_code = kernel_code[:max_kernel_chars] + "\n# ... (truncated)"

            prompt = CAT1_REVERSE_PROMPT.format(
                user_prompt=user_prompt[:500],
                kernel_code=kernel_code,
            )

            plan_text = await call_claude(client, prompt, max_tokens=1024)
            if not plan_text:
                return None

            plan_text = plan_text.strip()
            # Remove any tags Claude might have added
            plan_text = re.sub(r"</?plan>", "", plan_text).strip()

            if not validate_plan_code_consistency(plan_text, kernel_code):
                logger.debug("Skipping inconsistent plan for %s", sample.get("source"))
                return None

            # Build the format-aligned response
            formatted_response = f"<plan>\n{plan_text}\n</plan>\n<code>\n{kernel_code}\n</code>"

            return {
                "messages": [
                    {"role": "system", "content": FORMAT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": formatted_response},
                ],
                "source": f"format_aligned_cat1_{sample.get('source', 'unknown')}",
            }

    tasks = [process_one(s) for s in samples]
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            results.append(result)
            if len(results) % 50 == 0:
                logger.info("  Cat 1: %d / %d generated", len(results), len(samples))

    return results


async def generate_cat2_batch(client, problems, semaphore):
    """Generate general reasoning samples in plan+code format."""
    results = []

    async def process_one(problem):
        async with semaphore:
            prompt = CAT2_GEN_PROMPT.format(
                topic=problem["topic"],
                prompt=problem["prompt"],
            )
            response = await call_claude(client, prompt, max_tokens=4096)
            if not response:
                return None

            response = response.strip()
            # Require complete, non-truncated dual-segment structure.
            if not is_complete_dual_segment(response):
                return None

            return {
                "messages": [
                    {"role": "system", "content": FORMAT_SYSTEM_PROMPT},
                    {"role": "user", "content": problem["prompt"]},
                    {"role": "assistant", "content": response},
                ],
                "source": f"format_aligned_cat2_{problem['topic']}",
            }

    tasks = [process_one(p) for p in problems]
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            results.append(result)

    return results


async def generate_cat3_batch(client, problems, semaphore):
    """Generate complex CoT samples in plan+code format."""
    results = []

    async def process_one(problem):
        async with semaphore:
            prompt = CAT3_COT_PROMPT.format(problem=problem)
            response = await call_claude(client, prompt, max_tokens=4096)
            if not response:
                return None

            response = response.strip()
            if not is_complete_dual_segment(response):
                return None

            return {
                "messages": [
                    {"role": "system", "content": FORMAT_SYSTEM_PROMPT},
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": response},
                ],
                "source": "format_aligned_cat3_cot",
            }

    tasks = [process_one(p) for p in problems]
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            results.append(result)

    return results


def expand_cat2_problems(base_problems, target_count):
    """Expand problem list by generating variations."""
    expanded = list(base_problems)
    variations = [
        "with error handling and input validation",
        "optimized for large inputs",
        "with detailed docstrings",
        "using only built-in Python (no imports)",
    ]
    while len(expanded) < target_count:
        base = random.choice(base_problems)
        var = random.choice(variations)
        expanded.append({
            "topic": base["topic"],
            "prompt": f"{base['prompt']} ({var})",
        })
    return expanded[:target_count]


def expand_cat3_problems(base_problems, target_count):
    """Expand complex problem list."""
    expanded = list(base_problems)
    while len(expanded) < target_count:
        expanded.append(random.choice(base_problems))
    return expanded[:target_count]


# ── Main ─────────────────────────────────────────────────────────────────────

async def async_main(args):
    import anthropic

    api_key = get_api_key(args)
    custom_headers = parse_custom_headers()
    client_kwargs = {"api_key": api_key}
    if custom_headers:
        client_kwargs["default_headers"] = custom_headers
    # base_url is picked up automatically from ANTHROPIC_BASE_URL by the SDK.
    client = anthropic.AsyncAnthropic(**client_kwargs)
    logger.info("Using model=%s base_url=%s custom_headers=%s",
                CLAUDE_MODEL, os.environ.get("ANTHROPIC_BASE_URL", "(default)"),
                list(custom_headers.keys()) or "(none)")
    semaphore = asyncio.Semaphore(args.concurrency)

    # Load SFT data
    logger.info("Loading SFT data from %s ...", args.sft_data)
    sft_data = []
    with open(args.sft_data) as f:
        for line in f:
            sft_data.append(json.loads(line))
    logger.info("  Total SFT samples: %d", len(sft_data))

    # ── Category 1: kernel reverse annotation ──
    kernel_samples = filter_kernel_samples(sft_data)
    random.shuffle(kernel_samples)
    kernel_samples = kernel_samples[:args.max_cat1]
    logger.info("Cat 1: %d kernel samples selected for plan annotation", len(kernel_samples))

    cat1_results = await generate_cat1_batch(client, kernel_samples, semaphore, args.max_kernel_chars)
    logger.info("Cat 1: %d samples generated", len(cat1_results))

    # ── Category 2: general reasoning ──
    cat2_problems = expand_cat2_problems(CAT2_PROBLEMS, args.max_cat2)
    logger.info("Cat 2: generating %d general reasoning samples", len(cat2_problems))

    cat2_results = await generate_cat2_batch(client, cat2_problems, semaphore)
    logger.info("Cat 2: %d samples generated", len(cat2_results))

    # ── Category 3: complex CoT ──
    cat3_problems = expand_cat3_problems(CAT3_PROBLEMS, args.max_cat3)
    logger.info("Cat 3: generating %d complex reasoning samples", len(cat3_problems))

    cat3_results = await generate_cat3_batch(client, cat3_problems, semaphore)
    logger.info("Cat 3: %d samples generated", len(cat3_results))

    # ── Combine and split ──
    all_data = cat1_results + cat2_results + cat3_results
    random.shuffle(all_data)

    logger.info("Total format-aligned samples: %d (cat1=%d, cat2=%d, cat3=%d)",
                len(all_data), len(cat1_results), len(cat2_results), len(cat3_results))

    # 90/10 train/val split
    val_size = max(1, int(len(all_data) * 0.1))
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info("Train data: %d samples → %s", len(train_data), args.output)

    if args.val_output:
        os.makedirs(os.path.dirname(args.val_output), exist_ok=True)
        with open(args.val_output, "w") as f:
            for sample in val_data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info("Val data: %d samples → %s", len(val_data), args.val_output)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Format-Aligned Data Generation Complete")
    print(f"{'='*60}")
    print(f"  Category 1 (FlyDSL kernel):   {len(cat1_results)}")
    print(f"  Category 2 (General reasoning): {len(cat2_results)}")
    print(f"  Category 3 (Complex CoT):      {len(cat3_results)}")
    print(f"  Total:                         {len(all_data)}")
    print(f"  Train:                         {len(train_data)}")
    print(f"  Validation:                    {len(val_data)}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Generate format-aligned SFT data")
    parser.add_argument("--sft-data", required=True,
                        help="Path to v5e SFT training JSONL")
    parser.add_argument("--output", required=True,
                        help="Output path for training JSONL")
    parser.add_argument("--val-output", default=None,
                        help="Output path for validation JSONL")
    parser.add_argument("--api-key", default=None, help="Anthropic API key")
    parser.add_argument("--api-key-file", default=None,
                        help="Path to file containing Anthropic API key")
    parser.add_argument("--max-cat1", type=int, default=1400,
                        help="Max category 1 samples (kernel reverse annotation)")
    parser.add_argument("--max-cat2", type=int, default=800,
                        help="Max category 2 samples (general reasoning)")
    parser.add_argument("--max-cat3", type=int, default=400,
                        help="Max category 3 samples (complex CoT)")
    parser.add_argument("--max-kernel-chars", type=int, default=48000,
                        help="Truncate Cat 1 kernels longer than this many chars "
                             "(sized to fit max_seq_length=16384; <=0 disables truncation)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent API calls")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
