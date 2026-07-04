#!/usr/bin/env python3
"""Generate v5f SFT dataset: v5e data (untouched) + dual-segment copies + cat2/cat3.

Strategy: keep ALL v5e data verbatim (preserving original training signal),
and ADD new dual-segment (<plan>+<code>) copies of kernel samples as separate
entries. The model learns both:
  - Original system prompt → raw kernel code  (retains v5e ability)
  - Format system prompt → <plan>+<code>      (learns dual-segment format)

Data composition:
  - v5e ALL samples verbatim (~3889, original format, untouched)
  - v5e kernel dual-segment copies (~1500, new entries with plan tags)
  - Cat2: general reasoning in <plan>+<code> format (~700)
  - Cat3: complex CoT in <plan>+<code> format (~350)
  Total: ~6400 samples

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python generate_v5f_data.py \
        --sft-data /home/danyzhan/flydsl-agent-dataset/data/sft/train-00000-of-00001.jsonl \
        --val-data /home/danyzhan/flydsl-agent-dataset/data/sft/validation-00000-of-00001.jsonl \
        --output /home/danyzhan/flydsl-agent-dataset/data/format_aligned/train.jsonl \
        --val-output /home/danyzhan/flydsl-agent-dataset/data/format_aligned/validation.jsonl
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
from pathlib import Path

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CLAUDE_MODEL = os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "claude-sonnet-4-20250514")


def parse_custom_headers():
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


# ── System prompts ───────────────────────────────────────────────────────────

FORMAT_SYSTEM_PROMPT = (
    "You are a FlyDSL GPU kernel programming expert for AMD Instinct GPUs. "
    "Structure your response in two sections: "
    "first a <plan> section with brief optimization reasoning (under 200 words), "
    "then a <code> section with complete FlyDSL kernel code. "
    "Output raw Python code in the <code> section, never markdown blocks."
)

# ── Kernel source classification ─────────────────────────────────────────────

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

Write a concise plan section (4-8 sentences, under 200 words) that explains:
1. What the kernel computes and key hardware constraints
2. Tiling/blocking strategy: actual tile sizes from the code and why they were chosen
3. Memory hierarchy: LDS usage, pipeline depth, prefetch strategy (if used)
4. Conflict avoidance: swizzle pattern (if used) and why

Rules:
- Reference actual constants from the code (tile sizes, stage count, etc.)
- Explain WHY each decision was made, not just WHAT
- Do NOT reproduce the code — only reference decisions
- Use vague language for hardware specs ("multi-bank architecture", not "64 banks")
- Output ONLY the plan text (no <plan> tags, no code)"""


# ── Code decision extraction and plan-code consistency ───────────────────────

def _extract_code_decisions(code_text):
    """Extract structural design decisions from kernel code."""
    decisions = {}
    tile_pattern = (
        r"(?:BLOCK_SIZE_[MNK]|BLOCK_[MNK]|BLOCK_SIZE|BLOCK_THREADS|"
        r"tile_[mnk]|B[MNK]|TILE_[MNK]|"
        r"VEC_WIDTH|VEC|NUM_WARPS|WARP_SIZE)\s*[=:]\s*(\d+)"
    )
    for m in re.finditer(tile_pattern, code_text):
        key = m.group(0).split("=")[0].split(":")[0].strip().upper()
        decisions.setdefault("tiles", []).append((key, int(m.group(1))))
    if re.search(r"lds_stage|num_stages|NUM_STAGES|NUM_PREFETCH", code_text):
        for m in re.finditer(
            r"(?:lds_stage|num_stages|NUM_STAGES|NUM_PREFETCH_K)\s*[=:]\s*(\d+)", code_text
        ):
            decisions["pipeline_stages"] = int(m.group(1))
    if re.search(r"swizzle|SWIZZLE", code_text, re.I):
        decisions["swizzle"] = True
    if re.search(r"SmemAllocator|smem_alloc|flyc\.Stage|lds_stage", code_text):
        decisions["smem"] = True
    if re.search(r"rocdl\.mfma|mfma_|MFMA_LANE", code_text):
        decisions["mfma"] = True
    if re.search(r"split.?k|SPLIT.?K|num_ksplit|NUM_KSPLIT", code_text, re.I):
        decisions["splitk"] = True
    if re.search(r"VEC_WIDTH|VEC\s*=\s*\d|buffer_load.*x[248]|DMA_BYTES", code_text):
        decisions["vectorization"] = True
    if re.search(r"warp.*reduce|warp.*shuffle|DPP|dpp_|cross_lane|permute", code_text, re.I):
        decisions["warp_ops"] = True
    return decisions


def validate_plan_code_consistency(plan_text, code_text):
    """Verify plan references actual design decisions from the code."""
    decisions = _extract_code_decisions(code_text)
    plan_lower = plan_text.lower()
    score = 0
    checks = 0

    if decisions.get("tiles"):
        checks += 1
        code_tile_values = {str(v) for _, v in decisions["tiles"]}
        plan_numbers = set(re.findall(r"\b(\d{2,4})\b", plan_text))
        if code_tile_values & plan_numbers:
            score += 1

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

    checks += 1
    sentences = [s.strip() for s in re.split(r"[.。\n]", plan_text) if len(s.strip()) > 20]
    if len(sentences) >= 3:
        score += 1

    if checks == 0:
        return True
    return score / checks >= 0.6


def is_complete_dual_segment(text):
    return bool(
        re.search(r"<plan>.*?</plan>", text, re.S)
        and re.search(r"<code>.*?</code>", text, re.S)
    )


# ── Classify v5e samples ────────────────────────────────────────────────────

def classify_v5e_sample(sample):
    """Returns 'kernel' if sample should get plan tags, else 'passthrough'."""
    source = sample.get("source", "")
    if source not in KERNEL_SOURCES:
        return "passthrough"
    assistant = sample["messages"][-1]["content"]
    if len(assistant) < 200:
        return "passthrough"
    if not re.search(r"@flyc\.|import flydsl|fx\.\w+", assistant):
        return "passthrough"
    return "kernel"


# ── Cat2/Cat3 data (reused from generate_format_data.py) ────────────────────

CAT2_PROBLEMS = [
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
    {"topic": "prime sieve", "prompt": "Implement the Sieve of Eratosthenes to find all primes up to N."},
    {"topic": "modular exponentiation", "prompt": "Implement fast modular exponentiation (a^b mod m) using binary method."},
    {"topic": "matrix multiply", "prompt": "Implement matrix multiplication for two NxN matrices using the naive O(n^3) algorithm."},
    {"topic": "Newton's method", "prompt": "Implement Newton's method to find the square root of a number to a given precision."},
    {"topic": "FFT", "prompt": "Implement the Cooley-Tukey radix-2 FFT algorithm for a power-of-2 length signal."},
]

CAT2_GEN_PROMPT = """\
Generate a programming problem and solution in <plan>+<code> dual-segment format.

Topic: {topic}
Task: {prompt}

Your response MUST be in exactly this format:
<plan>
  1. Problem analysis: what we need to solve and constraints
  2. Algorithm choice: which approach and why (complexity analysis)
  3. Key implementation details: edge cases, data structures
  (Keep under 200 words)
</plan>
<code>
  Complete, correct Python implementation (no markdown, just raw code)
</code>

Output ONLY the <plan> and <code> sections."""

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
    "Implement a splay tree with splay-on-access for amortized O(log n) operations.",
    "Build a van Emde Boas tree supporting successor/predecessor in O(log log u).",
    "Implement Kosaraju's algorithm for strongly connected components using two DFS passes.",
    "Write a topological sort with Kahn's algorithm and detect cycles in a DAG.",
    "Implement the Knuth-Morris-Pratt failure function and full pattern search.",
    "Build a Fenwick (binary indexed) tree supporting range updates and point queries.",
    "Implement Edmonds-Karp maximum flow using BFS augmenting paths.",
    "Write Dinic's algorithm for maximum flow with level graphs and blocking flows.",
    "Implement the Boyer-Moore string search with bad-character and good-suffix heuristics.",
    "Build a rope data structure for efficient string concatenation and slicing.",
    "Implement a treap (randomized BST) with split and merge operations.",
    "Write a Huffman coding encoder/decoder with a priority queue.",
    "Implement the Floyd-Warshall all-pairs shortest path with path reconstruction.",
    "Build a disjoint-set with union by size, path compression, and rollback support.",
    "Implement a suffix automaton for counting distinct substrings.",
    "Write Manacher's algorithm for the longest palindromic substring in O(n).",
    "Implement the Z-algorithm for linear-time pattern matching.",
    "Build a k-d tree for nearest-neighbor queries in 2D space.",
    "Implement a quadtree for 2D spatial region and point queries.",
    "Write a min-cost max-flow solver using SPFA (queue-based Bellman-Ford).",
    "Implement the Hopcroft-Karp algorithm for maximum bipartite matching in O(E sqrt(V)).",
    "Build a persistent segment tree supporting historical range-sum queries.",
    "Implement a wavelet tree for rank/select and range k-th element queries.",
    "Write a heavy-light decomposition for path queries on a weighted tree.",
    "Implement lowest common ancestor with Euler tour and a sparse table.",
    "Build a monotonic stack solution for the largest rectangle in a histogram.",
    "Write a Miller-Rabin primality test with deterministic witnesses for 64-bit integers.",
    "Implement Pollard's rho algorithm for integer factorization.",
    "Build an extended Euclidean algorithm and modular multiplicative inverse solver.",
    "Implement the Chinese Remainder Theorem for a system of congruences.",
    "Write matrix exponentiation to compute linear recurrences in O(log n).",
    "Implement Gaussian elimination with partial pivoting for linear systems.",
    "Build a convex hull using Andrew's monotone chain algorithm.",
    "Implement a sweep-line algorithm to report all segment intersections.",
    "Write a simulated annealing optimizer for the traveling salesman problem.",
    "Implement a genetic algorithm for function optimization with crossover and mutation.",
    "Build a backtracking Sudoku solver with constraint propagation.",
    "Implement an N-Queens solver using bitmask backtracking.",
    "Write a recursive descent parser and evaluator for arithmetic with operator precedence.",
    "Implement the shunting-yard algorithm to convert infix to postfix and evaluate it.",
    "Build a finite state machine engine that tokenizes a small language.",
    "Implement a coroutine-based cooperative scheduler using Python generators.",
    "Write a thread pool executor with a bounded work queue and graceful shutdown.",
    "Implement a rate limiter using the token bucket algorithm with timed refill.",
    "Build a consistent hashing ring with virtual nodes for load distribution.",
    "Implement an LFU cache with O(1) get/put using frequency buckets.",
    "Write a trie-based autocomplete returning top-k suggestions by frequency.",
    "Implement reservoir sampling to pick k items from a stream of unknown length.",
    "Build a count-min sketch for approximate frequency counting on a data stream.",
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
  (Natural language, 4-10 sentences, under 200 words)
</plan>
<code>
  Complete, correct Python implementation
</code>

Output the <plan> and <code> sections. No other text."""


# ── API helpers ──────────────────────────────────────────────────────────────

def get_api_key(args):
    if args.api_key:
        return args.api_key
    if args.api_key_file and os.path.exists(args.api_key_file):
        return Path(args.api_key_file).read_text().strip()
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key and key != "dummy":
        return key
    if parse_custom_headers() or os.environ.get("ANTHROPIC_BASE_URL"):
        return key or "dummy"
    raise ValueError("No API key. Set ANTHROPIC_API_KEY or pass --api-key-file.")


async def call_claude(client, prompt, system="", max_tokens=4096, retries=3):
    import anthropic
    for attempt in range(retries):
        try:
            kwargs = {
                "model": CLAUDE_MODEL,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system
            response = await client.messages.create(**kwargs)
            return response.content[0].text
        except anthropic.RateLimitError:
            await asyncio.sleep(2 ** attempt * 5)
        except Exception as e:
            logger.error("API error (attempt %d): %s", attempt + 1, e)
            if attempt < retries - 1:
                await asyncio.sleep(2)
    return None


# ── Core generation functions ────────────────────────────────────────────────

PLAN_SUFFIX_VARIANTS = [
    " Explain your tiling and pipeline decisions.",
    " Include your optimization reasoning.",
    " Describe your design choices before the code.",
    " Walk through the key design decisions first.",
]


async def annotate_kernel_sample(client, sample, semaphore, max_kernel_chars=0):
    """Create a NEW dual-segment copy of a kernel sample (original is kept separately)."""
    async with semaphore:
        user_prompt = sample["messages"][1]["content"]
        kernel_code = sample["messages"][-1]["content"]

        if 0 < max_kernel_chars < len(kernel_code):
            kernel_code = kernel_code[:max_kernel_chars] + "\n# ... (truncated)"

        prompt = CAT1_REVERSE_PROMPT.format(
            user_prompt=user_prompt[:500],
            kernel_code=kernel_code,
        )

        plan_text = await call_claude(client, prompt, max_tokens=512)
        if not plan_text:
            return None

        plan_text = plan_text.strip()
        plan_text = re.sub(r"</?plan>", "", plan_text).strip()

        if not validate_plan_code_consistency(plan_text, kernel_code):
            logger.debug("Inconsistent plan for %s, skipping", sample.get("source"))
            return None

        formatted = f"<plan>\n{plan_text}\n</plan>\n<code>\n{kernel_code}\n</code>"

        # Slightly modify user prompt so this entry is distinct from the original
        suffix = random.choice(PLAN_SUFFIX_VARIANTS)
        dual_user_prompt = user_prompt.rstrip() + suffix

        return {
            "messages": [
                {"role": "system", "content": FORMAT_SYSTEM_PROMPT},
                {"role": "user", "content": dual_user_prompt},
                {"role": "assistant", "content": formatted},
            ],
            "source": f"v5f_dual_{sample.get('source', 'unknown')}",
        }


async def generate_cat2_sample(client, problem, semaphore):
    async with semaphore:
        prompt = CAT2_GEN_PROMPT.format(topic=problem["topic"], prompt=problem["prompt"])
        response = await call_claude(client, prompt, max_tokens=4096)
        if not response or not is_complete_dual_segment(response.strip()):
            return None
        return {
            "messages": [
                {"role": "system", "content": FORMAT_SYSTEM_PROMPT},
                {"role": "user", "content": problem["prompt"]},
                {"role": "assistant", "content": response.strip()},
            ],
            "source": f"v5f_cat2_{problem['topic']}",
        }


async def generate_cat3_sample(client, problem, semaphore):
    async with semaphore:
        prompt = CAT3_COT_PROMPT.format(problem=problem)
        response = await call_claude(client, prompt, max_tokens=6144)
        if not response or not is_complete_dual_segment(response.strip()):
            return None
        return {
            "messages": [
                {"role": "system", "content": FORMAT_SYSTEM_PROMPT},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": response.strip()},
            ],
            "source": "v5f_cat3_cot",
        }


def expand_problems(base, target):
    expanded = list(base)
    variations = ["with error handling", "optimized for large inputs",
                   "with detailed docstrings", "using only built-in Python"]
    while len(expanded) < target:
        b = random.choice(base)
        v = random.choice(variations)
        if isinstance(b, dict):
            expanded.append({"topic": b["topic"], "prompt": f"{b['prompt']} ({v})"})
        else:
            expanded.append(f"{b} ({v})")
    return expanded[:target]


# ── Main ─────────────────────────────────────────────────────────────────────

async def async_main(args):
    import anthropic

    api_key = get_api_key(args)
    client_kwargs = {"api_key": api_key}
    custom_headers = parse_custom_headers()
    if custom_headers:
        client_kwargs["default_headers"] = custom_headers
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    client = anthropic.AsyncAnthropic(**client_kwargs)
    semaphore = asyncio.Semaphore(args.concurrency)

    # ── Load v5e data (ALL kept verbatim) ──
    logger.info("Loading v5e SFT data from %s", args.sft_data)
    v5e_data = []
    with open(args.sft_data) as f:
        for line in f:
            v5e_data.append(json.loads(line))
    logger.info("  Total v5e samples: %d (all kept verbatim)", len(v5e_data))

    # Identify kernel samples for dual-segment copy generation
    kernel_samples = [s for s in v5e_data if classify_v5e_sample(s) == "kernel"]
    logger.info("  Kernel samples (will get dual-segment COPIES): %d", len(kernel_samples))
    logger.info("  Non-kernel samples: %d", len(v5e_data) - len(kernel_samples))

    # ── Generate dual-segment COPIES of kernel samples ──
    random.shuffle(kernel_samples)
    max_kernels = min(len(kernel_samples), args.max_kernel)
    kernel_batch = kernel_samples[:max_kernels]
    logger.info("Generating %d dual-segment copies...", len(kernel_batch))

    tasks = [annotate_kernel_sample(client, s, semaphore, args.max_kernel_chars)
             for s in kernel_batch]
    annotated = []
    done = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        done += 1
        if result:
            annotated.append(result)
        if done % 100 == 0:
            logger.info("  Kernel annotation: %d/%d done, %d passed consistency",
                        done, len(kernel_batch), len(annotated))

    logger.info("Kernel annotation complete: %d/%d passed (%d dropped for plan↔code inconsistency)",
                len(annotated), len(kernel_batch), len(kernel_batch) - len(annotated))

    # ── Cat2: general reasoning ──
    cat2_problems = expand_problems(CAT2_PROBLEMS, args.max_cat2)
    logger.info("Generating %d cat2 (general reasoning) samples...", len(cat2_problems))
    cat2_tasks = [generate_cat2_sample(client, p, semaphore) for p in cat2_problems]
    cat2_results = []
    for coro in asyncio.as_completed(cat2_tasks):
        result = await coro
        if result:
            cat2_results.append(result)
    logger.info("Cat2 complete: %d samples", len(cat2_results))

    # ── Cat3: complex CoT ──
    cat3_problems = expand_problems(CAT3_PROBLEMS, args.max_cat3)
    logger.info("Generating %d cat3 (complex CoT) samples...", len(cat3_problems))
    cat3_tasks = [generate_cat3_sample(client, p, semaphore) for p in cat3_problems]
    cat3_results = []
    for coro in asyncio.as_completed(cat3_tasks):
        result = await coro
        if result:
            cat3_results.append(result)
    logger.info("Cat3 complete: %d samples", len(cat3_results))

    # ── Assemble v5f dataset ──
    v5f_train = []

    # 1. ALL v5e data verbatim (original training signal preserved)
    v5f_train.extend(v5e_data)

    # 2. Dual-segment COPIES of kernel samples (new entries, not replacements)
    v5f_train.extend(annotated)

    # 3. Cat2 + Cat3 reasoning samples
    v5f_train.extend(cat2_results)
    v5f_train.extend(cat3_results)

    random.shuffle(v5f_train)

    logger.info("\n=== v5f Dataset Composition ===")
    logger.info("  v5e verbatim (ALL):        %d", len(v5e_data))
    logger.info("  Dual-segment copies (NEW): %d", len(annotated))
    logger.info("  Cat2 (reasoning):          %d", len(cat2_results))
    logger.info("  Cat3 (CoT):                %d", len(cat3_results))
    logger.info("  Total train:               %d", len(v5f_train))

    # ── Write train ──
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for sample in v5f_train:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info("Train → %s (%d samples)", args.output, len(v5f_train))

    # ── Validation set ──
    if args.val_data and args.val_output:
        logger.info("Copying validation data as-is from %s", args.val_data)
        os.makedirs(os.path.dirname(args.val_output), exist_ok=True)
        val_samples = []
        with open(args.val_data) as f:
            for line in f:
                val_samples.append(json.loads(line))
        with open(args.val_output, "w") as f:
            for sample in val_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info("Val → %s (%d samples)", args.val_output, len(val_samples))

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  v5f Dataset Generation Complete")
    print(f"{'='*60}")
    print(f"  v5e verbatim (ALL):        {len(v5e_data)}")
    print(f"  Dual-segment copies (NEW): {len(annotated)}")
    print(f"  Cat2 (reasoning):          {len(cat2_results)}")
    print(f"  Cat3 (CoT):                {len(cat3_results)}")
    print(f"  Total train:               {len(v5f_train)}")
    print(f"  Plan↔code consistency:     {len(annotated)}/{len(kernel_batch)}"
          f" ({len(annotated)/max(1,len(kernel_batch))*100:.0f}%)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Generate v5f SFT dataset")
    parser.add_argument("--sft-data", required=True, help="v5e train JSONL")
    parser.add_argument("--val-data", default=None, help="v5e validation JSONL")
    parser.add_argument("--output", required=True, help="v5f train JSONL output")
    parser.add_argument("--val-output", default=None, help="v5f validation JSONL output")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--max-kernel", type=int, default=1879,
                        help="Max kernel samples to annotate (default: all)")
    parser.add_argument("--max-kernel-chars", type=int, default=120000,
                        help="Truncate kernels longer than this (0=no truncation, default 120000 for SEQ_LEN=32768)")
    parser.add_argument("--max-cat2", type=int, default=700)
    parser.add_argument("--max-cat3", type=int, default=350)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
