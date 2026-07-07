"""RFT Step 2+3: Verify candidates in FlyDSL-Gym sandbox and apply diversity filter.

Reads candidates.jsonl, sends each to the sandbox for compilation verification,
filters by three gates (compile, patterns, non-trivial), and preserves all
passing candidates (diversity-preserving, not top-K).

Usage::

    python verify_candidates.py \
        --input /home/danyzhan/rft-results/candidates.jsonl \
        --output /home/danyzhan/rft-results/verified.jsonl \
        --metadata /home/danyzhan/rft-results/verify_stats.json
"""

import argparse
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from collections import Counter, defaultdict

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SANDBOX_IMAGE = "flydsl-gym:latest"
DOCKER_OPTS = [
    "--rm", "--device", "/dev/dri", "--device", "/dev/kfd",
    "--group-add", "video", "--group-add", "render",
    "--ipc=host", "--network=none",
    "--security-opt=seccomp=unconfined",
    "--memory=32g", "--pids-limit=256",
]
VERIFY_TIMEOUT = 120  # seconds


def verify_in_sandbox(code, spec=None, skip_runtime=False):
    """Run verify.py in Docker sandbox. Returns dict with results."""
    code = clean_code(code)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        spec_json = json.dumps(spec or {})
        cmd = [
            "docker", "run", *DOCKER_OPTS,
            "-v", f"{tmp_path}:/tmp/kernel.py:ro",
            SANDBOX_IMAGE,
            "python3", "/workspace/verify.py", "/tmp/kernel.py",
            "--spec", spec_json,
        ]
        if skip_runtime:
            cmd.append("--skip-runtime")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=VERIFY_TIMEOUT,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout.strip())
        else:
            return {
                "compiles": False,
                "error": result.stderr[-500:] if result.stderr else "Unknown error",
            }
    except subprocess.TimeoutExpired:
        return {"compiles": False, "error": "Timeout (120s)"}
    except Exception as e:
        return {"compiles": False, "error": str(e)}
    finally:
        os.unlink(tmp_path)


def clean_code(code):
    """Strip markdown code blocks, special tokens, and generation artifacts."""
    # Strip all special tokens (Qwen2.5 FIM, file sep, endoftext, etc.)
    code = re.sub(r"<\|fim_\w+\|>", "", code)
    code = re.sub(r"<\|file_sep\|>", "", code)
    code = re.sub(r"<\|endoftext\|>", "", code)
    code = re.sub(r"<\|im_start\|>.*?(?:<\|im_end\|>|$)", "", code, flags=re.DOTALL)
    # Strip system prompt leaks
    code = re.sub(r"You are a FlyDSL.*?(?:no explanations\.|code blocks\.)", "", code, flags=re.DOTALL)
    # Strip markdown code blocks (at start/end and in middle)
    code = re.sub(r"```\w*\n", "", code)
    code = re.sub(r"\n```", "", code)
    code = re.sub(r"^```", "", code)
    # Truncate at any remaining special token
    for tok in ["<|", "```"]:
        idx = code.find(tok)
        if idx > 50:  # only truncate if there's substantial code before it
            code = code[:idx]
    # Strip leading/trailing whitespace
    code = code.strip()
    return code


def static_verify(code):
    """Fast static checks without Docker (pre-filter)."""
    code = clean_code(code)

    # Gate 0: Valid Python syntax
    try:
        compile(code, "<kernel>", "exec")
    except SyntaxError as e:
        return {"pass": False, "gate": "syntax", "error": str(e)}

    # Gate 1: Has FlyDSL patterns
    has_flyc = bool(re.search(r"@flyc\.(kernel|jit)", code))
    has_import = bool(re.search(r"import flydsl|from flydsl|import flyc", code))
    has_fx = bool(re.search(r"fx\.\w+", code))

    if not (has_flyc or has_import):
        return {"pass": False, "gate": "no_flydsl", "error": "No FlyDSL imports or decorators"}

    # Gate 2: Non-trivial
    lines = [l for l in code.strip().split("\n") if l.strip() and not l.strip().startswith("#")]
    if len(lines) < 15:
        return {"pass": False, "gate": "trivial", "error": f"Only {len(lines)} non-comment lines"}

    return {
        "pass": True,
        "has_flyc": has_flyc, "has_import": has_import, "has_fx": has_fx,
        "line_count": len(lines),
    }


def edit_distance_ratio(a, b):
    """Rough edit distance ratio between two code strings."""
    lines_a = set(a.strip().split("\n"))
    lines_b = set(b.strip().split("\n"))
    if not lines_a and not lines_b:
        return 0.0
    common = lines_a & lines_b
    total = lines_a | lines_b
    return len(common) / len(total) if total else 1.0


def diversity_filter(candidates, threshold=0.9):
    """Remove candidates that are too similar (edit distance ratio > threshold)."""
    if len(candidates) <= 1:
        return candidates

    kept = [candidates[0]]
    for c in candidates[1:]:
        too_similar = False
        for k in kept:
            if edit_distance_ratio(c["code"], k["code"]) > threshold:
                too_similar = True
                break
        if not too_similar:
            kept.append(c)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--use-sandbox", action="store_true", default=False,
                        help="Use Docker sandbox for real compilation (slow)")
    args = parser.parse_args()

    with open(args.input) as f:
        records = [json.loads(l) for l in f if l.strip()]
    logger.info("Loaded %d specs with candidates", len(records))

    stats = {
        "total_specs": len(records),
        "total_candidates": 0,
        "passed_static": 0,
        "passed_sandbox": 0,
        "passed_runtime": 0,
        "passed_correctness": 0,
        "correctness_skipped": 0,
        "passed_diversity": 0,
        "by_operator": {},
    }

    results = []
    for i, rec in enumerate(records):
        spec_id = rec.get("spec_id", f"spec_{i}")
        op = rec["operator"]
        candidates = rec.get("candidates", [])
        stats["total_candidates"] += len(candidates)

        verified = []
        for j, cand in enumerate(candidates):
            raw_code = cand.get("code", "")
            code = clean_code(raw_code)

            # Static pre-filter (fast)
            static = static_verify(raw_code)
            if not static["pass"]:
                continue
            stats["passed_static"] += 1

            if args.use_sandbox:
                sandbox_result = verify_in_sandbox(raw_code, {
                    "operator": op,
                    "hardware": rec.get("hardware", "generic"),
                    **rec.get("params", {}),
                })
                if not sandbox_result.get("compiles", False):
                    continue
                stats["passed_sandbox"] += 1

                if sandbox_result.get("runs") is True:
                    stats["passed_runtime"] += 1
                if sandbox_result.get("correct") is True:
                    stats["passed_correctness"] += 1
                elif sandbox_result.get("correct") is None and sandbox_result.get("runs") is not False:
                    stats["correctness_skipped"] += 1

                cand["sandbox_result"] = sandbox_result
            else:
                stats["passed_sandbox"] += 1

            verified.append({
                "code": code,
                "style": cand.get("style", "unknown"),
                "static_analysis": static,
                "sandbox_result": cand.get("sandbox_result", {}),
            })

        # Diversity filter
        diverse = diversity_filter(verified, threshold=0.9)
        stats["passed_diversity"] += len(diverse)

        if diverse:
            results.append({
                "spec_id": spec_id,
                "operator": op,
                "hardware": rec.get("hardware", "generic"),
                "features": rec.get("features", []),
                "verified_candidates": diverse,
                "n_verified": len(diverse),
                "n_original": len(candidates),
            })

        if op not in stats["by_operator"]:
            stats["by_operator"][op] = {
                "specs": 0, "candidates": 0, "verified": 0,
                "runtime": 0, "correct": 0,
            }
        stats["by_operator"][op]["specs"] += 1
        stats["by_operator"][op]["candidates"] += len(candidates)
        stats["by_operator"][op]["verified"] += len(diverse)
        for v in verified:
            sr = v.get("sandbox_result", {}) if isinstance(v, dict) else {}
            if sr.get("runs") is True:
                stats["by_operator"][op]["runtime"] += 1
            if sr.get("correct") is True:
                stats["by_operator"][op]["correct"] += 1

        if (i + 1) % 20 == 0:
            logger.info("  [%d/%d] %d verified so far", i + 1, len(records), stats["passed_diversity"])

    # Write verified results
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.metadata:
        with open(args.metadata, "w") as f:
            json.dump(stats, f, indent=2)

    logger.info("=== Verification Summary ===")
    logger.info("  Specs: %d → %d with verified candidates", len(records), len(results))
    logger.info("  Candidates: %d → static %d → sandbox %d → runtime %d → correct %d → diverse %d",
                stats["total_candidates"], stats["passed_static"],
                stats["passed_sandbox"], stats["passed_runtime"],
                stats["passed_correctness"], stats["passed_diversity"])
    logger.info("  Pass rate: %.1f%%", stats["passed_diversity"] / max(stats["total_candidates"], 1) * 100)
    logger.info("  Output: %s", args.output)


if __name__ == "__main__":
    main()
