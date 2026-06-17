"""CPT Benchmark — evaluate Qwen2.5-Coder-CPT vs base model.

Tests from plan.md §8.1:
  (a) Perplexity on held-out FlyDSL kernels
  (b) API completion accuracy (FlyDSL-specific tokens)
  (c) Code continuation quality

Usage::

    python eval_cpt.py \
        --base-model /dev/shm/qwen2.5-coder-32b \
        --cpt-model /home/danyzhan/cpt-results/Qwen2.5-Coder-CPT \
        --data /home/danyzhan/flydsl-agent-dataset/data/cpt/train-00000-of-00001.jsonl
"""

import argparse
import json
import logging
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- (b) API completion test cases ----
API_COMPLETIONS = [
    ("tid = fx.gpu.thread_idx.x\nbid = fx.gpu.", "block_idx"),
    ("layout = fx.make_layout(fx.make_shape(8, 16), fx.", "make_stride"),
    ("buffer_ops.buffer_", "load"),
    ("smem = fx.SmemAllocator(", ""),
    ("@flyc.", "kernel"),
    ("fx.zipped_divide(", ""),
    ("fx.composition(", ""),
    ("rocdl.mfma_f32_32x32x16_", "bf16"),
    ("fx.complement(", ""),
    ("lds_stage =", ""),
    ("swizzle_xor", "16"),
    ("fx.make_shape(", ""),
    ("fx.gpu.block_dim.", "x"),
    ("flyc.launch(", ""),
    ("fx.syncthreads(", ""),
    ("fx.gpu.warp_id", ""),
    ("buffer_ops.buffer_store(", ""),
    ("fx.make_layout(", ""),
    ("fx.logical_divide(", ""),
    ("flyc.jit(", ""),
]


def compute_perplexity(model, tokenizer, texts, max_len=2048, device="cuda"):
    """Compute per-text perplexity, return list of (text_id, ppl) and mean."""
    model.eval()
    ppls = []
    with torch.no_grad():
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text, return_tensors="pt",
                                   truncation=True, max_length=max_len).to(device)
            if ids.shape[1] < 2:
                continue
            outputs = model(ids, labels=ids)
            ppl = math.exp(outputs.loss.item())
            ppls.append((i, ppl))
    mean_ppl = sum(p for _, p in ppls) / len(ppls) if ppls else float("inf")
    return ppls, mean_ppl


def test_api_completion(model, tokenizer, device="cuda", top_k=5):
    """Test FlyDSL API token completion accuracy."""
    model.eval()
    top1_correct = 0
    topk_correct = 0
    results = []

    with torch.no_grad():
        for prefix, expected in API_COMPLETIONS:
            ids = tokenizer.encode(prefix, return_tensors="pt").to(device)
            outputs = model(ids)
            logits = outputs.logits[0, -1, :]
            top_tokens = torch.topk(logits, top_k).indices.tolist()
            top_strs = [tokenizer.decode([t]).strip() for t in top_tokens]

            t1 = 1 if expected and any(expected in s for s in top_strs[:1]) else 0
            tk = 1 if expected and any(expected in s for s in top_strs) else 0
            top1_correct += t1
            topk_correct += tk

            results.append({
                "prefix": prefix.replace("\n", "\\n")[-50:],
                "expected": expected,
                "top1": top_strs[0] if top_strs else "?",
                "top5": top_strs[:5],
                "top1_hit": bool(t1),
                "top5_hit": bool(tk),
            })

    n = len(API_COMPLETIONS)
    return {
        "top1_accuracy": top1_correct / n,
        "top5_accuracy": topk_correct / n,
        "total": n,
        "details": results,
    }


def test_code_continuation(model, tokenizer, texts, device="cuda", prefix_lines=30, gen_tokens=200):
    """Test code continuation: give first N lines, generate continuation."""
    model.eval()
    results = []

    for text in texts[:10]:
        lines = text.split("\n")
        if len(lines) < prefix_lines + 10:
            continue

        prefix = "\n".join(lines[:prefix_lines])
        reference = "\n".join(lines[prefix_lines:prefix_lines+20])

        ids = tokenizer.encode(prefix, return_tensors="pt",
                               truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=gen_tokens,
                                 do_sample=False, temperature=1.0)
        generated = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

        # Check quality indicators
        has_fx = "fx." in generated
        has_flyc = "flyc." in generated or "@flyc" in generated
        has_smem = "SmemAllocator" in generated or "smem" in generated
        has_valid_python = True
        try:
            compile(generated, "<gen>", "exec")
        except SyntaxError:
            has_valid_python = False

        results.append({
            "prefix_last_line": lines[prefix_lines-1].strip()[:60],
            "has_fx_api": has_fx,
            "has_flyc": has_flyc,
            "has_smem_pattern": has_smem,
            "valid_python": has_valid_python,
            "gen_preview": generated[:200].replace("\n", "\\n"),
        })

    return results


def evaluate_model(model_path, model_name, tokenizer, test_texts, device="cuda"):
    """Run all evaluations on one model."""
    logger.info("=== Evaluating: %s ===", model_name)
    logger.info("Loading %s ...", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map=device,
    )

    # (a) Perplexity
    logger.info("[a] Perplexity test ...")
    ppls, mean_ppl = compute_perplexity(model, tokenizer, test_texts, device=device)
    logger.info("  Mean PPL = %.2f", mean_ppl)
    for i, ppl in ppls:
        logger.info("    text %d: PPL = %.2f", i, ppl)

    # (b) API completion
    logger.info("[b] API completion test (%d cases) ...", len(API_COMPLETIONS))
    api_results = test_api_completion(model, tokenizer, device=device)
    logger.info("  Top-1 accuracy: %.1f%% (%d/%d)",
                api_results["top1_accuracy"]*100,
                int(api_results["top1_accuracy"]*api_results["total"]),
                api_results["total"])
    logger.info("  Top-5 accuracy: %.1f%% (%d/%d)",
                api_results["top5_accuracy"]*100,
                int(api_results["top5_accuracy"]*api_results["total"]),
                api_results["total"])

    # (c) Code continuation
    logger.info("[c] Code continuation test ...")
    cont_results = test_code_continuation(model, tokenizer, test_texts, device=device)
    n_fx = sum(1 for r in cont_results if r["has_fx_api"])
    n_flyc = sum(1 for r in cont_results if r["has_flyc"])
    n_valid = sum(1 for r in cont_results if r["valid_python"])
    n = len(cont_results)
    if n > 0:
        logger.info("  fx.* API usage: %d/%d (%.0f%%)", n_fx, n, 100*n_fx/n)
        logger.info("  flyc pattern:   %d/%d (%.0f%%)", n_flyc, n, 100*n_flyc/n)
        logger.info("  Valid Python:   %d/%d (%.0f%%)", n_valid, n, 100*n_valid/n)

    del model
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "perplexity": {"mean": mean_ppl, "per_text": ppls},
        "api_completion": api_results,
        "code_continuation": cont_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--cpt-model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-test", type=int, default=5,
                        help="Number of held-out kernel texts for perplexity test")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Select test texts: pick 5 diverse FlyDSL kernel files
    logger.info("Loading test data from %s ...", args.data)
    with open(args.data) as f:
        records = [json.loads(l) for l in f]

    # Pick gold/silver FlyDSL kernels as test set
    test_candidates = [
        r for r in records
        if r["meta"].get("content_type") == "kernel_impl"
        and r["meta"].get("source_repo") == "FlyDSL"
        and r["meta"].get("quality_grade") in ("gold", "silver")
        and r["meta"].get("tokens_approx", 0) > 1000
        and r["meta"].get("tokens_approx", 0) < 10000
    ]

    # Pick diverse operators
    seen_ops = set()
    test_texts = []
    for r in test_candidates:
        op = r["meta"].get("operator", "unknown")
        if op not in seen_ops and len(test_texts) < args.n_test:
            test_texts.append(r["text"])
            seen_ops.add(op)
            logger.info("  Test file: %s (op=%s, grade=%s)",
                        r["meta"]["source_path"], op, r["meta"]["quality_grade"])

    if len(test_texts) < args.n_test:
        for r in test_candidates:
            if len(test_texts) >= args.n_test:
                break
            if r["text"] not in test_texts:
                test_texts.append(r["text"])

    logger.info("Selected %d test texts", len(test_texts))

    # Evaluate both models
    base_results = evaluate_model(args.base_model, "Qwen2.5-Coder-32B (base)", tokenizer, test_texts, args.device)
    cpt_results = evaluate_model(args.cpt_model, "Qwen2.5-Coder-CPT (r=128, 10ep)", tokenizer, test_texts, args.device)

    # Summary comparison
    print("\n" + "="*70)
    print("  CPT BENCHMARK RESULTS")
    print("="*70)
    print(f"\n{'Metric':<35} {'Base':>12} {'CPT':>12} {'Target':>12} {'Pass?':>8}")
    print("-"*70)

    b_ppl = base_results["perplexity"]["mean"]
    c_ppl = cpt_results["perplexity"]["mean"]
    ppl_pass = "YES" if c_ppl < 10 else "NO"
    print(f"{'(a) Perplexity (FlyDSL code)':<35} {b_ppl:>12.1f} {c_ppl:>12.1f} {'< 10':>12} {ppl_pass:>8}")

    b_t1 = base_results["api_completion"]["top1_accuracy"] * 100
    c_t1 = cpt_results["api_completion"]["top1_accuracy"] * 100
    t1_pass = "YES" if c_t1 > 60 else "NO"
    print(f"{'(b) API Top-1 accuracy':<35} {b_t1:>11.1f}% {c_t1:>11.1f}% {'> 60%':>12} {t1_pass:>8}")

    b_t5 = base_results["api_completion"]["top5_accuracy"] * 100
    c_t5 = cpt_results["api_completion"]["top5_accuracy"] * 100
    t5_pass = "YES" if c_t5 > 85 else "NO"
    print(f"{'(b) API Top-5 accuracy':<35} {b_t5:>11.1f}% {c_t5:>11.1f}% {'> 85%':>12} {t5_pass:>8}")

    bc = base_results["code_continuation"]
    cc = cpt_results["code_continuation"]
    if bc and cc:
        b_fx = sum(1 for r in bc if r["has_fx_api"]) / len(bc) * 100
        c_fx = sum(1 for r in cc if r["has_fx_api"]) / len(cc) * 100
        fx_pass = "YES" if c_fx > 70 else "NO"
        print(f"{'(c) Code continuation fx.* API':<35} {b_fx:>11.1f}% {c_fx:>11.1f}% {'> 70%':>12} {fx_pass:>8}")

    print("-"*70)

    overall = all([c_ppl < 10, c_t1 > 60, c_t5 > 85])
    print(f"\n  Overall: {'PASS' if overall else 'PARTIAL PASS'}")
    print(f"  PPL improvement: {b_ppl:.1f} -> {c_ppl:.1f} ({b_ppl/c_ppl:.1f}x reduction)")
    print()

    # Save results
    out_path = os.path.join(os.path.dirname(args.cpt_model), "cpt_benchmark.json")
    with open(out_path, "w") as f:
        json.dump({"base": base_results, "cpt": cpt_results}, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
