###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Post-hoc GSM8K evaluation of training checkpoints.

Produces an accuracy-vs-step curve (arXiv:2512.07611 style) by evaluating
each checkpoint saved during GRPO training on the GSM8K test set.

Usage:
  python eval_checkpoints_gsm8k.py \
      --output-dir /workspace/Lumen/outputs/benchmark/R1 \
      --base-model /dev/shm/model/llama-3.1-8b \
      --eval-samples 100 \
      --eval-base
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def _extract_boxed(text: str):
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def _normalize_answer(s: str) -> str:
    s = s.strip().replace(" ", "").replace("\\,", "").replace("{", "").replace("}", "").rstrip(".")
    try:
        return str(float(s))
    except ValueError:
        return s.lower()


def evaluate_model(model, tokenizer, eval_ds, max_new_tokens=256):
    """Run greedy GSM8K eval, returns (accuracy, correct, total)."""
    model.eval()
    correct = 0
    total = len(eval_ds)
    for i, ex in enumerate(eval_ds):
        q = ex["question"]
        gold_raw = ex["answer"]
        gold = gold_raw.split("####")[-1].strip() if "####" in gold_raw else gold_raw.strip()
        prompt = f"Solve the following math problem step by step. Put your final answer in \\boxed{{}}.\n\n{q}\n\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        comp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = _extract_boxed(comp)
        if pred is not None and _normalize_answer(pred) == _normalize_answer(gold):
            correct += 1
        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{total}] running acc={correct/(i+1):.1%}")
    acc = correct / total if total > 0 else 0.0
    return acc, correct, total


def find_checkpoints(output_dir: Path):
    """Find checkpoint-NNN directories and sort by step number."""
    ckpts = []
    for d in output_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[1])
                ckpts.append((step, d))
            except (ValueError, IndexError):
                pass
    ckpts.sort(key=lambda x: x[0])
    return ckpts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--eval-base", action="store_true", default=False,
                        help="Also evaluate the base model (step 0)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    curve_path = output_dir / "gsm8k_accuracy_curve.jsonl"

    print("Loading GSM8K test set...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if len(ds) > args.eval_samples:
        ds = ds.select(range(args.eval_samples))
    print(f"  Using {len(ds)} samples for evaluation")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    results = []

    if args.eval_base:
        print(f"\n=== Evaluating BASE model (step 0) ===")
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
        model.to("cuda:0")
        acc, correct, total = evaluate_model(model, tokenizer, ds, args.max_new_tokens)
        elapsed = time.time() - t0
        print(f"  Base model: {correct}/{total} = {acc:.1%}  ({elapsed:.0f}s)")
        results.append({"step": 0, "gsm8k_accuracy": acc, "correct": correct,
                        "total": total, "checkpoint": "base", "eval_time_s": round(elapsed, 1)})
        del model
        torch.cuda.empty_cache()

    checkpoints = find_checkpoints(output_dir)
    print(f"\nFound {len(checkpoints)} checkpoints: {[s for s, _ in checkpoints]}")

    for step, ckpt_path in checkpoints:
        print(f"\n=== Evaluating checkpoint step {step} ({ckpt_path.name}) ===")
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(str(ckpt_path), torch_dtype=torch.bfloat16)
        model.to("cuda:0")
        acc, correct, total = evaluate_model(model, tokenizer, ds, args.max_new_tokens)
        elapsed = time.time() - t0
        print(f"  Step {step}: {correct}/{total} = {acc:.1%}  ({elapsed:.0f}s)")
        results.append({"step": step, "gsm8k_accuracy": acc, "correct": correct,
                        "total": total, "checkpoint": ckpt_path.name, "eval_time_s": round(elapsed, 1)})
        del model
        torch.cuda.empty_cache()

        with open(curve_path, "w") as fh:
            for entry in results:
                fh.write(json.dumps(entry) + "\n")
        print(f"  Curve saved to {curve_path}")

    print(f"\n{'='*60}")
    print(f"GSM8K Accuracy Curve ({len(results)} points):")
    print(f"{'='*60}")
    for r in results:
        print(f"  Step {r['step']:>4d}: {r['gsm8k_accuracy']:.1%}  ({r['correct']}/{r['total']})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
