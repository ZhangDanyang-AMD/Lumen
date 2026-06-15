#!/usr/bin/env python3
"""Multi-model consensus annotation (Plan §4.4 Layer 2).

Sends annotation prompts to multiple AI models, collects responses,
and produces a consensus annotation via majority voting.

Usage:
  # Step 1: Run annotation with each model (repeat with different --model-name)
  python3 consensus_annotate.py annotate \
      --prompts /path/to/annotation_prompts.jsonl \
      --output /path/to/responses_claude.jsonl \
      --api-base https://api.anthropic.com/v1 \
      --api-key $ANTHROPIC_API_KEY \
      --model claude-sonnet-4-20250514 \
      --model-name claude \
      --provider anthropic

  python3 consensus_annotate.py annotate \
      --prompts /path/to/annotation_prompts.jsonl \
      --output /path/to/responses_gpt4o.jsonl \
      --api-base https://api.openai.com/v1 \
      --api-key $OPENAI_API_KEY \
      --model gpt-4o \
      --model-name gpt4o \
      --provider openai

  # Step 2: Merge all model outputs into consensus
  python3 consensus_annotate.py consensus \
      --responses responses_claude.jsonl responses_gpt4o.jsonl responses_gemini.jsonl responses_deepseek.jsonl \
      --output consensus_annotations.jsonl

  # Step 3: Apply consensus to manifest and rebuild datasets
  python3 consensus_annotate.py apply \
      --manifest /path/to/graded_manifest.json \
      --consensus consensus_annotations.jsonl \
      --output /path/to/annotated_manifest.json
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

TAXONOMY = {
    "operator": [
        "gemm", "flash_attn", "mla", "moe", "softmax", "rmsnorm", "layernorm",
        "rope", "topk", "paged_attn", "allreduce", "quant", "custom",
    ],
    "features": [
        "swizzle_xor16", "pipeline", "async_copy", "double_buffer", "preshuffle",
        "blockscale", "split_k", "epilogue_fusion", "multi_wave", "sage_attention",
        "mxfp4", "tdm", "wmma", "shared_allocator", "layout_algebra",
    ],
    "complexity": ["beginner", "intermediate", "advanced", "expert"],
}


def parse_json_response(text: str) -> dict:
    """Extract JSON from model response, handling markdown code blocks."""
    text = text.strip()
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()
    if not text.startswith('{'):
        brace = text.find('{')
        if brace >= 0:
            text = text[brace:]
    last_brace = text.rfind('}')
    if last_brace >= 0:
        text = text[:last_brace + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def call_openai_compatible(api_base: str, api_key: str, model: str, prompt: str,
                           max_retries: int = 3) -> str:
    """Call OpenAI-compatible API (works for OpenAI, DeepSeek, vLLM, etc.)."""
    import requests
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a GPU kernel code analyst. Output ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  API error after {max_retries} retries: {e}")
                return ""
    return ""


def call_anthropic(api_key: str, model: str, prompt: str, max_retries: int = 3) -> str:
    """Call Anthropic Messages API."""
    import requests
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 2000,
        "temperature": 0.1,
        "system": "You are a GPU kernel code analyst. Output ONLY valid JSON.",
        "messages": [{"role": "user", "content": prompt}],
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  API error after {max_retries} retries: {e}")
                return ""
    return ""


def call_google(api_key: str, model: str, prompt: str, max_retries: int = 3) -> str:
    """Call Google Gemini API."""
    import requests
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2000},
        "systemInstruction": {"parts": [{"text": "You are a GPU kernel code analyst. Output ONLY valid JSON."}]},
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            if resp.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  API error after {max_retries} retries: {e}")
                return ""
    return ""


def run_annotate(args):
    """Run annotation with a single model."""
    prompts = []
    with open(args.prompts) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    already_done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    already_done.add(rec["id"])
        print(f"Resuming: {len(already_done)} already annotated, {len(prompts) - len(already_done)} remaining")

    total = len(prompts)
    done = len(already_done)
    success = 0
    start_time = time.time()

    with open(args.output, "a", encoding="utf-8") as out:
        for i, p in enumerate(prompts):
            if p["id"] in already_done:
                continue

            prompt_text = p["prompt"]

            if args.provider == "anthropic":
                raw = call_anthropic(args.api_key, args.model, prompt_text)
            elif args.provider == "google":
                raw = call_google(args.api_key, args.model, prompt_text)
            else:
                raw = call_openai_compatible(args.api_base, args.api_key, args.model, prompt_text)

            parsed = parse_json_response(raw)
            if parsed:
                success += 1

            record = {
                "id": p["id"],
                "path": p["path"],
                "repo": p.get("repo"),
                "model_name": args.model_name,
                "model": args.model,
                "response": parsed,
                "raw_response": raw[:3000] if not parsed else "",
                "current_labels": p.get("current_labels", {}),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            done += 1

            elapsed = time.time() - start_time
            rate = (done - len(already_done)) / elapsed if elapsed > 0 else 0
            remaining = (total - done) / rate if rate > 0 else 0
            print(f"[{done}/{total}] {p['id'][:60]}... "
                  f"{'OK' if parsed else 'PARSE_FAIL'} "
                  f"({rate:.1f}/s, ~{remaining:.0f}s left)")

            if args.delay > 0:
                time.sleep(args.delay)

    print(f"\nDone: {done}/{total} annotated, {success} parsed successfully")
    print(f"Output: {args.output}")


def run_consensus(args):
    """Merge annotations from multiple models into consensus."""
    all_responses = {}  # id -> {model_name: response}

    for resp_file in args.responses:
        with open(resp_file) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rid = rec["id"]
                mname = rec.get("model_name", Path(resp_file).stem)
                if rid not in all_responses:
                    all_responses[rid] = {"path": rec.get("path"), "repo": rec.get("repo"),
                                          "current_labels": rec.get("current_labels", {}),
                                          "models": {}}
                all_responses[rid]["models"][mname] = rec.get("response", {})

    print(f"Loaded annotations for {len(all_responses)} entries from {len(args.responses)} models")

    results = []
    needs_human = []

    for rid, data in all_responses.items():
        models = data["models"]
        model_names = list(models.keys())
        n_models = len(model_names)

        consensus = {"id": rid, "path": data["path"], "repo": data.get("repo"),
                      "n_models": n_models, "model_names": model_names}

        disagreements = []

        # Enum fields: majority vote
        for field in ["operator", "complexity"]:
            values = [m.get(field) for m in models.values() if m.get(field)]
            if values:
                counter = Counter(values)
                winner, count = counter.most_common(1)[0]
                consensus[field] = winner
                consensus[f"{field}_agreement"] = f"{count}/{len(values)}"
                if count < max(2, len(values) * 0.5 + 1):
                    disagreements.append(field)
                    consensus[f"{field}_votes"] = dict(counter)
            else:
                consensus[field] = data.get("current_labels", {}).get(field, "custom")

        # List field (features): items with ≥2 votes (or ≥50% of models)
        all_features = []
        for m in models.values():
            feats = m.get("features", [])
            if isinstance(feats, list):
                all_features.extend(feats)
        if all_features:
            threshold = max(2, n_models // 2)
            feat_counter = Counter(all_features)
            consensus["features"] = sorted([f for f, c in feat_counter.items() if c >= threshold])
            low_agree = [f for f, c in feat_counter.items() if 1 <= c < threshold]
            if low_agree:
                consensus["features_disputed"] = low_agree
        else:
            consensus["features"] = data.get("current_labels", {}).get("features", [])

        # List field (hardware): union
        all_hw = []
        for m in models.values():
            hw = m.get("hardware", [])
            if isinstance(hw, list):
                all_hw.extend(hw)
        consensus["hardware"] = sorted(set(all_hw)) if all_hw else data.get("current_labels", {}).get("hardware", [])

        # Text fields: pick the longest non-empty response
        for field in ["description", "known_limitations", "sft_instruction"]:
            texts = [(mname, m.get(field, ""))
                     for mname, m in models.items() if m.get(field)]
            if texts:
                best = max(texts, key=lambda x: len(x[1]))
                consensus[field] = best[1]
                consensus[f"{field}_source"] = best[0]
            else:
                consensus[field] = ""

        # typical_shapes: merge unique shapes from all models
        all_shapes = []
        seen = set()
        for m in models.values():
            shapes = m.get("typical_shapes", [])
            if isinstance(shapes, list):
                for s in shapes:
                    key = json.dumps(s, sort_keys=True) if isinstance(s, dict) else str(s)
                    if key not in seen:
                        seen.add(key)
                        all_shapes.append(s)
        consensus["typical_shapes"] = all_shapes[:5]

        consensus["needs_human_review"] = disagreements
        results.append(consensus)
        if disagreements:
            needs_human.append(consensus)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nConsensus results: {len(results)} entries")
    print(f"Needs human review: {len(needs_human)} entries ({len(needs_human)*100/max(len(results),1):.1f}%)")
    if needs_human:
        print("\nDisagreement summary:")
        field_counts = Counter()
        for h in needs_human:
            for d in h["needs_human_review"]:
                field_counts[d] += 1
        for field, count in field_counts.most_common():
            print(f"  {field}: {count} entries")

    print(f"\nSaved to {args.output}")


def run_apply(args):
    """Apply consensus annotations to manifest."""
    with open(args.manifest) as f:
        manifest = json.load(f)

    consensus = {}
    with open(args.consensus) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                consensus[rec["id"]] = rec

    print(f"Applying {len(consensus)} consensus annotations to {len(manifest)} manifest entries")

    applied = 0
    for entry in manifest:
        rid = f"{entry.get('repo', 'unknown')}:{entry['path']}"
        if rid not in consensus:
            continue
        ann = consensus[rid]
        applied += 1

        if ann.get("operator") and ann["operator"] != "custom":
            entry["operator"] = ann["operator"]
        if ann.get("features"):
            existing = set(entry.get("features", []))
            entry["features"] = sorted(existing | set(ann["features"]))
        if ann.get("hardware"):
            existing = set(entry.get("hardware", []))
            entry["hardware"] = sorted(existing | set(ann["hardware"]))
        if ann.get("complexity"):
            entry["ai_complexity"] = ann["complexity"]

        for field in ["description", "known_limitations", "sft_instruction", "typical_shapes"]:
            if ann.get(field):
                entry[f"ai_{field}"] = ann[field]

        entry["ai_annotated"] = True
        entry["ai_models"] = ann.get("model_names", [])
        entry["ai_agreement"] = {
            "operator": ann.get("operator_agreement", ""),
            "complexity": ann.get("complexity_agreement", ""),
        }
        if ann.get("needs_human_review"):
            entry["needs_human_review"] = ann["needs_human_review"]

    print(f"Applied annotations to {applied} entries")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"Saved annotated manifest to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Multi-model consensus annotation")
    sub = parser.add_subparsers(dest="command")

    p_ann = sub.add_parser("annotate", help="Run annotation with one model")
    p_ann.add_argument("--prompts", required=True, help="Input prompts JSONL")
    p_ann.add_argument("--output", required=True, help="Output responses JSONL")
    p_ann.add_argument("--api-base", default="https://api.openai.com/v1")
    p_ann.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    p_ann.add_argument("--model", default="gpt-4o")
    p_ann.add_argument("--model-name", default="model", help="Short name for this model run")
    p_ann.add_argument("--provider", choices=["openai", "anthropic", "google"], default="openai")
    p_ann.add_argument("--delay", type=float, default=0.5, help="Delay between requests (seconds)")

    p_con = sub.add_parser("consensus", help="Merge model responses into consensus")
    p_con.add_argument("--responses", nargs="+", required=True, help="Response JSONL files from each model")
    p_con.add_argument("--output", required=True, help="Output consensus JSONL")

    p_app = sub.add_parser("apply", help="Apply consensus annotations to manifest")
    p_app.add_argument("--manifest", required=True, help="Input manifest JSON")
    p_app.add_argument("--consensus", required=True, help="Consensus JSONL")
    p_app.add_argument("--output", required=True, help="Output annotated manifest JSON")

    args = parser.parse_args()
    if args.command == "annotate":
        run_annotate(args)
    elif args.command == "consensus":
        run_consensus(args)
    elif args.command == "apply":
        run_apply(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
