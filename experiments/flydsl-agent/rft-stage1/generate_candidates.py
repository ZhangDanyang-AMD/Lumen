"""RFT Step 1: Generate kernel candidates from SFT model.

For each RL spec, generate N candidates using the SFT model with diverse
prompting styles. Saves raw generations for sandbox verification.

Usage::

    python generate_candidates.py \
        --model /home/danyzhan/sft-results/Qwen2.5-Coder-SFT-v5f \
        --specs /home/danyzhan/flydsl-agent-dataset/data/rl/train-00000-of-00001.jsonl \
        --output /home/danyzhan/rft-results/candidates.jsonl \
        --n-candidates 16 --max-specs 200 --device cuda:0
"""

import argparse
import json
import logging
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a FlyDSL GPU kernel programming expert. You write compilable, "
    "correct, high-performance GPU kernels using the FlyDSL framework for "
    "AMD Instinct GPUs. Always use @flyc.kernel decorator, fx.* expression "
    "API, and follow FlyDSL coding patterns including SmemAllocator, "
    "pipeline staging, and swizzle for bank conflict avoidance. "
    "Output only the complete Python code, no explanations."
)

STYLES = {
    "precise": (
        "Write a FlyDSL {op} kernel for AMD {hw}.\n"
        "Features: {features}\n"
        "Use @flyc.kernel decorator, fx.* expression API, "
        "SmemAllocator for shared memory, and proper MFMA instructions.\n"
        "Output the complete compilable Python code."
    ),
    "natural": (
        "Implement a high-performance {op_desc} kernel "
        "targeting AMD Instinct {hw_desc} ({hw}) using FlyDSL.\n"
        "The kernel should use proper tiling, shared memory management, "
        "and MFMA matrix instructions for optimal performance.\n"
        "Output only the complete Python source file."
    ),
    "optimization": (
        "Write an optimized FlyDSL kernel for {op} on {hw}.\n"
        "Requirements:\n"
        "- Use @flyc.kernel and @flyc.jit decorators\n"
        "- Use fx.make_layout for tiling and layout algebra\n"
        "- Use SmemAllocator or SharedAllocator for LDS\n"
        "- Optimize for high roofline efficiency\n"
        "Features to incorporate: {features}\n"
        "Output the complete code."
    ),
}

OP_DESCRIPTIONS = {
    "gemm": "matrix multiplication (GEMM)",
    "flash_attn": "FlashAttention forward pass",
    "moe": "Mixture-of-Experts GEMM",
    "softmax": "softmax normalization",
    "rmsnorm": "RMSNorm layer normalization",
    "layernorm": "LayerNorm",
    "rope": "Rotary Position Embedding (RoPE)",
    "topk": "top-K selection",
    "mla": "Multi-head Latent Attention decode",
    "paged_attn": "Paged Attention for KV-cache",
    "allreduce": "custom all-reduce",
    "quant": "quantization kernel",
    "custom": "custom GPU kernel",
}

HW_DESCRIPTIONS = {
    "gfx942": "MI300X", "gfx950": "MI350X", "gfx1250": "MI450", "generic": "AMD GPU",
}


def spec_to_prompt(spec, style="precise"):
    op = spec.get("operator", "unknown")
    hw = spec.get("hardware", "gfx950")
    features = ", ".join(spec.get("features", [])) or "standard"
    template = STYLES[style]
    return template.format(
        op=op, hw=hw, features=features,
        op_desc=OP_DESCRIPTIONS.get(op, op),
        hw_desc=HW_DESCRIPTIONS.get(hw, hw),
    )


def clean_response(text):
    """Strip special tokens, markdown, and generation artifacts from model output."""
    import re
    # Strip Qwen2.5 special tokens that skip_special_tokens may miss
    text = re.sub(r"<\|fim_\w+\|>", "", text)
    text = re.sub(r"<\|file_sep\|>", "", text)
    text = re.sub(r"<\|endoftext\|>", "", text)
    text = re.sub(r"<\|im_start\|>.*?(?:<\|im_end\|>|$)", "", text, flags=re.DOTALL)
    # Strip system prompt leaks
    text = re.sub(r"You are a (?:FlyDSL|GPU).*?(?:no explanations\.|code blocks\.|no markdown\.)", "", text, flags=re.DOTALL)
    # Strip markdown code blocks
    text = re.sub(r"```\w*\n", "", text)
    text = re.sub(r"\n```", "", text)
    text = re.sub(r"^```", "", text)
    # Truncate at any remaining special token
    for tok in ["<|"]:
        idx = text.find(tok)
        if idx > 50:
            text = text[:idx]
    return text.strip()


def generate_batch(model, tokenizer, prompts, device, max_new_tokens=6144,
                   temperature=0.8, top_p=0.95):
    results = []

    # Build stop token IDs for Qwen2.5 FIM tokens
    stop_strings = ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
                    "<|file_sep|>", "<|im_start|>"]
    stop_ids = []
    for s in stop_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            stop_ids.append(ids[0])

    for prompt in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        attention_mask = torch.ones_like(input_ids)

        gen_kwargs = dict(
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        # Add stop tokens if found
        if stop_ids:
            eos_list = [tokenizer.eos_token_id] + stop_ids
            gen_kwargs["eos_token_id"] = eos_list

        with torch.no_grad():
            out = model.generate(input_ids, **gen_kwargs)

        response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        response = clean_response(response)
        results.append(response)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--specs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-candidates", type=int, default=16)
    parser.add_argument("--max-specs", type=int, default=200)
    parser.add_argument("--hardware", type=str, default="gfx950",
                        help="Filter specs by hardware (default: gfx950)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load specs — filter to target hardware only
    with open(args.specs) as f:
        all_specs = [json.loads(l) for l in f if l.strip()]

    usable = [s for s in all_specs if s.get("hardware", "") == args.hardware]
    logger.info("Total specs: %d, %s specs: %d", len(all_specs), args.hardware, len(usable))

    # Sample: prioritize diverse operators, take up to max_specs
    from collections import defaultdict
    by_op = defaultdict(list)
    for s in usable:
        by_op[s["operator"]].append(s)

    selected = []
    per_op = max(1, args.max_specs // len(by_op))
    for op, specs in sorted(by_op.items()):
        sampled = random.sample(specs, min(per_op, len(specs)))
        selected.extend(sampled)
    selected = selected[:args.max_specs]
    random.shuffle(selected)
    logger.info("Selected %d specs across %d operators", len(selected), len(by_op))

    # Load model
    logger.info("Loading model from %s ...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map=args.device,
    )
    model.eval()

    # Generate candidates
    style_keys = list(STYLES.keys())
    total_generated = 0

    with open(args.output, "w") as fout:
        for i, spec in enumerate(selected):
            logger.info("[%d/%d] %s/%s — generating %d candidates ...",
                        i + 1, len(selected), spec["operator"], spec.get("hardware", "?"),
                        args.n_candidates)

            candidates = []
            for j in range(args.n_candidates):
                style = style_keys[j % len(style_keys)]
                prompt = spec_to_prompt(spec, style)
                try:
                    responses = generate_batch(
                        model, tokenizer, [prompt], args.device,
                        max_new_tokens=4096,
                        temperature=0.8, top_p=0.95,
                    )
                    code = responses[0]
                    candidates.append({
                        "code": code,
                        "style": style,
                        "prompt": prompt[:200],
                    })
                except Exception as e:
                    logger.warning("  Generation failed: %s", e)

            record = {
                "spec_id": spec.get("id", f"spec_{i}"),
                "operator": spec["operator"],
                "hardware": spec.get("hardware", "generic"),
                "features": spec.get("features", []),
                "candidates": candidates,
                "n_generated": len(candidates),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            total_generated += len(candidates)

            if (i + 1) % 10 == 0:
                logger.info("  Progress: %d specs, %d total candidates", i + 1, total_generated)

    logger.info("Done. %d specs, %d candidates → %s", len(selected), total_generated, args.output)


if __name__ == "__main__":
    main()
