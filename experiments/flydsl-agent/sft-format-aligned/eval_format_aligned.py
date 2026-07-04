#!/usr/bin/env python3
"""Evaluate Format-Aligned SFT model (used by eval_v5f.sh).

Three-part evaluation:
  Part A: API Score regression (must not drop below v5e baseline of 74%)
  Part B: Format compliance (<plan>+<code> dual-segment presence)
  Part C: Sandbox compilation (extract <code> segment, verify Python syntax)

Usage (v5f):
    python eval_format_aligned.py \
        --model /home/danyzhan/sft-results/Qwen2.5-Coder-SFT-v5f \
        --base-model /home/danyzhan/sft-results/Qwen2.5-Coder-SFT-v5e \
        --sandbox \
        --output /home/danyzhan/sft-results/benchmark_v5f.json
"""

import argparse
import json
import logging
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Format-aligned system prompt (matches training data)
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

# v5e system prompt (for baseline comparison)
V5E_SYSTEM_PROMPT = (
    "You are a FlyDSL GPU kernel programming expert. You write compilable, "
    "correct, high-performance GPU kernels using the FlyDSL framework for "
    "AMD Instinct GPUs. Always use @flyc.kernel decorator, fx.* expression "
    "API, and follow FlyDSL coding patterns."
)

# 25 test prompts (same as eval_sft.py)
TEST_PROMPTS = [
    {"level": 1, "id": "L1_vec_add", "prompt": "Write a FlyDSL kernel that performs element-wise vector addition C = A + B for two float32 vectors of length N. Use @flyc.kernel decorator and fx.gpu thread indexing."},
    {"level": 1, "id": "L1_relu", "prompt": "Write a FlyDSL kernel that applies ReLU activation (max(0, x)) element-wise to a BF16 tensor. Use @flyc.kernel and fx.gpu.thread_idx for parallelization."},
    {"level": 1, "id": "L1_scale", "prompt": "Write a FlyDSL kernel that scales a vector by a scalar constant: Y = alpha * X. Use @flyc.kernel decorator with proper thread indexing."},
    {"level": 1, "id": "L1_copy", "prompt": "Write a FlyDSL kernel that copies data from one buffer to another using buffer_ops.buffer_load and buffer_ops.buffer_store."},
    {"level": 1, "id": "L1_reduce", "prompt": "Write a FlyDSL kernel that computes the sum reduction of a float32 vector using shared memory and fx.syncthreads."},
    {"level": 2, "id": "L2_softmax", "prompt": "Write a FlyDSL kernel implementing row-wise softmax for a 2D BF16 tensor. Handle numerical stability with max subtraction. Use @flyc.kernel and fx.* APIs."},
    {"level": 2, "id": "L2_rmsnorm", "prompt": "Write a FlyDSL kernel implementing RMSNorm normalization for a hidden state tensor. Compute RMS = sqrt(mean(x^2) + eps), then normalize and scale by weight parameter."},
    {"level": 2, "id": "L2_layernorm", "prompt": "Write a FlyDSL kernel implementing LayerNorm with learnable gamma and beta parameters. Use shared memory for mean/variance computation."},
    {"level": 2, "id": "L2_silu", "prompt": "Write a FlyDSL kernel implementing SiLU (Swish) activation: y = x * sigmoid(x) for BF16 tensors. Use @flyc.kernel decorator."},
    {"level": 2, "id": "L2_rope", "prompt": "Write a FlyDSL kernel implementing Rotary Position Embedding (RoPE) for query/key tensors with cos/sin cached frequencies."},
    {"level": 3, "id": "L3_gemm_naive", "prompt": "Write a FlyDSL GEMM kernel computing C = A @ B for BF16 matrices. Use tiled computation with fx.make_layout, SmemAllocator for shared memory, and MFMA instructions via rocdl.mfma_f32_32x32x16_bf16."},
    {"level": 3, "id": "L3_topk", "prompt": "Write a FlyDSL kernel that finds the top-K elements and their indices from each row of a 2D tensor. Use shared memory for partial sorting."},
    {"level": 3, "id": "L3_fused_bias_relu", "prompt": "Write a FlyDSL kernel that fuses bias addition and ReLU activation: Y = max(0, X + bias). Optimize by avoiding extra memory traffic with kernel fusion."},
    {"level": 3, "id": "L3_gemv", "prompt": "Write a FlyDSL kernel for matrix-vector multiplication Y = A @ x where A is (M,K) BF16 and x is (K,) BF16. Use fx.make_layout for tiling."},
    {"level": 3, "id": "L3_concat", "prompt": "Write a FlyDSL kernel that concatenates two tensors along a given dimension with proper layout handling using fx.make_layout and fx.make_shape."},
    {"level": 4, "id": "L4_fp8_gemm", "prompt": "Write a FlyDSL FP8 GEMM kernel for gfx950 with 2-stage pipeline, swizzle_xor16 for LDS bank conflict avoidance, and preshuffle data layout. Use @flyc.kernel with SmemAllocator and lds_stage=2."},
    {"level": 4, "id": "L4_flash_attn", "prompt": "Write a FlyDSL FlashAttention forward kernel for BF16 on gfx950. Implement online softmax with running max/sum, tiled Q@K^T and score@V computation using MFMA instructions and shared memory."},
    {"level": 4, "id": "L4_paged_attn", "prompt": "Write a FlyDSL PagedAttention decode kernel for KV-cache serving with block_table lookup, supporting variable-length sequences and page_size=16."},
    {"level": 4, "id": "L4_gemm_splitk", "prompt": "Write a FlyDSL split-K GEMM kernel that partitions the K dimension across workgroups and uses an atomic reduction to accumulate partial results. Target gfx942 with BF16."},
    {"level": 4, "id": "L4_fused_norm_quant", "prompt": "Write a FlyDSL kernel that fuses RMSNorm + FP8 quantization in a single pass. Compute norm, then quantize the normalized output to fp8_e4m3fnuz with per-tensor scaling."},
    {"level": 5, "id": "L5_preshuffle_gemm", "prompt": "Write a FlyDSL preshuffle GEMM kernel for gfx950 FP8 with tile 128x128x256, double-buffered LDS pipeline (lds_stage=2), swizzle_xor16, 4 waves per workgroup, and epilogue fusion for bias+activation. Use SmemAllocator, fx.zipped_divide, fx.composition for layout algebra."},
    {"level": 5, "id": "L5_moe_2stage", "prompt": "Write a FlyDSL 2-stage MoE GEMM kernel supporting top-K expert routing with token permutation. Stage 1 computes gate_proj+up_proj, stage 2 computes down_proj. Use pipeline for LDS double buffering and expert-parallel tiling."},
    {"level": 5, "id": "L5_mla_decode", "prompt": "Write a FlyDSL MLA (Multi-head Latent Attention) decode kernel for inference. Support compressed KV with latent_dim=576, nope/rope split, and head_dim=128. Use MFMA instructions for BF16 matmul."},
    {"level": 5, "id": "L5_blockscale_gemm", "prompt": "Write a FlyDSL blockscale preshuffle GEMM kernel for gfx950 with per-block FP8 scaling (block_size=128), preshuffle data layout, 3-stage pipeline, and swizzle_xor16. Include proper scale factor broadcasting in the epilogue."},
    {"level": 5, "id": "L5_allreduce", "prompt": "Write a FlyDSL custom all-reduce kernel for multi-GPU ring reduction using DPP (Data Parallel Primitives) instructions. Support BF16 with warp-level shuffles and cross-workgroup synchronization."},
]

# Additional prompts with <code> tag wrapping (user explicitly requested)
CODE_TAG_PROMPTS = [
    {"id": "CT_vec_mul", "prompt": "<code>Write a FlyDSL kernel that multiplies two vectors element-wise C = A * B</code>"},
    {"id": "CT_softmax", "prompt": "<code>Write a FlyDSL kernel for row-wise softmax on BF16 tensor</code>"},
    {"id": "CT_gemm", "prompt": "<code>Write a simple FlyDSL GEMM kernel C = A @ B for BF16 matrices with tiled computation</code>"},
    {"id": "CT_rmsnorm", "prompt": "<code>Write a FlyDSL RMSNorm kernel</code>"},
    {"id": "CT_silu", "prompt": "<code>Write a FlyDSL SiLU activation kernel for BF16</code>"},
]

FLYDSL_PATTERNS = {
    "flyc_kernel": (r"@flyc\.kernel|@flyc\.jit", "Has @flyc.kernel/@flyc.jit"),
    "fx_api": (r"fx\.\w+", "Uses fx.* expression API"),
    "fx_gpu": (r"fx\.gpu\.", "Uses fx.gpu thread/block indexing"),
    "fx_layout": (r"fx\.make_layout|fx\.make_shape|fx\.make_stride", "Uses fx layout algebra"),
    "smem_alloc": (r"SmemAllocator|smem_alloc", "Uses SmemAllocator"),
    "buffer_ops": (r"buffer_ops\.|buffer_load|buffer_store", "Uses buffer_ops"),
    "mfma": (r"rocdl\.mfma|mfma_", "Uses MFMA instructions"),
    "syncthreads": (r"fx\.syncthreads|syncthreads", "Uses thread sync"),
    "swizzle": (r"swizzle_xor|swizzle", "Uses swizzle"),
    "pipeline": (r"lds_stage|pipeline|num_stages", "Uses pipeline"),
    "zipped_divide": (r"fx\.zipped_divide", "Uses fx.zipped_divide"),
    "composition": (r"fx\.composition", "Uses fx.composition"),
    "import_flyc": (r"import flydsl|from flydsl|import flyc|from flyc", "Imports FlyDSL"),
    "valid_python": (None, "Valid Python syntax"),
}

LEVEL_EXPECTED = {
    1: ["flyc_kernel", "fx_api", "import_flyc"],
    2: ["flyc_kernel", "fx_api", "import_flyc"],
    3: ["flyc_kernel", "fx_api", "fx_layout", "smem_alloc", "import_flyc"],
    4: ["flyc_kernel", "fx_api", "fx_layout", "smem_alloc", "mfma", "pipeline", "import_flyc"],
    5: ["flyc_kernel", "fx_api", "fx_layout", "smem_alloc", "mfma", "pipeline", "swizzle", "import_flyc"],
}


def extract_code_segment(response):
    """Extract <code>...</code> segment from response. Falls back to full response."""
    match = re.search(r"<code>\s*(.*?)\s*</code>", response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Remove markdown code blocks if present inside <code>
        code = re.sub(r"^```(?:python)?\s*\n?", "", code)
        code = re.sub(r"\n?```\s*$", "", code)
        return code
    # Fallback: try markdown code blocks
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if blocks:
        return "\n\n".join(blocks)
    return response


def extract_plan_segment(response):
    """Extract <plan>...</plan> segment from response."""
    match = re.search(r"<plan>\s*(.*?)\s*</plan>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def check_format_compliance(response):
    """Check if response has both <plan> and <code> tags."""
    has_plan = bool(re.search(r"<plan>.*?</plan>", response, re.DOTALL))
    has_code = bool(re.search(r"<code>.*?</code>", response, re.DOTALL))
    plan_before_code = True
    if has_plan and has_code:
        plan_pos = response.find("<plan>")
        code_pos = response.find("<code>")
        plan_before_code = plan_pos < code_pos
    return {
        "has_plan": has_plan,
        "has_code": has_code,
        "plan_before_code": plan_before_code,
        "compliant": has_plan and has_code and plan_before_code,
    }


def analyze_code(code):
    results = {}
    for name, (pattern, _) in FLYDSL_PATTERNS.items():
        if name == "valid_python":
            try:
                compile(code, "<gen>", "exec")
                results[name] = True
            except SyntaxError:
                results[name] = False
        else:
            results[name] = bool(re.search(pattern, code))
    return results


def score_response(analysis, level):
    expected = LEVEL_EXPECTED.get(level, [])
    if not expected:
        return 0.0
    return sum(1 for p in expected if analysis.get(p)) / len(expected)


def generate_response(model, tokenizer, system_prompt, user_prompt, device="cuda",
                      max_new_tokens=2048):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)


def sandbox_check(code):
    """Check if extracted code is valid Python with FlyDSL patterns."""
    issues = []
    try:
        compile(code, "<sandbox>", "exec")
        syntax_ok = True
    except SyntaxError as e:
        syntax_ok = False
        issues.append(f"SyntaxError: {e}")

    has_import = bool(re.search(r"import flydsl|from flydsl|import flyc", code))
    has_kernel = bool(re.search(r"@flyc\.(kernel|jit)", code))

    return {
        "syntax_valid": syntax_ok,
        "has_import": has_import,
        "has_kernel": has_kernel,
        "issues": issues,
        "pass": syntax_ok and has_import,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Format-aligned model path")
    parser.add_argument("--base-model", default=None, help="v5e baseline model for comparison")
    parser.add_argument("--output", default=None, help="Output JSON")
    parser.add_argument("--sandbox", action="store_true", help="Run sandbox compilation tests")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ── Part A: API Score (format-aligned model) ──
    logger.info("=" * 60)
    logger.info("Part A: API Score — Format-Aligned Model")
    logger.info("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map=args.device,
    )

    fa_results = []
    format_checks = []
    for test in TEST_PROMPTS:
        logger.info("  [L%d] %s ...", test["level"], test["id"])
        response = generate_response(model, tokenizer, FORMAT_SYSTEM_PROMPT,
                                     test["prompt"], args.device)

        # Format compliance
        fmt = check_format_compliance(response)
        format_checks.append({"id": test["id"], **fmt})

        # Extract <code> for API analysis
        code = extract_code_segment(response)
        analysis = analyze_code(code)
        score = score_response(analysis, test["level"])

        fa_results.append({
            "id": test["id"],
            "level": test["level"],
            "api_score": score,
            "analysis": analysis,
            "format": fmt,
            "response_len": len(response),
            "code_len": len(code),
            "response_preview": response[:400],
        })
        logger.info("    score=%.2f fmt=%s | %s",
                     score, "OK" if fmt["compliant"] else "FAIL",
                     {k: v for k, v in analysis.items() if v})

    del model
    torch.cuda.empty_cache()

    # ── Part A2: API Score (v5e baseline, optional) ──
    v5e_results = []
    if args.base_model:
        logger.info("=" * 60)
        logger.info("Part A2: API Score — v5e Baseline")
        logger.info("=" * 60)

        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map=args.device,
        )

        for test in TEST_PROMPTS:
            logger.info("  [L%d] %s ...", test["level"], test["id"])
            response = generate_response(base_model, tokenizer, V5E_SYSTEM_PROMPT,
                                         test["prompt"], args.device)
            analysis = analyze_code(response)
            score = score_response(analysis, test["level"])
            v5e_results.append({
                "id": test["id"],
                "level": test["level"],
                "api_score": score,
                "analysis": analysis,
            })
            logger.info("    score=%.2f", score)

        del base_model
        torch.cuda.empty_cache()

    # ── Part B: Format Compliance Summary ──
    logger.info("=" * 60)
    logger.info("Part B: Format Compliance")
    logger.info("=" * 60)

    compliant_count = sum(1 for f in format_checks if f["compliant"])
    compliance_rate = compliant_count / len(format_checks) * 100
    logger.info("  Compliance rate: %d/%d = %.0f%%",
                compliant_count, len(format_checks), compliance_rate)

    # ── Part C: Sandbox Tests (with <code> tag prompts) ──
    sandbox_results = []
    if args.sandbox:
        logger.info("=" * 60)
        logger.info("Part C: Sandbox Compilation Tests")
        logger.info("=" * 60)

        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map=args.device,
        )

        # Test standard prompts (extract <code> from response)
        for test in TEST_PROMPTS[:10]:  # L1-L2 subset
            logger.info("  [sandbox] %s ...", test["id"])
            response = generate_response(model, tokenizer, FORMAT_SYSTEM_PROMPT,
                                         test["prompt"], args.device)
            code = extract_code_segment(response)
            check = sandbox_check(code)
            sandbox_results.append({
                "id": test["id"],
                "prompt_type": "standard",
                **check,
                "code_preview": code[:300],
            })
            logger.info("    %s | syntax=%s import=%s",
                         "PASS" if check["pass"] else "FAIL",
                         check["syntax_valid"], check["has_import"])

        # Test <code> tag prompts
        for test in CODE_TAG_PROMPTS:
            logger.info("  [sandbox/code-tag] %s ...", test["id"])
            response = generate_response(model, tokenizer, FORMAT_SYSTEM_PROMPT,
                                         test["prompt"], args.device)
            code = extract_code_segment(response)
            check = sandbox_check(code)
            sandbox_results.append({
                "id": test["id"],
                "prompt_type": "code_tag",
                **check,
                "code_preview": code[:300],
            })
            logger.info("    %s | syntax=%s import=%s",
                         "PASS" if check["pass"] else "FAIL",
                         check["syntax_valid"], check["has_import"])

        del model
        torch.cuda.empty_cache()

    # ── Report ──
    print("\n" + "=" * 70)
    print("  FORMAT-ALIGNED SFT EVALUATION REPORT")
    print("=" * 70)

    # API Scores
    level_scores_fa = {}
    for r in fa_results:
        level_scores_fa.setdefault(r["level"], []).append(r["api_score"])
    overall_fa = sum(r["api_score"] for r in fa_results) / len(fa_results)

    print("\n  Part A: API Score (Format-Aligned)")
    for lv in sorted(level_scores_fa):
        avg = sum(level_scores_fa[lv]) / len(level_scores_fa[lv])
        print(f"    Level {lv}: {avg*100:.0f}%")
    print(f"    Overall: {overall_fa*100:.0f}%")

    if v5e_results:
        overall_v5e = sum(r["api_score"] for r in v5e_results) / len(v5e_results)
        delta = (overall_fa - overall_v5e) * 100
        print(f"\n  v5e Baseline: {overall_v5e*100:.0f}%")
        print(f"  Delta: {delta:+.0f}%")
        verdict = "PASS" if overall_fa >= overall_v5e * 0.95 else "FAIL"
        print(f"  Regression test: {verdict} (threshold: ≥ {overall_v5e*95:.0f}%)")

    # Format compliance
    print(f"\n  Part B: Format Compliance")
    print(f"    Rate: {compliance_rate:.0f}% (target ≥90%)")
    verdict = "PASS" if compliance_rate >= 90 else "FAIL"
    print(f"    Verdict: {verdict}")

    # Sandbox
    if sandbox_results:
        standard = [r for r in sandbox_results if r["prompt_type"] == "standard"]
        code_tag = [r for r in sandbox_results if r["prompt_type"] == "code_tag"]

        std_pass = sum(1 for r in standard if r["pass"]) / len(standard) * 100 if standard else 0
        ct_pass = sum(1 for r in code_tag if r["pass"]) / len(code_tag) * 100 if code_tag else 0
        all_pass = sum(1 for r in sandbox_results if r["pass"]) / len(sandbox_results) * 100

        print(f"\n  Part C: Sandbox Compilation")
        print(f"    Standard prompts: {std_pass:.0f}%")
        print(f"    <code> tag prompts: {ct_pass:.0f}%")
        print(f"    Overall: {all_pass:.0f}% (target ≥80%)")
        for r in sandbox_results:
            status = "PASS" if r["pass"] else "FAIL"
            print(f"      {status} {r['id']} [{r['prompt_type']}]")

    print("=" * 70)

    # Save output
    output = {
        "format_aligned_results": fa_results,
        "v5e_baseline_results": v5e_results,
        "format_checks": format_checks,
        "sandbox_results": sandbox_results,
        "summary": {
            "api_score_fa": overall_fa,
            "api_score_v5e": sum(r["api_score"] for r in v5e_results) / len(v5e_results) if v5e_results else None,
            "format_compliance_rate": compliance_rate,
            "sandbox_pass_rate": sum(1 for r in sandbox_results if r["pass"]) / len(sandbox_results) * 100 if sandbox_results else None,
        },
    }

    out_path = args.output or os.path.join(args.model, "benchmark_format_aligned.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
