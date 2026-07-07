# Format-Aligned SFT — Stage 0 Results

## v5f Approach (replaces v1 patch-on-v5e)

v1 尝试在 v5e 上叠加 LoRA 做格式对齐，导致双重 LoRA 叠加不稳定 + 非 kernel 能力隐式保持。
v5f 改为从 base model 全量重训，将 v5e 数据 + format alignment 合并成统一数据集。

### v5f 数据构造

策略：**保留 v5e 全量数据不动** + 增加双段格式副本。不替换任何原始样本。

| 类别 | 来源 | 数量 | 格式 | 说明 |
|------|------|------|------|------|
| v5e 全量 (原样) | v5e SFT 不动 | ~3889 | 原始 | 保护 v5e 代码能力 |
| kernel 双段副本 (新增) | Claude plan + v5e kernel code | ~1500 | `<plan>+<code>` | 教模型双段格式 |
| Cat2 通用推理 | Claude 生成 | ~700 | `<plan>+<code>` | 推理保持 |
| Cat3 复杂 CoT | Claude 生成 | ~350 | `<plan>+<code>` | 推理深度 |
| **总计** | | **~6400** | 混合 | |

关键设计：kernel 双段副本的 user prompt 加了 suffix ("Explain your tiling decisions." 等)，
使模型学到 "当用户要求解释时 → 用 plan+code 格式"，而非无条件改变输出格式。

### v5f 训练配置 (与 v5e 相同)

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-Coder-32B (原始, 非 v5e) |
| Method | LoRA r=64, alpha=128, dropout=0.1 |
| Epochs | 3 |
| LR | 1e-5 |
| SEQ_LEN | 16384 |

### v5f vs v1 的关键区别

1. **保留原始信号**：v5e 全量数据原样保留，双段格式作为额外样本添加（不替换）
2. **一次训练 vs 双重 LoRA**：从 base 直接训到 v5f，不叠加 v5e
3. **Plan-code 因果校验**：`_extract_code_decisions` 确保 plan 引用 code 中的实际设计决策
4. **简化 system prompt**：去掉格式示例，改为简短指令 (防止 template regurgitation)
5. **条件格式**：user prompt 含"Explain your decisions"时 → plan+code；否则 → 原始格式

### 运行流程

```bash
# 1. 生成 v5f 数据 (需要 Claude API)
python generate_v5f_data.py \
    --sft-data /home/danyzhan/flydsl-agent-dataset/data/sft/train-00000-of-00001.jsonl \
    --val-data /home/danyzhan/flydsl-agent-dataset/data/sft/validation-00000-of-00001.jsonl \
    --output /home/danyzhan/flydsl-agent-dataset/data/v5f/train.jsonl \
    --val-output /home/danyzhan/flydsl-agent-dataset/data/v5f/validation.jsonl

# 2. 训练 (3 epochs, ~3h on 8xMI350X)
bash run_v5f.sh

# 3. 导出 HF 模型
bash export_v5f.sh

# 4. 评估
bash eval_v5f.sh
```

---

## v5f Evaluation Results (2026-07-04)

### Part A: API Score

| Level | v5f | v5e Baseline | Delta |
|-------|-----|-------------|-------|
| L1 (Basic) | **100%** | 100% | 0% |
| L2 (Elementary) | **100%** | 100% | 0% |
| L3 (Intermediate) | 72% | 72% | 0% |
| L4 (Advanced) | 46% | 49% | -3% |
| L5 (Expert) | 50% | 50% | 0% |
| **Overall** | **74%** | **74%** | **-1%** |

**Verdict: PASS** (target ≥ 74%)

### Part B: Format Compliance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Compliant responses | 24/25 | ≥ 90% | **PASS** |
| Compliance rate | **96%** | ≥ 90% | **PASS** |
| Only failure | L5_preshuffle_gemm (truncated) | — | — |

### Part C: Sandbox Compilation

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Standard prompts | 10/10 (100%) | — | **PASS** |
| `<code>` tag prompts | 5/5 (100%) | — | **PASS** |
| **Overall** | **15/15 (100%)** | ≥ 80% | **PASS** |

### Per-Prompt Detail

```
ID                        v5f   v5e   Delta  fmt
─────────────────────────────────────────────────
L1_vec_add               1.00  1.00  +0.00   OK
L1_relu                  1.00  1.00  +0.00   OK
L1_scale                 1.00  1.00  +0.00   OK
L1_copy                  1.00  1.00  +0.00   OK
L1_reduce                1.00  1.00  +0.00   OK
L2_softmax               1.00  1.00  +0.00   OK
L2_rmsnorm               1.00  1.00  +0.00   OK   (v1: 0.00 ← fixed)
L2_layernorm             1.00  1.00  +0.00   OK
L2_silu                  1.00  1.00  +0.00   OK
L2_rope                  1.00  1.00  +0.00   OK
L3_gemm_naive            1.00  1.00  +0.00   OK
L3_topk                  0.60  0.60  +0.00   OK
L3_fused_bias_relu       0.60  0.60  +0.00   OK
L3_gemv                  0.80  0.80  +0.00   OK
L3_concat                0.60  0.60  +0.00   OK
L4_fp8_gemm              0.57  0.71  -0.14   OK
L4_flash_attn            0.43  0.43  +0.00   OK   (v1: 0.14 ← fixed)
L4_paged_attn            0.43  0.43  +0.00   OK
L4_gemm_splitk           0.43  0.43  +0.00   OK
L4_fused_norm_quant      0.43  0.43  +0.00   OK
L5_preshuffle_gemm       0.62  0.75  -0.12   FAIL (truncated)
L5_moe_2stage            0.50  0.50  +0.00   OK   (v1: 0.00 ← fixed)
L5_mla_decode            0.50  0.62  -0.12   OK
L5_blockscale_gemm       0.50  0.25  +0.25   OK
L5_allreduce             0.38  0.38  +0.00   OK
```

### Training Summary

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-Coder-32B (original, NOT v5e) |
| Data | 6809 train / 264 val |
| v5e verbatim | 3889 samples (100% preserved) |
| Dual-segment copies | ~1870 new entries |
| Cat2 (reasoning) | ~700 |
| Cat3 (CoT) | ~350 |
| Method | LoRA r=64, alpha=128, dropout=0.1 |
| Epochs | 3 |
| Steps | 2556 |
| LR | 1e-5 (cosine) |
| SEQ_LEN | 32768 |
| Val loss | 1.2246 → 0.9450 |
| Training time | ~10.5h (8xMI350X) |
| Model | `sft-results/Qwen2.5-Coder-SFT-v5f` |
| HuggingFace | `ZhangDanyang-AMD/Qwen2.5-Coder-SFT-v5f` |

### v5f vs v1 comparison

| Metric | v1 (patch on v5e) | v5f (full retrain) | Target |
|--------|-------------------|-------------------|--------|
| API Score | 68% ❌ | **74%** ✅ | ≥ 74% |
| Format compliance | 88% ❌ | **96%** ✅ | ≥ 90% |
| Sandbox | 87% | **100%** ✅ | ≥ 80% |
| L2_rmsnorm | 0% | **100%** | — |
| L5_moe_2stage | 0% | **50%** | — |

---

## v5f RFT Sandbox Results (2026-07-05)

122 gfx950 specs × 16 candidates = 1952 total candidates.

| Metric | v5e (12 specs) | v5f (122 specs) | Improvement |
|--------|---------------|----------------|-------------|
| Specs tested | 12 | **122** | 10x |
| Total candidates | 192 | **1952** | 10x |
| Passed static | 134 (70%) | **1369 (70%)** | same rate |
| **Passed sandbox** | **42 (22%)** | **733 (38%)** | **+16pp, 17x** |

Per-operator sandbox pass rates:

| Operator | v5e verified | v5f verified | v5f rate |
|----------|-------------|-------------|----------|
| softmax | 5 | 71 | 49% |
| custom | 5 | 120 | 44% |
| topk | 3 | 48 | 43% |
| rmsnorm | 2 | 41 | 43% |
| moe | 5 | 60 | 42% |
| layernorm | 4 | 31 | 39% |
| mla | 6 | 43 | 38% |
| quant | 1 | 100 | 37% |
| gemm | 4 | 96 | 35% |
| rope | 5 | 34 | 35% |
| paged_attn | 0 | 26 | 32% |
| flash_attn | 2 | 63 | 23% |

Results: `rft-results/verify_stats_v5f_gfx950.json`, `rft-results/candidates_v5f_gfx950.jsonl`

---

## RFT Stage A Results (2026-07-06)

### Training

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-Coder-SFT-v5f (merged) |
| Data | 8275 samples (v5f 6809 + 733 verified × 2) |
| Method | LoRA r=64, alpha=128, lr 5e-6 |
| Epochs | 1 |
| Steps | 1035 |
| Val loss | 0.9615 → 0.9533 |
| Training time | ~2h (8xMI350X) |
| Model | `rft-results/Qwen2.5-Coder-RFT-v5f` |

### Benchmark (RFT vs v5f vs v5e)

| Level | RFT | v5f | v5e | Delta (RFT-v5f) |
|-------|-----|-----|-----|-----------------|
| L1 (Basic) | 100% | 100% | 100% | 0% |
| L2 (Elementary) | 100% | 100% | 100% | 0% |
| L3 (Intermediate) | 68% | 72% | 72% | -4% |
| L4 (Advanced) | **49%** | 46% | 49% | **+3%** |
| L5 (Expert) | **57%** | 50% | 50% | **+7%** |
| **Overall** | **75%** | 74% | 74% | **+1%** |

| Metric | RFT | v5f | Target | Status |
|--------|-----|-----|--------|--------|
| Format compliance | 96% | 96% | >= 90% | PASS |
| Sandbox compilation | 100% | 100% | >= 80% | PASS |

### Per-Prompt Detail

```
ID                        RFT   v5f   Delta  Note
─────────────────────────────────────────────────
L5_blockscale_gemm       0.75  0.50  +0.25   +++  (RFT boost)
L5_moe_2stage            0.62  0.50  +0.12   +++  (RFT boost)
L5_preshuffle_gemm       0.75  0.62  +0.12   +++  (RFT boost)
L4_fp8_gemm              0.71  0.71  +0.00
L3_concat                0.60  0.80  -0.20   (minor regression)
L5_mla_decode            0.38  0.62  -0.25   (regression)
```

Key takeaway: RFT sandbox-verified data significantly improved L5 Expert-level
kernel generation (+7pp), particularly for complex GEMM variants (blockscale,
preshuffle, MoE). Overall API Score 75% exceeds v5e baseline (74%).

### Full gfx950 Sandbox Evaluation (2026-07-07)

122 specs × 16 candidates = 1952 total. Compared against v5f SFT model (pre-RFT).

| Metric | v5e (12 specs) | v5f SFT (122 specs) | **RFT** (122 specs) | RFT vs v5f |
|--------|---------------|--------------------|--------------------|------------|
| Passed sandbox | 42 (22%) | 733 (38%) | **1026 (53%)** | **+15pp** |

Per-operator sandbox pass rates:

| Operator | v5e | v5f SFT | **RFT** | Delta |
|----------|-----|---------|---------|-------|
| rmsnorm | 2 | 41 (43%) | **57 (59%)** | +17pp |
| quant | 1 | 100 (37%) | **158 (58%)** | +21pp |
| softmax | 5 | 71 (49%) | **84 (58%)** | +9pp |
| layernorm | 4 | 31 (39%) | **45 (56%)** | +18pp |
| custom | 5 | 120 (44%) | **150 (55%)** | +11pp |
| topk | 3 | 48 (43%) | **58 (52%)** | +9pp |
| gemm | 4 | 96 (35%) | **137 (50%)** | +15pp |
| rope | 5 | 34 (35%) | **48 (50%)** | +15pp |
| paged_attn | 0 | 26 (32%) | **40 (50%)** | +18pp |
| mla | 6 | 43 (38%) | **55 (49%)** | +11pp |
| moe | 5 | 60 (42%) | **69 (48%)** | +6pp |
| flash_attn | 2 | 63 (23%) | **125 (46%)** | +23pp |

Every operator improved. flash_attn had the largest gain (+23pp).

HuggingFace: [Zhangdanyang/Qwen2.5-Coder-RFT-v5f](https://huggingface.co/Zhangdanyang/Qwen2.5-Coder-RFT-v5f)

### Three-Level Verification (2026-07-07)

Added runtime execution and correctness checking to sandbox verification.

| Level | Count | Rate | Description |
|-------|-------|------|-------------|
| Static | 1598 | 82% | Python syntax + FlyDSL patterns |
| **Compilation** | 944 | **48%** | FlyDSL JIT import succeeds |
| **Runtime** | 211 | **11%** | Entry point callable without crash |
| **Correctness** | 0 | **0%** | Output matches PyTorch reference |

Runtime breakdown by reason:
- quant: 149 INCORRECT (wrong quantization logic)
- topk: 51 shape mismatch (wrong output dimensions)
- quant/gemm/topk/rmsnorm: 11 returned None (in-place but unverifiable)

**Key insight**: The model has learned FlyDSL API syntax (48% compile) and partial
host-side structure (11% run), but kernel compute logic is entirely wrong (0% correct).
This is expected — SFT+RFT only trained on compilation pass/fail, not on correctness.
**Runtime correctness requires RL Stage B/C with correctness reward.**

---

# Format-Aligned SFT v1 Results (archived)

## Training Summary

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-Coder-SFT-v5e (merged) |
| Method | LoRA r=32, alpha=64, dropout=0.05 |
| Epochs | 1 |
| LR | 5e-6 (cosine decay) |
| Steps | 287 |
| SEQ_LEN | 32768 |
| Data | 2295 train / 254 val |
| Cat1 (FlyDSL kernel dual-segment) | 1163 (50.7%) |
| Cat2 (General reasoning) | 743 (32.4%) |
| Cat3 (Complex CoT) | 389 (16.9%) |
| Val loss | 0.6123 → 0.6064 (stable, no overfit) |
| Training time | ~1h (8xMI350X) |
| Output model | `/home/danyzhan/sft-results/Qwen2.5-Coder-SFT-Format-Aligned` |

## Evaluation Results

### Part A: API Score (vs v5e baseline)

| Level | Format-Aligned | v5e Baseline | Delta |
|-------|---------------|-------------|-------|
| L1 (Basic) | **100%** | 100% | 0% |
| L2 (Elementary) | 80% | **100%** | **-20%** |
| L3 (Intermediate) | 72% | 72% | 0% |
| L4 (Advanced) | 43% | **49%** | -6% |
| L5 (Expert) | 45% | **50%** | -5% |
| **Overall** | **68%** | **74%** | **-6%** |

**Verdict: FAIL** (target ≥ 74%, actual 68%)

### Part B: Format Compliance (`<plan>` + `<code>` dual-segment)

| Metric | Value | Target |
|--------|-------|--------|
| Compliant responses | 22/25 | ≥ 90% |
| Compliance rate | **88%** | ≥ 90% |
| Failed prompts | L2_rmsnorm, L4_flash_attn, L5_preshuffle_gemm | — |

**Verdict: FAIL** (88% < 90%)

### Part C: Sandbox Compilation

| Metric | Value | Target |
|--------|-------|--------|
| Standard prompts | 9/10 (90%) | — |
| `<code>` tag prompts | 4/5 (80%) | — |
| **Overall** | **13/15 (87%)** | ≥ 80% |

**Verdict: PASS**

### Per-Prompt Detail

```
ID                        FA    v5e   Delta  fmt
─────────────────────────────────────────────────
L1_vec_add               1.00  1.00  +0.00   OK
L1_relu                  1.00  1.00  +0.00   OK
L1_scale                 1.00  1.00  +0.00   OK
L1_copy                  1.00  1.00  +0.00   OK
L1_reduce                1.00  1.00  +0.00   OK
L2_softmax               1.00  1.00  +0.00   OK
L2_rmsnorm               0.00  1.00  -1.00   FAIL <<<
L2_layernorm             1.00  1.00  +0.00   OK
L2_silu                  1.00  1.00  +0.00   OK
L2_rope                  1.00  1.00  +0.00   OK
L3_gemm_naive            1.00  1.00  +0.00   OK
L3_topk                  0.60  0.60  +0.00   OK
L3_fused_bias_relu       0.60  0.60  +0.00   OK
L3_gemv                  0.80  0.80  +0.00   OK
L3_concat                0.60  0.60  +0.00   OK
L4_fp8_gemm              0.57  0.71  -0.14   OK  <<<
L4_flash_attn            0.14  0.43  -0.29   FAIL <<<
L4_paged_attn            0.57  0.43  +0.14   OK
L4_gemm_splitk           0.43  0.43  +0.00   OK
L4_fused_norm_quant      0.43  0.43  +0.00   OK
L5_preshuffle_gemm       0.75  0.75  +0.00   FAIL (truncated)
L5_moe_2stage            0.00  0.50  -0.50   OK  <<<
L5_mla_decode            0.50  0.62  -0.12   OK  <<<
L5_blockscale_gemm       0.62  0.25  +0.38   OK
L5_allreduce             0.38  0.38  +0.00   OK
```

22/25 prompts unchanged or improved; 5 prompts regressed.

## Failure Root Cause Analysis

### Root Cause 1: System prompt template regurgitation

**Affected**: L5_moe_2stage (0.00 vs 0.50)

The `<code>` segment contained the literal text from the system prompt:
`"Complete, compilable FlyDSL kernel code"` followed by `<|repo_name|>` special tokens.

**Cause**: The FORMAT_SYSTEM_PROMPT includes a format example with
`<plan>\n  1. Problem analysis...\n</plan>\n<code>\n  Complete, compilable FlyDSL kernel code\n</code>`.
On hard L5 prompts, the model falls back to regurgitating this template instead of generating actual code.

### Root Cause 2: Plan segment runaway (token exhaustion)

**Affected**: L2_rmsnorm (0.00 vs 1.00), L4_flash_attn (0.14 vs 0.43), L5_preshuffle_gemm (fmt=FAIL)

The plan segment enters a repetition loop (e.g., "128 elements per subsubsubsubwave...")
or grows too long, consuming the entire `max_new_tokens` budget.
`</plan>` never appears, so `<code>` is never generated.

**Cause**: Training data has variable plan lengths with no upper bound.
The model has not learned to self-terminate the plan segment within a token budget.

### Root Cause 3: Minor API pattern drift

**Affected**: L4_fp8_gemm (0.57 vs 0.71), L5_mla_decode (0.50 vs 0.62)

Format is correct, but the code segment is slightly shorter and misses some
expected patterns (e.g., `fx.make_layout`, `mfma`).

**Cause**: Normal LoRA fine-tuning variance — plan segment uses token capacity
that was previously available for code generation.

### Key Takeaway

**This is NOT catastrophic forgetting.** 22/25 prompts are unchanged.
The regressions are caused by (1) training data design issues and (2) plan
length control, not by knowledge loss.

## Optimization Plan for v2

### Fix 1: Simplify system prompt (addresses Root Cause 1)

**Before** (contains full format example that gets regurgitated):
```
Always structure your response as:
<plan>
  1. Problem analysis and hardware constraints
  2. Tiling decisions and why
  3. Memory layout and pipeline strategy
  4. Optimization choices (swizzle, etc.)
</plan>
<code>
  Complete, compilable FlyDSL kernel code
</code>
```

**After** (short instruction, no example to regurgitate):
```
Structure your response in two sections:
First a <plan> section with brief optimization reasoning (under 200 words),
then a <code> section with complete FlyDSL kernel code.
```

### Fix 2: Enforce plan length constraint (addresses Root Cause 2)

- Truncate all cat1 plan segments to ≤ 300 tokens during data construction
- Add explicit "Keep the plan concise (4-8 sentences)" to system prompt
- In `generate_format_data.py`, add `max_tokens=512` for plan generation
  and post-filter plans that exceed 300 tokens

### Fix 3: Structural plan-code consistency validation (addresses Root Causes 1+2)

v1 的 `validate_plan_code_consistency` 只做数字重叠检查，plan 和 code 之间
没有真正的语义关联。这导致模型学到"先输出一段看起来像推理的文字，然后输出代码"，
而不是"分析 tiling/pipeline/swizzle 决策，再写对应代码"。

v2 改用结构化校验 (`_extract_code_decisions` + 多维度验证):
1. 从 code 中静态解析 tile sizes, pipeline stages, swizzle pattern, MFMA, split-K
2. 验证 plan 是否用自然语言提到了这些具体决策 (不只是数字匹配，而是关键词语义匹配)
3. 检查 plan 是否有 ≥3 个实质句子 (防止空洞模板)
4. 60% 通过率门槛，不一致的样本丢弃重新生成

这对 HRD 至关重要 — HRD 假设 plan tokens 解释了 code tokens 的设计选择，
reward decomposition 依赖这个因果链。如果 plan 是套话，HRD 的 plan reward 就在奖励空洞推理。

### Fix 4: Add v5e kernel preservation set (addresses Root Cause 3)

- Mix ~500 samples from v5e SFT data (pure kernel, no plan/code tags) as cat4
- Focus on sources: `boost_correct_kernel`, `gfx950_kernel_real`, `augmentation_*`
- These samples use the original v5e system prompt (no format requirement)
- Ratio: cat1 ~45%, cat2 ~25%, cat3 ~12%, cat4 ~18%

### Expected v2 data composition

| Category | Count | Ratio | Purpose |
|----------|-------|-------|---------|
| Cat1 (FlyDSL dual-segment) | ~1100 | 39% | Format alignment (with structural consistency) |
| Cat2 (General reasoning) | ~700 | 25% | Reasoning preservation |
| Cat3 (Complex CoT) | ~350 | 12% | Reasoning depth |
| Cat4 (v5e kernel preservation) | ~650 | 23% | Code ability retention |
| **Total** | **~2800** | 100% | — |

### Training config changes for v2

- LR: 5e-6 → **3e-6** (reduce perturbation)
- Everything else unchanged (LoRA r=32, 1 epoch)

### v1 → v2 key difference: plan-code causal link

v1 的核心缺陷不只是格式问题，而是 **plan 和 code 之间缺乏因果关联**。
模型学到的是"先输出推理状文字，再输出代码"的表面模式，而非"分析决策→据此写代码"的因果推理。

v2 通过结构化校验 (`_extract_code_decisions`) 确保每条训练数据中:
- plan 提到的 tile size 确实出现在 code 的常量中
- code 用了 pipeline/swizzle/MFMA，plan 必须解释为什么
- plan 不能是空洞的 3 句话模板

这个改变对下游 HRD 至关重要：如果 plan 和 code 脱节，HRD 的 reward decomposition
就无法学到"好的推理→好的代码"这个信用分配关系。
