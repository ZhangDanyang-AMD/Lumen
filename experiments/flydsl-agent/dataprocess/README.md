# FlyDSL Agent Data Processing Pipeline (v5e)

Processes [FlyDSL](https://github.com/ROCm/FlyDSL), [aiter](https://github.com/ROCm/aiter),
and AMD GPU docs to generate training data for the FlyDSL GPU kernel code generation agent.

## Current Status

| Stage | Model | Overall | Sandbox Compile | HuggingFace |
|-------|-------|---------|-----------------|-------------|
| **SFT v5e** | Qwen2.5-Coder-32B | **74.1%** | **21.9% (42/192)** | [Zhangdanyang/Qwen2.5-Coder-SFT-v5e](https://huggingface.co/Zhangdanyang/Qwen2.5-Coder-SFT-v5e) |
| RFT v1 | (in progress) | — | — | — |

## Pipeline Architecture

```
FlyDSL + aiter + gpu-docs
         │
         ▼
  process_all_v2.py ────── Base pipeline: manifest → CPT/SFT/RL raw data
         │
         ├── enhance_sft_data.py ─────── v2: kernel extraction + resampling
         ├── fix_import_sft.py ───────── v3: import correction pairs
         ├── clean_hw_features.py ────── v4: hw-feature mismatch cleanup
         ├── enhance_sft_v5.py ───────── v5: gfx950 kernels + API fixes
         ├── add_gluon_tutorials.py ──── v5: Gluon GEMM optimization tutorials
         ├── fix_import_chain.py ─────── v5b: import chain navigation
         ├── fix_api_hallucination.py ── v5b: API reference + hallucination fixes
         ├── add_module_digest.py ────── v5c/v5d: 30-module structure digest
         └── boost_correct_kernels.py ── v5e: correct kernel pattern boost
                  │
                  ▼
         SFT v5e dataset (3,889 samples, 52% correct kernel code)
                  │
                  ▼
         RFT pipeline (rft-stage1/)
           generate_candidates.py → verify_candidates.py → build_rft_dataset.py
```

## Scripts

### Core Pipeline (v1-v2 era)

| Script | Description |
|--------|-------------|
| `process_all_v2.py` | Main pipeline: dual-repo scan → manifest → CPT/SFT/RL |
| `validate_dataset.py` | Format, content, dedup, distribution checks |
| `package_hf_dataset.py` | Package into HuggingFace dataset repo |
| `taxonomy.yaml` | Label taxonomy: operators, hardware, features |

### GPU Benchmarking

| Script | Description |
|--------|-------------|
| `benchmark_filter.py` | 3-gate quality filter: compile → correctness → grade |
| `perf_benchmark.py` | Performance benchmarks: latency, TFLOPS, roofline |
| `update_manifest_perf.py` | Merge benchmark results into manifest |

### Multi-Model AI Annotation

| Script | Description |
|--------|-------------|
| `ai_annotate.py` | Generate annotation prompts from manifest |
| `batch_annotate_cursor.py` | Rule-based expert annotator |
| `consensus_annotate.py` | Multi-model annotation + majority voting |
| `rebuild_with_annotations.py` | Rebuild datasets with AI-enriched metadata |

### SFT Data Enhancement (v2 → v5e)

| Script | Version | Description |
|--------|---------|-------------|
| `enhance_sft_data.py` | v2 | Kernel extraction + weighted resampling (18% → 59% kernel) |
| `fix_import_sft.py` | v3 | 60 import correction pairs (81% import hallucination fix) |
| `clean_hw_features.py` | v4 | Remove gfx1250 features (wmma/tdm/mxfp4) from gfx950 samples |
| `enhance_sft_v5.py` | v5 | +156 real gfx950 kernels, no-markdown templates, API type fixes |
| `add_gluon_tutorials.py` | v5 | Gluon GEMM v0→v9 progressive optimization tutorials |
| `fix_import_chain.py` | v5b | Import chain navigation: "need X → which module → how to import" |
| `fix_api_hallucination.py` | v5b | Module API reference + hallucination correction pairs |
| `add_module_digest.py` | v5c/v5d | Complete 30-module structure digest + exhaustive "does not exist" list |
| `boost_correct_kernels.py` | v5e | 247 correct kernels ×3 prompts + 8 mini-kernels ×5 + flyc boundary QA |

### Orchestration

| Script | Description |
|--------|-------------|
| `run_in_docker.sh` | Docker orchestrator for base pipeline |
| `run_consensus_pipeline.sh` | One-click 5-model annotation pipeline |
| `Dockerfile` | Container image for the pipeline |

## SFT Version History

| Version | Samples | Kernel% | Overall | Sandbox | Key Change |
|---------|---------|---------|---------|---------|------------|
| v1 | 2,808 | 18% | 56% | — | Baseline |
| v2 | 2,916 | 59% | 72% | 0% | Kernel extraction + resampling |
| v3 | 3,096 | 60% | — | 0% | Import correction pairs |
| v4 | 2,344 | 55% | 60% | 0.5% | hw-feature mismatch cleanup |
| v5 | 2,596 | 58% | 69% | 0% | Real gfx950 kernels |
| v5b | 2,792 | 60% | 76.5% | 4.7% | Import chains + API reference |
| v5c | 2,943 | 62% | 73.6% | 10.4% | 30-module structure digest |
| v5d | 3,102 | 63% | — | 9.9% | Expanded negative list (whack-a-mole failed) |
| **v5e** | **3,889** | **52%** | **74.1%** | **21.9%** | **Correct kernel boost (positive flooding)** |

## Key Lessons

1. **Positive examples beat negative examples** — v5d showed that adding "does not exist" entries is whack-a-mole (fix one hallucination, model invents another). v5e's positive flooding (kernel ratio 35%→52%) doubled sandbox compile rate.
2. **Data quality > data quantity** — v4 removed 24% of data (hw-feature mismatches) and the model improved on the key metric (sandbox compilation).
3. **Hallucinations are layered** — v3 fixed top-level imports, v5 discovered second-level API hallucinations, v5c/v5d found the long tail never ends.
4. **Post-processing matters** — 66% of "syntax errors" were special token leakage, fixed in code without retraining.

## Output: flydsl-agent-dataset/

| Split | Train | Val | Format |
|-------|------:|----:|--------|
| SFT (v5e) | 3,889 | 264 | `{messages, source, metadata}` — 52% correct kernel |
| RL | 2,563 | 287 | `{id, operator, hardware, params}` |
