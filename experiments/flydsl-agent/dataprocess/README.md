# FlyDSL Agent Data Processing Pipeline (v3.0)

Processes [FlyDSL](https://github.com/ROCm/FlyDSL), [aiter](https://github.com/ROCm/aiter),
and AMD GPU architecture docs (gpu-docs) to generate training data for the FlyDSL GPU kernel
code generation agent. Produces three training splits:

- **CPT** — domain corpus for continued pre-training
- **SFT** — instruction-response pairs + refusal boundary training (§13)
- **RL** — task specs for GRPO-based kernel quality optimization

## Pipeline Architecture

```
FlyDSL + aiter + gpu-docs
         │
         ▼
  process_all_v2.py ─────────────────── Main pipeline (manifest → CPT/SFT/RL)
         │
    ┌────┼────────────────────┐
    ▼    ▼                    ▼
  CPT  SFT  RL           manifest.json
    │    │    │                │
    ▼    ▼    ▼                ▼
  validate_dataset.py    benchmark_filter.py ─── GPU quality grading
         │                     │
         ▼                     ▼
  package_hf_dataset.py   graded_manifest.json
         │                     │
         ▼                     ▼
  flydsl-agent-dataset/   ai_annotate.py ──────── Prompt generation
                               │
                               ▼
                          consensus_annotate.py ── 5-model voting
                               │
                               ▼
                          rebuild_with_annotations.py
```

## Scripts

### Core Pipeline

| Script | Lines | Description |
|--------|------:|-------------|
| `process_all_v2.py` | 1,410 | **Main pipeline**: dual-repo scan → manifest → CPT/SFT/RL generation (sources A-I) |
| `validate_dataset.py` | 315 | Format, content, dedup, distribution, and consistency checks |
| `package_hf_dataset.py` | 345 | Package into HuggingFace dataset repo with README card |
| `taxonomy.yaml` | 66 | Label taxonomy: operators, hardware, features, complexity |

### GPU Benchmarking

| Script | Lines | Description |
|--------|------:|-------------|
| `benchmark_filter.py` | 270 | 3-gate quality filter: compile → correctness → grade assignment |
| `perf_benchmark.py` | 298 | Performance benchmarks: latency, TFLOPS, roofline efficiency |
| `update_manifest_perf.py` | 168 | Merge benchmark results into manifest |

### Multi-Model AI Annotation

| Script | Lines | Description |
|--------|------:|-------------|
| `ai_annotate.py` | 218 | Generate structured annotation prompts from manifest |
| `batch_annotate_cursor.py` | 246 | Rule-based expert annotator (Model 0) |
| `consensus_annotate.py` | 451 | Multi-model annotation + majority voting + manifest merge |
| `rebuild_with_annotations.py` | 211 | Rebuild datasets with AI-enriched metadata |

### SFT Data Enhancement (v2)

| Script | Lines | Description |
|--------|------:|-------------|
| `enhance_sft_data.py` | 230 | SFT v2: kernel code extraction + weighted resampling (19% → 59% kernel) |

### Orchestration

| Script | Lines | Description |
|--------|------:|-------------|
| `run_in_docker.sh` | 73 | Docker orchestrator: LLVM → FlyDSL → benchmark |
| `run_consensus_pipeline.sh` | 150 | One-click 5-model annotation pipeline |
| `Dockerfile` | 26 | Container image for the pipeline |

**Total: 16 active files, ~4,480 lines**

> Archived v1 scripts (superseded by `process_all_v2.py`) are in `_archived_v1/`.

## Quick Start

### Full Pipeline (one command)

```bash
python3 process_all_v2.py
```

Environment variables:
- `FLYDSL_ROOT` — path to FlyDSL repo (default: `/home/danyzhan/FlyDSL`)
- `AITER_ROOT` — path to aiter repo (default: `/home/danyzhan/aiter`)
- `OUTPUT_DIR` — output directory (default: `/home/danyzhan`)

### Full Pipeline (Docker)

```bash
bash run_in_docker.sh
```

### GPU Benchmark Grading (requires MI350X)

```bash
python3 benchmark_filter.py \
    --manifest manifest.json \
    --output graded_manifest.json
```

### Multi-Model AI Annotation

```bash
bash run_consensus_pipeline.sh
```

Or step by step:

```bash
# Generate prompts
python3 ai_annotate.py --manifest graded_manifest.json

# Run 5-model annotation
python3 consensus_annotate.py annotate --prompts prompts.jsonl ...

# Consensus voting
python3 consensus_annotate.py consensus --responses resp_*.jsonl --output consensus.jsonl

# Apply + rebuild
python3 consensus_annotate.py apply --manifest graded.json --consensus consensus.jsonl
python3 rebuild_with_annotations.py --manifest annotated.json --output-dir data/ --merge-existing
```

### Validate & Package

```bash
python3 validate_dataset.py
python3 package_hf_dataset.py
```

### SFT Data Enhancement

```bash
python3 enhance_sft_data.py \
    --input  /home/danyzhan/flydsl-agent-dataset/data/sft/train-00000-of-00001.jsonl \
    --rl-specs /home/danyzhan/flydsl-agent-dataset/data/rl/train-00000-of-00001.jsonl \
    --cpt-data /home/danyzhan/flydsl-agent-dataset/data/cpt/train-00000-of-00001.jsonl \
    --metadata-dir /home/danyzhan/flydsl-agent-metadata \
    --output /home/danyzhan/flydsl-agent-dataset/data/sft/train-00000-of-00001.jsonl
```

## Output: flydsl-agent-dataset/

| Split | Train | Val | Format |
|-------|------:|----:|--------|
| CPT | 1,967 | — | `{text, meta}` |
| SFT (v2) | 2,916 | 264 | `{messages, source, metadata}` — 59% kernel code (was 19%) |
| RL | 2,591 | 287 | `{id, operator, hardware, params}` |

## Extending

1. Add new taxonomy labels → `taxonomy.yaml`
2. Add new SFT source → `process_all_v2.py` (add `source_X_*` function)
3. Add new validation → `validate_dataset.py`
