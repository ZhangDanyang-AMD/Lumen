# Data Processing Pipeline for FlyDSL Code Agent Training

This pipeline processes the [aiter](https://github.com/ROCm/aiter.git) repository
to generate training data for a FlyDSL GPU kernel code generation model.
It produces three types of training data:

- **CPT (Continued Pre-Training)** — domain-adapted corpus for next-token prediction
- **SFT (Supervised Fine-Tuning)** — instruction-output pairs for instruction following
- **RL (Reinforcement Learning)** — task specifications for GRPO-based performance optimization

## Pipeline Overview

```
aiter repo ──► generate_manifest.py  (Layer 1: static analysis annotation)
                       │
                       ▼
                 manifest.json
                  ┌────┼────┐
                  ▼    ▼    ▼
    extract_cpt_data  generate_sft_data  prepare_rl_specs
           │               │                   │
           ▼               ▼                   ▼
     cpt_data.jsonl   sft_data.jsonl      rl_specs.json
           │               │                   │
           └───────┬───────┘                   │
                   ▼                           │
          validate_dataset.py ◄────────────────┘
                   │
                   ▼
          export_for_training.py
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   cpt_train.txt  sft_*.json  rl_specs.json
   (plain text)   (ChatML /    (task specs)
                   Alpaca)
```

## Scripts

| Script | Description |
|--------|-------------|
| `taxonomy.yaml` | Unified label taxonomy: operator types, hardware targets, features, complexity levels, content types, and priority tiers |
| `generate_manifest.py` | Scans the aiter repo and produces `manifest.json` with per-file metadata inferred from file paths, code patterns, and structure |
| `extract_cpt_data.py` | Reads the manifest, loads source files, wraps each in a structured document format with metadata headers, and assigns sampling weights |
| `generate_sft_data.py` | Generates instruction-output pairs from kernel code (reverse annotation), tests (parameter extraction), docs (QA conversion), tuned configs, and git history |
| `prepare_rl_specs.py` | Builds a library of RL task specifications from manifest entries, tuned config CSVs, and exploration parameter ranges |
| `validate_dataset.py` | Runs format, content, deduplication, distribution, and consistency checks on all generated datasets |
| `export_for_training.py` | Converts processed data into ChatML (for LLaMA-Factory), Alpaca, and plain text formats with train/val split |
| `run_pipeline.py` | One-click orchestrator that runs all steps in sequence |

## Quick Start (Docker)

### 1. Build the Docker image

```bash
cd dataprocess
docker build -t flydsl-dataprocess .
```

### 2. Clone the aiter repository (if not already present)

```bash
git clone --recursive https://github.com/ROCm/aiter.git /path/to/aiter
```

### 3. Run the full pipeline

```bash
docker run --rm \
  -v /path/to/aiter:/workspace/aiter:ro \
  -v $(pwd)/output:/workspace/dataprocess/output \
  flydsl-dataprocess
```

The output files will appear in `./output/`.

### 4. Run individual steps

```bash
# Only generate the manifest
docker run --rm \
  -v /path/to/aiter:/workspace/aiter:ro \
  -v $(pwd)/output:/workspace/dataprocess/output \
  flydsl-dataprocess \
  --repo /workspace/aiter --steps manifest

# Manifest + CPT only
docker run --rm \
  -v /path/to/aiter:/workspace/aiter:ro \
  -v $(pwd)/output:/workspace/dataprocess/output \
  flydsl-dataprocess \
  --repo /workspace/aiter --steps manifest,cpt
```

### 5. Skip git history extraction (if running outside a git repo)

```bash
docker run --rm \
  -v /path/to/aiter:/workspace/aiter:ro \
  -v $(pwd)/output:/workspace/dataprocess/output \
  flydsl-dataprocess \
  --repo /workspace/aiter --skip-git
```

## Running Without Docker

```bash
pip install pyyaml
python run_pipeline.py --repo /path/to/aiter --output-dir ./output
```

## Output Files

| File | Format | Description |
|------|--------|-------------|
| `manifest.json` | JSON | Per-file metadata for the entire aiter repo |
| `cpt_data.jsonl` | JSONL | Weighted CPT documents with metadata headers |
| `cpt_data_stats.json` | JSON | CPT extraction statistics |
| `cpt_train.txt` | Plain text | Weight-expanded CPT corpus for training |
| `sft_data.jsonl` | JSONL | Raw SFT instruction-output pairs |
| `sft_train.jsonl` / `sft_val.jsonl` | JSONL | Train/val split |
| `sft_train_chatml.json` | JSON | ChatML format for LLaMA-Factory |
| `sft_train_alpaca.json` | JSON | Alpaca format |
| `rl_specs.json` | JSON | RL task specifications |
| `validation_report.json` | JSON | Full validation results |

## Data Sources from aiter

| Source | Path | Content | Estimated Volume |
|--------|------|---------|------------------|
| Triton kernels | `aiter/ops/triton/` | Triton kernel implementations | ~320 files |
| Python ops | `aiter/ops/` | Op wrappers and dispatchers | ~415 files |
| C++ kernels | `csrc/` | HIP C++ kernel implementations | ~360 files |
| HSA binaries | `hsa/` | Pre-compiled GPU binaries | binary (excluded) |
| Operator tests | `op_tests/` | Unit tests with parameterized configs | ~50 files |
| Documentation | `docs/` | ISA optimization, autotuning, etc. | ~15 files |
| Tuned configs | `aiter/configs/` | CSV files with optimal parameters | ~20 CSV files |
| Claude skills | `.claude/skills/` | Expert-level programming guides | varies |

## Sampling Weight Strategy

CPT data is assigned sampling weights based on three factors:

- **Priority** (P0=3.0x, P1=1.5x, P2=0.5x)
- **Quality grade** (gold=1.5x, silver=1.0x, bronze=0.7x)
- **Content type** (expert_skill=2.5x, repo_guide=2.0x, kernel_impl=1.2x, ...)

The final weight is: `priority_weight × grade_weight × type_weight`

Higher-weighted documents are seen more frequently during CPT training,
ensuring the model deeply internalizes critical patterns (kernel implementations,
API documentation, expert guides).

## SFT Data Sources

| Source | Method | Estimated Pairs |
|--------|--------|-----------------|
| Kernel reverse-annotation | Generate instructions from existing kernel code | 100-300 |
| Test parameterization | Extract pytest parameter combos | 30-80 |
| Documentation QA | Convert doc sections to Q&A pairs | 50-150 |
| Tuned config recommendations | CSV configs → parameter advice | 20-40 |
| Git history | Extract bugfix/optimization commit context | 50-200 |

## Extending the Pipeline

### Adding new data sources

1. Add new label values to `taxonomy.yaml` if needed
2. Update pattern matching in `generate_manifest.py`
3. Add a new generator function in `generate_sft_data.py`
4. Add corresponding validation in `validate_dataset.py`

### Integrating with FlyDSL data

This pipeline processes only the aiter repository. To incorporate FlyDSL data,
mount the FlyDSL repo alongside aiter and adjust the `--repo` paths. The manifest
generator and CPT extractor are designed to work with any Python/Markdown codebase.

### AI-assisted annotation (Layer 2 — Multi-Model Consensus)

The pipeline now includes a full multi-model consensus annotation system (Plan §4.4):

| Script | Description |
|--------|-------------|
| `ai_annotate.py` | Generate annotation prompts from manifest |
| `batch_annotate_cursor.py` | Rule-based expert annotator (Model 0) |
| `consensus_annotate.py` | Multi-model annotation + voting + manifest merge |
| `rebuild_with_annotations.py` | Rebuild datasets using AI-enriched manifest |
| `run_consensus_pipeline.sh` | One-click orchestrator for the full annotation pipeline |

**Run multi-model annotation:**

```bash
# Set API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AIza..."
export DEEPSEEK_API_KEY="sk-..."

# Run full consensus pipeline
bash run_consensus_pipeline.sh

# Or run individual models:
python3 consensus_annotate.py annotate \
    --prompts metadata/annotation_prompts.jsonl \
    --output responses_claude.jsonl \
    --api-key $ANTHROPIC_API_KEY \
    --model claude-sonnet-4-20250514 \
    --model-name claude \
    --provider anthropic

# Merge consensus
python3 consensus_annotate.py consensus \
    --responses responses_*.jsonl \
    --output consensus_annotations.jsonl

# Apply to manifest
python3 consensus_annotate.py apply \
    --manifest graded_manifest.json \
    --consensus consensus_annotations.jsonl \
    --output annotated_manifest.json

# Rebuild datasets with AI annotations
python3 rebuild_with_annotations.py \
    --manifest annotated_manifest.json \
    --output-dir /home/danyzhan/flydsl-agent-dataset/data \
    --merge-existing
```
