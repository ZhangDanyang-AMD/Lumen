# FlyDSL Agent — CPT (Continued Pre-Training)

Stage 1 of the FlyDSL Agent three-stage post-training pipeline (CPT -> SFT -> RL).
Injects FlyDSL/aiter/AMD GPU domain knowledge into Qwen2.5-Coder-32B via
next-token prediction with LoRA.

## Quick Start

```bash
# 1. Download base model to /dev/shm (fast RAM-backed storage)
python download_model.py

# 2. Build Docker image
bash build.sh

# 3. Run training (8x MI355X)
bash run_cpt.sh

# Smoke test (5 steps only)
MAX_STEPS=5 bash run_cpt.sh
```

## Architecture

```
Qwen2.5-Coder-32B (frozen, BF16)
    + LoRA adapters (r=64, alpha=128, trainable)
    + FSDP sharding (8-way data parallel)
    + Gradient checkpointing (memory efficiency)
                |
    CPT Dataset (1,967 JSONL docs, weighted sampling)
                |
    Next-token prediction loss (all tokens)
                |
    Cosine LR schedule (2e-5 -> 0, warmup 6 steps)
```

### Why LoRA r=64 (not full fine-tuning)?

- 32B params in BF16 = 64GB per GPU for full fine-tuning optimizer states
- LoRA r=64 trains ~1-2% of parameters, fitting in 8x MI355X memory
- CPT is knowledge injection (vocabulary + patterns), not capability change
- Higher rank (64 vs 32 for SFT) because CPT needs to learn entirely new domain tokens

### Why weighted sampling?

The dataset has heterogeneous quality. Expert-authored Claude skills (14 guides)
have 7.5x weight while build scripts get 0.2x. This ensures the model sees
high-value content proportionally more often per epoch.

| Content Type | Weight | Rationale |
|---|---:|---|
| Claude expert skills | 7.5 | Highest information density |
| CLAUDE.md project graph | 6.0 | Repository navigation + GPU tables |
| Gold kernels (>70% roofline) | 5.4 | Production-quality code |
| FlyDSL documentation | 4.5 | Official API guides |
| Framework API source | 3.0 | Core DSL infrastructure |
| gpu-docs hardware specs | 0.5 | Internalize but don't over-memorize |
| Build scripts | 0.2 | Low-priority background |

## Training Hyperparameters

From plan.md section 7.2:

| Parameter | Value | Rationale |
|---|---|---|
| Base model | Qwen2.5-Coder-32B | Best open-source code model at 32B scale |
| Method | LoRA | Memory-efficient, 32B model on 8x MI355X |
| LoRA rank | 64 | CPT needs more capacity than SFT |
| LoRA alpha | 128 | alpha = 2 x rank |
| LoRA targets | q/k/v/o/gate/up/down_proj | All attention + MLP projections |
| Learning rate | 2e-5 | Higher LR for learning new domain |
| LR schedule | Cosine with warmup | 6-step warmup (~5% of 125 steps) |
| Weight decay | 0.01 | Standard regularization |
| Sequence length | 8192 | FlyDSL kernels average 500-2000 lines |
| Micro batch size | 2 per GPU | |
| Gradient accumulation | 2 | Effective batch = 2 x 8 x 2 = 32 |
| Epochs | 3 | Small dataset needs multiple passes |
| Total steps | ~125 | 33M tokens / (32 x 8192) |
| Precision | BF16 | No FP8 for simplicity (CPT is only ~1.5h) |

## Expected Results

From plan.md section 8.1:

| Metric | Before CPT | After CPT | Threshold |
|---|---|---|---|
| FlyDSL perplexity | >50 | <10 | <15 |
| API completion (Top-1) | ~5% | >60% | >60% |
| API completion (Top-5) | ~15% | >85% | >85% |
| HumanEval regression | baseline | <5pt drop | <5pt |

### Training time

~1.5 hours on 8x MI355X (33M tokens at ~6K tokens/sec).

## Post-Training: LoRA Merge

After CPT, merge the LoRA adapter back into the base weights to create a
clean starting point for SFT:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(
    "/dev/shm/qwen2.5-coder-32b",
    torch_dtype="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, "./checkpoints/final")
merged = model.merge_and_unload()
merged.save_pretrained("./merged_cpt_model")
```

The merged model becomes the base for SFT Stage 2.

## File Structure

```
cpt/
├── Dockerfile          # Docker image (ROCm + AITER + Lumen + PEFT)
├── README.md           # This file
├── build.sh            # Build Docker image
├── run_cpt.sh          # Launch training in Docker
├── config_cpt.sh       # Training hyperparameters
├── train_cpt.py        # Training entry point (FSDP + LoRA)
├── dataset.py          # CPT dataset with weighted sampling
└── download_model.py   # Download Qwen2.5-Coder-32B to /dev/shm
```

## Docker Usage

The Docker image is based on the same ROCm base as `examples/llama2/Dockerfile`.
It includes AITER (for Lumen operator patching) and HuggingFace PEFT (for LoRA).

```bash
# Build
bash build.sh

# Run with custom paths
HOST_MODEL=/path/to/model \
HOST_DATA=/path/to/dataset \
HOST_RESULTS=/path/to/output \
bash run_cpt.sh
```

The launch script mounts host Lumen code into the container (overlay), so code
changes don't require rebuilding the image.

## Customization

Override any hyperparameter via environment variables:

```bash
# Smaller batch for memory debugging
MBS=1 GRAD_ACCUM=1 bash run_cpt.sh

# Different model
MODEL=/path/to/other/model bash run_cpt.sh

# More steps
MAX_STEPS=200 bash run_cpt.sh

# Without Docker (if AITER/Lumen are installed locally)
source config_cpt.sh
torchrun --nproc_per_node=8 train_cpt.py \
    --model-name-or-path /dev/shm/qwen2.5-coder-32b \
    --train-data-path /home/danyzhan/flydsl-agent-dataset/data/cpt/train-00000-of-00001.jsonl
```
