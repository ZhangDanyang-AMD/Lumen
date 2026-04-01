# LLaMA2 SFT

Full fine-tuning or LoRA on LLaMA2 (7B / 13B / 70B) with FP8 attention, packed sequences, and early stopping.

## Quick Start

```bash
# 1. Prepare data and model checkpoint
bash examples/llama2/scripts/prepare_data_and_model.sh

# 2. Run training — Megatron backend (default)
BACKEND=megatron bash examples/llama2/run_finetune.sh

# 2. Or: FSDP backend (no Megatron dependency)
BACKEND=fsdp bash examples/llama2/run_finetune.sh
```

The training script (`finetune_llama2.py`) selects the backend via `--backend megatron|fsdp`.

## TP=1 DP=8 MLPerf-Aligned Training (8x MI300X)

Llama2-70B LoRA SFT with FP8 quantization, TP=1 DP=8 parallelism, aligned with the
MLPerf `MI300X_EPYC_9575F_pytorch_llama2_70b` reference submission.

### Prerequisites

- **Docker image**: `lumen_unit_test:latest` (contains Megatron-LM-AMD, ROCm, RCCL, AITER)
- **GPUs**: 8x AMD MI300X (192 GB HBM each)
- **Host RAM**: >= 256 GB (checkpoint loading uses mmap but still needs headroom)
- **Disk**: ~300 GB free on `/data1/` (model ~140 GB, dataset ~30 GB, checkpoints/results)
- **Model**: `NousResearch/Llama-2-70b-hf` converted to Megatron TP=1 format
- **Dataset**: `regisss/scrolls_gov_report_preprocessed_mlperf_2` preprocessed to `.npy`

### Step 1 — Download and convert model + dataset

```bash
# Download HuggingFace model
python examples/llama2/scripts/download_model.py

# Convert HF checkpoint to Megatron TP=1 format
python examples/llama2/scripts/convert_to_megatron.py \
    --hf-dir /data1/lumen/nous_llama2_70b_hf \
    --out-dir /data1/lumen/megatron_ckpt_nous_tp1 \
    --tp 1

# Download and preprocess dataset
python examples/llama2/scripts/download_dataset.py
python examples/llama2/scripts/convert_dataset.py \
    --tokenizer examples/llama2/tokenizer \
    --out-dir /data1/lumen/data \
    --seq-len 8192
```

### Step 2 — Launch training

```bash
bash examples/llama2/run_tp1_dp8.sh
```

### Step 3 — Monitor training

Logs stream to stdout and are tee'd to `~/tp1_dp8_v3.log`:

```bash
# Watch loss and grad norms
tail -f ~/tp1_dp8_v3.log | grep -E "iteration|lm loss|grad_norm"

# Watch validation eval
tail -f ~/tp1_dp8_v3.log | grep "eval"

# Quick GPU memory check
rocm-smi --showmeminfo vram
```

### Script chain

`run_tp1_dp8.sh` orchestrates the full pipeline inside a Docker container:

```
[Host]  bash run_tp1_dp8.sh
  └── docker run lumen_unit_test:latest bash -c '...'
        │
        ├── pip install runtime deps (huggingface-hub, sentencepiece, peft, ...)
        ├── Fix numpy.product → numpy.prod (Megatron compat)
        │
        ├── python scripts/patch_gpt_layer_specs.py   # RMSNorm / FusedRMSNorm compat
        ├── python scripts/patch_checkpointing.py     # LoRA base_layer key remap + mmap
        ├── python scripts/patch_requires_grad.py     # Grad flow fix for LoRA + recompute
        │
        ├── python scripts/verify_fp8_quality.py      # FP8 quantization diagnostic
        │
        └── CONFIG=config_MI300X_tp1_dp8.sh bash run_finetune.sh
              └── torchrun --nproc_per_node=8 finetune_llama2.py \
                    --linear-fp8 --fp8-param-storage --lora-rank 16 ...
```

The three Megatron patches are applied at runtime because they modify the container's
Megatron-LM-AMD installation (not part of the Lumen repo). The patches are idempotent
and skip themselves if already applied.

### Key training parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Tensor Parallel | 1 | Matches MLPerf reference |
| Data Parallel | 8 | One rank per GPU |
| Global Batch Size | 8 | MBS=1 x DP=8 |
| Sequence Length | 8192 | |
| Learning Rate | 2e-4 | Stable for Lumen FP8 (MLPerf uses 4e-4 with TE) |
| LR Schedule | Cosine decay | Over full 1024 steps, min\_lr=0 |
| LR Warmup Steps | 100 | Ramp from 0 to peak LR |
| Synthetic Warmup | 5 steps | Zero loss\_mask — calibrates FP8 scales without updating LoRA weights |
| Weight Decay | 1e-4 | Matches MLPerf |
| Gradient Clip | 0.3 | Matches MLPerf |
| Adam Betas | (0.9, 0.999) | Matches MLPerf |
| LoRA | rank=16, alpha=32, dropout=0.1 | Matches MLPerf |
| FP8 Format | E4M3 | Delayed scaling, amax history=4, most\_recent algorithm |
| FP8 Param Storage | Enabled | Weights stored in FP8 to save memory |
| Activation Recompute | 80 layers (full) | Required for TP=1 memory budget |
| Distributed Optimizer | Enabled | Shards optimizer states across DP ranks |
| Seed | 1234 | Fixed for reproducibility |

### MLPerf alignment notes

The configuration matches the MLPerf MI300X submission for all parameters except:

| Parameter | Lumen | MLPerf | Reason |
|-----------|-------|--------|--------|
| Learning Rate | 2e-4 | 4e-4 | Lumen FP8 (AITER backend) is more sensitive to high LR than TransformerEngine |
| LR Warmup | 100 steps | 0 | Compensates for the FP8 precision gap at training start |
| Activation Recompute Layers | 80 (full) | 21 | TP=1 puts all parameters on one GPU; full recompute needed to fit in 192 GB |

Root cause of the LR sensitivity: Lumen's AITER GEMM kernels do not guarantee FP32
accumulation for partial sums, and delayed scaling uses stale amax from the previous
iteration. These compound across 80 layers at high learning rates.

### Configuration

All training parameters live in `config_MI300X_tp1_dp8.sh`. Key variables to customize:

```bash
export LR=2e-4           # Learning rate (lower = more stable, higher = faster convergence)
export TRAIN_STEPS=1024   # Total training iterations
export CKPT_DIR="/data1/lumen/megatron_ckpt_nous_tp1"  # Megatron-format checkpoint
export TRAIN_DATA="/data1/lumen/data/train.npy"         # Training data
export VALID_DATA="/data1/lumen/data/validation.npy"    # Validation data
export WARMUP_STEPS=5     # Synthetic warmup steps (FP8 scale calibration)
export LR_WARMUP_STEPS=100  # LR linear ramp-up steps
```

### Expected results

With the default configuration (lr=2e-4, 1024 steps, seed=1234):

| Metric | Value |
|--------|-------|
| Initial loss (step 6, after warmup) | ~4.0 |
| Loss at step 100 | ~1.3 |
| Loss at step 500 | ~1.0 |
| Best validation loss | ~0.96 |
| MLPerf target | 0.925 |
| Peak GPU memory per device | ~170 GB / 192 GB |

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `SIGKILL` during checkpoint load | CPU OOM — 8 ranks each loading 128 GB | Ensure `patch_checkpointing.py` ran (adds `mmap=True`) |
| `HIP out of memory` in forward pass | Activation memory overflow | Increase `RECOMPUTE_NUM_LAYERS` (default 80 = full recompute) |
| `grad_norm: 0.000` every step | Broken autograd chain with LoRA + recompute | Ensure `patch_requires_grad.py` ran |
| NCCL timeout on step 1 | AITER kernel tuning takes > default timeout | Set `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200` (already in `run_tp1_dp8.sh`) |
| Loss spikes / divergence | LR too high for Lumen FP8 | Lower `LR` (2e-4 is known stable) and/or increase `LR_WARMUP_STEPS` |
| `numpy.product` error on save | Deprecated numpy API in Megatron | Already patched in `run_tp1_dp8.sh`; ensure the `sed` line runs |
| `IndexError` in validation | Validation dataset too short | Not critical — training results are unaffected |

## CLI Flags

| Feature | CLI Flag |
|---------|----------|
| Attention backend | `--lumen-attn-backend {aiter_csrc,aiter_triton,aiter_triton_fp8,aiter_csrc_fp8}` |
| FP8 quantised training | `--linear-fp8 --fp8-format e4m3` |
| MXFP8 block sizes | `--mxfp8-block-m-fwd 128 ...` (6 independent dims) |
| LoRA | `--lora-rank 16 --lora-alpha 32` |
| LoRA A2A comm opt | `--lora-a2a` |
| Synthetic warmup | `--warmup-steps 5` |
| Early stopping | `--val-loss-target 1.5` |
| Context Parallelism | `--context-parallel-size 2` |

See `run_finetune.sh` for the full list of environment variables and defaults.

## Megatron Patches

When running inside the Docker container, three patches are applied to the container's
Megatron-LM-AMD installation. These live in `scripts/` and are applied at launch:

| Patch | Purpose |
|-------|---------|
| `patch_gpt_layer_specs.py` | Creates `MegatronFusedRMSNorm` wrapper; patches `gpt_layer_specs.py` and `transformer_block.py` to use it when RMSNorm is detected |
| `patch_checkpointing.py` | Remaps checkpoint keys for LoRA `base_layer` wrapping; injects `mmap=True` into `torch.load` to prevent CPU OOM with 8 ranks loading a 128 GB checkpoint |
| `patch_requires_grad.py` | Forces `hidden_states.requires_grad_(True)` before `_checkpointed_forward` so LoRA gradients flow through activation checkpointing |

Debug/diagnostic patches are available in `scripts/debug/`.

## Reference Logs

See [`results/`](results/) for full training logs from LLaMA2-70B SFT runs on 8x MI355X GPUs across different quantization configurations (BF16, FP8 blockwise, MXFP8, FSDP).
