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

## MLPerf-Aligned Training: Llama2-70B LoRA SFT (8x MI300X)

Llama2-70B LoRA SFT with FP8 quantization, TP=1 DP=8 parallelism, aligned with the
AMD MLPerf v5.1 `MI300X_EPYC_9575F_pytorch_llama2_70b` reference submission.

**Lumen v33 passes the MLPerf target (val_loss < 0.925)** with best val_loss = 0.9208.

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
# MLPerf-aligned training with data shuffling (recommended)
LUMEN_SHUFFLE_TRAIN=1 bash examples/llama2/run_tp1_dp8.sh
```

The `LUMEN_SHUFFLE_TRAIN=1` environment variable enables epoch-level data shuffling,
matching the AMD MLPerf reference behavior. This is **critical for convergence** —
without it, val_loss stalls at ~0.937 and never reaches the 0.925 target.

### Step 3 — Monitor training

Logs stream to stdout and are tee'd to `~/tp1_dp8_v3.log`:

```bash
# Watch loss and grad norms
tail -f ~/tp1_dp8_v3.log | grep -E "iteration|lm loss|grad_norm"

# Watch validation eval
tail -f ~/tp1_dp8_v3.log | grep "validation loss"

# Quick GPU memory check
rocm-smi --showmeminfo vram
```

### Script chain

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
| Tensor Parallel | 1 | Single-GPU model, same as MLPerf reference |
| Data Parallel | 8 | One rank per GPU |
| Global Batch Size | 8 | MBS=1 x DP=8 |
| Sequence Length | 8192 | |
| Learning Rate | 4e-4 | Matches MLPerf reference |
| LR Schedule | Cosine decay | Over full 1024 steps, min\_lr=0 |
| LR Warmup Steps | 0 | Matches MLPerf reference |
| Synthetic Warmup | 5 steps | Zero loss\_mask — calibrates FP8 scales without updating LoRA weights |
| Weight Decay | 1e-4 | Matches MLPerf |
| Gradient Clip | 0.3 | Matches MLPerf |
| Adam Betas | (0.9, 0.999) | Matches MLPerf |
| LoRA | rank=16, alpha=32, dropout=0.1 | Matches MLPerf |
| FP8 Format | E4M3 | Delayed scaling, amax history=4, most\_recent algorithm |
| FP8 Param Storage | Enabled | Weights stored in FP8 to save memory |
| Activation Recompute | 80 layers (full) | Required for TP=1 memory budget |
| Distributed Optimizer | Enabled | Shards optimizer states across DP ranks |
| Data Shuffling | Enabled (`LUMEN_SHUFFLE_TRAIN=1`) | Epoch-level shuffle matching NeMo reference |
| Seed | 1234 | Fixed for reproducibility |

### Data shuffling

Data shuffling is the single most important factor for reaching the MLPerf target.

The training dataset (`train.npy`) contains 3901 pre-packed samples of 8192 tokens each.
The AMD MLPerf reference (NeMo) shuffles all sample indices into a random permutation at
epoch start (seeded by the training seed). Without shuffling, consecutive mini-batches
contain adjacent packed sequences that are highly correlated, degrading early convergence.

| Setting | Best val_loss | Passes MLPerf? | Steps to target |
|---------|--------------|----------------|-----------------|
| `LUMEN_SHUFFLE_TRAIN=0` (v20) | 0.9371 | No | Never |
| `LUMEN_SHUFFLE_TRAIN=1` (v33) | **0.9208** | **Yes** | 672 |

Implementation: `lumen/models/llama2/dataset.py` — `LLaMA2SFTDataset._build_samples_mapping()`
creates a permuted index array and remaps `__getitem__` through it. Controlled by `LUMEN_SHUFFLE_TRAIN`
env var, passed via `shuffle=True` in `lumen/models/llama2/megatron/sft.py`.

### Expected results

With the recommended configuration (lr=4e-4, 1024 steps, seed=1234, `LUMEN_SHUFFLE_TRAIN=1`):

| Metric | Value |
|--------|-------|
| Initial loss (step 6, after warmup) | ~4.1 |
| Loss at step 100 | ~1.3 |
| Loss at step 500 | ~1.3 |
| Best validation loss | **0.9208** (step 960) |
| Final validation loss (step 1024) | **0.9221** |
| MLPerf target | 0.925 |
| First step under target | 672 |
| Peak GPU memory per device | ~185 GB / 192 GB (96.2%) |
| Step time | ~7.9 s |

See [`results/mlperf_llama2_70b_lora/`](results/mlperf_llama2_70b_lora/) for the full
comparison against the AMD MLPerf reference.

### MLPerf alignment status

All training parameters now match the AMD MLPerf v5.1 reference:

| Parameter | Lumen (v33) | MLPerf Reference | Status |
|-----------|-------------|-----------------|--------|
| Learning Rate | 4e-4 | 4e-4 | Matched |
| LR Warmup | 0 | 0 | Matched |
| LR Schedule | Cosine, 1024 steps | Cosine, 1024 steps | Matched |
| LoRA rank/alpha | 16/32 | 16/32 | Matched |
| FP8 Format | E4M3 hybrid | E4M3 hybrid | Matched |
| Data Shuffling | Epoch-level | Epoch-level | **Matched (v33 fix)** |
| Activation Recompute | 80 layers (full) | 21 layers | Different (TP=1 memory) |
| FP8 Engine | AITER (CK + Triton) | TransformerEngine | Different (kernel impl) |
| Attention | AITER CK FMHA | TE fused CK v3 | Different (same CK kernel) |
| RMSNorm | AITER Triton | TE Triton / apex | Different (higher precision) |

Remaining implementation differences (FP8 engine, attention, RMSNorm) each contribute
< 0.005 val_loss delta individually, as verified by systematic A/B experiments (v25-v32).

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `SIGKILL` during checkpoint load | CPU OOM — 8 ranks each loading 128 GB | Ensure `patch_checkpointing.py` ran (adds `mmap=True`) |
| `HIP out of memory` in forward pass | Activation memory overflow | Increase `RECOMPUTE_NUM_LAYERS` (default 80 = full recompute) |
| `grad_norm: 0.000` every step | Broken autograd chain with LoRA + recompute | Ensure `patch_requires_grad.py` ran |
| NCCL timeout on step 1 | AITER kernel tuning takes > default timeout | Set `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200` (already in `run_tp1_dp8.sh`) |
| Loss spikes / divergence | LoRA scaling bug or LR too high | Ensure `patch_lora_scaling.py` ran; use lr=4e-4 |
| `numpy.product` error on save | Deprecated numpy API in Megatron | Already patched in `run_tp1_dp8.sh`; ensure the `sed` line runs |
| val_loss stuck at ~0.937 | Data not shuffled | Set `LUMEN_SHUFFLE_TRAIN=1` |

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

See [`results/`](results/) for full training logs from LLaMA2-70B SFT runs on MI300X/MI355X GPUs across different quantization configurations and MLPerf comparisons.
