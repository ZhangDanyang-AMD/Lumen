# Lumen

A lightweight, AMD-native quantized training framework for large language models.

Lumen manages the **quantized training lifecycle** — the vertical path a low-precision tensor takes through forward, backward, optimizer, and communication. It supports **FP8 (E4M3/E5M2)**, **MXFP8**, and **FP4** formats with a unified `QuantConfig` interface. It also integrates high-performance **FP8/MXFP8 attention kernels** (Triton + CK backends) directly, eliminating the dependency on Transformer Engine for end-to-end finetuning.

## Why Lumen?

| | TransformerEngine | Lumen |
|---|---|---|
| Codebase | ~200K lines C++/CUDA/Python | Lightweight Python + Triton |
| Scope | Monolith (attention, norms, GEMM, FP8, comm) | Quantized lifecycle + attention |
| AMD support | Hipified fork | AMD-native (AITER kernels) |
| Install time | Hours (C++ compilation) | Seconds (pure Python + AITER) |
| Integration | Requires module replacement | Non-invasive `quant.enable(model)` |

## Architecture

Lumen owns the quantized training lifecycle and delegates everything else (optimizer, data loading, distributed orchestration) to the training backend:

```
┌──────────────────────────────────────────────────┐
│  MODEL LIBRARY                                   │
│     LLaMA2 SFT / LLaMA 3.1 Pretrain             │
│     Megatron-LM or FSDP backend                  │
│     LoRA, early stopping, synthetic warmup       │
├──────────────────────────────────────────────────┤
│  NON-INVASIVE QUANTIZATION  quant.enable(model)  │
│     Patches nn.Linear in-place (no module swap)  │
│     FP8 E4M3 / E5M2 / MXFP8 / FP4 formats      │
│     QuantConfig — one object for all settings    │
├────────────────────────┬─────────────────────────┤
│  SCALING MANAGER       │  DISTRIBUTED MANAGEMENT  │
│  Per-tensor / block /  │  Param & grad buffer     │
│    MXFP8 scaling       │    (FP8/BF16 contiguous) │
│  Amax history with     │  Distributed optimizer   │
│    delayed scaling     │    (shard + all-gather)   │
│  AMD FNUZ auto-detect  │  FP8 param-gather        │
│  FP8 param lifecycle:  │    (uint8 comm, 2× save) │
│    quantize / dequant  │  Overlap: AG ↔ fwd,      │
│    scale & amax sync   │    RS ↔ bwd              │
├────────────────────────┼─────────────────────────┤
│  ATTENTION KERNELS     │  QUANTIZED LINEAR        │
│  aiter_csrc — CK FA   │  Fused: quant → GEMM     │
│  aiter_triton — Triton │    → dequant (one op)    │
│  aiter_triton_fp8 —   │  AITER hipBLASLt or      │
│    FP8 block / MXFP8  │    Triton backend         │
│  aiter_csrc_fp8 — CK  │  torch.compile compatible│
│  Context Parallelism   │                          │
├────────────────────────┬─────────────────────────┤
│  AITER                 │  MORI                    │
│  (kernel provider)     │  (communication provider)│
│  CK asm kernels        │  MORI-CCL (AG, RS, AR)   │
│  hipBLASLt, Triton     │  MORI-EP  (MoE dispatch) │
│       ▲                │  Device-side RDMA / FP8  │
│       │                │  AINIC / CX-7 / Thor2    │
│  serves: QUANTIZED     │       ▲                  │
│  LINEAR + ATTENTION    │       │                  │
│                        │  serves: DISTRIBUTED     │
│                        │  MANAGEMENT              │
└────────────────────────┴─────────────────────────┘
```

## Quick Start

### Quantized Training (non-invasive patching)

```python
import lumen.quantize as quant
from lumen.quantize import AmaxAlgo, QuantConfig, QuantFormat, ScalingType

# Full config object
config = QuantConfig(
    format=QuantFormat.FP8_E4M3,       # FP8_E5M2, HYBRID, MXFP8
    scaling=ScalingType.DELAYED,        # DYNAMIC, BLOCKWISE
    amax_algo=AmaxAlgo.MAX,             # or MOST_RECENT
    history_len=16,
    quantize_activation=True,           # False → weight-only quantization
    quantize_grad="fp8",                # None, "fp8", "mxfp8", "fp4"
)
quant.enable(model, config=config)

# Or use string shorthand — same effect
quant.enable(model, format="fp8_e4m3", scaling="delayed")

# Training loop is unchanged
output = model(input)       # Lumen handles quantized dispatch
loss.backward()             # Lumen handles quantized gradients
optimizer.step()
```

### FP8 Attention (module API)

```python
from lumen.modules import LumenAttention

# BF16 attention via AITER CK (default)
attn = LumenAttention(causal=True, backend_type="aiter_csrc")

# FP8 blockwise attention via AITER Triton
attn = LumenAttention(causal=True, backend_type="aiter_triton", quant_type="fp8_blockwise")

# MXFP8 attention via Triton (gfx950, configurable block sizes)
attn = LumenAttention(
    causal=True,
    backend_type="aiter_triton",
    quant_type="mxfp8",
    block_m_fwd=64, block_n_fwd=64,     # forward pass tile sizes
    block_m_dq_bwd=64, block_n_dq_bwd=64,  # backward dQ tile sizes
    block_m_dkv_bwd=64, block_n_dkv_bwd=64, # backward dKV tile sizes
    quant_block_size=32,
)

output = attn(q, k, v)       # q, k, v: [B, S, H, D]
```

### Functional API

```python
from lumen.ops.attention import attention, attention_fp8_quant

# Standard attention — auto-selects aiter_csrc if available, else aiter_triton
output = attention(q, k, v, causal=True, backend_type="auto")

# FP8 blockwise attention (Triton only)
output = attention_fp8_quant(q, k, v, causal=True, quant_type="fp8_blockwise")

# MXFP8 attention (Triton only)
output = attention_fp8_quant(q, k, v, causal=True, quant_type="mxfp8",
                             quant_block_size=32)

# Context Parallelism (All-to-All)
output = attention(q, k, v, causal=True,
                   cp_param_bundle={"cp_group": cp_group, "cp_comm_type": "a2a"})
```

## Training Backends: Megatron vs FSDP

Every model in Lumen supports **two independent training stacks**, selected via `--backend megatron|fsdp`. Shared logic (FP8 enablement, LoRA, warmup, early stopping) is factored into `lumen.models.megatron` and `lumen.models.fsdp`; model-specific code (batch construction, dataset, CLI args) stays in the per-model subpackages.

```
                         ┌──────────────────────────────────────────┐
                         │            Training Script               │
                         │  --backend megatron    --backend fsdp    │
                         └────────┬─────────────────────┬──────────┘
                                  │                     │
              ┌───────────────────▼────────┐ ┌──────────▼──────────────────┐
              │   Megatron-LM stack    │ │     PyTorch FSDP stack      │
              │                            │ │                             │
              │  • TP / PP / CP / VP / SP  │ │  • FSDP sharding            │
              │  • Megatron pretrain()     │ │  • HuggingFace Transformers │
              │  • LumenDotProductAttention│ │  • HuggingFace PEFT (LoRA)  │
              │    (replaces core_attn)    │ │  • Gradient checkpointing   │
              │  • LumenRMSNorm            │ │                             │
              │    (Triton-accelerated)    │ │                             │
              │  • Megatron LoRA adapters  │ │                             │
              └───────────┬────────────────┘ └─────────────┬───────────────┘
                          │                                │
              ┌───────────▼────────────────────────────────▼───────────────┐
              │              Lumen (shared across both stacks)             │
              │                                                           │
              │  quant.enable(model)    — non-invasive FP8 patching       │
              │  LumenAttention         — FP8 / MXFP8 / BF16 attention   │
              │  ScalingManager         — per-layer amax / scale / quant  │
              │  DistributedManager     — param & grad buffer, dist-opt   │
              │    FP8 param-gather     — uint8 all-gather, 2× BW saving │
              │    overlap AG ↔ fwd     — bucket-wise async pipeline     │
              │    overlap RS ↔ bwd     — grad reduce with backward      │
              │  reset_fp8_state()      — post-warmup scale reset        │
              └──────────────┬──────────────────────────┬──────────────┘
                             │                          │
              ┌──────────────▼──────────┐ ┌─────────────▼──────────────┐
              │  AITER (kernel backend) │ │  MORI (comm backend)       │
              │                         │ │                            │
              │  CK flash-attention     │ │  MORI-CCL — AG, RS, AR    │
              │  hipBLASLt FP8 GEMM     │ │  MORI-EP  — MoE dispatch  │
              │  Triton FP8 / MXFP8     │ │  Device-side RDMA / FP8   │
              │  Triton RMSNorm         │ │  AINIC / CX-7 / Thor2     │
              └─────────────────────────┘ └────────────────────────────┘
```

### Megatron stack (`lumen.models.megatron`)

Uses [Megatron-LM](https://github.com/ROCm/Megatron-LM) as the training backbone. Lumen injects itself at four points:

1. **Aiter Attention** — `LumenDotProductAttention` for Megatron (the same as `DotProductAttention`) in every transformer layer via spec patching. Supports AITER backends.
2. **FP8 quantized linear** — `quant.enable(model)` patches all `ColumnParallelLinear` / `RowParallelLinear` layers with FP8 forward/backward, preserving Megatron's sequence-parallel all-gather / reduce-scatter communication.
3. **Distributed management** — FP8/BF16 contiguous param & grad buffers, distributed optimizer with shard + all-gather, FP8 param-gather (uint8 communication, 2x memory saving), and communication-computation overlap (all-gather ↔ forward, reduce-scatter ↔ backward).
4. **MORI communication** — Native RDMA + GPU communication via MORI-CCL (all-gather, reduce-scatter, all-reduce) and MORI-EP (MoE expert dispatch), with device-side RDMA for FP8 payloads.

Supports full parallelism: TP, PP, CP (All-to-All), VP, SP, distributed optimizer.

### FSDP stack (`lumen.models.fsdp`)

Uses PyTorch FSDP + HuggingFace Transformers. No Megatron dependency. Lumen provides:

1. **FP8 quantized training** — same `quant.enable(model)` API, patches all `nn.Linear` layers.
2. **LoRA** — via HuggingFace PEFT (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
3. **FP8 state management** — `reset_fp8_state()` with FSDP-aware `.module` unwrapping.

### Usage

```python
# Megatron backend — full Megatron parallelism
from lumen.models.llama2.megatron import (
    lumen_gpt_builder, forward_step, apply_fp8_training,
    apply_lora, add_finetune_args, train_valid_test_datasets_provider,
)

# FSDP backend — lightweight, no Megatron dependency
from lumen.models.llama2.fsdp import FSDPTrainer, get_args
args = get_args()
trainer = FSDPTrainer(args)
trainer.train()
```

## What the Framework Does NOT Do

Staying lightweight means explicitly excluding:

| Feature | Why Excluded | Use Instead |
|---------|-------------|-------------|
| Attention kernels | FlashAttention-CK is better than anything we'd write | FlashAttention |
| Fused norms (RMSNorm, LayerNorm) | Liger Kernel already has AMD-tuned Triton versions | Liger Kernel |
| Fused activations (SwiGLU, GeLU) | Liger Kernel | Liger Kernel |
| RL loss kernels (GRPO, DPO) | Liger Kernel has these with 80% mem savings | Liger Kernel |
| MoE dispatch | AITER has 3x faster fused MoE | AITER |
| MLA/MHA kernels | AITER has 17x/14x faster versions | AITER |
| Distributed training orchestration | FSDP2/DeepSpeed already handle this | FSDP2/DeepSpeed |
| RL training loop | veRL/OpenRLHF/TRL own this | veRL |
| Model definitions | HuggingFace Transformers | HuggingFace |

**The framework is ~1000-2000 lines of Python.** It owns FP8 lifecycle. Everything else is delegated.

## Installation

**Requirements**: PyTorch 2.x, ROCm, Triton.

### User Install (recommended)

```bash
# Core (Triton-only attention backends)
pip install lumen

# With AITER CK attention backend
pip install lumen[aiter]

# All optional dependencies
pip install lumen[all]
```

### Developer Install

```bash
git clone git@github.com:ZhangDanyang-AMD/Lumen.git
cd Lumen

# Editable install with dev dependencies
pip install -e ".[dev]"
```

### Third-party Libraries

| Library | PyPI Package | Purpose |
|---------|-------------|---------|
| [AITER](https://github.com/ROCm/aiter) | `amd-aiter` | AMD-optimised kernels: FP8 quantization, hipBLASLt GEMM, CK attention (MHA) |
| [Composable Kernel (CK)](https://github.com/ROCm/composable_kernel) | *(bundled in aiter)* | High-performance GPU kernel primitives used by AITER |
| [MORI](https://github.com/ROCm/mori) | `mori` | Native RDMA + GPU communication: MORI-CCL (collective ops), MORI-EP (MoE dispatch) |


## Examples

### LLaMA2 SFT with Megatron-LM

Full fine-tuning or LoRA on LLaMA2 (7B / 13B / 70B) with FP8 attention, packed sequences, and early stopping.

```bash
# 1. Prepare data and model checkpoint
bash examples/llama2/scripts/prepare_data_and_model.sh

# 2. Run training — Megatron backend (default)
BACKEND=megatron bash examples/llama2/run_finetune.sh

# 2. Or: FSDP backend (no Megatron dependency)
BACKEND=fsdp bash examples/llama2/run_finetune.sh
```

The training script (`examples/llama2/finetune_llama2.py`) selects the backend via `--backend megatron|fsdp`:

| Feature | CLI Flag |
|---------|----------|
| Attention backend | `--tl-attn-backend {aiter_csrc,aiter_triton,aiter_triton_fp8,aiter_csrc_fp8}` |
| FP8 quantised training | `--linear-fp8 --fp8-format e4m3` |
| MXFP8 block sizes | `--mxfp8-block-m-fwd 128 ...` (6 independent dims) |
| LoRA | `--lora-rank 16 --lora-alpha 32` |
| LoRA A2A comm opt | `--lora-a2a` |
| Synthetic warmup | `--warmup-steps 5` |
| Early stopping | `--val-loss-target 1.5` |
| Context Parallelism | `--context-parallel-size 2` |

See `run_finetune.sh` for the full list of environment variables and defaults.

### LLaMA 3.1 Pretraining

Pretraining LLaMA 3.1 (8B) with FP8 hybrid training and MXFP8 attention, aligned with MLPerf LLM pretraining config.

```bash
# 1. Prepare data and model checkpoint
bash examples/llama31/scripts/prepare_data_and_model.sh

# 2. Run training — Megatron backend (default)
BACKEND=megatron bash examples/llama31/run_pretrain.sh

# 2. Or: FSDP backend (no Megatron dependency)
BACKEND=fsdp bash examples/llama31/run_pretrain.sh
```

The entry point (`examples/llama31/pretrain_llama31.py`) selects the backend via `--backend megatron|fsdp`:

| Feature | CLI Flag | Default |
|---------|----------|---------|
| Model size | `SIZE=8b` (env var) | 8b |
| MXFP8 attention | `--tl-fp8-quant-type mxfp8` | mxfp8 |
| FP8 training | `--linear-fp8` | enabled |
| Amax algorithm | `--linear-fp8-amax-algo most_recent` | most_recent |
| Amax history | `--linear-fp8-amax-history 4` | 4 |
| Learning rate | `MAX_LR=8e-4` (env var) | 8e-4 |
| Cosine LR warmup | `LR_WARMUP_STEPS=128` | 128 |
| GQA (8 KV heads) | auto from SIZE | 32 heads / 8 KV groups |

See `run_pretrain.sh` for all environment variables.

## Testing

```bash
# Run all tests
pytest tests/ -v

# FP8 attention correctness: Lumen vs TransformerEngine AMD
pytest tests/module/test_fp8_attention.py -v -s
```

## Software Stack

```
┌──────────────────────────────────────────────────────┐
│  Training Script (train.py)                          │
├──────────────────────────────────────────────────────┤
│  Lumen                                               │
│    quant.enable(model, config=QuantConfig)           │
│    ├─ ScalingManager                                 │
│    │   ├─ Per-tensor / block / MXFP8 scaling         │
│    │   ├─ Amax history & delayed scaling             │
│    │   └─ FP8 param lifecycle (quant/dequant/sync)   │
│    ├─ DistributedManager                             │
│    │   ├─ Param & grad buffer (FP8/BF16 contiguous)  │
│    │   ├─ Distributed optimizer (shard + all-gather) │
│    │   ├─ FP8 param-gather (uint8 comm, 2× save)    │
│    │   └─ Overlap: AG ↔ fwd, RS ↔ bwd               │
│    ├─ QuantizedLinearFunction (autograd)             │
│    ├─ Attention Kernels                              │
│    │   ├─ Triton FP8 blockwise                      │
│    │   ├─ Triton MXFP8 (gfx950)                    │
│    │   └─ AITER CK attention                        │
│    └─ Models                                        │
│        ├─ LLaMA2 SFT (LoRA, FP8, CP)               │
│        └─ LLaMA 3.1 Pretrain (FP8, MXFP8)          │
├──────────────────────────┬───────────────────────────┤
│  AITER ← quant kernels  │  MORI ← comm backend      │
│  CK flash-attention      │  MORI-CCL (AG, RS, AR)    │
│  hipBLASLt FP8 GEMM     │  MORI-EP (MoE dispatch)   │
│  Triton FP8 / MXFP8     │  Device-side RDMA / FP8   │
├──────────────────────────┴───────────────────────────┤
│  PyTorch + ROCm + RCCL + Triton                      │
└──────────────────────────────────────────────────────┘
```

## Project Structure

```
Lumen/
├── lumen/                     # Main Python package
│   ├── core/                  #   FP8 dtype helpers, gradient quantization, device detection
│   ├── kernels/               #   Triton GPU kernels (FP8/MXFP8 flash attention)
│   ├── ops/                   #   Stateless ops API (attention, quantize, normalization)
│   ├── modules/               #   nn.Module wrappers (LumenAttention, LumenLinear, Megatron drop-in)
│   ├── quantize/              #   Quantization lifecycle (enable/disable, config, scaling manager)
│   └── models/                #   Training utilities & model definitions
│       ├── megatron.py        #     Shared Megatron stack (spec patching, FP8, LoRA)
│       ├── fsdp.py            #     Shared FSDP stack (FP8, LoRA, state mgmt)
│       ├── llama2/            #     LLaMA2 SFT (dataset, megatron/, fsdp/)
│       └── llama31/           #     LLaMA 3.1 Pretrain (dataset, megatron/, fsdp/)
├── third_party/               # Git submodules
│   └── aiter/                 #   AMD AITER — CK attention, FP8 quant, hipBLASLt kernels
├── examples/                  # End-to-end training examples (Dockerfile, launcher, scripts)
│   ├── llama2/                #   LLaMA2 SFT
│   └── llama31/               #   LLaMA 3.1 Pretrain
└── tests/                     # Test suite
```

## License

Apache License 2.0
