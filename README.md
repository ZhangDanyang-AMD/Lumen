# Transformer Light

A lightweight, AMD-native quantized training framework for large language models.

Transformer Light manages the **quantized training lifecycle** — the vertical path a low-precision tensor takes through forward, backward, optimizer, and communication. It supports **FP8 (E4M3/E5M2)**, **MXFP8**, and **FP4** formats with a unified `QuantConfig` interface. It also integrates high-performance **FP8/MXFP8 attention kernels** (Triton + CK backends) directly, eliminating the dependency on Transformer Engine for end-to-end finetuning.

## Why Transformer Light?

| | TransformerEngine | Transformer Light |
|---|---|---|
| Codebase | ~200K lines C++/CUDA/Python | Lightweight Python + Triton |
| Scope | Monolith (attention, norms, GEMM, FP8, comm) | Quantized lifecycle + attention |
| AMD support | Hipified fork | AMD-native (AITER + Triton kernels) |
| Install time | Hours (C++ compilation) | Seconds (pure Python + AITER) |
| Integration | Requires module replacement | Non-invasive `quant.enable(model)` |

## Architecture

Transformer Light owns the following and delegates everything else:

```
┌──────────────────────────────────────────────────┐
│  1. SCALING MANAGER                              │
│     - Per-tensor scale tracking                  │
│     - Amax history (delayed scaling)             │
│     - Dynamic / block / MXFP8 scaling policy     │
│     - AMD format handling (FNUZ)                 │
├──────────────────────────────────────────────────┤
│  2. QUANTIZED AUTOGRAD WRAPPERS                  │
│     - Linear: quant fwd + quant bwd (weight grad)│
│     - Fused: quant → GEMM → dequant as one op   │
│     - Uses AITER kernels underneath              │
│     - torch.compile compatible                   │
├──────────────────────────────────────────────────┤
│  3. QUANTIZED COMMUNICATION                      │
│     - Quantized all-gather / reduce-scatter      │
│     - mori RDMA path for inter-node              │
│     - RCCL path for intra-node                   │
│     - Quantize-on-send, dequantize-on-receive    │
├──────────────────────────────────────────────────┤
│  4. FP8/MXFP8 ATTENTION KERNELS                  │
│     - Triton Flash Attention v2 (FP8 blockwise)  │
│     - Triton MXFP8 Attention (gfx950)            │
│     - CK backend via AITER                       │
│     - Context parallelism (All-to-All)           │
│     - TransformerLightAttention module            │
├──────────────────────────────────────────────────┤
│  5. MODEL LIBRARY                                │
│     - LLaMA2 SFT (Megatron / FSDP)              │
│     - LLaMA 3.1 Pretrain (Megatron / FSDP)      │
│     - LoRA / PEFT with A2A comm optimisation     │
│     - FP8 quantised training (non-invasive)      │
│     - MXFP8 attention (MLPerf config)            │
│     - Synthetic warmup + early stopping          │
└──────────────────────────────────────────────────┘
```

## Quick Start

### Quantized Training (non-invasive patching)

```python
import transformer_light.quantize as quant
from transformer_light.quantize import QuantConfig, QuantFormat, ScalingType

# Configure quantization
config = QuantConfig(format=QuantFormat.FP8_E4M3,
                     scaling=ScalingType.DELAYED)

# Non-invasive: patch existing model, no module replacement
quant.enable(model, config=config)

# Training loop is unchanged
output = model(input)       # framework handles quantized dispatch
loss.backward()             # framework handles quantized gradients
optimizer.step()            # stays FP32 (or quantized with opt-in)
```

### FP8 Attention (module API)

```python
from transformer_light.pytorch.modules import TransformerLightAttention

attn = TransformerLightAttention(
    causal=True,
    backend_type="triton",
    quant_type="mxfp8",       # or "fp8_blockwise", or None for BF16
    quant_block_size=32,
)

output = attn(q, k, v)       # q, k, v: [B, S, H, D]
```

### Functional API

```python
from transformer_light.pytorch.ops.attention import attention, attention_fp8_quant

# Standard attention (CK or Triton backend)
output = attention(q, k, v, causal=True, backend_type="triton")

# FP8 quantized attention
output = attention_fp8_quant(q, k, v, causal=True, quant_type="mxfp8")
```

### LLaMA2 SFT (library API)

```python
from transformer_light.models.llama2 import (
    tl_gpt_builder,
    apply_lora,
    apply_fp8_training,
    forward_step,
    add_finetune_args,
    train_valid_test_datasets_provider,
)
```

## What It Does NOT Do

| Feature | Why Excluded | Use Instead |
|---------|-------------|-------------|
| Fused norms | Liger Kernel has AMD-tuned Triton versions | Liger Kernel |
| Fused activations | Liger Kernel (SwiGLU, GeLU) | Liger Kernel |
| RL losses | Liger Kernel (GRPO, DPO, 80% mem savings) | Liger Kernel |
| MoE dispatch | AITER has 3x faster fused MoE | AITER |
| Distributed orchestration | FSDP2/DeepSpeed handle this | FSDP2/DeepSpeed |
| Model definitions | HuggingFace Transformers | HuggingFace |

## Installation

**Requirements**: PyTorch 2.x, ROCm, Triton.

### User Install (recommended)

```bash
# Core (Triton-only attention backends)
pip install transformer_light

# With AITER CK attention backend
pip install transformer_light[aiter]

# All optional dependencies
pip install transformer_light[all]
```

### Developer Install

```bash
git clone git@github.com:ZhangDanyang-AMD/Transformer_Light.git
cd Transformer_Light

# Editable install with dev dependencies
pip install -e ".[dev]"
```

### Third-party Libraries

| Library | PyPI Package | Purpose |
|---------|-------------|---------|
| [AITER](https://github.com/ROCm/aiter) | `amd-aiter` | AMD-optimised kernels: FP8 quantization, hipBLASLt GEMM, CK attention (MHA) |
| [Composable Kernel (CK)](https://github.com/ROCm/composable_kernel) | *(bundled in aiter)* | High-performance GPU kernel primitives used by AITER |

AITER is **optional** — if not installed, Transformer Light falls back to Triton-only backends.

## Examples

### LLaMA2 SFT with Megatron-LM-AMD

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
| Attention backend | `--tl-attn-backend {aiter,triton,triton_fp8}` |
| FP8 quantised training | `--fp8-training --fp8-format fp8_e4m3` |
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
| FP8 training | `--fp8-training` | enabled |
| Amax algorithm | `--fp8-amax-algo most_recent` | most_recent |
| Amax history | `--fp8-amax-history 4` | 4 |
| Learning rate | `MAX_LR=8e-4` (env var) | 8e-4 |
| Cosine LR warmup | `LR_WARMUP_STEPS=128` | 128 |
| GQA (8 KV heads) | auto from SIZE | 32 heads / 8 KV groups |

See `run_pretrain.sh` for all environment variables.

## Testing

```bash
# Run all tests
pytest tests/ -v

# FP8 attention correctness: TransformerLight vs TransformerEngine AMD
pytest tests/module/test_fp8_attention.py -v -s
```

## Software Stack

```
┌─────────────────────────────────────────────┐
│  Training Script (train.py)                 │
├─────────────────────────────────────────────┤
│  Transformer Light                          │
│    quant.enable(model, config=QuantConfig)  │
│    ├─ ScalingManager (per-layer amax)       │
│    ├─ QuantizedLinearFunction (autograd)    │
│    ├─ Attention Kernels                     │
│    │   ├─ Triton FP8 blockwise             │
│    │   ├─ Triton MXFP8 (gfx950)           │
│    │   └─ CK backend (via AITER)           │
│    └─ Models                               │
│        ├─ LLaMA2 SFT (LoRA, FP8, CP)      │
│        └─ LLaMA 3.1 Pretrain (FP8, MXFP8) │
├─────────────────────────────────────────────┤
│  AITER          ← quant kernels + hipBLASLt │
│  Liger Kernel   ← fused ops + RL losses     │
│  mori           ← device-side RDMA          │
├─────────────────────────────────────────────┤
│  PyTorch + ROCm + RCCL + Triton             │
└─────────────────────────────────────────────┘
```

## Project Structure

```
Transformer_Light/
├── transformer_light/                  # Main Python package
│   ├── quantize/                       # Quantization lifecycle management
│   │   ├── config.py                   # QuantConfig, QuantFormat, ScalingType
│   │   └── scaling_manager.py          # Per-layer scale tracking + amax history
│   ├── triton/                         # Triton GPU kernels
│   │   ├── attention/                  # FP8 blockwise & MXFP8 flash attention kernels
│   │   └── quantize/                   # FP8 / MXFP8 quantization kernels
│   ├── pytorch/                        # PyTorch integration layer
│   │   ├── core/                       # FP8 dtypes & device capability detection
│   │   ├── kernels/                    # Kernel dispatch (Triton + AITER)
│   │   ├── ops/                        # High-level ops API
│   │   │   ├── attention/              # attention(), attention_fp8_quant()
│   │   │   │   ├── attention.py        # Functional API + autograd.Function
│   │   │   │   ├── attention_megatron.py  # Megatron DotProductAttention replacement
│   │   │   │   └── attention_with_cp_a2a.py  # Context Parallelism (All-to-All)
│   │   └── quantize/                  # Quantization ops
│   │       ├── ops.py                 # Pure quant/dequant functions
│   │       └── linear.py             # QuantizedLinearFunction (autograd)
│   │   └── modules/                    # nn.Module wrappers
│   │       ├── attention.py            # TransformerLightAttention
│   │       └── quantize.py            # TransformerLightLinear
│   └── models/                         # Reusable model definitions
│       ├── utils.py                    # Common helpers (peek_backend, download, sha256)
│       ├── perf_env.sh                 # AMD MI GPU perf tuning env vars (sourceable)
│       ├── llama2/            # LLaMA2 SFT
│       │   ├── dataset.py              # LLaMA2SFTDataset (shared, no framework dep)
│       │   ├── megatron/sft.py         # Megatron-LM-AMD backend
│       │   └── fsdp/sft.py            # PyTorch FSDP backend
│       └── llama31/           # LLaMA 3.1 Pretraining
│           ├── dataset.py              # PretrainTextDataset (shared)
│           ├── megatron/pretrain.py    # Megatron-LM-AMD backend
│           └── fsdp/pretrain.py       # PyTorch FSDP backend
├── examples/                           # Training examples
│   ├── llama2/                # LLaMA2 SFT (Megatron or FSDP)
│   │   ├── finetune_llama2.py          # Unified entry point (--backend switch)
│   │   ├── run_finetune.sh             # Unified launcher (BACKEND env var)
│   │   └── scripts/                    # Data/model download & conversion
│   └── llama31/               # LLaMA 3.1 Pretrain (Megatron or FSDP)
│       ├── pretrain_llama31.py         # Unified entry point (--backend switch)
│       ├── run_pretrain.sh             # Unified launcher (BACKEND env var)
│       └── scripts/                    # Data/model download & conversion
├── tests/                              # Test suite
│   └── module/                         # Module-level tests
│       └── test_fp8_attention.py       # TL vs TE FP8 attention correctness
├── setup.py
└── README.md
```

## License

Apache License 2.0
