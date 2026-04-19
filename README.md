# Transformer Light

A lightweight, AMD-native quantized training engine for large language models.

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

# Or use string shorthand
quant.enable(model, format="fp8_e4m3", scaling="delayed")

# Training loop is unchanged
output = model(input)       # engine handles quantized dispatch
loss.backward()             # engine handles quantized gradients
optimizer.step()            # stays FP32 (or quantized with opt-in)

# Communication is automatic
# - intra-node: RCCL with quantized tensors
# - inter-node: mori RDMA with quantized tensors
```

### FP8 Attention (direct use)

```python
from transformer_light.modules import TransformerLightAttention

attn = TransformerLightAttention(
    causal=True,
    backend_type="triton",
    quant_type="mxfp8",       # or "fp8_blockwise"
    quant_block_size=32,
)

output = attn(q, k, v)
```

### Functional API

```python
from transformer_light.ops.attention import attention, attention_fp8_quant

# Standard attention (CK or Triton backend)
output = attention(q, k, v, causal=True, backend_type="triton")

# FP8 quantized attention
output = attention_fp8_quant(q, k, v, causal=True, quant_type="mxfp8")
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

```bash
# Clone with third-party dependencies
git clone --recursive git@github.com:ZhangDanyang-AMD/Transformer_Light.git
cd Transformer_Light

# Or, if already cloned, initialize submodules
git submodule update --init --recursive

# Install Transformer Light
pip install -e .

# (Optional) Build AITER from bundled source
pip install -e 3rdparty/aiter
```

**Requirements**: PyTorch 2.x, ROCm, Triton.

### Third-party Libraries

Transformer Light bundles the following as git submodules under `3rdparty/`:

| Library | Path | Purpose |
|---------|------|---------|
| [AITER](https://github.com/ROCm/aiter) | `3rdparty/aiter/` | AMD-optimised kernels: FP8 quantization, hipBLASLt GEMM, CK attention (MHA) |
| [Composable Kernel (CK)](https://github.com/ROCm/composable_kernel) | `3rdparty/aiter/3rdparty/composable_kernel/` | High-performance GPU kernel primitives used by AITER |

AITER is **optional** — if not installed, Transformer Light falls back to Triton-only backends. When the CK attention backend (`backend_type="ck"`) is needed, AITER must be built with CK support.

## Examples

### LLaMA 2 70B LoRA Finetune

```bash
# 1. Prepare data and model
bash examples/llama2_finetune/scripts/prepare_data_and_model.sh

# 2. Run training (8 GPUs)
cd examples/llama2_finetune
bash run_and_time.sh
```

## Software Stack

```
┌─────────────────────────────────────────────┐
│  Training Script (train.py)                 │
├─────────────────────────────────────────────┤
│  Transformer Light                          │
│    quant.enable(model, config=QuantConfig)  │
│    ├─ ScalingManager                        │
│    ├─ QuantLinear (autograd)                │
│    ├─ QuantAllGather (communication)        │
│    └─ Attention Kernels                     │
│       ├─ Triton FP8 blockwise              │
│       ├─ Triton MXFP8 (gfx950)            │
│       └─ CK backend (via AITER)           │
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
├── 3rdparty/                      # Third-party dependencies (git submodules)
│   └── aiter/                     # AMD AITER — FP8 kernels, hipBLASLt, CK attention
│       └── 3rdparty/
│           └── composable_kernel/ # ROCm Composable Kernel (CK)
├── transformer_light/             # Main Python package
│   ├── quantize/                  # Quantization lifecycle management
│   │   ├── config.py              # QuantConfig, QuantFormat, ScalingType
│   │   ├── scaling_manager.py     # Per-tensor scale tracking (ScalingManager)
│   │   ├── autograd.py            # Quantized linear autograd (QuantLinear)
│   │   ├── communication.py       # Quantized all-gather / RDMA (QuantAllGather)
│   │   └── ops.py                 # Quantization kernel dispatch
│   ├── triton/                    # Triton GPU kernels
│   │   ├── attention/
│   │   │   ├── attention_kernel.py        # FP8 blockwise flash attention
│   │   │   └── mxfp8_attention_kernel.py  # MXFP8 attention (gfx950)
│   │   └── quantize/
│   │       ├── quant_mxfp8.py     # MXFP8 quantization kernel
│   │       └── quant_blockwise.py # Blockwise FP8 quantization
│   ├── kernels/                   # Kernel dispatch layer
│   │   └── attention/
│   │       ├── attention_triton_impl.py   # Triton attention dispatch
│   │       └── attention_csrc_impl.py     # CK/AITER attention dispatch
│   ├── ops/                       # High-level ops API
│   │   └── attention/
│   │       ├── attention.py       # attention() + attention_fp8_quant()
│   │       ├── attention_utils.py # Scaling utilities
│   │       └── attention_cp_dispatcher.py  # Context parallelism
│   ├── modules/                   # nn.Module wrappers
│   │   └── attention.py           # TransformerLightAttention
│   └── core/                      # Core utilities
│       ├── float8.py              # FP8 dtype definitions
│       └── utils.py               # Device capability detection
├── examples/                      # Training examples
│   └── llama2_finetune/           # LLaMA 2 70B LoRA finetune
├── setup.py
└── README.md
```

## License

Apache License 2.0
