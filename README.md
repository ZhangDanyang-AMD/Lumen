# Lumen

A lightweight, AMD-native quantized training framework for large language models.

Lumen manages the **quantized training lifecycle** — the vertical path a low-precision tensor takes through forward, backward, optimizer, and communication. It supports **FP8 (E4M3/E5M2)**, **MXFP8**, and **FP4** formats with a unified `QuantConfig` interface. It also integrates high-performance **FP8/MXFP8 attention kernels** (Triton + CK backends) directly, eliminating the dependency on Transformer Engine for end-to-end finetuning. Lumen leverages **[MORI](https://github.com/ROCm/mori)** for distributed communication management, including collective operations (all-gather, reduce-scatter, all-reduce) and MoE expert dispatch.

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

<table>
  <tr>
    <th colspan="2" align="center">MODEL LIBRARY</th>
  </tr>
  <tr>
    <td colspan="2">
      <b>LLaMA2 SFT / LLaMA 3.1 Pretrain</b><br>
      Megatron-LM or FSDP backend<br>
      LoRA, early stopping, synthetic warmup
    </td>
  </tr>
  <tr>
    <th colspan="2" align="center">NON-INVASIVE QUANTIZATION &mdash; <code>quant.enable(model)</code></th>
  </tr>
  <tr>
    <td colspan="2">
      Patches <code>nn.Linear</code> in-place (no module swap)<br>
      FP8 E4M3 / E5M2 / MXFP8 / FP4 formats<br>
      <code>QuantConfig</code> &mdash; one object for all settings
    </td>
  </tr>
  <tr>
    <th>SCALING MANAGER</th>
    <th>DISTRIBUTED MANAGEMENT</th>
  </tr>
  <tr>
    <td valign="top">
      Per-tensor / block / MXFP8 scaling<br>
      Amax history with delayed scaling<br>
      AMD FNUZ auto-detect<br>
      FP8 param lifecycle: quantize / dequant / scale &amp; amax sync
    </td>
    <td valign="top">
      Param &amp; grad buffer (FP8/BF16 contiguous)<br>
      Distributed optimizer (shard + all-gather)<br>
      FP8 param-gather (uint8 comm, 2&times; BW saving)<br>
      Overlap: AG &harr; fwd, RS &harr; bwd
    </td>
  </tr>
  <tr>
    <th>ATTENTION KERNELS</th>
    <th>QUANTIZED LINEAR</th>
  </tr>
  <tr>
    <td valign="top">
      <code>aiter_csrc</code> &mdash; CK flash-attention<br>
      <code>aiter_triton</code> &mdash; Triton<br>
      <code>aiter_triton_fp8</code> &mdash; FP8 block / MXFP8<br>
      <code>aiter_csrc_fp8</code> &mdash; CK<br>
      Context Parallelism
    </td>
    <td valign="top">
      Fused: quant &rarr; GEMM &rarr; dequant (one op)<br>
      AITER hipBLASLt or Triton backend<br>
      <code>torch.compile</code> compatible
    </td>
  </tr>
  <tr>
    <th><a href="https://github.com/ROCm/aiter">AITER</a> <em>(kernel provider)</em></th>
    <th><a href="https://github.com/ROCm/mori">MORI</a> <em>(communication provider)</em></th>
  </tr>
  <tr>
    <td valign="top">
      CK asm kernels<br>
      hipBLASLt, Triton<br>
      <br>
      &uarr; serves: <b>QUANTIZED LINEAR + ATTENTION</b>
    </td>
    <td valign="top">
      MORI-CCL (AG, RS, AR)<br>
      MORI-EP (MoE dispatch)<br>
      Device-side RDMA / FP8<br>
      AINIC / CX-7 / Thor2<br>
      <br>
      &uarr; serves: <b>DISTRIBUTED MANAGEMENT</b>
    </td>
  </tr>
</table>

## Software Stack

<table>
  <tr>
    <th colspan="2" align="center">
      Training Script<br>
      <code>--backend megatron</code> &nbsp;|&nbsp; <code>--backend fsdp</code>
    </th>
  </tr>
  <tr>
    <td colspan="2" align="center">&darr;</td>
  </tr>
  <tr>
    <th>Megatron-LM Stack</th>
    <th>PyTorch FSDP Stack</th>
  </tr>
  <tr>
    <td valign="top">
      TP / PP / CP / VP / SP<br>
      Megatron <code>pretrain()</code><br>
      <code>LumenDotProductAttention</code> (replaces core_attn)<br>
      <code>LumenRMSNorm</code> (Triton-accelerated)<br>
      Megatron LoRA adapters
    </td>
    <td valign="top">
      FSDP sharding<br>
      HuggingFace Transformers<br>
      HuggingFace PEFT (LoRA)<br>
      Gradient checkpointing
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">&darr;</td>
  </tr>
  <tr>
    <th colspan="2" align="center">Lumen <em>(shared across both stacks)</em></th>
  </tr>
  <tr>
    <td colspan="2">
      <code>quant.enable(model)</code> &mdash; non-invasive FP8 patching<br>
      <code>LumenAttention</code> &mdash; FP8 / MXFP8 / BF16 attention<br>
      <code>ScalingManager</code> &mdash; per-layer amax / scale / quant<br>
      <code>DistributedManager</code> &mdash; param &amp; grad buffer, dist-opt<br>
      &nbsp;&nbsp;&bull; FP8 param-gather &mdash; uint8 all-gather, 2&times; BW saving<br>
      &nbsp;&nbsp;&bull; overlap AG &harr; fwd &mdash; bucket-wise async pipeline<br>
      &nbsp;&nbsp;&bull; overlap RS &harr; bwd &mdash; grad reduce with backward<br>
      <code>reset_fp8_state()</code> &mdash; post-warmup scale reset
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">&darr;</td>
  </tr>
  <tr>
    <th><a href="https://github.com/ROCm/aiter">AITER</a> <em>(kernel backend)</em></th>
    <th><a href="https://github.com/ROCm/mori">MORI</a> <em>(comm backend)</em></th>
  </tr>
  <tr>
    <td valign="top">
      CK flash-attention<br>
      hipBLASLt FP8 GEMM<br>
      Triton FP8 / MXFP8<br>
      Triton RMSNorm
    </td>
    <td valign="top">
      MORI-CCL &mdash; AG, RS, AR<br>
      MORI-EP &mdash; MoE dispatch<br>
      Device-side RDMA / FP8<br>
      AINIC / CX-7 / Thor2
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">&darr;</td>
  </tr>
  <tr>
    <td colspan="2" align="center"><b>PyTorch</b> + <b>ROCm</b> + <b>RCCL</b> + <b>Triton</b></td>
  </tr>
</table>

## Training Backends: Megatron vs FSDP

Every model in Lumen supports **two independent training stacks**, selected via `--backend megatron|fsdp`. See [`lumen/models/`](lumen/models/) for detailed documentation on both stacks.

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

See [`lumen/modules/`](lumen/modules/) for the full module API with usage examples.

### Functional API

See [`lumen/ops/`](lumen/ops/) for the stateless functional API with usage examples.

### Training Backends

See [`lumen/models/`](lumen/models/) for Megatron and FSDP stack documentation and usage examples.


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

| Example | Description | Docs |
|---------|-------------|------|
| **LLaMA2 SFT** | Fine-tuning / LoRA on LLaMA2 7B–70B with FP8 attention, packed sequences, early stopping | [`examples/llama2/`](examples/llama2/) |
| **LLaMA 3.1 Pretrain** | Pretraining LLaMA 3.1 8B with FP8 hybrid training and MXFP8 attention (MLPerf-aligned) | [`examples/llama31/`](examples/llama31/) |

## Testing

See [`tests/`](tests/) for test instructions.

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
