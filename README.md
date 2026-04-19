# Lumen

A lightweight, AMD-native quantized training engine for large language models.

Lumen manages the **quantized training lifecycle** — the vertical path a low-precision tensor takes through forward, backward, optimizer, and communication.

- **Quantized Formats** — FP8 (E4M3 / E5M2), MXFP8, and FP4 (Not supported yet) with a unified `QuantConfig` interface
- **[Aiter Kernels](https://github.com/ROCm/aiter)** — high-performance quantized MHA, Linear, MLA, MoE kernels
- **[MORI](https://github.com/ROCm/mori)** — high-performance RDMA + GPU communication library for distributed training (MORI-CCL: all-gather, reduce-scatter, all-reduce; MORI-EP: MoE expert dispatch)

## Architecture

Lumen owns the quantized training lifecycle and delegates everything else (optimizer, data loading, distributed orchestration) to the training backend:

<div align="center">
<table>
  <tr>
    <th colspan="2" align="center">SFT / PRETRAIN / RL </th>
  </tr>
  <tr>
    <td colspan="2" align="center">
      Megatron-LM or FSDP backend<br>
      LoRA, early stopping, synthetic warmup
    </td>
  </tr>
  <tr>
    <th colspan="2" align="center">NON-INVASIVE QUANTIZATION &mdash; <code>quant.enable(model)</code></th>
  </tr>
  <tr>
    <td colspan="2" align="center">
      FP8 E4M3 / E5M2 / MXFP8 / FP4 (Not supported yet) formats<br>
      <code>QuantConfig</code> &mdash; one object for all settings
    </td>
  </tr>
  <tr>
    <th align="center">SCALING MANAGER</th>
    <th align="center">DISTRIBUTED MANAGEMENT</th>
  </tr>
  <tr>
    <td align="center" valign="top">
      Per-tensor / block / MXFP8 scaling<br>
      Amax history with delayed scaling<br>
      AMD FNUZ auto-detect<br>
      FP8 param lifecycle: quantize / dequant / scale &amp; amax sync
    </td>
    <td align="center" valign="top">
      Param &amp; grad buffer (FP8/BF16 contiguous)<br>
      Distributed optimizer (shard + all-gather)<br>
      FP8 param-gather (uint8 comm, 2&times; BW saving)<br>
      Overlap: AG &harr; fwd, RS &harr; bwd
    </td>
  </tr>
  <tr>
    <th align="center">ATTENTION KERNELS</th>
    <th align="center">QUANTIZED LINEAR</th>
  </tr>
  <tr>
    <td align="center" valign="top">
      <code>aiter_csrc</code> <br>
      <code>aiter_triton</code> <br>
      <code>aiter_triton_fp8</code> <br>
      <code>aiter_csrc_fp8</code> <br>
      Context Parallelism
    </td>
    <td align="center" valign="top">
      Fused: quant &rarr; GEMM &rarr; dequant (one op)<br>
      AITER C++, ASM or Triton backend<br>
      <code>torch.compile</code> compatible
    </td>
  </tr>
  <tr>
    <th align="center"><a href="https://github.com/ROCm/aiter">AITER</a> <em>(kernel provider)</em></th>
    <th align="center"><a href="https://github.com/ROCm/mori">MORI</a> <em>(communication provider)</em></th>
  </tr>
  <tr>
    <td align="center" valign="top">
      Asm kernels<br>
      hipBLASLt, Triton<br>
      <br>
      &uarr; serves: <b>QUANTIZED LINEAR + ATTENTION</b>
    </td>
    <td align="center" valign="top">
      MORI-CCL (AG, RS, AR)<br>
      MORI-EP (MoE dispatch)<br>
      Device-side RDMA / FP8<br>
      AINIC / CX-7 / Thor2<br>
      <br>
      &uarr; serves: <b>DISTRIBUTED MANAGEMENT</b>
    </td>
  </tr>
</table>
</div>

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


### User Install (recommended)

```bash
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
| [AITER](https://github.com/ROCm/aiter) | `amd-aiter` | AMD-optimised kernels: high-performance quantized MHA, Linear, MLA, MoE kernels |
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
│   └── aiter/                 #   AMD AITER — High-performance quantized MHA, Linear, MLA, MoE kernels
├── examples/                  # End-to-end training examples (Dockerfile, launcher, scripts)
│   ├── llama2/                #   LLaMA2 SFT
│   └── llama31/               #   LLaMA 3.1 Pretrain
└── tests/                     # Test suite
```

## License

Apache License 2.0
