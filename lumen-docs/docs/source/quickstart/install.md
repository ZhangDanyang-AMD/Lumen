# Installation

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | >= 3.10 |
| PyTorch | >= 2.4 |
| ROCm | >= 6.2 |
| Hardware | AMD Instinct MI300X (CDNA3) |

## User Install (recommended)

```bash
pip install lumen[all]
```

## Developer Install

```bash
git clone git@github.com:ZhangDanyang-AMD/Lumen.git
cd Lumen
pip install -e ".[dev]"
```

## Third-party Libraries

Lumen delegates kernel execution and communication to two AMD-native libraries:

| Library | PyPI Package | Purpose |
|---------|-------------|---------|
| [AITER](https://github.com/ROCm/aiter) | `amd-aiter` | GPU kernels: attention, GEMM, normalization, RoPE, MoE, fused MLP, cross-entropy, quantization (ASM / CK / Triton backends) |
| [MORI](https://github.com/ROCm/mori) | `mori` | RDMA + GPU communication: MORI-CCL (collective ops), MORI-EP (MoE expert dispatch) |

Both are installed automatically with `pip install lumen[all]`.

## Verify Installation

```python
import lumen
print(lumen.__version__)

import lumen.quantize as quant
from lumen.quantize import QuantConfig, QuantFormat, ScalingType
print("Lumen ready.")
```
