# Quick Start

This page shows how to enable Lumen's quantized training on an existing model with minimal code changes.

## Non-invasive Quantized Training

Lumen's core workflow is a single call — `quant.enable(model)` — that patches a model's linear, attention, and MLP layers for FP8 execution without modifying model code or checkpoint format.

```python
import lumen.quantize as quant
from lumen.quantize import AmaxAlgo, QuantConfig, QuantFormat, ScalingType

# 1. Build or load your model as usual
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. Configure quantization
config = QuantConfig(
    format=QuantFormat.FP8_E4M3,       # FP8_E5M2, HYBRID, MXFP8
    scaling=ScalingType.DELAYED,        # DYNAMIC, BLOCKWISE
    amax_algo=AmaxAlgo.MAX,             # or MOST_RECENT
    history_len=16,
    quantize_activation=True,           # False → weight-only quantization
    quantize_grad="fp8",                # None, "fp8", "mxfp8", "fp4"
)
quant.enable(model, config=config)

# 3. Train as usual — Lumen handles quantized dispatch
output = model(input_ids)
loss = output.loss
loss.backward()
optimizer.step()
```

### String Shorthand

For quick experiments, `quant.enable` also accepts string arguments:

```python
quant.enable(model, format="fp8_e4m3", scaling="delayed")
```

## FP8 Attention (Module API)

For more fine-grained control, use the module API to replace specific layers:

```python
from lumen.modules import LumenAttention, LumenDotProductAttention
```

See {doc}`/api/modules` for the full list of drop-in module replacements.

## Functional API

For custom kernels and operator-level control:

```python
from lumen.ops.attention import fp8_flash_attention_fwd
from lumen.ops.quantize import fp8_quantize, fp8_dequantize
```

See {doc}`/api/ops` for the stateless functional API backed by AITER kernels.

## Training Backends

Lumen integrates with two distributed training backends:

| Backend | Use case | Documentation |
|---------|----------|---------------|
| **FSDP / FSDP2** | HuggingFace ecosystem, medium scale | {doc}`/advance/distributed` |
| **Megatron-LM** | Large-scale pre-training, TP/PP/EP | {doc}`/advance/distributed` |

See {doc}`/examples/llama2_sft` and {doc}`/examples/llama31_pretrain` for end-to-end training examples.
