import importlib as _importlib

from lumen.ops.normalization import LumenLayerNorm, LumenRMSNorm

from .attention import LumenAttention
from .quantize import LumenLinear

# Megatron-dependent modules are loaded only when Megatron is installed.
# This lets FSDP/HF-only training import lumen.modules (e.g. ep_moe)
# without hitting a hard Megatron dependency.
_HAS_MEGATRON = _importlib.util.find_spec("megatron") is not None

if _HAS_MEGATRON:
    from .attention_megatron import LumenDotProductAttention
    from .attention_mla import LumenDotProductAttentionMLA
    from .cross_entropy import lumen_parallel_cross_entropy
    from .fused_mlp import LumenFusedMLP, LumenGatedMLP
    from .grouped_linear import (
        LumenColumnParallelGroupedLinear,
        LumenGroupedLinear,
        LumenRowParallelGroupedLinear,
    )
    from .layernorm_linear import LumenLayerNormLinear
    from .parallel_linear import LumenColumnParallelLinear, LumenRowParallelLinear
