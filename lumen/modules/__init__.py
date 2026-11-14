# Lazy re-exports.
#
# Importing a submodule (for example ``lumen.modules.fused_mlp``) must not
# pull Megatron. The parallel-linear modules import it, and the ATOM Hugging
# Face path treats that dependency as optional. Names below still resolve
# through ``from lumen.modules import ...``.

from __future__ import annotations

_EXPORTS = {
    "LumenLayerNorm": ("lumen.ops.normalization", "LumenLayerNorm"),
    "LumenRMSNorm": ("lumen.ops.normalization", "LumenRMSNorm"),
    "LumenAttention": (".attention", "LumenAttention"),
    "LumenDotProductAttention": (".attention_megatron", "LumenDotProductAttention"),
    "LumenDotProductAttentionMLA": (".attention_mla", "LumenDotProductAttentionMLA"),
    "lumen_parallel_cross_entropy": (".cross_entropy", "lumen_parallel_cross_entropy"),
    "LumenFusedMLP": (".fused_mlp", "LumenFusedMLP"),
    "LumenGatedMLP": (".fused_mlp", "LumenGatedMLP"),
    "LumenColumnParallelGroupedLinear": (".grouped_linear", "LumenColumnParallelGroupedLinear"),
    "LumenGroupedLinear": (".grouped_linear", "LumenGroupedLinear"),
    "LumenRowParallelGroupedLinear": (".grouped_linear", "LumenRowParallelGroupedLinear"),
    "LumenLayerNormLinear": (".layernorm_linear", "LumenLayerNormLinear"),
    "LumenColumnParallelLinear": (".parallel_linear", "LumenColumnParallelLinear"),
    "LumenRowParallelLinear": (".parallel_linear", "LumenRowParallelLinear"),
    "LumenLinear": (".quantize", "LumenLinear"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attr = target
    module = importlib.import_module(module_name, __name__ if module_name.startswith(".") else None)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
