# Lazy subpackage re-exports (PEP 562).
# Avoids eagerly importing GPU kernel code when only a single subpackage
# (e.g. ops.quantize) is needed.

# Order matters twice over, because __getattr__ below imports each entry in turn
# until one has the requested name:
#
#   * An entry that raises on import blocks everything after it. "gemm" does: its
#     __init__ imports .epilogue and .fp8_output, neither of which exists, so it
#     raises ModuleNotFoundError rather than being skipped. Keep new subpackages
#     ahead of it.
#   * Every entry before the one that answers gets imported as a side effect. So
#     a subpackage with light dependencies belongs early, or reaching it drags in
#     the heavy ones. "conv" is first for that reason: it needs only flydsl, and
#     resolving lumen.ops.conv3d should not have to import AITER attention
#     kernels on the way.
_SUBMODULES = (
    "conv",
    "attention",
    "normalization",
    "quantize",
    "gemm",
    "sdma",
    "mlp",
    "fused_residual_norm",
)


def __getattr__(name):
    import importlib

    for sub in _SUBMODULES:
        mod = importlib.import_module(f".{sub}", __name__)
        if hasattr(mod, name):
            globals()[name] = getattr(mod, name)
            return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
