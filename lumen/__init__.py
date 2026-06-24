from lumen.config import LumenConfig
from lumen.core.float8 import float8_e4m3, float8_e5m2

__version__ = "0.4.0"


def __getattr__(name):
    if name == "quantize":
        import lumen.quantize as quantize

        return quantize
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
