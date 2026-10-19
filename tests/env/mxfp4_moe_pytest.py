"""Explicit P0a fixture CLI: load with -p tests.env.mxfp4_moe_pytest."""

from types import SimpleNamespace

from lumen.models.mxfp4_moe_options import add_mxfp4_moe_arguments, validate_p0_selector
from lumen.utils.mxfp4_moe_ledger import Mxfp4StateError

_SELECTED = None
_OPTION_NAMES = (
    "lumen_moe_mxfp4",
    "lumen_moe_mxfp4_backend",
    "lumen_moe_mxfp4_strict",
    "lumen_moe_mxfp4_grouped_tuning_policy",
)


def pytest_addoption(parser):
    add_mxfp4_moe_arguments(parser, pytest=True)


def pytest_configure(config):
    global _SELECTED
    _SELECTED = SimpleNamespace(
        **{name: config.getoption(name) for name in _OPTION_NAMES}
    )
    if _SELECTED.lumen_moe_mxfp4:
        # Reject an unsupported P0 selector before any runtime fixture/PG setup.
        validate_p0_selector(_SELECTED)


def pytest_unconfigure(config):
    global _SELECTED
    _SELECTED = None


def fixture_options():
    if _SELECTED is None:
        raise Mxfp4StateError(
            "P0_CLI_PLUGIN_REQUIRED", "use -p tests.env.mxfp4_moe_pytest"
        )
    validate_p0_selector(_SELECTED)
    return SimpleNamespace(**vars(_SELECTED))
