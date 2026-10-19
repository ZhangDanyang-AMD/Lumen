"""Host oracle goldens only; not public-wrapper or adapter numerical evidence."""

import importlib.util
from pathlib import Path


def _reference():
    path = Path(__file__).with_name("mxfp4_moe_sr_reference.py")
    spec = importlib.util.spec_from_file_location("independent_sr_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_published_philox_zero_vector():
    ref = _reference()
    assert tuple(ref.philox_word(0, lane) for lane in range(4)) == (
        0x6627E8D5,
        0xE169C58D,
        0xBC57AC4C,
        0x9B00DBD8,
    )


def test_fraction_threshold_boundary_and_signed_zero():
    ref = _reference()
    for lo, hi, low_code in (
        (0, 0.5, 0),
        (0.5, 1, 1),
        (1, 1.5, 2),
        (1.5, 2, 3),
        (2, 3, 4),
        (3, 4, 5),
        (4, 6, 6),
    ):
        midpoint = (lo + hi) / 2
        assert ref.sr_code(midpoint, 2**31 - 1) == low_code + 1
        assert ref.sr_code(midpoint, 2**31) == low_code
        assert ref.sr_code(-midpoint, 2**31) == low_code | 8
    assert ref.sr_code(-0.0, 0) == 8
    assert ref.sr_code(7.0, 2**32 - 1) == 7
    assert ref.sr_code(2**-149, 0) == 1
    assert ref.sr_code(2**-149, 1) == 0
