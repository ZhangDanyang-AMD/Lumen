"""Independent raw-bit/Fraction/list-stage checks of the new Lumen reference."""

import importlib.util
import math
import struct
from fractions import Fraction
from pathlib import Path


def _f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _rn32(value):
    return _f32(_bits(value))


def _rne(value):
    choices = tuple(map(Fraction, (0, 0.5, 1, 1.5, 2, 3, 4, 6)))
    a = Fraction.from_float(abs(value))
    code = min(range(8), key=lambda index: (abs(a - choices[index]), index % 2, index))
    return code | (8 if math.copysign(1, value) < 0 else 0)


def _list_stages(values):
    current = [_rn32(value * 0.25) for value in values]
    stages = []
    for half in (1, 2, 4, 8):
        following = current[:]
        for start in range(0, 16, half * 2):
            for index in range(half):
                left, right = current[start + index], current[start + half + index]
                following[start + index] = _rn32(left + right)
                following[start + half + index] = _rn32(left - right)
        current = following
        stages.append(tuple(_bits(value) for value in current))
    return stages


def _recipe():
    path = Path(__file__).with_name("mxfp4_moe_recipe_reference.py")
    spec = importlib.util.spec_from_file_location("recipe_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_host_fraction_ties_and_list_stage_goldens():
    assert [_rne(value) for value in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5)] == [
        0,
        2,
        2,
        4,
        4,
        6,
        6,
    ]
    assert _rne(-0.25) == 8
    final = _list_stages([-0.0] * 16)[-1]
    assert final[0] == 0x80000000 and final[1:] == (0,) * 15
    impulse = [0.0] * 16
    impulse[1] = 4.0
    assert _list_stages(impulse)[-1] == tuple(
        _bits(1.0 if index % 2 == 0 else -1.0) for index in range(16)
    )


def test_reference_raw_boundaries_ties_and_max():
    import torch

    recipe = _recipe()
    # Expected selector scale bytes are frozen independent values, not another
    # invocation of the reference selector. Constant rows preserve each amax.
    cases = (
        (0, 0, 0),
        (0x80000000, 0, 8),
        (0x00010000, 0, 0),
        (0x00400000, 0, 2),
        (0x00800000, 0, 4),
        (0x01000000, 0, 6),
        (0x3FDFFFFF, 125, 7),
        (0x3FE00000, 126, 6),
        (0x3FE00001, 126, 6),
        (0x7F7F0000, 254, 4),
    )
    for raw, expected_scale, expected_code in cases:
        value = _f32(raw)
        tensor = torch.full((1, 32), value, dtype=torch.float32)
        codes, scale, decoded = recipe.quantize_reference(tensor, return_encoding=True)
        assert int(scale[0, 0]) == expected_scale
        assert int(codes[0, 0]) == expected_code
        expected = (0, 0.5, 1, 1.5, 2, 3, 4, 6)[expected_code & 7] * 2.0 ** (
            expected_scale - 127
        )
        if expected_code & 8:
            expected = -expected
        assert float(decoded[0, 0]) == expected
        assert bool(torch.signbit(decoded[0, 0])) == (expected_code == 8)
        if raw == 0x7F7F0000:
            assert math.isfinite(expected) and torch.isinf(decoded.float()[0, 0])
    ties = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5)
    # amax=6 forces unit scale; fillers do not alter the tie's independent oracle.
    tensor = torch.tensor([list(ties) + [-value for value in ties] + [6.0] * 18])
    codes, scale, _ = recipe.quantize_reference(tensor, return_encoding=True)
    assert int(scale[0, 0]) == 127
    assert codes[0, :14].tolist() == [
        _rne(value) for value in (*ties, *(-value for value in ties))
    ]
    for value in (float("nan"), float("inf"), -float("inf")):
        try:
            recipe.quantize_reference(torch.full((1, 32), value))
        except ValueError:
            pass
        else:
            raise AssertionError("finite recipe accepted nonfinite source")


def test_reference_h16_stages_direction_pad_and_signedzero():
    import torch

    recipe = _recipe()
    for values in (
        [-0.0] * 16,
        [0.0, 4.0] + [0.0] * 14,
        [float(index - 8) for index in range(16)],
        [_f32(0x00010000)] * 16,
    ):
        observed = []
        source = torch.tensor(values, dtype=torch.bfloat16).reshape(16, 1)
        result = recipe.h16_pad_transpose(source, stage_observer=observed.append)
        expected = _list_stages(source.float().flatten().tolist())
        for stage, bits in zip(observed, expected):
            assert tuple(_bits(value) for value in stage[0, :, 0].tolist()) == bits
        assert result.shape == (1, 32)
        assert tuple(_bits(value) for value in result[0, :16].tolist()) == expected[-1]
        assert torch.count_nonzero(result[0, 16:]) == 0
    source = torch.tensor([[4.0]], dtype=torch.bfloat16)
    result = recipe.h16_pad_transpose(source)
    assert torch.equal(result[0, :16], torch.ones(16))
    assert torch.count_nonzero(result[0, 16:]) == 0
