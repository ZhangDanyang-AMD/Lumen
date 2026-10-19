"""Independent scalar Fraction SR oracle; no production or GPU imports."""

import math
from fractions import Fraction

_MAGNITUDES = tuple(map(Fraction, (0, 0.5, 1, 1.5, 2, 3, 4, 6)))


def philox_word(seed, draw):
    """Published Philox4x32-10 recurrence, direct element lane mapping."""
    if not 0 <= seed < 2**64 or not 0 <= draw < 2**64:
        raise ValueError("unsigned64 seed/draw required")
    mask = 2**32 - 1
    counter, lane = divmod(draw, 4)
    c = (counter & mask, counter >> 32, 0, 0)
    k = (seed & mask, seed >> 32)
    for _ in range(10):
        p0, p2 = 0xD2511F53 * c[0], 0xCD9E8D57 * c[2]
        c = ((p2 >> 32) ^ c[1] ^ k[0], p2 & mask, (p0 >> 32) ^ c[3] ^ k[1], p0 & mask)
        k = ((k[0] + 0x9E3779B9) & mask, (k[1] + 0xBB67AE85) & mask)
    return c[lane]


def sr_code(normalized_fp32, word):
    """Exact dyadic threshold; caller supplies already-FP32 normalized value."""
    if not math.isfinite(normalized_fp32) or not 0 <= word < 2**32:
        raise ValueError("finite operand and unsigned32 word required")
    sign = 8 if math.copysign(1, normalized_fp32) < 0 else 0
    a = min(Fraction.from_float(abs(float(normalized_fp32))), Fraction(6))
    for code, value in enumerate(_MAGNITUDES):
        if a == value:
            return code | sign
        if a < value:
            lo = _MAGNITUDES[code - 1]
            probability = (a - lo) / (value - lo)
            scaled = probability * 2**32
            threshold = -(-scaled.numerator // scaled.denominator)
            return (code if word < threshold else code - 1) | sign
    raise AssertionError("clamped magnitude outside E2M1")
