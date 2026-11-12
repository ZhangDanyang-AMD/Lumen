"""MXFP4 public AITER adapters with explicit legacy capability fallbacks."""

import functools
import logging
import random

import torch

from lumen.ops.dispatch import Backend, _probe_aiter_mxfp4_api, try_backends

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def _warn_legacy(name, reason):
    logger.warning("%s: using legacy Lumen kernel (%s)", name, reason)


def _legacy(name, args, kwargs, reason):
    from lumen.ops.quantize import ops

    _warn_legacy(name, reason)
    return getattr(ops, "_legacy_" + name)(*args, **kwargs)


def _run(api, name, run, args, kwargs):
    def fallback():
        return _legacy(
            name, args, kwargs, "upstream API unavailable or rejected this call"
        )

    if not _probe_aiter_mxfp4_api(api):
        return fallback()
    # Shape/layout-specific keys prevent a degraded winner leaking into other calls.
    key = (
        (
            tuple(getattr(arg, "shape", ())),
            str(getattr(arg, "dtype", None)),
            str(getattr(arg, "device", None)),
        )
        for arg in args
    )
    contract = tuple(
        (k, v) for k, v in kwargs.items() if k not in ("philox_seed", "philox_offset")
    )
    op_name = f"{name}:{tuple(key)}:{contract}"
    return try_backends(
        [
            (Backend.TRITON, run, "aiter_public"),
            # Legacy kernels retain compatibility with unsupported installations.
            (Backend.TRITON, fallback, "legacy_lumen"),
        ],
        op_name=op_name,
        slow_labels=("legacy_lumen",),
    )


def _rng(use_sr, seed, offset):
    if not use_sr:
        return None, 0
    if use_sr:
        seed = random.randint(0, 2**31 - 2) if seed is None else seed
        offset = random.randint(0, 2**31 - 2) if offset is None else offset
    return seed, 0 if offset is None else offset


def convert_to_mxfp4(
    data_hp,
    block_size=32,
    axis=-1,
    use_sr=False,
    philox_seed=None,
    philox_offset=None,
    swizzle_scale=False,
):
    args = (data_hp,)
    kwargs = dict(
        block_size=block_size,
        axis=axis,
        use_sr=use_sr,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        swizzle_scale=swizzle_scale,
    )
    if block_size != 32:
        return _legacy(
            "convert_to_mxfp4",
            args,
            kwargs,
            "upstream quantization uses 32-value blocks",
        )

    def run():
        from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant

        from lumen.ops.quantize import ops

        x = data_hp.transpose(-2, -1).contiguous() if axis in (0, -2) else data_hp
        shape = x.shape
        x = x.reshape(-1, shape[-1]).contiguous()
        seed, offset = _rng(use_sr, philox_seed, philox_offset)
        # Force Triton so an AITER upgrade cannot silently choose another quantizer.
        packed, scales = dynamic_mxfp4_quant(
            x,
            use_sr=use_sr,
            philox_seed=seed,
            philox_offset=offset,
            backend="triton",
        )
        packed, scales = (
            packed.view(torch.uint8).contiguous(),
            scales.view(torch.uint8).contiguous(),
        )
        packed = packed.reshape(*shape[:-1], shape[-1] // 2)
        if swizzle_scale:
            if axis in (0, -2):
                raise ValueError("swizzled scales cannot be transposed afterwards")
            return packed, ops.swizzle_mxfp4_scale(scales)
        scales = scales.reshape(*shape[:-1], shape[-1] // block_size)
        if axis in (0, -2):
            return (
                packed.transpose(-2, -1).contiguous(),
                scales.transpose(-2, -1).contiguous(),
            )
        return packed, scales

    return _run("dynamic_mxfp4_quant", "convert_to_mxfp4", run, args, kwargs)


def convert_to_mxfp4_2d(
    data_hp,
    block_size=32,
    use_sr=False,
    philox_seed=None,
    philox_offset=None,
    shuffle_data=False,
):
    args = (data_hp,)
    kwargs = dict(
        block_size=block_size,
        use_sr=use_sr,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        shuffle_data=shuffle_data,
    )
    if block_size != 32 or use_sr:
        return _legacy(
            "convert_to_mxfp4_2d",
            args,
            kwargs,
            "upstream blockscale supports 32x32 RTN only",
        )

    def run():
        from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant_blockscale

        from lumen.ops.quantize.linear import _shuffle_mxfp4_weight

        shape = data_hp.shape
        packed, scales = dynamic_mxfp4_quant_blockscale(
            data_hp.reshape(-1, shape[-1]).contiguous(),
        )
        if shuffle_data:
            packed = _shuffle_mxfp4_weight(packed)
        return packed.reshape(*shape[:-1], shape[-1] // 2), scales

    return _run(
        "dynamic_mxfp4_quant_blockscale", "convert_to_mxfp4_2d", run, args, kwargs
    )


def transpose_packed_fp4(data_fp4, shuffle_data=False, in_shuffled=False):
    args = (data_fp4,)
    kwargs = dict(shuffle_data=shuffle_data, in_shuffled=in_shuffled)
    if in_shuffled:
        return _legacy(
            "transpose_packed_fp4",
            args,
            kwargs,
            "upstream transpose reads row-major payloads only",
        )

    def run():
        from aiter.ops.triton.quant.fast_transpose import (
            transpose_packed_fp4 as transpose,
        )

        from lumen.ops.quantize.linear import _shuffle_mxfp4_weight

        out = transpose(data_fp4)
        return _shuffle_mxfp4_weight(out) if shuffle_data else out

    return _run("transpose_packed_fp4", "transpose_packed_fp4", run, args, kwargs)


def dual_layout_quant_mxfp4(
    x,
    sign_vector,
    block_size=32,
    g=16,
    use_sr_row=True,
    use_sr_transposed=True,
    philox_seed=None,
    philox_offset=None,
    swizzle_scale=False,
    shuffle_col=False,
):
    args = (x, sign_vector)
    seed, offset = _rng(use_sr_row or use_sr_transposed, philox_seed, philox_offset)
    kwargs = dict(
        block_size=block_size,
        g=g,
        use_sr_row=use_sr_row,
        use_sr_transposed=use_sr_transposed,
        philox_seed=seed,
        philox_offset=offset,
        swizzle_scale=swizzle_scale,
        shuffle_col=shuffle_col,
    )

    def run():
        from aiter.ops.triton.quant.fused_quant_transpose_hadamard_mxfp4 import (
            fused_quant_transpose_hadamard_mxfp4,
        )

        return fused_quant_transpose_hadamard_mxfp4(x, sign_vector, **kwargs)

    return _run(
        "fused_quant_transpose_hadamard_mxfp4",
        "dual_layout_quant_mxfp4",
        run,
        args,
        kwargs,
    )


def dequant_hadamard_quant_mxfp4(
    data_fp4,
    scales,
    sign_vector,
    block_size=32,
    g=16,
    use_sr=False,
    philox_seed=None,
    philox_offset=None,
    swizzle_scale=False,
    shuffle_data=False,
    in_scale_swizzled=False,
):
    args = (data_fp4, scales, sign_vector)
    seed, offset = _rng(use_sr, philox_seed, philox_offset)
    kwargs = dict(
        block_size=block_size,
        g=g,
        use_sr=use_sr,
        philox_seed=seed,
        philox_offset=offset,
        swizzle_scale=swizzle_scale,
        shuffle_data=shuffle_data,
        in_scale_swizzled=in_scale_swizzled,
    )

    def run():
        from aiter.ops.triton.quant.dequant_hadamard_quant_mxfp4 import (
            dequant_hadamard_quant_mxfp4 as dequant,
        )

        return dequant(data_fp4, scales, sign_vector, **kwargs)

    return _run(
        "dequant_hadamard_quant_mxfp4",
        "dequant_hadamard_quant_mxfp4",
        run,
        args,
        kwargs,
    )
