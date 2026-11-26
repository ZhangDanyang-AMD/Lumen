###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""TransformerEngine ``MXFP8BlockScaling`` numerics in plain PyTorch (reference only, not a fast path).

TE (``transformer_engine/common``, MXFP8 cast kernels) quantizes each 1x32 block as:

* ``amax`` = max |x| over the block (fp32);
* E8M0 scale exponent = ``ceil(log2(amax / 448))`` (``float_to_e8m0(amax * max_norm_rcp)``,
  round toward +inf), biased by 127 and clamped to [0, 254]; ``amax == 0`` gives byte 0;
* ``q = satfinite_e4m3_rne(x * 2^(127 - biased))``.

Rounding the exponent up keeps every block's amax at or below 448 after scaling, so nothing
saturates except through the final RNE. aiter's ``_convert_to_mxfp8_kernel`` (used by Lumen's
``quantized_linear(scaling_type="mxfp8")``) follows torchao instead: exponent =
``floor(log2(rne_3bit(amax))) - 8``. Blocks whose amax mantissa rounds down (e.g. amax in
(448, 480), or 230 = 1.797 * 2^7) then get a scale one power of two smaller, and their largest
values clip at 448.

Operand axes (TE ``pytorch/module/linear.py``), for ``Y = X W^T`` with X ``[M, K]``, W ``[N, K]``:

* fprop: X rowwise (blocks along K) x W rowwise (along K);
* dgrad: dY rowwise (along N) x W columnwise (along N);
* wgrad: dY columnwise (along M) x X columnwise (along M), accumulated in fp32
  (``main_grad`` with ``fuse_wgrad_accumulation``).

The GEMMs here multiply the dequantized operands in fp32: identical products to an MXFP8 MMA
with fp32 accumulation, up to summation order.
"""

import torch

MXFP8_BLOCK = 32
E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448
E8M0_BIAS = 127
E8M0_MAX_BIASED = 254  # 255 encodes NaN


def e8m0_scale_exponent(amax: torch.Tensor) -> torch.Tensor:
    """Biased E8M0 byte per block: ``ceil(log2(amax / 448)) + 127``, clamped to [0, 254]; 0 for amax 0."""
    v = amax.float() * (1.0 / E4M3_MAX)
    m, e = torch.frexp(v)  # v = m * 2^e, m in [0.5, 1)
    ceil_log2 = torch.where(m == 0.5, e - 1, e)
    biased = (ceil_log2 + E8M0_BIAS).clamp(0, E8M0_MAX_BIASED)
    return torch.where(v > 0, biased, torch.zeros_like(biased)).to(torch.int32)


def _blocks(x, axis):
    xt = x.float().movedim(axis, -1)
    n = xt.shape[-1]
    pad = (-n) % MXFP8_BLOCK
    xp = torch.nn.functional.pad(xt, (0, pad)) if pad else xt
    return xp.reshape(*xp.shape[:-1], -1, MXFP8_BLOCK), xp.shape, n


def quantize(x: torch.Tensor, axis: int):
    """(e4m3 data blocked along ``axis``, biased E8M0 bytes ``[..., n_blocks]`` with ``axis`` moved last).

    A trailing partial block is padded with zeros for the amax (TE requires multiples of 32;
    the padding only matters for small test shapes).
    """
    blocks, padded_shape, n = _blocks(x, axis)
    biased = e8m0_scale_exponent(blocks.abs().amax(-1))
    rcp = torch.exp2((E8M0_BIAS - biased).float()).unsqueeze(-1)
    q = (blocks * rcp).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q.reshape(padded_shape)[..., :n].movedim(-1, axis), biased


def fake_quantize(x: torch.Tensor, axis: int) -> torch.Tensor:
    """Dequantized fp32 values TE's MXFP8 GEMM sees for ``x`` blocked along ``axis``."""
    blocks, padded_shape, n = _blocks(x, axis)
    biased = e8m0_scale_exponent(blocks.abs().amax(-1))
    rcp = torch.exp2((E8M0_BIAS - biased).float()).unsqueeze(-1)
    q = (blocks * rcp).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()
    return (q / rcp).reshape(padded_shape)[..., :n].movedim(-1, axis)


def _mm_fp32(a, b):
    """fp32 GEMM without TF32, so the result equals fp32 accumulation of exact fp8 x fp8 products."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return a @ b
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


class TEMXFP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        x2 = x.reshape(-1, x.shape[-1])
        ctx.save_for_backward(x2, w)
        ctx.x_shape = x.shape
        y = _mm_fp32(fake_quantize(x2, -1), fake_quantize(w, -1).t())
        return y.to(x.dtype).reshape(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x2, w = ctx.saved_tensors
        dy2 = dy.reshape(-1, dy.shape[-1])
        dx = _mm_fp32(fake_quantize(dy2, -1), fake_quantize(w, 0)).to(x2.dtype).reshape(ctx.x_shape)
        dw = _mm_fp32(fake_quantize(dy2, 0).t(), fake_quantize(x2, 0))
        main_grad = getattr(w, "main_grad", None)
        if main_grad is not None:
            main_grad.add_(dw)
            return dx, None
        return dx, dw.to(w.dtype)


def te_mxfp8_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``x @ w.T`` with TE MXFP8BlockScaling numerics in fprop, dgrad and wgrad.

    When ``w.main_grad`` (fp32) exists, wgrad is accumulated into it in fp32 and ``w.grad`` stays
    unset, as with Megatron's ``gradient_accumulation_fusion``.
    """
    return TEMXFP8Linear.apply(x, w)


def attach_main_grad(params):
    """Give each parameter a zeroed fp32 ``main_grad`` buffer (Megatron-style wgrad accumulation)."""
    for p in params:
        p.main_grad = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
    return params
