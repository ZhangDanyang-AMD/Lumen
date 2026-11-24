"""aiter's FlyDSL implicit-GEMM convolution for SimWAM's Wan2.2 VAE, local to this example.

Lumen's library conv path (``LumenConfig(aiter_conv=True)``, ``lumen.ops.conv``) is aiter's
Triton conv; the library has no FlyDSL option. This module patches the VAE with aiter's
``aiter.ops.flydsl.conv_kernels.flydsl_conv_implicit`` instead, the way
``examples/perf-veomni`` does for Wan2.1 / Qwen-Image, without touching ``lumen/``.

Two rewrites, both exact:

* **T = 1 without a feature cache** -- the 1-frame encode the NAVSIM GRPO rollout and the
  navtest eval run, and the first chunk of any encode. A ``CausalConv3d`` prepends
  ``kT - 1`` zero frames, so only the last temporal tap of the kernel ever meets data:
  the call is the 2-D convolution with ``weight[:, :, -1]``.
* Every other call keeps the module's own causal padding and cache concat; only the
  convolution under it moves to the kernel.

The kernel picks launch tiles from per-model tuned tables
(``aiter/configs/model_configs/*_bf16_tuned_conv3d.csv``). The Wan2.2 table comes with
ROCm/aiter#7372. Without it every lookup falls back to the heuristic tile: the encode is
still correct but slower, and :func:`warn_if_untuned` logs it. :func:`lookup_stats` counts
how each lookup resolved (exact row, a row borrowed from the same layer at another
resolution, or the heuristic), because a table that never matches looks like one that
works except in the timing.

The kernel is forward-only, so a call that is not BF16, not on the GPU, or needs a
gradient goes to torch; :func:`conv_calls` counts both.
"""

import functools
import logging
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_lookups = {"exact": 0, "borrowed": 0, "heuristic": 0, "unkeyed": 0}
_shapes = {}
_calls = {"flydsl": 0, "torch": 0}
_trace = None


def _count_lookups(mod):
    """Wrap the kernel's tuned-row lookup so that every call's resolution is counted."""
    lookup = mod._lookup_tuned_tile
    if getattr(lookup, "_wan22_counted", False):
        return

    @functools.lru_cache(maxsize=1)
    def _device():
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx

        return (get_gfx(), get_cu_num())

    def counted(key, device):
        hit = lookup(key, device)
        if key is None:
            _lookups["unkeyed"] += 1
            return hit
        if hit is None:
            how = "heuristic"
        elif (*_device(), *key) in mod._load_tuned_table():
            how = "exact"
        else:
            how = "borrowed"
        _lookups[how] += 1
        _shapes[key] = how
        return hit

    counted._wan22_counted = True
    mod._lookup_tuned_tile = counted


@functools.lru_cache(maxsize=1)
def load():
    """``flydsl_conv_implicit`` from aiter, or None if this install cannot provide it."""
    try:
        import aiter.ops.flydsl.conv_kernels as mod

        fn = mod.flydsl_conv_implicit
    except (ImportError, OSError, AttributeError, RuntimeError) as exc:
        logger.warning("FlyDSL conv unavailable (%s: %s)", type(exc).__name__, exc)
        return None
    _count_lookups(mod)
    rows = len(mod._load_tuned_table())
    if not rows:
        logger.warning("FlyDSL conv: tuned table is empty; every convolution will take the heuristic tile")
    logger.info("FlyDSL conv: %s with %d tuned rows", mod.__name__, rows)
    return fn


def conv(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """``F.conv2d`` / ``F.conv3d`` by filter rank: the FlyDSL kernel where it applies, torch otherwise."""
    needs_grad = torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad)
    eligible = not needs_grad and x.is_cuda and x.dtype is torch.bfloat16 and weight.dtype is torch.bfloat16
    fn = load() if eligible else None
    if fn is None:
        _calls["torch"] += 1
        op = F.conv3d if weight.dim() == 5 else F.conv2d
        return op(x, weight, None if bias is None else bias.to(x.dtype), stride, padding, dilation, groups)
    _calls["flydsl"] += 1
    if _trace is not None:
        key = (tuple(x.shape), tuple(weight.shape), _tuple(stride, weight), _tuple(padding, weight),
               _tuple(dilation, weight), groups, bias is not None)  # fmt: skip
        _trace[key] = _trace.get(key, 0) + 1
    return fn(x, weight, bias, stride, padding, dilation, groups=groups)


def _tuple(v, weight):
    n = weight.dim() - 2
    return tuple(v) if isinstance(v, (tuple, list)) else (v,) * n


def _is_causal_conv3d(m):
    # SimWAM's CausalConv3d keeps its causal padding as a 6-tuple ``_padding`` and
    # convolves with ``padding=0``; matched on that rather than on a class name.
    return isinstance(m, nn.Conv3d) and len(getattr(m, "_padding", ())) == 6


def _t1_spatial_pad(m):
    """The (padH, padW) of the 2-D convolution a T=1 uncached call reduces to, or None."""
    pad_w_l, pad_w_r, pad_h_t, pad_h_b, pad_t_front, pad_t_back = m._padding
    if pad_w_l != pad_w_r or pad_h_t != pad_h_b or pad_t_back != 0 or pad_t_front != m.kernel_size[0] - 1:
        return None
    if m.stride[0] != 1 or m.dilation[0] != 1 or tuple(m.padding) != (0, 0, 0):
        return None
    return (pad_h_t, pad_w_l)


def _bind_kernel(m):
    def _conv_forward(self, input, weight, bias):
        return conv(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    m._conv_forward = types.MethodType(_conv_forward, m)


def _bind_t1_conv2d(m, spatial_pad):
    # The VAE is frozen, so the last temporal tap is materialised once.
    weight2d = m.weight[:, :, -1].contiguous()
    original = m.forward

    def forward(self, x, cache_x=None, _w=weight2d, _pad=spatial_pad, _orig=original):
        # A cache means the leading frames are features, not zeros: the identity does not hold.
        if cache_x is not None or x.dim() != 5 or x.shape[2] != 1:
            return _orig(x, cache_x)
        out = conv(x[:, :, 0], _w, self.bias, self.stride[1:], _pad, self.dilation[1:], self.groups)
        return out.unsqueeze(2)

    m.forward = types.MethodType(forward, m)


def patch_vae(vae):
    """Route every convolution of ``vae`` (a SimWAM ``WanVideoVAE38``) to the FlyDSL kernel.

    Returns ``(n_causal_conv3d, n_conv2d, n_t1_rewrites)``. Only ``forward`` and
    ``_conv_forward`` are rebound; parameters and the module tree are untouched.
    """
    n_c3 = n_c2 = n_t1 = 0
    for m in vae.modules():
        if _is_causal_conv3d(m):
            n_c3 += 1
            pad = _t1_spatial_pad(m)
            if pad is not None:
                _bind_t1_conv2d(m, pad)
                n_t1 += 1
            _bind_kernel(m)
        elif type(m) is nn.Conv2d and m.padding_mode == "zeros" and not isinstance(m.padding, str):
            n_c2 += 1
            _bind_kernel(m)
    return n_c3, n_c2, n_t1


def conv_calls():
    """Calls of :func:`conv` that ran on the FlyDSL kernel and on torch."""
    return dict(_calls)


def lookup_stats():
    """Tuned-table lookups resolved by exact row, borrowed row or heuristic (calls and distinct shapes)."""
    distinct = {k: 0 for k in _lookups if k != "unkeyed"}
    for how in _shapes.values():
        distinct[how] += 1
    return {"calls": dict(_lookups), "shapes": distinct}


def warn_if_untuned():
    """Log the shapes that took the heuristic tile; returns how many there were."""
    misses = sorted(k for k, how in _shapes.items() if how != "exact")
    if misses:
        logger.warning(
            "FlyDSL conv: %d of %d looked-up shapes have no exact tuned row (%s); the encode is correct "
            "but slower. The Wan2.2 table comes with ROCm/aiter#7372 "
            "(aiter/configs/model_configs/wan22_vae_bf16_tuned_conv3d.csv).",
            len(misses), len(_shapes), lookup_stats()["shapes"],
        )  # fmt: skip
    return len(misses)


class Trace:
    """Collect the distinct FlyDSL calls (shapes and conv parameters) made while active."""

    def __enter__(self):
        global _trace
        _trace = {}
        self.calls = _trace
        return self

    def __exit__(self, *exc):
        global _trace
        _trace = None
        return False


def ternary_check(call, generator):
    """One traced FlyDSL call on {-1, 0, 1} BF16 data against a CPU fp32 reference.

    Every product and partial sum is an integer far below 2**24, so an implicit-GEMM
    kernel accumulating in fp32 gets the exact sum, and the BF16 rounding is the same on
    both sides: the result has to equal the reference bit for bit.
    Returns (flydsl_calls, max|d|, kernel output, reference).
    """
    x_shape, w_shape, stride, padding, dilation, groups, has_bias = call
    x = torch.randint(-1, 2, x_shape, generator=generator).to(torch.bfloat16)
    w = torch.randint(-1, 2, w_shape, generator=generator).to(torch.bfloat16)
    b = torch.randint(-1, 2, (w_shape[0],), generator=generator).to(torch.bfloat16) if has_bias else None
    op = F.conv3d if len(w_shape) == 5 else F.conv2d
    ref = op(x.float(), w.float(), None if b is None else b.float(), stride, padding, dilation, groups).bfloat16()
    before = _calls["flydsl"]
    with torch.no_grad():
        got = conv(x.cuda(), w.cuda(), None if b is None else b.cuda(), stride, padding, dilation, groups).cpu()
    return _calls["flydsl"] - before, (got.float() - ref.float()).abs().max().item(), got, ref
