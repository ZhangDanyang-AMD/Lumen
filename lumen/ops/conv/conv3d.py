###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Convolution via the FlyDSL implicit-GEMM kernel, with a torch fallback.

Motivating workload is the Qwen-Image VAE, whose convolutions are all 3x3. On
gfx950 the FlyDSL kernel runs one VAE forward's worth of convolution 1.64x faster
than ``F.conv2d``, frequency-weighted, and 2.06x if the caller already holds
channels-last activations.

Two properties of the kernel shape this module:

* **Forward only.** FlyDSL provides no dgrad/wgrad, so anything that needs
  gradients is routed to torch instead of silently losing them. Precedent for a
  forward-only op is :func:`lumen.ops.rope.apply_rotary_pos_emb_2d`.
* **BF16 only.** Other dtypes go to torch.

The kernel is channels-last internally, so an NCHW input costs a pre-transpose --
20% of total convolution time on this workload, and half of it on the downsample
layers. ``input_layout`` / ``output_layout`` are therefore exposed rather than
fixed, so a caller that keeps a whole VAE stage in channels-last pays the
transpose zero times instead of once per layer.
"""

import logging
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumen.ops.dispatch import FLYDSL_FALLBACK_ORDER, Backend, _probe_flydsl_conv3d, build_fallback_chain, try_backends

logger = logging.getLogger(__name__)

_LAYOUTS = {3: ("NCDHW", "NDHWC"), 2: ("NCHW", "NHWC"), 1: ("NCW", "NWC")}
_PADDING_MODES = ("zeros", "reflect", "replicate", "circular")

IntOrSeq = Union[int, Sequence[int]]


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _get_flydsl_conv3d():
    from lumen.kernels.conv.conv3d_implicit import conv3d_implicit

    return conv3d_implicit


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def _check_positive(value, name, rank):
    seq = (value,) if isinstance(value, int) else tuple(value)
    if len(seq) not in (1, rank):
        raise ValueError(f"{name} must be an int or a sequence of 1 or {rank} ints, got {value!r}")
    if min(seq) < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")


def _validate_args(x, weight, bias, stride, padding, dilation, groups, padding_mode, layouts, rank):
    """Reject arguments no backend could accept; raised to the caller.

    Kept separate from :func:`_flydsl_precheck` deliberately. These are caller
    mistakes, and demoting to torch would only turn a clear error into a
    confusing one further down. Capability limits go in the pre-check instead, so
    that they *do* demote.
    """
    if x.dim() not in (weight.dim(), weight.dim() - 1):
        raise ValueError(f"x rank {x.dim()} incompatible with weight rank {weight.dim()}")

    for name, layout in zip(("input_layout", "output_layout"), layouts):
        if layout not in _LAYOUTS[rank]:
            raise ValueError(f"{name} for a {rank}D filter must be one of {_LAYOUTS[rank]}, got {layout!r}")

    # Mixed dtypes are rejected by torch too, so this is an argument error, not a
    # capability limit -- demoting would just produce a murkier message.
    if x.dtype is not weight.dtype:
        raise ValueError(f"x and weight must have the same dtype; got {x.dtype} and {weight.dtype}")

    k = weight.shape[0]
    if bias is not None and (bias.dim() != 1 or bias.numel() != k):
        raise ValueError(f"bias must be 1-D with {k} elements, one per output channel; got shape {tuple(bias.shape)}")

    _check_positive(stride, "stride", rank)
    _check_positive(dilation, "dilation", rank)
    if isinstance(padding, str):
        if padding not in ("same", "valid"):
            raise ValueError(f"padding string must be 'same' or 'valid', got {padding!r}")
    else:
        pads = (padding,) if isinstance(padding, int) else tuple(padding)
        if len(pads) not in (1, rank):
            raise ValueError(f"padding must be an int or a sequence of 1 or {rank} ints, got {padding!r}")
        if min(pads) < 0:
            raise ValueError(f"negative padding is not supported, got {padding!r}")
    if padding_mode not in _PADDING_MODES:
        raise ValueError(f"padding_mode must be one of {_PADDING_MODES}, got {padding_mode!r}")

    groups = int(groups)
    if groups < 1:
        raise ValueError(f"groups must be >= 1, got {groups}")
    # Channel axis is innermost only in the channels-last layouts.
    c = x.shape[-1] if layouts[0] == _LAYOUTS[rank][1] else x.shape[-rank - 1]
    if c % groups or k % groups:
        raise ValueError(f"in-channels {c} and out-channels {k} must both be divisible by groups {groups}")
    if weight.shape[1] != c // groups:
        raise ValueError(f"weight in-channels {weight.shape[1]} != C/groups = {c // groups}")


def _flydsl_supported(x, weight, bias):
    """Can the FlyDSL kernel handle this at all? Decided up front, not by trying.

    Dtype and device are knowable without running anything, so they are settled
    here rather than raised for ``try_backends`` to catch. That matters because
    the dispatcher caches the winning backend after
    ``_BACKEND_WARMUP_CALLS`` successes and the cached path has no try/except: a
    later unsupported input would surface as an exception to the caller instead of
    demoting. Deciding in advance keeps the torch fallback reachable for the
    lifetime of the process.

    Limits that are *not* knowable in advance -- output extent, reflect/circular
    pad versus input extent -- stay asserts inside the kernel, and
    :func:`_conv_flydsl` converts them so the chain can still demote on a first
    call.
    """
    if x.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        return False
    return all(t is None or t.is_cuda for t in (x, weight, bias))


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _conv_flydsl(x, weight, bias, stride, padding, dilation, groups, padding_mode, layouts, rank, **kw):
    fn = _get_flydsl_conv3d()
    try:
        return fn(
            x,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            input_layout=layouts[0],
            output_layout=layouts[1],
            **kw,
        )
    except AssertionError as exc:
        # Constraints the pre-check does not cover stay asserts inside the kernel;
        # an AssertionError here would abort the chain instead of demoting.
        raise ValueError(f"FlyDSL conv rejected this problem: {exc}") from exc


def _to_channels_first(t, layout, rank):
    """Permute a channels-last tensor to channels-first for the torch path."""
    if layout != _LAYOUTS[rank][1]:
        return t
    # (N, *spatial, C) -> (N, C, *spatial); an unbatched input has no N.
    dims = list(range(t.dim()))
    lead = dims[:-1][: t.dim() - rank - 1]
    spatial = dims[len(lead) : -1]
    return t.permute(*lead, dims[-1], *spatial).contiguous()


def _from_channels_first(t, layout, rank):
    if layout != _LAYOUTS[rank][1]:
        return t
    dims = list(range(t.dim()))
    lead = dims[: t.dim() - rank - 1]
    return t.permute(*lead, *dims[len(lead) + 1 :], dims[len(lead)])


def _conv_torch(x, weight, bias, stride, padding, dilation, groups, padding_mode, layouts, rank, **_kw):
    """Reference path. Also the autograd path, since FlyDSL has no backward."""
    x_cf = _to_channels_first(x, layouts[0], rank)
    unbatched = x_cf.dim() == weight.dim() - 1
    if unbatched:
        x_cf = x_cf.unsqueeze(0)

    if padding_mode != "zeros" and padding != 0:
        # torch's conv only pads with zeros; materialize the other modes, which
        # is what the FlyDSL kernel does internally for large inputs too.
        pads = (padding,) * rank if isinstance(padding, int) else tuple(padding)
        pad_arg = []
        for p in reversed(pads):
            pad_arg += [p, p]
        x_cf = F.pad(x_cf, pad_arg, mode=padding_mode)
        padding = 0

    conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[rank]
    bias_cast = bias.to(x_cf.dtype) if bias is not None else None
    out = conv(x_cf, weight, bias=bias_cast, stride=stride, padding=padding, dilation=dilation, groups=groups)
    if unbatched:
        out = out.squeeze(0)
    return _from_channels_first(out, layouts[1], rank)


_conv_chain = None


def _get_conv_chain():
    global _conv_chain
    if _conv_chain is None:
        candidates = {Backend.TORCH: _conv_torch}
        if _probe_flydsl_conv3d():
            candidates[Backend.FLYDSL] = _conv_flydsl
        else:
            logger.info("conv: flydsl unavailable, using torch")
        _conv_chain = build_fallback_chain(candidates, order=FLYDSL_FALLBACK_ORDER)
    return _conv_chain


# ---------------------------------------------------------------------------
# Public functional API
# ---------------------------------------------------------------------------


def _dispatch_conv(
    x,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    padding_mode,
    input_layout,
    output_layout,
    rank,
    **kw,
):
    if weight.dim() - 2 != rank:
        raise ValueError(f"expected a {rank}D filter, i.e. weight.dim() == {rank + 2}, got {weight.dim()}")
    chan_first = _LAYOUTS[rank][0]
    layouts = (
        chan_first if input_layout is None else input_layout,
        chan_first if output_layout is None else output_layout,
    )
    args = (x, weight, bias, stride, padding, dilation, groups, padding_mode, layouts, rank)
    _validate_args(*args)

    # FlyDSL has no backward, so never let it see a tensor that needs one --
    # torch would otherwise be reached only on failure, and the result would
    # silently have no grad_fn.
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        return _conv_torch(*args, **kw)

    if not _flydsl_supported(x, weight, bias):
        return _conv_torch(*args, **kw)

    return try_backends(_get_conv_chain(), *args, op_name=f"conv{rank}d", **kw)


def conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: IntOrSeq = 1,
    padding: Union[IntOrSeq, str] = 0,
    dilation: IntOrSeq = 1,
    groups: int = 1,
    padding_mode: str = "zeros",
    input_layout: Optional[str] = None,
    output_layout: Optional[str] = None,
    **kw,
) -> torch.Tensor:
    """2D convolution over a BF16 input, via FlyDSL where possible.

    Args:
        x: ``(N, C, H, W)``, or ``(N, H, W, C)`` when ``input_layout="NHWC"``.
            The batch axis may be omitted.
        weight: ``(K, C // groups, R, S)``.
        bias: Optional ``(K,)``.
        stride, padding, dilation, groups: torch semantics. ``padding`` also
            takes ``"same"`` or ``"valid"``; ``"same"`` requires stride 1.
        padding_mode: ``"zeros"``, ``"reflect"``, ``"replicate"``, ``"circular"``.
        input_layout, output_layout: ``"NCHW"`` (default) or ``"NHWC"``,
            independent. Channels-last skips a transpose on that side.

    Returns:
        BF16 output in ``output_layout``.

    Falls back to ``F.conv2d`` when flydsl is absent, the problem is outside the
    kernel's support, or gradients are required.
    """
    return _dispatch_conv(
        x, weight, bias, stride, padding, dilation, groups, padding_mode, input_layout, output_layout, 2, **kw
    )


def conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: IntOrSeq = 1,
    padding: Union[IntOrSeq, str] = 0,
    dilation: IntOrSeq = 1,
    groups: int = 1,
    padding_mode: str = "zeros",
    input_layout: Optional[str] = None,
    output_layout: Optional[str] = None,
    **kw,
) -> torch.Tensor:
    """3D convolution over a BF16 input, via FlyDSL where possible.

    Args:
        x: ``(N, C, D, H, W)``, or ``(N, D, H, W, C)`` when
            ``input_layout="NDHWC"``. The batch axis may be omitted.
        weight: ``(K, C // groups, T, R, S)``.

    Other arguments and the fallback behaviour match :func:`conv2d`.

    Note:
        A Qwen-Image ``CausalConv3d`` 3x3x3 at ``T == 1`` is *not* this call with
        ``padding=1``: causal padding puts 2 zeros before the time axis and 0
        after, so only ``weight[:, :, 2]`` sees real pixels. Use
        :func:`conv2d` on that slice instead; symmetric padding would pad time on
        both sides and is not causal.
    """
    return _dispatch_conv(
        x, weight, bias, stride, padding, dilation, groups, padding_mode, input_layout, output_layout, 3, **kw
    )


# ---------------------------------------------------------------------------
# nn.Module API
# ---------------------------------------------------------------------------


class LumenConv3d(nn.Module):
    """Conv3d backed by the FlyDSL implicit-GEMM kernel.

    Argument names and defaults follow ``nn.Conv3d`` so an existing module can be
    swapped for this one. Inference-oriented: with ``requires_grad`` parameters
    the forward routes to torch, because the kernel has no backward.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: int or 3-tuple.
        stride, padding, dilation, groups, bias, padding_mode: as ``nn.Conv3d``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrSeq,
        stride: IntOrSeq = 1,
        padding: Union[IntOrSeq, str] = 0,
        dilation: IntOrSeq = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        ks = (kernel_size,) * 3 if isinstance(kernel_size, int) else tuple(kernel_size)
        if in_channels % groups:
            raise ValueError(f"in_channels {in_channels} not divisible by groups {groups}")
        self.stride, self.padding, self.dilation = stride, padding, dilation
        self.groups, self.padding_mode = groups, padding_mode
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *ks, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv3d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            padding_mode=self.padding_mode,
        )

    def extra_repr(self) -> str:
        k, cg = self.weight.shape[0], self.weight.shape[1]
        ks = tuple(self.weight.shape[2:])
        return (
            f"{cg * self.groups}, {k}, kernel_size={ks}, stride={self.stride}, "
            f"padding={self.padding}, groups={self.groups}, bias={self.bias is not None}"
        )
