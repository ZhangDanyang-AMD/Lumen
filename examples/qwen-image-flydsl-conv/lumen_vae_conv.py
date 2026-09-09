#!/usr/bin/env python3
"""Route a diffusers VAE's convolutions through Lumen's FlyDSL kernel.

Used by ``train_dit_lumen.py`` (LUMEN_PATCH=vae_conv / vae_conv_video) and by
``verify_vae_patch.py`` and ``verify_video_vae_patch.py``, which check it against
the unpatched model.

Why this is safe here
---------------------
``DiTTrainer._freeze_model_module`` calls ``condition_model.requires_grad_(False)``
and the task is ``online_training``, so the VAE runs forward-only inside
``@torch.no_grad()``. FlyDSL has no dgrad/wgrad, and that is exactly the
constraint this workload already satisfies.

Two rewrites, because images and video are different problems
-------------------------------------------------------------
``QwenImageCausalConv3d`` and ``WanCausalConv3d`` are the same module written
twice: both subclass ``nn.Conv3d``, pad time asymmetrically (``kT-1`` zero frames
in front, none behind, minus whatever an inference feature cache supplies), and
then convolve with ``padding=0``.

**T == 1 and no feature cache** -- an image, or a video VAE's first chunk. Every
kernel slice but the last multiplies zeros, so

    conv3d(pad(x), w)  ==  conv2d(x, w[:, :, -1])

exactly -- not approximately. Symmetric ``padding=1`` on a 3-D convolution would
*not* be causal and would give a different answer. This is where most of the
Qwen-Image win comes from: torch's conv3d is poor on a degenerate time axis, and
leaving conv3d is worth more than the kernel swap is (2.27x versus 1.13x at
256^2, see the example README).

**Everything else** -- a real clip, or a T == 1 chunk whose leading frames come
from the feature cache rather than from zeros. No 2-D identity exists: the
cached frames are real data, so dropping the other kernel slices would change
the result. Here the module's own padding is left exactly as it is and only the
convolution underneath it is replaced, with Lumen's ``conv3d`` on the unmodified
5-D filter. ``nn.Conv3d.forward`` is a single call to
``self._conv_forward(input, self.weight, self.bias)``, so rebinding
``_conv_forward`` swaps the kernel and nothing else -- the causal padding and the
cache handling are reused verbatim instead of being reimplemented here, which is
the part that would be easy to get wrong.

The second rewrite is what ``video=True`` adds, and it is where a video VAE's
time goes: at 17 frames of 480x832, 83.3% of one Wan encode's convolution time
is on T>1 or feature-cache calls, and FlyDSL is 1.60x torch on them
(``trace_video_vae_convs.py``).

What is left on torch, and why
------------------------------
Pointwise convolutions -- ``conv_shortcut``, ``quant_conv``, and Wan's
``time_conv`` (3x1x1, so its *spatial* kernel is 1) -- are launch-bound and
measure 0.88-1.05x through Lumen. ``min_spatial_kernel`` skips them.

The plain ``nn.Conv2d`` spatial resamplers are resolution-dependent and so are
tied to ``video``: 0.88x at Qwen-Image's 256^2, but 1.19-1.86x on Wan's
481x833, where they are 4.7% of convolution time. Off for images, on for video.
"""

import types

import torch
import torch.nn as nn


# Both are nn.Conv3d subclasses with an identical causal forward. Matching on the
# name rather than the type keeps this file from importing either autoencoder
# module, which would pull in the model the caller may not be using.
_CAUSAL_CONV3D = ("QwenImageCausalConv3d", "WanCausalConv3d")


def _is_causal_conv3d(m):
    return type(m).__name__ in _CAUSAL_CONV3D


def _degenerate_to_conv2d(m):
    """Can this module's 3-D convolution be rewritten as a 2-D one at T=1?

    Returns the spatial (padH, padW) to hand the 2-D convolution, or None.
    """
    pad = getattr(m, "_padding", None)
    if pad is None or len(pad) != 6:
        return None
    pad_w_l, pad_w_r, pad_h_t, pad_h_b, pad_t_front, pad_t_back = pad
    if pad_w_l != pad_w_r or pad_h_t != pad_h_b:
        return None  # asymmetric spatial padding needs an explicit F.pad
    kt = m.kernel_size[0]
    if pad_t_back != 0 or pad_t_front != kt - 1:
        return None
    if m.stride[0] != 1 or m.dilation[0] != 1:
        return None
    if tuple(m.padding) != (0, 0, 0):
        return None  # the module is expected to have moved padding into _padding
    return (pad_h_t, pad_w_l)


def _bind_lumen_kernel(m):
    """Replace the convolution inside an ``nn.Conv{2,3}d``, leaving its padding alone.

    Applies at any time extent, which is the point: the caller keeps whatever
    padding, cropping or cache concatenation it already does and only the kernel
    changes.
    """
    import lumen.ops.conv as conv_ops

    op = conv_ops.conv3d if isinstance(m, nn.Conv3d) else conv_ops.conv2d

    def _conv_forward(self, input, weight, bias, _op=op):
        return _op(
            input,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    m._conv_forward = types.MethodType(_conv_forward, m)


def _bind_t1_conv2d(m, spatial_pad):
    """Rebind forward so a T=1 uncached call becomes an exact 2-D convolution."""
    import lumen.ops.conv as conv_ops

    # The VAE is frozen, so the last time-slice of the kernel is constant for the
    # life of the run and worth materialising once instead of per call.
    weight2d = m.weight[:, :, -1].contiguous()
    stride2d = tuple(m.stride[1:])
    dilation2d = tuple(m.dilation[1:])
    groups = m.groups
    original = m.forward

    def forward(
        self,
        x,
        cache_x=None,
        _w=weight2d,
        _pad=spatial_pad,
        _orig=original,
        _stride=stride2d,
        _dil=dilation2d,
        _groups=groups,
    ):
        # A cache means the leading frames are real features, not zeros, so the
        # 2-D identity does not hold and the module's own forward has to run.
        if cache_x is not None or x.dim() != 5 or x.shape[2] != 1:
            return _orig(x, cache_x)
        out = conv_ops.conv2d(
            x[:, :, 0],
            _w,
            self.bias,
            stride=_stride,
            padding=_pad,
            dilation=_dil,
            groups=_groups,
        )
        return out.unsqueeze(2)

    m.forward = types.MethodType(forward, m)


def patch_vae_convs(vae, min_spatial_kernel=2, video=False, verbose=None):
    """Rebind the convolutions of a diffusers VAE. Returns a summary.

    Args:
        vae: ``AutoencoderKLQwenImage``, ``AutoencoderKLWan``, or any VAE built
            from ``nn.Conv3d`` subclasses that move causal padding into
            ``_padding``.
        min_spatial_kernel: Skip convolutions whose spatial kernel is smaller
            than this. The default of 2 leaves the pointwise layers on torch,
            where they measure faster.
        video: Also take the shapes the T=1 identity cannot -- real clips and
            cached chunks -- plus the plain 2-D resamplers. Required for a video
            VAE, where those are nearly all of the time; measures slower than
            torch at 256^2, so off by default.
        verbose: Optional ``print``-like callable for a one-line summary.

    Only ``forward`` and ``_conv_forward`` are rebound; parameters, dtypes and the
    module tree are untouched, which is what keeps this usable under
    ``init_device=meta`` and FSDP2.
    """
    stats = {"patched": [], "skipped": []}

    for name, m in vae.named_modules():
        causal = _is_causal_conv3d(m)
        plain = video and type(m) in (nn.Conv2d, nn.Conv3d)
        if not (causal or plain):
            continue

        ks = tuple(m.kernel_size)
        if max(ks[-2:]) < min_spatial_kernel:
            stats["skipped"].append((name, f"pointwise {ks}"))
            continue
        if m.padding_mode != "zeros" or isinstance(m.padding, str):
            # nn.Conv's own _conv_forward pads for these modes; replicating that
            # here would duplicate logic for no measured gain.
            stats["skipped"].append((name, f"padding {m.padding_mode} {m.padding!r}"))
            continue

        how = []
        if causal:
            spatial_pad = _degenerate_to_conv2d(m)
            if spatial_pad is not None:
                _bind_t1_conv2d(m, spatial_pad)
                how.append("T=1->conv2d")
        if video:
            # Puts the kernel underneath the module's own padding, so the calls
            # the identity above cannot claim are covered too.
            _bind_lumen_kernel(m)
            how.append("conv3d" if isinstance(m, nn.Conv3d) else "conv2d")

        if not how:
            stats["skipped"].append((name, "not reducible to conv2d"))
            continue
        stats["patched"].append((name, ks, "+".join(how)))

    if verbose:
        if video:
            detail = "convolutions"
            label = "vae_conv_video"
        else:
            detail = "QwenImageCausalConv3d"
            label = "vae_conv"
        verbose(f"{label}: patched {len(stats['patched'])} {detail}, skipped {len(stats['skipped'])}")
    return stats


def reference_encode(vae, x):
    """Latent posterior parameters, for comparing patched against unpatched."""
    with torch.no_grad():
        return vae.encode(x).latent_dist.parameters


def snr_db(ref, test):
    signal = torch.norm(ref.float()).pow(2)
    noise = torch.norm(ref.float() - test.float()).pow(2)
    if noise.item() == 0.0:
        return float("inf")
    return (10.0 * torch.log10(signal / noise)).item()


__all__ = ["patch_vae_convs", "reference_encode", "snr_db"]
