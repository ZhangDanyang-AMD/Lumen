#!/usr/bin/env python3
"""Route the Qwen-Image VAE's convolutions through Lumen's FlyDSL kernel.

Used by ``train_dit_lumen.py`` (LUMEN_PATCH=vae_conv) and by
``verify_vae_patch.py``, which checks it against the unpatched model.

Why this is safe here
---------------------
``DiTTrainer._freeze_model_module`` calls ``condition_model.requires_grad_(False)``
and the task is ``online_training``, so the VAE runs forward-only inside
``@torch.no_grad()``. FlyDSL has no dgrad/wgrad, and that is exactly the
constraint this workload already satisfies.

What gets replaced, and what deliberately does not
--------------------------------------------------
Only ``QwenImageCausalConv3d`` with a spatial kernel > 1. A trace of one 256x256
encode (``trace_vae_convs.py``) shows where the time is:

    layer                        calls   torch     lumen NCHW
    down_blocks.0.conv1              4   165.5us    44.7us   3.70x
    down_blocks.9.conv1              8   116.0us    41.2us   2.82x
    down_blocks.6.conv2              3   125.8us    42.0us   3.00x
    resample.1 (plain nn.Conv2d)     1    39.4us    44.9us   0.88x
    conv_short (1x1x1)               1    26.1us    30.8us   0.85x

The win is not really "FlyDSL beats torch at convolution". It is that the module
convolves a 5-D tensor whose time extent is 1, and torch's conv3d is poor at that
shape. Dropping to a 2-D convolution is most of the gain; FlyDSL supplies the
rest. Where the model already calls a 2-D convolution -- the ``resample``
downsamplers -- Lumen is *slower* at 256x256, so those are left alone. Same for
the 1x1 convolutions, which are launch-bound.

The T=1 degeneracy
------------------
``QwenImageCausalConv3d`` pads time asymmetrically (``kT-1`` zero frames in
front, none behind) and then convolves with ``padding=0``. With one real frame,
every kernel slice but the last multiplies zeros, so

    conv3d(pad(x), w)  ==  conv2d(x, w[:, :, -1])

exactly -- not approximately. Symmetric ``padding=1`` on a 3-D convolution would
*not* be causal and would give a different answer. Anything that breaks the
assumption (a real video with T > 1, an inference feature cache, a strided or
dilated time axis) falls back to the module's own forward.
"""

import types

import torch


def _is_causal_conv3d(m):
    return type(m).__name__ == "QwenImageCausalConv3d"


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


def patch_vae_convs(vae, min_spatial_kernel=2, verbose=None):
    """Rebind the forward of every eligible VAE convolution. Returns a summary.

    ``min_spatial_kernel`` skips pointwise convolutions, where the Lumen path
    measures slower than torch. Only ``forward`` is rebound; parameters, dtypes
    and the module tree are untouched, which is what keeps this usable under
    ``init_device=meta`` and FSDP2.
    """
    import lumen.ops.conv as conv_ops

    stats = {"patched": [], "skipped": []}

    for name, m in vae.named_modules():
        if not _is_causal_conv3d(m):
            continue
        spatial_k = max(m.kernel_size[1], m.kernel_size[2])
        if spatial_k < min_spatial_kernel:
            stats["skipped"].append((name, f"pointwise {m.kernel_size}"))
            continue
        spatial_pad = _degenerate_to_conv2d(m)
        if spatial_pad is None:
            stats["skipped"].append((name, "not reducible to conv2d"))
            continue

        # The VAE is frozen, so the last time-slice of the kernel is constant for
        # the life of the run and worth materialising once instead of per call.
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
        stats["patched"].append((name, tuple(m.kernel_size), tuple(m.stride)))

    if verbose:
        verbose(f"vae_conv: patched {len(stats['patched'])} QwenImageCausalConv3d, skipped {len(stats['skipped'])}")
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
