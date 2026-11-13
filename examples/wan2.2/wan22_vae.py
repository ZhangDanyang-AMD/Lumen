"""SimWAM's Wan2.2-TI2V-5B VAE (WanVideoVAE38) at its NAVSIM shapes, on Lumen's aiter conv.

The class is SimWAM's own, imported from ``$SIMWAM_DIR/src/simwam/models/wan22/
wan_video_vae.py`` (H-EmbodVis/SimWAM, MIT, pinned as ``SIMWAM_COMMIT``) rather
than copied here. That file needs only torch, einops and tqdm, so it is loaded on
its own and SimWAM's package, with its training dependencies, is never imported.
Weights load the way SimWAM's loader does (helpers/loader.py: the checkpoint's
keys under ``model.``), but strictly.

NAVSIM, as SimWAM's NAVSIM config runs it: 9 frames of 384x672 encode to a
48x3x24x42 latent -- temporal (9-1)/4+1, spatial 384/16 and 672/16, the 16 being
patchify 2 and three 2x downsamples -- and decode back to 9x384x672. The DiT then
patchifies the latent [1, 2, 2] (configs/model/simwam_navsim.yaml): 12x21 tokens
per latent frame, 756 in all.

The convolutions are patched with ``LumenConfig(aiter_conv=True).enable(vae)``,
which routes every ``nn.Conv2d`` / ``nn.Conv3d`` forward to aiter's Triton conv
and leaves ``CausalConv3d.forward`` (causal pad, feature cache) as it is. It
falls back to torch per call -- fp32, CPU, or a kernel error -- without saying
so beyond one warning, which is why :class:`ConvCalls` exists.
"""

import importlib.util
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

SIMWAM_COMMIT = "1433ee7e06a62bd8b89da74455df0d82bda7b726"
NAVSIM_VIDEO = (1, 3, 9, 384, 672)
NAVSIM_LATENT = (1, 48, 3, 24, 42)
DIT_PATCH = (1, 2, 2)

_MODULE = "simwam_wan_video_vae"


def wan_video_vae(simwam_dir=None):
    """SimWAM's ``wan_video_vae`` module."""
    if _MODULE in sys.modules:
        return sys.modules[_MODULE]
    root = simwam_dir or os.environ.get("SIMWAM_DIR", "/work/SimWAM")
    path = os.path.join(root, "src", "simwam", "models", "wan22", "wan_video_vae.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found; clone H-EmbodVis/SimWAM @ {SIMWAM_COMMIT} to $SIMWAM_DIR")
    spec = importlib.util.spec_from_file_location(_MODULE, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE] = mod
    spec.loader.exec_module(mod)
    return mod


def load(path=None, dtype=torch.float32, device="cuda"):
    """``WanVideoVAE38`` with the Wan2.2-TI2V-5B VAE weights, in eval mode."""
    path = path or os.environ.get("WAN22_VAE", "/work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    vae = wan_video_vae().WanVideoVAE38()
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("model_state", state)
    vae.load_state_dict({f"model.{k}": v for k, v in state.items()}, strict=True)
    return vae.to(device=device, dtype=dtype).eval().requires_grad_(False)


def encode(vae, video):
    """Normalised latent mean, as SimWAM's ``single_encode``."""
    with torch.no_grad():
        return vae.model.encode(video, vae.scale)


def decode(vae, latent):
    """Video in [-1, 1], as SimWAM's ``single_decode``."""
    with torch.no_grad():
        return vae.model.decode(latent, vae.scale).clamp_(-1, 1)


def dit_tokens(latent_shape, patch=DIT_PATCH):
    """(tokens per latent frame, total) after the DiT's patchify."""
    _, _, t, h, w = latent_shape
    per_frame = (h // patch[1]) * (w // patch[2])
    return per_frame, math.ceil(t / patch[0]) * per_frame


def convs(model):
    return [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Conv3d))]


class ConvCalls:
    """Count every convolution call in ``model`` as aiter or torch while active.

    ``total`` counts calls of each conv's ``_conv_forward``; ``aiter`` counts the
    aiter conv2d/conv3d calls Lumen's patch made and that returned. Whatever is
    left -- unpatched modules, Lumen's dtype/device fallback, a kernel that raised
    and fell back -- is ``torch``, so a silent fallback shows up here.
    """

    def __init__(self, model, lumen_conv):
        self.model, self.lc = model, lumen_conv
        self.total = self.aiter = 0

    @property
    def torch(self):
        return self.total - self.aiter

    def as_dict(self):
        return {"aiter": self.aiter, "torch": self.torch}

    def _counting(self, getter):
        def get():
            fn = getter()
            if fn is None:
                return None

            def run(*args, **kwargs):
                out = fn(*args, **kwargs)
                self.aiter += 1
                return out

            return run

        return get

    def __enter__(self):
        self._getters = (self.lc._conv2d, self.lc._conv3d)
        self.lc._conv2d = self._counting(self._getters[0])
        self.lc._conv3d = self._counting(self._getters[1])
        self._saved = []
        for m in convs(self.model):
            inner = m._conv_forward
            self._saved.append((m, m.__dict__.get("_conv_forward")))

            def counted(x, w, b, _inner=inner):
                self.total += 1
                return _inner(x, w, b)

            m._conv_forward = counted
        return self

    def __exit__(self, *exc):
        self.lc._conv2d, self.lc._conv3d = self._getters
        for m, own in self._saved:
            if own is None:
                del m._conv_forward
            else:
                m._conv_forward = own
        return False


def conv_signatures(vae):
    """Every distinct convolution one NAVSIM encode and one decode make, traced on meta.

    Each is ``(spatial_rank, x_shape, w_shape, stride, padding)`` as
    ``_conv_forward`` -- and so aiter -- receives it: after the causal pad and the
    cache concat.
    """
    with torch.device("meta"):
        meta = type(vae)()
    seen = {}
    for m in convs(meta):

        def record(self, x, w, b, _orig=m._conv_forward):
            key = (w.dim() - 2, tuple(x.shape), tuple(w.shape), tuple(self.stride), tuple(self.padding))
            seen.setdefault(key, None)
            return _orig(x, w, b)

        m._conv_forward = record.__get__(m)
    with torch.no_grad():
        z = meta.model.encode(torch.empty(NAVSIM_VIDEO, device="meta"), meta.scale)
        meta.model.decode(z, meta.scale)
    return list(seen)


def aiter_route(sig):
    """aiter's own route name for a signature, or None if it cannot be resolved."""
    rank, x, w, stride, padding = sig
    try:
        if rank == 3:
            from aiter.ops.triton.conv.conv3d import _resolve_route

            N, C, D, H, W = x
            return _resolve_route(*w[2:], stride, (1, 1, 1), N, C, D, H, W, w[0], "ncdhw", padding).name
        from aiter.ops.triton.conv.conv2d import _resolve_route

        N, C, H, W = x
        return _resolve_route(*w[2:], stride, (1, 1), N, C, H, W, w[0], "nchw", padding).name
    except Exception:  # private aiter API; the caller treats None as unknown
        return None


def is_winograd(route):
    return route is not None and ("WINOGRAD" in route or "WF4X3" in route)


def ternary_conv_check(sig, lumen_conv, generator):
    """One signature through Lumen's patched conv on {-1, 0, 1} BF16 data.

    Every product and partial sum is then an integer far below 2**24, so any
    direct, GEMM or implicit-GEMM kernel accumulating in fp32 gets the exact sum,
    and rounding it to BF16 is the same on both sides: the result has to equal a
    CPU fp32 reference bit for bit. A Winograd transform (fractional constants,
    an fp16 transform buffer for BF16 input) cannot be exact; :func:`aiter_route`
    says which calls take it. Returns (aiter_calls, max|d|, patched, reference).
    """
    from lumen.config import LumenConfig

    rank, x_shape, w_shape, stride, padding = sig
    Conv = nn.Conv3d if rank == 3 else nn.Conv2d
    conv = Conv(w_shape[1], w_shape[0], w_shape[2:], stride=stride, padding=padding)
    x = torch.randint(-1, 2, x_shape, generator=generator).to(torch.bfloat16)
    with torch.no_grad():
        conv.weight.copy_(torch.randint(-1, 2, w_shape, generator=generator))
        conv.bias.copy_(torch.randint(-1, 2, (w_shape[0],), generator=generator))
    ref = (F.conv3d if rank == 3 else F.conv2d)(x.float(), conv.weight, conv.bias, stride, padding).bfloat16()
    conv = conv.cuda().bfloat16()
    LumenConfig(aiter_conv=True).enable(conv)
    with ConvCalls(conv, lumen_conv) as calls, torch.no_grad():
        got = conv(x.cuda()).cpu()
    return calls.aiter, (got.float() - ref.float()).abs().max().item(), got, ref
