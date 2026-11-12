"""SimWAM's Wan2.2 VAE (WanVideoVAE38) at NAVSIM's shapes with LumenConfig(aiter_conv=True).

    SIMWAM_DIR=... WAN22_VAE=... HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python -m pytest -q test_wan22_vae.py

CPU: the model's structure and shapes, and which modules the patch binds. GPU
(aiter with Triton conv3d): the patch against the stock model at rtol = atol =
1e-5, with every call counted as aiter or torch so that a silent fallback cannot
pass. Everything here skips without a SimWAM checkout.
"""

import copy
import importlib.util
import os
import sys
import types

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wan22_vae as W  # noqa: E402

_LUMEN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RTOL = ATOL = 1e-5


def _simwam():
    pytest.importorskip("einops")
    try:
        return W.wan_video_vae()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _lumen_gpu():
    """``lumen.ops.conv`` with aiter's Triton conv3d behind it, or skip."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    try:
        import lumen.ops.conv as lc
    except ImportError as exc:
        pytest.skip(f"lumen/aiter not importable: {exc}")
    if lc._conv3d() is None:
        pytest.skip("aiter Triton conv3d unavailable (install aiter >= c8325e00c or set AITER_SRC)")
    return lc


def _weights():
    path = os.environ.get("WAN22_VAE", "/work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    if not os.path.isfile(path):
        pytest.skip(f"{path} not found (Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth)")
    return path


# --- CPU -------------------------------------------------------------------------


def test_structure_and_shapes_match_navsim():
    """Full width on the meta device: SimWAM NAVSIM's VAE and its shapes."""
    V = _simwam()
    with torch.device("meta"):
        vae = V.WanVideoVAE38()
    m = vae.model
    assert sum(p.numel() for p in vae.parameters()) == 704_688_668  # 0.70B
    assert (vae.z_dim, vae.upsampling_factor, vae.temporal_downsample_factor) == (48, 16, 4)
    # 16x spatial = patchify 2 (12 input channels) and three 2x downsamples.
    assert m.encoder.conv1.in_channels == 12
    assert sum(isinstance(r, V.Resample) and r.mode.startswith("downsample") for r in m.encoder.modules()) == 3
    assert sum(type(c) is V.CausalConv3d for c in vae.modules()) == 62
    assert sum(type(c) is nn.Conv2d for c in vae.modules()) == 10

    flops = []
    for c in W.convs(vae):

        def _conv_forward(self, x, w, b, _orig=c._conv_forward):
            y = _orig(x, w, b)
            flops.append(y.numel() * w[0].numel() * 2)
            return y

        c._conv_forward = types.MethodType(_conv_forward, c)

    def run(fn, shape):
        flops.clear()
        with torch.no_grad():
            out = fn(torch.empty(shape, device="meta"), vae.scale)
        return tuple(out.shape), len(flops), sum(flops)

    # Convolution calls and FLOPs of SimWAM's own trace of this model.
    assert run(m.encode, W.NAVSIM_VIDEO) == (W.NAVSIM_LATENT, 92, 8_404_617_166_848)
    assert run(m.decode, W.NAVSIM_LATENT) == (W.NAVSIM_VIDEO, 116, 43_850_007_355_392)
    assert W.dit_tokens(W.NAVSIM_LATENT) == (12 * 21, 756)
    assert len(W.conv_signatures(vae)) == 58


def _patch_conv_without_aiter(monkeypatch):
    """``lumen.ops.conv.patch_conv`` -- what ``LumenConfig(aiter_conv=True)`` calls --
    on a machine without aiter. ``import lumen`` needs aiter and a GPU, so the file
    is loaded on its own, and aiter's two conv modules are stand-ins that raise if
    called: on CPU tensors Lumen never calls them."""

    def unavailable(*args, **kwargs):
        raise RuntimeError("aiter stand-in")

    for name in ("aiter", "aiter.ops", "aiter.ops.triton", "aiter.ops.triton.conv"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    for name in ("conv2d", "conv3d"):
        mod = types.ModuleType(f"aiter.ops.triton.conv.{name}")
        setattr(mod, name, unavailable)
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    spec = importlib.util.spec_from_file_location("_lumen_ops_conv", os.path.join(_LUMEN_ROOT, "lumen/ops/conv.py"))
    lc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lc)
    return lc


def test_patch_binds_every_conv_and_keeps_causal_forward(monkeypatch):
    """Which modules the patch takes, on CPU. Numerics are the GPU tests' job: on
    CPU every call is torch either way."""
    V = _simwam()
    lc = _patch_conv_without_aiter(monkeypatch)
    torch.manual_seed(0)
    vae = V.VideoVAE38_(dim=16, dec_dim=16).eval()
    assert lc.patch_conv(vae) == (10, 62)
    assert all(getattr(c._conv_forward, "_lumen_conv", False) for c in W.convs(vae))
    assert all(type(c).forward is V.CausalConv3d.forward for c in vae.modules() if type(c) is V.CausalConv3d)
    with W.ConvCalls(vae, lc) as calls, torch.no_grad():
        vae.decode(vae.encode(torch.rand(1, 3, 9, 64, 96) * 2 - 1, [torch.zeros(48), torch.ones(48)]),
                   [torch.zeros(48), torch.ones(48)])  # fmt: skip
    assert calls.as_dict() == {"aiter": 0, "torch": 92 + 116}


# --- GPU: the aiter path in BF16 against the stock model, rtol = atol = 1e-5 -----


@pytest.fixture(scope="module")
def navsim_bf16():
    """Stock and patched BF16 VAE with the real weights, and the stock outputs."""
    lc = _lumen_gpu()
    _simwam()
    from lumen.config import LumenConfig

    base = W.load(_weights(), dtype=torch.bfloat16, device="cpu")
    stock = copy.deepcopy(base).cuda()
    patched = copy.deepcopy(base).cuda()
    LumenConfig(aiter_conv=True).enable(patched)
    torch.manual_seed(0)
    x = (torch.rand(W.NAVSIM_VIDEO, device="cuda") * 2 - 1).bfloat16()
    z = W.encode(stock, x)
    video = W.decode(stock, z)
    assert tuple(z.shape) == W.NAVSIM_LATENT and tuple(video.shape) == W.NAVSIM_VIDEO
    return lc, stock, patched, x, z, video


def test_bf16_encode_matches_stock(navsim_bf16):
    lc, _, patched, x, z, _ = navsim_bf16
    with W.ConvCalls(patched, lc) as calls:
        got = W.encode(patched, x)
    assert calls.as_dict() == {"aiter": 92, "torch": 0}
    torch.testing.assert_close(got, z, rtol=RTOL, atol=ATOL)


def test_bf16_decode_matches_stock(navsim_bf16):
    lc, _, patched, _, z, video = navsim_bf16
    with W.ConvCalls(patched, lc) as calls:
        got = W.decode(patched, z)
    assert calls.as_dict() == {"aiter": 116, "torch": 0}
    torch.testing.assert_close(got, video, rtol=RTOL, atol=ATOL)


def test_aiter_kernels_exact_on_navsim_conv_shapes():
    """Each of the 58 distinct NAVSIM convolutions on {-1, 0, 1} BF16 data, where the
    exact answer is known (wan22_vae.ternary_conv_check). Non-Winograd routes must
    match the CPU fp32 reference at rtol = atol = 1e-5; Winograd routes are not
    exact by construction and are not asserted on."""
    lc = _lumen_gpu()
    V = _simwam()
    with torch.device("meta"):
        sigs = W.conv_signatures(V.WanVideoVAE38())
    gen = torch.Generator().manual_seed(0)
    failed, winograd = [], []
    for sig in sigs:
        route = W.aiter_route(sig)
        aiter_calls, d, got, ref = W.ternary_conv_check(sig, lc, gen)
        assert aiter_calls == 1, (sig, route, "fell back to torch")
        if W.is_winograd(route):
            winograd.append((sig, route, d))
            continue
        try:
            torch.testing.assert_close(got, ref, rtol=RTOL, atol=ATOL)
        except AssertionError:
            failed.append((sig, route, d))
    for sig, route, d in winograd:
        print(f"not asserted (Winograd): {route} {sig} max|d| {d:.3e}")
    assert not failed, failed
