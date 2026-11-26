###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kimi-K3 Tiny-L13 (``Qinghao/Kimi-K3-Tiny-L13``) KDA weights as aiter ``KDALayerWeights``.

Tiny-L13 keeps the real K3 widths (hidden 7168, 96 KDA heads x 128, conv width
4, F-decay rank 128) and has 13 decoder layers. ``linear_attn_config`` in its
``config.json`` is 1-indexed: KDA layers [1,2,3,5,6,7,9,10,11] and gated-MLA
layers [4,8,12,13], i.e. 0-indexed KDA 0-2, 4-6, 8-10 and MLA 3, 7, 11, 12.

Checkpoint layout:

* ``A_log`` is one decay per head, padded to [128] for 96 heads; aiter#8739
  (bbed79a+) reads ``A_log[:heads]``.
* The config sets ``gate_lower_bound = -5``: the safe gate
  ``g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))``, passed
  to ``kda_layer_fwd(lower_bound=...)``.
* Conv weights, ``A_log``, ``dt_bias`` and ``o_norm`` are fp32 and stay fp32.

The checkpoint directory comes from ``path``, ``LUMEN_KIMI_TINY_L13_PATH`` or
``AITER_KIMI_TINY_L13_PATH`` (a local snapshot, e.g. from
``huggingface-cli download Qinghao/Kimi-K3-Tiny-L13``). Without one, weights
are random at the same shapes.
"""

import json
import os

import torch

TINY_L13_REPO = "Qinghao/Kimi-K3-Tiny-L13"
TINY_L13_PATH_ENVS = ("LUMEN_KIMI_TINY_L13_PATH", "AITER_KIMI_TINY_L13_PATH")

HIDDEN, HEADS, HEAD_DIM, CONV, GATE_RANK = 7168, 96, 128, 4, 128
A_LOG_LEN = 128
FP32_FIELDS = ("conv_q", "conv_k", "conv_v", "A_log", "dt_bias", "norm_w")
NUM_LAYERS = 13
KDA_LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10)
MLA_LAYERS = (3, 7, 11, 12)
RMS_NORM_EPS = 1e-5
GATE_LOWER_BOUND = -5.0

HF_PREFIX = "language_model.model.layers.{}.self_attn."
HF_KEYS = {
    "w_q": "q_proj.weight",
    "w_k": "k_proj.weight",
    "w_v": "v_proj.weight",
    "conv_q": "q_conv1d.weight",
    "conv_k": "k_conv1d.weight",
    "conv_v": "v_conv1d.weight",
    "f_a": "f_a_proj.weight",
    "f_b": "f_b_proj.weight",
    "A_log": "A_log",
    "dt_bias": "dt_bias",
    "w_beta": "b_proj.weight",
    "g_proj": "g_proj.weight",
    "norm_w": "o_norm.weight",
    "w_o": "o_proj.weight",
}


def checkpoint_path(path=None):
    if path:
        return path
    for env in TINY_L13_PATH_ENVS:
        if os.environ.get(env):
            return os.environ[env]
    return None


def _kda_cfg_from_text_config(tc):
    """KDA geometry from a K3 / K3.1 ``text_config``; layer lists in config are 1-indexed."""
    la = tc["linear_attn_config"]
    return dict(
        hidden=tc["hidden_size"],
        heads=la["num_heads"],
        head_dim=la["head_dim"],
        conv=la["short_conv_kernel_size"],
        gate_rank=la["head_dim"],
        num_layers=tc["num_hidden_layers"],
        kda_layers=tuple(i - 1 for i in la["kda_layers"]),
        mla_layers=tuple(i - 1 for i in la["full_attn_layers"]),
        eps=tc.get("rms_norm_eps", RMS_NORM_EPS),
        gate_lower_bound=la.get("gate_lower_bound"),
        full_rank_gate=la.get("use_full_rank_gate", False),
        vocab=tc.get("vocab_size"),
    )


# moonshotai/Kimi-K3.1-Pro config.json: 93 layers, KDA at every layer except 4k (1-indexed) and 93;
# the full-attention layers are DSA (dsa_config.dsa_layers), not gated MLA. KDA widths match K3.
_K31_FULL_ATTN = tuple(range(4, 93, 4)) + (93,)
K31_PRO_TEXT_CONFIG = {
    "hidden_size": 7168,
    "num_hidden_layers": 93,
    "vocab_size": 163840,
    "rms_norm_eps": 1e-5,
    "linear_attn_config": {
        "full_attn_layers": list(_K31_FULL_ATTN),
        "kda_layers": [i for i in range(1, 94) if i not in _K31_FULL_ATTN],
        "gate_lower_bound": -5.0,
        "head_dim": 128,
        "num_heads": 96,
        "short_conv_kernel_size": 4,
        "use_full_rank_gate": True,
    },
}
PRESETS = ("k3-tiny-l13", "k3.1-pro")


def load_config(path=None, preset="k3-tiny-l13"):
    """KDA geometry and layer split from ``path``'s config.json, else from ``preset``."""
    path = checkpoint_path(path) if preset == "k3-tiny-l13" else path
    if path is None and preset == "k3.1-pro":
        return _kda_cfg_from_text_config(K31_PRO_TEXT_CONFIG)
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}, expected one of {PRESETS}")
    cfg = {
        "hidden": HIDDEN,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "conv": CONV,
        "gate_rank": GATE_RANK,
        "num_layers": NUM_LAYERS,
        "kda_layers": KDA_LAYERS,
        "mla_layers": MLA_LAYERS,
        "eps": RMS_NORM_EPS,
        "gate_lower_bound": GATE_LOWER_BOUND,
        "full_rank_gate": True,
        "vocab": 163840,
    }
    if path is None:
        return cfg
    with open(os.path.join(path, "config.json")) as f:
        tc = json.load(f)
    cfg.update(_kda_cfg_from_text_config(tc.get("text_config", tc)))
    return cfg


def expected_shapes(cfg):
    D, H, K, W, R = (cfg[n] for n in ("hidden", "heads", "head_dim", "conv", "gate_rank"))
    HK = H * K
    return {
        "w_q": (HK, D), "w_k": (HK, D), "w_v": (HK, D),
        "conv_q": (HK, W), "conv_k": (HK, W), "conv_v": (HK, W),
        "f_a": (R, D), "f_b": (HK, R), "A_log": (max(H, A_LOG_LEN),), "dt_bias": (HK,),
        "w_beta": (H, D), "g_proj": (HK, D), "norm_w": (K,), "w_o": (D, HK),
    }  # fmt: skip


def _hf_init_a_log(heads, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.log(torch.empty(heads).uniform_(1, 16, generator=g))


def random_tensors(cfg, dtype, seed=0):
    """{field: tensor} at Tiny-L13 shapes, scaled like the HF init."""
    shapes = expected_shapes(cfg)
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for name, shape in shapes.items():
        if name == "A_log":
            out[name] = _hf_init_a_log(shape[0], seed)
        elif name == "norm_w":
            out[name] = 1.0 + torch.randn(shape, generator=g) * 0.1
        elif name.startswith("conv"):
            out[name] = torch.randn(shape, generator=g) * 0.5
        elif name == "dt_bias":
            out[name] = torch.randn(shape, generator=g)
        else:
            out[name] = torch.randn(shape, generator=g) * shape[1] ** -0.5
    return {n: t if n in FP32_FIELDS else t.to(dtype) for n, t in out.items()}


def _read(path, names):
    from safetensors import safe_open

    idx = os.path.join(path, "model.safetensors.index.json")
    wmap = json.load(open(idx))["weight_map"] if os.path.exists(idx) else None
    by_file = {}
    for n in names:
        by_file.setdefault(wmap[n] if wmap is not None else "model.safetensors", []).append(n)
    out = {}
    for fname, ns in by_file.items():
        with safe_open(os.path.join(path, fname), framework="pt", device="cpu") as f:
            for n in ns:
                out[n] = f.get_tensor(n)
    return out


def load_kda_tensors(layer, path=None, cfg=None):
    """{field: CPU tensor} for 0-indexed decoder ``layer``, in checkpoint dtype."""
    cfg = cfg or load_config(path)
    if layer not in cfg["kda_layers"]:
        raise ValueError(f"layer {layer} is not a KDA layer; KDA layers are {cfg['kda_layers']}")
    path = checkpoint_path(path)
    if path is None:
        return random_tensors(cfg, torch.float32, seed=layer)
    names = {field: HF_PREFIX.format(layer) + key for field, key in HF_KEYS.items()}
    raw = _read(path, names.values())
    ts = {field: raw[name] for field, name in names.items()}
    for c in ("conv_q", "conv_k", "conv_v"):
        ts[c] = ts[c].reshape(ts[c].shape[0], -1)  # [C, 1, W] -> [C, W]
    ts["A_log"] = ts["A_log"].reshape(-1)
    for n, want in expected_shapes(cfg).items():
        if n == "A_log":
            if ts[n].numel() < cfg["heads"]:
                raise ValueError(f"{names[n]}: {ts[n].numel()} values, expected >= {cfg['heads']} (one per head)")
        elif tuple(ts[n].shape) != want:
            raise ValueError(f"{names[n]}: shape {tuple(ts[n].shape)}, expected {want}")
    return ts


def load_kda_layer(layer, path=None, dtype=torch.bfloat16, device="cuda", cfg=None):
    """aiter ``KDALayerWeights`` for 0-indexed decoder ``layer``; ``FP32_FIELDS`` stay fp32."""
    from aiter.ops.kda_fused_layer import KDALayerWeights

    ts = load_kda_tensors(layer, path, cfg)
    return KDALayerWeights(
        **{
            n: t.to(device=device, dtype=torch.float32 if n in FP32_FIELDS else dtype).contiguous()
            for n, t in ts.items()
        }
    )


def load_hf_model(path=None, device="cuda"):
    """Full remote-code Tiny-L13 (``trust_remote_code``), checkpoint dtypes kept.

    The remote code initializes ``A_log`` as [num_heads] while the checkpoint
    stores it padded to [128], so the model is loaded with
    ``ignore_mismatched_sizes`` and the checkpoint ``A_log`` is restored as fp32.
    """
    import transformers

    path = checkpoint_path(path)
    if path is None:
        raise ValueError(f"no Tiny-L13 checkpoint; set one of {TINY_L13_PATH_ENVS}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype="auto", device_map=device, ignore_mismatched_sizes=True
    )
    cfg = load_config(path)
    names = {i: HF_PREFIX.format(i) + "A_log" for i in cfg["kda_layers"]}
    raw = _read(path, names.values())
    for mod in model.modules():
        if type(mod).__name__ == "KimiDeltaAttention":
            t = raw[names[mod.layer_idx]].reshape(-1).float().to(mod.A_log.device)
            mod.A_log = torch.nn.Parameter(t)
    return model
