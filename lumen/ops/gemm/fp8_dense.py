###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Blockwise FP8 dense linears for HF Qwen3 training (q/k/v, o, gate/up, down).

q/k/v and gate/up each run as one GEMM on the concatenated weight. The weight
is quantized with ATOM's 128x128 function once per optimizer step and reused by
every forward and DGrad until the next ``optimizer.step()``; a global
optimizer post-step hook bumps the generation and frees the cache.

Forward kernels (``LUMEN_FP8_DENSE_FWD``):
  ``triton``      aiter ``gemm_a8w8_blockscale``, the call ATOM rollout makes
                  with ATOM_USE_TRITON_GEMM=1, on bit-identical tiles from
                  ``configs/fp8_dense_triton_gfx950.json``. Use it whenever
                  training logprobs must match rollout.
  ``ck``          aiter CK ``gemm_a8w8_blockscale``.
  ``bpreshuffle`` aiter blockscale bpreshuffle on a cached (16, 16)-shuffled
                  weight, kernels from ``configs/fp8_dense_bpreshuffle_gfx950.csv``
                  or ``LUMEN_FP8_DENSE_CK_TABLE``. Every forward-shape row is a
                  CK bpreshuffle or cktile 192x256x128_4x2x1 kernel, which all
                  give the same bits at any M; with the table shared by ATOM
                  rollout (ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=1,
                  AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE) training gets
                  the rollout bits, faster than ``triton``.
DGrad uses ``LUMEN_FP8_DENSE_DGRAD`` (same choices). WGrad
(``LUMEN_FP8_DENSE_WGRAD``) is ``fp8`` (Triton blockscale on the 128-row
aligned token prefix, BF16 for the remaining < 128 rows) or ``bf16``.
"""

from __future__ import annotations

import logging
import os
import weakref

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

logger = logging.getLogger(__name__)

_GEN = [0]
_HOOK = [None]
_KINDS = ("triton", "ck", "bpreshuffle")
_OWNERS = weakref.WeakSet()


def bump_generation(*_args, **_kwargs) -> None:
    """Invalidate every cached FP8 weight and free it (each holds a full,
    unsharded copy of its weight), so rollout never runs next to the cache."""
    _GEN[0] += 1
    for owner in list(_OWNERS):
        owner.__dict__.pop("_lumen_fp8_dense_w", None)
    _OWNERS.clear()


def _ensure_hook() -> None:
    if _HOOK[0] is None:
        from torch.optim.optimizer import register_optimizer_step_post_hook

        _HOOK[0] = register_optimizer_step_post_hook(bump_generation)


def _fp8():
    from aiter import dtypes

    return dtypes.fp8


def _act_quant(x2):
    from aiter import QuantType, get_hip_quant

    return get_hip_quant(QuantType.per_1x128)(x2, quant_dtype=_fp8(), transpose_scale=False)


def _weight_quant(w):
    from atom.quantization.quark.utils import quantize_weight_to_fp8_128x128_blockscale

    return quantize_weight_to_fp8_128x128_blockscale(w.detach(), _fp8())


@triton.jit
def _store_tile(P, tile, row0, col0, width, SHUF: tl.constexpr):
    """Store a 128x128 FP8 tile at (row0, col0) of a row-major [*, width] matrix,
    plain or in aiter's shuffle_weight(layout=(16, 16)) order.

    In the shuffle each 16-row group of a 128-column span is one contiguous
    2048-byte run, so the tile is permuted in registers and stored as 8 runs.
    """
    if SHUF:
        runs = tl.reshape(tl.permute(tl.reshape(tile, (8, 16, 4, 2, 16)), (0, 2, 3, 1, 4)), (8, 2048))
        g16 = row0 // 16 + tl.arange(0, 8)
        off = (g16[:, None] * (width // 32) + col0 // 32) * 512 + tl.arange(0, 2048)[None, :]
    else:
        runs = tile
        off = (row0 + tl.arange(0, 128))[:, None] * width + col0 + tl.arange(0, 128)[None, :]
    tl.store(P + off, runs)


@triton.jit
def _round_e4m3(x):
    """Round fp32 (|x| <= 448) to the nearest-even e4m3fn value, in fp32, so the
    following cast is exact (a direct cast rounds twice)."""
    e = ((x.to(tl.uint32, bitcast=True) >> 23) & 0xFF).to(tl.int32) - 127
    e = tl.maximum(e, -6) - 3
    step = ((e + 127) << 23).to(tl.uint32).to(tl.float32, bitcast=True)
    inv_step = ((127 - e) << 23).to(tl.uint32).to(tl.float32, bitcast=True)
    return libdevice.nearbyint(x * inv_step) * step


@triton.jit
def _quant_w128_kernel(W, Q, T, S, ST, stride_w, K, NT, row_off,
                       Q_SHUF: tl.constexpr, T_SHUF: tl.constexpr):
    """ATOM's quantize_weight_to_fp8_128x128_blockscale on one 128x128 tile of
    one piece of a row-stacked weight, bit for bit: ``amax / 448`` is torch's
    multiply by the fp32 reciprocal, ``w / scale`` an IEEE division, then the
    clamp and a round-to-nearest-even cast.

    Writes the FP8 tile and its transpose, each plain or shuffled, and both
    scale layouts.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    r = pid_n * 128 + tl.arange(0, 128)
    c = pid_k * 128 + tl.arange(0, 128)
    w = tl.load(W + r[:, None] * stride_w + c[None, :]).to(tl.float32)
    amax = tl.max(tl.max(tl.abs(w), axis=1), axis=0)
    s = tl.maximum(amax * tl.math.div_rn(1.0, 448.0), 1.1754943508222875e-38)
    q = _round_e4m3(tl.clamp(tl.math.div_rn(w, s), -448.0, 448.0)).to(Q.dtype.element_ty)
    row0 = row_off + pid_n * 128
    _store_tile(Q, q, row0, pid_k * 128, K, Q_SHUF)
    _store_tile(T, tl.trans(q), pid_k * 128, row0, NT, T_SHUF)
    nb = row0 // 128
    tl.store(S + nb * (K // 128) + pid_k, s)
    tl.store(ST + pid_k * (NT // 128) + nb, s)


def _quant_weights(weights, q_shuf, t_shuf):
    """(q, s, t, t_s) of the row-stacked weights; q is [N, K], t is q^T [K, N]."""
    k = weights[0].shape[1]
    nt = sum(w.shape[0] for w in weights)
    dev = weights[0].device
    fp8 = _fp8()
    q = torch.empty(nt, k, dtype=fp8, device=dev)
    t = torch.empty(k, nt, dtype=fp8, device=dev)
    s = torch.empty(nt // 128, k // 128, dtype=torch.float32, device=dev)
    t_s = torch.empty(k // 128, nt // 128, dtype=torch.float32, device=dev)
    off = 0
    for w in weights:
        n = w.shape[0]
        if n % 128 or k % 128 or w.stride(1) != 1:
            raise RuntimeError(f"FP8 dense weight quant needs 128-aligned, row-major pieces, got {tuple(w.shape)}")
        _quant_w128_kernel[(n // 128, k // 128)](w, q, t, s, t_s, w.stride(0), k, nt, off,
                                                 Q_SHUF=q_shuf, T_SHUF=t_shuf, num_warps=8)
        off += n
    return q, s, t, t_s


class _WeightEntry:
    __slots__ = ("gen", "q", "s", "q_sh", "t", "t_s", "t_sh")

    def __init__(self, gen, weights, kinds):
        nograd_kind, dgrad_kind, _, grad_kind = kinds
        fwd_kinds = {nograd_kind, grad_kind}
        q_shuf = fwd_kinds == {"bpreshuffle"}
        t_shuf = dgrad_kind == "bpreshuffle"
        q, self.s, t, self.t_s = _quant_weights(weights, q_shuf, t_shuf)
        self.gen = gen
        self.q, self.q_sh = (None, q) if q_shuf else (q, None)
        self.t, self.t_sh = (None, t) if t_shuf else (t, None)

    def fwd(self, kind):
        if kind == "bpreshuffle" and self.q_sh is None:
            from aiter.ops.shuffle import shuffle_weight

            self.q_sh = shuffle_weight(self.q, layout=(16, 16))
        return (self.q_sh if kind == "bpreshuffle" else self.q), self.s

    def bwd(self, kind):
        return (self.t_sh if kind == "bpreshuffle" else self.t), self.t_s


def _weight_entry(owner, weights, kinds) -> _WeightEntry:
    entry = owner.__dict__.get("_lumen_fp8_dense_w")
    if entry is None or entry.gen != _GEN[0]:
        with torch.no_grad():
            entry = _WeightEntry(_GEN[0], [w.detach() for w in weights], kinds)
        owner.__dict__["_lumen_fp8_dense_w"] = entry
        _OWNERS.add(owner)
    return entry


_TRITON_TABLE = [None]


def _triton_config(m, n, k):
    """Training-only tile for the Triton blockscale GEMM, or None for aiter's.

    Every entry keeps NUM_KSPLIT=1, BLOCK_SIZE_K=128 and aiter's MFMA shape, and
    was checked to give the same bits as aiter's own config, so a row equals
    the row rollout computes at any M.
    """
    if _TRITON_TABLE[0] is None:
        table = {}
        if os.environ.get("LUMEN_FP8_DENSE_TRITON_TUNED", "1") == "1":
            import json

            path = os.path.join(os.path.dirname(__file__), "configs", "fp8_dense_triton_gfx950.json")
            from aiter.ops.triton.utils._triton import arch_info

            if arch_info.get_arch() == "gfx950" and os.path.exists(path):
                with open(path) as f:
                    table = json.load(f)
        _TRITON_TABLE[0] = table
    by_m = _TRITON_TABLE[0].get(f"{n},{k}")
    if not by_m or m < 1024:
        return None
    return dict(by_m["4096" if m > 2048 else "2048"])


def _gemm(kind, a_q, a_s, b, b_s):
    """Y[M, N] = A[M, K] @ B[N, K]^T with 1x128 A scales and 128x128 B scales."""
    if kind == "triton":
        from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale

        config = _triton_config(a_q.shape[0], b.shape[0], a_q.shape[1])
        return gemm_a8w8_blockscale(a_q, b, a_s, b_s, dtype=torch.bfloat16, config=config)
    import aiter

    if kind == "ck":
        return aiter.gemm_a8w8_blockscale(a_q, b, a_s, b_s, torch.bfloat16)
    a_s_t = a_s.t().contiguous().view(*a_s.shape)
    m, n, k = a_q.shape[0], b.shape[0], a_q.shape[1]
    config = _bpreshuffle_config(m, n, k)
    if config is None:
        return aiter.gemm_a8w8_blockscale_bpreshuffle(a_q, b, a_s_t, b_s, torch.bfloat16)
    from aiter.ops import gemm_op_a8w8 as ops

    y = torch.empty(m, n, dtype=torch.bfloat16, device=a_q.device)
    lib, name = config["libtype"], str(config["kernelName"])
    if lib == "cktile":
        return ops.gemm_a8w8_blockscale_bpreshuffle_cktile(a_q, b, a_s_t, b_s, y, kernelName=name)
    if lib == "ck":
        return ops.gemm_a8w8_blockscale_bpreshuffle_ck(a_q, b, a_s_t, b_s, y, kernelName=name)
    return ops.gemm_a8w8_blockscale_bpreshuffle_asm(
        a_q, b, y, a_s_t, b_s, splitK=int(config["splitK"]), kernelName=name
    )


_BPRESHUFFLE = {}


def _bpreshuffle_config(m, n, k):
    """Kernel from Lumen's training table (Qwen3-8B shapes), kept out of aiter's
    global table so rollout keeps its own kernels.

    ``LUMEN_FP8_DENSE_CK_TABLE`` replaces the table.
    """
    key = (m, n, k)
    if key not in _BPRESHUFFLE:
        config = None
        path = os.environ.get("LUMEN_FP8_DENSE_CK_TABLE") or os.path.join(
            os.path.dirname(__file__), "configs", "fp8_dense_bpreshuffle_gfx950.csv")
        if os.environ.get("LUMEN_FP8_DENSE_CK_TUNED", "1") == "1" and os.path.exists(path):
            from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config

            config = get_CKGEMM_config(m, n, k, path)
            if config is not None and config["libtype"] not in ("cktile", "ck", "asm"):
                config = None
        _BPRESHUFFLE[key] = config
    return _BPRESHUFFLE[key]


def _wgrad_fp8(g2, x2):
    """dW[N, K] = g2[M, N]^T @ x2[M, K]; FP8 on the first M//128*128 rows."""
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale

    from lumen.ops.quantize.fast_transpose import fast_transpose_fp8
    from lumen.ops.quantize.ops import quant_fp8_blockwise_impl

    m = g2.shape[0]
    ma = m // 128 * 128
    if ma == 0:
        return g2.t().mm(x2)
    fp8 = _fp8()
    g_col, g_s = quant_fp8_blockwise_impl(g2[:ma], dtype=fp8, axis=0, block_size=128)
    x_col, x_s = quant_fp8_blockwise_impl(x2[:ma], dtype=fp8, axis=0, block_size=128)
    dw = gemm_a8w8_blockscale(
        fast_transpose_fp8(g_col),
        fast_transpose_fp8(x_col),
        g_s.t().contiguous(),
        x_s.t().contiguous(),
        dtype=torch.bfloat16,
    )
    if ma < m:
        dw.addmm_(g2[ma:].t(), x2[ma:])
    return dw


class FP8DenseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, owner, kinds, fwd_kind, *weights):
        k = x.shape[-1]
        x2 = x.reshape(-1, k)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        entry = _weight_entry(owner, weights, kinds)
        x_q, x_s = _act_quant(x2)
        b, b_s = entry.fwd(fwd_kind)
        y = _gemm(fwd_kind, x_q, x_s, b, b_s)
        ctx.save_for_backward(x2)
        ctx.entry = entry
        ctx.kinds = kinds
        ctx.sizes = tuple(w.shape[0] for w in weights)
        ctx.x_shape = x.shape
        ctx.w_dtype = weights[0].dtype
        return y.view(*x.shape[:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, grad_out):
        (x2,) = ctx.saved_tensors
        _, dgrad_kind, wgrad_kind, _ = ctx.kinds
        g2 = grad_out.reshape(-1, grad_out.shape[-1])
        if not g2.is_contiguous():
            g2 = g2.contiguous()
        if g2.dtype != torch.bfloat16:
            g2 = g2.bfloat16()
        g_q, g_s = _act_quant(g2)
        b, b_s = ctx.entry.bwd(dgrad_kind)
        dx = _gemm(dgrad_kind, g_q, g_s, b, b_s).view(ctx.x_shape)
        dw = _wgrad_fp8(g2, x2) if wgrad_kind == "fp8" else g2.t().mm(x2)
        if dw.dtype != ctx.w_dtype:
            dw = dw.to(ctx.w_dtype)
        grads = dw.split(ctx.sizes, dim=0) if len(ctx.sizes) > 1 else (dw,)
        return (dx, None, None, None, *grads)


def _dense(x, owner, kinds, *weights):
    fwd_kind = kinds[3] if torch.is_grad_enabled() else kinds[0]
    return FP8DenseFn.apply(x, owner, kinds, fwd_kind, *weights)


def _kinds(forward_kind=None):
    """(no-grad forward, DGrad, WGrad, grad-mode forward) kernels.

    ``LUMEN_FP8_DENSE_FWD_GRAD`` picks the forward of the actor update apart
    from the no-grad forward (old logprobs); it defaults to the same kernel.
    """
    fwd = os.environ.get("LUMEN_FP8_DENSE_FWD") or forward_kind or "triton"
    dgrad = os.environ.get("LUMEN_FP8_DENSE_DGRAD", "triton")
    wgrad = os.environ.get("LUMEN_FP8_DENSE_WGRAD", "fp8")
    grad_fwd = os.environ.get("LUMEN_FP8_DENSE_FWD_GRAD") or fwd
    if any(kd not in _KINDS for kd in (fwd, dgrad, grad_fwd)) or wgrad not in ("fp8", "bf16"):
        raise ValueError(
            f"bad FP8 dense kernels fwd={fwd} dgrad={dgrad} wgrad={wgrad} fwd_grad={grad_fwd}"
        )
    return fwd, dgrad, wgrad, grad_fwd


def _no_bias(*mods):
    for mod in mods:
        if getattr(mod, "bias", None) is not None:
            raise RuntimeError(f"FP8 dense linear needs bias-free projections ({type(mod).__name__})")


def _fused_silu(mlp):
    """The fused silu_and_mul Function if Lumen's MLP activation patch replaced
    this MLP's forward, else None. It takes the fused [gate | up] output as is."""
    fwd = mlp.__dict__.get("forward")
    if getattr(fwd, "__module__", "") != "lumen.ops.mlp.mlp_activation_patch":
        return None
    from lumen.ops.silu_and_mul import silu_and_mul_fn

    return silu_and_mul_fn()


def patch_qwen3_fp8_dense(model, *, forward_kind=None) -> int:
    """Route Qwen3 attention and MLP projections through :class:`FP8DenseFn`."""
    kinds = _kinds(forward_kind)
    _ensure_hook()
    count = 0
    for module in model.modules():
        name = type(module).__name__
        if name in ("Qwen3Attention", "Qwen3MoeAttention") and all(
            hasattr(module, p) for p in ("q_proj", "k_proj", "v_proj", "o_proj")
        ):
            q_mod, k_mod, v_mod, o_mod = module.q_proj, module.k_proj, module.v_proj, module.o_proj
            _no_bias(q_mod, k_mod, v_mod, o_mod)
            pending = {}

            def q_forward(hidden, q_mod=q_mod, k_mod=k_mod, v_mod=v_mod, pending=pending):
                fused = _dense(hidden, q_mod, kinds, q_mod.weight, k_mod.weight, v_mod.weight)
                q, k, v = fused.split(
                    (q_mod.weight.shape[0], k_mod.weight.shape[0], v_mod.weight.shape[0]), dim=-1
                )
                pending["k"], pending["v"] = (hidden, k), (hidden, v)
                return q

            def k_forward(hidden, pending=pending):
                src, k = pending.pop("k")
                if src is not hidden:
                    raise RuntimeError("fused FP8 qkv: k_proj input differs from q_proj input")
                return k

            def v_forward(hidden, pending=pending):
                src, v = pending.pop("v")
                if src is not hidden:
                    raise RuntimeError("fused FP8 qkv: v_proj input differs from q_proj input")
                return v

            def o_forward(hidden, o_mod=o_mod):
                return _dense(hidden, o_mod, kinds, o_mod.weight)

            q_mod.forward, k_mod.forward, v_mod.forward = q_forward, k_forward, v_forward
            o_mod.forward = o_forward
            count += 1
        elif name == "Qwen3MLP" and all(hasattr(module, p) for p in ("gate_proj", "up_proj", "down_proj")):
            gate_mod, up_mod, down_mod = module.gate_proj, module.up_proj, module.down_proj
            _no_bias(gate_mod, up_mod, down_mod)

            def mlp_forward(hidden, module=module, silu=_fused_silu(module)):
                gate_up = _dense(hidden, module.gate_proj, kinds, module.gate_proj.weight, module.up_proj.weight)
                if silu is not None:
                    act = silu.apply(gate_up)
                else:
                    gate, up = gate_up.split(
                        (module.gate_proj.weight.shape[0], module.up_proj.weight.shape[0]), dim=-1)
                    act = module.act_fn(gate) * up
                return _dense(act, module.down_proj, kinds, module.down_proj.weight)

            module.forward = mlp_forward
            count += 1
    logger.info(
        "FP8 dense linears on %d Qwen3 attention/MLP modules (fwd=%s dgrad=%s wgrad=%s fwd_grad=%s)",
        count, *kinds,
    )
    return count
