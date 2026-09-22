"""Microbench AITER SonicMoE vs FlyDSL pre-routed experts on Qwen3 shapes.

One GPU, EP=8 local experts (E=16). Default T matches one training microbatch
after EP all-to-all: MBS=2, seq=4096, topk=8 → T = 2*4096*8/8 = 8192.

    python tests/kernels/bench_sonic_moe_aiter_vs_flydsl.py
    python tests/kernels/bench_sonic_moe_aiter_vs_flydsl.py --tk 32768
    python tests/kernels/bench_sonic_moe_aiter_vs_flydsl.py --sweep-tiles
    python tests/kernels/bench_sonic_moe_aiter_vs_flydsl.py --trace-regions
    pytest tests/kernels/bench_sonic_moe_aiter_vs_flydsl.py -s
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import os
import statistics
from functools import lru_cache
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from lumen.modules.sonic_moe import SonicMoEExperts


E = 16
H = 2048
I = 768
WARMUP = 5
ITERS = 20
DEFAULT_TK = 2 * 4096 * 8 // 8  # MBS=2, seq=4096, topk=8, EP=8


def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    return torch.device("cuda")


@lru_cache(maxsize=1)
def load_roctx():
    candidates = [
        "/opt/rocm/lib/librocprofiler-sdk-roctx.so",
        *glob.glob(
            "/opt/venv/lib/python*/site-packages/"
            "_rocm_sdk_devel/lib/librocprofiler-sdk-roctx.so"
        ),
        *glob.glob("/opt/rocm/lib/librocprofiler-sdk-roctx.so*"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError("librocprofiler-sdk-roctx.so not found")
    lib = ctypes.CDLL(path)
    lib.roctxProfilerResume.argtypes = [ctypes.c_uint64]
    lib.roctxProfilerPause.argtypes = [ctypes.c_uint64]
    lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
    return lib


def trace_one(name, fn):
    torch.cuda.synchronize()
    roctx = load_roctx()

    assert roctx.roctxProfilerResume(0) == 0
    roctx.roctxRangePushA(name.encode())
    try:
        result = fn()
        torch.cuda.synchronize()
    finally:
        roctx.roctxRangePop()
        assert roctx.roctxProfilerPause(0) == 0

    return result


def _median_ms(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def one():
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    for _ in range(warmup):
        one()
    return statistics.median(one() for _ in range(iters))


def _make_counts(tokens: int, num_experts: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(num_experts, generator=g)
    probs = torch.softmax(logits, dim=0)
    assignments = torch.multinomial(probs, tokens, replacement=True, generator=g)
    counts = torch.bincount(assignments, minlength=num_experts).to(torch.int32)
    leftover = int(tokens - int(counts.sum()))
    if leftover:
        counts[int(counts.argmax())] += leftover
    return counts


class _Linear(nn.Module):
    def __init__(self, out_features: int, in_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))


class _GroupedExperts(nn.Module):
    def __init__(self, num_experts: int, hidden: int, intermediate: int):
        super().__init__()
        self.config = SimpleNamespace(
            add_bias_linear=False,
            gated_linear_unit=True,
            expert_tensor_parallel_size=1,
        )
        self.num_local_experts = num_experts
        self.linear_fc1 = nn.Module()
        self.linear_fc2 = nn.Module()
        for index in range(num_experts):
            self.linear_fc1.register_parameter(
                f"weight{index}",
                nn.Parameter(torch.empty(2 * intermediate, hidden)),
            )
            self.linear_fc2.register_parameter(
                f"weight{index}",
                nn.Parameter(torch.empty(hidden, intermediate)),
            )


def _build_module(backend: str, device: torch.device) -> SonicMoEExperts:
    os.environ["SONIC_MOE_GEMM_BACKEND"] = backend
    os.environ.setdefault("SONIC_MOE_GROUPED_GEMM_BACKEND", "triton")
    os.environ.setdefault("SONIC_MOE_USE_QWEN3_TUNED_GEMM", "1")
    torch.manual_seed(0)
    experts = _GroupedExperts(E, H, I)
    with torch.no_grad():
        scale1 = H**-0.5
        scale2 = I**-0.5
        for index in range(E):
            getattr(experts.linear_fc1, f"weight{index}").normal_(std=scale1)
            getattr(experts.linear_fc2, f"weight{index}").normal_(std=scale2)
    module = SonicMoEExperts(experts).to(device=device, dtype=torch.bfloat16)
    return module


def _inputs(tokens: int, device: torch.device):
    counts = _make_counts(tokens, E).to(device)
    hidden = torch.randn(
        tokens, H, dtype=torch.bfloat16, device=device, requires_grad=True
    )
    scores = torch.rand(tokens, dtype=torch.float32, device=device, requires_grad=True)
    grad = torch.randn(tokens, H, dtype=torch.bfloat16, device=device)
    return counts, hidden, scores, grad


def _bench_module(name: str, module: SonicMoEExperts, counts, hidden, scores, grad):
    def fwd():
        with torch.no_grad():
            module(hidden.detach(), counts, scores.detach())

    def e2e():
        module.zero_grad(set_to_none=True)
        hidden.grad = None
        scores.grad = None
        out, _ = module(hidden, counts, scores)
        out.backward(grad)

    out, _ = module(hidden, counts, scores)
    out.backward(grad)

    def bwd():
        module.zero_grad(set_to_none=True)
        hidden.grad = None
        scores.grad = None
        fresh, _ = module(hidden, counts, scores)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fresh.backward(grad)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    for _ in range(WARMUP):
        bwd()
    bwd_ms = statistics.median(bwd() for _ in range(ITERS))
    fwd_ms = _median_ms(fwd)
    e2e_ms = _median_ms(e2e)
    print(
        f"  {name:8s}  fwd {fwd_ms:7.2f} ms   bwd {bwd_ms:7.2f} ms   "
        f"e2e {e2e_ms:7.2f} ms"
    )
    return {"fwd": fwd_ms, "bwd": bwd_ms, "e2e": e2e_ms}


def _bench_flydsl_parts(counts, hidden, scores, w1, w2):
    from lumen.ops.moe.flydsl_grouped import (
        build_pre_routed_metadata,
        flydsl_grouped_dgrad,
        flydsl_grouped_wgrad,
        flydsl_pre_routed,
        _cu_seqlens_from_counts,
        _preshuffled_weights,
    )

    tokens = hidden.shape[0]
    cu = _cu_seqlens_from_counts(counts, E, hidden.device)
    dh = torch.randn(tokens, 2 * I, dtype=torch.bfloat16, device=hidden.device)
    dy = torch.randn(tokens, H, dtype=torch.bfloat16, device=hidden.device)
    dx = torch.empty(tokens, H, dtype=torch.bfloat16, device=hidden.device)
    da = torch.empty(tokens, I, dtype=torch.bfloat16, device=hidden.device)
    dw1 = torch.empty_like(w1)
    dw2 = torch.empty_like(w2)
    a_prime = torch.randn(tokens, I, dtype=torch.bfloat16, device=hidden.device)

    _preshuffled_weights(w1, w2)
    print("  flydsl parts (cached weights)")
    print(
        f"    metadata     {_median_ms(lambda: build_pre_routed_metadata(cu, tokens)):7.2f} ms"
    )
    print(
        f"    gemm1+gemm2  {_median_ms(lambda: flydsl_pre_routed(hidden.detach(), counts, scores.detach(), w1, w2)):7.2f} ms"
    )
    print(f"    up dgrad     {_median_ms(lambda: flydsl_grouped_dgrad(dh, w1, cu, dx)):7.2f} ms")
    print(f"    down dgrad   {_median_ms(lambda: flydsl_grouped_dgrad(dy, w2, cu, da)):7.2f} ms")
    print(f"    up wgrad     {_median_ms(lambda: flydsl_grouped_wgrad(hidden.detach(), dh, cu, dw1)):7.2f} ms")
    print(
        f"    down wgrad   {_median_ms(lambda: flydsl_grouped_wgrad(a_prime, dy, cu, dw2)):7.2f} ms"
    )


def _bench_aiter_gemms(counts, hidden, w1, w2):
    from aiter.ops.triton._triton_kernels.moe.sonicmoe.grouped_gemm_triton import (
        grouped_gemm,
    )

    tokens = hidden.shape[0]
    cu = torch.zeros(E + 1, dtype=torch.int32, device=hidden.device)
    cu[1:] = counts.to(hidden.device).cumsum(0)
    h = torch.randn(tokens, 2 * I, dtype=torch.bfloat16, device=hidden.device)
    a = torch.randn(tokens, I, dtype=torch.bfloat16, device=hidden.device)
    y = torch.randn(tokens, H, dtype=torch.bfloat16, device=hidden.device)
    os.environ["SONIC_MOE_GEMM_BACKEND"] = "triton"
    os.environ["SONIC_MOE_GROUPED_GEMM_BACKEND"] = "triton"
    os.environ["SONIC_MOE_USE_QWEN3_TUNED_GEMM"] = "1"
    print("  aiter grouped GEMMs (triton tuned)")
    specs = (
        ("up fwd", lambda: grouped_gemm(hidden.detach(), w1, cu)),
        ("down fwd", lambda: grouped_gemm(a, w2, cu)),
        ("up dgrad", lambda: grouped_gemm(h, w1, cu, B_is_transposed=True)),
        ("down dgrad", lambda: grouped_gemm(y, w2, cu, B_is_transposed=True)),
        ("up wgrad", lambda: grouped_gemm(hidden.detach(), h, cu, A_is_transposed=True)),
        ("down wgrad", lambda: grouped_gemm(a, y, cu, A_is_transposed=True)),
    )
    total = 0.0
    for name, fn in specs:
        ms = _median_ms(fn)
        total += ms
        print(f"    {name:10s}  {ms:7.2f} ms")
    print(f"    {'sum':10s}  {total:7.2f} ms")


SWEEP_WARMUP = 3
SWEEP_ITERS = 8

_FWD_GEMM1_BASE = (128, 128, 64, 8, 0)
_FWD_GEMM2_BASE = (64, 256, 64, 0, 0)
_DX_BASE = (128, 128, 64)
_DA_BASE = (128, 128, 64)
_DW1_BASE = (128, 128, 64, 0, 2, 2)
_DW2_BASE = (128, 128, 64, 0, 2, 2)


def _product(*axes):
    from itertools import product

    return list(product(*axes))


def _fwd_gemm1_candidates():
    tiles = []
    for bm, bn, bk in _product((64, 128), (128, 192), (64, 128)):
        tiles.append((bm, bn, bk, 8, 0))
    tiles.append((128, 192, 64, 0, 0))
    return list(dict.fromkeys(tiles))


def _fwd_gemm2_candidates():
    tiles = []
    for bm, bn, bk in _product((64, 128), (128, 256), (64, 128)):
        if (bm * bk) % 2048:
            continue
        tiles.append((bm, bn, bk, 0, 0))
    return list(dict.fromkeys(tiles))


def _nn_candidates(output_size: int, contraction_size: int):
    bns = [bn for bn in (128, 192, 256) if output_size % bn == 0]
    bks = [bk for bk in (64, 128) if contraction_size % bk == 0]
    tiles = []
    for bm, bn, bk in _product((64, 128), bns, bks):
        n_waves = 4 if bn >= 256 else 2
        if bn % (n_waves * 16):
            continue
        ab = 2 * (bm + bn) * bk * 2
        c = bm * bn * 2
        if max(ab, c) > 163840:
            continue
        tiles.append((bm, bn, bk))
    return tiles


def _tn_candidates(output_m: int, output_n: int):
    bms = [bm for bm in (128, 192, 256) if output_m % bm == 0]
    bns = [bn for bn in (128, 192, 256) if output_n % bn == 0]
    tiles = []
    for bm, bn, bk in _product(bms, bns, (64,)):
        m_waves = 4 if bm >= 256 else 2
        n_waves = 4 if bn >= 256 else 2
        if bm % (m_waves * 16) or bn % (n_waves * 16):
            continue
        tiles.append((bm, bn, bk, 0, m_waves, n_waves))
    return tiles


def _time_or_fail(fn, warmup: int = SWEEP_WARMUP, iters: int = SWEEP_ITERS):
    try:
        ms = _median_ms(fn, warmup=warmup, iters=iters)
        return ms, None
    except Exception as exc:  # noqa: BLE001 — compile/runtime skips are the point
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        return None, f"{type(exc).__name__}: {exc}"


def _print_sweep_table(title: str, rows: list[tuple], aiter_ms: float | None):
    print(f"\n  {title}", flush=True)
    print(
        f"    {'tiles':40s}  {'ms':>8s}  {'vs_base':>8s}  {'vs_aiter':>8s}",
        flush=True,
    )
    valid = [(tile, ms) for tile, ms, err in rows if ms is not None]
    if not valid:
        for tile, _, err in rows:
            print(f"    {str(tile):40s}  FAIL  {err}")
        return None
    best_ms = min(ms for _, ms in valid)
    base_ms = rows[0][1] if rows[0][1] is not None else best_ms
    for tile, ms, err in rows:
        if ms is None:
            print(f"    {str(tile):40s}  {'FAIL':>8s}  {err}")
            continue
        vs_base = ms / base_ms
        vs_aiter = f"{ms / aiter_ms:8.2f}" if aiter_ms else f"{'n/a':>8s}"
        mark = " *" if ms == best_ms else ""
        print(
            f"    {str(tile):40s}  {ms:8.2f}  {vs_base:8.2f}  {vs_aiter}{mark}",
            flush=True,
        )
    winner = min(valid, key=lambda item: item[1])
    print(f"    winner {winner[0]}  {winner[1]:.2f} ms")
    return winner


def run_tile_sweep(tokens: int) -> None:
    from lumen.ops.moe.flydsl_grouped import (
        flydsl_grouped_dgrad,
        flydsl_grouped_wgrad,
        flydsl_pre_routed,
        flydsl_tile_override,
        _cu_seqlens_from_counts,
        _preshuffled_weights,
    )

    device = _require_gpu()
    counts, hidden, scores, _grad = _inputs(tokens, device)
    module = _build_module("flydsl", device)
    w1 = module.w1.detach()
    w2 = module.w2.detach()
    _preshuffled_weights(w1, w2)
    cu = _cu_seqlens_from_counts(counts, E, device)
    hidden_d = hidden.detach()
    scores_d = scores.detach()
    dh = torch.randn(tokens, 2 * I, dtype=torch.bfloat16, device=device)
    dy = torch.randn(tokens, H, dtype=torch.bfloat16, device=device)
    dx = torch.empty(tokens, H, dtype=torch.bfloat16, device=device)
    da = torch.empty(tokens, I, dtype=torch.bfloat16, device=device)
    dw1 = torch.empty_like(w1)
    dw2 = torch.empty_like(w2)
    a_prime = torch.randn(tokens, I, dtype=torch.bfloat16, device=device)

    print(
        f"\nFlyDSL tile sweep  E={E} T={tokens} H={H} I={I}  "
        f"counts min={int(counts.min())} max={int(counts.max())}",
        flush=True,
    )
    print("First row of each table is the current wrap default.", flush=True)

    aiter_times = {}
    try:
        from aiter.ops.triton._triton_kernels.moe.sonicmoe.grouped_gemm_triton import (
            grouped_gemm,
        )

        os.environ["SONIC_MOE_USE_QWEN3_TUNED_GEMM"] = "1"
        a_act = torch.randn(tokens, I, dtype=torch.bfloat16, device=device)
        h_act = torch.randn(tokens, 2 * I, dtype=torch.bfloat16, device=device)
        y_act = torch.randn(tokens, H, dtype=torch.bfloat16, device=device)
        aiter_times["up dgrad"] = _median_ms(
            lambda: grouped_gemm(h_act, w1, cu, B_is_transposed=True)
        )
        aiter_times["down dgrad"] = _median_ms(
            lambda: grouped_gemm(y_act, w2, cu, B_is_transposed=True)
        )
        aiter_times["up wgrad"] = _median_ms(
            lambda: grouped_gemm(hidden_d, h_act, cu, A_is_transposed=True)
        )
        aiter_times["down wgrad"] = _median_ms(
            lambda: grouped_gemm(a_act, y_act, cu, A_is_transposed=True)
        )
        def aiter_fwd():
            grouped_gemm(hidden_d, w1, cu)
            grouped_gemm(a_act, w2, cu)

        aiter_times["up+down fwd"] = _median_ms(aiter_fwd)
        print("  AITER reference")
        for name, ms in aiter_times.items():
            print(f"    {name:12s} {ms:7.2f} ms")
    except Exception as exc:  # noqa: BLE001
        print(f"  AITER reference unavailable: {exc}")

    def sweep(op, base, candidates, launch, aiter_key):
        ordered = [base] + [tile for tile in candidates if tile != base]
        rows = []
        for tile in ordered:
            with flydsl_tile_override(**{op: tile}):
                ms, err = _time_or_fail(launch)
            rows.append((tile, ms, err))
        return _print_sweep_table(op, rows, aiter_times.get(aiter_key))

    sweep(
        "gemm1",
        _FWD_GEMM1_BASE,
        _fwd_gemm1_candidates(),
        lambda: flydsl_pre_routed(hidden_d, counts, scores_d, w1, w2),
        "up+down fwd",
    )
    sweep(
        "gemm2",
        _FWD_GEMM2_BASE,
        _fwd_gemm2_candidates(),
        lambda: flydsl_pre_routed(hidden_d, counts, scores_d, w1, w2),
        "up+down fwd",
    )
    sweep(
        "dx",
        _DX_BASE,
        _nn_candidates(H, 2 * I),
        lambda: flydsl_grouped_dgrad(dh, w1, cu, dx),
        "up dgrad",
    )
    sweep(
        "da",
        _DA_BASE,
        _nn_candidates(I, H),
        lambda: flydsl_grouped_dgrad(dy, w2, cu, da),
        "down dgrad",
    )
    sweep(
        "dw1",
        _DW1_BASE,
        _tn_candidates(H, 2 * I),
        lambda: flydsl_grouped_wgrad(hidden_d, dh, cu, dw1),
        "up wgrad",
    )
    sweep(
        "dw2",
        _DW2_BASE,
        _tn_candidates(I, H),
        lambda: flydsl_grouped_wgrad(a_prime, dy, cu, dw2),
        "down wgrad",
    )


def run_bench(tokens: int) -> None:
    device = _require_gpu()
    counts, hidden, scores, grad = _inputs(tokens, device)
    print(
        f"\nSonicMoE pre-routed  E={E} T={tokens} H={H} I={I} BF16  "
        f"tokens/expert min={int(counts.min())} max={int(counts.max())}"
    )
    print(
        "Projected step ≈ 48 layers × 128 microbatches × e2e  "
        f"(GBS=256 / MBS=2)"
    )

    aiter = _build_module("triton", device)
    hidden_a = hidden.detach().clone().requires_grad_(True)
    scores_a = scores.detach().clone().requires_grad_(True)
    aiter_times = _bench_module("aiter", aiter, counts, hidden_a, scores_a, grad)

    flydsl = _build_module("flydsl", device)
    hidden_f = hidden.detach().clone().requires_grad_(True)
    scores_f = scores.detach().clone().requires_grad_(True)
    flydsl_times = _bench_module("flydsl", flydsl, counts, hidden_f, scores_f, grad)

    ratio = flydsl_times["e2e"] / aiter_times["e2e"]
    print(f"  flydsl / aiter e2e  {ratio:5.2f}x")
    layers, microbatches = 48, 128
    print(
        f"  projected step  aiter {aiter_times['e2e'] * layers * microbatches / 1000:6.2f} s   "
        f"flydsl {flydsl_times['e2e'] * layers * microbatches / 1000:6.2f} s"
    )

    _bench_aiter_gemms(counts, hidden, aiter.w1.detach(), aiter.w2.detach())
    if flydsl.native_weight_layout:
        print(
            "  flydsl parts: native retained-state lifecycle; "
            "standalone grouped-GEMM decomposition is not applicable"
        )
    else:
        _bench_flydsl_parts(
            counts, hidden_f.detach(), scores_f.detach(), flydsl.w1, flydsl.w2
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_bench_sonic_moe_aiter_vs_flydsl():
    try:
        from kernels.moe.sonic_grouped_a16w16 import compile_sonic_grouped_a16w16_nn
    except ImportError:
        pytest.skip("FlyDSL is not available")
    del compile_sonic_grouped_a16w16_nn
    run_bench(DEFAULT_TK)


def run_roctx_trace(tokens: int) -> None:
    """One ROCTx-selected forward and one backward after compile + warmup.

    Uses the process ``SONIC_MOE_GEMM_BACKEND`` and wrap tiles as-is.  Does not
    time under rocprof; call ``run_bench`` without a profiler for numbers.
    """
    device = _require_gpu()
    backend = os.environ.get("SONIC_MOE_GEMM_BACKEND", "flydsl")
    print(f"ROCTx trace backend={backend} T={tokens} WARMUP={WARMUP}", flush=True)
    load_roctx()

    counts, hidden, scores, grad = _inputs(tokens, device)
    module = _build_module(backend, device)

    def run_forward():
        with torch.no_grad():
            return module(hidden.detach(), counts, scores.detach())

    run_forward()
    torch.cuda.synchronize()
    for _ in range(WARMUP):
        run_forward()
    torch.cuda.synchronize()
    print("tracing flydsl_forward", flush=True)
    trace_one("flydsl_forward", run_forward)

    output, _ = module(hidden, counts, scores)
    torch.cuda.synchronize()

    def run_backward():
        return torch.autograd.grad(
            output,
            (hidden, module.w1, module.w2, scores),
            grad,
            retain_graph=True,
        )

    for _ in range(WARMUP):
        run_backward()
    torch.cuda.synchronize()
    print("tracing flydsl_backward", flush=True)
    trace_one("flydsl_backward", run_backward)
    print("ROCTx regions done", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tk", type=int, default=DEFAULT_TK)
    parser.add_argument(
        "--sweep-tiles",
        action="store_true",
        help="Sweep FlyDSL gemm1/gemm2/NN/TN tiles on this shape",
    )
    parser.add_argument(
        "--trace-regions",
        action="store_true",
        help="ROCTx-selected one forward and one backward after warmup",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP required")
    if args.sweep_tiles:
        run_tile_sweep(args.tk)
    elif args.trace_regions:
        run_roctx_trace(args.tk)
    else:
        run_bench(args.tk)
