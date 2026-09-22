"""Compare native fixed-top-k AITER and FlyDSL SonicMoE training paths.

This benchmark intentionally runs before Lumen's EP dispatcher.  Both backends
consume the same ``x``, fixed top-k ids/scores, logical weights, and ``dout``;
both include their own fixed-top-k sorting, expert forward, and backward.

The default is FlyDSL's measured Qwen3-30B-A3B forward profile:
T=4096, H=2048, I=768, E=128, K=8.  This is a single-GPU kernel comparison,
not the Lumen EP=8 local workload (E=16, one already-routed row per token).

Run inside the Lumen image with FlyDSL on PYTHONPATH:

    python tests/kernels/bench_sonic_moe_native_aiter_vs_flydsl.py
    python tests/kernels/bench_sonic_moe_native_aiter_vs_flydsl.py --tokens 8192
"""

from __future__ import annotations

import argparse
import math
import statistics

import torch


HIDDEN = 2048
INTERMEDIATE = 768
EXPERTS = 128
TOPK = 8
WARMUP = 3
ITERS = 10


def _median_ms(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def once() -> float:
        torch.cuda.synchronize()
        begin.record()
        result = fn()
        end.record()
        end.synchronize()
        del result
        return begin.elapsed_time(end)

    for _ in range(warmup):
        once()
    return statistics.median(once() for _ in range(iters))


def _fixed_topk_routing(tokens: int, device: torch.device):
    # Deterministic, exactly balanced routing with distinct ids per token.
    slots = torch.arange(TOPK, device=device, dtype=torch.int32)
    bases = (torch.arange(tokens, device=device, dtype=torch.int32) * TOPK) % EXPERTS
    ids = (bases[:, None] + slots[None, :]).remainder(EXPERTS).contiguous()
    scores = torch.rand(tokens, TOPK, device=device, dtype=torch.float32)
    scores /= scores.sum(dim=-1, keepdim=True)
    return ids, scores


def _aiter_fixed_topk(
    x: torch.Tensor,
    ids: torch.Tensor,
    scores: torch.Tensor,
    w1_kn: torch.Tensor,
    w2_kn: torch.Tensor,
) -> torch.Tensor:
    """AITER fixed-top-k path without router-logit matmul/softmax."""
    from aiter.ops.triton._triton_kernels.moe.sonicmoe import (
        _DownProjection,
        _UpProjection,
    )
    from aiter.ops.triton._triton_kernels.moe.sonicmoe.enums import ActivationType
    from aiter.ops.triton._triton_kernels.moe.sonicmoe.routing import (
        TC_topk_router_metadata_triton,
    )

    tokens, topk = ids.shape
    routes = tokens * topk
    device = x.device
    frequency = torch.empty(EXPERTS, dtype=torch.int32, device=device)
    offsets = torch.empty(EXPERTS + 1, dtype=torch.int32, device=device)
    gather = torch.empty(routes, dtype=torch.int32, device=device)
    score_scatter = torch.empty(routes, dtype=torch.int32, device=device)
    score_reverse = torch.empty(routes, dtype=torch.int32, device=device)
    TC_topk_router_metadata_triton(
        ids,
        EXPERTS,
        frequency,
        offsets,
        gather,
        score_scatter,
        score_reverse,
    )
    activation, preactivation = _UpProjection.apply(
        x,
        w1_kn,
        None,
        offsets,
        routes,
        topk,
        gather,
        score_scatter,
        score_reverse,
        None,
        False,
        ActivationType.SWIGLU,
        False,
        True,   # Megatron/Qwen W1 is [gate half | up half].
        True,   # grouped weight layout [E, K, N].
        False,
    )
    return _DownProjection.apply(
        activation,
        preactivation,
        w2_kn,
        None,
        scores,
        offsets,
        tokens,
        topk,
        gather,
        score_scatter,
        score_reverse,
        None,
        False,
        ActivationType.SWIGLU,
        True,
        True,
    )


def _flydsl_config():
    from kernels.moe.sonic import SonicMoEConfig

    # Exact measured dense Qwen3 profile from
    # default_sonic_moe_candidates(), not Lumen's former BM64 wrapper.
    return SonicMoEConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts=EXPERTS,
        top_k=TOPK,
        tile_m=128,
        tile_n=192,
        tile_k=64,
        down_tile_m=64,
        down_tile_n=256,
        down_tile_k=64,
        renormalize=False,
        stage1_xcd_swizzle=8,
        stage1_k_wave=1,
        stage2_xcd_swizzle=0,
        stage2_pipeline_stages=None,
        stage1_write_padded_rows=True,
        stage1_lds_swizzle=True,
        activation="swiglu",
        compute_dtype="bf16",
    )


def run(tokens: int, warmup: int, iters: int) -> None:
    from kernels.moe.sonic import SonicMoE, prepare_sonic_bf16_weights
    from kernels.moe.sonic_backward import sonic_moe_backward

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU required")
    device = torch.device("cuda")
    torch.manual_seed(20260913)

    x = torch.randn(tokens, HIDDEN, device=device, dtype=torch.bfloat16)
    dout = torch.randn_like(x)
    w1 = (
        torch.randn(
            EXPERTS,
            2 * INTERMEDIATE,
            HIDDEN,
            device=device,
            dtype=torch.float32,
        )
        / math.sqrt(HIDDEN)
    ).to(torch.bfloat16)
    w2 = (
        torch.randn(
            EXPERTS,
            HIDDEN,
            INTERMEDIATE,
            device=device,
            dtype=torch.float32,
        )
        / math.sqrt(INTERMEDIATE)
    ).to(torch.bfloat16)
    ids, scores = _fixed_topk_routing(tokens, device)

    # AITER grouped weights are [E,K,N]. Keep these as independent leaves so
    # timing includes the same dx/dw1/dw2/dscore products as FlyDSL.
    aiter_w1 = w1.transpose(1, 2).contiguous()
    aiter_w2 = w2.transpose(1, 2).contiguous()

    config = _flydsl_config()
    flydsl = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    flydsl.reserve(tokens)

    def aiter_forward():
        with torch.no_grad():
            return _aiter_fixed_topk(x, ids, scores, aiter_w1, aiter_w2)

    def flydsl_forward():
        with torch.no_grad():
            return flydsl.forward_topk_training(x, ids, scores)

    def aiter_e2e():
        xa = x.detach().requires_grad_(True)
        s = scores.detach().requires_grad_(True)
        aw1 = aiter_w1.detach().requires_grad_(True)
        aw2 = aiter_w2.detach().requires_grad_(True)
        out = _aiter_fixed_topk(xa, ids, s, aw1, aw2)
        return torch.autograd.grad(out, (xa, aw1, aw2, s), dout)

    def flydsl_e2e():
        out, state = flydsl.forward_topk_training(x, ids, scores)
        del out
        return sonic_moe_backward(
            x,
            w1,
            w2,
            ids,
            scores,
            dout,
            config,
            forward_state=state,
        )

    def aiter_backward_once() -> float:
        xa = x.detach().requires_grad_(True)
        s = scores.detach().requires_grad_(True)
        aw1 = aiter_w1.detach().requires_grad_(True)
        aw2 = aiter_w2.detach().requires_grad_(True)
        out = _aiter_fixed_topk(xa, ids, s, aw1, aw2)
        torch.cuda.synchronize()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        grads = torch.autograd.grad(out, (xa, aw1, aw2, s), dout)
        end.record()
        end.synchronize()
        del grads
        return begin.elapsed_time(end)

    def flydsl_backward_once() -> float:
        _, state = flydsl.forward_topk_training(x, ids, scores)
        torch.cuda.synchronize()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        grads = sonic_moe_backward(
            x,
            w1,
            w2,
            ids,
            scores,
            dout,
            config,
            forward_state=state,
        )
        end.record()
        end.synchronize()
        del grads, state
        return begin.elapsed_time(end)

    # Compile all paths before checking or timing.
    aiter_e2e()
    flydsl_e2e()
    torch.cuda.synchronize()

    with torch.no_grad():
        aiter_out = aiter_forward()
        flydsl_out, _ = flydsl_forward()
        diff = (aiter_out.float() - flydsl_out.float()).abs()
        denom = aiter_out.float().norm().clamp_min(1e-12)
        rel_l2 = float(diff.norm() / denom)
        max_abs = float(diff.max())
        del aiter_out, flydsl_out, diff

    for _ in range(warmup):
        aiter_backward_once()
        flydsl_backward_once()
    aiter_bwd = statistics.median(aiter_backward_once() for _ in range(iters))
    flydsl_bwd = statistics.median(flydsl_backward_once() for _ in range(iters))
    aiter_fwd = _median_ms(aiter_forward, warmup, iters)
    flydsl_fwd = _median_ms(flydsl_forward, warmup, iters)
    aiter_total = _median_ms(aiter_e2e, warmup, iters)
    flydsl_total = _median_ms(flydsl_e2e, warmup, iters)

    print(
        f"\nNative fixed-top-k SonicMoE T={tokens} H={HIDDEN} I={INTERMEDIATE} "
        f"E={EXPERTS} K={TOPK} BF16"
    )
    print(f"routes={tokens * TOPK}; balanced rows/expert={(tokens * TOPK) // EXPERTS}")
    print(f"forward agreement: rel_l2={rel_l2:.4e}, max_abs={max_abs:.4e}")
    print("                 forward   backward        e2e")
    print(f"AITER Triton   {aiter_fwd:8.2f}  {aiter_bwd:9.2f}  {aiter_total:9.2f} ms")
    print(f"FlyDSL tuned   {flydsl_fwd:8.2f}  {flydsl_bwd:9.2f}  {flydsl_total:9.2f} ms")
    print(
        f"FlyDSL/AITER   {flydsl_fwd / aiter_fwd:8.2f}x "
        f"{flydsl_bwd / aiter_bwd:9.2f}x {flydsl_total / aiter_total:9.2f}x"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    args = parser.parse_args()
    run(args.tokens, args.warmup, args.iters)
