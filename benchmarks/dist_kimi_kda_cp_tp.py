###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CP2-vs-CP1 and TP2-vs-TP1 for the KDA recurrence under Megatron's ``kda_backend="aiter"`` contract.

Every rank calls ``aiter.ops.kda.chunk_kda`` with Megatron's keywords (pre-normalized q/k, raw gate,
fp32 post-sigmoid beta, safe gate -5, A_log/dt_bias in the kernel), as ``KimiDeltaAttention``
does with ``gdn_pre_gated_delta_rule_fusion``:

* CP: packed sequences split contiguously across the group (``build_kda_cp_context``); outputs and
  input grads gathered and compared with one rank running the unsplit batch.
* TP: heads sharded across the group (Megatron's head partition); compared with all heads on one rank.
* Loss curve: 32 SGD steps on the q/k/v/g inputs with the #6872 continuation lr; per-step
  |loss_CP2 - loss_CP1| <= 1e-3.

    torchrun --nproc_per_node 2 benchmarks/dist_kimi_kda_cp_tp.py [--heads 96 --seq 4096]
"""

import argparse
import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

TOL_REL = 2e-2  # bf16 recurrence, different chunk boundaries per rank
LOSS_TOL = 1e-3  # Megatron-LM#6872 continuation protocol


def megatron_kw(A_log, dt_bias):
    return dict(
        A_log=A_log, dt_bias=dt_bias, use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0,
        use_qk_l2norm_in_kernel=False, output_final_state=False,
    )  # fmt: skip


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def inputs(T, H, K, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, device=device)  # noqa: E731
    q = F.normalize(r(1, T, H, K), dim=-1).bfloat16()
    k = F.normalize(r(1, T, H, K), dim=-1).bfloat16()
    v = r(1, T, H, K).bfloat16()
    graw = r(1, T, H, K).bfloat16()
    beta = torch.sigmoid(r(1, T, H))
    A_log = torch.log(torch.empty(H, device=device).uniform_(1, 16, generator=g))
    dt_bias = r(H * K)
    return q, k, v, graw, beta, A_log, dt_bias


def lr(step):
    return 1e-6 + 0.5 * (1e-5 - 1e-6) * (1 + math.cos(math.pi * min(step, 50) / 50))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--seqlens", default="700,900,448", help="packed sequence lengths (sum must equal --seq)")
    p.add_argument("--steps", type=int, default=32)
    args = p.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda")
    from aiter.ops.kda import build_kda_cp_context, chunk_kda

    T, H, K = args.seq, args.heads, args.head_dim
    lens = [int(x) for x in args.seqlens.split(",")]
    assert sum(lens) == T and T % world == 0 and H % world == 0
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device=dev, dtype=torch.int32)
    q, k, v, graw, beta, A_log, dt_bias = inputs(T, H, K, 0, dev)
    dy = torch.randn(1, T, H, K, device=dev, generator=torch.Generator(device=dev).manual_seed(1)).bfloat16()
    ok = True

    def leaves(*ts):
        return [t.detach().clone().requires_grad_(True) for t in ts]

    # CP1 reference (every rank, unsplit)
    lq, lk, lv, lg = leaves(q, k, v, graw)
    o1, _ = chunk_kda(lq, lk, lv, lg, beta, cu_seqlens=cu, **megatron_kw(A_log, dt_bias))
    o1.backward(dy)
    ref = [o1.detach(), lq.grad, lk.grad, lv.grad, lg.grad]

    # CP2: contiguous token split
    ctx = build_kda_cp_context(cu, dist.group.WORLD)
    part = T // world
    sl = slice(rank * part, (rank + 1) * part)
    cq, ck, cv, cg = leaves(q[:, sl], k[:, sl], v[:, sl], graw[:, sl])
    o2, _ = chunk_kda(cq, ck, cv, cg, beta[:, sl].contiguous(), cu_seqlens=ctx.cu_seqlens, cp_context=ctx,
                      **megatron_kw(A_log, dt_bias))  # fmt: skip
    o2.backward(dy[:, sl])
    for name, local, full in zip(["o", "dq", "dk", "dv", "dg"], [o2.detach(), cq.grad, ck.grad, cv.grad, cg.grad], ref):
        parts = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(parts, local.contiguous())
        e = rel(torch.cat(parts, 1), full)
        ok &= e < TOL_REL
        if rank == 0:
            print(f"CP{world} vs CP1 {name}: {e:.3e}")

    # TP2: head sharding
    hs = slice(rank * H // world, (rank + 1) * H // world)
    tq, tk, tv, tg = leaves(q[:, :, hs], k[:, :, hs], v[:, :, hs], graw[:, :, hs])
    kh = H // world
    o3, _ = chunk_kda(tq, tk, tv, tg, beta[:, :, hs].contiguous(), cu_seqlens=cu,
                      **megatron_kw(A_log[hs], dt_bias.view(H, K)[hs].reshape(kh * K)))  # fmt: skip
    o3.backward(dy[:, :, hs])
    for name, local, full in zip(["o", "dq", "dk", "dv", "dg"], [o3.detach(), tq.grad, tk.grad, tv.grad, tg.grad], ref):
        parts = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(parts, local.contiguous())
        e = rel(torch.cat(parts, 2), full)
        ok &= e < TOL_REL
        if rank == 0:
            print(f"TP{world} vs TP1 {name}: {e:.3e}")

    # loss curve, CP2 vs CP1, on the q/k/v/g inputs
    target = torch.randn(1, T, H, K, device=dev, generator=torch.Generator(device=dev).manual_seed(2)).bfloat16()
    curves = {}
    for mode in ("cp1", f"cp{world}"):
        params = leaves(q, k, v, graw)
        curve = []
        for step in range(args.steps):
            if mode == "cp1":
                o, _ = chunk_kda(*params, beta, cu_seqlens=cu, **megatron_kw(A_log, dt_bias))
                loss = (o.float() - target.float()).pow(2).mean()
            else:
                o, _ = chunk_kda(*(t[:, sl] for t in params), beta[:, sl].contiguous(), cu_seqlens=ctx.cu_seqlens,
                                 cp_context=ctx, **megatron_kw(A_log, dt_bias))  # fmt: skip
                local = (o.float() - target[:, sl].float()).pow(2).sum()
                loss = local / target.numel()
            loss.backward()
            total = loss.detach().clone()
            if mode != "cp1":
                dist.all_reduce(total)
                for t in params:
                    dist.all_reduce(t.grad)  # each rank holds the grads of its token slice
            curve.append(total.item())
            with torch.no_grad():
                for t in params:
                    t.sub_(lr(step) * t.grad * T)  # T rescales the per-element mean so the step is visible
                    t.grad = None
        curves[mode] = curve
    d = [abs(a - b) for a, b in zip(*curves.values())]
    ok &= max(d) <= LOSS_TOL
    if rank == 0:
        print("loss cp1:", [f"{x:.6f}" for x in curves["cp1"]])
        print(f"loss cp{world}:", [f"{x:.6f}" for x in curves[f"cp{world}"]])
        print(f"[primary] max |dloss| {max(d):.3e} (tol {LOSS_TOL:g})")
    flag = torch.tensor(int(ok), device=dev)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    dist.destroy_process_group()
    raise SystemExit(0 if flag.item() else 1)


if __name__ == "__main__":
    main()
