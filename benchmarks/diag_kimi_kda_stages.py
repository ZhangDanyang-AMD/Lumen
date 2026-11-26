###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Localize a KDA eager-vs-aiter mismatch on one layer, stage by stage.

Same weights and input on both sides. Reports ||aiter - eager|| / ||eager|| for:

1. forward stages: q/k after conv + l2norm, v after conv, gate g, beta, chunk o,
   gated-RMSNorm + o_proj output y;
2. the chunk forward piece alone, fed the eager q/k/v/g/beta;
3. every gradient (dx and the 14 weights) of the fused backward, once with the
   selected pieces and once per stage forced to torch: the ``AITER_KDA_PIECES``
   pieces (chunk_fwd, chunk_bwd, conv_bwd) and the GPU-only stages that are not
   pieces (``proj``: FlyDSL GEMM-epilogue gate/output-gate projections;
   ``rowops``: FlyDSL l2norm / gated-RMSNorm backward). The stage whose torch
   override removes the error is the wrong one. ``all=torch`` must match eager.

The eager side is ``aiter.ops.kda_fused_layer`` ``backend="torch"``.

    python benchmarks/diag_kimi_kda_stages.py --dtype fp32 --widths small
    python benchmarks/diag_kimi_kda_stages.py --dtype bf16 --widths full --seq-len 512
"""

import argparse

import torch

from lumen.models.kimi import tiny_l13


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--widths", choices=["small", "full"], default="small")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--no-lower-bound", action="store_true", help="use the softplus gate instead of the safe gate")
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=["bf16", "mxfp8", "mxfp8-lumen"], default="bf16",
                   help="KDA projection GEMMs for section [4] (Megatron-LM#6872 reference vs aiter)")
    return p.parse_args()


def main():
    args = parse_args()
    from aiter.ops import kda_fused_layer as kfl

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    cfg = tiny_l13.load_config(None)
    if args.widths == "small":
        cfg = dict(cfg, hidden=64, heads=2, head_dim=16, gate_rank=16)
    H, eps = cfg["heads"], cfg["eps"]
    lb = None if args.no_lower_bound else cfg["gate_lower_bound"]
    ts = tiny_l13.random_tensors(cfg, torch.float32, seed=args.layer)
    w = kfl.KDALayerWeights(
        **{n: t.to(args.device).to(torch.float32 if n in tiny_l13.FP32_FIELDS else dtype) for n, t in ts.items()}
    )
    g = torch.Generator(device=args.device).manual_seed(0)
    x = torch.randn(1, args.seq_len, cfg["hidden"], generator=g, device=args.device).to(dtype)
    dy = torch.randn(x.shape, generator=g, device=args.device).to(dtype)
    print(f"dtype={args.dtype} widths={args.widths} H={H} K={cfg['head_dim']} T={args.seq_len} "
          f"eps={eps} lower_bound={lb} pieces={kfl.kda_piece_choices()}")  # fmt: skip

    import contextlib

    @contextlib.contextmanager
    def forced(over):
        saved = (kfl._proj_ops, kfl._row_ops, kfl._default_gated_rmsnorm_bwd)
        pieces = {p: "torch" for p in over if p in kfl.KDA_PIECE_DEFAULTS}
        if "proj" in over:
            kfl._proj_ops = lambda x, w: kfl._TorchProj
        if "rowops" in over:
            kfl._row_ops = lambda x: kfl._TorchOps
            kfl._default_gated_rmsnorm_bwd = lambda on_gpu: kfl._TorchOps.gated_rmsnorm_bwd
        kfl.set_kda_pieces(**pieces)
        try:
            yield
        finally:
            kfl._proj_ops, kfl._row_ops, kfl._default_gated_rmsnorm_bwd = saved
            kfl.set_kda_pieces(**{p: None for p in pieces})

    stages = list(kfl.KDA_PIECE_DEFAULTS) + ["proj", "rowops"]
    variants = [("selected", ())] + [(f"{p}=torch", (p,)) for p in stages] + [("all=torch", tuple(stages))]

    print("\n[1] forward stages, aiter vs eager")
    q, k, v, gg, beta = kfl.kda_layer_preprocess(x, w, H, eps, lb)
    scale = q.shape[-1] ** -0.5
    o_ref = kfl.chunk_kda_torch(q, k, v, gg, beta, scale, args.chunk_size)
    y_ref = kfl.kda_layer_postprocess(o_ref, x, w, eps)
    _, s = kfl._kda_layer_fused_fwd(x, w, H, args.chunk_size, eps, lower_bound=lb)
    y, _ = kfl._kda_layer_fused_fwd(x, w, H, args.chunk_size, eps, for_backward=False, lower_bound=lb)
    for name, ref, out in [("q (conv+l2norm)", q, s["q"]), ("k (conv+l2norm)", k, s["k"]), ("v (conv)", v, s["v"]),
                           ("g (gate)", gg, s["g"]), ("beta", beta, s["beta"]), ("o (chunk)", o_ref, s["o"]),
                           ("y (rmsnorm*gate, o_proj)", y_ref.float(), y.float())]:  # fmt: skip
        print(f"  {name:28s} {rel(out, ref):.3e}")
    print("  pieces:", dict(sorted(kfl.LAST_PIECES.items())))
    for label, over in variants[1:]:
        with forced(over):
            yv, _ = kfl._kda_layer_fused_fwd(x, w, H, args.chunk_size, eps, for_backward=False, lower_bound=lb)
        print(f"  y with {label:16s} {rel(yv.float(), y_ref.float()):.3e}")

    print("\n[2] chunk forward piece alone, eager inputs")
    o_piece = kfl._chunk_fwd(q, k, v, gg, beta, scale, args.chunk_size, dtype)
    print(f"  {kfl.LAST_PIECES.get('chunk_fwd')}: {rel(o_piece, o_ref):.3e}")
    for qn in ("q", "k", "v"):
        t = {"q": q, "k": k, "v": v}[qn]
        print(f"  bf16 rounding of {qn} alone: {rel(t.bfloat16().float(), t):.3e}")

    print("\n[2b] conv backward piece alone (fp32 inputs) vs fp64 autograd")
    a_in = s["a"].float()
    C = a_in.shape[-1]
    dc = torch.randn(*a_in.shape[:-1], C, device=a_in.device)
    hk, hv = w.w_q.shape[0], w.w_v.shape[0]
    da_k, dw_k = kfl.conv_silu_bwd(a_in, s["conv_qkv"], *dc.split([hk, hk, hv], -1))
    with torch.enable_grad():
        a64 = a_in.double().requires_grad_(True)
        w64 = s["conv_qkv"].double().requires_grad_(True)
        da64, dw64 = torch.autograd.grad(kfl._short_conv(a64, w64), [a64, w64], dc.double())
    print(f"  {kfl.LAST_PIECES.get('conv_bwd')}: da {rel(da_k, da64):.3e}  dw {rel(dw_k, dw64):.3e}")

    print("\n[3] backward, aiter vs eager, per piece forced to torch")
    dx_ref, dw_ref = kfl.kda_layer_bwd(dy, x, w, H, args.chunk_size, eps, backend="torch", lower_bound=lb)
    names = ["dx"] + [f.name for f in kfl.fields(kfl.KDALayerWeights)]
    refs = [dx_ref, *dw_ref.tensors()]
    table = {}
    for label, over in variants:
        with forced(over):
            dx, dw = kfl.kda_layer_bwd(dy, x, w, H, args.chunk_size, eps, backend="fused", lower_bound=lb)
            if x.is_cuda:
                torch.cuda.synchronize()
        table[label] = ([rel(o, r) for o, r in zip([dx, *dw.tensors()], refs)], dict(sorted(kfl.LAST_PIECES.items())))
    print("  " + f"{'grad':10s}" + "".join(f"{lbl:>16s}" for lbl, _ in variants))
    for i, n in enumerate(names):
        print("  " + f"{n:10s}" + "".join(f"{table[lbl][0][i]:16.3e}" for lbl, _ in variants))
    for lbl, _ in variants:
        print(f"  pieces[{lbl}]: gemm={table[lbl][1].get('gemm')} {table[lbl][1]}")
    pad = dw_ref.A_log[H:].abs().max().item() if dw_ref.A_log.numel() > H else 0.0
    print(f"\n  padded A_log grad max |.|: eager {pad}")

    from lumen.models.kimi import kda_reference

    if not kda_reference.fla_available():
        print("\n[4] skipped: Megatron-LM#6872 reference needs fla-core (fla.ops.kda)")
        return
    print(f"\n[4] vs Megatron-LM#6872 KDA layer (FLA recurrence), projections {args.precision}")

    def run(fn):
        xs = x.detach().requires_grad_(True)
        ws = [t.detach().requires_grad_(True) for t in w.tensors()]
        y = fn(xs, kfl.KDALayerWeights.from_tensors(ws))
        dxw = torch.autograd.grad(y, [xs, *ws], dy)
        return [y.detach(), *dxw]

    meg = run(lambda xs, ww: kda_reference.megatron_kda_forward(xs, ww, H, lb, eps, precision=args.precision))
    print(f"  Megatron reference recurrence: {kda_reference.LAST_BACKEND} (FLA backward: {kda_reference.fla_backward_status()})")
    extra = {} if args.precision == "bf16" else {"gemm_precision": "mxfp8"}
    from lumen.models.kimi.kda_aiter import set_aiter_kda_precision

    # bf16: aiter rounds like #6872; fp32: aiter's default (fp32) precision, a pure wiring check
    aiter_mode = "megatron" if dtype == torch.bfloat16 else "default"
    set_aiter_kda_precision(aiter_mode)
    try:
        fused = run(lambda xs, ww: kfl.kda_layer_fwd(xs, ww, H, args.chunk_size, eps, "fused", lower_bound=lb, **extra))
    finally:
        set_aiter_kda_precision(None)
    gemm = kfl.LAST_PIECES.get("gemm")
    torch_b = run(lambda xs, ww: kfl.kda_layer_fwd(xs, ww, H, args.chunk_size, eps, "torch", lower_bound=lb))
    print(f"  {'tensor':10s}{'aiter fused':>16s}{'aiter torch':>16s}   (||x - megatron|| / ||megatron||, gemm={gemm}; fused in aiter {aiter_mode} precision)")
    for n, m, f, t in zip(["y"] + names, meg, fused, torch_b):
        print(f"  {n:10s}{rel(f.float(), m.float()):16.3e}{rel(t.float(), m.float()):16.3e}")


if __name__ == "__main__":
    main()
