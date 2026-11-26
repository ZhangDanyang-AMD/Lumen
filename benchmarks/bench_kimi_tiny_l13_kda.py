###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kimi-K3 Tiny-L13 benchmark: aiter KDA vs the model's own KDA.

Tiers:

* ``kda-only`` (default): the 9 Tiny-L13 KDA layers stacked (embedding, RMSNorm +
  KDA + residual blocks, final norm, LM head; MLA and MoE omitted), random init
  from the Tiny-L13 config. No checkpoint, no compressed-tensors, no fla-core.
  The eager side is aiter's ``torch`` reference layer.
* ``full``: the 13-layer remote-code checkpoint (``LUMEN_KIMI_TINY_L13_PATH``).

Reporting follows NVIDIA/Megatron-LM#6872 ("Released recipes and benchmarks"
and section 1.5) where one GPU allows:

* median iteration time and median TFLOP/s/GPU over a step window after warm-up
  (Megatron reports steps 6-50 of the full decoder; ``--warmup 5 --steps 50``),
  plus tokens/s/GPU and peak allocated memory;
* LM-loss parity as the maximum per-step |loss difference| over a 32-step run
  from the same weights and data, within ``1e-3``, with the grad-norm trajectory;
* fixed-length mock documents (``--data mock``), as in the throughput runs.

Each mode (``eager``: ``LUMEN_KDA_AITER=0``; ``aiter``: ``LUMEN_KDA_AITER=1``)
starts from the same weights and token stream. Per-piece aiter kernels follow
``AITER_KDA_PIECES``.

    python benchmarks/bench_kimi_tiny_l13_kda.py                     # tier kda-only
    AITER_KDA_PIECES=chunk_fwd=flydsl python benchmarks/bench_kimi_tiny_l13_kda.py
    LUMEN_KIMI_TINY_L13_PATH=/path/to/Kimi-K3-Tiny-L13 \\
    python benchmarks/bench_kimi_tiny_l13_kda.py --tier full --seq-len 4096
"""

import argparse
import json
import math
import os
import statistics
import time

import torch

from lumen.models.kimi import kda_reference, kda_stack, tiny_l13
from lumen.models.kimi.kda_aiter import (
    KDA_AITER_ENV,
    invalidate_aiter_kda_weight_caches,
    set_kda_is_first_microbatch,
    maybe_patch_kda_layers,
    unpatch_kda_layers,
)

LOSS_PARITY_TOL = 1e-3  # Megatron-LM#6872 continuation protocol: max LM-loss difference within 1e-3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", choices=["kda-only", "full"], default="kda-only")
    p.add_argument("--model", choices=list(tiny_l13.PRESETS), default="k3-tiny-l13",
                   help="kda-only config preset: K3 Tiny-L13 (9 KDA layers) or K3.1-Pro (69 KDA layers)")
    p.add_argument("--num-layers", type=int, default=None,
                   help="kda-only: first N config KDA layers (default: all for k3-tiny-l13, 9 for k3.1-pro)")
    p.add_argument("--ckpt", default=None, help="tier full: Tiny-L13 snapshot (default: $LUMEN_KIMI_TINY_L13_PATH)")
    p.add_argument("--seq-len", type=int, default=None, help="default 2048 (kda-only) / 4096 (full)")
    p.add_argument("--grad-check", action=argparse.BooleanOptionalAction, default=None,
                   help="compare every parameter gradient at step 1 (default: on for kda-only)")
    p.add_argument("--grad-tol", type=float, default=kda_stack.BF16_VS_FP32_FLOOR,
                   help="per-parameter ||g_aiter - g_eager|| / ||g_eager||")
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--steps", type=int, default=50, help="total steps per mode (timing and loss curve)")
    p.add_argument("--warmup", type=int, default=5, help="steps excluded from the medians")
    p.add_argument("--parity-steps", type=int, default=32, help="leading steps used for loss parity")
    p.add_argument("--modes", default="eager,aiter", help="eager = the --reference KDA, aiter = kda_layer_fwd")
    p.add_argument("--reference", choices=list(kda_stack.REFERENCES), default="megatron-fla",
                   help="eager KDA: Megatron-LM#6872 KDA layer with the FLA recurrence (primary), or aiter torch")
    p.add_argument("--precision", choices=list(kda_reference.PRECISIONS), default=None,
                   help="KDA projection GEMMs on both sides (default: mxfp8 for megatron-fla, bf16 for aiter-torch)")
    p.add_argument("--aiter-precision", choices=["default", "megatron"], default=None,
                   help="aiter KDA rounding mode (#8739 AITER_KDA_PRECISION; default: megatron with megatron-fla)")
    p.add_argument("--lr-schedule", choices=["megatron-continuation", "constant"], default=None,
                   help="megatron-continuation: lr 1e-5 cosine to 1e-6 over 50 iters, no warm-up "
                        "(default with megatron-fla); constant: --lr")
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--main-grad", action=argparse.BooleanOptionalAction, default=True,
                   help="fp32 master weights; with --precision mxfp8 the reference accumulates KDA wgrad "
                        "into fp32 main_grad (Megatron gradient_accumulation_fusion)")
    p.add_argument("--data", choices=["mock"], default="mock")
    p.add_argument("--json", default=None, help="write per-step records here")
    return p.parse_args()


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad and p.is_floating_point()]


def kda_only_flops_per_step(cfg, n_layers, vocab, tokens):
    """Fwd+bwd (3x forward) GEMM FLOPs of the KDA stack; the chunked recurrence is not counted."""
    D, HK, R, H = cfg["hidden"], cfg["heads"] * cfg["head_dim"], cfg["gate_rank"], cfg["heads"]
    per_layer = 5 * HK * D + D * R + R * HK + H * D
    return 3 * 2 * tokens * (n_layers * per_layer + D * vocab)


def model_flops_per_step(cfg_text, tokens, seq_len):
    """Fwd+bwd model FLOPs (3x forward): dense GEMMs as 2*params*token, MoE at top-k, causal MLA scores.

    Packed MXFP4 experts are counted from the config shapes, not from their storage numel.
    """
    D = cfg_text.hidden_size
    la = cfg_text.linear_attn_config
    H, K = la["num_heads"], la["head_dim"]
    n_layers = cfg_text.num_hidden_layers
    kda = len(la["kda_layers"])
    mla = n_layers - kda
    kda_params = 5 * H * K * D + D * K + K * H * K + H * D  # q, k, v, g, o; f_a, f_b; beta
    qk_hd = getattr(cfg_text, "qk_nope_head_dim", 128) + getattr(cfg_text, "qk_rope_head_dim", 64)
    v_hd = getattr(cfg_text, "v_head_dim", 128)
    q_lat = getattr(cfg_text, "q_lora_rank", 1536)
    kv_lat = getattr(cfg_text, "kv_lora_rank", 512)
    n_heads = cfg_text.num_attention_heads
    mla_params = (
        D * q_lat + q_lat * n_heads * qk_hd + D * (kv_lat + 64) + kv_lat * n_heads * (qk_hd - 64 + v_hd)
        + n_heads * v_hd * D + D * n_heads * v_hd  # o_proj, output gate
    )  # fmt: skip
    latent = cfg_text.routed_expert_hidden_size
    inter = cfg_text.moe_intermediate_size
    topk = cfg_text.num_experts_per_token
    shared = getattr(cfg_text, "num_shared_experts", 0)
    moe_layers = n_layers - getattr(cfg_text, "first_k_dense_replace", 0)
    dense_layers = n_layers - moe_layers
    routed = topk * 3 * latent * inter + 2 * D * latent
    shared_p = shared * 3 * D * inter
    dense_p = 3 * D * getattr(cfg_text, "intermediate_size", inter)
    active = (
        kda * kda_params + mla * mla_params + moe_layers * (routed + shared_p + D * cfg_text.num_experts)
        + dense_layers * dense_p + D * cfg_text.vocab_size
    )  # fmt: skip
    gemm = 2 * active * tokens
    mla_scores = mla * 2 * (tokens * seq_len / 2) * n_heads * (qk_hd + v_hd)  # causal: half the T x T
    return 3 * (gemm + mla_scores)


def lm_loss(model, ids):
    out = model(input_ids=ids, labels=ids)
    return out[0] if isinstance(out, tuple) else out.loss


def run_mode(model, mode, args, init, ids_steps):
    os.environ[KDA_AITER_ENV] = "1" if mode == "aiter" else "0"
    unpatch_kda_layers(model)
    patched = maybe_patch_kda_layers(model, gemm_precision=args.precision, kda_precision=args.aiter_precision)
    with torch.no_grad():
        for p, p0 in zip(trainable(model), init):
            p.copy_(p0)
    invalidate_aiter_kda_weight_caches()
    params = trainable(model)
    masters = None
    if args.main_grad:
        # Megatron: fp32 master weights; TE MXFP8 wgrad accumulates into fp32 main_grad
        masters = [p.detach().float().clone().requires_grad_(True) for p in params]
        if mode == "eager" and args.precision == "mxfp8":
            from lumen.ops.quantize.mxfp8_te_reference import attach_main_grad

            attach_main_grad([p for n, p in model.named_parameters() if p.dim() == 2 and ".self_attn." in n])
    opt_params = masters if masters is not None else params
    opt = (
        torch.optim.SGD(opt_params, lr=args.lr)
        if args.optimizer == "sgd"
        else torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=0.0)
    )
    torch.cuda.reset_peak_memory_stats()
    records = []
    for step, ids in enumerate(ids_steps, start=1):
        if args.lr_schedule == "megatron-continuation":
            t = min(step - 1, 50) / 50
            for group in opt.param_groups:
                group["lr"] = 1e-6 + 0.5 * (1e-5 - 1e-6) * (1 + math.cos(math.pi * t))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        set_kda_is_first_microbatch(model, True)  # one microbatch per step: rebuild aiter weight caches
        loss = lm_loss(model, ids)
        loss.backward()
        if masters is not None:
            for p in params:
                mg = getattr(p, "main_grad", None)
                if mg is not None:
                    p.grad = mg.clone()
                    mg.zero_()
        if step == 1 and args.grad_check:
            grads1 = {n: p.grad.detach().cpu() for n, p in model.named_parameters() if p.grad is not None}
        gnorm = torch.nn.utils.clip_grad_norm_(params, float("inf"))
        if masters is not None:
            for p, m in zip(params, masters):
                m.grad = None if p.grad is None else p.grad.float()
            opt.step()
            with torch.no_grad():
                for p, m in zip(params, masters):
                    p.copy_(m)
        else:
            opt.step()
        invalidate_aiter_kda_weight_caches()  # in-place updates need not bump _version
        torch.cuda.synchronize()
        records.append(
            {"step": step, "loss": loss.item(), "grad_norm": gnorm.item(), "time_s": time.perf_counter() - t0}
        )
        print(f"[{mode}] step {step:3d} loss {records[-1]['loss']:.6f} grad_norm {records[-1]['grad_norm']:.4e} "
              f"time {records[-1]['time_s']:.3f}s")  # fmt: skip
    del opt
    for p in params:
        if hasattr(p, "main_grad"):
            del p.main_grad
    unpatch_kda_layers(model)
    pieces = None
    if mode == "aiter":
        from aiter.ops import kda_fused_layer as kfl

        pieces = {"selected": kfl.kda_piece_choices(), "ran": dict(sorted(kfl.LAST_PIECES.items()))}
    return {
        "mode": mode,
        "kda_layers_patched": patched,
        "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
        "pieces": pieces,
        "records": records,
        "grads_step1": grads1 if args.grad_check else None,
    }


def summarize(res, args, flops):
    window = [r["time_s"] for r in res["records"][args.warmup :]]
    med = statistics.median(window)
    tokens = args.micro_batch * args.seq_len
    return {
        "median_iter_s": med,
        "median_tflops_gpu": statistics.median(flops / t / 1e12 for t in window),
        "tokens_per_s_gpu": tokens / med,
        "window": f"{args.warmup + 1}-{len(res['records'])}",
    }


def main():
    args = parse_args()
    assert "gfx950" in torch.cuda.get_device_properties(0).gcnArchName, "benchmark targets one gfx950 GPU"
    if args.seq_len is None:
        args.seq_len = 2048 if args.tier == "kda-only" else 4096
    if args.grad_check is None:
        args.grad_check = args.tier == "kda-only"
    if args.precision is None:
        args.precision = "mxfp8" if args.reference.startswith("megatron") else "bf16"
    if args.aiter_precision is None:
        args.aiter_precision = "megatron" if args.reference.startswith("megatron") else "default"
    if args.lr_schedule is None:
        args.lr_schedule = "megatron-continuation" if args.reference.startswith("megatron") else "constant"
    if args.reference == "megatron-aiter" and not kda_reference.aiter_recurrence_available():
        raise SystemExit("--reference megatron-aiter needs aiter.ops.kda (ROCm/aiter#8755)")
    if args.reference == "megatron-fla" and not kda_reference.fla_available():
        raise SystemExit("--reference megatron-fla needs fla-core (fla.ops.kda); or pass --reference aiter-torch")
    tokens = args.micro_batch * args.seq_len
    if args.tier == "kda-only":
        cfg = tiny_l13.load_config(None, preset=args.model)
        if args.num_layers is None and args.model == "k3.1-pro":
            args.num_layers = 9  # all 69 K3.1 KDA layers (~41B params + grads) do not fit one GPU
        fp32_params = (
            kda_stack.MEGATRON_FP32_KDA_PARAMS if args.reference.startswith("megatron") else kda_stack.FP32_KDA_PARAMS
        )
        model = kda_stack.build_kda_stack(cfg, num_layers=args.num_layers, fp32_params=fp32_params)
        kda_stack.set_reference(model, args.reference, args.precision if args.reference.startswith("megatron") else "bf16")
        vocab = cfg["vocab"]
        recipe = (f"{args.model} KDA-only ({len(model.layers)} of {len(cfg['kda_layers'])} KDA layers), "
                  f"KDA GEMMs {args.precision}")  # fmt: skip
        flops = kda_only_flops_per_step(cfg, len(model.layers), vocab, tokens)
    else:
        model = tiny_l13.load_hf_model(args.ckpt)
        if args.reference.startswith("megatron"):
            kda_reference.patch_kda_reference(
                model, args.precision, backend="aiter" if args.reference == "megatron-aiter" else "fla"
            )
        cfg_text = model.config.get_text_config()
        vocab, recipe = cfg_text.vocab_size, f"Tiny-L13 full, KDA GEMMs {args.precision}"
        flops = model_flops_per_step(cfg_text, tokens, args.seq_len)
    model.train()
    init = [p.detach().cpu().clone() for p in trainable(model)]
    g = torch.Generator(device="cpu").manual_seed(1234)
    ids_steps = [
        torch.randint(0, vocab, (args.micro_batch, args.seq_len), generator=g).cuda() for _ in range(args.steps)
    ]  # fixed-length mock documents, identical for every mode

    results = [run_mode(model, m, args, init, ids_steps) for m in args.modes.split(",")]
    gpu = torch.cuda.get_device_name(0)
    print(f"\nAITER_KDA_PIECES={os.environ.get('AITER_KDA_PIECES', '')!r}\n")
    print("| Recipe | Measured hardware | TP / PP / EP / CP | GBS | Seq | Median iteration time "
          "| Median TFLOP/s/GPU | Tokens/s/GPU | Peak alloc |")  # fmt: skip
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for res in results:
        s = summarize(res, args, flops)
        res["summary"] = s
        print(f"| {recipe}, KDA={res['mode']} (steps {s['window']}) | 1 {gpu} | 1 / 1 / 1 / 1 "
              f"| {args.micro_batch} | {args.seq_len} | {s['median_iter_s']:.4f} s | {s['median_tflops_gpu']:.1f} "
              f"| {s['tokens_per_s_gpu']:.0f} | {res['peak_alloc_gib']:.1f} GiB |")  # fmt: skip

    by_mode = {r["mode"]: r for r in results}
    if {"eager", "aiter"} <= by_mode.keys():
        n = min(args.parity_steps, args.steps)
        ref, got = (by_mode[m]["records"][:n] for m in ("eager", "aiter"))
        dloss = [abs(a["loss"] - b["loss"]) for a, b in zip(ref, got)]
        dgn = [abs(a["grad_norm"] - b["grad_norm"]) / max(abs(a["grad_norm"]), 1e-12) for a, b in zip(ref, got)]
        ok = max(dloss) <= LOSS_PARITY_TOL
        print(f"\n[primary] Megatron-LM#6872 LM-loss parity vs {args.reference} ({args.precision}, "
              f"aiter precision {args.aiter_precision}, "
              f"lr {args.lr_schedule}) over {n} steps: max |dloss| = {max(dloss):.3e} "
              f"(step {dloss.index(max(dloss)) + 1}), final {dloss[-1]:.3e} -> {'PASS' if ok else 'FAIL'} "
              f"(tol {LOSS_PARITY_TOL:g})")  # fmt: skip
        print(f"[secondary] max rel grad-norm diff {max(dgn):.3e}")
        print("aiter pieces:", by_mode["aiter"]["pieces"], "gemm:", by_mode["aiter"]["pieces"]["ran"].get("gemm"))
        if args.grad_check:
            gr, gg = by_mode["eager"]["grads_step1"], by_mode["aiter"]["grads_step1"]
            assert gr.keys() == gg.keys(), "parameters with gradients differ between modes"
            heads = next(m.num_heads for m in model.modules() if hasattr(m, "num_heads") and hasattr(m, "A_log"))
            rel, problems, diag = kda_stack.grad_parity(gr, gg, heads)
            bad = [n for n, e in rel.items() if e > args.grad_tol]
            print(f"[secondary] step-1 grads ({len(rel)} params), worst ||a-b||/||b||:")
            for n, e in list(rel.items())[:5]:
                print(f"  {n}: {e:.3e}")
            for n, rows in list(diag.items())[:3]:
                print(f"  {n} largest |diff| heads (idx, eager, aiter): {rows}")
            print(f"-> {'PASS' if not bad and not problems else 'FAIL ' + str(problems + bad[:5])} "
                  f"(tol {args.grad_tol:g})")  # fmt: skip

    for r in results:
        r.pop("grads_step1", None)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"args": vars(args), "flops_per_step": flops, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
