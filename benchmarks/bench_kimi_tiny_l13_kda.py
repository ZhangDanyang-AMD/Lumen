###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Whole-network Kimi-K3 Tiny-L13 benchmark: aiter KDA vs the model's own KDA.

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

    LUMEN_KIMI_TINY_L13_PATH=/path/to/Kimi-K3-Tiny-L13 \\
    AITER_KDA_PIECES=chunk_fwd=flydsl \\
    python benchmarks/bench_kimi_tiny_l13_kda.py --seq-len 4096 --steps 50 --warmup 5
"""

import argparse
import json
import os
import statistics
import time

import torch

from lumen.models.kimi import tiny_l13
from lumen.models.kimi.kda_aiter import KDA_AITER_ENV, maybe_patch_kda_layers, unpatch_kda_layers

LOSS_PARITY_TOL = 1e-3  # Megatron-LM#6872 continuation protocol: max LM-loss difference within 1e-3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="Tiny-L13 snapshot (default: $LUMEN_KIMI_TINY_L13_PATH)")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--steps", type=int, default=50, help="total steps per mode (timing and loss curve)")
    p.add_argument("--warmup", type=int, default=5, help="steps excluded from the medians")
    p.add_argument("--parity-steps", type=int, default=32, help="leading steps used for loss parity")
    p.add_argument("--modes", default="eager,aiter")
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--data", choices=["mock"], default="mock")
    p.add_argument("--json", default=None, help="write per-step records here")
    return p.parse_args()


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad and p.is_floating_point()]


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


def run_mode(model, mode, args, init, ids_steps):
    os.environ[KDA_AITER_ENV] = "1" if mode == "aiter" else "0"
    unpatch_kda_layers(model)
    patched = maybe_patch_kda_layers(model)
    with torch.no_grad():
        for p, p0 in zip(trainable(model), init):
            p.copy_(p0)
    params = trainable(model)
    opt = (
        torch.optim.SGD(params, lr=args.lr)
        if args.optimizer == "sgd"
        else torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    )
    torch.cuda.reset_peak_memory_stats()
    records = []
    for step, ids in enumerate(ids_steps, start=1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, float("inf"))
        opt.step()
        torch.cuda.synchronize()
        records.append(
            {"step": step, "loss": loss.item(), "grad_norm": gnorm.item(), "time_s": time.perf_counter() - t0}
        )
        print(f"[{mode}] step {step:3d} loss {records[-1]['loss']:.6f} grad_norm {records[-1]['grad_norm']:.4e} "
              f"time {records[-1]['time_s']:.3f}s")  # fmt: skip
    del opt
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
    model = tiny_l13.load_hf_model(args.ckpt)
    model.train()
    cfg_text = model.config.get_text_config()
    init = [p.detach().cpu().clone() for p in trainable(model)]
    g = torch.Generator(device="cpu").manual_seed(1234)
    ids_steps = [
        torch.randint(0, cfg_text.vocab_size, (args.micro_batch, args.seq_len), generator=g).cuda()
        for _ in range(args.steps)
    ]  # fixed-length mock documents, identical for every mode
    flops = model_flops_per_step(cfg_text, args.micro_batch * args.seq_len, args.seq_len)

    results = [run_mode(model, m, args, init, ids_steps) for m in args.modes.split(",")]
    gpu = torch.cuda.get_device_name(0)
    print(f"\nAITER_KDA_PIECES={os.environ.get('AITER_KDA_PIECES', '')!r}\n")
    print("| Recipe | Measured hardware | TP / PP / EP / CP | GBS | Seq | Median iteration time "
          "| Median TFLOP/s/GPU | Tokens/s/GPU | Peak alloc |")  # fmt: skip
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for res in results:
        s = summarize(res, args, flops)
        res["summary"] = s
        print(f"| Tiny-L13 BF16, KDA={res['mode']} (steps {s['window']}) | 1 {gpu} | 1 / 1 / 1 / 1 "
              f"| {args.micro_batch} | {args.seq_len} | {s['median_iter_s']:.4f} s | {s['median_tflops_gpu']:.1f} "
              f"| {s['tokens_per_s_gpu']:.0f} | {res['peak_alloc_gib']:.1f} GiB |")  # fmt: skip

    by_mode = {r["mode"]: r for r in results}
    if {"eager", "aiter"} <= by_mode.keys():
        n = min(args.parity_steps, args.steps)
        ref, got = (by_mode[m]["records"][:n] for m in ("eager", "aiter"))
        dloss = [abs(a["loss"] - b["loss"]) for a, b in zip(ref, got)]
        dgn = [abs(a["grad_norm"] - b["grad_norm"]) / max(abs(a["grad_norm"]), 1e-12) for a, b in zip(ref, got)]
        ok = max(dloss) <= LOSS_PARITY_TOL
        print(f"\nLM-loss parity over {n} steps: max |dloss| = {max(dloss):.3e} "
              f"(step {dloss.index(max(dloss)) + 1}), final {dloss[-1]:.3e}, "
              f"max rel grad-norm diff {max(dgn):.3e} -> {'PASS' if ok else 'FAIL'} (tol {LOSS_PARITY_TOL:g})")  # fmt: skip
        print("aiter pieces:", by_mode["aiter"]["pieces"])
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"args": vars(args), "flops_per_step": flops, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
