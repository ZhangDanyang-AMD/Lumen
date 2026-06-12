# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3-8B LoRA SFT — PyTorch FSDP + Lumen FP8 blockwise2d.

Mirrors the Llama2-7B FSDP fp8_blockwise2d recipe (LumenConfig.enable applies FP8
blockwise2d linear quantization + LoRA; attention/norm stay BF16), adapted for
Qwen3-8B:
  - AutoModelForCausalLM + Qwen3DecoderLayer FSDP wrap policy
  - Qwen3 tokenizer + chat template (the gov_report .npy is Llama2-tokenized and
    cannot be reused), alpaca-style jsonl, answer-only loss mask
"""
import argparse
import json
import logging
import math
import os
import time
from argparse import Namespace
from collections import Counter, defaultdict
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from lumen.config import LumenConfig
# Reuse the FSDP trainer's rank0 logger so startup/quant/LoRA lines share the
# `INFO:lumen.models.fsdp:` / `INFO:lumen.quantize:` style of the llama2 runs.
from lumen.models.fsdp import _rank0_print as rank0


class StepProfiler:
    """rank0-only torch.profiler over a step window, env-gated.

    Off unless LUMEN_PROF_START is set. Captures CPU+CUDA activity for steps
    [start, end], writes a self_cuda_time table (and optional chrome trace).
    Env: LUMEN_PROF_START, LUMEN_PROF_END (default start+3),
         LUMEN_PROF_OUTPUT (default /results/qwen3_profile.txt), LUMEN_PROF_TRACE.

    LUMEN_COPY_TRACE=1 additionally attributes aten::copy_ / contiguous() calls to
    their lumen call sites (count + MB), written to <out>_copy_trace.txt — use to
    locate where the copy time comes from.
    """

    def __init__(self, enabled):
        self.start = int(os.environ.get("LUMEN_PROF_START") or 0)
        self.enabled = enabled and self.start > 0
        self.end = int(os.environ.get("LUMEN_PROF_END") or (self.start + 3))
        self.out = os.environ.get("LUMEN_PROF_OUTPUT") or "/results/qwen3_profile.txt"
        self.trace = os.environ.get("LUMEN_PROF_TRACE") or ""
        self.copy_trace = os.environ.get("LUMEN_COPY_TRACE", "0") == "1"
        self.prof = None
        self._tracing = False
        self._copy_n = Counter(); self._copy_mb = defaultdict(float)
        self._contig_n = Counter(); self._contig_mb = defaultdict(float)
        if self.enabled:
            rank0(f"> Profiler armed: steps {self.start}-{self.end} -> {self.out}"
                  + (" (+copy trace)" if self.copy_trace else ""))

    def _short_stack(self):
        import traceback
        rel = []
        for f in traceback.extract_stack()[:-2]:
            if any(k in f.filename for k in ("lumen/", "examples/")):
                rel.append(f"{f.filename.split('/workspace/Lumen/')[-1]}:{f.lineno} {f.name}")
        return " <- ".join(rel[-3:]) if rel else "(non-lumen)"

    def _install_copy_patches(self):
        self._orig_copy = torch.Tensor.copy_
        self._orig_contig = torch.Tensor.contiguous
        prof = self

        def _copy(self, src, *a, **k):
            if prof._tracing:
                key = prof._short_stack()
                prof._copy_n[key] += 1
                prof._copy_mb[key] += self.nelement() * self.element_size() / 1e6
            return prof._orig_copy(self, src, *a, **k)

        def _contig(self, *a, **k):
            if prof._tracing and not self.is_contiguous(*a, **k):
                key = prof._short_stack()
                prof._contig_n[key] += 1
                prof._contig_mb[key] += self.nelement() * self.element_size() / 1e6
            return prof._orig_contig(self, *a, **k)

        torch.Tensor.copy_ = _copy
        torch.Tensor.contiguous = _contig

    def _dump_copy_trace(self):
        torch.Tensor.copy_ = self._orig_copy
        torch.Tensor.contiguous = self._orig_contig
        nsteps = max(1, self.end - self.start + 1)
        path = self.out.replace(".txt", "_copy_trace.txt")
        with open(path, "w") as f:
            f.write(f"copy_/contiguous trace, steps {self.start}-{self.end} ({nsteps} steps)\n\n")
            for title, n, mb in (("copy_", self._copy_n, self._copy_mb),
                                 ("contiguous", self._contig_n, self._contig_mb)):
                f.write(f"--- {title} by total MB (top 25) ---\n")
                f.write(f"{'calls/step':>10} {'MB/step':>10}  site\n")
                for key, m in sorted(mb.items(), key=lambda x: -x[1])[:25]:
                    f.write(f"{n[key]/nsteps:>10.0f} {m/nsteps:>10.1f}  {key}\n")
                f.write("\n")
        rank0(f"> Profiler wrote copy trace {path}")

    def step_begin(self, step):
        if self.enabled and step == self.start:
            self.prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            )
            self.prof.__enter__()
            if self.copy_trace:
                self._install_copy_patches()
                self._tracing = True

    def step_end(self, step):
        if self.enabled and self.prof is not None and step == self.end:
            self._tracing = False
            self.prof.__exit__(None, None, None)
            table = self.prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
            with open(self.out, "w") as f:
                f.write(f"Qwen3 FP8 blockwise2d profile, steps {self.start}-{self.end}\n\n{table}")
            rank0(f"> Profiler wrote {self.out}")
            if self.trace:
                self.prof.export_chrome_trace(self.trace)
                rank0(f"> Profiler wrote chrome trace {self.trace}")
            if self.copy_trace:
                self._dump_copy_trace()
            self.prof = None


class AlpacaDataset(Dataset):
    """jsonl rows {instruction, input, output} -> Qwen3 chat, answer-only mask."""

    def __init__(self, path, tokenizer, seq_length, num_samples=None):
        self.tok = tokenizer
        self.seq_length = seq_length
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        self.n = num_samples or len(self.rows)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __len__(self):
        return self.n

    def _chat(self, msgs, add_gen):
        # transformers 5.x returns a BatchEncoding when tokenize=True; return_dict
        # gives the flat input_ids list.
        try:
            o = self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=add_gen, enable_thinking=False, return_dict=True)
        except TypeError:
            o = self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=add_gen, return_dict=True)
        return list(o["input_ids"])

    def __getitem__(self, idx):
        r = self.rows[idx % len(self.rows)]
        prompt = r["instruction"].strip()
        if r.get("input", "").strip():
            prompt += "\n" + r["input"].strip()
        p_ids = self._chat([{"role": "user", "content": prompt}], add_gen=True)
        f_ids = self._chat([{"role": "user", "content": prompt}, {"role": "assistant", "content": r["output"]}], add_gen=False)
        mask = [0] * len(p_ids) + [1] * max(0, len(f_ids) - len(p_ids))
        ids = list(f_ids)
        L = self.seq_length + 1
        ids, mask = ids[:L], mask[:L]
        if len(ids) < L:
            pad = L - len(ids)
            ids += [self.pad_id] * pad
            mask += [0] * pad
        return {"input_ids": torch.LongTensor(ids), "loss_mask": torch.LongTensor(mask)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--tokenizer-name-or-path", default=None)
    p.add_argument("--mode", choices=["bf16", "fp8_blockwise2d"], default="fp8_blockwise2d")
    p.add_argument("--train-data-path", required=True)
    p.add_argument("--val-data-path", default=None)
    p.add_argument("--seq-length", type=int, default=2048)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--lr-warmup-steps", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=0.3)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--cache-frozen-weight", action="store_true",
                   help="cache the frozen base weight's FP8 quant (skip per-fwd re-quant)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--val-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", 0))
    # Surface lumen.* INFO logs (quant enable, LoRA) on rank0 only, like FSDPTrainer.
    logging.basicConfig(level=logging.INFO if global_rank == 0 else logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed)

    rank0(f"> Loading Qwen3 from {args.model_name_or_path} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rank0("> Gradient checkpointing enabled")

    # ---- Lumen LoRA (+ optional FP8 blockwise2d), same recipe as llama2 ----
    # mode=bf16 -> LoRA only (FP8 off); mode=fp8_blockwise2d -> FP8 blockwise2d linears.
    use_fp8 = args.mode == "fp8_blockwise2d"
    cfg = LumenConfig.from_args(Namespace(
        linear_fp8=use_fp8, linear_fp8_format="fp8_e4m3", linear_fp8_scaling="blockwise2d",
        linear_fp8_block_size=128, linear_fp8_amax_algo="max", linear_fp8_amax_history=16,
        linear_fp8_reduce_amax=False, linear_fp8_activation=True, linear_fp8_wgrad=True,
        linear_fp8_cache_frozen_weight=args.cache_frozen_weight,
        grad_quant_type=None, first_last_layers_bf16=False, lumen_norm=False,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    ))
    _manager, model = cfg.enable(model)  # logs INFO:lumen.quantize + INFO:lumen.config (LoRA + trainable)
    # FSDP1 flatten needs uniform dtype: cast PEFT LoRA adapters (fp32) to bf16.
    for nme, prm in model.named_parameters():
        if "lora_" in nme and prm.dtype == torch.float32:
            prm.data = prm.data.to(torch.bfloat16)

    model = FSDP(
        model,
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen3DecoderLayer}),
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    rank0(f"> FSDP model ready (version=1, sharding=full_shard, world_size={world_size})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-5, weight_decay=args.weight_decay)

    def lr_lambda(step):
        w, T, mx, mn = args.lr_warmup_steps, args.max_steps, args.lr, args.min_lr
        if step < w:
            return float(step) / max(w, 1)
        prog = float(step - w) / max(T - w, 1)
        return (mn / mx if mx > 0 else 0.0) + (1 - (mn / mx if mx > 0 else 0.0)) * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    tok = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path or args.model_name_or_path)

    def make_loader(path, n):
        ds = AlpacaDataset(path, tok, args.seq_length, num_samples=n)
        rank0(f"Loaded {len(ds)} samples from {path}")
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=global_rank, shuffle=True) if world_size > 1 else None
        return DataLoader(ds, batch_size=args.micro_batch_size, sampler=sampler, shuffle=(sampler is None),
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)

    train_loader = make_loader(args.train_data_path, None)
    val_loader = make_loader(args.val_data_path, args.val_samples) if args.val_data_path else None

    def loss_on(b):
        ids = b["input_ids"][:, :-1].to(local_rank)
        labels = b["input_ids"][:, 1:].to(local_rank)
        lm = b["loss_mask"][:, 1:].to(local_rank).float()
        logits = model(input_ids=ids).logits
        per = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1), reduction="none")
        return (per * lm.reshape(-1)).sum() / lm.sum().clamp(min=1)

    @torch.no_grad()
    def validate():
        model.eval()
        tot_l, nb = 0.0, 0
        for b in val_loader:
            tot_l += loss_on(b).item(); nb += 1
            if nb >= 10:
                break
        model.train()
        avg = tot_l / max(nb, 1)
        if dist.is_initialized():
            t = torch.tensor([avg], device="cuda"); dist.all_reduce(t, op=dist.ReduceOp.AVG); avg = t.item()
        return avg

    model.train()
    ga = args.gradient_accumulation_steps
    it = iter(train_loader)
    profiler = StepProfiler(enabled=(global_rank == 0))
    for step in range(1, args.max_steps + 1):
        profiler.step_begin(step)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        opt.zero_grad()
        acc = 0.0
        for _ in range(ga):
            try:
                b = next(it)
            except StopIteration:
                it = iter(train_loader); b = next(it)
            l = loss_on(b)
            (l / ga).backward()
            acc += l.item()
        if args.max_grad_norm > 0:
            model.clip_grad_norm_(args.max_grad_norm)
        opt.step(); sched.step()
        torch.cuda.synchronize(); step_time_ms = (time.perf_counter() - t0) * 1e3
        if step % args.log_interval == 0:
            rank0(f"  step {step}/{args.max_steps} | loss {acc/ga:.4f} | lr {sched.get_last_lr()[0]:.2e} | step_time_ms {step_time_ms:.1f}")
        profiler.step_end(step)
        if val_loader and step % args.eval_interval == 0:
            rank0(f"  step {step}/{args.max_steps} | val_loss {validate():.4f}")

    rank0(f"> Training complete after {args.max_steps} steps.")


if __name__ == "__main__":
    main()
