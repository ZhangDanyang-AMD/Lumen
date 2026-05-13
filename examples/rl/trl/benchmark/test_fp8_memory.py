"""FP8 memory savings benchmark — measures actual GPU memory for different
FP8 configurations on a single GPU with a small training loop.

Configs tested:
  1. BF16 baseline (AdamW)
  2. FP8ParamManager (true FP8 weight storage + dequant hooks)
  3. FP8ParamManager + 8-bit Adam (bitsandbytes)
  4. FP8 Attention (dpa) via LumenConfig

NOTE: For accurate results, run each config in a separate process to avoid
GPU memory leaks between configs. When run via --configs all in a single
process, the second config may inherit residual GPU memory from the first.
The --output JSON is still useful for per-step data within each config.

Usage:
    python test_fp8_memory.py --model /dev/shm/model/llama-3.1-8b --configs all
    python test_fp8_memory.py --model /dev/shm/model/llama-3.1-8b --configs bf16
    python test_fp8_memory.py --model /dev/shm/model/llama-3.1-8b --configs fp8params
"""

import argparse
import gc
import json

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_memory_mb():
    return {
        "allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
        "current_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
    }


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_dummy_batch(tokenizer, batch_size=2, seq_len=256, device="cuda"):
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def run_bf16_baseline(model_path, steps=3, batch_size=2, seq_len=256):
    """Config 1: BF16 baseline with AdamW."""
    print("\n" + "=" * 70)
    print("CONFIG 1: BF16 Baseline (AdamW)")
    print("=" * 70)

    reset_memory()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    total_p, trainable_p = count_params(model)
    print(f"  Params: {total_p / 1e9:.2f}B total, {trainable_p / 1e9:.2f}B trainable")

    mem_after_model = get_memory_mb()
    print(f"  After model load: {mem_after_model['allocated_mb']:.0f} MB alloc, "
          f"{mem_after_model['reserved_mb']:.0f} MB reserved")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)

    reset_memory()
    results = {"config": "bf16_baseline", "steps": []}

    for step in range(steps):
        batch = make_dummy_batch(tokenizer, batch_size, seq_len)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        mem = get_memory_mb()
        step_result = {"step": step + 1, **mem, "loss": loss.item()}
        results["steps"].append(step_result)
        print(f"  Step {step + 1}: loss={loss.item():.4f}, "
              f"peak_alloc={mem['allocated_mb']:.0f} MB, peak_res={mem['reserved_mb']:.0f} MB")

    del model, optimizer
    reset_memory()
    return results


def run_fp8_param_manager(model_path, steps=3, batch_size=2, seq_len=256):
    """Config 2: FP8ParamManager — true FP8 weight storage with dequant hooks."""
    print("\n" + "=" * 70)
    print("CONFIG 2: FP8ParamManager (True FP8 Weight Storage)")
    print("=" * 70)

    reset_memory()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    mem_before_fp8 = get_memory_mb()
    print(f"  Before FP8ParamManager: {mem_before_fp8['current_allocated_mb']:.0f} MB current alloc")

    from lumen.quantize.fp8_params import FP8ParamManager
    mgr = FP8ParamManager(fp8_dtype=torch.float8_e4m3fn)
    n_quantized = mgr.quantize_params(model)
    n_hooks = mgr.register_dequant_hooks(model)
    print(f"  FP8ParamManager: quantized {n_quantized} params, {n_hooks} dequant hooks")

    mem_after_fp8 = get_memory_mb()
    savings = mgr.memory_savings_bytes(model)
    print(f"  After FP8ParamManager: {mem_after_fp8['current_allocated_mb']:.0f} MB current alloc")
    print(f"  Theoretical weight savings: {savings / 1024**2:.0f} MB "
          f"({savings / 1024**3:.2f} GB)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)

    reset_memory()
    results = {"config": "fp8_param_manager", "n_quantized": n_quantized, "steps": []}

    for step in range(steps):
        batch = make_dummy_batch(tokenizer, batch_size, seq_len)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        mem = get_memory_mb()
        step_result = {"step": step + 1, **mem, "loss": loss.item()}
        results["steps"].append(step_result)
        print(f"  Step {step + 1}: loss={loss.item():.4f}, "
              f"peak_alloc={mem['allocated_mb']:.0f} MB, peak_res={mem['reserved_mb']:.0f} MB")

    mgr.remove_hooks()
    del model, optimizer, mgr
    reset_memory()
    return results


def run_fp8_param_8bit_adam(model_path, steps=3, batch_size=2, seq_len=256):
    """Config 3: FP8ParamManager + 8-bit Adam (bitsandbytes)."""
    print("\n" + "=" * 70)
    print("CONFIG 3: FP8ParamManager + 8-bit Adam (bitsandbytes)")
    print("=" * 70)

    try:
        import bitsandbytes as bnb
        print(f"  bitsandbytes version: {bnb.__version__}")
    except ImportError:
        print("  ERROR: bitsandbytes not installed. Skipping.")
        return {"config": "fp8_param_8bit_adam", "error": "bitsandbytes not installed"}

    reset_memory()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    from lumen.quantize.fp8_params import FP8ParamManager
    mgr = FP8ParamManager(fp8_dtype=torch.float8_e4m3fn)
    n_quantized = mgr.quantize_params(model)
    n_hooks = mgr.register_dequant_hooks(model)
    print(f"  FP8ParamManager: quantized {n_quantized} params, {n_hooks} dequant hooks")

    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=5e-6)
    print(f"  Using 8-bit Adam from bitsandbytes")

    reset_memory()
    results = {"config": "fp8_param_8bit_adam", "n_quantized": n_quantized, "steps": []}

    for step in range(steps):
        batch = make_dummy_batch(tokenizer, batch_size, seq_len)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        mem = get_memory_mb()
        step_result = {"step": step + 1, **mem, "loss": loss.item()}
        results["steps"].append(step_result)
        print(f"  Step {step + 1}: loss={loss.item():.4f}, "
              f"peak_alloc={mem['allocated_mb']:.0f} MB, peak_res={mem['reserved_mb']:.0f} MB")

    mgr.remove_hooks()
    del model, optimizer, mgr
    reset_memory()
    return results


def run_fp8_attention(model_path, steps=3, batch_size=2, seq_len=256):
    """Config 4: FP8 Attention (dpa) via LumenConfig."""
    print("\n" + "=" * 70)
    print("CONFIG 4: FP8 Attention (dpa) via LumenConfig")
    print("=" * 70)

    reset_memory()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    from lumen.config import LumenConfig
    cfg = LumenConfig(fp8_attn="dpa")
    manager, model = cfg.enable(model)
    print(f"  LumenConfig enabled: fp8_attn=dpa")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)

    reset_memory()
    results = {"config": "fp8_attention_dpa", "steps": []}

    for step in range(steps):
        batch = make_dummy_batch(tokenizer, batch_size, seq_len)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        mem = get_memory_mb()
        step_result = {"step": step + 1, **mem, "loss": loss.item()}
        results["steps"].append(step_result)
        print(f"  Step {step + 1}: loss={loss.item():.4f}, "
              f"peak_alloc={mem['allocated_mb']:.0f} MB, peak_res={mem['reserved_mb']:.0f} MB")

    del model, optimizer, manager
    reset_memory()
    return results


CONFIG_MAP = {
    "bf16": run_bf16_baseline,
    "fp8params": run_fp8_param_manager,
    "fp8_8bit": run_fp8_param_8bit_adam,
    "fp8attn": run_fp8_attention,
}


def main():
    parser = argparse.ArgumentParser(description="FP8 memory savings benchmark")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--configs", default="all",
                        help="Comma-separated configs: bf16,fp8params,fp8_8bit,fp8attn or 'all'")
    parser.add_argument("--steps", type=int, default=3, help="Training steps per config")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--output", default=None, help="JSON output file")
    args = parser.parse_args()

    if args.configs == "all":
        configs = list(CONFIG_MAP.keys())
    else:
        configs = [c.strip() for c in args.configs.split(",")]

    print(f"Model: {args.model}")
    print(f"Configs: {configs}")
    print(f"Steps: {args.steps}, Batch: {args.batch_size}, SeqLen: {args.seq_len}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    all_results = []
    for cfg_name in configs:
        if cfg_name not in CONFIG_MAP:
            print(f"\nWARNING: Unknown config '{cfg_name}', skipping.")
            continue
        fn = CONFIG_MAP[cfg_name]
        try:
            result = fn(args.model, steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len)
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR in {cfg_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"config": cfg_name, "error": str(e)})

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<25} {'Peak Alloc (MB)':<18} {'Peak Res (MB)':<18} {'vs BF16':<12}")
    print("-" * 70)

    bf16_peak = None
    for r in all_results:
        if "error" in r:
            print(f"{r['config']:<25} {'ERROR':<18} {r.get('error', '')}")
            continue
        last_step = r["steps"][-1] if r["steps"] else {}
        peak_alloc = last_step.get("allocated_mb", 0)
        peak_res = last_step.get("reserved_mb", 0)
        if r["config"] == "bf16_baseline":
            bf16_peak = peak_alloc
            delta_str = "baseline"
        elif bf16_peak:
            delta = (peak_alloc - bf16_peak) / bf16_peak * 100
            delta_str = f"{delta:+.1f}%"
        else:
            delta_str = "N/A"
        print(f"{r['config']:<25} {peak_alloc:<18.0f} {peak_res:<18.0f} {delta_str:<12}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
