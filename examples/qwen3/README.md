# Qwen3-8B LoRA SFT — PyTorch FSDP + Lumen FP8 blockwise2d

Supervised fine-tuning of **Qwen3-8B** with LoRA on **8×MI308X**, using PyTorch
FSDP for sharding and **Lumen** for FP8 **blockwise2d** linear quantization. Same
recipe as the Llama2-7B FSDP `fp8_blockwise2d` run (`../llama2`): `LumenConfig.enable`
applies FP8 blockwise2d linear quant + LoRA; attention and norm stay BF16.

## Layout

| Path | Purpose |
|---|---|
| `train_qwen3_fsdp_fp8_blockwise2d.py` | Training script (FSDP + `LumenConfig.enable`) |
| `run_qwen3_fsdp_mi308.sh` | Docker launcher for 8×MI308X |
| `scripts/download_model.py` | Fetch the HF model checkpoint |
| `scripts/download_dataset.py` | Fetch an alpaca-style dataset as jsonl |
| `requirements.txt` | Python deps (`transformers`, `peft`, `datasets`, ...) |
| `results/` | Saved training logs + performance notes/report |

## Data

The script consumes alpaca-style jsonl rows `{instruction, input, output}`,
rendered through the **Qwen3 chat template** with an **answer-only loss mask**
(loss only on assistant tokens). The Llama2 gov_report `.npy` cannot be reused —
it is pre-tokenized with the Llama2 tokenizer.

```bash
python scripts/download_model.py   --model_name Qwen/Qwen3-8B --output_dir /data/Qwen3-8B
python scripts/download_dataset.py --dataset_name tatsu-lab/alpaca --output_dir /data/alpaca
```

> The reference run used a local Chinese alpaca set
> (`alpaca_zh-{train,valid}-general.jsonl`); any `{instruction, input, output}`
> jsonl works.

## Run

```bash
HOST_MODEL=/data/Qwen3-8B HOST_DATA=/data/alpaca \
TRAIN_FILE=train.jsonl VAL_FILE=validation.jsonl \
  bash run_qwen3_fsdp_mi308.sh
```

Overridable env: `HOST_MODEL`, `HOST_DATA`, `HOST_RESULTS`, `TRAIN_FILE`,
`VAL_FILE`, `SEQ_LENGTH` (default 2048), `MAX_STEPS` (200), `EVAL_INTERVAL` (50),
`PYTORCH_TUNABLEOP_ENABLED` (0), plus the optimization flags below.

Direct `torchrun` (inside the container):

```bash
torchrun --nproc_per_node=8 train_qwen3_fsdp_fp8_blockwise2d.py \
  --model-name-or-path /model-qwen3 \
  --train-data-path /data/train.jsonl --val-data-path /data/validation.jsonl \
  --seq-length 2048 --max-steps 200 --eval-interval 50 --seed 1234
```

## Optimizations

The base recipe above is the **unoptimized FP8 baseline** (~2.78 s/step, ~61% slower
than BF16 — overhead-bound). A series of **lossless** (loss bit-identical) and
**memory-for-speed** optimizations bring it down to **~634 ms/step** (faster than BF16).
Full analysis: [`results/qwen3_fp8_perf_notes.md`](results/qwen3_fp8_perf_notes.md) and the
visual report [`results/qwen3_fp8_optimization_report.html`](results/qwen3_fp8_optimization_report.html);
per-operator breakdown in `results/qwen3_fsdp2_operator_breakdown.xlsx`.

| Env flag | Effect | Type |
|---|---|---|
| `CACHE_FROZEN_WEIGHT=1` | Quantize the frozen LoRA base weight to FP8 **once** (not every forward); also caches the DGrad transpose and skips the frozen-weight WGrad | lossless |
| `BPRESHUFFLE=1` | Pre-shuffle the cached FP8 weight into an engine-friendly layout → blockscale GEMM ~2.6× (needs `CACHE_FROZEN_WEIGHT`) | lossless |
| `AITER_ATTN=1` | AITER CK FMHA v3 attention instead of PyTorch AOTriton SDPA | lossless |
| `LUMEN_NORM=1` | Fused RMSNorm (`Qwen3RMSNorm` → `LumenRMSNorm`) | lossless |
| `FUSE_ROPE=1` | Fused RoPE (HF `apply_rotary_pos_emb` → AITER autograd RoPE) | lossless |
| `GRAD_CKPT=0` | Disable gradient checkpointing — no backward recompute (needs activation memory) | memory-for-speed |
| `SHARDING=shard_grad_op` | ZeRO-2: params stay resident, no backward all-gather (pair with `GRAD_CKPT=0`) | memory-for-speed |
| `LIMIT_ALL_GATHERS=0` `FORWARD_PREFETCH=1` | Loosen FSDP all-gather throttling / prefetch | memory-for-speed |
| `FSDP_VERSION=2` | Use FSDP2 (`fully_shard`, per-parameter sharding) | — |
| `FSDP_FP8_PARAM_STORAGE=1` | Store the frozen base as FP8 **in the shard** (all-gather FP8, not BF16) — for large models that can't cache the full FP8 weight (e.g. 70B); needs FSDP2. Not the fastest 8B path | memory |

Fastest 8B config (memory permitting):

```bash
FSDP_VERSION=2 CACHE_FROZEN_WEIGHT=1 BPRESHUFFLE=1 GRAD_CKPT=0 \
SHARDING=shard_grad_op LIMIT_ALL_GATHERS=0 FORWARD_PREFETCH=1 \
AITER_ATTN=1 LUMEN_NORM=1 FUSE_ROPE=1 MODE=fp8_blockwise2d \
HOST_MODEL=/data/Qwen3-8B HOST_DATA=/data/alpaca \
  bash run_qwen3_fsdp_mi308.sh
```

| stage | step time |
|---|---|
| FP8 baseline | 2784 ms |
| + frozen-weight cache + skip WGrad | 1912 ms |
| + no grad-ckpt + SHARD_GRAD_OP | 1542 ms |
| + bpreshuffle GEMM (FSDP1 best) | 960 ms |
| + FSDP2 + AITER attn + fused RMSNorm/RoPE | **634 ms** |

When memory is tight (large model / batch / seq), drop `GRAD_CKPT=0` and `SHARDING`;
the lossless lib-level flags alone still give ~−31%.

## Recipe

| Item | Value |
|---|---|
| Sharding | FSDP1 `FULL_SHARD` (or FSDP2 `fully_shard` via `FSDP_VERSION=2`), `use_orig_params=True`, DP=8 |
| Precision | FP8 e4m3 **blockwise2d** (128×128 tiles) on linears; attention/norm BF16 |
| LoRA | rank 16, alpha 32, dropout 0.1 — q/k/v/o/gate/up/down (~0.53% trainable) |
| Seq length / micro-batch / global batch | 2048 / 1 / 8 |
| Optimizer / schedule | AdamW (β=0.9,0.95, eps=1e-5, wd=1e-4) / cosine→0, grad clip 0.3 |
| LR / steps | 4e-4 / 200 |

## Reference result (8×MI308X)

| step | val_loss (answer-only) |
|---|---|
| 50 | 1.5968 |
| 100 | 1.6086 |
| 150 | 1.5771 |
| 200 | 1.5741 |

0 NaN; val_loss is **bit-identical** across all optimization stages (the optimizations
change only kernel selection / scheduling, not the math). Logs:

| Log | Config |
|---|---|
| `results/qwen3_fsdp_train_bf16_lora_8b.log` | BF16 reference (~1.73 s/step) |
| `results/qwen3_fsdp1_train_fp8_blockwise2d_lora_8b.log` | FSDP1 FP8 baseline (~2.78 s/step) |
| `results/qwen3_fsdp2_train_fp8_blockwise2d_best_lora_8b.log` | FSDP2 fully-optimized (~634 ms/step) |
