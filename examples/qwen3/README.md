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
| `results/` | Saved training logs |

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
`PYTORCH_TUNABLEOP_ENABLED` (0).

Direct `torchrun` (inside the container):

```bash
torchrun --nproc_per_node=8 train_qwen3_fsdp_fp8_blockwise2d.py \
  --model-name-or-path /model-qwen3 \
  --train-data-path /data/train.jsonl --val-data-path /data/validation.jsonl \
  --seq-length 2048 --max-steps 200 --eval-interval 50 --seed 1234
```

## Recipe

| Item | Value |
|---|---|
| Sharding | FSDP1 `FULL_SHARD`, `use_orig_params=True`, DP=8 |
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

~2.86 s/step (steady), 0 NaN. Full log:
`results/qwen3_fsdp_train_fp8_blockwise2d_lora_8b.log`.
