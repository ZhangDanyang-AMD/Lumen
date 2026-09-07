# Qwen3-30B-A3B checkpoint conversion

This directory contains only the model-specific layer needed by Megatron's
checkpoint converter. Generic converter modules (`saver_base`, `schema_base`,
and `utils`) are imported from the bundled Megatron-LM checkout.

## Layout

- `convert_hf_to_megatron.sh`: converts HF safetensors to TP=1, PP=1, EP=8.
- `loader_qwen3_moe_hf.py`: streams Qwen3 tensors into converter messages.
- `saver_qwen3_moe.py`: writes those messages into MCore checkpoints.
- `schema_qwen3_moe.py`: maps Qwen3 fields to MCore module paths.
- `qwen3_moe_mapping.py`: pure QKV, SwiGLU, and expert-sharding helpers.
- `export_megatron_blockwise_fp8.py`: exports a final BF16 MCore checkpoint as
  Lumen 128x128 blockwise FP8 inference weights.
- `tests/`: unit tests for tensor layouts and EP partitioning.

## Usage

Run inside the benchmark container:

```bash
bash checkpoint/convert_hf_to_megatron.sh
```

Defaults:

- HF input: `/nobackup/model/Qwen3-30B-A3B`
- output: `/nobackup/checkpoints/Qwen3-30B-A3B-tp1-pp1-ep8`
- Megatron-LM: `/workspace/Megatron-LM`

Override these with `HF_DIR`, `SAVE_DIR`, and `MEGATRON_PATH`.

## Export the final BF16 checkpoint to FP8

The FP8 training path keeps BF16 master weights in its resumable checkpoints.
After training, export the selected iteration with:

```bash
python3 checkpoint/export_megatron_blockwise_fp8.py \
  --load-dir /nobackup/checkpoints/Qwen3-30B-A3B-tp1-pp1-ep8 \
  --save-dir /nobackup/checkpoints/Qwen3-30B-A3B-blockwise-fp8
```

Run this in the Lumen training image on a GPU. The exporter uses the same
production `_quant_blockwise2d_weight` implementation as training. Each rank
gets a `model_fp8.pt` containing E4M3 weights and sibling `weight_scale`
dequantization factors. Embeddings, output heads, norms, and routers stay in
their checkpoint dtype.

The result is a Lumen inference artifact, not a resumable Megatron training
checkpoint and not yet a Hugging Face `weight_scale_inv` checkpoint.
