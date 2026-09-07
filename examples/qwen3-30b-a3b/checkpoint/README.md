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
  official-style 128×128 E4M3 weights plus `weight_scale_inv`.
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
`_quant_blockwise2d_weight` kernel as training. Coverage matches official
Qwen3-30B-A3B-FP8: attn QKV/proj and expert fc1/fc2 (fused `w1`/`w2` in
Sonic). Each rank gets a `model_fp8.pt` with E4M3 weights and sibling
`weight_scale_inv` (dequant multiplier, `fp8.float() * scale`). Embeddings,
output heads, norms, and routers stay in their checkpoint dtype. Training
also runs FP8 dgrad/wgrad on those GEMMs; that is not part of the HF
inference dump.

The result is a Lumen/Megatron inference artifact, not a resumable training
checkpoint and not a drop-in Hugging Face `Qwen/Qwen3-30B-A3B-FP8` directory
(Transformers module names and CUDA `e4m3fn` still need a separate converter).
FSDP/HF training has no equivalent exporter.
