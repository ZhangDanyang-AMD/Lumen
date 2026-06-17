"""Export DCP checkpoint to HuggingFace format with LoRA merged.

Converts DCP shards → full state dict → load into PEFT model → merge → save HF.

Usage::

    python export_hf.py \
        --base-model /dev/shm/qwen2.5-coder-32b \
        --dcp-path /home/danyzhan/cpt-results/rank_128/final/final \
        --output /home/danyzhan/cpt-results/Qwen2.5-Coder-CPT \
        --lora-rank 128 --lora-alpha 256
"""

import argparse
import logging
import os

import torch

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--dcp-path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--lora-alpha", type=float, default=256.0)
    args = parser.parse_args()

    # Step 1: Convert DCP to single state dict
    logger.info("Converting DCP shards ...")
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    tmp_ckpt = args.output + "_tmp.pt"
    dcp_to_torch_save(args.dcp_path, tmp_ckpt)

    dcp_sd = torch.load(tmp_ckpt, map_location="cpu", weights_only=False)
    logger.info("DCP state dict: %d keys, %d lora keys",
                len(dcp_sd),
                sum(1 for k in dcp_sd if "lora" in k))

    # Step 2: Build PEFT model with same structure
    logger.info("Building PEFT model from %s ...", args.base_model)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="cpu",
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    for n, p in model.named_parameters():
        if "lora_" in n and p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)

    # Step 3: Load DCP state dict with strict matching
    model_sd = model.state_dict()
    matched, skipped = 0, 0
    for key in dcp_sd:
        if key in model_sd:
            if model_sd[key].shape == dcp_sd[key].shape:
                model_sd[key] = dcp_sd[key].to(model_sd[key].dtype)
                matched += 1
            else:
                logger.warning("Shape mismatch: %s model=%s dcp=%s",
                               key, model_sd[key].shape, dcp_sd[key].shape)
                skipped += 1
        else:
            skipped += 1

    logger.info("Matched %d / %d keys (skipped %d)", matched, len(dcp_sd), skipped)

    # Verify LoRA weights are loaded (not zeros)
    lora_loaded = sum(1 for k, v in model_sd.items()
                      if "lora_A" in k and v.abs().sum() > 0)
    lora_total = sum(1 for k in model_sd if "lora_A" in k)
    logger.info("LoRA_A non-zero: %d / %d", lora_loaded, lora_total)

    model.load_state_dict(model_sd)
    del dcp_sd, model_sd

    # Step 4: Merge LoRA
    logger.info("Merging LoRA ...")
    merged = model.merge_and_unload()

    # Step 5: Save
    os.makedirs(args.output, exist_ok=True)
    logger.info("Saving to %s ...", args.output)
    merged.save_pretrained(args.output, max_shard_size="5GB")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)

    os.remove(tmp_ckpt)

    n_files = len([f for f in os.listdir(args.output) if f.endswith('.safetensors')])
    total_gb = sum(os.path.getsize(os.path.join(args.output, f))
                   for f in os.listdir(args.output)) / (1024**3)
    logger.info("Done! %d files, %.1f GB", n_files, total_gb)


if __name__ == "__main__":
    main()
