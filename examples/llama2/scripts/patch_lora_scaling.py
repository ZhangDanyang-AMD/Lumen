"""Patch Megatron-LM-AMD's LoraAdapter to use alpha/rank scaling (matching NeMo/HuggingFace PEFT).

Megatron-LM-AMD's LoraAdapter uses `self.lora_alpha * output` as the scaling factor,
where lora_alpha = alpha (e.g., 32). NeMo and HuggingFace PEFT use `alpha / rank` (e.g.,
32/16 = 2.0). This 16x discrepancy causes Lumen's LoRA updates to be far too large,
leading to loss plateau around 7.0 instead of converging to 0.925.
"""

import sys

megatron_root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/megatron_lm"
lora_path = f"{megatron_root}/megatron/core/transformer/lora_adapter.py"

with open(lora_path) as f:
    src = f.read()

old = "        self.lora_alpha = alpha"
new = "        self.lora_alpha = alpha / rank if rank > 0 else alpha"

if old in src:
    src = src.replace(old, new, 1)
    with open(lora_path, "w") as f:
        f.write(src)
    print(f"[LoRA Scaling Fix] Patched {lora_path}: alpha -> alpha/rank")
else:
    if "alpha / rank" in src:
        print(f"[LoRA Scaling Fix] Already patched {lora_path}")
    else:
        print(f"[LoRA Scaling Fix] WARNING: target not found in {lora_path}")
        sys.exit(1)
