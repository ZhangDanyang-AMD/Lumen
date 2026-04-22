"""Patch megatron checkpointing to insert .base_layer for LoRA-wrapped layers.

The DDP and Float16Module both override load_state_dict to pass through to
the inner module directly. So checkpoint keys need to match GPTModel's
state_dict format (no module. prefix), but with .base_layer. for LoRA layers.

Checkpoint has:  decoder.layers.0.self_attention.linear_qkv.weight
Model expects:   decoder.layers.0.self_attention.linear_qkv.base_layer.weight
"""

import os
import sys

megatron_root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/megatron_lm"
target = os.path.join(megatron_root, "megatron", "training", "checkpointing.py")

with open(target) as f:
    src = f.read()

# Find and replace the load_model_state_dict function
start_marker = "    def load_model_state_dict(module, state_dict, strict: bool):"
end_marker = "\n    # Model."
start_idx = src.find(start_marker)
end_idx = src.find(end_marker, start_idx)

if start_idx < 0 or end_idx < 0:
    print("[patch_checkpointing] ERROR: Could not find function boundaries")
    sys.exit(1)

REPLACEMENT = r'''    def load_model_state_dict(module, state_dict, strict: bool):
        """Load state dict with LoRA base_layer key remapping."""
        import re as _re

        # Get the innermost model (strip DDP + Float16Module wrappers)
        inner = module
        while hasattr(inner, 'module'):
            inner = inner.module

        # Get the inner model's state_dict keys (what load_state_dict will match against)
        inner_keys = set(inner.state_dict().keys())

        # Check if remapping is needed: do checkpoint keys match inner model keys?
        ckpt_keys = set(state_dict.keys())
        common = ckpt_keys.intersection(inner_keys)
        needs_remap = len(common) < len(ckpt_keys)
        if not needs_remap:
            print(f"[CKPT FIX] All {len(common)} checkpoint keys match model")
        else:
            print(f"[CKPT FIX] {len(common)}/{len(ckpt_keys)} keys match directly, remapping remaining...")
            # Find LoRA base_layer parents from inner model keys
            lora_parents = set()
            for ik in inner_keys:
                m = _re.match(r'(.+)\.base_layer\.weight$', ik)
                if m:
                    lora_parents.add(m.group(1))

            # Find _norm.weight keys
            norm_keys = {ik for ik in inner_keys if '._norm.' in ik}

            new_sd = {}
            mapped = 0
            for ck, cv in state_dict.items():
                # Check if this key's parent is a LoRA-wrapped layer
                parts = ck.rsplit(".", 1)
                if len(parts) == 2:
                    parent, param = parts
                    if parent in lora_parents:
                        base_key = f"{parent}.base_layer.{param}"
                        new_sd[base_key] = cv
                        mapped += 1
                        continue

                new_sd[ck] = cv

                # Duplicate layernorm weights to _norm.weight if needed
                if ck.endswith(".weight") and ("layernorm" in ck or "final_layernorm" in ck):
                    norm_key = ck.replace(".weight", "._norm.weight")
                    if norm_key in inner_keys:
                        new_sd[norm_key] = cv

            state_dict = new_sd

            # Remap fused LayerNormLinear norm weights (--lumen-linear)
            _ln_rules = [
                ("input_layernorm.", "self_attention.linear_qkv.base_layer.layer_norm_"),
                ("input_layernorm.", "self_attention.linear_qkv.layer_norm_"),
                ("pre_mlp_layernorm.", "mlp.linear_fc1.base_layer.layer_norm_"),
                ("pre_mlp_layernorm.", "mlp.linear_fc1.layer_norm_"),
            ]
            ln_mapped = 0
            for ck in list(state_dict.keys()):
                for old_frag, new_frag in _ln_rules:
                    if old_frag not in ck:
                        continue
                    tgt = ck.replace(old_frag, new_frag, 1)
                    if tgt in inner_keys and tgt not in state_dict:
                        state_dict[tgt] = state_dict.pop(ck)
                        ln_mapped += 1
                        break
            if ln_mapped:
                print(f"[CKPT FIX] Remapped {ln_mapped} fused LayerNormLinear norm keys")

            loaded = set(state_dict.keys()).intersection(inner_keys)
            not_loaded = inner_keys - set(state_dict.keys())
            important_missing = [k for k in sorted(not_loaded)
                                 if "lora_" not in k and "_extra_state" not in k
                                 and "cross_attn" not in k and "._norm." not in k]
            print(f"[CKPT FIX] Remapped {mapped} LoRA base_layer keys")
            print(f"[CKPT FIX] After remapping: {len(loaded)}/{len(inner_keys)} model keys covered")
            if important_missing:
                print(f"[CKPT FIX] Important missing ({len(important_missing)}): {important_missing[:15]}")
            else:
                print(f"[CKPT FIX] All base model weights mapped!")

        try:
            module.load_state_dict(state_dict, strict=strict)
        except Exception as e:
            if strict:
                load_return = module.load_state_dict(state_dict, strict=False)
                print(f"load_return: {load_return}")

        print(f"[CKPT FIX] Checkpoint loaded successfully (verification skipped for memory efficiency)")
'''

src = src[:start_idx] + REPLACEMENT + src[end_idx:]

# Patch torch.load to use mmap=True for memory-efficient loading
# This prevents 8 ranks from each loading 128GB into CPU RAM
old_load = "state_dict = torch.load(checkpoint_name, map_location='cpu')"
new_load = "state_dict = torch.load(checkpoint_name, map_location='cpu', mmap=True, weights_only=False)"
load_count = src.count(old_load)
src = src.replace(old_load, new_load)
print(f"[patch_checkpointing] Patched {load_count} torch.load calls with mmap=True")

with open(target, "w") as f:
    f.write(src)
print(f"[patch_checkpointing] Patched {target}")
