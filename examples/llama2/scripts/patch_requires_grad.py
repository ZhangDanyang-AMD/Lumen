"""Patch TransformerBlock.forward to force hidden_states.requires_grad_(True).

With LoRA fine-tuning + activation checkpointing, the embedding layer is
frozen so its output has requires_grad=False. make_viewless_tensor only
sets requires_grad on views, so non-view tensors keep requires_grad=False.
CheckpointFunction.apply then receives no grad-requiring inputs, producing
an output with requires_grad=False. This breaks the entire backward chain:
backward_step sees output_tensor[0].requires_grad=False, skips backward(),
so no gradients ever flow and training stalls at grad_norm=0.000.

Fix: After make_viewless_tensor, unconditionally call requires_grad_() on
hidden_states. This is safe because:
  1. In non-LoRA training, hidden_states already has requires_grad=True
     (from the embedding weight), so this is a no-op.
  2. In LoRA training with checkpointing, this ensures CheckpointFunction
     builds a proper autograd graph through its recompute-in-backward path.
"""

import sys


def patch(megatron_root):
    target = f"{megatron_root}/megatron/core/transformer/transformer_block.py"
    with open(target) as f:
        src = f.read()

    marker = "# LUMEN-PATCH-REQUIRES-GRAD"
    if marker in src:
        print("[patch_requires_grad] Already patched")
        return

    old = "        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)\n"

    # Find the FIRST occurrence (in TransformerBlock.forward, before _checkpointed_forward)
    idx = src.find(old)
    if idx == -1:
        print("[patch_requires_grad] WARNING: make_viewless_tensor target not found")
        return

    inject = (
        f"\n        {marker}\n"
        "        # LoRA fine-tuning: embedding is frozen so hidden_states.requires_grad=False.\n"
        "        # Force requires_grad=True so CheckpointFunction builds autograd graph.\n"
        "        if not hidden_states.requires_grad:\n"
        "            hidden_states = hidden_states.detach().requires_grad_(True)\n"
    )

    src = src[: idx + len(old)] + inject + src[idx + len(old) :]

    with open(target, "w") as f:
        f.write(src)
    print("[patch_requires_grad] Patched: force hidden_states.requires_grad_(True) " "before _checkpointed_forward")


if __name__ == "__main__":
    patch(sys.argv[1])
