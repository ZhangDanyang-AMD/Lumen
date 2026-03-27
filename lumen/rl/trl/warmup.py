###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Synthetic warmup helper for TRL + Lumen RL training."""

import torch

from lumen.models.fsdp import reset_fp8_state

__all__ = ["maybe_run_synthetic_warmup"]


def maybe_run_synthetic_warmup(model, args, *, device):
    """Run synthetic warmup steps on a single device when enabled."""

    warmup_steps = getattr(args, "warmup_steps", 0)
    if warmup_steps <= 0:
        return 0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    batch_size = getattr(args, "micro_batch_size", 1)
    seq_length = getattr(args, "seq_length", 1)

    for _ in range(warmup_steps):
        optimizer.zero_grad()
        input_ids = torch.ones(batch_size, seq_length, dtype=torch.long, device=device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()
        optimizer.step()

    if getattr(args, "linear_fp8", False):
        reset_fp8_state(model)
    return warmup_steps
