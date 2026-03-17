###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Tests for lumen.ops.cross_entropy: vocab-parallel cross-entropy.

Covers:
  - Forward — compare against cross_entropy_ref from conftest
  - Forward + backward — compare gradient on logits
  - Label smoothing
"""

import pytest
import torch
from conftest import compute_snr, cross_entropy_ref

from lumen.ops.cross_entropy import parallel_cross_entropy

# ---------------------------------------------------------------------------
# Parametrize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch,vocab", [(32, 32000), (128, 50000)], ids=["b32_v32k", "b128_v50k"])
def test_cross_entropy_fwd(batch, vocab):
    """Compare cross-entropy forward against cross_entropy_ref from conftest."""
    seq_len = 1  # parallel_cross_entropy expects [B, SQ, V]
    device = "cuda"
    dtype = torch.bfloat16

    logits = torch.randn(batch, seq_len, vocab, device=device, dtype=dtype) * 0.1
    target = torch.randint(0, vocab, (batch, seq_len), device=device, dtype=torch.long)

    loss_ref = cross_entropy_ref(logits, target, label_smoothing=0.0, ignore_idx=-100)
    loss_ref = loss_ref.reshape(batch, seq_len)

    loss_lumen = parallel_cross_entropy(
        logits,
        target,
        label_smoothing=0.0,
        reduce_loss=False,
        dist_process_group=None,
        ignore_idx=-100,
    )

    snr = compute_snr(loss_ref.float(), loss_lumen.float())
    assert snr > 20, f"Cross-entropy fwd SNR: {snr:.1f} dB (expected > 20)"


@pytest.mark.parametrize("batch,vocab", [(32, 32000), (128, 50000)], ids=["b32_v32k", "b128_v50k"])
def test_cross_entropy_fwd_bwd(batch, vocab):
    """Forward + backward, compare gradient on logits."""
    seq_len = 1
    device = "cuda"
    dtype = torch.bfloat16

    logits = torch.randn(batch, seq_len, vocab, device=device, dtype=dtype, requires_grad=True) * 0.1
    target = torch.randint(0, vocab, (batch, seq_len), device=device, dtype=torch.long)

    logits_ref = logits.detach().clone().requires_grad_(True)

    loss_ref = cross_entropy_ref(logits_ref, target, label_smoothing=0.0, ignore_idx=-100)
    loss_ref.mean().backward()

    loss_lumen = parallel_cross_entropy(
        logits,
        target,
        label_smoothing=0.0,
        reduce_loss=False,
        dist_process_group=None,
        ignore_idx=-100,
    )
    loss_lumen.mean().backward()

    grad_snr = compute_snr(logits_ref.grad, logits.grad)
    assert grad_snr > 15, f"Cross-entropy bwd grad SNR: {grad_snr:.1f} dB (expected > 15)"


@pytest.mark.parametrize("batch,vocab", [(32, 32000), (128, 50000)], ids=["b32_v32k", "b128_v50k"])
def test_cross_entropy_label_smoothing(batch, vocab):
    """Cross-entropy with label_smoothing > 0."""
    seq_len = 1
    device = "cuda"
    dtype = torch.bfloat16
    label_smoothing = 0.1

    logits = torch.randn(batch, seq_len, vocab, device=device, dtype=dtype) * 0.1
    target = torch.randint(0, vocab, (batch, seq_len), device=device, dtype=torch.long)

    loss_ref = cross_entropy_ref(logits, target, label_smoothing=label_smoothing, ignore_idx=-100)
    loss_ref = loss_ref.reshape(batch, seq_len)

    loss_lumen = parallel_cross_entropy(
        logits,
        target,
        label_smoothing=label_smoothing,
        reduce_loss=False,
        dist_process_group=None,
        ignore_idx=-100,
    )

    snr = compute_snr(loss_ref.float(), loss_lumen.float())
    assert snr > 15, f"Cross-entropy label_smoothing SNR: {snr:.1f} dB (expected > 15)"
