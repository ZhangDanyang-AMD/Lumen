"""Independent BF16 recipe reference; never calls AITER or production quantization."""

import torch


def h16_pad_transpose(x, *, stage_observer=None):
    m, width = x.shape
    p = (m + 31) // 32 * 32
    padded = torch.zeros((p, width), dtype=torch.bfloat16, device=x.device)
    padded[:m] = x
    # Separate stage tensors preserve the signed-zero and RN32 stage contract.
    values = padded.float().reshape(-1, 16, width) * 0.25
    for half in (1, 2, 4, 8):
        following = torch.empty_like(values)
        for start in range(0, 16, 2 * half):
            left = values[:, start : start + half]
            right = values[:, start + half : start + 2 * half]
            following[:, start : start + half] = left + right
            following[:, start + half : start + 2 * half] = left - right
        values = following
        if stage_observer is not None:
            stage_observer(values.detach().clone())
    return values.reshape(p, width).T.contiguous()


def quantize_reference(x, row_group=1, words=None, *, return_encoding=False):
    """RNE E2M1; compact E8M0 selector is independently evaluated in FP64."""
    rows, cols = x.shape
    if not torch.isfinite(x).all():
        raise ValueError(
            "finite recipe operand required; poison codec tested separately"
        )
    if x.numel() == 0:
        if words is not None and len(words):
            raise ValueError("empty operand reserves zero words")
        return x.float()
    groups = x.double().reshape(rows // row_group, row_group, cols // 32, 32)
    amax = groups.abs().amax((1, 3))
    # Independent interval selector: carry at 1.75 times the binade power.
    binade = torch.floor(torch.log2(torch.where(amax == 0, 1.0, amax)))
    carry = amax >= 1.75 * torch.pow(2.0, binade)
    exponent = torch.where(amax == 0, -127.0, binade + carry - 2).clamp(-127, 127)
    exponent = torch.where((binade == 127) & carry, 127.0, exponent)
    scale = (
        torch.pow(2.0, exponent)
        .repeat_interleave(row_group, 0)
        .repeat_interleave(32, 1)
    )
    # Exact power-of-two division in FP64 followed by one RN32 cast avoids
    # endpoint divisor underflow/double rounding in a two-step FP32 multiply.
    normalized = (x.double() / scale).float().double()
    if words is not None:
        from mxfp4_moe_sr_reference import sr_code

        if len(words) != x.numel():
            raise ValueError("one independent unsigned32 word per element required")
        codes = torch.tensor(
            [
                sr_code(value, word)
                for value, word in zip(normalized.cpu().flatten().tolist(), words)
            ],
            device=x.device,
        ).reshape(x.shape)
        magnitudes = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device
        )
        decoded = magnitudes[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0) * scale
        return (codes, exponent + 127, decoded) if return_encoding else decoded.float()
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device, dtype=torch.float64
    )
    distance = (normalized.abs().unsqueeze(-1) - values).abs()
    minimum = distance.amin(-1, keepdim=True)
    ties = distance == minimum
    priority = torch.tensor([0, 9, 2, 11, 4, 13, 6, 15], device=x.device)
    codes = torch.where(ties, priority, 100).argmin(-1)
    negative = torch.signbit(normalized)
    decoded = values[codes] * torch.where(negative, -1.0, 1.0) * scale
    signed_codes = codes | (negative.to(torch.int64) * 8)
    return (
        (signed_codes, exponent + 127, decoded) if return_encoding else decoded.float()
    )


def bf16_recipe(x, weight, dy):
    """Unquantized mathematical baselines, separately from the quantized recipe."""
    return (
        x.float() @ weight.float().T,
        dy.float() @ weight.float(),
        dy.float().T @ x.float(),
    )


def quantized_recipe(x, weight, dy, *, row_seed, row_base, h16_seed, h16_base):
    """Independent F/D/W recipe with supplied route-ready receipt seeds/bases."""
    from mxfp4_moe_sr_reference import philox_word

    qw = quantize_reference(weight, 32)
    qx = quantize_reference(x)
    row_words = [philox_word(row_seed, row_base + index) for index in range(dy.numel())]
    qdy = quantize_reference(dy, words=row_words)
    tx, tdy = h16_pad_transpose(x), h16_pad_transpose(dy)
    h16_words = [
        philox_word(h16_seed, h16_base + index) for index in range(tdy.numel())
    ]
    qtx = quantize_reference(tx)
    qtdy = quantize_reference(tdy, words=h16_words)
    return qx @ qw.T, qdy @ qw, qtdy @ qtx.T
