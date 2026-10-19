"""Independent decode for the frozen E2M1/E8M0 formats, used only in GPU tests."""

import torch


def decode(payload, scales, rows, cols, row_group=1):
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=payload.device
    )
    codes = torch.stack((payload & 15, payload >> 4), dim=-1).reshape(rows, cols).long()
    magnitude = values[codes & 7]
    sign = torch.where(codes & 8 != 0, -1.0, 1.0)
    expanded = scales.float().repeat_interleave(row_group, 0).repeat_interleave(32, 1)
    return magnitude * sign * torch.pow(2.0, expanded - 127)


def assert_metrics(reference, result, floor=30):
    difference = reference.float() - result.float()
    signal = reference.float().square().sum()
    noise = difference.square().sum()
    assert noise == 0 or 10 * torch.log10(signal / noise) >= floor
    assert (difference.abs() <= 0.5 + 0.02 * reference.abs()).float().mean() >= 0.99
