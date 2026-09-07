import pytest
import torch
import torch.nn as nn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
def test_blockwise2d_linear_accepts_zero_token_batch():
    from lumen.quantize import disable, enable
    from lumen.quantize.config import QuantConfig

    module = nn.Linear(256, 128, bias=False).cuda().to(torch.bfloat16)
    enable(
        module,
        config=QuantConfig.from_str(scaling="blockwise2d", block_size=128),
    )
    empty = torch.empty(
        0, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    output = module(empty)
    assert output.shape == (0, 128)
    output.sum().backward()
    assert empty.grad is not None and empty.grad.shape == empty.shape
    assert module.weight.grad is not None
    torch.testing.assert_close(
        module.weight.grad, torch.zeros_like(module.weight.grad)
    )
    disable(module)
