"""Gated MLP partition stride reaches both weight initialization paths."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from lumen.modules.layernorm_linear import LumenLayerNormLinear


@pytest.mark.parametrize("cpu", [False, True])
def test_gated_partition_stride(cpu):
    config = SimpleNamespace(
        params_dtype=torch.bfloat16, perform_initialization=True,
        use_cpu_initialization=cpu, sequence_parallel=False,
        normalization="RMSNorm", layernorm_epsilon=1e-5,
        layernorm_zero_centered_gamma=False,
    )
    base = "lumen.modules.layernorm_linear."
    with (
        patch(base + "_get_tp_group", return_value=None),
        patch(base + "_pg_size", return_value=1),
        patch(base + "_pg_rank", return_value=0),
        patch(base + "_use_sdma_from_args", return_value=False),
        patch(base + "_initialize_affine_weight_gpu") as gpu_init,
        patch(base + "_initialize_affine_weight_cpu") as cpu_init,
        patch("megatron.core.tensor_parallel.layers.condition_init_method", side_effect=lambda c, f: f),
    ):
        LumenLayerNormLinear(64, 128, config=config, init_method=lambda w: w, stride=2)
        selected = cpu_init if cpu else gpu_init
        assert selected.call_args.kwargs["stride"] == 2
        assert not (gpu_init if cpu else cpu_init).called
