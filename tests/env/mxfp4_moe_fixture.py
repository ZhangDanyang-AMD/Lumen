"""Real pinned Core fixture builder; no process/resource launch at import."""

import json
from pathlib import Path

import torch


def signed_config(contract_path, *, ep, overlap):
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig

    record = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    values = dict(record["fixture"]["effective_core_config"])
    values["params_dtype"] = torch.bfloat16
    values["activation_func"] = torch.nn.functional.silu
    config = TransformerConfig(
        num_layers=2,
        num_attention_heads=32,
        tensor_model_parallel_size=ep,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=1,
        sequence_parallel=True,
        **values
    )
    for name, expected in values.items():
        actual = getattr(config, name)
        assert (
            actual is expected
            if callable(expected) or isinstance(expected, torch.dtype)
            else actual == expected
        )
        if (
            expected is not None
            and not callable(expected)
            and not isinstance(expected, torch.dtype)
        ):
            assert type(actual) is type(expected), name
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=overlap,
        num_distributed_optimizer_instances=1,
        overlap_param_gather=False,
        delay_wgrad_compute=False,
    )
    return config, ddp_config


def build_core_layer(config):
    from megatron.core.transformer.mlp import MLPSubmodules
    from megatron.core.transformer.moe.experts import TEGroupedMLP
    from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
    from megatron.core.transformer.spec_utils import ModuleSpec

    from lumen.modules.grouped_linear import (
        LumenColumnParallelGroupedLinear,
        LumenRowParallelGroupedLinear,
    )

    expert = ModuleSpec(
        module=TEGroupedMLP,
        submodules=MLPSubmodules(
            linear_fc1=LumenColumnParallelGroupedLinear,
            linear_fc2=LumenRowParallelGroupedLinear,
        ),
    )
    return MoELayer(
        config=config, submodules=MoESubmodules(experts=expert), layer_number=1
    )
