###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Grouped-linear (MoE) modules with tensor-parallel support.

For MoE, TP communication is handled by the MoE token dispatcher — the
grouped linear itself does **not** perform all-gather / reduce-scatter.
When ``is_expert=True`` and TP > 1 (or EP > 1), the weight dimensions
are sharded manually and ``parallel_mode`` is set to ``None`` so the
module's internal communication is skipped.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from torch.nn.parameter import Parameter

from lumen.modules.parallel_linear import _DeferredWgrad, _get_tp_group, _pg_size

__all__ = [
    "LumenGroupedLinear",
    "LumenColumnParallelGroupedLinear",
    "LumenRowParallelGroupedLinear",
]


class LumenGroupedLinear(nn.Module):
    """Grouped linear for MoE: ``num_gemms`` independent linear layers
    whose weights are stored as ``nn.ParameterList``.

    Forward signature: ``forward(x, m_splits)`` where ``m_splits`` is a
    list of per-expert token counts.
    """

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: Optional[str],
        config,
        init_method: Callable,
        bias: bool = True,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__()

        self.config = config
        self.num_gemms = num_gemms
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert

        self.tp_group = _get_tp_group(tp_group, is_expert)
        tp_size = _pg_size(self.tp_group)

        self.expert_parallel = getattr(config, "expert_model_parallel_size", 1) > 1
        self.explicit_expert_comm = is_expert and (tp_size > 1 or self.expert_parallel)

        if self.explicit_expert_comm:
            if parallel_mode == "column":
                output_size = divide(output_size, tp_size)
            elif parallel_mode == "row":
                input_size = divide(input_size, tp_size)

        self.in_features = input_size
        self.out_features = output_size

        # FP8 config
        self.scaling_type = "none"
        self.scaling_manager = None
        from lumen.quantize.config import _get_float8_e4m3

        self.fp8_dtype = _get_float8_e4m3()
        self.block_size = 128
        self.gradient_accumulation_fusion = False
        self.delay_wgrad = False
        self.fp8_activation_store = False
        self._deferred_wgrad = _DeferredWgrad()

        # Per-expert weights
        self.weights = nn.ParameterList(
            [
                Parameter(
                    torch.empty(
                        output_size,
                        input_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
                for _ in range(num_gemms)
            ]
        )
        if getattr(config, "perform_initialization", True):
            for w in self.weights:
                init_method(w)

        if bias:
            self.biases = nn.ParameterList(
                [
                    Parameter(
                        torch.zeros(
                            output_size,
                            device=torch.cuda.current_device(),
                            dtype=config.params_dtype,
                        )
                    )
                    for _ in range(num_gemms)
                ]
            )
        else:
            self.biases = None

        for param in self.parameters():
            setattr(param, "allreduce", not (is_expert and self.expert_parallel))

        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(f"{prefix}_extra_state")
        )

    def forward(self, x: torch.Tensor, m_splits, m_splits_gpu=None):
        """Forward: sequentially process each expert's tokens.

        Args:
            x: [total_tokens, in_features]
            m_splits: list/tensor of per-expert token counts.
        Returns:
            (output, bias)
        """
        outputs = []
        offset = 0
        for i in range(self.num_gemms):
            count = int(m_splits[i]) if not isinstance(m_splits[i], int) else m_splits[i]
            if count == 0:
                continue
            xi = x[offset : offset + count]
            bias_i = None if self.skip_bias_add else (self.biases[i] if self.biases else None)
            if self.scaling_type != "none" or self.delay_wgrad:
                from lumen.ops.quantize.linear import quantized_linear

                yi = quantized_linear(
                    xi,
                    self.weights[i],
                    bias_i,
                    scaling_manager=self.scaling_manager,
                    scaling_type=self.scaling_type,
                    fp8_dtype=self.fp8_dtype,
                    block_size=self.block_size,
                    gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                    delay_wgrad=self.delay_wgrad,
                    deferred_wgrad=self._deferred_wgrad if self.delay_wgrad else None,
                )
            else:
                yi = F.linear(xi, self.weights[i], bias_i)
            outputs.append(yi)
            offset += count

        if not outputs:
            output = x.new_empty(0, self.out_features)
        else:
            output = torch.cat(outputs, dim=0)

        if self.skip_bias_add and self.biases is not None:
            output_bias = torch.cat([b.unsqueeze(0) for b in self.biases], dim=0)
        else:
            output_bias = None
        return output, output_bias

    def enable_fp8(self, scaling_manager=None, scaling_type="dynamic", fp8_dtype=None, block_size=None):
        from lumen.quantize import QuantConfig, ScalingManager

        self.scaling_type = scaling_type
        self.scaling_manager = scaling_manager or ScalingManager(QuantConfig())
        if fp8_dtype is not None:
            self.fp8_dtype = fp8_dtype
        if block_size is not None:
            self.block_size = block_size

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(state_dict, prefix, {}, sharded_offsets)

    def set_extra_state(self, state):
        pass

    def get_extra_state(self):
        return None

    def execute_deferred_wgrad(self):
        """Execute any deferred weight gradient computation."""
        self._deferred_wgrad.execute()

    def backward_dw(self):
        """Megatron-compatible API: execute deferred weight gradient."""
        self._deferred_wgrad.execute()


class LumenColumnParallelGroupedLinear(LumenGroupedLinear):
    """Column-parallel grouped linear for MoE."""

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        config,
        init_method: Callable,
        bias: bool = True,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(
            num_gemms=num_gemms,
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group,
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        state_dict = self.state_dict(prefix="", keep_vars=True)
        shard_map = {f"weights.{i}": 0 for i in range(self.num_gemms)}
        if self.biases is not None:
            shard_map.update({f"biases.{i}": 0 for i in range(self.num_gemms)})
        return make_sharded_tensors_for_checkpoint(state_dict, prefix, shard_map, sharded_offsets)


class LumenRowParallelGroupedLinear(LumenGroupedLinear):
    """Row-parallel grouped linear for MoE."""

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        config,
        init_method: Callable,
        bias: bool = True,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(
            num_gemms=num_gemms,
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group,
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        state_dict = self.state_dict(prefix="", keep_vars=True)
        shard_map = {f"weights.{i}": 1 for i in range(self.num_gemms)}
        return make_sharded_tensors_for_checkpoint(state_dict, prefix, shard_map, sharded_offsets)
