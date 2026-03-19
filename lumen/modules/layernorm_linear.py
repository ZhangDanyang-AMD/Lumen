###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Fused LayerNorm/RMSNorm + column-parallel linear.

Applies normalization, optional FP8 quantization, then a column-parallel
GEMM — all in one module with a single set of TP communication calls.

The norm runs on the *local* (possibly sequence-parallel) input; the
all-gather (if sequence_parallel) happens *after* the norm but *before*
the GEMM.
"""

import warnings
from typing import Callable, Optional

import torch
import torch.nn as nn
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
    set_tensor_model_parallel_attributes,
)
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from torch.nn.parameter import Parameter

from lumen.modules.parallel_linear import (
    _do_gemm,
    _get_tp_group,
    _pg_rank,
    _pg_size,
    _use_sdma_from_args,
)

__all__ = ["LumenLayerNormLinear"]


class LumenLayerNormLinear(nn.Module):
    """Fused Norm + ColumnParallelLinear using Lumen GEMM.

    Norm type is selected via ``config.normalization`` (``"LayerNorm"`` or
    ``"RMSNorm"``).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config,
        init_method: Callable,
        gather_output: bool = False,
        bias: bool = True,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        use_fsdp2: bool = False,
    ):
        super().__init__()
        self.use_fsdp2 = use_fsdp2

        if gather_output:
            raise ValueError("LumenLayerNormLinear does not support gather_output=True")
        if is_expert:
            raise ValueError("LumenLayerNormLinear does not support MoE experts")
        if skip_weight_param_allocation:
            raise ValueError("LumenLayerNormLinear does not support skip_weight_param_allocation")

        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add

        self.tp_group = _get_tp_group(tp_group, is_expert=False)
        self.tp_size = _pg_size(self.tp_group)
        tp_rank = _pg_rank(self.tp_group)
        self.output_size_per_partition = divide(output_size, self.tp_size)

        self.sequence_parallel = getattr(config, "sequence_parallel", False)
        if self.sequence_parallel and self.tp_size <= 1:
            warnings.warn("sequence_parallel disabled: tp_size <= 1")
            self.sequence_parallel = False

        self.use_sdma = _use_sdma_from_args()
        self._sdma_comm = None

        # Norm type
        normalization = getattr(config, "normalization", "LayerNorm")
        self.use_rmsnorm = normalization == "RMSNorm"
        eps = getattr(config, "layernorm_epsilon", 1e-5)
        zero_centered = getattr(config, "layernorm_zero_centered_gamma", False)

        # Norm parameters
        self.ln_weight = Parameter(
            torch.ones(input_size, dtype=config.params_dtype, device=torch.cuda.current_device())
        )
        if self.use_rmsnorm:
            self.register_parameter("ln_bias", None)
        else:
            self.ln_bias = Parameter(
                torch.zeros(input_size, dtype=config.params_dtype, device=torch.cuda.current_device())
            )
        self.ln_eps = eps
        self.zero_centered_gamma = zero_centered

        # FP8 config
        self.scaling_type = "none"
        self.scaling_manager = None
        self.fp8_dtype = torch.float8_e4m3fn
        self.block_size = 128

        # Linear weight [output_size_per_partition, input_size]
        if getattr(config, "use_cpu_initialization", False):
            self.weight = Parameter(torch.empty(self.output_size_per_partition, input_size, dtype=config.params_dtype))
            if getattr(config, "perform_initialization", True):
                from megatron.core.tensor_parallel.layers import condition_init_method

                _initialize_affine_weight_cpu(
                    self.weight,
                    output_size,
                    input_size,
                    self.output_size_per_partition,
                    0,
                    condition_init_method(config, init_method),
                    stride=1,
                    return_master_weight=False,
                    rank=tp_rank,
                    world_size=self.tp_size,
                    skip_set_tensor_parallel_attributes=True,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    input_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if getattr(config, "perform_initialization", True):
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=0,
                    stride=1,
                )
        setattr(self.weight, "allreduce", True)

        if bias:
            if getattr(config, "use_cpu_initialization", False):
                self.bias = Parameter(torch.empty(self.output_size_per_partition, dtype=config.params_dtype))
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
            set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
            if getattr(config, "perform_initialization", True):
                with torch.no_grad():
                    self.bias.zero_()
            setattr(self.bias, "allreduce", True)
        else:
            self.register_parameter("bias", None)

        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(f"{prefix}_extra_state")
        )

    def _norm(self, x):
        """Apply normalization."""
        w = self.ln_weight
        if self.zero_centered_gamma:
            w = w + 1.0
        if self.use_rmsnorm:
            from lumen.ops.normalization.rmsnorm import rmsnorm

            return rmsnorm(x, w, self.ln_eps)
        else:
            from lumen.ops.normalization.layernorm import layernorm

            return layernorm(x, w, self.ln_bias, self.ln_eps)

    def _get_sdma_comm(self):
        if self._sdma_comm is None:
            from lumen.modules.sdma_comm import SdmaTpComm

            self._sdma_comm = SdmaTpComm.get(self.tp_group)
        return self._sdma_comm

    def forward(self, x: torch.Tensor):
        """Forward: Norm → (all-gather if SP) → GEMM.

        Returns:
            (output, bias) where bias is ``None`` unless ``skip_bias_add``.
        """
        ln_out = self._norm(x)

        if self.sequence_parallel:
            if self.use_sdma and self.tp_size > 1:
                from lumen.modules.sdma_comm import sdma_gather_from_sequence_parallel_region

                ln_out = sdma_gather_from_sequence_parallel_region(
                    ln_out,
                    self._get_sdma_comm(),
                )
            else:
                ln_out = gather_from_sequence_parallel_region(
                    ln_out,
                    tensor_parallel_output_grad=True,
                    group=self.tp_group,
                )

        gemm_bias = self.bias if not self.skip_bias_add else None
        output = _do_gemm(
            ln_out,
            self.weight,
            gemm_bias,
            self.scaling_manager,
            self.scaling_type,
            self.fp8_dtype,
            self.block_size,
        )

        output_bias = self.bias if self.skip_bias_add else None
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
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {"weight": 0, "bias": 0},
            sharded_offsets,
        )

    def set_extra_state(self, state):
        pass

    def get_extra_state(self):
        return None

    def __repr__(self):
        norm = "RMSNorm" if self.use_rmsnorm else "LayerNorm"
        return (
            f"{type(self).__name__}({norm}, in={self.input_size}, out={self.output_size}, "
            f"bias={self.bias is not None}, TP={self.tp_size}, fp8={self.scaling_type})"
        )
