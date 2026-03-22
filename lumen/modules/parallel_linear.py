###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Tensor-parallel linear modules using Lumen FP8 GEMM kernels.

These use Megatron's own TP communication primitives (autograd-aware
all-gather, reduce-scatter, etc.) and route the GEMM through Lumen's
:func:`~lumen.ops.quantize.linear.quantized_linear`.

When ``scaling_type="none"`` (the default) and ``delay_wgrad=False``,
plain BF16 ``F.linear`` is used.  When ``delay_wgrad=True``, even BF16
routes through ``quantized_linear`` (with ``scaling_type="none"``) so
that the autograd backward can defer wgrad computation.

To enable FP8, call :func:`enable_fp8` on the module or set
``scaling_type`` to one of the supported quantization modes.
"""

import warnings
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
    set_tensor_model_parallel_attributes,
)
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from torch.nn.parameter import Parameter

__all__ = ["LumenColumnParallelLinear", "LumenRowParallelLinear", "_DeferredWgrad"]


class _DeferredWgrad:
    """Stores and executes deferred weight-gradient computations.

    Used with ``delay_wgrad=True`` in parallel linear modules to split the
    backward pass into dgrad (immediate) and wgrad (deferred).  The deferred
    wgrad GEMM can then overlap with the next layer's communication.

    Each ``defer()`` call stores a self-contained closure that computes the
    wgrad and accumulates it into the correct gradient buffer (``main_grad``
    or ``weight.grad``).  ``execute()`` runs the pending closure.
    """

    def __init__(self):
        self._pending = None

    def defer(self, fn_or_weight, compute_fn=None):
        """Store a deferred wgrad closure.

        Two calling conventions are supported:

        1. ``defer(fn)`` — ``fn()`` is a self-contained closure that
           computes the weight gradient **and** accumulates it into the
           correct buffer.  Used by the autograd backward.

        2. ``defer(weight, compute_fn)`` — legacy / benchmark API.
           ``compute_fn()`` returns the gradient tensor.  ``execute()``
           will accumulate it into ``weight.main_grad`` or ``weight.grad``.
        """
        if compute_fn is not None:
            weight = fn_or_weight

            # Legacy wrapper: unconditionally targets main_grad when present.
            # Autograd closures gate on gradient_accumulation_fusion internally.
            def _wrapped():
                gw = compute_fn()
                if hasattr(weight, "main_grad"):
                    weight.main_grad.add_(gw)
                elif weight.grad is not None:
                    weight.grad.add_(gw)
                else:
                    weight.grad = gw

            self._pending = _wrapped
        else:
            self._pending = fn_or_weight

    def execute(self):
        """Run the pending wgrad computation, if any."""
        if self._pending is not None:
            self._pending()
            self._pending = None

    @property
    def has_pending(self):
        return self._pending is not None


def _use_sdma_from_args() -> bool:
    """Read ``--use-sdma`` from Megatron args (safe fallback to False)."""
    try:
        from megatron.training import get_args

        return getattr(get_args(), "use_sdma", False)
    except Exception:
        return False


def _tp_comm_overlap_from_args() -> bool:
    """Read ``--lumen-tp-comm-overlap`` from args (safe fallback to False)."""
    try:
        from megatron.training import get_args

        return getattr(get_args(), "lumen_tp_comm_overlap", False)
    except Exception:
        return False


def _get_tp_group(tp_group, is_expert):
    if tp_group is not None:
        return tp_group
    from megatron.core.tensor_parallel.layers import get_tensor_model_parallel_group_if_none

    return get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)


def _pg_size(group):
    if group is None or not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group=group)


def _pg_rank(group):
    if group is None or not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank(group=group)


def _do_gemm(
    input_,
    weight,
    bias,
    scaling_manager,
    scaling_type,
    fp8_dtype,
    block_size,
    gradient_accumulation_fusion=False,
    delay_wgrad=False,
    deferred_wgrad=None,
):
    """Route to Lumen FP8 GEMM or standard F.linear.

    When ``delay_wgrad=True``, always routes through
    :func:`~lumen.ops.quantize.linear.quantized_linear` (even for BF16)
    so that the autograd backward can defer the wgrad computation.
    """
    from lumen.ops.quantize.linear import quantized_linear

    if scaling_type != "none" or delay_wgrad:
        return quantized_linear(
            input_,
            weight,
            bias,
            scaling_manager=scaling_manager,
            scaling_type=scaling_type,
            fp8_dtype=fp8_dtype,
            block_size=block_size,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            delay_wgrad=delay_wgrad,
            deferred_wgrad=deferred_wgrad,
        )
    return F.linear(input_, weight, bias)


# ---------------------------------------------------------------------------
# ColumnParallelLinear
# ---------------------------------------------------------------------------


class LumenColumnParallelLinear(nn.Module):
    """Column-parallel linear using Lumen GEMM.

    Weight is ``[output_size // tp_size, input_size]``.
    Output is ``[*, output_size // tp_size]`` (each TP rank holds a shard).

    Can be used directly in Megatron layer specs.
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

        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        self.use_fsdp2 = use_fsdp2

        self.tp_group = _get_tp_group(tp_group, is_expert)
        self.tp_size = _pg_size(self.tp_group)
        tp_rank = _pg_rank(self.tp_group)

        self.expert_parallel = getattr(config, "expert_model_parallel_size", 1) > 1
        self.explicit_expert_comm = is_expert and (self.tp_size > 1 or self.expert_parallel)
        self.output_size_per_partition = divide(output_size, self.tp_size)

        self.sequence_parallel = getattr(config, "sequence_parallel", False)
        if self.sequence_parallel and self.tp_size <= 1:
            warnings.warn("sequence_parallel disabled: tp_size <= 1")
            self.sequence_parallel = False

        self.allreduce_dgrad = self.tp_size > 1 and not self.sequence_parallel and not self.explicit_expert_comm

        self.use_sdma = _use_sdma_from_args()
        self.tp_comm_overlap = getattr(config, "lumen_tp_comm_overlap", False) or _tp_comm_overlap_from_args()
        self._sdma_comm = None

        # FP8 config (disabled by default; enabled via enable_fp8())
        self.scaling_type = "none"
        self.scaling_manager = None
        self.fp8_dtype = torch.float8_e4m3fn
        self.block_size = 128
        self.gradient_accumulation_fusion = False
        self.delay_wgrad = False
        self._deferred_wgrad = _DeferredWgrad()

        # Weight allocation
        if not skip_weight_param_allocation:
            if getattr(config, "use_cpu_initialization", False):
                self.weight = Parameter(
                    torch.empty(self.output_size_per_partition, input_size, dtype=config.params_dtype)
                )
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
                        is_expert=is_expert,
                    )
            setattr(self.weight, "allreduce", not (is_expert and self.expert_parallel))
        else:
            self.weight = None

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
            setattr(self.bias, "allreduce", not (is_expert and self.expert_parallel))
        else:
            self.register_parameter("bias", None)

        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(f"{prefix}_extra_state")
        )

    def _get_sdma_comm(self):
        if self._sdma_comm is None:
            from lumen.modules.sdma_comm import SdmaTpComm

            self._sdma_comm = SdmaTpComm.get(self.tp_group)
        return self._sdma_comm

    def forward(self, input_: torch.Tensor, weight=None, runtime_gather_output=None):
        if weight is None:
            weight = self.weight

        if (
            self.use_sdma
            and self.tp_size > 1
            and self.tp_comm_overlap
            and self.sequence_parallel
            and not self.explicit_expert_comm
        ):
            output_parallel, gemm_bias = self._forward_sdma_overlap_column(input_, weight)
        else:
            if self.use_sdma and self.tp_size > 1:
                input_parallel = self._forward_sdma_pre_gemm(input_)
            else:
                if self.allreduce_dgrad or self.sequence_parallel or self.explicit_expert_comm:
                    input_parallel = input_
                else:
                    input_parallel = copy_to_tensor_model_parallel_region(input_, group=self.tp_group)

                if self.sequence_parallel and not self.explicit_expert_comm:
                    input_parallel = gather_from_sequence_parallel_region(
                        input_parallel,
                        tensor_parallel_output_grad=True,
                        group=self.tp_group,
                    )

            gemm_bias = self.bias if not self.skip_bias_add else None
            output_parallel = _do_gemm(
                input_parallel,
                weight,
                gemm_bias,
                self.scaling_manager,
                self.scaling_type,
                self.fp8_dtype,
                self.block_size,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                delay_wgrad=self.delay_wgrad,
                deferred_wgrad=self._deferred_wgrad if self.delay_wgrad else None,
            )

        gather = self.gather_output
        if runtime_gather_output is not None:
            gather = runtime_gather_output

        if self.use_sdma and self.tp_size > 1 and gather:
            from lumen.modules.sdma_comm import sdma_gather_from_tensor_model_parallel_region

            output = sdma_gather_from_tensor_model_parallel_region(
                output_parallel,
                self._get_sdma_comm(),
            )
        elif gather:
            output = gather_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def _forward_sdma_pre_gemm(self, input_: torch.Tensor) -> torch.Tensor:
        """SDMA-based input preparation for column-parallel GEMM."""
        from lumen.modules.sdma_comm import (
            sdma_copy_to_tensor_model_parallel_region,
            sdma_gather_from_sequence_parallel_region,
        )

        comm = self._get_sdma_comm()
        if self.allreduce_dgrad or self.sequence_parallel or self.explicit_expert_comm:
            input_parallel = input_
        else:
            input_parallel = sdma_copy_to_tensor_model_parallel_region(input_, comm)

        if self.sequence_parallel and not self.explicit_expert_comm:
            input_parallel = sdma_gather_from_sequence_parallel_region(input_parallel, comm)

        return input_parallel

    def _forward_sdma_overlap_column(
        self, input_: torch.Tensor, weight: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Overlap allgather with GEMM: compute local-shard GEMM while allgather runs."""
        comm = self._get_sdma_comm()
        if getattr(self, "_sdma_stream", None) is None:
            self._sdma_stream = torch.cuda.Stream(device=input_.device)
        sdma_stream = self._sdma_stream
        compute_stream = torch.cuda.current_stream(input_.device)

        input_parallel = input_.contiguous()
        local_dim0 = input_parallel.shape[0]
        comm.allgather_dim0_async(input_parallel, stream=sdma_stream)

        gemm_bias = self.bias if not self.skip_bias_add else None
        out_local = _do_gemm(
            input_parallel,
            weight,
            gemm_bias,
            self.scaling_manager,
            self.scaling_type,
            self.fp8_dtype,
            self.block_size,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            delay_wgrad=self.delay_wgrad,
            deferred_wgrad=self._deferred_wgrad if self.delay_wgrad else None,
        )

        input_gathered = comm.wait_allgather_dim0(stream=sdma_stream)
        compute_stream.wait_stream(sdma_stream)

        out_remaining = _do_gemm(
            input_gathered[local_dim0:, ...],
            weight,
            None,
            self.scaling_manager,
            self.scaling_type,
            self.fp8_dtype,
            self.block_size,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            delay_wgrad=self.delay_wgrad,
            deferred_wgrad=self._deferred_wgrad if self.delay_wgrad else None,
        )
        output_parallel = torch.cat([out_local, out_remaining], dim=0)
        return output_parallel, self.bias if self.skip_bias_add else None

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

    def execute_deferred_wgrad(self):
        """Execute any deferred weight gradient computation."""
        self._deferred_wgrad.execute()

    def backward_dw(self):
        """Megatron-compatible API: execute deferred weight gradient.

        Megatron's fine-grained 1F1B scheduler calls ``module.backward_dw()``
        to run the previously deferred wgrad GEMM.  This is a thin wrapper
        around :meth:`execute_deferred_wgrad`.
        """
        self._deferred_wgrad.execute()

    def __repr__(self):
        return (
            f"{type(self).__name__}(in={self.input_size}, out={self.output_size}, "
            f"bias={self.bias is not None}, TP={self.tp_size}, "
            f"fp8={self.scaling_type})"
        )


# ---------------------------------------------------------------------------
# RowParallelLinear
# ---------------------------------------------------------------------------


class LumenRowParallelLinear(nn.Module):
    """Row-parallel linear using Lumen GEMM.

    Weight is ``[output_size, input_size // tp_size]``.
    Input is already split across TP ranks; output is all-reduced.

    Row-parallel linear using Lumen GEMM.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config,
        init_method: Callable,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        use_fsdp2: bool = False,
    ):
        super().__init__()

        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        self.use_fsdp2 = use_fsdp2

        self.tp_group = _get_tp_group(tp_group, is_expert)
        self.tp_size = _pg_size(self.tp_group)
        tp_rank = _pg_rank(self.tp_group)

        self.expert_parallel = getattr(config, "expert_model_parallel_size", 1) > 1
        self.explicit_expert_comm = is_expert and (self.tp_size > 1 or self.expert_parallel)
        self.input_size_per_partition = divide(input_size, self.tp_size)

        self.sequence_parallel = getattr(config, "sequence_parallel", False)
        if self.sequence_parallel and not input_is_parallel:
            raise RuntimeError("sequence_parallel requires input_is_parallel=True")

        self.use_sdma = _use_sdma_from_args()
        self.tp_comm_overlap = getattr(config, "lumen_tp_comm_overlap", False) or _tp_comm_overlap_from_args()
        self._sdma_comm = None

        # FP8 config
        self.scaling_type = "none"
        self.scaling_manager = None
        self.fp8_dtype = torch.float8_e4m3fn
        self.block_size = 128
        self.gradient_accumulation_fusion = False
        self.delay_wgrad = False
        self._deferred_wgrad = _DeferredWgrad()

        # Weight
        if getattr(config, "use_cpu_initialization", False):
            self.weight = Parameter(torch.empty(output_size, self.input_size_per_partition, dtype=config.params_dtype))
            if getattr(config, "perform_initialization", True):
                from megatron.core.tensor_parallel.layers import condition_init_method

                _initialize_affine_weight_cpu(
                    self.weight,
                    output_size,
                    input_size,
                    self.input_size_per_partition,
                    1,
                    condition_init_method(config, init_method),
                    stride=1,
                    return_master_weight=False,
                    params_dtype=config.params_dtype,
                    rank=tp_rank,
                    world_size=self.tp_size,
                    skip_set_tensor_parallel_attributes=True,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    output_size,
                    self.input_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if getattr(config, "perform_initialization", True):
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=1,
                    stride=1,
                    is_expert=is_expert,
                )
        setattr(self.weight, "allreduce", not (is_expert and self.expert_parallel))

        if bias:
            if getattr(config, "use_cpu_initialization", False):
                self.bias = Parameter(torch.empty(output_size, dtype=config.params_dtype))
            else:
                self.bias = Parameter(
                    torch.empty(
                        output_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
            if getattr(config, "perform_initialization", True):
                with torch.no_grad():
                    self.bias.zero_()
            setattr(self.bias, "allreduce", not (is_expert and self.expert_parallel))
            setattr(self.bias, "sequence_parallel", self.sequence_parallel)
        else:
            self.register_parameter("bias", None)

        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(f"{prefix}_extra_state")
        )

    def _get_sdma_comm(self):
        if self._sdma_comm is None:
            from lumen.modules.sdma_comm import SdmaTpComm

            self._sdma_comm = SdmaTpComm.get(self.tp_group)
        return self._sdma_comm

    def forward(self, input_: torch.Tensor):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            from megatron.core.tensor_parallel.mappings import scatter_to_tensor_model_parallel_region

            input_parallel = scatter_to_tensor_model_parallel_region(input_, group=self.tp_group)

        output_parallel = _do_gemm(
            input_parallel,
            self.weight,
            None,
            self.scaling_manager,
            self.scaling_type,
            self.fp8_dtype,
            self.block_size,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            delay_wgrad=self.delay_wgrad,
            deferred_wgrad=self._deferred_wgrad if self.delay_wgrad else None,
        )

        if self.explicit_expert_comm:
            output_ = output_parallel
        elif self.use_sdma and self.tp_size > 1 and self.tp_comm_overlap:
            output_ = self._forward_sdma_overlap_row(output_parallel)
        elif self.use_sdma and self.tp_size > 1:
            output_ = self._forward_sdma_post_gemm(output_parallel)
        elif self.sequence_parallel:
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel, group=self.tp_group)
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)

        if not self.skip_bias_add:
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias

    def _forward_sdma_post_gemm(self, output_parallel: torch.Tensor) -> torch.Tensor:
        """SDMA-based output reduction for row-parallel GEMM."""
        from lumen.modules.sdma_comm import (
            sdma_reduce_from_tensor_model_parallel_region,
            sdma_reduce_scatter_to_sequence_parallel_region,
        )

        comm = self._get_sdma_comm()
        if self.sequence_parallel:
            return sdma_reduce_scatter_to_sequence_parallel_region(output_parallel, comm)
        else:
            return sdma_reduce_from_tensor_model_parallel_region(output_parallel, comm)

    def _forward_sdma_overlap_row(self, output_parallel: torch.Tensor) -> torch.Tensor:
        """Overlap reduce-scatter / allreduce with GEMM via async SDMA."""
        comm = self._get_sdma_comm()
        if getattr(self, "_sdma_stream", None) is None:
            self._sdma_stream = torch.cuda.Stream(device=output_parallel.device)
        sdma_stream = self._sdma_stream
        compute_stream = torch.cuda.current_stream(output_parallel.device)
        sdma_stream.wait_stream(compute_stream)

        if self.sequence_parallel:
            comm.reduce_scatter_dim0_async(output_parallel, stream=sdma_stream)
            return comm.wait_reduce_scatter_dim0(stream=sdma_stream)
        else:
            comm.allreduce_sum_async(output_parallel, stream=sdma_stream)
            comm.wait_allreduce_sum(stream=sdma_stream)
            return output_parallel

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
            {"weight": 1},
            sharded_offsets,
        )

    def set_extra_state(self, state):
        pass

    def get_extra_state(self):
        return None

    def execute_deferred_wgrad(self):
        """Execute any deferred weight gradient computation."""
        self._deferred_wgrad.execute()

    def backward_dw(self):
        """Megatron-compatible API: execute deferred weight gradient.

        Megatron's fine-grained 1F1B scheduler calls ``module.backward_dw()``
        to run the previously deferred wgrad GEMM.  This is a thin wrapper
        around :meth:`execute_deferred_wgrad`.
        """
        self._deferred_wgrad.execute()

    def __repr__(self):
        return (
            f"{type(self).__name__}(in={self.input_size}, out={self.output_size}, "
            f"bias={self.bias is not None}, TP={self.tp_size}, "
            f"fp8={self.scaling_type})"
        )
