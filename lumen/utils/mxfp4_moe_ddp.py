# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0
"""Scoped first-order gradient handoff to existing Megatron DDP accumulators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from lumen.quantize.mxfp4_moe_state import (
    MainGradBinding,
    tensor_range,
    validate_disjoint_main_grads,
)
from lumen.utils.mxfp4_moe_ledger import Mxfp4StateError


@dataclass
class ZeroLease:
    binding: MainGradBinding
    emission: torch.Tensor
    sink: torch.Tensor
    zero_event: object
    w_event: object
    consumed_event: object
    armed: bool = False
    consumed: bool = False

    def validate(self):
        import torch

        b = self.binding
        b.validate(window=b.window, weight_epoch=b.weight_epoch)
        regions = [b.region, b.parameter_region]
        for value in (self.emission, self.sink):
            if (
                value.dtype != torch.bfloat16
                or tuple(value.shape) != b.shape
                or str(value.device) != b.device
                or not value.is_contiguous()
            ):
                raise Mxfp4StateError("ZERO_LEASE_MISMATCH")
            region = tensor_range(value)
            if any(
                region.key == prior.key
                and max(region.start, prior.start) < min(region.end, prior.end)
                for prior in regions
            ):
                raise Mxfp4StateError("ZERO_LEASE_MISMATCH", "alias")
            regions.append(region)


class Mxfp4DDPHandoff:
    """Install after DDP construction; never replace its hook or allocate main_grad."""

    def __init__(self, ddp, parameters, *, current_stream=None):
        if current_stream is None:
            from torch.cuda import current_stream
        self._current_stream = current_stream
        from megatron.core.distributed.distributed_data_parallel import (
            DistributedDataParallel,
        )

        if not isinstance(ddp, DistributedDataParallel):
            raise Mxfp4StateError("MAIN_GRAD_MISMATCH", "real Core DDP required")
        config = ddp.ddp_config
        if (
            not config.grad_reduce_in_fp32
            or config.num_distributed_optimizer_instances != 1
            or config.overlap_param_gather
            or config.delay_wgrad_compute
        ):
            raise Mxfp4StateError("UNSUPPORTED_DDP_CONFIGURATION")
        self.ddp = ddp
        self.parameters = tuple(parameters)
        self.owners, self.nodes, self.handles = {}, {}, []
        self.pending = {}
        self.leases = []
        self.window = None
        for parameter in self.parameters:
            if (
                not parameter.is_leaf
                or not parameter.requires_grad
                or getattr(parameter, "allreduce", True)
            ):
                raise Mxfp4StateError(
                    "MAIN_GRAD_MISMATCH", "routed dense leaf required"
                )
            if getattr(parameter, "_backward_hooks", None) or getattr(
                parameter, "_post_accumulate_grad_hooks", None
            ):
                raise Mxfp4StateError("UNKNOWN_GRADIENT_HOOK_SUBSCRIPTION")
            candidates = [
                buffer
                for buffer in ddp.expert_parallel_buffers
                if parameter in buffer.param_index_map
            ]
            if len(candidates) != 1 or parameter not in ddp.param_to_bucket_group:
                raise Mxfp4StateError(
                    "MAIN_GRAD_MISMATCH", "unique expert buffer required"
                )
            buffer = candidates[0]
            self.owners[parameter] = buffer
            node = parameter.expand_as(parameter).grad_fn.next_functions[0][0]
            if node not in ddp.grad_accs:
                raise Mxfp4StateError("MAIN_GRAD_MISMATCH", "original DDP hook absent")
            self.nodes[parameter] = node
            self.handles.append(parameter.register_hook(self._prehook(parameter)))
            self.handles.append(node.register_hook(self._observer(parameter)))

    def open_after_zero(self, window, epoch, zero_event):
        if self.window is not None or self.pending:
            raise Mxfp4StateError("ACCUMULATION_WINDOW_MISMATCH")
        zero_event.synchronize()
        bindings = []
        for parameter in self.parameters:
            if parameter.grad is not None or parameter.grad_added_to_main_grad:
                raise Mxfp4StateError(
                    "ACCUMULATION_WINDOW_MISMATCH", "DDP zero boundary required"
                )
            binding = MainGradBinding.capture(
                parameter,
                self.owners[parameter].grad_data,
                window=window,
                weight_epoch=epoch,
            )
            start, end, _ = self.owners[parameter].param_index_map[parameter]
            if binding.region.start != start * 4 or binding.region.end != end * 4:
                raise Mxfp4StateError("MAIN_GRAD_MISMATCH", "DDP indexed range differs")
            bindings.append(binding)
        validate_disjoint_main_grads(bindings)
        self.window = window
        self.bindings = dict(zip(self.parameters, bindings))
        self.zero_event = zero_event
        return tuple(bindings)

    def arm(self, lease):
        lease.validate()
        parameter = lease.binding.parameter
        if (
            self.window != lease.binding.window
            or self.bindings.get(parameter) is not lease.binding
            or parameter in self.pending
            or lease.armed
            or lease.consumed
        ):
            raise Mxfp4StateError("DUPLICATE_WGRAD_OR_GRADIENT_EDGE")
        stream = self._current_stream(parameter.device)
        stream.wait_event(lease.w_event)
        stream.wait_event(lease.zero_event)
        parameter.grad_added_to_main_grad = True
        lease.armed = True
        self.pending[parameter] = lease
        self.leases.append(lease)
        return lease.emission

    def _prehook(self, parameter):
        def hook(gradient):
            lease = self.pending.get(parameter)
            if lease is None:
                raise Mxfp4StateError("UNREGISTERED_GRADIENT_WRITER")
            lease.validate()
            if (
                tensor_range(gradient) != tensor_range(lease.emission)
                or gradient.dtype != lease.emission.dtype
                or gradient.stride() != lease.emission.stride()
            ):
                raise Mxfp4StateError(
                    "ZERO_LEASE_MISMATCH", "materialized or merged gradient"
                )
            if parameter.grad is not None or not parameter.grad_added_to_main_grad:
                raise Mxfp4StateError("GRADIENT_BOOKKEEPING_MISMATCH")
            stream = self._current_stream(parameter.device)
            stream.wait_event(lease.w_event)
            stream.wait_event(lease.zero_event)
            stream.wait_event(self.zero_event)
            parameter.grad = lease.sink
            return gradient

        return hook

    def _observer(self, parameter):
        def hook(*unused):
            lease = self.pending.get(parameter)
            if lease is None or parameter.grad is not None:
                raise Mxfp4StateError("DDP_HOOK_NOT_CONSUMED")
            lease.consumed_event.record(self._current_stream(parameter.device))
            lease.consumed = True
            del self.pending[parameter]

        return hook

    def before_grad_sync(self):
        if self.pending or any(not lease.consumed for lease in self.leases):
            raise Mxfp4StateError("DDP_HOOK_NOT_CONSUMED")
        for lease in self.leases:
            stream = self._current_stream(lease.binding.parameter.device)
            stream.wait_event(lease.w_event)
            stream.wait_event(lease.consumed_event)

    def close_after_grad_sync(self, reducer_event):
        self.before_grad_sync()
        reducer_event.synchronize()
        for lease in self.leases:
            lease.consumed_event.synchronize()
        self.leases.clear()
        self.window = None

    def remove(self):
        if self.window is not None or self.pending:
            raise Mxfp4StateError("UNSAFE_HOOK_REMOVAL")
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
