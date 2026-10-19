# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0
"""Explicit fixture/driver boundaries around actual DDP zero/reduction/update."""

from lumen.utils.mxfp4_moe_ledger import Mxfp4StateError


class RoutedLifecycle:
    def __init__(
        self,
        ddp,
        handoff,
        weights,
        rng,
        ledger,
        *,
        event_factory=None,
        current_stream=None
    ):
        if event_factory is None or current_stream is None:
            import torch

            event_factory = event_factory or torch.cuda.Event
            current_stream = current_stream or torch.cuda.current_stream
        self._current_stream = current_stream
        self.ddp, self.handoff, self.weights = ddp, handoff, weights
        self.rng, self.ledger = rng, ledger
        self.active = {}
        self.layer_bindings = []
        self.zero_event = event_factory()
        self.reducer_event = event_factory()
        self.sync_event = event_factory()
        for event in (self.zero_event, self.reducer_event, self.sync_event):
            event.record(self._current_stream())

    def zero_boundary(self):
        if self.active:
            raise Mxfp4StateError("ACCUMULATION_WINDOW_MISMATCH")
        self.ddp.zero_grad_buffer()
        self.zero_event.record(self._current_stream())
        self.zero_event.synchronize()
        window = self.weights.open_window(zero_events=(self.zero_event,))
        return self.handoff.open_after_zero(
            window, self.weights.committed_weight_epoch, self.zero_event
        )

    def retain(self, invocation):
        key = invocation.scope
        if key in self.active:
            raise Mxfp4StateError("DUPLICATE_ROUTED_SCOPE")
        self.weights.acquire(key, (invocation,))
        self.active[key] = invocation

    def finish_grad_sync(self):
        self.handoff.before_grad_sync()
        self.ddp.finish_grad_sync()
        self.reducer_event.record(self._current_stream())
        self.reducer_event.synchronize()
        self.handoff.close_after_grad_sync(self.reducer_event)
        for key, invocation in self.active.items():
            events = tuple(
                lease.consumed_event
                for execution in invocation.executions.values()
                for lease in execution.zero_leases
            ) + (self.reducer_event,)
            if not self.weights.release(key, events=events):
                raise Mxfp4StateError("CACHE_LIFETIME_MISMATCH")
            self.rng.retire_scope(key, events=events)
            for execution in invocation.executions.values():
                execution.retire_storage(events)
        self.active.clear()
        if self.ledger.counts()["open"] or self.ledger.counts()["pending_physical"]:
            raise Mxfp4StateError("OPEN_LEDGER_AT_OPTIMIZER_BOUNDARY")
        self.weights.close_window(reducer_events=(self.reducer_event,))

    def optimizer_result(self, success):
        if type(success) is not bool:
            raise Mxfp4StateError("WEIGHT_EPOCH_MISMATCH")
        if success:
            # Explicit full parameter synchronization, not local shard-copy readiness.
            self.ddp.start_param_sync(force_sync=True)
            for group in (
                self.ddp.bucket_groups + self.ddp.expert_parallel_bucket_groups
            ):
                group.finish_param_sync()
            self.sync_event.record(self._current_stream())
            self.sync_event.synchronize()
        self.weights.optimizer_result(
            success=success, full_bf16_sync_events=(self.sync_event,) if success else ()
        )
        if success and not self.weights.commit_ready_update():
            raise Mxfp4StateError("WEIGHT_EPOCH_MISMATCH")
        if success:
            for binding in self.layer_bindings:
                binding.invalidate_cache_after_consumers((self.sync_event,))
