"""Real CPU driver state and full-microbatch boundaries; no GPU imports."""

from contextlib import contextmanager, nullcontext

from lumen.quantize.mxfp4_moe_rng import ScopeKey
from lumen.utils.mxfp4_moe_ledger import Mxfp4StateError


class RoutedDriver:
    def zero_boundary(self, microbatches):
        if (
            type(microbatches) is not int
            or not 0 < microbatches <= self.max_live_invocations_per_layer
        ):
            raise Mxfp4StateError("LIVE_LEASE_CAPACITY_VIOLATION")
        self._driver_call(self._check_step_microbatches)
        self._driver_call(self.lifecycle.zero_boundary)
        step = self.weights._window
        self.scopes = {
            (microbatch, index): ScopeKey(
                self.rng.rng_epoch,
                step,
                microbatch,
                index,
                self.rng.frontier + microbatch * 2 + index,
            )
            for microbatch in range(microbatches)
            for index in range(2)
        }
        self.rng.declare(self.scopes.values())
        self._microbatch_count = microbatches
        self._next_microbatch = 0

    @contextmanager
    def microbatch_scope(self, microbatch):
        """Enclose one complete forward AND backward, including checkpoint replay."""
        self._driver_call(self._check_microbatch_entry, microbatch)
        previous = self.microbatch
        self.microbatch = microbatch
        self._microbatch_scope_active = microbatch
        try:
            # Only Core owns its readiness flags, reducer and no_sync restoration.
            context = (
                self.ddp.no_sync()
                if microbatch + 1 < self._microbatch_count
                else nullcontext()
            )
            with context:
                yield
                self._driver_call(self._check_microbatch_consumed, microbatch)
        except Exception as exc:
            self._driver_failure(exc)
            raise
        finally:
            self.microbatch = previous
            self._microbatch_scope_active = None
        self._next_microbatch += 1

    def _check_microbatch_entry(self, microbatch):
        if (
            type(microbatch) is not int
            or self._microbatch_scope_active is not None
            or microbatch != self._next_microbatch
            or not 0 <= microbatch < self._microbatch_count
        ):
            raise Mxfp4StateError("MICROBATCH_SCOPE_MISMATCH")

    def _check_microbatch_consumed(self, microbatch):
        # Every FC's original Core hook must run its final observer before exit.
        for layer in range(2):
            invocation = self.lifecycle.active.get(self.scopes[(microbatch, layer)])
            if invocation is None or set(invocation.executions) != {1, 2}:
                raise Mxfp4StateError("MICROBATCH_BACKWARD_INCOMPLETE")
            for execution in invocation.executions.values():
                if (
                    not execution.backward_started
                    or not execution.zero_leases
                    or any(not lease.consumed for lease in execution.zero_leases)
                ):
                    raise Mxfp4StateError("MICROBATCH_BACKWARD_INCOMPLETE")

    def _check_step_microbatches(self):
        if self._microbatch_scope_active is not None or (
            self._microbatch_count > 1
            and self._next_microbatch != self._microbatch_count
        ):
            raise Mxfp4StateError("MICROBATCH_BACKWARD_INCOMPLETE")

    def _check_forward_microbatch(self, microbatch):
        if (
            type(microbatch) is not int
            or not 0 <= microbatch < self._microbatch_count
            or (
                self._microbatch_scope_active is not None
                and self._microbatch_scope_active != microbatch
            )
        ):
            raise Mxfp4StateError("MICROBATCH_SCOPE_MISMATCH")
        if self._microbatch_count > 1 and self._microbatch_scope_active is None:
            raise Mxfp4StateError("MICROBATCH_SCOPE_REQUIRED")

    def forward(self, hidden, microbatch, *, checkpoint=False):
        self._driver_call(self._check_forward_microbatch, microbatch)
        previous = self.microbatch
        self.microbatch = microbatch

        def run(x):
            for layer in self.layers:
                x, _ = layer(x)
            return x

        def execute():
            if checkpoint:
                from megatron.core.tensor_parallel.random import (
                    checkpoint as core_checkpoint,
                )

                from lumen.models.mxfp4_moe import CheckpointReceiptTape

                tape = CheckpointReceiptTape()
                return core_checkpoint(tape.wrap(run), False, hidden)
            return run(hidden)

        try:
            return self._driver_call(execute)
        finally:
            self.microbatch = previous

    def _driver_call(self, operation, *args):
        # A caught error must not allow a later reducer/optimizer/zero submission.
        if self._driver_failed or self.control.failed:
            raise Mxfp4StateError("ABORTED_DRIVER_SCOPE")
        try:
            return operation(*args)
        except Exception as exc:
            self._driver_failure(exc)
            raise

    def _driver_failure(self, exc):
        if self._driver_failed:
            return
        self._driver_failed = True
        if not isinstance(exc, Mxfp4StateError):
            try:
                self.control.async_abort(exc)
            except Exception as secondary:
                if secondary is exc:
                    self.failure_diagnostic_secondary = secondary.__cause__
                    raise
                self.failure_diagnostic_secondary = secondary
                raise exc from secondary

    def finish_grad_sync(self):
        self._driver_call(self._check_step_microbatches)
        return self._driver_call(self.lifecycle.finish_grad_sync)

    def optimizer_result(self, success):
        self._driver_call(self._check_step_microbatches)
        return self._driver_call(self.lifecycle.optimizer_result, success)
