# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0
"""Translate actual routed public operations into the existing obligation ledger."""

from lumen.utils.mxfp4_moe_diagnostics import FailureSnapshot
from lumen.utils.mxfp4_moe_ledger import Mxfp4StateError, ObligationKey


class FCObligations:
    def __init__(self, ledger, scope, fc, route, epoch, tokens, prepared):
        self.ledger, self.route, self.prepared = ledger, route, prepared
        self.keys = tuple(
            ObligationKey(
                scope.rng_epoch,
                scope.attempted_step,
                scope.microbatch_ordinal,
                scope.global_layer_ordinal,
                scope.logical_invocation_id,
                fc,
                pass_id,
            )
            for pass_id in range(3)
        )
        ledger.declare(self.keys)
        for key in self.keys:
            config = prepared.config_view.gemm(key.pass_id).config
            ledger.bind(
                key,
                logical_binding_key=key.record(),
                config_ids=[list(config)],
                routing_epoch=route.routing_epoch,
                weight_epoch=epoch,
                rng_receipts=[token.record() for token in tokens],
                shape=[route.sorted_rows, 4096, prepared.static_spec[6]],
                dtype="torch.bfloat16" if key.pass_id != 2 else "torch.float32",
                layout_ids=[
                    "mxfp4_weight_tile32x32_v1",
                    "mxfp4_segmented_row_and_t_v1",
                    "plain_dense_v1",
                ],
                requested_mode="sequential",
                selected_mode="sequential",
                executed_mode=None,
                kernel_family=None,
                precision=None,
                selection_events=[],
                status=0,
                failure_phase=None,
                reason=None,
            )
        self.attempts = {}
        self.first_failure = None
        self.failure_context = None
        self.failure_output_attempted = False
        self.failure_output_written = False

    def begin(self, pass_id, replay=False):
        key = self.keys[pass_id]
        attempt = self.ledger.begin_attempt(key, replay=replay)
        self.attempts[pass_id] = attempt

    def enqueued(self, pass_id, event):
        from lumen.utils.mxfp4_moe_children import dense_children

        descriptors = dense_children(
            self.prepared.config_view,
            self.route.counts,
            pass_id,
            replay=self.attempts[pass_id] > 0,
        )
        self.ledger.enqueue(
            self.keys[pass_id],
            self.attempts[pass_id],
            sublaunches=descriptors,
            events=(event,),
            execution=(
                ("sequential", "TRITON", "MXFP4")
                if any(item["op"] == "gemm" for item in descriptors)
                else None
            ),
        )

    def completed(self, pass_id):
        if not self.ledger.finish(
            self.keys[pass_id], self.attempts[pass_id], empty=not self.route.sorted_rows
        ):
            raise Mxfp4StateError(
                "EVENT_CONTRACT_VIOLATION", "healthy boundary incomplete"
            )

    def attach_journal(self, pass_id, journal, *, materialization=False, replay=False):
        from lumen.utils.mxfp4_moe_children import dense_children

        if materialization:
            known = journal.known_prefix()
        else:
            # Use exactly the success mapper, restricted to proven public returns.
            # In particular, an unentered empty W must not acquire a zero child.
            returned_experts = {item["expert"] for item in journal.known_prefix()}
            known = dense_children(
                self.prepared.config_view,
                self.route.counts,
                pass_id,
                replay=replay,
                experts=returned_experts,
            )
        facts = tuple(
            dict(descriptor, submission=state)
            for descriptor, state in zip(journal.descriptors, journal.states)
        )
        self.ledger.attach_submission(
            self.keys[pass_id],
            facts=facts,
            known_sublaunches=known,
            unknown_operation=journal.pending_device_operation,
        )

    def emit_failure(self, control, error, phase, *, asynchronous):
        """State is already saved; unexpected diagnostics retain primary and cause."""
        if self.failure_output_attempted:
            return self.failure_output_written
        self.first_failure = error
        self.failure_output_attempted = True
        control.failed = True
        try:
            snapshot = FailureSnapshot(
                phase,
                asynchronous,
                tuple((key.record(), self.ledger._entry(key)) for key in self.keys),
            )
            if asynchronous:
                written = control.async_abort(error, snapshot=snapshot)
            else:
                written = control.failure_record(
                    "submission_failure", primary=error, snapshot=snapshot
                )
            self.failure_context = control.failure_context
            self.failure_output_written = written is True
            self.failure_output_error = control.failure_output_error
        except Exception as secondary:
            if secondary is error:
                self.failure_diagnostic_secondary = secondary.__cause__
                self.failure_output_error = type(secondary.__cause__).__name__
                raise
            self.failure_diagnostic_secondary = secondary
            self.failure_output_error = type(secondary).__name__
            raise error from secondary
        return self.failure_output_written

    def fail(self, pass_id, reason, phase, asynchronous=False, numeric_status=None):
        self.ledger.fail(
            self.keys[pass_id],
            reason=(
                reason
                if isinstance(reason, str)
                else (
                    reason.reason
                    if isinstance(reason, Mxfp4StateError)
                    else type(reason).__name__
                )
            ),
            phase=phase,
            asynchronous=asynchronous,
            numeric_status=numeric_status,
        )
        if pass_id == 0:
            for key in self.keys[1:]:
                if (
                    self.ledger._entry(key).state in ("DECLARED", "BOUND")
                    and not self.ledger._entry(key).attempts
                ):
                    self.ledger.abort_unlaunched(key, "CAUSAL_FORWARD_FAILURE")
