# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0
"""Feature-scoped original/recompute contexts around existing Core experts."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from lumen.ops.mxfp4_moe import routed_linear
from lumen.quantize.mxfp4_moe_rng import RoutingMetadataV1
from lumen.utils.mxfp4_moe_ledger import Mxfp4StateError

_ACTIVE = ContextVar("lumen_mxfp4_routed_scope", default=None)
_TAPE = ContextVar("lumen_mxfp4_checkpoint_tape", default=None)


@dataclass
class RoutedInvocation:
    scope: object
    route: RoutingMetadataV1
    executions: dict = field(default_factory=dict)
    replay: bool = False
    cursor: int = 1


@contextmanager
def routed_invocation(invocation):
    if _ACTIVE.get() is not None:
        raise Mxfp4StateError("NESTED_ROUTED_INVOCATION")
    invocation.cursor = 1
    token = _ACTIVE.set(invocation)
    try:
        yield invocation
        if invocation.cursor != 3:
            raise Mxfp4StateError("INCOMPLETE_TWO_FC_INVOCATION")
    finally:
        _ACTIVE.reset(token)


def module_forward(module, x, m_splits):
    invocation = _ACTIVE.get()
    if invocation is None:
        raise Mxfp4StateError("MISSING_ROUTE_READY_SCOPE")
    fc = module._mxfp4_fc
    if (
        invocation.cursor != fc
        or tuple(m_splits) != invocation.route.counts
        or x.shape[0] != invocation.route.sorted_rows
    ):
        raise Mxfp4StateError("ROUTING_SCHEMA_MISMATCH")
    if invocation.replay:
        execution = invocation.executions[fc]
    else:
        execution = module._mxfp4_materializer(x, invocation, fc)
        if execution.route is not invocation.route:
            raise Mxfp4StateError("ROUTING_SCHEMA_MISMATCH", "metadata identity")
        invocation.executions[fc] = execution
    invocation.cursor += 1
    return routed_linear(x, execution, replay=invocation.replay), None


class CoreExpertsScope:
    """Wrap only a real experts instance; its activation/probability body is unchanged."""

    def __init__(
        self,
        experts,
        *,
        scope_supplier,
        sender_rows_supplier,
        rng,
        preflight,
        fc1_materializer,
        fc2_materializer
    ):
        from megatron.core.transformer.moe.experts import TEGroupedMLP

        from lumen.modules.grouped_linear import LumenGroupedLinear

        if not isinstance(experts, TEGroupedMLP) or not all(
            isinstance(fc, LumenGroupedLinear)
            for fc in (experts.linear_fc1, experts.linear_fc2)
        ):
            raise Mxfp4StateError("UNSUPPORTED_EXPERTS_SHELL")
        self.experts, self.rng = experts, rng
        self.scope_supplier, self.sender_rows_supplier = (
            scope_supplier,
            sender_rows_supplier,
        )
        self.preflight = preflight
        self.original_forward = experts.forward
        for index, (fc, materializer) in enumerate(
            zip(
                (experts.linear_fc1, experts.linear_fc2),
                (fc1_materializer, fc2_materializer),
            ),
            1,
        ):
            if (
                fc.use_bias
                or not fc.is_expert
                or fc.num_gemms not in (32, 64)
                or (fc.out_features, fc.in_features)
                != (4096, 4096 if index == 1 else 2048)
            ):
                raise Mxfp4StateError("UNSUPPORTED_ROUTED_FC")
            fc._mxfp4_fc, fc._mxfp4_materializer = index, materializer
        experts.forward = self.forward

    def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        tape = _TAPE.get()
        if tape is not None and tape.calls == 2 and _ACTIVE.get() is None:
            invocation = tape.next_replay()
            with routed_invocation(invocation):
                return self.forward(
                    permuted_local_hidden_states, tokens_per_expert, permuted_probs
                )
        active = _ACTIVE.get()
        if active is not None:
            if not active.replay:
                raise Mxfp4StateError("DUPLICATE_ROUTED_SCOPE")
            counts = tuple(tokens_per_expert.tolist())
            if counts != active.route.counts:
                raise Mxfp4StateError("REPLAY_ROUTING_MISMATCH")
            return self.original_forward(
                permuted_local_hidden_states, tokens_per_expert, permuted_probs
            )
        if tokens_per_expert.device.type != "cpu":
            raise Mxfp4StateError("READY_CPU_COUNTS_REQUIRED")
        counts = tuple(tokens_per_expert.tolist())
        scope, epoch = self.scope_supplier()
        route = RoutingMetadataV1(
            counts,
            permuted_local_hidden_states.shape[0],
            tuple(self.sender_rows_supplier()),
            routing_epoch=scope.logical_invocation_id,
        )
        self.rng.close_scope(scope, route, epoch, preflight=self.preflight)
        receipts = self.rng.reserve_next()
        if receipts is None or receipts[0].scope != scope:
            raise Mxfp4StateError("DECLARATION_FRONTIER_NOT_READY")
        invocation = RoutedInvocation(scope, route)
        if tape is not None:
            tape.retain(invocation)
        with routed_invocation(invocation):
            result = self.original_forward(
                permuted_local_hidden_states, tokens_per_expert, permuted_probs
            )
        return result


class CheckpointReceiptTape:
    """Closure-owned tape around original Core checkpoint; never rewinds SR counters."""

    def __init__(self):
        self.invocations = []
        self.calls = 0
        self.cursor = 0

    def retain(self, invocation):
        if self.calls != 1:
            raise Mxfp4StateError("CHECKPOINT_TAPE_PHASE_MISMATCH")
        self.invocations.append(invocation)

    def next_replay(self):
        if self.calls != 2 or self.cursor >= len(self.invocations):
            raise Mxfp4StateError("CHECKPOINT_TAPE_CURSOR_MISMATCH")
        invocation = self.invocations[self.cursor]
        self.cursor += 1
        invocation.replay = True
        return invocation

    def wrap(self, run_function):
        def closure(*args):
            self.calls += 1
            if self.calls > 2:
                raise Mxfp4StateError("DUPLICATE_CHECKPOINT_REPLAY")
            self.cursor = 0
            token = _TAPE.set(self)
            try:
                result = run_function(*args)
            finally:
                _TAPE.reset(token)
            if self.calls == 2 and self.cursor != len(self.invocations):
                raise Mxfp4StateError("CHECKPOINT_TAPE_CURSOR_MISMATCH")
            return result

        return closure
