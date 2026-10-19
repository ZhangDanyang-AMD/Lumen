"""Actual routed FC host orchestration with explicit runtime operations."""

from dataclasses import dataclass

from lumen.utils.mxfp4_moe_control import ControlSequence
from lumen.utils.mxfp4_moe_ledger import Mxfp4StateError


@dataclass
class RoutedFCExecution:
    route: object
    bounds: tuple
    inputs: tuple
    zero_leases: tuple
    handoff: object
    control: object
    identity: tuple
    y: object
    dx: object
    status: object
    mirror: object
    forward_event: object
    backward_event: object
    status_event: object
    obligations: object
    operations: object
    producer_frontier: int = 0
    post_forward_frontier: int = -1
    backward_started: bool = False
    replayed: bool = False

    def __post_init__(self):
        from lumen.utils.mxfp4_moe_submission import SubmissionJournal

        descriptors = tuple(
            {
                "expert": index,
                "logical_M": count,
                "op": "forward_composition" if count else "empty_no_launch",
            }
            for index, count in enumerate(self.route.counts)
        )
        self.forward_submission = SubmissionJournal(descriptors)
        self.replay_submission = SubmissionJournal(descriptors)
        self.backward_submission = SubmissionJournal(
            tuple(
                dict(
                    item,
                    op=(
                        "backward_composition"
                        if item["logical_M"]
                        else "empty_no_launch"
                    ),
                )
                for item in descriptors
            )
        )

    def sequence(self, phase, physical=0):
        return ControlSequence(*self.identity, phase, physical)

    def validate(self):
        if len(self.bounds) != len(self.route.counts) or len(self.zero_leases) != len(
            self.bounds
        ):
            raise Mxfp4StateError("ROUTING_SCHEMA_MISMATCH")
        if (
            self.y.shape != (sum(self.route.counts), 4096)
            or self.dx.shape[0] != self.y.shape[0]
        ):
            raise Mxfp4StateError("RUNTIME_OPERAND_CONTRACT_VIOLATION")
        for count, bound, lease in zip(
            self.route.counts, self.bounds, self.zero_leases
        ):
            lease.validate()
            if (
                bound.logical_M != count
                or bound.buffers["w_out"] is not lease.binding.main_grad
                or bound.buffers["status"] is not self.status
            ):
                raise Mxfp4StateError("MAIN_GRAD_MISMATCH")
            if bound.prepared.static_spec[9:11] != (1, 1):
                raise Mxfp4StateError("MAIN_GRAD_MISMATCH", "ACCUMULATE_FP32 required")

    def _submission_failed(
        self,
        journal,
        passes,
        exc,
        phase,
        *,
        replay=False,
        unknown=None,
        numeric_status=None,
    ):
        self.submission_failure = journal.failure()
        for pass_id in passes:
            self.obligations.attach_journal(pass_id, journal, replay=replay)
        if unknown is None:
            unknown = journal.unknown()
        for pass_id in passes:
            self.obligations.fail(
                pass_id, exc, phase, asynchronous=unknown, numeric_status=numeric_status
            )
        self.obligations.emit_failure(self.control, exc, phase, asynchronous=unknown)

    def status_boundary(self, phase, event, physical=0):
        passes = (0,) if phase == "FC_POSTF_STATUS" else (1, 2)
        journal = (
            (self.replay_submission if physical else self.forward_submission)
            if passes == (0,)
            else self.backward_submission
        )
        try:
            stream = journal.device_call(
                "status.current_stream",
                self.operations.current_stream,
                self.status.device,
            )
            journal.device_call("status.wait_event", stream.wait_event, event)
            journal.device_call(
                "status.copy", self.mirror.copy_, self.status, non_blocking=True
            )
            journal.device_call("status.event_record", self.status_event.record, stream)
            journal.device_call("status.synchronize", self.status_event.synchronize)
        except Exception as exc:
            self._submission_failed(
                journal, passes, exc, phase, replay=bool(physical), unknown=True
            )
            raise
        code, voting = None, False
        try:
            code = int(self.mirror[0])
            voting = True
            self.control.vote(
                self.sequence(phase, physical),
                code=code,
                reason="OK" if not code else "NUMERIC_STATUS_POSTLAUNCH",
            )
        except Exception as exc:
            # This reason is inspected only at the CPU vote site, never inside a GPU call.
            health_failure = (
                isinstance(exc, Mxfp4StateError)
                and exc.reason == "COMPONENT_HEALTH_FAILURE"
            )
            self._submission_failed(
                journal,
                passes,
                exc,
                phase,
                replay=bool(physical),
                unknown=voting and not health_failure,
                numeric_status=code,
            )
            raise

    def forward(self, replay=False):
        api = self.operations.probe_api()
        if api is None:
            raise Mxfp4StateError("STATIC_CAPABILITY_MISS")
        physical = int(replay)
        self.validate()
        if replay:
            if (
                self.backward_started
                or self.replayed
                or self.post_forward_frontier != self.producer_frontier
            ):
                raise Mxfp4StateError("REPLAY_PRODUCER_FRONTIER_MISMATCH")
            self.control.vote(self.sequence("MATERIALIZATION_HEALTH", physical))
            self.replayed = True
        self.control.vote(self.sequence("FC_PRELAUNCH", physical))
        self.obligations.begin(0, replay=replay)
        journal = self.replay_submission if replay else self.forward_submission
        try:
            for index, bound in enumerate(self.bounds):
                if replay:
                    journal.call(index, api.mxfp4_dense_launch_packed_forward_v1, bound)
                else:
                    journal.call(
                        index,
                        api.mxfp4_dense_launch_forward_v1,
                        bound,
                        self.inputs[index],
                    )
            self.producer_frontier += 1
            stream = journal.device_call(
                "forward.current_stream",
                self.operations.current_stream,
                self.status.device,
            )
            journal.device_call(
                "forward.event_record", self.forward_event.record, stream
            )
            self.obligations.enqueued(0, self.forward_event)
        except Exception as exc:
            self._submission_failed(journal, (0,), exc, "FORWARD_LAUNCH", replay=replay)
            raise
        self.status_boundary("FC_POSTF_STATUS", self.forward_event, physical)
        self.obligations.completed(0)
        self.post_forward_frontier = self.producer_frontier
        if not replay:
            # AITER keeps only source identity/metadata, not BF16 activation storage.
            self.inputs = ()
        return self.y

    def backward(self, dy):
        if self.backward_started or self.operations.is_grad_enabled():
            raise Mxfp4StateError("DUPLICATE_WGRAD_OR_HIGHER_ORDER")
        api = self.operations.probe_api()

        def attach():
            self.validate()
            if (
                dy.dtype != self.operations.activation_dtype
                or dy.shape != self.y.shape
                or not dy.is_contiguous()
            ):
                raise Mxfp4StateError("RUNTIME_OPERAND_CONTRACT_VIOLATION")
            starts = [0]
            for count in self.route.counts:
                starts.append(starts[-1] + count)
            return tuple(
                api.mxfp4_dense_attach_backward_v1(bound, dy[start : start + count])
                for bound, start, count in zip(self.bounds, starts, self.route.counts)
            )

        try:
            runtime = self.control.metadata_check(
                self.sequence("BACKWARD_PREATTACH"), attach
            )
        except Exception as exc:
            for pass_id in (1, 2):
                self.obligations.fail(
                    pass_id,
                    exc,
                    "BACKWARD_PREATTACH",
                    asynchronous=not isinstance(exc, Mxfp4StateError),
                )
            raise
        self.backward_started = True
        for pass_id in (1, 2):
            self.obligations.begin(pass_id)
        journal = self.backward_submission
        try:
            for index, (bound, operands) in enumerate(zip(self.bounds, runtime)):
                journal.call(index, api.mxfp4_dense_launch_backward_v1, bound, operands)
            self.producer_frontier += 1
            stream = journal.device_call(
                "backward.current_stream",
                self.operations.current_stream,
                self.status.device,
            )
            journal.device_call(
                "backward.event_record", self.backward_event.record, stream
            )
            for pass_id in (1, 2):
                self.obligations.enqueued(pass_id, self.backward_event)
        except Exception as exc:
            self._submission_failed(journal, (1, 2), exc, "BACKWARD_LAUNCH")
            raise
        self.status_boundary("BACKWARD_POST_STATUS", self.backward_event)
        for pass_id in (1, 2):
            self.obligations.completed(pass_id)
        for lease in self.zero_leases:
            lease.w_event.record(stream)
        return self.dx, tuple(self.handoff.arm(lease) for lease in self.zero_leases)
