"""Direct host failure behavior; fake device boundaries are not GPU-fault evidence."""

import importlib.util
import io
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[2]


def load_host_modules():
    spec = importlib.util.spec_from_file_location(
        "mxfp4_host_imports", ROOT / "tests/mxfp4_host_imports.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_host_modules()


HOST = load_host_modules()
Mxfp4StateError = HOST.ledger.Mxfp4StateError
ObligationLedger = HOST.ledger.ObligationLedger
FCObligations = HOST.execution_ledger.FCObligations
SubmissionJournal = HOST.submission.SubmissionJournal


class View:
    w_output_mode = 1
    w_output_dtype = 1

    def gemm(self, pass_id):
        return SimpleNamespace(config=(100 + pass_id, 1))

    def primitive(self, op_id, *, variant, output_dtype=0):
        return SimpleNamespace(config=(200 + op_id * 10 + variant, output_dtype))


class Control(HOST.control.Mxfp4Control):
    def __init__(self):
        self.failed = False
        self.votes = []
        self.async_errors = []
        self.rank = 0
        self.physical_run_id = "host-control-boundary"
        self._journal = io.StringIO()
        self._failure_reporter = HOST.diagnostics.FailureReporter(
            self._journal, self.rank, self.physical_run_id
        )

    def vote(self, sequence, *, code=0, reason="OK"):
        self.votes.append((sequence.phase, code))
        if code:
            self.failed = True
            raise Mxfp4StateError("COMPONENT_HEALTH_FAILURE")
        return {0: 0}

    def metadata_check(self, sequence, function):
        return function()

    def async_abort(self, error, *, snapshot=None):
        self.async_errors.append(error)
        return super().async_abort(error, snapshot=snapshot)


class Tensor:
    dtype = "bf16"
    device = "device0"

    def __init__(self, shape, log, value=0):
        self.shape, self.log, self.value = shape, log, value
        self.copy_error = None

    def is_contiguous(self):
        return True

    def __getitem__(self, key):
        return self.value if key == 0 else self

    def copy_(self, source, **kwargs):
        self.log.append("copy")
        if self.copy_error:
            raise self.copy_error
        self.value = source.value
        return self


class Event:
    def __init__(self, log):
        self.log = log
        self.record_error = None

    def record(self, stream):
        self.log.append("event.record")
        if self.record_error:
            raise self.record_error

    def synchronize(self):
        self.log.append("event.synchronize")

    def query(self):
        self.log.append("event.query")
        return True


def obligations(counts=(1, 1), mode=1):
    view = View()
    view.w_output_mode = mode
    prepared = SimpleNamespace(
        config_view=view, static_spec=(0, 0, 0, 0, 0, 4096, 2048, 0, 0, mode, 1)
    )
    route = SimpleNamespace(counts=counts, sorted_rows=sum(counts), routing_epoch=0)
    scope = SimpleNamespace(
        rng_epoch=0,
        attempted_step=1,
        microbatch_ordinal=0,
        global_layer_ordinal=0,
        logical_invocation_id=0,
    )
    ledger = ObligationLedger("a" * 64)
    fc = FCObligations(ledger, scope, 1, route, 0, (), prepared)
    return fc


def materializer(function):
    def call(x, **kwargs):
        return HOST.materialization.materialize_with_journal(function, x, **kwargs)

    return call


def execution(counts=(1, 1)):
    fc, log, control = obligations(counts), [], Control()
    api = SimpleNamespace()
    calls = []
    api.mxfp4_dense_launch_forward_v1 = lambda bound, source: calls.append(("F", bound))
    api.mxfp4_dense_launch_packed_forward_v1 = lambda bound: calls.append(("R", bound))
    api.mxfp4_dense_attach_backward_v1 = lambda bound, dy: dy
    api.mxfp4_dense_launch_backward_v1 = lambda bound, operands: calls.append(
        ("B", bound)
    )
    stream = SimpleNamespace(wait_event=lambda event: log.append("wait_event"))

    def current_stream(device):
        log.append("current_stream")
        return stream

    operations = SimpleNamespace(
        activation_dtype="bf16",
        is_grad_enabled=lambda: False,
        current_stream=current_stream,
        probe_api=lambda: api,
    )
    rows = sum(counts)
    obj = HOST.execution.RoutedFCExecution(
        fc.route,
        tuple(range(len(counts))),
        tuple(object() for _ in counts),
        (),
        None,
        control,
        (1, 0, 0, 1),
        Tensor((rows, 4096), log),
        Tensor((rows, 2048), log),
        Tensor((1,), log),
        Tensor((1,), log),
        Event(log),
        Event(log),
        Event(log),
        fc,
        operations=operations,
    )
    # The fixture starts from already validated tensors; this test targets failure boundaries.
    obj.validate = lambda: None
    return obj, api, calls, log


def fail_after_return(journal, error, index=0):
    returned = journal.returned

    def injected(current):
        returned(current)
        if current == index:
            raise error

    journal.returned = injected


class PartialFailureTests(unittest.TestCase):
    def test_materialize_before_first_is_prelaunch_and_causal(self):
        fc, control, error = obligations(), Control(), RuntimeError("host metadata")

        def work(x, progress, **kwargs):
            raise error

        with self.assertRaises(RuntimeError) as caught:
            materializer(work)(
                None,
                prepared=fc.prepared,
                bindings=(1, 2),
                obligations=fc,
                control=control,
            )
        self.assertIs(caught.exception, error)
        self.assertEqual(
            [fc.ledger._entry(key).state for key in fc.keys],
            ["FAILED_PRELAUNCH", "ABORTED", "ABORTED"],
        )
        self.assertEqual(fc.ledger._entry(fc.keys[0]).known_sublaunches, ())
        self.assertEqual(control.async_errors, [])

    def test_materialize_known_prefix_keeps_expert_config_and_first_error(self):
        fc, control, error = obligations(), Control(), RuntimeError("between calls")

        def work(x, progress, **kwargs):
            for index in (0, 1, 2):
                progress["journal"].call(index, lambda: None)
            raise error

        with self.assertRaises(RuntimeError) as caught:
            materializer(work)(
                None,
                prepared=fc.prepared,
                bindings=(1, 2),
                obligations=fc,
                control=control,
            )
        self.assertIs(caught.exception, error)
        entry = fc.ledger._entry(fc.keys[0])
        self.assertEqual(entry.state, "ABORTED")
        self.assertEqual(
            [(item["expert"], item["op"]) for item in entry.known_sublaunches],
            [(None, "status_init"), (0, "cold_status_init"), (0, "weight_quant")],
        )
        self.assertEqual(
            entry.known_sublaunches[-1]["config_id"],
            tuple(fc.prepared.config_view.primitive(1, variant=0).config),
        )
        self.assertEqual(
            [fc.ledger._entry(key).state for key in fc.keys[1:]], ["ABORTED", "ABORTED"]
        )
        self.assertEqual(control.async_errors, [])

    def test_materialize_opaque_exception_never_double_fails_causal_terminals(self):
        fc, control = obligations(), Control()
        error = Mxfp4StateError(
            "COMPONENT_HEALTH_FAILURE", "from inside opaque primitive"
        )

        def work(x, progress, **kwargs):
            progress["journal"].call(0, lambda: None)

            def unknown():
                raise error

            progress["journal"].call(2, unknown)

        with self.assertRaises(Mxfp4StateError) as caught:
            materializer(work)(
                None,
                prepared=fc.prepared,
                bindings=(1, 2),
                obligations=fc,
                control=control,
            )
        self.assertIs(caught.exception, error)
        self.assertEqual(
            [fc.ledger._entry(key).state for key in fc.keys],
            ["INCOMPLETE_ASYNC", "ABORTED", "ABORTED"],
        )
        self.assertEqual(fc.ledger.counts()["open"], 1)
        self.assertEqual(len(control.async_errors), 1)

    def test_materialize_status_and_empty_zero_children(self):
        fc, control = obligations((0,)), Control()

        def work(x, progress, **kwargs):
            for index in range(len(progress["journal"].states)):
                progress["journal"].call(index, lambda: None)
            progress["numeric_status"] = 4
            progress["control_pending"] = True
            raise Mxfp4StateError("COMPONENT_HEALTH_FAILURE")

        with self.assertRaises(Mxfp4StateError):
            materializer(work)(
                None,
                prepared=fc.prepared,
                bindings=(1,),
                obligations=fc,
                control=control,
            )
        entry = fc.ledger._entry(fc.keys[0])
        self.assertEqual((entry.state, entry.semantic["status"]), ("ABORTED", 4))
        self.assertEqual(
            [item["op"] for item in entry.known_sublaunches][-2:], ["Z_zero", "G_zero"]
        )
        self.assertNotIn("gemm", [item["op"] for item in entry.known_sublaunches])

    def test_forward_f06_is_known_host_abort_not_completion(self):
        obj, _, calls, log = execution()
        error = RuntimeError("F06 after public return")
        fail_after_return(obj.forward_submission, error)
        with self.assertRaises(RuntimeError) as caught:
            obj.forward()
        self.assertIs(caught.exception, error)
        entry = obj.obligations.ledger._entry(obj.obligations.keys[0])
        self.assertEqual(entry.state, "ABORTED")
        self.assertEqual(
            [item["op"] for item in entry.known_sublaunches], ["X_dual", "gemm"]
        )
        self.assertEqual([item["expert"] for item in entry.known_sublaunches], [0, 0])
        self.assertEqual(calls, [("F", 0)])
        self.assertEqual(log, [])
        self.assertEqual(obj.control.async_errors, [])
        with self.assertRaises(RuntimeError):
            obj.forward_submission.call(1, lambda: calls.append("forbidden"))
        self.assertEqual(calls, [("F", 0)])

    def test_first_forward_call_rejection_has_no_launch_claim(self):
        obj, _, calls, log = execution()

        def before(index):
            raise RuntimeError("before first child")

        obj.forward_submission.enter = before
        with self.assertRaises(RuntimeError):
            obj.forward()
        entry = obj.obligations.ledger._entry(obj.obligations.keys[0])
        self.assertEqual(entry.state, "FAILED_PRELAUNCH")
        self.assertEqual(entry.known_sublaunches, ())
        self.assertEqual(calls + log, [])

    def test_gpu_event_error_before_enqueue_preserves_prefix_and_does_no_cleanup(self):
        obj, _, calls, log = execution()
        obj.forward_event.record_error = Mxfp4StateError("COMPONENT_HEALTH_FAILURE")
        with self.assertRaises(Mxfp4StateError):
            obj.forward()
        entry = obj.obligations.ledger._entry(obj.obligations.keys[0])
        self.assertEqual(entry.state, "INCOMPLETE_ASYNC")
        self.assertEqual(entry.unknown_operation, "forward.event_record")
        self.assertEqual(len(entry.known_sublaunches), 4)
        self.assertEqual(log, ["current_stream", "event.record"])
        self.assertEqual(calls, [("F", 0), ("F", 1)])
        self.assertFalse(entry.attempts[0].finished)

    def test_status_gpu_error_is_unknown_even_with_health_error_class(self):
        obj, _, _, log = execution()
        obj.mirror.copy_error = Mxfp4StateError("COMPONENT_HEALTH_FAILURE")
        with self.assertRaises(Mxfp4StateError):
            obj.forward()
        entry = obj.obligations.ledger._entry(obj.obligations.keys[0])
        self.assertEqual(entry.state, "INCOMPLETE_ASYNC")
        self.assertEqual(entry.unknown_operation, "status.copy")
        self.assertNotIn("event.synchronize", log)
        self.assertNotIn("event.query", log)

    def test_postlaunch_numeric_status_is_preserved(self):
        obj, _, _, _ = execution()
        obj.status.value = 4
        with self.assertRaises(Mxfp4StateError):
            obj.forward()
        entry = obj.obligations.ledger._entry(obj.obligations.keys[0])
        self.assertEqual(
            (entry.state, entry.semantic["status"], entry.numeric_status),
            ("ABORTED", 4, 4),
        )
        self.assertEqual(obj.control.async_errors, [])

    def test_failed_replay_has_nonterminal_domain_and_keeps_original_attempt(self):
        obj, api, _, log = execution()
        obj.forward()
        log.clear()

        def unknown(bound):
            raise RuntimeError("opaque replay error")

        api.mxfp4_dense_launch_packed_forward_v1 = unknown
        with self.assertRaises(RuntimeError):
            obj.forward(replay=True)
        ledger = obj.obligations.ledger
        entry = ledger._entry(obj.obligations.keys[0])
        self.assertEqual(entry.state, "INCOMPLETE_ASYNC")
        self.assertEqual(
            [a.status for a in entry.attempts], ["COMPLETED", "INCOMPLETE_ASYNC"]
        )
        self.assertEqual(ledger.counts()["COMPLETED"], 0)
        self.assertEqual(
            (ledger.counts()["open"], ledger.counts()["pending_physical"]), (1, 1)
        )
        self.assertEqual(log, [])
        with self.assertRaises(Mxfp4StateError):
            ledger.checkpoint(last_committed_step=0)

    def test_backward_returned_prefix_counts_shared_producer_once(self):
        obj, _, calls, log = execution()
        obj.forward()
        calls.clear()
        log.clear()
        fail_after_return(obj.backward_submission, RuntimeError("B host after return"))
        with self.assertRaises(RuntimeError):
            obj.backward(Tensor(obj.y.shape, log))
        ledger, keys = obj.obligations.ledger, obj.obligations.keys
        d, w = ledger._entry(keys[1]), ledger._entry(keys[2])
        self.assertEqual((d.state, w.state), ("ABORTED", "ABORTED"))
        self.assertEqual(
            [item["op"] for item in d.known_sublaunches], ["dY_dual", "gemm"]
        )
        self.assertEqual([item["op"] for item in w.known_sublaunches], ["gemm"])
        self.assertEqual(calls, [("B", 0)])
        self.assertEqual(log, [])

    def test_backward_status_failure_retains_numeric_bits_for_both_passes(self):
        obj, _, calls, log = execution()
        obj.forward()
        obj.status.value = 4
        with self.assertRaises(Mxfp4StateError):
            obj.backward(Tensor(obj.y.shape, log))
        ledger, keys = obj.obligations.ledger, obj.obligations.keys
        self.assertEqual(ledger._entry(keys[0]).state, "COMPLETED")
        for key in keys[1:]:
            entry = ledger._entry(key)
            self.assertEqual((entry.state, entry.numeric_status), ("ABORTED", 4))
        before = list(calls)
        with self.assertRaises(Mxfp4StateError):
            obj.backward(Tensor(obj.y.shape, log))
        self.assertEqual(calls, before)

    def test_cpu_vote_timeout_is_incomplete_without_gpu_recovery(self):
        obj, _, _, log = execution()
        vote = obj.control.vote

        def timeout(sequence, **kwargs):
            if sequence.phase == "FC_POSTF_STATUS":
                raise RuntimeError("store timeout")
            return vote(sequence, **kwargs)

        obj.control.vote = timeout
        with self.assertRaises(RuntimeError):
            obj.forward()
        entry = obj.obligations.ledger._entry(obj.obligations.keys[0])
        self.assertEqual((entry.state, entry.numeric_status), ("INCOMPLETE_ASYNC", 0))
        self.assertEqual(log[-1], "event.synchronize")
        self.assertNotIn("event.query", log)
        self.assertEqual(len(obj.control.async_errors), 1)

    def test_empty_accumulate_and_overwrite_failure_do_not_invent_gemm(self):
        for mode in (0, 1):
            fc = obligations((0,), mode)
            fc.begin(2)
            journal = SubmissionJournal([{"expert": 0, "logical_M": 0, "op": "empty"}])
            journal.call(0, lambda: None)
            fc.attach_journal(2, journal)
            fc.fail(2, RuntimeError("known empty boundary"), "BACKWARD_LAUNCH")
            entry = fc.ledger._entry(fc.keys[2])
            self.assertEqual(
                [item["op"] for item in entry.known_sublaunches],
                ["zero"] if mode == 0 else [],
            )
            self.assertEqual(
                entry.state, "ABORTED" if mode == 0 else "FAILED_PRELAUNCH"
            )
