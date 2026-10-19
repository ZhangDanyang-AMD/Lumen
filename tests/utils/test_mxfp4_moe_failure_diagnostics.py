"""PF-01/02: real control journal methods, with host-only submission fixtures."""

import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path

from lumen.utils.mxfp4_moe_control import Mxfp4Control
from lumen.utils.mxfp4_moe_ledger import Mxfp4StateError

_spec = importlib.util.spec_from_file_location(
    "partial_failure_fixtures_for_diagnostics",
    Path(__file__).with_name("test_mxfp4_moe_partial_failure.py"),
)
fixtures = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fixtures
_spec.loader.exec_module(fixtures)


class JournalStream(io.StringIO):
    def __init__(self, failure=None, verify=None):
        super().__init__()
        self.failure = failure
        self.verify = verify
        self.writes = 0
        self.flushes = 0
        self.state_verified_before_write = False

    def write(self, value):
        self.writes += 1
        if self.verify:
            self.verify()
            self.state_verified_before_write = True
        if self.failure == "write":
            raise OSError("disk write failed")
        return super().write(value)

    def flush(self):
        self.flushes += 1
        if self.verify:
            self.verify()
        if self.failure == "flush":
            raise OSError("disk flush failed")
        return super().flush()


def real_control(stream):
    # Do not construct a process group/store. record/failure_record/async_abort are real.
    control = object.__new__(Mxfp4Control)
    control.failed = False
    control.rank = 0
    control.physical_run_id = "diagnostic-host-fixture"
    control._journal = stream
    control._failure_reporter = fixtures.HOST.diagnostics.FailureReporter(
        stream, control.rank, control.physical_run_id
    )
    control.vote = fixtures.Control().vote
    return control


class FailureDiagnosticTests(unittest.TestCase):
    def test_execution_write_or_flush_failure_cannot_mask_first_error(self):
        deferred = []
        for failure in ("write", "flush"):
            with self.subTest(failure=failure):
                obj, api, calls, log = fixtures.execution()
                original = RuntimeError("first opaque launch error")
                ledger, key = obj.obligations.ledger, obj.obligations.keys[0]

                def verify_saved(ledger=ledger, key=key, obj=obj, original=original):
                    entry = ledger._entry(key)
                    self.assertEqual(entry.state, "INCOMPLETE_ASYNC")
                    self.assertEqual(len(entry.known_sublaunches), 2)
                    self.assertEqual(entry.submission_facts[1]["submission"], "ENTERED")
                    self.assertFalse(entry.attempts[0].finished)
                    self.assertIs(obj.obligations.first_failure, original)

                stream = JournalStream(failure, verify_saved)
                obj.control = real_control(stream)

                def launch(bound, source, *, calls=calls, original=original):
                    calls.append(bound)
                    if bound == 1:
                        raise original

                deferred.append((verify_saved, launch, original, calls))
                api.mxfp4_dense_launch_forward_v1 = launch
                with self.assertRaises(RuntimeError) as caught:
                    obj.forward()
                self.assertIs(caught.exception, original)
                verify_saved()
                self.assertTrue(stream.state_verified_before_write)
                self.assertEqual(stream.writes, 1)
                self.assertFalse(obj.obligations.failure_output_written)
                self.assertEqual(obj.control.failure_output_error, "OSError")
                self.assertEqual(log, [])
        for verify, launch, primary, recorded in deferred:
            verify()
            with self.assertRaises(RuntimeError) as caught:
                launch(1, None)
            self.assertIs(caught.exception, primary)
            self.assertEqual(recorded[-1], 1)

    def test_materializer_write_or_flush_failure_keeps_prefix_and_causal_states(self):
        deferred = []
        for failure in ("write", "flush"):
            with self.subTest(failure=failure):
                fc = fixtures.obligations()
                original = RuntimeError("first materializer primitive error")

                def verify_saved(fc=fc, original=original):
                    self.assertEqual(
                        [fc.ledger._entry(key).state for key in fc.keys],
                        ["INCOMPLETE_ASYNC", "ABORTED", "ABORTED"],
                    )
                    entry = fc.ledger._entry(fc.keys[0])
                    self.assertEqual(
                        [item["op"] for item in entry.known_sublaunches],
                        ["status_init", "cold_status_init"],
                    )
                    self.assertIs(fc.first_failure, original)

                stream = JournalStream(failure, verify_saved)
                control = real_control(stream)

                def work(x, progress, *, original=original, **kwargs):
                    for index in (0, 1):
                        progress["journal"].call(index, lambda: None)

                    def broken():
                        raise original

                    progress["journal"].call(2, broken)

                deferred.append((verify_saved, work, original))
                with self.assertRaises(RuntimeError) as caught:
                    fixtures.materializer(work)(
                        None,
                        prepared=fc.prepared,
                        bindings=(1, 2),
                        obligations=fc,
                        control=control,
                    )
                self.assertIs(caught.exception, original)
                verify_saved()
                self.assertTrue(stream.state_verified_before_write)
                self.assertEqual(stream.writes, 1)
                self.assertFalse(fc.failure_output_written)
        for verify, work, primary in deferred:
            verify()
            journal = fixtures.SubmissionJournal({"slot": index} for index in range(3))
            with self.assertRaises(RuntimeError) as caught:
                work(None, {"journal": journal})
            self.assertIs(caught.exception, primary)

    def test_failed_replay_keeps_old_attempt_when_real_record_flush_fails(self):
        obj, api, _, log = fixtures.execution()
        obj.forward()
        log.clear()
        original = RuntimeError("replay first error")
        stream = JournalStream("flush")
        obj.control = real_control(stream)

        def broken(bound):
            raise original

        api.mxfp4_dense_launch_packed_forward_v1 = broken
        with self.assertRaises(RuntimeError) as caught:
            obj.forward(replay=True)
        self.assertIs(caught.exception, original)
        entry = obj.obligations.ledger._entry(obj.obligations.keys[0])
        self.assertEqual(
            [attempt.status for attempt in entry.attempts],
            ["COMPLETED", "INCOMPLETE_ASYNC"],
        )
        payload = obj.obligations.failure_context["obligations"][0]
        self.assertEqual(payload["physical_attempt"], 1)
        self.assertEqual(payload["physical_attempts"][0]["state"], "COMPLETED")
        self.assertFalse(payload["physical_attempts"][1]["finished"])
        self.assertEqual(log, [])

    def test_healthy_f06_payload_is_written_once_outside_canonical_state(self):
        obj, _, _, log = fixtures.execution()
        original = RuntimeError("healthy F06")
        stream = JournalStream()
        obj.control = real_control(stream)
        chain = obj.obligations.ledger.chain
        fixtures.fail_after_return(obj.forward_submission, original)
        with self.assertRaises(RuntimeError) as caught:
            obj.forward()
        self.assertIs(caught.exception, original)
        record = json.loads(stream.getvalue())
        self.assertEqual(record["event"], "submission_failure")
        failure = record["failure"]
        self.assertEqual(
            failure["first_error"],
            {"type": "builtins.RuntimeError", "message": "healthy F06"},
        )
        self.assertFalse(failure["asynchronous_or_unknown"])
        current = failure["obligations"][0]
        self.assertEqual(current["logical_key"], obj.obligations.keys[0].record())
        self.assertEqual(
            (current["physical_attempt"], current["state"]), (0, "ABORTED")
        )
        self.assertEqual(
            [child["op"] for child in current["known_prefix"]], ["X_dual", "gemm"]
        )
        self.assertTrue(all("config_id" in child for child in current["known_prefix"]))
        self.assertEqual(
            [item["submission"] for item in current["submission_facts"]],
            ["PUBLIC_RETURNED", "NOT_ENTERED"],
        )
        self.assertIsNone(current["unknown_operation"])
        self.assertIsNone(current["numeric_snapshot"])
        self.assertTrue(obj.obligations.failure_output_written)
        self.assertEqual(obj.obligations.ledger.chain, chain)
        self.assertNotIn(
            "failure", obj.obligations.ledger._entry(obj.obligations.keys[0]).semantic
        )
        obj.obligations.emit_failure(
            obj.control, RuntimeError("secondary"), "SECONDARY", asynchronous=True
        )
        self.assertEqual(stream.writes, 1)
        self.assertIs(obj.obligations.first_failure, original)
        self.assertEqual(log, [])

    def test_async_payload_exports_unknown_operation(self):
        obj, _, _, _ = fixtures.execution()
        original = RuntimeError("event API error")
        obj.forward_event.record_error = original
        stream = JournalStream()
        obj.control = real_control(stream)
        with self.assertRaises(RuntimeError) as caught:
            obj.forward()
        self.assertIs(caught.exception, original)
        record = json.loads(stream.getvalue())
        self.assertEqual(record["event"], "async_abort")
        entry = record["failure"]["obligations"][0]
        self.assertEqual(entry["unknown_operation"], "forward.event_record")
        self.assertEqual(entry["state"], "INCOMPLETE_ASYNC")
        self.assertEqual(len(entry["known_prefix"]), 4)

    def test_successful_numeric_failure_output_contains_real_snapshot(self):
        obj, _, _, _ = fixtures.execution()
        obj.status.value = 4
        stream = JournalStream()
        obj.control = real_control(stream)
        with self.assertRaises(Mxfp4StateError) as caught:
            obj.forward()
        self.assertIs(obj.obligations.first_failure, caught.exception)
        entry = json.loads(stream.getvalue())["failure"]["obligations"][0]
        self.assertEqual((entry["state"], entry["numeric_snapshot"]), ("ABORTED", 4))
        self.assertTrue(obj.obligations.failure_output_written)
