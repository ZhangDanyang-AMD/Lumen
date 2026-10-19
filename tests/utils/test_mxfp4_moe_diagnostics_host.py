"""Real reporter/control/ledger behavior with actual in-memory stream failures."""

import importlib.util
import io
import json
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def load_host_modules():
    spec = importlib.util.spec_from_file_location(
        "mxfp4_host_imports", ROOT / "tests/mxfp4_host_imports.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_host_modules()


HOST = load_host_modules()


class FaultStream(io.StringIO):
    def __init__(self, stage=None, secondary=None, entry=None):
        super().__init__()
        self.stage = stage
        self.secondary = secondary
        self.entry = entry
        self.writes = 0
        self.flushes = 0
        self.state_saved = False

    def write(self, text):
        self.writes += 1
        if self.entry is not None:
            self.state_saved = self.entry.state == "INCOMPLETE_ASYNC"
            if not self.state_saved:
                raise AssertionError("failure state was not saved before output")
        if self.stage == "write":
            raise self.secondary
        return super().write(text)

    def flush(self):
        self.flushes += 1
        if self.stage == "flush":
            raise self.secondary
        return super().flush()


class BadString(RuntimeError):
    def __init__(self, secondary):
        super().__init__("primary")
        self.secondary = secondary

    def __str__(self):
        raise self.secondary


def failure_fixture(primary):
    view = SimpleNamespace(
        gemm=lambda pass_id: SimpleNamespace(config=(4, pass_id)),
        primitive=lambda *args, **kwargs: SimpleNamespace(config=(2, 2)),
        w_output_mode=1,
        w_output_dtype=1,
    )
    scope = SimpleNamespace(
        rng_epoch=0,
        attempted_step=0,
        microbatch_ordinal=0,
        global_layer_ordinal=0,
        logical_invocation_id=0,
    )
    route = SimpleNamespace(counts=(1,), sorted_rows=1, routing_epoch=0)
    prepared = SimpleNamespace(
        config_view=view, static_spec=(0, 0, 0, 0, 0, 4096, 2048)
    )
    fc = HOST.execution_ledger.FCObligations(
        HOST.ledger.ObligationLedger("a" * 64), scope, 1, route, 0, (), prepared
    )
    fc.begin(0)
    journal = HOST.submission.SubmissionJournal(({"expert": 0, "logical_M": 1},))
    journal.enter(0)
    journal.returned(0)
    fc.attach_journal(0, journal)
    fc.fail(0, primary, "FORWARD_LAUNCH", asynchronous=True)
    return fc


def control_for(stream):
    control = HOST.control.Mxfp4Control.__new__(HOST.control.Mxfp4Control)
    control.rank, control.physical_run_id, control.failed = 0, "diagnostic-test", False
    control._journal = stream
    control._failure_reporter = HOST.diagnostics.FailureReporter(
        stream, 0, "diagnostic-test"
    )
    return control


class ReporterTests(unittest.TestCase):
    def test_write_flush_and_encoding_failures_return_visible_results_once(self):
        for stage, secondary in (
            ("write", OSError("write")),
            ("flush", OSError("flush")),
            ("write", UnicodeError("encoding")),
        ):
            with self.subTest(stage=stage, error=type(secondary).__name__):
                primary = RuntimeError("primary")
                fc = failure_fixture(primary)
                stream = FaultStream(stage, secondary, fc.ledger._entry(fc.keys[0]))
                control = control_for(stream)
                self.assertFalse(
                    fc.emit_failure(
                        control, primary, "FORWARD_LAUNCH", asynchronous=True
                    )
                )
                result = control.failure_diagnostic_result
                self.assertIs(result.secondary, secondary)
                self.assertEqual(result.stage, stage)
                self.assertTrue(stream.state_saved)
                self.assertIs(fc.first_failure, primary)
                self.assertEqual(stream.writes, 1)
                fc.emit_failure(
                    control, RuntimeError("later"), "OTHER", asynchronous=True
                )
                self.assertIs(fc.first_failure, primary)
                self.assertEqual(stream.writes, 1)

    def test_actual_closed_stream_value_error_is_reported_not_retried(self):
        primary = RuntimeError("primary")
        fc = failure_fixture(primary)
        stream = FaultStream(entry=fc.ledger._entry(fc.keys[0]))
        stream.close()
        control = control_for(stream)
        self.assertFalse(
            fc.emit_failure(control, primary, "FORWARD_LAUNCH", asynchronous=True)
        )
        self.assertEqual(control.failure_diagnostic_result.stage, "write")
        self.assertIsInstance(control.failure_diagnostic_result.secondary, ValueError)
        self.assertEqual(stream.writes, 1)

    def test_unexpected_write_and_recursion_errors_chain_to_primary(self):
        for secondary in (
            RuntimeError("unexpected write"),
            RecursionError("write recursion"),
            ValueError("not closed"),
        ):
            primary = RuntimeError("primary")
            fc = failure_fixture(primary)
            stream = FaultStream("write", secondary, fc.ledger._entry(fc.keys[0]))
            control = control_for(stream)
            with self.assertRaises(RuntimeError) as caught:
                fc.emit_failure(control, primary, "FORWARD_LAUNCH", asynchronous=True)
            self.assertIs(caught.exception, primary)
            self.assertIs(primary.__cause__, secondary)
            self.assertIs(control.failure_diagnostic_result.secondary, secondary)
            self.assertIs(fc.failure_diagnostic_secondary, secondary)
            self.assertTrue(stream.state_saved)
            self.assertEqual(stream.writes, 1)

    def test_payload_string_failure_preserves_saved_state_and_primary_identity(self):
        for secondary in (
            RuntimeError("str failed"),
            RecursionError("str recursion"),
            ValueError("payload value"),
        ):
            primary = BadString(secondary)
            fc = failure_fixture(primary)
            stream = FaultStream(entry=fc.ledger._entry(fc.keys[0]))
            control = control_for(stream)
            with self.assertRaises(BadString) as caught:
                fc.emit_failure(control, primary, "FORWARD_LAUNCH", asynchronous=True)
            self.assertIs(caught.exception, primary)
            self.assertIs(primary.__cause__, secondary)
            self.assertEqual(fc.ledger._entry(fc.keys[0]).state, "INCOMPLETE_ASYNC")
            self.assertEqual(control.failure_diagnostic_result.stage, "payload")
            self.assertEqual(stream.writes, 0)

    def test_format_error_chains_without_any_output(self):
        primary = RuntimeError("primary")
        fc = failure_fixture(primary)
        fc.ledger._entry(fc.keys[0]).numeric_status = object()
        stream = FaultStream()
        control = control_for(stream)
        with self.assertRaises(RuntimeError) as caught:
            fc.emit_failure(control, primary, "FORWARD_LAUNCH", asynchronous=True)
        self.assertIs(caught.exception, primary)
        self.assertIsInstance(primary.__cause__, TypeError)
        self.assertEqual(control.failure_diagnostic_result.stage, "format")
        self.assertEqual(stream.writes, 0)

    def test_success_has_fixed_payload_prefix_and_single_output(self):
        primary = RuntimeError("primary")
        fc = failure_fixture(primary)
        entry = fc.ledger._entry(fc.keys[0])
        entry.numeric_status = 4
        stream = FaultStream(entry=entry)
        control = control_for(stream)
        self.assertTrue(
            fc.emit_failure(control, primary, "FORWARD_LAUNCH", asynchronous=True)
        )
        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["failure"]["obligations"][0]["numeric_snapshot"], 4)
        self.assertEqual(len(payload["failure"]["obligations"][0]["known_prefix"]), 2)
        self.assertEqual((stream.writes, stream.flushes), (1, 1))
        self.assertNotIn("secondary", payload)

    def test_driver_callback_error_retains_primary_and_visible_cause(self):
        primary, secondary = RuntimeError("primary"), RecursionError("callback")
        driver = HOST.driver.RoutedDriver()
        driver._driver_failed = False

        def fail_report(error):
            raise secondary

        driver.control = SimpleNamespace(async_abort=fail_report)
        with self.assertRaises(RuntimeError) as caught:
            driver._driver_failure(primary)
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, secondary)
        self.assertTrue(driver._driver_failed)
        self.assertIs(driver.failure_diagnostic_secondary, secondary)

    def test_driver_preserves_reporter_secondary_on_primary_rethrow(self):
        primary = RuntimeError("primary")
        secondary = RecursionError("reporter write")
        driver = HOST.driver.RoutedDriver()
        driver._driver_failed = False
        driver.control = control_for(FaultStream("write", secondary))
        with self.assertRaises(RuntimeError) as caught:
            driver._driver_failure(primary)
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, secondary)
        self.assertIs(driver.failure_diagnostic_secondary, secondary)
        self.assertTrue(driver._driver_failed)

    def test_control_owns_journal_until_explicit_close(self):
        control = control_for(io.StringIO())
        control._journal_lifetime = ExitStack()
        control._journal_lifetime.enter_context(control._journal)
        self.assertFalse(control._journal.closed)
        control.close()
        self.assertTrue(control._journal.closed)


if __name__ == "__main__":
    unittest.main()
