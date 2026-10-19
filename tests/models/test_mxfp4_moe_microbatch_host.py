"""Real driver protocol tests; normal pinned-Core import is a separate guarded case."""

import hashlib
import importlib
import importlib.util
import os
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
CORE_SOURCE_ENV = "MXFP4_CORE_SOURCE_ROOT"
CORE_SHA256 = {
    "distributed_data_parallel.py": "e7eeb8db32c1521f64722b6f03c1c8221ed2d9dba938ac84295d48d60f767f7f",
    "param_and_grad_buffer.py": "88ba45d759c6ad0484ee82c0b990bec85ad591d221183bfebff697a8be477220",
}


def load_host_modules():
    spec = importlib.util.spec_from_file_location(
        "mxfp4_host_imports", ROOT / "tests/mxfp4_host_imports.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_host_modules()


HOST = load_host_modules()
Driver = HOST.driver.RoutedDriver
StateError = HOST.ledger.Mxfp4StateError


def core_source_directory():
    configured = os.environ.get(CORE_SOURCE_ENV)
    if not configured:
        raise unittest.SkipTest(f"UNRUN: set {CORE_SOURCE_ENV} to pinned Core 1b754411")
    directory = Path(configured) / "megatron/core/distributed"
    for filename, expected in CORE_SHA256.items():
        path = directory / filename
        if not path.is_file():
            raise ValueError(f"configured Core source missing: {path}")
        data = path.read_text(encoding="utf-8").encode("utf-8")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Core 1b754411 source identity mismatch: {path}")
    return directory.resolve()


class ContextBoundary:
    """Record calls to the injected context API; do not implement a Core reducer."""

    def __init__(self):
        self.active = False
        self.enters = 0
        self.exits = 0

    @contextmanager
    def no_sync(self):
        self.enters += 1
        self.active = True
        try:
            yield
        finally:
            self.active = False
            self.exits += 1


def fixture(count=2):
    runtime = Driver.__new__(Driver)
    runtime._driver_failed = False
    runtime._microbatch_count, runtime._next_microbatch = count, 0
    runtime._microbatch_scope_active = None
    runtime.microbatch = None
    runtime.scopes = {
        (micro, layer): (micro, layer) for micro in range(count) for layer in range(2)
    }
    reports = []
    runtime.control = SimpleNamespace(failed=False, async_abort=reports.append)
    boundary = ContextBoundary()
    runtime.ddp = boundary
    runtime.layers = (lambda value: (value, None), lambda value: (value, None))
    runtime.lifecycle = SimpleNamespace(
        active={}, finish_grad_sync=lambda: None, optimizer_result=lambda success: None
    )
    return runtime, boundary, reports


def consume(runtime, microbatch, *, empty=False):
    # Explicit completed-hook receipts; not tensors, Core hooks or GPU execution.
    for layer in range(2):
        executions = {
            fc: SimpleNamespace(
                backward_started=True,
                zero_leases=(SimpleNamespace(consumed=True),),
                logical_empty=empty,
            )
            for fc in (1, 2)
        }
        runtime.lifecycle.active[(microbatch, layer)] = SimpleNamespace(
            executions=executions
        )


class MicrobatchHostTests(unittest.TestCase):
    def test_two_microbatches_keep_nonlast_context_through_backward(self):
        for empty in (False, True):
            with self.subTest(empty=empty):
                runtime, boundary, reports = fixture()
                for microbatch in range(2):
                    with runtime.microbatch_scope(microbatch):
                        self.assertEqual(boundary.active, microbatch == 0)
                        value = object()
                        self.assertIs(runtime.forward(value, microbatch), value)
                        self.assertEqual(runtime.microbatch, microbatch)
                        consume(runtime, microbatch, empty=empty)
                        self.assertEqual(boundary.active, microbatch == 0)
                    self.assertFalse(boundary.active)
                    self.assertIsNone(runtime.microbatch)
                runtime.finish_grad_sync()
                self.assertEqual((boundary.enters, boundary.exits), (1, 1))
                self.assertFalse(reports)

    def test_scope_exit_without_backward_or_hook_consumption_is_fatal(self):
        for missing in ("all", "fc", "backward", "hook"):
            runtime, boundary, _ = fixture()
            with self.assertRaisesRegex(
                StateError, "MICROBATCH_BACKWARD_INCOMPLETE"
            ), runtime.microbatch_scope(0):
                runtime.forward(object(), 0)
                if missing != "all":
                    consume(runtime, 0)
                    execution = runtime.lifecycle.active[(0, 0)].executions[1]
                    if missing == "fc":
                        del runtime.lifecycle.active[(0, 0)].executions[2]
                    elif missing == "backward":
                        execution.backward_started = False
                    else:
                        execution.zero_leases[0].consumed = False
            self.assertEqual(runtime._next_microbatch, 0)
            self.assertFalse(boundary.active)
            with self.assertRaisesRegex(StateError, "ABORTED_DRIVER_SCOPE"):
                runtime.finish_grad_sync()

    def test_backward_exception_restores_context_and_blocks_step(self):
        runtime, boundary, reports = fixture()
        failure = RuntimeError("backward failed")
        with self.assertRaises(RuntimeError) as caught, runtime.microbatch_scope(0):
            runtime.forward(object(), 0)
            raise failure
        self.assertIs(caught.exception, failure)
        self.assertEqual(reports, [failure])
        self.assertFalse(boundary.active)
        self.assertIsNone(runtime._microbatch_scope_active)
        self.assertIsNone(runtime.microbatch)
        with self.assertRaisesRegex(StateError, "ABORTED_DRIVER_SCOPE"):
            runtime.optimizer_result(True)

    def test_multi_bare_forward_rejected_but_single_batch_compatible(self):
        runtime, _, _ = fixture()
        with self.assertRaisesRegex(StateError, "MICROBATCH_SCOPE_REQUIRED"):
            runtime.forward(object(), 0)
        single, _, _ = fixture(count=1)
        value = object()
        self.assertIs(single.forward(value, 0), value)
        self.assertIsNone(single.microbatch)
        for index in (None, True, 2):
            runtime, _, _ = fixture()
            with self.assertRaisesRegex(StateError, "MICROBATCH_SCOPE_MISMATCH"):
                runtime.forward(object(), index)

    def test_order_nested_and_early_step_are_rejected(self):
        runtime, _, _ = fixture()
        with self.assertRaisesRegex(
            StateError, "MICROBATCH_SCOPE_MISMATCH"
        ), runtime.microbatch_scope(1):
            self.fail("out-of-order context entered")
        runtime, boundary, _ = fixture()
        with self.assertRaisesRegex(
            StateError, "MICROBATCH_SCOPE_MISMATCH"
        ), runtime.microbatch_scope(0), runtime.microbatch_scope(0):
            self.fail("nested context entered")
        self.assertFalse(boundary.active)
        runtime, _, _ = fixture()
        with self.assertRaisesRegex(StateError, "MICROBATCH_BACKWARD_INCOMPLETE"):
            runtime.finish_grad_sync()


class PinnedCoreImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("MXFP4_AUTHORIZED_RUNTIME") != "1":
            raise unittest.SkipTest(
                "UNRUN: normal Core import requires the authorized environment"
            )
        directory = core_source_directory()
        if importlib.util.find_spec("megatron") is None:
            raise unittest.SkipTest("UNRUN: pinned Megatron-Core is not installed")
        ddp = importlib.import_module(
            "megatron.core.distributed.distributed_data_parallel"
        )
        buckets = importlib.import_module(
            "megatron.core.distributed.param_and_grad_buffer"
        )
        if (
            Path(ddp.__file__).resolve() != directory / "distributed_data_parallel.py"
            or Path(buckets.__file__).resolve()
            != directory / "param_and_grad_buffer.py"
        ):
            raise ValueError(
                "imported Core does not match the configured pinned source"
            )
        cls.no_sync = staticmethod(ddp.DistributedDataParallel.no_sync)
        cls.register_ready = staticmethod(
            buckets._ParamAndGradBucketGroup.register_grad_ready
        )

    def test_imported_core_no_sync_defers_actual_ready_method(self):
        dispatches = []
        group = SimpleNamespace(
            is_last_microbatch=True,
            params=("p",),
            params_with_grad=set(),
            param_to_bucket={"p": None},
            ddp_config=SimpleNamespace(overlap_grad_reduce=True),
            start_grad_sync=lambda: dispatches.append("sync"),
        )
        ddp = SimpleNamespace(bucket_groups=[group], expert_parallel_bucket_groups=[])
        no_sync = MethodType(self.no_sync, ddp)
        ready = MethodType(self.register_ready, group)
        with no_sync():
            ready("p")
            self.assertFalse(group.params_with_grad)
            self.assertFalse(dispatches)
        ready("p")
        self.assertEqual(group.params_with_grad, {"p"})
        self.assertEqual(dispatches, ["sync"])


if __name__ == "__main__":
    unittest.main()
