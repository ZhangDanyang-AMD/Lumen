"""Exact host-method behavior with bounded API doubles, not CUDA/Core evidence."""

import gc
import importlib.util
import unittest
import weakref
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
LEDGER = HOST.ledger


class GPU:
    def __init__(self):
        self.trace, self.events = [], []

    def Event(self):
        self.trace.append("allocate_event")
        event = Event(self)
        self.events.append(event)
        return event

    def current_stream(self, *unused):
        return self

    def wait_event(self, event):
        self.trace.append(("wait", event))

    def empty(self, *args, **kwargs):
        self.trace.append(("allocate_pinned", kwargs))
        return Payload()

    def zeros(self, *args, **kwargs):
        return self.empty(*args, **kwargs)

    def pinned_word(self, zero):
        factory = self.zeros if zero else self.empty
        return factory((1,), dtype="int32", device="cpu", pin_memory=True)


class Event:
    def __init__(self, gpu):
        self.gpu, self.ready = gpu, False

    def record(self, *unused):
        self.gpu.trace.append(("record", self))
        self.ready = False

    def query(self):
        return self.ready

    def synchronize(self):
        self.gpu.trace.append(("synchronize", self))
        for event in self.gpu.events:
            event.ready = True


class Payload:
    pass


class ResourceDriverTests(unittest.TestCase):
    def test_prepare_trace_and_pool_reuse_do_not_allocate_on_acquire(self):
        gpu = GPU()
        cls = HOST.resources.PreparedResources
        factories = {
            "event_factory": gpu.Event,
            "current_stream": gpu.current_stream,
            "pinned_word_factory": gpu.pinned_word,
        }
        with self.assertRaisesRegex(LEDGER.Mxfp4StateError, "HOST_PLAN_BREACH"):
            cls(fc_leases=2, experts=3, pinned_limit_bytes=15, **factories)
        self.assertEqual(gpu.trace, [])
        pool = cls(fc_leases=2, experts=3, pinned_limit_bytes=16, **factories)
        self.assertEqual(pool.telemetry["event_count"], 26)
        self.assertEqual(sum(item == "allocate_event" for item in gpu.trace), 26)
        self.assertEqual(
            sum(isinstance(item, tuple) and item[0] == "record" for item in gpu.trace),
            26,
        )
        pinned = [
            item[1]
            for item in gpu.trace
            if isinstance(item, tuple) and item[0] == "allocate_pinned"
        ]
        self.assertEqual(len(pinned), 4)
        self.assertTrue(
            all(item["pin_memory"] and item["device"] == "cpu" for item in pinned)
        )
        trace = list(gpu.trace)
        slot = pool.acquire("first")
        mirror = slot["mirror"]
        pool.acquire("second")
        with self.assertRaisesRegex(LEDGER.Mxfp4StateError, "DUPLICATE_RESOURCE_LEASE"):
            pool.acquire("first")
        with self.assertRaisesRegex(
            LEDGER.Mxfp4StateError, "LIVE_LEASE_CAPACITY_VIOLATION"
        ):
            pool.acquire("third")
        pending = slot["events"][0]
        pending.ready = False
        with self.assertRaisesRegex(
            LEDGER.Mxfp4StateError, "RESOURCE_CONSUMERS_NOT_READY"
        ):
            pool.release("first", (pending,))
        self.assertEqual(slot["owner"], "first")
        pending.ready = True
        pool.release("first", (pending,))
        self.assertIs(pool.acquire("third"), slot)
        self.assertIs(slot["mirror"], mirror)
        self.assertEqual(gpu.trace, trace)

    def test_lifecycle_strong_refs_live_until_consumer_and_reducer_ready(self):
        gpu = GPU()
        state_cls = HOST.state.WeightEpochState
        weights = state_cls()
        zero = gpu.Event()
        zero.synchronize()
        weights.open_window(zero_events=(zero,))
        consumed = gpu.Event()
        calls = []
        lifecycle_cls = HOST.lifecycle.RoutedLifecycle
        handoff = SimpleNamespace(
            before_grad_sync=lambda: calls.append("before"),
            close_after_grad_sync=lambda event: calls.append("close"),
        )
        lifecycle = lifecycle_cls(
            SimpleNamespace(finish_grad_sync=lambda: calls.append("reduce")),
            handoff,
            weights,
            SimpleNamespace(retire_scope=lambda key, events: calls.append("rng")),
            SimpleNamespace(counts=lambda: {"open": 0, "pending_physical": 0}),
            event_factory=gpu.Event,
            current_stream=gpu.current_stream,
        )
        payload = Payload()
        reference = weakref.ref(payload)
        execution = SimpleNamespace(
            zero_leases=(SimpleNamespace(consumed_event=consumed),),
            metadata=payload,
            retire_storage=lambda events: calls.append("storage"),
        )
        invocation = Payload()
        invocation.scope, invocation.executions = "scope", {1: execution}
        lifecycle.retain(invocation)
        del payload, invocation, execution
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertFalse(weights.release("scope", events=(consumed,)))
        self.assertIn("scope", lifecycle.active)
        lifecycle.finish_grad_sync()
        self.assertEqual(calls, ["before", "reduce", "close", "rng", "storage"])
        self.assertFalse(lifecycle.active)
        self.assertFalse(weights._leases)
        gc.collect()
        self.assertIsNone(reference())

    def test_handoff_waits_and_observer_gate_without_new_event_allocation(self):
        gpu = GPU()
        cls = HOST.ddp.Mxfp4DDPHandoff
        handoff = cls.__new__(cls)
        handoff._current_stream = gpu.current_stream
        parameter = Payload()
        parameter.device, parameter.grad = "cuda", None
        parameter.grad_added_to_main_grad = False
        binding = SimpleNamespace(parameter=parameter, window=0)
        emission = SimpleNamespace(
            dtype="bf16",
            device="cuda",
            shape=(1, 4),
            stride=lambda: (4, 1),
            element_size=lambda: 2,
            storage_offset=lambda: 0,
            untyped_storage=lambda: SimpleNamespace(
                nbytes=lambda: 8, data_ptr=lambda: 1234
            ),
        )
        sink = Payload()
        lease = SimpleNamespace(
            binding=binding,
            emission=emission,
            sink=sink,
            zero_event=gpu.Event(),
            w_event=gpu.Event(),
            consumed_event=gpu.Event(),
            armed=False,
            consumed=False,
            validate=lambda: None,
        )
        handoff.window, handoff.bindings = 0, {parameter: binding}
        handoff.pending, handoff.leases = {}, []
        handoff.zero_event = gpu.Event()
        prepared_events = len(gpu.events)
        self.assertIs(handoff.arm(lease), emission)
        self.assertTrue(parameter.grad_added_to_main_grad)
        self.assertEqual(
            gpu.trace[-2:], [("wait", lease.w_event), ("wait", lease.zero_event)]
        )
        with self.assertRaisesRegex(LEDGER.Mxfp4StateError, "DDP_HOOK_NOT_CONSUMED"):
            handoff.before_grad_sync()
        self.assertIs(handoff._prehook(parameter)(emission), emission)
        self.assertIs(parameter.grad, sink)
        self.assertEqual(
            gpu.trace[-3:],
            [
                ("wait", lease.w_event),
                ("wait", lease.zero_event),
                ("wait", handoff.zero_event),
            ],
        )
        with self.assertRaisesRegex(LEDGER.Mxfp4StateError, "DDP_HOOK_NOT_CONSUMED"):
            handoff._observer(parameter)()
        # Model only the original Core hook's observable grad-clear postcondition.
        parameter.grad = None
        handoff._observer(parameter)()
        self.assertTrue(lease.consumed)
        self.assertFalse(handoff.pending)
        handoff.before_grad_sync()
        self.assertEqual(
            gpu.trace[-2:], [("wait", lease.w_event), ("wait", lease.consumed_event)]
        )
        self.assertEqual(len(gpu.events), prepared_events)

    def runtime(self, fail, *, layer=False, reporting_fails=False):
        cls = HOST.driver.RoutedDriver
        runtime = cls.__new__(cls)
        runtime._driver_failed = False
        runtime._microbatch_count = 1
        runtime._next_microbatch = 0
        runtime._microbatch_scope_active = None
        runtime.microbatch = None
        calls = []

        def operation(*args):
            calls.append("operation")
            raise fail

        def abort(exc):
            calls.append(("abort", exc))
            if reporting_fails:
                raise OSError("journal unavailable")

        runtime.control = SimpleNamespace(failed=False, async_abort=abort)
        runtime.lifecycle = SimpleNamespace(
            zero_boundary=operation,
            finish_grad_sync=operation,
            optimizer_result=operation,
        )
        runtime.max_live_invocations_per_layer = 1
        runtime.layers = (operation,) if layer else ()
        return runtime, calls

    def test_forward_preserves_first_exception_clears_scope_and_blocks_step(self):
        for failure in (
            RuntimeError("launch unknown"),
            LEDGER.Mxfp4StateError("COMPONENT_HEALTH_FAILURE"),
        ):
            runtime, calls = self.runtime(failure, layer=True, reporting_fails=True)
            with self.assertRaises(type(failure)) as caught:
                runtime.forward(object(), 0)
            self.assertIs(caught.exception, failure)
            self.assertIsNone(runtime.microbatch)
            self.assertEqual(
                sum(isinstance(item, tuple) for item in calls),
                0 if isinstance(failure, LEDGER.Mxfp4StateError) else 1,
            )
            previous = list(calls)
            for operation in (
                lambda runtime=runtime: runtime.forward(object(), 0),
                runtime.finish_grad_sync,
                lambda runtime=runtime: runtime.optimizer_result(True),
                lambda runtime=runtime: runtime.zero_boundary(1),
            ):
                with self.assertRaisesRegex(
                    LEDGER.Mxfp4StateError, "ABORTED_DRIVER_SCOPE"
                ):
                    operation()
            self.assertEqual(calls, previous)

    def test_zero_reducer_and_optimizer_errors_stop_later_submissions(self):
        for name in ("zero_boundary", "finish_grad_sync", "optimizer_result"):
            failure = RuntimeError(name)
            runtime, calls = self.runtime(failure)
            args = (
                (1,)
                if name == "zero_boundary"
                else ((True,) if name == "optimizer_result" else ())
            )
            with self.assertRaises(RuntimeError) as caught:
                getattr(runtime, name)(*args)
            self.assertIs(caught.exception, failure)
            self.assertEqual(calls, ["operation", ("abort", failure)])
            with self.assertRaisesRegex(LEDGER.Mxfp4StateError, "ABORTED_DRIVER_SCOPE"):
                runtime.optimizer_result(True)
            self.assertEqual(len(calls), 2)

    def test_existing_control_failure_rejects_before_driver_operation(self):
        runtime, calls = self.runtime(RuntimeError("must not run"))
        runtime.control.failed = True
        with self.assertRaisesRegex(LEDGER.Mxfp4StateError, "ABORTED_DRIVER_SCOPE"):
            runtime.finish_grad_sync()
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
