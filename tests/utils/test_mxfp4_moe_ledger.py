"""Graph/physical accounting and deterministic replay tests, without GPU calls."""

import copy
import unittest

from lumen.utils.mxfp4_moe_ledger import (
    Mxfp4StateError,
    ObligationKey,
    ObligationLedger,
    aggregate_rank_records,
)


class _Event:
    def __init__(self, ready=True):
        self.ready = ready

    def query(self):
        return self.ready


def _semantic(key):
    return {
        "logical_binding_key": key.record(),
        "config_ids": [1, 2, 3],
        "routing_epoch": 0,
        "weight_epoch": 0,
        "rng_receipts": [],
        "shape": [32, 4096, 4096],
        "dtype": "BF16/FP32",
        "layout_ids": [1, 2, 0],
        "requested_mode": "sequential",
        "selected_mode": "sequential",
        "executed_mode": None,
        "kernel_family": None,
        "precision": None,
        "selection_events": [],
        "status": 0,
        "failure_phase": None,
        "reason": None,
    }


class TestMxfp4MoeLedger(unittest.TestCase):
    def _finish(self, ledger, key, *, replay=False, empty=False, event=None):
        attempt = ledger.begin_attempt(key, replay=replay)
        ledger.enqueue(
            key,
            attempt,
            sublaunches=(
                []
                if empty
                else [{"op": "gemm", "expert": 0, "config_id": 1, "ordinal": 0}]
            ),
            events=(event or _Event(),),
            execution=None if empty else ("sequential", "TRITON", "MXFP4"),
        )
        return ledger.finish(key, attempt, empty=empty)

    def test_graph_conservation_with_children_empty_and_failures(self):
        ledger = ObligationLedger("a" * 64)
        keys = [ObligationKey(0, 0, 0, 0, 0, fc, p) for fc in (1, 2) for p in range(3)]
        ledger.declare(reversed(keys))
        for key in keys:
            ledger.bind(key, **_semantic(key))
        self._finish(ledger, keys[0])
        self._finish(ledger, keys[1], empty=True)
        ledger.fail(keys[2], reason="BAD_CAPACITY", phase="BIND")
        ledger.abort_unlaunched(keys[3], "DEPENDENT_FAILURE")
        counts = ledger.counts()
        self.assertEqual(counts["declared"], 6)
        self.assertEqual(
            [
                counts[s]
                for s in ("COMPLETED", "EMPTY", "FAILED_PRELAUNCH", "ABORTED", "open")
            ],
            [1, 1, 1, 1, 2],
        )
        with self.assertRaises(Mxfp4StateError):
            ledger.checkpoint(last_committed_step=0)

    def test_event_completion_not_host_return(self):
        ledger = ObligationLedger("0" * 64)
        key = ObligationKey(0, 0, 0, 0, 0, 1, 2)
        ledger.declare([key])
        ledger.bind(key, **_semantic(key))
        event = _Event(False)
        self.assertFalse(self._finish(ledger, key, event=event))
        self.assertEqual(ledger.counts()["open"], 1)
        with self.assertRaises(Mxfp4StateError):
            ledger.checkpoint(last_committed_step=0)
        event.ready = True
        self.assertTrue(ledger.finish(key, 0))
        self.assertEqual(ledger.counts()["COMPLETED"], 1)
        with self.assertRaises(Mxfp4StateError):
            ledger.begin_attempt(key, replay=True)
        with self.assertRaises(Mxfp4StateError):
            ledger.begin_attempt(key)

    def test_physical_recompute_no_new_obligation(self):
        ledger = ObligationLedger("1" * 64)
        key = ObligationKey(0, 0, 0, 0, 0, 1, 0)
        ledger.declare([key])
        ledger.bind(key, **_semantic(key))
        self._finish(ledger, key)
        self._finish(ledger, key, replay=True)
        counts = ledger.counts()
        self.assertEqual(
            (counts["declared"], counts["COMPLETED"], counts["pending_physical"]),
            (1, 1, 0),
        )
        with self.assertRaises(Mxfp4StateError):
            ledger.begin_attempt(key, replay=True)

    def test_canonical_frontier_independent_of_completion_order(self):
        results = []
        for reverse in (False, True):
            ledger = ObligationLedger("b" * 64)
            keys = [
                ObligationKey(0, 0, 0, layer, layer, fc, p)
                for layer in range(2)
                for fc in (1, 2)
                for p in range(3)
            ]
            ledger.declare(reversed(keys))
            for key in keys:
                ledger.bind(key, **_semantic(key))
            for key in reversed(keys) if reverse else keys:
                self._finish(ledger, key)
            snapshot = ledger.checkpoint(last_committed_step=0)
            restored = ObligationLedger.restore(snapshot, manifest_digest="b" * 64)
            self.assertEqual(
                (ledger.chain, ledger.epoch), (restored.chain, restored.epoch)
            )
            next_key = ObligationKey(0, 1, 0, 0, 2, 1, 0)
            for branch in (ledger, restored):
                branch.declare([next_key])
                branch.bind(next_key, **_semantic(next_key))
                self._finish(branch, next_key)
            self.assertEqual(
                ledger.checkpoint(last_committed_step=1),
                restored.checkpoint(last_committed_step=1),
            )
            results.append(snapshot)
        self.assertEqual(results[0], results[1])

    def test_ephemeral_fields_and_malformed_frontier_rejected(self):
        ledger = ObligationLedger("c" * 64)
        key = ObligationKey(0, 0, 0, 0, 0, 1, 0)
        ledger.declare([key])
        with self.assertRaises(Mxfp4StateError):
            ledger.bind(key, **_semantic(key), binding_id=1234)
        ledger.bind(key, **_semantic(key))
        self._finish(ledger, key)
        snapshot = ledger.checkpoint(last_committed_step=0)
        bad = copy.deepcopy(snapshot)
        bad["frontier"]["declared"] += 1
        with self.assertRaises(Mxfp4StateError):
            ObligationLedger.restore(bad, manifest_digest="c" * 64)
        with self.assertRaises(Mxfp4StateError):
            ObligationLedger.restore(snapshot, manifest_digest="d" * 64)
        with self.assertRaises(Mxfp4StateError):
            ledger.declare([key])

    def test_async_error_never_queries_bad_device_or_closes_unknown_work(self):
        ledger = ObligationLedger("d" * 64)
        key = ObligationKey(0, 0, 0, 0, 0, 1, 0)
        ledger.declare([key])
        ledger.bind(key, **_semantic(key))
        attempt = ledger.begin_attempt(key)

        class BadDevice:
            def query(self):
                raise AssertionError("must not query broken GPU")

        ledger.enqueue(key, attempt, sublaunches=[], events=(BadDevice(),))
        ledger.fail(key, reason="ASYNC_DEVICE_ERROR", phase="ASYNC", asynchronous=True)
        self.assertEqual(ledger.counts()["open"], 1)
        self.assertEqual(ledger.counts()["COMPLETED"], 0)
        with self.assertRaises(Mxfp4StateError):
            ledger.checkpoint(last_committed_step=0)

    def test_postlaunch_numeric_error_is_aborted_not_candidate_miss(self):
        ledger = ObligationLedger("e" * 64)
        key = ObligationKey(0, 0, 0, 0, 0, 1, 0)
        ledger.declare([key])
        ledger.bind(key, **_semantic(key))
        attempt = ledger.begin_attempt(key)
        ledger.enqueue(key, attempt, sublaunches=[], events=(_Event(),))
        ledger.fail(key, reason="NUMERIC_STATUS_POSTLAUNCH", phase="POST_F")
        self.assertEqual(ledger.counts()["ABORTED"], 1)
        self.assertEqual(ledger.counts()["FAILED_PRELAUNCH"], 0)

    def test_all_rank_counts_digest_and_missing_rank(self):
        records = {
            rank: {"counts": ObligationLedger("a" * 64).counts(), "chain": "b" * 64}
            for rank in (0, 1)
        }
        result = aggregate_rank_records(records, (0, 1))
        self.assertEqual(result["counts"]["open"], 0)
        self.assertEqual(
            result,
            aggregate_rank_records(dict(reversed(list(records.items()))), (1, 0)),
        )
        with self.assertRaises(Mxfp4StateError):
            aggregate_rank_records({0: records[0]}, (0, 1))


if __name__ == "__main__":
    unittest.main()
