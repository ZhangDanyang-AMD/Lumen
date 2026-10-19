# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0
"""Host-only MXFP4 obligation accounting and deterministic replay frontiers."""

import hashlib
import json
from dataclasses import dataclass, field
from threading import RLock

UINT64_MAX = (1 << 64) - 1
INT64_MAX = (1 << 63) - 1
SCHEMA = "lumen.replay_hash_chain.v1"
TERMINALS = ("COMPLETED", "EMPTY", "FAILED_PRELAUNCH", "ABORTED")


class Mxfp4StateError(RuntimeError):
    """A training-state failure, never an implicit backend fallback."""

    def __init__(self, reason, detail=""):
        self.reason = reason
        self.disposition = "RUN_FATAL"
        super().__init__(f"{reason}: {detail}")


def uint(value, name, maximum=UINT64_MAX):
    if type(value) is not int or not 0 <= value <= maximum:
        raise Mxfp4StateError("CAPACITY_OR_INDEX_OVERFLOW", name)
    return value


def canonical_bytes(value):
    """Encode explicit semantic records; arbitrary objects/floats are forbidden."""

    def check(item):
        if item is None or type(item) in (str, int, bool):
            return
        if type(item) in (list, tuple):
            for child in item:
                check(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                check(child)
            return
        raise Mxfp4StateError("REPLAY_FRONTIER_MISMATCH", "noncanonical value")

    check(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def completed_events(events):
    """Query caller-recorded completion events without synchronizing or allocating."""
    events = tuple(events)
    if not events or any(not callable(getattr(e, "query", None)) for e in events):
        raise Mxfp4StateError("EVENT_CONTRACT_VIOLATION", "recorded events required")
    return all(e.query() for e in events)


@dataclass(frozen=True, order=True)
class ObligationKey:
    rng_epoch: int
    attempted_step: int
    microbatch: int
    layer: int
    invocation: int
    fc: int
    pass_id: int

    def __post_init__(self):
        for name in (
            "rng_epoch",
            "attempted_step",
            "microbatch",
            "layer",
            "invocation",
        ):
            uint(getattr(self, name), name)
        if type(self.fc) is not int or self.fc not in (1, 2):
            raise Mxfp4StateError("INVALID_LOGICAL_KEY", "FC")
        if type(self.pass_id) is not int or self.pass_id not in (0, 1, 2):
            raise Mxfp4StateError("INVALID_LOGICAL_KEY", "pass")

    def record(self):
        return list(self.__dict__.values())


# Process-local IDs/addresses are deliberately absent from this schema.
SEMANTIC_FIELDS = frozenset(
    {
        "logical_binding_key",
        "config_ids",
        "routing_epoch",
        "weight_epoch",
        "rng_receipts",
        "shape",
        "dtype",
        "layout_ids",
        "requested_mode",
        "selected_mode",
        "executed_mode",
        "kernel_family",
        "precision",
        "selection_events",
        "status",
        "failure_phase",
        "reason",
    }
)


@dataclass
class _Attempt:
    kind: str
    sublaunches: list = field(default_factory=list)
    events: tuple = ()
    finished: bool = False
    status: str = "BOUND"


@dataclass
class _Entry:
    state: str = "DECLARED"
    semantic: dict = field(default_factory=dict)
    attempts: list = field(default_factory=list)


class ObligationLedger:
    """Count graph obligations, not launches; retain pending device work explicitly."""

    def __init__(self, manifest_digest):
        if type(manifest_digest) is not str or len(manifest_digest) != 64:
            raise Mxfp4StateError("REPLAY_FRONTIER_MISMATCH", "manifest digest")
        try:
            bytes.fromhex(manifest_digest)
        except ValueError as exc:
            raise Mxfp4StateError(
                "REPLAY_FRONTIER_MISMATCH", "manifest digest"
            ) from exc
        self.manifest = manifest_digest
        self.chain = hashlib.sha256(
            SCHEMA.encode() + manifest_digest.encode()
        ).hexdigest()
        self.epoch = 0
        self.prefix_leaf_count = 0
        self._counts = {s: 0 for s in TERMINALS}
        self._declared = 0
        self._entries = {}
        self._last_key = None
        self._lock = RLock()

    def declare(self, keys):
        keys = tuple(sorted(keys))
        with self._lock:
            if len(set(keys)) != len(keys) or any(
                not isinstance(k, ObligationKey) for k in keys
            ):
                raise Mxfp4StateError("DUPLICATE_LOGICAL_OBLIGATION")
            if any(
                k in self._entries
                or (self._last_key is not None and k <= self._last_key)
                for k in keys
            ):
                raise Mxfp4StateError(
                    "REPLAY_FRONTIER_MISMATCH", "duplicate or old declaration"
                )
            self._entries.update({k: _Entry() for k in keys})
            self._declared += len(keys)

    def _entry(self, key):
        if key not in self._entries:
            raise Mxfp4StateError("UNKNOWN_LOGICAL_OBLIGATION")
        return self._entries[key]

    def bind(self, key, **semantic):
        with self._lock:
            entry = self._entry(key)
            if entry.state != "DECLARED" or set(semantic) != SEMANTIC_FIELDS:
                raise Mxfp4StateError(
                    "LEDGER_BIND_MISMATCH", "state or semantic fields"
                )
            if semantic["logical_binding_key"] != key.record() or any(
                semantic[name] is not None
                for name in ("executed_mode", "kernel_family", "precision")
            ):
                raise Mxfp4StateError("LEDGER_BIND_MISMATCH", "bind is not execution")
            if semantic["requested_mode"] not in (
                "auto",
                "grouped",
                "sequential",
            ) or semantic["selected_mode"] not in ("grouped", "sequential"):
                raise Mxfp4StateError("LEDGER_BIND_MISMATCH", "mode")
            # Copy through canonical serialization so caller mutation cannot rewrite history.
            entry.semantic = json.loads(canonical_bytes(semantic))
            entry.state = "BOUND"

    def begin_attempt(self, key, *, replay=False):
        with self._lock:
            entry = self._entry(key)
            if type(replay) is not bool:
                raise Mxfp4StateError("INVALID_LOGICAL_KEY", "replay")
            if replay:
                if (
                    key.pass_id != 0
                    or len(entry.attempts) != 1
                    or entry.state not in ("COMPLETED", "EMPTY")
                ):
                    raise Mxfp4StateError("RECOMPUTE_CONTRACT_VIOLATION")
            elif entry.state != "BOUND" or entry.attempts:
                raise Mxfp4StateError("DUPLICATE_PHYSICAL_ATTEMPT")
            entry.attempts.append(_Attempt("REPLAY" if replay else "ORIGINAL"))
            return len(entry.attempts) - 1

    def enqueue(self, key, attempt, *, sublaunches, events, execution=None):
        with self._lock:
            entry = self._entry(key)
            physical = self._attempt(entry, attempt)
            if physical.status != "BOUND":
                raise Mxfp4StateError("DUPLICATE_ENQUEUE")
            descriptors = []
            for descriptor in sublaunches:
                if set(descriptor) != {"op", "expert", "config_id", "ordinal"}:
                    raise Mxfp4StateError(
                        "REPLAY_FRONTIER_MISMATCH", "sublaunch schema"
                    )
                descriptors.append(json.loads(canonical_bytes(descriptor)))
            if execution is not None:
                if (
                    type(execution) is not tuple
                    or len(execution) != 3
                    or execution[0] != entry.semantic["selected_mode"]
                    or execution[1] not in ("TRITON", "FLYDSL", "ASM")
                    or execution[2] not in ("MXFP4", "BF16")
                ):
                    raise Mxfp4StateError("LEDGER_EXECUTION_MISMATCH")
                if (
                    attempt
                    and tuple(
                        entry.semantic[n]
                        for n in ("executed_mode", "kernel_family", "precision")
                    )
                    != execution
                ):
                    raise Mxfp4StateError(
                        "RECOMPUTE_CONTRACT_VIOLATION", "execution changed"
                    )
            elif any(d["op"] == "gemm" for d in descriptors):
                raise Mxfp4StateError(
                    "LEDGER_EXECUTION_MISMATCH", "GEMM execution missing"
                )
            events = tuple(events)
            if not events:
                raise Mxfp4StateError("EVENT_CONTRACT_VIOLATION")
            if execution is not None:
                entry.semantic.update(
                    zip(("executed_mode", "kernel_family", "precision"), execution)
                )
            physical.events = events
            physical.sublaunches = descriptors
            physical.status = "ENQUEUED"
            if not attempt:
                entry.state = "ENQUEUED"

    @staticmethod
    def _attempt(entry, attempt):
        uint(attempt, "attempt")
        if attempt >= len(entry.attempts):
            raise Mxfp4StateError("UNKNOWN_PHYSICAL_ATTEMPT")
        return entry.attempts[attempt]

    def finish(self, key, attempt, *, empty=False):
        with self._lock:
            entry = self._entry(key)
            physical = self._attempt(entry, attempt)
            if type(empty) is not bool or physical.status != "ENQUEUED":
                raise Mxfp4StateError("LEDGER_TRANSITION_MISMATCH")
            if empty and (
                entry.semantic.get("executed_mode") is not None
                or any(d["op"] == "gemm" for d in physical.sublaunches)
            ):
                raise Mxfp4StateError(
                    "LEDGER_EXECUTION_MISMATCH", "EMPTY cannot claim a GEMM"
                )
            if not empty and entry.semantic.get("executed_mode") is None:
                raise Mxfp4StateError(
                    "LEDGER_EXECUTION_MISMATCH", "completion without execution"
                )
            if not completed_events(physical.events):
                return False
            physical.finished = True
            physical.status = "EMPTY" if empty else "COMPLETED"
            if attempt == 0:
                entry.state = physical.status
            return True

    def attach_submission(
        self, key, *, facts, known_sublaunches, unknown_operation=None
    ):
        """Failure diagnostics stay outside canonical success/replay semantics."""
        with self._lock:
            entry = self._entry(key)
            entry.submission_facts = tuple(dict(item) for item in facts)
            entry.known_sublaunches = tuple(dict(item) for item in known_sublaunches)
            entry.unknown_operation = unknown_operation

    def fail(self, key, *, reason, phase, asynchronous=False, numeric_status=None):
        with self._lock:
            entry = self._entry(key)
            if entry.state in TERMINALS and not any(
                not a.finished for a in entry.attempts
            ):
                raise Mxfp4StateError("LEDGER_TRANSITION_MISMATCH")
            launched = any(a.status == "ENQUEUED" for a in entry.attempts) or bool(
                getattr(entry, "known_sublaunches", ())
            )
            entry.semantic.update(reason=reason, failure_phase=phase)
            if numeric_status is not None:
                entry.numeric_status = uint(
                    numeric_status, "numeric status", (1 << 31) - 1
                )
                entry.semantic["status"] = entry.numeric_status
            if asynchronous:
                # Preserve earlier completed attempts, but never close the failed domain.
                entry.semantic["status"] = "INCOMPLETE_ASYNC"
                entry.state = "INCOMPLETE_ASYNC"
                for attempt in entry.attempts:
                    if not attempt.finished:
                        attempt.status = "INCOMPLETE_ASYNC"
                return
            entry.state = "ABORTED" if launched else "FAILED_PRELAUNCH"
            for attempt in entry.attempts:
                if not attempt.finished:
                    attempt.finished = True
                    attempt.status = entry.state

    def abort_unlaunched(self, key, reason):
        with self._lock:
            entry = self._entry(key)
            if entry.state not in ("DECLARED", "BOUND") or entry.attempts:
                raise Mxfp4StateError("LEDGER_TRANSITION_MISMATCH")
            entry.state = "ABORTED"
            entry.semantic.update(reason=reason, failure_phase="CAUSAL_ABORT")

    def counts(self):
        with self._lock:
            counts = dict(self._counts)
            for entry in self._entries.values():
                if entry.state in TERMINALS:
                    counts[entry.state] += 1
            counts["declared"] = self._declared
            counts["open"] = self._declared - sum(counts[s] for s in TERMINALS)
            counts["pending_physical"] = sum(
                not a.finished for e in self._entries.values() for a in e.attempts
            )
            return counts

    def checkpoint(self, *, last_committed_step):
        """Seal a quiescent segment on both uninterrupted and resumed branches."""
        with self._lock:
            uint(last_committed_step, "step")
            counts = self.counts()
            if (
                counts["open"]
                or counts["pending_physical"]
                or counts["ABORTED"]
                or counts["FAILED_PRELAUNCH"]
            ):
                raise Mxfp4StateError(
                    "REPLAY_FRONTIER_MISMATCH", "not a healthy quiescent boundary"
                )
            for key, entry in sorted(self._entries.items()):
                event = {
                    "key": key.record(),
                    "semantic": entry.semantic,
                    "terminal": entry.state,
                    "attempts": [
                        {
                            "kind": a.kind,
                            "status": a.status,
                            "sublaunches": a.sublaunches,
                        }
                        for a in entry.attempts
                    ],
                }
                self.chain = hashlib.sha256(
                    bytes.fromhex(self.chain) + canonical_bytes(event)
                ).hexdigest()
                self.prefix_leaf_count += 1
                self._last_key = key
            frontier = {
                "schema": SCHEMA,
                "manifest": self.manifest,
                "epoch": self.epoch,
                "chain": self.chain,
                "prefix_leaf_count": self.prefix_leaf_count,
                "counts": {s: counts[s] for s in TERMINALS},
                "declared": self._declared,
                "last_key": self._last_key.record() if self._last_key else None,
                "last_committed_step": last_committed_step,
            }
            parent = digest(frontier)
            child = hashlib.sha256(
                bytes.fromhex(self.chain)
                + canonical_bytes(
                    {"parent_frontier_digest": parent, "segment": self.epoch + 1}
                )
            ).hexdigest()
            record = {
                "frontier": frontier,
                "parent_frontier_digest": parent,
                "child_chain": child,
            }
            self.chain = child
            self.epoch += 1
            self._counts = dict(frontier["counts"])
            self._entries.clear()
            return record

    @classmethod
    def restore(cls, record, *, manifest_digest):
        try:
            frontier = record["frontier"]
            if (
                frontier["schema"] != SCHEMA
                or frontier["manifest"] != manifest_digest
                or digest(frontier) != record["parent_frontier_digest"]
            ):
                raise ValueError("frontier identity")
            child = hashlib.sha256(
                bytes.fromhex(frontier["chain"])
                + canonical_bytes(
                    {
                        "parent_frontier_digest": record["parent_frontier_digest"],
                        "segment": frontier["epoch"] + 1,
                    }
                )
            ).hexdigest()
            if child != record["child_chain"] or set(frontier["counts"]) != set(
                TERMINALS
            ):
                raise ValueError("child/counts")
            for value in frontier["counts"].values():
                uint(value, "count")
            if (
                sum(frontier["counts"].values()) != frontier["declared"]
                or frontier["counts"]["ABORTED"]
                or frontier["counts"]["FAILED_PRELAUNCH"]
            ):
                raise ValueError("terminal conservation")
            obj = cls(manifest_digest)
            obj.chain, obj.epoch = child, uint(frontier["epoch"], "epoch") + 1
            obj.prefix_leaf_count = uint(frontier["prefix_leaf_count"], "leaves")
            obj._counts, obj._declared = dict(frontier["counts"]), uint(
                frontier["declared"], "declared"
            )
            obj._last_key = (
                ObligationKey(*frontier["last_key"]) if frontier["last_key"] else None
            )
            return obj
        except (KeyError, TypeError, ValueError) as exc:
            raise Mxfp4StateError("REPLAY_FRONTIER_MISMATCH", str(exc)) from exc


def aggregate_rank_records(records, expected_ranks):
    """Aggregate caller-collected CPU records; absent/crashed ranks remain incomplete."""
    expected = tuple(sorted(expected_ranks))
    if len(set(expected)) != len(expected) or set(records) != set(expected):
        raise Mxfp4StateError("INCOMPLETE_RANK_LEDGER")
    if not expected:
        raise Mxfp4StateError("INCOMPLETE_RANK_LEDGER")
    for rank in expected:
        uint(rank, "global rank", (1 << 32) - 1)
        counts = records[rank]["counts"]
        if set(counts) != {*TERMINALS, "declared", "open", "pending_physical"}:
            raise Mxfp4StateError("INCOMPLETE_RANK_LEDGER", "count schema")
        for value in counts.values():
            uint(value, "count")
        if counts["declared"] != counts["open"] + sum(counts[s] for s in TERMINALS):
            raise Mxfp4StateError("INCOMPLETE_RANK_LEDGER", "conservation")
    ordered = [{"rank": rank, "record": records[rank]} for rank in expected]
    return {
        "ranks": ordered,
        "digest": digest(ordered),
        "counts": {
            name: sum(records[r]["counts"][name] for r in expected)
            for name in (*TERMINALS, "declared", "open", "pending_physical")
        },
    }
