"""Bounded CPU failure records with explicit primary and secondary outcomes."""

import json
import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FailureSnapshot:
    phase: str
    asynchronous: bool
    entries: tuple


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    written: bool
    stage: str
    secondary: Exception | None = None
    context: dict | None = None

    @property
    def secondary_type(self):
        return type(self.secondary).__name__ if self.secondary is not None else None


class FailureReporter:
    """Own fixed payload construction and one write/flush on the existing stream."""

    def __init__(self, stream, rank, physical_run_id):
        self.stream = stream
        self.rank = rank
        self.physical_run_id = physical_run_id
        self.last_result = DiagnosticResult(False, "not_attempted")

    @staticmethod
    def _context(primary, snapshot):
        if snapshot is None:
            return None
        entries = []
        limit = 321
        for logical_key, entry in snapshot.entries[:3]:
            facts = getattr(entry, "submission_facts", ())
            known = getattr(entry, "known_sublaunches", ())
            entries.append(
                {
                    "logical_key": logical_key,
                    "physical_attempt": (
                        len(entry.attempts) - 1 if entry.attempts else None
                    ),
                    "state": entry.state,
                    "physical_attempts": [
                        {
                            "index": index,
                            "kind": attempt.kind,
                            "state": attempt.status,
                            "finished": attempt.finished,
                        }
                        for index, attempt in enumerate(entry.attempts[:2])
                    ],
                    "known_prefix": [dict(item) for item in known[:limit]],
                    "submission_facts": [dict(item) for item in facts[:limit]],
                    "unknown_operation": getattr(entry, "unknown_operation", None),
                    "numeric_snapshot": getattr(entry, "numeric_status", None),
                    "facts_truncated": len(facts) > limit
                    or len(known) > limit
                    or len(entry.attempts) > 2,
                }
            )
        return {
            "schema": "lumen.mxfp4.failure_submission.v1",
            "phase": snapshot.phase,
            "asynchronous_or_unknown": snapshot.asynchronous,
            "first_error": {
                "type": type(primary).__module__ + "." + type(primary).__qualname__,
                "message": str(primary)[:1024],
            },
            "obligations": entries,
            "obligations_truncated": len(snapshot.entries) > 3,
        }

    def report(self, event, *, primary, snapshot=None):
        stage, context = "payload", None
        self.last_result = DiagnosticResult(False, stage)
        try:
            if event not in ("async_abort", "submission_failure"):
                raise ValueError("unsupported failure event")
            context = self._context(primary, snapshot)
            fields = {
                "event": event,
                "rank": self.rank,
                "physical_run_id": self.physical_run_id,
                "monotonic_ns": time.monotonic_ns(),
            }
            if event == "async_abort":
                fields.update(reason=str(primary)[:1024], ledger_may_be_incomplete=True)
            if context is not None:
                fields["failure"] = context
            stage = "format"
            text = json.dumps(fields, sort_keys=True) + "\n"
            stage = "write"
            self.stream.write(text)
            stage = "flush"
            self.stream.flush()
        except (OSError, UnicodeError) as secondary:
            self.last_result = DiagnosticResult(False, stage, secondary, context)
            if stage not in ("write", "flush"):
                raise primary from secondary
            return self.last_result
        except ValueError as secondary:
            self.last_result = DiagnosticResult(False, stage, secondary, context)
            if stage in ("write", "flush") and self.stream.closed:
                return self.last_result
            raise primary from secondary
        except Exception as secondary:
            self.last_result = DiagnosticResult(False, stage, secondary, context)
            raise primary from secondary
        self.last_result = DiagnosticResult(True, "complete", context=context)
        return self.last_result
