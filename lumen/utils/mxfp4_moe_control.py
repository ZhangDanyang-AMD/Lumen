# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0
"""PP1 MXFP4 healthy control boundaries on the existing torchrun agent store."""

import json
import os
import signal
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from lumen.utils.mxfp4_moe_diagnostics import FailureReporter
from lumen.utils.mxfp4_moe_ledger import Mxfp4StateError, canonical_bytes, digest, uint

PHASES = (
    "STARTUP",
    "ROUTE_READY_PREBIND",
    "MATERIALIZATION_HEALTH",
    "FC_PRELAUNCH",
    "FC_POSTF_STATUS",
    "BACKWARD_PREATTACH",
    "BACKWARD_POST_STATUS",
    "STEP_BOUNDARY",
)


@dataclass(frozen=True)
class ControlSequence:
    attempted_step: int
    microbatch: int
    layer: int
    fc: int
    phase: str
    physical_recompute_attempt: int

    def __post_init__(self):
        for name in (
            "attempted_step",
            "microbatch",
            "layer",
            "fc",
            "physical_recompute_attempt",
        ):
            uint(getattr(self, name), name)
        if (
            self.fc not in (0, 1, 2)
            or self.phase not in PHASES
            or self.physical_recompute_attempt not in (0, 1)
        ):
            raise Mxfp4StateError("CONTROL_SEQUENCE_MISMATCH")

    def record(self):
        return dict(self.__dict__)


def dependency_component(rank, memberships):
    """Compute the PP-stage EP plus expert-DP transitive dependency closure."""
    component = {rank}
    for member, groups in memberships.items():
        if len(groups) != 2 or any(member not in group for group in groups):
            raise Mxfp4StateError("CONTROL_MEMBERSHIP_MISMATCH")
    while True:
        expanded = set(component)
        for groups in memberships.values():
            for group in groups:
                if component.intersection(group):
                    expanded.update(group)
        if expanded == component:
            return tuple(sorted(component))
        component = expanded


class Mxfp4Control:
    """No new server, process group, polling thread or failure-time GPU collective."""

    def __init__(self, ddp, *, physical_run_id, journal_path):
        import torch.distributed as dist
        from torch.distributed.elastic.timer import FileTimerClient

        if not physical_run_id or not physical_run_id.isascii():
            raise Mxfp4StateError("CONTROL_NAMESPACE_MISMATCH")
        if os.environ.get("TORCHELASTIC_USE_AGENT_STORE", "").lower() != "true":
            raise Mxfp4StateError("AGENT_STORE_REQUIRED")
        timer_file = os.environ.get("TORCHELASTIC_TIMER_FILE")
        if os.environ.get("TORCHELASTIC_ENABLE_FILE_TIMER") != "1" or not timer_file:
            raise Mxfp4StateError("FILE_TIMER_REQUIRED")
        if dist.get_world_size() != 8 or ddp.pp_group.size() != 1:
            raise Mxfp4StateError("UNSUPPORTED_CONTROL_TOPOLOGY")
        self.rank = dist.get_rank()
        self.physical_run_id = physical_run_id
        # An independent client timeout never changes the PG's shared store options.
        client = dist.TCPStore(
            os.environ["MASTER_ADDR"],
            int(os.environ["MASTER_PORT"]),
            world_size=None,
            is_master=False,
            timeout=timedelta(seconds=30),
            wait_for_workers=False,
        )
        self.store = dist.PrefixStore("lumen-mxfp4/" + physical_run_id + "/", client)
        self.timer = FileTimerClient(timer_file, signal=signal.SIGKILL)
        self._seen = set()
        self.failed = False
        with ExitStack() as journal_lifetime:
            self._journal = journal_lifetime.enter_context(
                Path(journal_path).open("x", encoding="utf-8")
            )
            self._journal_lifetime = journal_lifetime.pop_all()
        self._failure_reporter = FailureReporter(
            self._journal, self.rank, self.physical_run_id
        )
        local = (
            tuple(dist.get_process_group_ranks(ddp.ep_group)),
            tuple(dist.get_process_group_ranks(ddp.expt_dp_group)),
        )
        key = "membership/" + str(self.rank)
        self.store.set(key, canonical_bytes(local))
        keys = ["membership/" + str(r) for r in range(8)]
        self.store.wait(keys, timedelta(seconds=30))
        memberships = {
            r: json.loads(self.store.get(k).decode()) for r, k in enumerate(keys)
        }
        if any(
            any(
                any(type(r) is not int or r not in range(8) for r in group)
                for group in groups
            )
            for groups in memberships.values()
        ):
            raise Mxfp4StateError("CONTROL_MEMBERSHIP_MISMATCH")
        self.component = dependency_component(self.rank, memberships)
        if self.component != tuple(range(8)):
            raise Mxfp4StateError(
                "UNSUPPORTED_CONTROL_TOPOLOGY",
                "signed PP1 component must close across EP/expert-DP",
            )
        self.record(
            "control_startup",
            {
                "rank": self.rank,
                "memberships": memberships,
                "component": self.component,
                "pid": os.getpid(),
                "agent_store_client_only": True,
                "control_timeout_seconds": 30,
            },
        )

    def record(self, event, fields):
        self._journal.write(
            json.dumps(
                {
                    "event": event,
                    "rank": self.rank,
                    "physical_run_id": self.physical_run_id,
                    "monotonic_ns": time.monotonic_ns(),
                    **fields,
                },
                sort_keys=True,
            )
            + "\n"
        )
        self._journal.flush()

    @contextmanager
    def active_scope(self, name):
        from torch.distributed.elastic import timer

        if self.failed:
            raise Mxfp4StateError("ABORTED_CONTROL_SCOPE")
        with timer.expires(120, scope="mxfp4/" + name, client=self.timer):
            yield

    def vote(self, sequence, *, code=0, reason="OK"):
        """Vote only after a healthy CPU/recorded-completion boundary is reached."""
        if self.failed or not isinstance(sequence, ControlSequence):
            raise Mxfp4StateError("ABORTED_CONTROL_SCOPE")
        uint(code, "status code", (1 << 31) - 1)
        key = digest(sequence.record())
        if key in self._seen:
            raise Mxfp4StateError("DUPLICATE_CONTROL_VOTE")
        self._seen.add(key)
        message = {
            "sequence": sequence.record(),
            "rank": self.rank,
            "code": code,
            "reason": reason,
        }
        self.record("vote_arrival", message)
        try:
            self.store.set(key + "/" + str(self.rank), canonical_bytes(message))
            keys = [key + "/" + str(r) for r in self.component]
            self.store.wait(keys, timedelta(seconds=30))
            records = [json.loads(self.store.get(k).decode()) for k in keys]
        except Exception:
            self.failed = True
            self.record("control_store_failure", {"sequence": sequence.record()})
            raise
        if any(
            r["sequence"] != sequence.record() or r["rank"] != rank
            for rank, r in zip(self.component, records)
        ):
            self.failed = True
            raise Mxfp4StateError("CONTROL_SEQUENCE_MISMATCH")
        failures = sorted(
            (r for r in records if r["code"]), key=lambda r: (r["rank"], r["code"])
        )
        self.record("vote_result", {"sequence": sequence.record(), "statuses": records})
        if failures:
            self.failed = True
            raise Mxfp4StateError("COMPONENT_HEALTH_FAILURE", str(failures[0]))
        return {r["rank"]: r["code"] for r in records}

    def metadata_check(self, sequence, function):
        """Coordinate validation errors before any selected device work begins."""
        result, error = None, None
        try:
            result = function()
        except (Mxfp4StateError, ValueError, TypeError) as exc:
            error = exc
        self.vote(
            sequence,
            code=0 if error is None else 1,
            reason="OK" if error is None else str(error),
        )
        return result

    def failure_record(self, event, *, primary, snapshot=None):
        """Retain a visible result; unexpected diagnostic errors chain to primary."""
        self.failed = True
        try:
            result = self._failure_reporter.report(
                event, primary=primary, snapshot=snapshot
            )
        finally:
            result = self._failure_reporter.last_result
            self.failure_diagnostic_result = result
            self.failure_output_error = result.secondary_type
            self.failure_context = result.context
        return result.written

    def async_abort(self, reason, *, snapshot=None):
        """Report to the existing stream; callers still terminate on their first error."""
        return self.failure_record("async_abort", primary=reason, snapshot=snapshot)

    def close(self):
        self._journal_lifetime.close()
