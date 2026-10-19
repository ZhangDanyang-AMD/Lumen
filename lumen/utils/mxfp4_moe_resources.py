# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0
"""Pre-recorded event and pinned-word leases for the bounded isolated fixture."""

from lumen.utils.mxfp4_moe_ledger import Mxfp4StateError


class PreparedResources:
    def __init__(
        self,
        *,
        fc_leases,
        experts,
        pinned_limit_bytes,
        event_factory=None,
        current_stream=None,
        pinned_word_factory=None
    ):
        required = fc_leases * 8
        if required > pinned_limit_bytes:
            raise Mxfp4StateError("HOST_PLAN_BREACH")
        if (
            event_factory is None
            or current_stream is None
            or pinned_word_factory is None
        ):
            import torch

            event_factory = event_factory or torch.cuda.Event
            current_stream = current_stream or torch.cuda.current_stream
            if pinned_word_factory is None:

                def allocate_pinned_word(zero):
                    factory = torch.zeros if zero else torch.empty
                    return factory(
                        (1,), dtype=torch.int32, device="cpu", pin_memory=True
                    )

                pinned_word_factory = allocate_pinned_word
        self.slots = []
        for _ in range(fc_leases):
            events = tuple(event_factory() for _ in range(4 + 3 * experts))
            for event in events:
                event.record(current_stream())
            events[-1].synchronize()
            mirror = pinned_word_factory(False)
            zero = pinned_word_factory(True)
            self.slots.append(
                {"events": events, "mirror": mirror, "zero": zero, "owner": None}
            )
        self.telemetry = {
            "pinned_host_bytes": required,
            "event_count": fc_leases * (4 + 3 * experts),
            "events_first_recorded_in_prepare": True,
            "event_runtime_internal_bytes": "unknown-runtime-measurement-required",
        }

    def acquire(self, key):
        if any(slot["owner"] == key for slot in self.slots):
            raise Mxfp4StateError("DUPLICATE_RESOURCE_LEASE")
        for slot in self.slots:
            if slot["owner"] is None:
                slot["owner"] = key
                return slot
        raise Mxfp4StateError("LIVE_LEASE_CAPACITY_VIOLATION")

    def release(self, key, events):
        if not events or not all(event.query() for event in events):
            raise Mxfp4StateError("RESOURCE_CONSUMERS_NOT_READY")
        slot = next((slot for slot in self.slots if slot["owner"] == key), None)
        if slot is None:
            raise Mxfp4StateError("UNKNOWN_RESOURCE_LEASE")
        slot["owner"] = None
