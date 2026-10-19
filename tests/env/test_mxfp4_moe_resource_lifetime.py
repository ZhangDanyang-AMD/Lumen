"""Real CUDA event/pinned lease behavior; authorized runtime only, unrun."""

import pytest
import torch

from lumen.utils.mxfp4_moe_ledger import Mxfp4StateError
from lumen.utils.mxfp4_moe_resources import PreparedResources

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA resources required"
)


def test_prepared_events_and_pinned_words_exist_before_acquire():
    pool = PreparedResources(fc_leases=2, experts=32, pinned_limit_bytes=16)
    assert pool.telemetry["event_count"] == 200
    slots = tuple(pool.slots)
    identities = tuple(tuple(id(event) for event in slot["events"]) for slot in slots)
    assert all(
        slot["mirror"].is_pinned() and slot["zero"].is_pinned() for slot in slots
    )
    assert all(event.query() for slot in slots for event in slot["events"])
    first = pool.acquire(("scope", 1))
    second = pool.acquire(("scope", 2))
    assert first is slots[0] and second is slots[1]
    assert identities == tuple(
        tuple(id(event) for event in slot["events"]) for slot in pool.slots
    )
    with pytest.raises(Mxfp4StateError, match="DUPLICATE_RESOURCE_LEASE"):
        pool.acquire(("scope", 1))
    with pytest.raises(Mxfp4StateError, match="LIVE_LEASE_CAPACITY_VIOLATION"):
        pool.acquire(("other", 1))
    with pytest.raises(Mxfp4StateError, match="RESOURCE_CONSUMERS_NOT_READY"):
        pool.release(("scope", 1), ())
    assert first["owner"] == ("scope", 1)
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    event = first["events"][0]
    value = torch.empty((1024,), device="cuda", dtype=torch.int32)
    with torch.cuda.stream(producer):
        value.fill_(7)
        event.record()
    with torch.cuda.stream(consumer):
        consumer.wait_event(event)
        observed = value + 1
        completion = first["events"][1]
        completion.record()
    completion.synchronize()
    assert torch.equal(observed, torch.full_like(observed, 8))
    pool.release(("scope", 1), (completion,))
    assert pool.acquire(("new-scope", 1)) is first
    assert first["mirror"] is slots[0]["mirror"]


def test_pinned_budget_rejects_before_event_preparation():
    with pytest.raises(Mxfp4StateError, match="HOST_PLAN_BREACH"):
        PreparedResources(fc_leases=2, experts=64, pinned_limit_bytes=15)
