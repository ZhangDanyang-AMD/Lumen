"""Optional healthy-runtime assertions for the existing exact-route paired entry.

No new Tensor/Node hooks are installed. Extra snapshots/synchronization belong to
this test observer only, not the product path or performance measurements.
"""

import torch


class ResourceObserver:
    def __init__(self):
        self.executions = []
        self.weights_after_w = {}
        self.backward_count = 0

    @staticmethod
    def identities(runtime):
        return tuple(
            (
                tuple(id(event) for event in slot["events"]),
                id(slot["mirror"]),
                id(slot["zero"]),
            )
            for slot in runtime.resources.slots
        )

    def __call__(self, phase, runtime, execution=None, emissions=None):
        if phase == "prepared":
            self.prepared = self.identities(runtime)
            self.original_nodes = tuple(runtime.ddp.grad_accs)
            assert runtime.installation_receipt["original_core_hooks_preserved"]
            assert all(slot["owner"] is None for slot in runtime.resources.slots)
            assert all(
                slot["mirror"].is_pinned() and slot["zero"].is_pinned()
                for slot in runtime.resources.slots
            )
            assert all(
                event.query()
                for slot in runtime.resources.slots
                for event in slot["events"]
            )
            return
        assert self.prepared == self.identities(
            runtime
        ), "event/pinned resource identities changed"
        assert (
            tuple(runtime.ddp.grad_accs) == self.original_nodes
        ), "original Core accumulators changed"
        if phase == "materialized":
            self.executions.append(execution)
            for lease in execution.zero_leases:
                lease.validate()
                parameter = lease.binding.parameter
                assert runtime.handoff.nodes[parameter] in self.original_nodes
                assert parameter.grad is None
            return
        if phase == "backward":
            self.backward_count += 1
            assert len(emissions) == len(execution.zero_leases)
            for lease, emission in zip(execution.zero_leases, emissions):
                assert emission is lease.emission, "Z was cloned/materialized"
                assert emission.dtype == torch.bfloat16
                assert tuple(emission.shape) == lease.binding.shape
                assert emission.stride() == lease.sink.stride()
                assert emission.data_ptr() != lease.sink.data_ptr()
                assert lease.armed and not lease.consumed
                self.weights_after_w[id(lease)] = (
                    lease.binding.main_grad.detach().clone()
                )
            return
        if phase == "consumed":
            assert len(self.executions) == self.backward_count == 4
            assert not runtime.handoff.pending
            assert len(runtime.lifecycle.active) == 2
            for observed_execution in self.executions:
                for lease in observed_execution.zero_leases:
                    parameter = lease.binding.parameter
                    assert lease.armed and lease.consumed
                    assert parameter.grad is None and parameter.grad_added_to_main_grad
                    # Healthy test-only observation, never an error cleanup/readback.
                    lease.consumed_event.synchronize()
                    assert lease.consumed_event.query()
                    assert (
                        torch.count_nonzero(lease.emission.view(torch.int16)).item()
                        == 0
                    )
                    assert torch.count_nonzero(lease.sink.view(torch.int16)).item() == 0
                    assert torch.equal(
                        lease.binding.main_grad, self.weights_after_w[id(lease)]
                    ), "Core hook duplicated/changed W"
            return
        assert phase == "retired"
        assert not runtime.lifecycle.active
        assert not runtime.handoff.pending and not runtime.handoff.leases
        assert all(slot["owner"] is None for slot in runtime.resources.slots)
        assert runtime.ledger.counts()["open"] == 0
        assert runtime.ledger.counts()["pending_physical"] == 0
