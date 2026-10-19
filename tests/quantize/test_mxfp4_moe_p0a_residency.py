"""Host ownership regressions only; real CUDA storage/HBM tests remain separate."""

import itertools
import unittest

from lumen.quantize.mxfp4_moe_rng import RoutingMetadataV1
from lumen.quantize.mxfp4_moe_state import (
    BudgetedStorage,
    CacheBudgetMiss,
    MainGradBinding,
    MemoryPlan,
    PlannedInput,
    plan_dense_layer,
    tensor_range,
)
from lumen.utils.mxfp4_moe_ledger import Mxfp4StateError

_ids = itertools.count(1)


class _Impl:
    def __init__(self, nbytes):
        self.identity = next(_ids)
        self.nbytes = nbytes
        self.pointer = self.identity * (1 << 30) if nbytes else 0


class _Wrapper:
    def __init__(self, impl):
        self.impl = impl
        self._cdata = impl.identity

    def nbytes(self):
        return self.impl.nbytes

    def data_ptr(self):
        return self.impl.pointer


class _Tensor:
    def __init__(
        self,
        nbytes,
        dtype="torch.uint8",
        shape=None,
        *,
        impl=None,
        offset=0,
        device="cuda:0"
    ):
        self.impl = impl if impl is not None else _Impl(nbytes)
        self.dtype, self.device, self.offset = dtype, device, offset
        self.width = {"torch.uint8": 1, "torch.bfloat16": 2, "torch.float32": 4}[dtype]
        self.shape = shape if shape is not None else (nbytes // self.width,)

    def untyped_storage(self):
        # Deliberately return fresh wrappers around a stable underlying StorageImpl.
        return _Wrapper(self.impl)

    def element_size(self):
        return self.width

    def stride(self):
        return (self.shape[1], 1) if len(self.shape) == 2 else (1,)

    def storage_offset(self):
        return self.offset

    def is_contiguous(self):
        return True


class _Done:
    def query(self):
        return True


def _inputs():
    parameter = _Tensor(16, "torch.bfloat16", (2, 4))
    owner = _Tensor(32, "torch.float32", (2, 4))
    parameter.main_grad = owner
    binding = MainGradBinding.capture(parameter, owner, window=0, weight_epoch=0)
    planned = (
        PlannedInput(parameter, binding, "weight", tensor_range(parameter)),
        PlannedInput(owner, binding, "main_grad", tensor_range(owner)),
    )
    return parameter, owner, planned


class TestP0aResidency(unittest.TestCase):
    def test_independent_empty_storage_and_temporary_wrappers(self):
        a, b = _Tensor(0), _Tensor(0)
        self.assertNotEqual(tensor_range(a).key, tensor_range(b).key)
        self.assertEqual(tensor_range(a).key, tensor_range(a).key)
        shared = _Tensor(0, impl=a.impl)
        self.assertEqual(tensor_range(a).key, tensor_range(shared).key)
        budget = BudgetedStorage(100, device="cuda:0")
        budget.begin_plan(MemoryPlan(0, 0, 0, 0, 0), free_hbm_bytes=0)
        budget.allocate("x-empty", "reusable_workspace_bytes", 0, lambda: a)
        budget.allocate("y-empty", "reusable_workspace_bytes", 0, lambda: b)
        budget.retain_view("alias", shared)
        self.assertEqual(budget.telemetry()["reusable_workspace_bytes"], 0)
        budget.seal()
        budget.retire(("x-empty", "y-empty", "alias"), events=(_Done(),))
        budget.collect_retired()
        self.assertFalse(budget._names)

    def test_nonempty_aliases_still_rejected_or_explicitly_deduplicated(self):
        tensor = _Tensor(16)
        alias = _Tensor(16, impl=tensor.impl)
        budget = BudgetedStorage(100)
        budget.begin_plan(MemoryPlan(32, 32, 0, 0, 32), free_hbm_bytes=32)
        budget.allocate("one", "persistent_cache_bytes", 16, lambda: tensor)
        budget.retain_view("shared", alias)
        self.assertEqual(budget.telemetry()["persistent_cache_bytes"], 16)
        with self.assertRaises(Mxfp4StateError):
            budget.allocate("duplicate", "persistent_cache_bytes", 16, lambda: alias)

    def test_zero_partial_full_residency_and_total_unchanged(self):
        parameter, grad, inputs = _inputs()
        plan = MemoryPlan(0, 0, 0, 0, 48, inputs)
        for tensors, required, resident in (
            ((), 48, 0),
            ((parameter,), 32, 16),
            ((parameter, grad), 0, 48),
            ((parameter, parameter, grad), 0, 48),
        ):
            with self.subTest(required=required, count=len(tensors)):
                budget = BudgetedStorage(100, device="cuda:0")
                budget.begin_plan(
                    plan, free_hbm_bytes=required, resident_inputs=tensors
                )
                self.assertEqual(
                    budget.telemetry()["already_resident_device_bytes"], resident
                )
                self.assertEqual(
                    budget.telemetry()["incremental_preflight_bytes"], required
                )
                self.assertEqual(budget.telemetry()["device_total_preflight_bytes"], 48)
                self.assertEqual(plan.device_total_preflight_bytes, 48)

    def test_unknown_or_wrong_device_storage_not_credited(self):
        parameter, _grad, inputs = _inputs()
        plan = MemoryPlan(0, 0, 0, 0, 48, inputs)
        for tensor in (
            _Tensor(48),
            _Tensor(16, "torch.bfloat16", (2, 4), impl=parameter.impl, device="cuda:1"),
        ):
            budget = BudgetedStorage(100, device="cuda:0")
            with self.assertRaises(Mxfp4StateError):
                budget.begin_plan(plan, free_hbm_bytes=100, resident_inputs=(tensor,))
            self.assertIsNone(budget._plan)
        with self.assertRaises(Mxfp4StateError):
            MemoryPlan(0, 0, 0, 0, 47, inputs)

    def test_free_shortage_does_not_change_reservations(self):
        parameter, _grad, inputs = _inputs()
        budget = BudgetedStorage(100, device="cuda:0")
        plan = MemoryPlan(0, 0, 0, 0, 48, inputs)
        before = budget.telemetry()
        with self.assertRaises(CacheBudgetMiss):
            budget.begin_plan(plan, free_hbm_bytes=31, resident_inputs=(parameter,))
        self.assertIsNone(budget._plan)
        self.assertEqual(before, budget.telemetry())

    def test_owned_hot_cache_is_credited_once_not_future_reclaimed(self):
        budget = BudgetedStorage(100, device="cuda:0")
        plan = MemoryPlan(60, 60, 20, 0, 80)
        budget.begin_plan(plan, free_hbm_bytes=80)
        weight = budget.allocate(
            "weight", "persistent_cache_bytes", 60, lambda: _Tensor(60)
        )
        budget.retain_view("transpose", _Tensor(60, impl=weight.impl))
        budget.allocate("ZG", "reusable_workspace_bytes", 20, lambda: _Tensor(20))
        budget.seal()
        with self.assertRaises(Mxfp4StateError):
            budget.begin_plan(plan, free_hbm_bytes=0)
        budget.begin_plan(plan, free_hbm_bytes=0, planned_owned_names=budget.live_names)
        self.assertEqual(budget.telemetry()["already_resident_device_bytes"], 80)
        self.assertEqual(budget.telemetry()["device_total_preflight_bytes"], 80)

    def test_other_resident_ranges_union_credit_and_future_demand(self):
        route = RoutingMetadataV1((0,) * 32, 0, (0,) * 8)
        other = _Tensor(64)
        alias = _Tensor(64, impl=other.impl)
        plan, _ = plan_dense_layer(
            route, other_device_bytes=96, other_resident_tensors=(other, alias)
        )
        budget = BudgetedStorage(1 << 40, device="cuda:0")
        required = plan.device_total_preflight_bytes - 64
        budget.begin_plan(plan, free_hbm_bytes=required, resident_inputs=(other, alias))
        self.assertEqual(budget.telemetry()["already_resident_device_bytes"], 64)
        self.assertEqual(budget.telemetry()["incremental_preflight_bytes"], required)
        with self.assertRaises(CacheBudgetMiss):
            BudgetedStorage(1 << 40).begin_plan(
                plan, free_hbm_bytes=required - 1, resident_inputs=(other,)
            )
        with self.assertRaises(Mxfp4StateError):
            plan_dense_layer(
                route, other_device_bytes=63, other_resident_tensors=(other,)
            )

    def test_other_residency_mutation_and_unregistered_credit_rejected(self):
        other = _Tensor(64)
        plan = MemoryPlan(
            0, 0, 0, 0, 64, (PlannedInput(other, None, "other", tensor_range(other)),)
        )
        other.offset = 1
        with self.assertRaises(Mxfp4StateError):
            BudgetedStorage(100).begin_plan(
                plan, free_hbm_bytes=0, resident_inputs=(other,)
            )
        unknown = _Tensor(64)
        with self.assertRaises(Mxfp4StateError):
            BudgetedStorage(100).begin_plan(
                MemoryPlan(0, 0, 0, 0, 64), free_hbm_bytes=0, resident_inputs=(unknown,)
            )

    def test_mutated_planned_binding_rejected_before_credit(self):
        parameter, grad, inputs = _inputs()
        plan = MemoryPlan(0, 0, 0, 0, 48, inputs)
        parameter.main_grad = _Tensor(32, "torch.float32", (2, 4))
        budget = BudgetedStorage(100)
        with self.assertRaises(Mxfp4StateError):
            budget.begin_plan(plan, free_hbm_bytes=0, resident_inputs=(parameter, grad))
        self.assertIsNone(budget._plan)


if __name__ == "__main__":
    unittest.main()
