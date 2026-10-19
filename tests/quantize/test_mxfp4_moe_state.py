"""Host lifecycle models; real Tensor/autograd/DDP/event qualification is separate."""

import unittest

from lumen.quantize.mxfp4_moe_rng import (
    PhiloxDrawAllocator,
    RoutingMetadataV1,
    ScopeKey,
)
from lumen.quantize.mxfp4_moe_state import (
    BudgetedStorage,
    CacheBudgetMiss,
    CacheKey,
    MainGradBinding,
    MemoryPlan,
    RecomputeTape,
    WeightEpochState,
    checkpoint_sidecar,
    plan_dense_layer,
    restore_sidecar,
    source_weight_bytes,
    validate_disjoint_main_grads,
)
from lumen.utils.mxfp4_moe_ledger import Mxfp4StateError, ObligationLedger


class _Event:
    def __init__(self, ready=True):
        self.ready = ready

    def query(self):
        return self.ready


class _Storage:
    def __init__(self, nbytes):
        self.size = nbytes

    def nbytes(self):
        return self.size

    def data_ptr(self):
        return id(self)


class _Tensor:
    """Metadata-only test double; no fake torch module and no numerical claim."""

    def __init__(self, nbytes, dtype="torch.uint8", shape=None, storage=None, offset=0):
        self.dtype = dtype
        self.device = "cuda:0"
        self._size = {"torch.uint8": 1, "torch.bfloat16": 2, "torch.float32": 4}[dtype]
        self._storage = storage if storage is not None else _Storage(nbytes)
        self.shape = shape if shape is not None else (nbytes // self._size,)
        self.offset = offset

    def untyped_storage(self):
        return self._storage

    def element_size(self):
        return self._size

    def stride(self):
        return (self.shape[1], 1) if len(self.shape) == 2 else (1,)

    def storage_offset(self):
        return self.offset

    def is_contiguous(self):
        return True


def _cache_refs():
    scale = _Tensor(128 * 128, shape=(128, 128))
    scale_t = _Tensor(128 * 128, shape=(128, 128), storage=scale.untyped_storage())
    scale_t.stride = lambda: (1, 128)
    return {
        "w_fp4": _Tensor(4096 * 2048, shape=(4096, 2048)),
        "wt_fp4": _Tensor(4096 * 2048, shape=(4096, 2048)),
        "w_scales": scale,
        "wt_scales": scale_t,
    }


def _parameter():
    return _Tensor(2 * 4096 * 4096, "torch.bfloat16", shape=(4096, 4096))


def _key(parameter, state, prepared="plain_dense_v1"):
    return CacheKey(
        id(parameter),
        state.committed_weight_epoch,
        state.cache_generation,
        "lumen_a4w4_direct_training_v1",
        "lumen_fp4_e2m1_e8m0_scale_block_32x32_s1024_v1",
        "none",
        "mxfp4_weight_tile32x32_v1",
        prepared,
        "cuda:0",
        "gfx950",
        256,
        (4096, 4096),
        32,
    )


class TestMxfp4MoeState(unittest.TestCase):
    def test_planner_real_fc_shapes_empty_and_multiple_live_leases(self):
        for experts, gib in ((32, 3), (64, 6)):
            route = RoutingMetadataV1((0,) * experts, 0, (0,) * (256 // experts))
            plan, details = plan_dense_layer(
                route, live_leases=2, other_device_bytes=1024
            )
            self.assertEqual(details["ZG_per_lease"], gib * 1024**3)
            self.assertEqual(details["ZG_all_live_leases"], 2 * gib * 1024**3)
            self.assertEqual(details["compact_operands_per_lease"], 0)
            self.assertEqual(plan.reusable_workspace_bytes, 2 * (gib * 1024**3 + 8) + 8)
            self.assertEqual(plan.transient_launch_bytes, 0)
            self.assertEqual(
                plan.device_total_preflight_bytes,
                details["B_ref"]
                + details["DDP_main_grad"]
                + 1024
                + plan.peak_cache_build_bytes
                + plan.reusable_workspace_bytes,
            )
            BudgetedStorage(details["B_ref"]).begin_plan(
                plan, free_hbm_bytes=plan.device_total_preflight_bytes
            )
            nonempty = RoutingMetadataV1(
                (33,) + (0,) * (experts - 1), 33, (100,) * (256 // experts)
            )
            nonempty_plan, nonempty_details = plan_dense_layer(nonempty)
            self.assertEqual(
                nonempty_plan.persistent_cache_bytes, plan.persistent_cache_bytes
            )
            self.assertEqual(
                nonempty_details["compact_operands_per_lease"],
                (33 + 64) * (4096 + 4096 + 2048 + 4096) * 17 // 32,
            )

    def test_parameter_storage_rebind_and_cache_source_alias_rejected(self):
        parameter = _parameter()
        owner = _Tensor(4 * 4096 * 4096, "torch.float32", shape=(4096, 4096))
        parameter.main_grad = owner
        binding = MainGradBinding.capture(parameter, owner, window=0, weight_epoch=0)
        parameter._storage = _Storage(parameter.untyped_storage().nbytes())
        with self.assertRaises(Mxfp4StateError):
            binding.validate(window=0, weight_epoch=0)
        state = WeightEpochState()
        state.open_window(zero_events=(_Event(),))
        refs = _cache_refs()
        refs["w_fp4"] = _Tensor(
            4096 * 2048, shape=(4096, 2048), storage=parameter.untyped_storage()
        )
        with self.assertRaises(Mxfp4StateError):
            state.publish_cache(
                _key(parameter, state),
                parameter,
                refs,
                component_ranks=(0, 1),
                votes={0: 0, 1: 0},
                completion_events=(_Event(),),
            )

    def test_compact_sidecar_and_no_rotation_on_unsafe_snapshot(self):
        weights = WeightEpochState()
        rng = PhiloxDrawAllocator(17, 0)
        ledger = ObligationLedger("f" * 64)
        saved = checkpoint_sidecar(weights, rng, ledger, last_committed_step=0)
        w2, r2, l2 = restore_sidecar(saved, execution_manifest="f" * 64)
        self.assertEqual(w2.checkpoint(), weights.checkpoint())
        self.assertEqual(r2.checkpoint(), rng.checkpoint())
        self.assertEqual((l2.chain, l2.epoch), (ledger.chain, ledger.epoch))
        self.assertNotIn("cache", saved["weights"])
        self.assertEqual(w2.cache_generation, 1)
        previous = ledger.chain
        weights.open_window(zero_events=(_Event(),))
        with self.assertRaises(Mxfp4StateError):
            checkpoint_sidecar(weights, rng, ledger, last_committed_step=0)
        self.assertEqual(ledger.chain, previous)
        with self.assertRaises(Mxfp4StateError):
            restore_sidecar(saved, execution_manifest="e" * 64)

    def test_live_alias_prevents_retirement_and_no_silent_budget_raise(self):
        with self.assertRaises(Mxfp4StateError):
            BudgetedStorage(
                100,
                persistent_numerator=101,
                persistent_denominator=100,
                override_provenance="test",
            )
        with self.assertRaises(Mxfp4StateError):
            BudgetedStorage(100, persistent_numerator=82, persistent_denominator=100)
        budget = BudgetedStorage(100)
        budget.begin_plan(MemoryPlan(70, 70, 0, 0, 70), free_hbm_bytes=100)
        value = budget.allocate(
            "base", "persistent_cache_bytes", 70, lambda: _Tensor(70)
        )
        budget.retain_view("alias", _Tensor(70, storage=value.untyped_storage()))
        budget.seal()
        with self.assertRaises(Mxfp4StateError):
            budget.retire(("base",), events=(_Event(),))
        self.assertEqual(budget.telemetry()["persistent_cache_bytes"], 70)
        budget.retire(("base", "alias"), events=(_Event(),))
        self.assertEqual(budget.collect_retired()["persistent_cache_bytes"], 0)

    def test_cache_cannot_publish_separately_requantized_transpose_scales(self):
        state = WeightEpochState()
        parameter = _parameter()
        state.open_window(zero_events=(_Event(),))
        refs = _cache_refs()
        refs["wt_scales"] = _Tensor(16384, shape=(128, 128))
        with self.assertRaises(Mxfp4StateError):
            state.publish_cache(
                _key(parameter, state),
                parameter,
                refs,
                component_ranks=(0, 1),
                votes={0: 0, 1: 0},
                completion_events=(_Event(),),
            )

    def test_epoch_only_after_full_sync_skip_keeps_epoch(self):
        state = WeightEpochState()
        event = _Event(False)
        window = state.open_window(zero_events=(_Event(),))
        self.assertEqual(window, 0)
        for microbatch in range(3):
            self.assertEqual(state.acquire(microbatch, (object(),)), (0, 0, 0))
            self.assertFalse(state.release(microbatch, events=(event,)))
            event.ready = True
            self.assertTrue(state.release(microbatch, events=(event,)))
            event.ready = False
        state.close_window(reducer_events=(_Event(),))
        state.optimizer_result(success=True, full_bf16_sync_events=(event,))
        self.assertFalse(state.commit_ready_update())
        self.assertEqual(state.committed_weight_epoch, 0)
        with self.assertRaises(Mxfp4StateError):
            state.open_window(zero_events=(_Event(),))
        event.ready = True
        self.assertTrue(state.commit_ready_update())
        self.assertEqual((state.committed_weight_epoch, state.cache_generation), (1, 1))
        self.assertEqual(state.open_window(zero_events=(_Event(),)), 1)
        state.close_window(reducer_events=(_Event(),))
        state.optimizer_result(success=False)
        self.assertEqual((state.committed_weight_epoch, state.cache_generation), (1, 1))
        self.assertEqual(state.checkpoint()["attempted_step"], 2)

    def test_unsafe_external_dirty_load_and_in_place_invalidation(self):
        state = WeightEpochState()
        checkpoint = state.checkpoint()
        parameter = _parameter()
        state.open_window(zero_events=(_Event(),))
        key = _key(parameter, state)
        refs = _cache_refs()
        state.publish_cache(
            key,
            parameter,
            refs,
            component_ranks=(0, 1),
            votes={0: 0, 1: 0},
            completion_events=(_Event(),),
        )
        self.assertEqual(state.cache(key, parameter), refs)
        for call in (
            lambda: state.mark_mxfp4_weights_dirty(full_bf16_sync_events=(_Event(),)),
            lambda: state.load_checkpoint(checkpoint),
        ):
            with self.assertRaises(Mxfp4StateError):
                call()
        state.close_window(reducer_events=(_Event(),))
        state.optimizer_result(success=False)
        state.load_checkpoint(checkpoint)
        self.assertEqual(state.committed_weight_epoch, 0)
        self.assertEqual(state.cache_generation, 1)
        with self.assertRaises(Mxfp4StateError):
            state.cache(key, parameter)
        state.mark_mxfp4_weights_dirty(full_bf16_sync_events=(_Event(),))
        self.assertEqual(state.committed_weight_epoch, 1)
        state.invalidate_device_dtype_move(completion_events=(_Event(),))
        self.assertEqual((state.committed_weight_epoch, state.cache_generation), (1, 3))

    def test_publish_requires_all_component_votes_and_events(self):
        state = WeightEpochState()
        parameter = _parameter()
        state.open_window(zero_events=(_Event(),))
        for votes, event in (
            ({0: 0}, _Event()),
            ({0: 0, 1: 1}, _Event()),
            ({0: 0, 1: 0}, _Event(False)),
        ):
            with self.assertRaises(Mxfp4StateError):
                state.publish_cache(
                    _key(parameter, state),
                    parameter,
                    _cache_refs(),
                    component_ranks=(0, 1),
                    votes=votes,
                    completion_events=(event,),
                )
        state.publish_cache(
            _key(parameter, state),
            parameter,
            _cache_refs(),
            component_ranks=(0, 1),
            votes={0: 0, 1: 0},
            completion_events=(_Event(),),
        )
        with self.assertRaises(Mxfp4StateError):
            state.publish_cache(
                _key(parameter, state),
                parameter,
                _cache_refs(),
                component_ranks=(0, 1),
                votes={0: 0, 1: 0},
                completion_events=(_Event(),),
            )

    def test_budget_default_override_and_allocation_fence(self):
        budget = BudgetedStorage(100)
        plan = MemoryPlan(82, 90, 20, 2, 112)
        with self.assertRaises(CacheBudgetMiss):
            budget.begin_plan(plan, free_hbm_bytes=1000)
        budget = BudgetedStorage(
            100,
            persistent_numerator=82,
            persistent_denominator=100,
            override_provenance="explicit test config",
        )
        budget.begin_plan(plan, free_hbm_bytes=1000)
        weight = budget.allocate(
            "packed", "persistent_cache_bytes", 82, lambda: _Tensor(82)
        )
        budget.retain_view(
            "transpose-scales", _Tensor(82, storage=weight.untyped_storage())
        )
        self.assertEqual(budget.telemetry()["persistent_cache_bytes"], 82)
        budget.allocate("builder", "peak_cache_build_bytes", 8, lambda: _Tensor(8))
        budget.allocate("ZG", "reusable_workspace_bytes", 20, lambda: _Tensor(20))
        called = []
        with self.assertRaises(Mxfp4StateError):
            budget.allocate(
                "too-big", "persistent_cache_bytes", 1, lambda: called.append(True)
            )
        self.assertEqual(called, [])
        budget.seal()
        with self.assertRaises(Mxfp4StateError):
            budget.allocate(
                "late", "transient_launch_bytes", 1, lambda: called.append(True)
            )
        self.assertEqual(called, [])
        self.assertEqual(budget.telemetry()["peak_cache_build_highwater_bytes"], 90)

    def test_retired_storage_stays_charged_and_plan_breach(self):
        budget = BudgetedStorage(100)
        budget.begin_plan(MemoryPlan(70, 90, 0, 0, 90), free_hbm_bytes=100)
        budget.allocate("old", "persistent_cache_bytes", 70, lambda: _Tensor(70))
        budget.seal()
        event = _Event(False)
        budget.retire(("old",), events=(event,))
        self.assertEqual(budget.collect_retired()["retired_storage_bytes"], 70)
        with self.assertRaises(CacheBudgetMiss):
            budget.begin_plan(MemoryPlan(60, 60, 0, 0, 60), free_hbm_bytes=100)
        event.ready = True
        self.assertEqual(budget.collect_retired()["persistent_cache_bytes"], 0)
        budget.begin_plan(MemoryPlan(60, 60, 0, 0, 60), free_hbm_bytes=100)
        with self.assertRaises(Mxfp4StateError):
            budget.allocate(
                "bad-real-size", "persistent_cache_bytes", 60, lambda: _Tensor(61)
            )
        with self.assertRaises(Mxfp4StateError):
            budget.seal()

    def test_bref_unique_storage_and_ddp_slice_validation(self):
        storage = _Storage(32)
        p = _Tensor(16, "torch.bfloat16", shape=(2, 4), storage=storage)
        q = _Tensor(16, "torch.bfloat16", shape=(2, 4), storage=storage, offset=8)
        self.assertEqual(source_weight_bytes((p, q)), 32)
        owner = _Tensor(64, "torch.float32")
        p.main_grad = _Tensor(
            32, "torch.float32", shape=(2, 4), storage=owner.untyped_storage()
        )
        q.main_grad = _Tensor(
            32, "torch.float32", shape=(2, 4), storage=owner.untyped_storage(), offset=8
        )
        left = MainGradBinding.capture(p, owner, window=0, weight_epoch=0)
        right = MainGradBinding.capture(q, owner, window=0, weight_epoch=0)
        validate_disjoint_main_grads((left, right))
        left.validate(window=0, weight_epoch=0)
        q.main_grad.offset = 4
        overlapping = MainGradBinding.capture(q, owner, window=0, weight_epoch=0)
        with self.assertRaises(Mxfp4StateError):
            validate_disjoint_main_grads((left, overlapping))
        with self.assertRaises(Mxfp4StateError):
            right.validate(window=0, weight_epoch=0)
        p.main_grad = _Tensor(32, "torch.float32", shape=(2, 4))
        with self.assertRaises(Mxfp4StateError):
            left.validate(window=0, weight_epoch=0)

    def test_recompute_identity_compact_retention_and_backward_once(self):
        tape = RecomputeTape((0, 0))
        scopes = (ScopeKey(0, 0, 0, 0, 0), ScopeKey(0, 0, 0, 1, 1))
        counts = ((0, 1, 33) + (0,) * 29, (17, 0, 2) + (0,) * 29)
        routes = [RoutingMetadataV1(c, sum(c), (100,) * 8) for c in counts]
        bounds, receipts = [object(), object()], [object(), object()]
        compact = {
            key: _Tensor(10)
            for key in ("x_row", "x_row_scales", "x_h16t", "x_h16t_scales")
        }
        self.assertEqual(tape.begin(), "ORIGINAL")
        for i, scope in enumerate(scopes):
            tape.original(
                scope,
                route=routes[i],
                bound=bounds[i],
                receipts=receipts[i],
                weight_epoch=0,
                cache_generation=0,
                compact_operands=compact,
                leases=(object(),),
                events=(_Event(),),
                producer_frontier=(1, 0, 0),
                post_f_status=0,
            )
        tape.end()
        self.assertEqual(tape.begin(), "REPLAY")
        with self.assertRaises(Mxfp4StateError):
            tape.replay(
                scopes[0],
                counts=counts[0],
                weight_epoch=0,
                cache_generation=0,
                producer_frontier=(2, 0, 0),
            )
        for i, scope in enumerate(scopes):
            replay = tape.replay(
                scope,
                counts=counts[i],
                weight_epoch=0,
                cache_generation=0,
                producer_frontier=(1, 0, 0),
            )
            self.assertIs(replay["route"], routes[i])
            self.assertIs(replay["bound"], bounds[i])
            self.assertIs(replay["receipts"], receipts[i])
            self.assertIs(replay["compact_operands"]["x_row"], compact["x_row"])
        tape.end()
        with self.assertRaises(Mxfp4StateError):
            tape.begin()
        for scope in reversed(scopes):
            for fc in (2, 1):
                tape.claim_backward(scope, fc)
        with self.assertRaises(Mxfp4StateError):
            tape.claim_backward(scopes[0], 1)
        event = _Event(False)
        self.assertFalse(tape.retire(consumer_events=(event,)))
        event.ready = True
        self.assertTrue(tape.retire(consumer_events=(event,)))


if __name__ == "__main__":
    unittest.main()
