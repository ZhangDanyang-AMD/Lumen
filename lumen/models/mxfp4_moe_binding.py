# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0
"""Both-FC preflight and selected, metered original binding for isolated Core cells."""

import torch

from lumen.ops.mxfp4_moe_materialize import materialize_fc
from lumen.quantize.mxfp4_moe_state import CacheKey, MemoryPlan, plan_dense_layer
from lumen.utils.mxfp4_moe_control import ControlSequence
from lumen.utils.mxfp4_moe_execution_ledger import FCObligations
from lumen.utils.mxfp4_moe_ledger import Mxfp4StateError


class RoutedLayerBinding:
    def __init__(
        self,
        *,
        fc1,
        fc2,
        prepared,
        lifecycle,
        budget,
        control,
        resource_pool,
        other_device_bytes,
        other_resident_tensors,
        cu_count,
        cache_disabled=False
    ):
        self.modules = (fc1, fc2)
        self.prepared = tuple(prepared)
        self.lifecycle, self.budget, self.control = lifecycle, budget, control
        self.resource_pool = resource_pool
        self.other_device_bytes = other_device_bytes
        self.other_resident_tensors = tuple(other_resident_tensors)
        self.cu_count = cu_count
        self.pending = {}
        self.cache_disabled = cache_disabled
        self.owned_cache_names = set()
        self.scope_names = {}
        lifecycle.layer_bindings.append(self)

    def bindings(self, fc):
        module = self.modules[fc - 1]
        return tuple(
            self.lifecycle.handoff.bindings[getattr(module, "weight" + str(slot))]
            for slot in range(module.num_gemms)
        )

    def preflight(self, scope, route):
        identity = (
            scope.attempted_step,
            scope.microbatch_ordinal,
            scope.global_layer_ordinal,
            0,
        )

        def check():
            bindings = self.bindings(1) + self.bindings(2)
            for fc, prepared in enumerate(self.prepared, 1):
                if (
                    prepared.static_spec[9:11] != (1, 1)
                    or max(route.counts) > prepared.static_spec[7]
                ):
                    raise Mxfp4StateError("STATIC_CAPABILITY_MISS")
                for binding in self.bindings(fc):
                    binding.validate(
                        window=self.lifecycle.weights._window,
                        weight_epoch=self.lifecycle.weights.committed_weight_epoch,
                    )
            # This budget owns one layer; other-layer totals are explicit external demand.
            live = 1
            plan, details = plan_dense_layer(
                route,
                live_leases=live,
                source_bindings=bindings,
                other_device_bytes=self.other_device_bytes,
                other_resident_tensors=self.other_resident_tensors,
                cold_status_words=2 * len(route.counts) + 2,
            )
            current = self.budget.telemetry()
            cache_extra = (
                current["persistent_cache_bytes"] if self.cache_disabled else 0
            )
            workspace_extra = current["reusable_workspace_bytes"]
            plan = MemoryPlan(
                plan.persistent_cache_bytes + cache_extra,
                plan.peak_cache_build_bytes + cache_extra,
                plan.reusable_workspace_bytes + workspace_extra,
                plan.transient_launch_bytes,
                plan.device_total_preflight_bytes + cache_extra + workspace_extra,
                plan.planned_inputs,
            )
            resident = (
                tuple(
                    tensor
                    for binding in bindings
                    for tensor in (binding.parameter, binding.main_grad)
                )
                + self.other_resident_tensors
            )
            free, _ = torch.cuda.mem_get_info(self.modules[0].weight0.device)
            self.budget.begin_plan(
                plan,
                free_hbm_bytes=free,
                resident_inputs=resident,
                planned_owned_names=self.budget.live_names,
            )
            self.pending[scope] = (route, details)

        self.control.metadata_check(
            ControlSequence(*identity, "ROUTE_READY_PREBIND", 0), check
        )

    def cache_key(self, binding):
        weights = self.lifecycle.weights
        return CacheKey(
            id(binding.parameter),
            weights.committed_weight_epoch,
            weights.cache_generation,
            "lumen_a4w4_direct_training_v1",
            "lumen_fp4_e2m1_e8m0_scale_block_32x32_s1024_v1",
            "none",
            "mxfp4_weight_tile32x32_v1",
            "plain_dense_v1",
            binding.device,
            "gfx950",
            self.cu_count,
            binding.shape,
            self.modules[0].num_gemms,
        )

    def materialize(self, x, invocation, fc):
        scope, route = invocation.scope, invocation.route
        if scope not in self.pending or self.pending[scope][0] is not route:
            raise Mxfp4StateError("MISSING_BOTH_FC_PREFLIGHT")
        bindings = self.bindings(fc)
        tokens = tuple(
            token
            for token in self.lifecycle.rng.receipts(
                scope,
                route=route,
                weight_epoch=self.lifecycle.weights.committed_weight_epoch,
            )
            if token.fc == fc
        )
        prepared = self.prepared[fc - 1]
        refs = tuple(
            (
                None
                if self.cache_disabled
                else self.lifecycle.weights.cache(
                    self.cache_key(binding), binding.parameter
                )
            )
            for binding in bindings
        )
        # Materializer handles per-expert hot/cold without changing registered vote count.
        obligations = FCObligations(
            self.lifecycle.ledger,
            scope,
            fc,
            route,
            self.lifecycle.weights.committed_weight_epoch,
            tokens,
            prepared,
        )
        identity = (
            scope.attempted_step,
            scope.microbatch_ordinal,
            scope.global_layer_ordinal,
            fc,
        )
        before = set(self.budget.live_names)
        execution = materialize_fc(
            x,
            prepared=prepared,
            route=route,
            tokens=tokens,
            bindings=bindings,
            handoff=self.lifecycle.handoff,
            control=self.control,
            identity=identity,
            allocate=self.budget.allocate,
            resources=self.resource_pool.acquire((scope, fc)),
            obligations=obligations,
            cache_refs=refs,
        )
        new_names = set(self.budget.live_names) - before
        cache_names = {
            name
            for name in new_names
            if name.rsplit("/", 1)[-1] in ("w_fp4", "w_scales", "wt_fp4")
        }
        self.scope_names.setdefault(scope, set()).update(
            new_names if self.cache_disabled else new_names - cache_names
        )
        self.owned_cache_names.update(cache_names if not self.cache_disabled else ())
        for binding, old, cache in zip(bindings, refs, execution.candidate_caches):
            if old is None and not self.cache_disabled:
                self.lifecycle.weights.publish_cache(
                    self.cache_key(binding),
                    binding.parameter,
                    cache,
                    component_ranks=self.control.component,
                    votes=execution.materialization_votes,
                    completion_events=(execution.materialization_event,),
                )
        if fc == 2:
            self.budget.seal()
            del self.pending[scope]
            self.lifecycle.retain(invocation)

        def retire(events):
            self.resource_pool.release((scope, fc), events)
            self.retire_scope(scope, events)

        execution.retire_storage = retire
        return execution

    def retire_scope(self, scope, events):
        names = self.scope_names.pop(scope, None)
        if names:
            self.budget.retire(tuple(names), events=events)
            self.budget.collect_retired()

    def invalidate_cache_after_consumers(self, events):
        if self.scope_names or self.pending:
            raise Mxfp4StateError("UNSAFE_WEIGHT_MUTATION")
        if self.owned_cache_names:
            self.budget.retire(tuple(self.owned_cache_names), events=events)
            self.budget.collect_retired()
            self.owned_cache_names.clear()
