# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0
"""Isolated routed-Core startup and explicit driver execution contexts."""

import os
from pathlib import Path

import torch
import torch.distributed as dist

from lumen.models.mxfp4_moe import CoreExpertsScope
from lumen.models.mxfp4_moe_binding import RoutedLayerBinding
from lumen.models.mxfp4_moe_driver import RoutedDriver
from lumen.models.mxfp4_moe_lifecycle import RoutedLifecycle
from lumen.ops.mxfp4_moe import _probe_aiter_mxfp4_training
from lumen.quantize.mxfp4_moe_rng import PhiloxDrawAllocator
from lumen.quantize.mxfp4_moe_state import (
    BudgetedStorage,
    WeightEpochState,
    source_weight_bytes,
)
from lumen.utils.mxfp4_moe_control import ControlSequence, Mxfp4Control
from lumen.utils.mxfp4_moe_ddp import Mxfp4DDPHandoff
from lumen.utils.mxfp4_moe_ledger import Mxfp4StateError, ObligationLedger


class RoutedCoreRuntime(RoutedDriver):
    """Own DDP construction through immediate installation in the isolated fixture.

    Original Core hooks remain. External Node-hook additions after installation
    are unsupported; this boundary does not enumerate arbitrary Node subscribers.
    """

    def __init__(
        self,
        module,
        layers,
        *,
        options,
        config,
        ddp_config,
        run_seed,
        manifest_digest,
        physical_run_id,
        journal_path,
        sender_rows,
        framework_resident_tensors,
        framework_future_bytes,
        host_limit_bytes,
        max_expert_rows=257,
        max_live_invocations_per_layer=1
    ):
        from lumen.models.mxfp4_moe_options import validate_p0_options

        self.selector_receipt = validate_p0_options(options, config)
        from aiter.ops.triton.utils.config_utils import resolve_config_dir

        if len(layers) != 2 or len(set(map(id, layers))) != 2:
            raise Mxfp4StateError("ISOLATED_TWO_LAYER_FIXTURE_REQUIRED")
        if (
            os.environ.get("LUMEN_DSV4_LINEAR_FP8")
            or os.environ.get("LUMEN_MXFP4_DGRAD_HADAMARD") == "1"
        ):
            raise Mxfp4StateError("INCOMPATIBLE_QUANTIZATION_FLAGS")
        from megatron.core.distributed.distributed_data_parallel import (
            DistributedDataParallel,
        )

        if isinstance(module, DistributedDataParallel):
            raise Mxfp4StateError("PREBUILT_DDP_INSTALLATION_UNSUPPORTED")
        if any(
            not any(candidate is layer for candidate in module.modules())
            for layer in layers
        ):
            raise Mxfp4StateError("ISOLATED_LAYER_OWNERSHIP_MISMATCH")
        parameters = tuple(
            getattr(fc, "weight" + str(slot))
            for layer in layers
            for fc in (layer.experts.linear_fc1, layer.experts.linear_fc2)
            for slot in range(fc.num_gemms)
        )
        ddp = DistributedDataParallel(
            config=config, ddp_config=ddp_config, module=module
        )
        self.handoff = Mxfp4DDPHandoff(ddp, parameters)
        self.ddp, self.layers = ddp, tuple(layers)
        self.sender_rows = tuple(sender_rows)
        self.max_live_invocations_per_layer = max_live_invocations_per_layer
        self.max_expert_rows = max_expert_rows
        self.scope = None
        self.microbatch = None
        self._driver_failed = False
        self._microbatch_count = 0
        self._next_microbatch = 0
        self._microbatch_scope_active = None
        self.bindings = []
        from lumen.utils.mxfp4_moe_resources import PreparedResources

        self.resources = PreparedResources(
            fc_leases=4 * max_live_invocations_per_layer,
            experts=layers[0].experts.linear_fc1.num_gemms,
            pinned_limit_bytes=host_limit_bytes,
        )
        self.installation_receipt = {
            "boundary": "exclusive_core_ddp_construction_immediate_adapter_install",
            "original_core_hooks_preserved": True,
            "arbitrary_node_hooks_enumerated": False,
            "external_node_subscription_after_install": "unsupported",
        }
        self.weights = WeightEpochState()
        self.rng = PhiloxDrawAllocator(run_seed, dist.get_rank())
        self.ledger = ObligationLedger(manifest_digest)
        self.lifecycle = RoutedLifecycle(
            ddp, self.handoff, self.weights, self.rng, self.ledger
        )
        self.control = Mxfp4Control(
            ddp, physical_run_id=physical_run_id, journal_path=journal_path
        )
        self.control.record("p0_selector", self.selector_receipt)
        self.control.record("controlled_ddp_installation", self.installation_receipt)
        self.control.record("prepared_resources", self.resources.telemetry)
        self.control.vote(ControlSequence(0, 0, 0, 0, "STARTUP", 0))
        api = _probe_aiter_mxfp4_training()
        if api is None:
            raise Mxfp4StateError("STATIC_CAPABILITY_MISS")
        cu = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        manifest = str(
            Path(resolve_config_dir("gemm", "GEMM-AFP4WFP4"))
            / "GEMM-AFP4WFP4-CONTRACT-MANIFEST-V1.json"
        )
        prepared = tuple(
            api.mxfp4_dense_prepare_kernel_v1(
                (
                    1,
                    0,
                    950,
                    cu,
                    1,
                    4096,
                    k,
                    max_expert_rows,
                    0,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    0,
                    1,
                ),
                manifest,
            )
            for k in (4096, 2048)
        )
        routed_ids = set(map(id, parameters))
        framework_ddp = tuple(
            tensor
            for parameter in ddp.params_with_grad
            if id(parameter) not in routed_ids
            for tensor in (parameter, parameter.main_grad)
        )
        self.framework_resident = tuple(framework_resident_tensors) + framework_ddp
        self.framework_future_bytes = framework_future_bytes
        self.shells = []
        for layer_index, layer in enumerate(layers):
            fc1, fc2 = layer.experts.linear_fc1, layer.experts.linear_fc2
            local = tuple(
                getattr(fc, "weight" + str(slot))
                for fc in (fc1, fc2)
                for slot in range(fc.num_gemms)
            )
            budget = BudgetedStorage(
                source_weight_bytes(local), device=fc1.weight0.device
            )
            binding = RoutedLayerBinding(
                fc1=fc1,
                fc2=fc2,
                prepared=prepared,
                lifecycle=self.lifecycle,
                budget=budget,
                control=self.control,
                resource_pool=self.resources,
                other_device_bytes=0,
                other_resident_tensors=(),
                cu_count=cu,
            )
            self.bindings.append(binding)

            def supplier(index=layer_index):
                if self.microbatch is None:
                    raise Mxfp4StateError("MISSING_DRIVER_MICROBATCH")
                return (
                    self.scopes[(self.microbatch, index)],
                    self.weights.committed_weight_epoch,
                )

            self.shells.append(
                CoreExpertsScope(
                    layer.experts,
                    scope_supplier=supplier,
                    sender_rows_supplier=lambda: self.sender_rows,
                    rng=self.rng,
                    preflight=lambda scope, route, item=binding: self.preflight(
                        item, scope, route
                    ),
                    fc1_materializer=binding.materialize,
                    fc2_materializer=binding.materialize,
                )
            )

    def preflight(self, binding, scope, route):
        # Register disjoint real external ranges; never credit an entire DDP bucket twice.
        tensors = list(self.framework_resident)
        future = self.framework_future_bytes
        for other in self.bindings:
            if other is binding:
                continue
            for fc in (1, 2):
                for owner in other.bindings(fc):
                    tensors.extend((owner.parameter, owner.main_grad))
            tensors.extend(other.budget._names.values())
            experts = other.modules[0].num_gemms
            sender_total = sum(self.sender_rows)
            # Bounds are demand only, not a fabricated future route or RNG declaration.
            weight_elements = experts * (4096 * 4096 + 4096 * 2048)
            max_m = min(6 * sender_total, experts * self.max_expert_rows)
            max_p = max_m + 31 * experts
            cache = weight_elements + weight_elements // 1024
            workspace = self.max_live_invocations_per_layer * (
                (max_m + max_p) * (4096 + 4096 + 2048 + 4096) * 17 // 32
                + 2 * max_m * (4096 + 4096 + 4096 + 2048)
                + 4 * weight_elements
                + 8
            ) + 4 * (2 * experts + 2)
            owned = other.budget.telemetry()
            future += max(
                0,
                cache
                + workspace
                - owned["persistent_cache_bytes"]
                - owned["reusable_workspace_bytes"],
            )
        from lumen.quantize.mxfp4_moe_state import _union_size, tensor_range

        ranges = {}
        for tensor in tensors:
            region = tensor_range(tensor)
            ranges.setdefault(region.key, []).append((region.start, region.end))
        binding.other_device_bytes = _union_size(ranges) + future
        binding.other_resident_tensors = tuple(tensors)
        binding.preflight(scope, route)
