# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0
"""MXFP4 host lifecycle, budgeted storage and compact recompute retention.

No tensor allocation, quantization, hooks or optimizer monkey-patching occurs on
import. The scoped adapter supplies real recorded events and allocation factories.
"""

from dataclasses import dataclass
from threading import RLock

from lumen.utils.mxfp4_moe_ledger import (
    INT64_MAX,
    Mxfp4StateError,
    completed_events,
    uint,
)

CATEGORIES = (
    "persistent_cache_bytes",
    "peak_cache_build_bytes",
    "reusable_workspace_bytes",
    "transient_launch_bytes",
)


@dataclass(frozen=True)
class StorageRange:
    key: tuple
    start: int
    end: int
    nbytes: int


def tensor_range(tensor):
    """Inspect real tensor metadata, retaining full backing-storage accounting."""
    storage = tensor.untyped_storage()
    nbytes = uint(storage.nbytes(), "storage bytes", INT64_MAX)
    element = uint(tensor.element_size(), "element size", INT64_MAX)
    shape, stride = tuple(tensor.shape), tuple(tensor.stride())
    if len(shape) != len(stride) or element == 0:
        raise Mxfp4StateError("MAIN_GRAD_MISMATCH", "tensor metadata")
    for value in (*shape, *stride):
        uint(value, "shape/stride", INT64_MAX)
    start = uint(tensor.storage_offset() * element, "storage offset", INT64_MAX)
    span = (
        0
        if 0 in shape
        else element + sum((s - 1) * t * element for s, t in zip(shape, stride))
    )
    end = uint(start + span, "tensor end", INT64_MAX)
    if end > nbytes or (span and any(s == 0 for s in stride)):
        raise Mxfp4StateError("MAIN_GRAD_MISMATCH", "capacity/stride")
    pointer = uint(storage.data_ptr(), "storage pointer")
    if nbytes:
        if pointer == 0:
            raise Mxfp4StateError("MAIN_GRAD_MISMATCH", "nonempty null storage")
        key = (str(tensor.device), pointer)
    else:
        # Empty allocations have no byte address; StorageImpl identity survives
        # temporary Python storage wrappers and is shared by true storage views.
        identity = uint(storage._cdata, "empty StorageImpl identity")
        if identity == 0:
            raise Mxfp4StateError("MAIN_GRAD_MISMATCH", "missing StorageImpl identity")
        key = (str(tensor.device), "empty_storage", identity)
    return StorageRange(key, start, end, nbytes)


def overlaps(a, b):
    return a.key == b.key and max(a.start, b.start) < min(a.end, b.end)


def source_weight_bytes(parameters):
    """B_ref counts unique local routed BF16 storage, never optimizer storage."""
    sizes = {}
    for parameter in parameters:
        if str(parameter.dtype) != "torch.bfloat16":
            raise Mxfp4StateError("WEIGHT_SCHEMA_MISMATCH")
        region = tensor_range(parameter)
        sizes[region.key] = region.nbytes
    return uint(sum(sizes.values()), "B_ref", INT64_MAX)


@dataclass(frozen=True)
class CacheKey:
    parameter_identity: int
    committed_weight_epoch: int
    cache_generation: int
    training_recipe_id: str
    operand_quant_recipe_id: str
    operand_transform_id: str
    physical_layout_id: str
    prepared_layout_id: str
    device: str
    gfx: str
    cu_count: int
    shape: tuple
    local_experts: int

    def __post_init__(self):
        for field in (
            "parameter_identity",
            "committed_weight_epoch",
            "cache_generation",
            "cu_count",
        ):
            uint(getattr(self, field), field)
        if (
            self.gfx != "gfx950"
            or self.shape not in ((4096, 4096), (4096, 2048))
            or self.local_experts not in (32, 64)
        ):
            raise Mxfp4StateError("WEIGHT_SCHEMA_MISMATCH")
        if (
            self.training_recipe_id != "lumen_a4w4_direct_training_v1"
            or self.operand_quant_recipe_id
            != "lumen_fp4_e2m1_e8m0_scale_block_32x32_s1024_v1"
        ):
            raise Mxfp4StateError("NUMERIC_LAYOUT_MISMATCH")
        if (
            self.operand_transform_id != "none"
            or self.physical_layout_id != "mxfp4_weight_tile32x32_v1"
            or self.prepared_layout_id != "plain_dense_v1"
        ):
            raise Mxfp4StateError("NUMERIC_LAYOUT_MISMATCH")


@dataclass(frozen=True)
class PlannedInput:
    """Strong, process-local evidence for storage explicitly included in a plan."""

    tensor: object
    binding: object
    role: str
    region: StorageRange

    def validate(self):
        if self.role == "other":
            if self.binding is not None or tensor_range(self.tensor) != self.region:
                raise Mxfp4StateError(
                    "PLAN_RESIDENCY_MISMATCH", "other storage changed"
                )
            return
        if not isinstance(self.binding, MainGradBinding) or self.role not in (
            "weight",
            "main_grad",
        ):
            raise Mxfp4StateError("PLAN_RESIDENCY_MISMATCH", "unrecognized owner")
        self.binding.validate(
            window=self.binding.window, weight_epoch=self.binding.weight_epoch
        )
        expected = (
            self.binding.parameter if self.role == "weight" else self.binding.main_grad
        )
        if self.tensor is not expected or tensor_range(self.tensor) != self.region:
            raise Mxfp4StateError("PLAN_RESIDENCY_MISMATCH", "owner/range changed")

    @property
    def covered(self):
        return (
            (0, self.region.nbytes)
            if self.role == "weight"
            else (self.region.start, self.region.end)
        )


def _union_size(ranges):
    total = 0
    for intervals in ranges.values():
        high = -1
        for start, end in sorted(intervals):
            total += max(0, end - max(start, high))
            high = max(high, end)
    return total


def _covered_by(interval, allowed):
    start, end = interval
    frontier = start
    for a, b in sorted(allowed):
        if a <= frontier:
            frontier = max(frontier, b)
    return frontier >= end


@dataclass(frozen=True)
class MemoryPlan:
    persistent_cache_bytes: int
    peak_cache_build_bytes: int
    reusable_workspace_bytes: int
    transient_launch_bytes: int
    device_total_preflight_bytes: int
    planned_inputs: tuple = ()

    def __post_init__(self):
        for name in (*CATEGORIES, "device_total_preflight_bytes"):
            uint(getattr(self, name), name, INT64_MAX)
        if type(self.planned_inputs) is not tuple:
            raise Mxfp4StateError("PLAN_RESIDENCY_MISMATCH")
        ranges = {}
        devices = set()
        for item in self.planned_inputs:
            if not isinstance(item, PlannedInput):
                raise Mxfp4StateError("PLAN_RESIDENCY_MISMATCH")
            item.validate()
            devices.add(str(item.tensor.device))
            ranges.setdefault(item.region.key, []).append(item.covered)
        if (
            len(devices) > 1
            or _union_size(ranges)
            > self.device_total_preflight_bytes
            - self.peak_cache_build_bytes
            - self.reusable_workspace_bytes
            - self.transient_launch_bytes
        ):
            raise Mxfp4StateError(
                "PLAN_RESIDENCY_MISMATCH", "input storage not covered by total"
            )
        if self.peak_cache_build_bytes < self.persistent_cache_bytes:
            raise Mxfp4StateError("ALLOCATOR_PLAN_BREACH", "peak below persistent")
        if (
            self.device_total_preflight_bytes
            < self.peak_cache_build_bytes
            + self.reusable_workspace_bytes
            + self.transient_launch_bytes
        ):
            raise Mxfp4StateError(
                "ALLOCATOR_PLAN_BREACH", "device total omits owned storage"
            )


def plan_dense_layer(
    route,
    *,
    live_leases=1,
    retired_cache_bytes=0,
    cache_alignment_bytes=0,
    workspace_alignment_bytes=0,
    other_device_bytes=0,
    cold_status_words=2,
    source_bindings=(),
    other_resident_tensors=(),
):
    """Predict both signed FCs, including inactive experts and full Z/G leases.

    other_device_bytes includes resident and future framework/master/other-layer
    demand. Register resident ranges in other_resident_tensors, then pass them to
    begin_plan(resident_inputs=...) for credit against current free HBM. Unregistered
    bytes remain future demand. Source weights/main_grad are already added here;
    other ranges must be disjoint from them (DDP owner slices, not whole buckets).
    The selected single-split profile has zero algorithm/launch scratch.
    """
    from lumen.quantize.mxfp4_moe_rng import RoutingMetadataV1

    if not isinstance(route, RoutingMetadataV1):
        raise Mxfp4StateError("ROUTING_SCHEMA_MISMATCH")
    for value in (
        live_leases,
        retired_cache_bytes,
        cache_alignment_bytes,
        workspace_alignment_bytes,
        other_device_bytes,
        cold_status_words,
    ):
        uint(value, "memory plan input", INT64_MAX)
    if not live_leases:
        raise Mxfp4StateError("ALLOCATOR_PLAN_BREACH", "at least one live lease")
    e, m, p = len(route.counts), sum(route.counts), sum(route.padded_counts)
    weight_elements = e * (4096 * 4096 + 4096 * 2048)
    b_ref = 2 * weight_elements
    cache = weight_elements + weight_elements // 1024
    compact = sum((m + p) * width * 17 // 32 for width in (4096, 4096, 2048, 4096))
    output = 2 * m * (4096 + 4096 + 4096 + 2048)
    zg = 4 * weight_elements
    status = 2 * 4
    cold_status = 4 * cold_status_words
    workspace = (
        live_leases * (compact + output + zg + status)
        + cold_status
        + workspace_alignment_bytes
    )
    persistent = cache + retired_cache_bytes + cache_alignment_bytes
    main_grad = 4 * weight_elements
    inputs = []
    if source_bindings:
        bindings = tuple(source_bindings)
        if len(bindings) != 2 * e or any(
            not isinstance(b, MainGradBinding) for b in bindings
        ):
            raise Mxfp4StateError(
                "PLAN_RESIDENCY_MISMATCH", "complete two-FC DDP bindings required"
            )
        validate_disjoint_main_grads(bindings)
        if sorted(b.shape for b in bindings) != sorted(
            [(4096, 4096)] * e + [(4096, 2048)] * e
        ):
            raise Mxfp4StateError("PLAN_RESIDENCY_MISMATCH", "different layer shapes")
        b_ref = source_weight_bytes(b.parameter for b in bindings)
        for b in bindings:
            for role, tensor in (("weight", b.parameter), ("main_grad", b.main_grad)):
                item = PlannedInput(tensor, b, role, tensor_range(tensor))
                item.validate()
                inputs.append(item)
    other_ranges = {}
    for tensor in other_resident_tensors:
        region = tensor_range(tensor)
        if any(
            region.key == item.region.key
            and max(region.start, item.covered[0]) < min(region.end, item.covered[1])
            for item in inputs
            if item.role != "other"
        ):
            raise Mxfp4StateError(
                "PLAN_RESIDENCY_MISMATCH", "other overlaps layer input"
            )
        item = PlannedInput(tensor, None, "other", region)
        item.validate()
        inputs.append(item)
        other_ranges.setdefault(region.key, []).append(item.covered)
    if _union_size(other_ranges) > other_device_bytes:
        raise Mxfp4StateError("PLAN_RESIDENCY_MISMATCH", "other demand omits residency")
    total = b_ref + main_grad + other_device_bytes + persistent + workspace
    plan = MemoryPlan(persistent, persistent, workspace, 0, total, tuple(inputs))
    details = {
        "B_ref": b_ref,
        "weight_cache_payload_scales": cache,
        "DDP_main_grad": main_grad,
        "retired_cache": retired_cache_bytes,
        "compact_operands_per_lease": compact,
        "outputs_per_lease": output,
        "ZG_per_lease": zg,
        "ZG_all_live_leases": zg * live_leases,
        "status_device_per_lease": status,
        "status_pinned_host_per_lease": status,
        "cold_status_device_and_host_each": cold_status,
        "live_leases": live_leases,
        "other_device_bytes": other_device_bytes,
        "cache_alignment_bytes": cache_alignment_bytes,
        "workspace_alignment_bytes": workspace_alignment_bytes,
    }
    return plan, details


class CacheBudgetMiss(Mxfp4StateError):
    def __init__(self, detail):
        super().__init__("CACHE_BUDGET_MISS", detail)
        self.disposition = "CANDIDATE_MISS"


class BudgetedStorage:
    """Reserve before calling any allocator; freeze allocation at bind commit.

    Categories are disjoint storage ownership: peak_cache_build_bytes records
    temporary builder scratch here; telemetry reports persistent plus that scratch.
    Aliased views must be retained via retain_view, not passed as new allocations.
    """

    def __init__(
        self,
        b_ref,
        *,
        persistent_numerator=3,
        persistent_denominator=4,
        override_provenance=None,
        device=None,
    ):
        self.b_ref = uint(b_ref, "B_ref", INT64_MAX)
        num = uint(persistent_numerator, "budget numerator")
        den = uint(persistent_denominator, "budget denominator")
        if not den or num > den or not num:
            raise Mxfp4StateError("INVALID_CACHE_BUDGET")
        if (num, den) != (3, 4) and not override_provenance:
            raise Mxfp4StateError(
                "INVALID_CACHE_BUDGET", "explicit override provenance required"
            )
        self.persistent_limit = self.b_ref * num // den
        self.override_provenance = override_provenance
        self.device = str(device) if device is not None else None
        self._resident_preflight_bytes = 0
        self._incremental_preflight_bytes = 0
        self._storage = {}
        self._names = {}
        self._retired = {}
        self._plan = None
        self._sealed = False
        self._failed = False
        self._highwater = 0
        self._lock = RLock()

    def telemetry(self):
        sizes = {
            c: sum(
                r["range"].nbytes for r in self._storage.values() if r["category"] == c
            )
            for c in CATEGORIES
        }
        sizes["peak_cache_build_bytes"] += sizes["persistent_cache_bytes"]
        self._highwater = max(self._highwater, sizes["peak_cache_build_bytes"])
        return {
            **sizes,
            "peak_cache_build_highwater_bytes": self._highwater,
            "device_total_preflight_bytes": (
                self._plan.device_total_preflight_bytes if self._plan else 0
            ),
            "retired_storage_bytes": sum(
                self._storage[k]["range"].nbytes for k in self._retired
            ),
            "persistent_limit": self.persistent_limit,
            "peak_limit": self.b_ref,
            "override_provenance": self.override_provenance,
            "already_resident_device_bytes": self._resident_preflight_bytes,
            "incremental_preflight_bytes": self._incremental_preflight_bytes,
        }

    @property
    def live_names(self):
        return tuple(sorted(self._names))

    def begin_plan(
        self, plan, *, free_hbm_bytes, resident_inputs=(), planned_owned_names=()
    ):
        with self._lock:
            if self._failed or (self._plan is not None and not self._sealed):
                raise Mxfp4StateError("ALLOCATOR_PLAN_BREACH", "unfinished/failed plan")
            if not isinstance(plan, MemoryPlan):
                raise Mxfp4StateError("ALLOCATOR_PLAN_BREACH", "plan type")
            live = self.telemetry()
            if (
                plan.persistent_cache_bytes > self.persistent_limit
                or plan.peak_cache_build_bytes > self.b_ref
            ):
                raise CacheBudgetMiss("persistent/peak quota")
            if any(live[c] > getattr(plan, c) for c in CATEGORIES):
                raise CacheBudgetMiss("live or retired storage exceeds proposed plan")
            planned_owned_names = tuple(planned_owned_names)
            if len(set(planned_owned_names)) != len(planned_owned_names) or set(
                planned_owned_names
            ) != set(self._names):
                raise Mxfp4StateError(
                    "PLAN_RESIDENCY_MISMATCH",
                    "plan must name all still-live owned allocations",
                )
            covered, resident = {}, {}
            devices = set()
            for item in plan.planned_inputs:
                item.validate()
                devices.add(str(item.tensor.device))
                covered.setdefault(item.region.key, []).append(item.covered)
            for record in self._storage.values():
                actual = tensor_range(record["tensor"])
                if actual != record["range"]:
                    raise Mxfp4StateError(
                        "PLAN_RESIDENCY_MISMATCH", "owned allocation changed"
                    )
                devices.add(str(record["tensor"].device))
                resident.setdefault(actual.key, []).append((0, actual.nbytes))
            for tensor in resident_inputs:
                region = tensor_range(tensor)
                devices.add(str(tensor.device))
                interval = (region.start, region.end)
                for item in plan.planned_inputs:
                    if item.tensor is tensor:
                        interval = item.covered
                        break
                if region.key not in covered or not _covered_by(
                    interval, covered[region.key]
                ):
                    raise Mxfp4StateError(
                        "PLAN_RESIDENCY_MISMATCH", "not included in this plan"
                    )
                resident.setdefault(region.key, []).append(interval)
            if len(devices) > 1 or (
                self.device is not None and devices and devices != {self.device}
            ):
                raise Mxfp4StateError("PLAN_RESIDENCY_MISMATCH", "wrong device")
            resident_bytes = _union_size(resident)
            if resident_bytes > plan.device_total_preflight_bytes:
                raise Mxfp4StateError(
                    "PLAN_RESIDENCY_MISMATCH", "resident exceeds total"
                )
            incremental = plan.device_total_preflight_bytes - resident_bytes
            if incremental > uint(free_hbm_bytes, "free HBM", INT64_MAX):
                raise CacheBudgetMiss("incremental device allocation exceeds free HBM")
            # Publish no reservation changes until all ownership/capacity checks pass.
            self._plan, self._sealed = plan, False
            self._resident_preflight_bytes = resident_bytes
            self._incremental_preflight_bytes = incremental
            if devices:
                self.device = next(iter(devices))

    def allocate(self, name, category, nbytes, factory):
        with self._lock:
            if (
                self._plan is None
                or self._sealed
                or self._failed
                or name in self._names
                or category not in CATEGORIES
            ):
                raise Mxfp4StateError(
                    "ALLOCATOR_PLAN_BREACH",
                    "allocation outside selected materialization",
                )
            nbytes = uint(nbytes, "reservation", INT64_MAX)
            live = self.telemetry()
            prospective = dict(live)
            if category == "persistent_cache_bytes":
                prospective[category] += nbytes
                prospective["peak_cache_build_bytes"] += nbytes
            else:
                prospective[category] += nbytes
            if any(prospective[c] > getattr(self._plan, c) for c in CATEGORIES):
                raise Mxfp4StateError(
                    "ALLOCATOR_PLAN_BREACH", "reservation exceeds approved plan"
                )
            # The lock holds the reservation across allocation; exceptions are selected-path fatal.
            try:
                tensor = factory()
                region = tensor_range(tensor)
            except BaseException:
                self._failed = True
                raise
            if (
                region.key in self._storage
                or region.nbytes != nbytes
                or (self.device is not None and str(tensor.device) != self.device)
            ):
                self._failed = True
                self._names[name] = (
                    tensor  # Retain unexpected allocation for diagnosis; never reuse.
                )
                raise Mxfp4StateError(
                    "ALLOCATOR_PLAN_BREACH", "actual backing allocation differs"
                )
            self.device = str(tensor.device)
            self._storage[region.key] = {
                "range": region,
                "category": category,
                "refs": {name},
                "tensor": tensor,
            }
            self._names[name] = tensor
            self.telemetry()
            return tensor

    def retain_view(self, name, tensor):
        with self._lock:
            if self._sealed or self._failed or name in self._names:
                raise Mxfp4StateError("ALLOCATOR_PLAN_BREACH", "late view retention")
            region = tensor_range(tensor)
            if (
                region.key not in self._storage
                or region.nbytes != self._storage[region.key]["range"].nbytes
            ):
                raise Mxfp4StateError("ALLOCATOR_PLAN_BREACH", "unaccounted storage")
            self._storage[region.key]["refs"].add(name)
            self._names[name] = tensor

    def seal(self):
        with self._lock:
            if self._plan is None or self._failed or self._sealed:
                raise Mxfp4StateError("ALLOCATOR_PLAN_BREACH")
            self._sealed = True

    def retire(self, names, *, events):
        with self._lock:
            names, events = tuple(names), tuple(events)
            if not events or any(name not in self._names for name in names):
                raise Mxfp4StateError("CACHE_LIFETIME_MISMATCH")
            keys = {tensor_range(self._names[name]).key for name in names}
            if any(not self._storage[key]["refs"].issubset(names) for key in keys):
                raise Mxfp4StateError(
                    "CACHE_LIFETIME_MISMATCH", "another alias remains live"
                )
            for key in keys:
                self._retired[key] = self._retired.get(key, ()) + events

    def collect_retired(self):
        with self._lock:
            for key, events in list(self._retired.items()):
                if completed_events(events):
                    for name in self._storage[key]["refs"]:
                        del self._names[name]
                    del self._storage[key]
                    del self._retired[key]
            return self.telemetry()


class WeightEpochState:
    """Explicit zero/update boundaries; local shard copy is not a BF16 commit."""

    def __init__(self):
        self.committed_weight_epoch = 0
        self.cache_generation = 0
        self.attempted_step = 0
        self._window = None
        self._leases = {}
        self._update = None
        self._update_due = False
        self._cache = {}
        self._lock = RLock()

    def open_window(self, *, zero_events):
        with self._lock:
            if (
                self._window is not None
                or self._leases
                or self._update is not None
                or self._update_due
            ):
                raise Mxfp4StateError("ACCUMULATION_WINDOW_MISMATCH")
            if not completed_events(zero_events):
                raise Mxfp4StateError("ZERO_BOUNDARY_NOT_READY")
            self._window = self.attempted_step
            self.attempted_step = uint(self.attempted_step + 1, "attempted step")
            return self._window

    def acquire(self, logical_key, refs):
        with self._lock:
            if self._window is None or logical_key in self._leases:
                raise Mxfp4StateError("ACCUMULATION_WINDOW_MISMATCH")
            self._leases[logical_key] = tuple(refs)
            return self.committed_weight_epoch, self.cache_generation, self._window

    def release(self, logical_key, *, events):
        with self._lock:
            if logical_key not in self._leases:
                raise Mxfp4StateError("CACHE_LIFETIME_MISMATCH")
            if not completed_events(events):
                return False
            del self._leases[logical_key]
            return True

    def close_window(self, *, reducer_events):
        with self._lock:
            if (
                self._window is None
                or self._leases
                or not completed_events(reducer_events)
            ):
                raise Mxfp4StateError(
                    "ACCUMULATION_WINDOW_MISMATCH", "consumers/reducer not drained"
                )
            self._window = None
            self._update_due = True

    def optimizer_result(self, *, success, full_bf16_sync_events=()):
        with self._lock:
            if (
                self._window is not None
                or self._leases
                or self._update is not None
                or not self._update_due
                or type(success) is not bool
            ):
                raise Mxfp4StateError("WEIGHT_EPOCH_MISMATCH")
            if not success:
                self._update_due = False
                return
            events = tuple(full_bf16_sync_events)
            if not events:
                raise Mxfp4StateError(
                    "WEIGHT_EPOCH_MISMATCH", "full BF16/all-gather events required"
                )
            self._update = events
            self._update_due = False

    def commit_ready_update(self):
        with self._lock:
            if self._update is None:
                raise Mxfp4StateError(
                    "WEIGHT_EPOCH_MISMATCH", "no successful update pending"
                )
            if not completed_events(self._update):
                return False
            self._invalidate(advance_epoch=True)
            self._update = None
            return True

    def _safe(self):
        if (
            self._window is not None
            or self._leases
            or self._update is not None
            or self._update_due
        ):
            raise Mxfp4StateError("UNSAFE_WEIGHT_MUTATION")

    def _invalidate(self, *, advance_epoch):
        generation = uint(self.cache_generation + 1, "cache generation")
        epoch = uint(self.committed_weight_epoch + int(advance_epoch), "weight epoch")
        self.cache_generation, self.committed_weight_epoch = generation, epoch
        self._cache.clear()

    def mark_mxfp4_weights_dirty(self, *, full_bf16_sync_events):
        """Commit a supported external mutation; naked .data writes are unsupported."""
        with self._lock:
            self._safe()
            if not completed_events(full_bf16_sync_events):
                raise Mxfp4StateError(
                    "WEIGHT_EPOCH_MISMATCH", "external commit not ready"
                )
            self._invalidate(advance_epoch=True)

    def invalidate_device_dtype_move(self, *, completion_events):
        with self._lock:
            self._safe()
            if not completed_events(completion_events):
                raise Mxfp4StateError("WEIGHT_EPOCH_MISMATCH", "move not complete")
            self._invalidate(advance_epoch=False)

    def publish_cache(
        self, key, parameter, packed_refs, *, component_ranks, votes, completion_events
    ):
        """Accept the adapter's completed all-component health vote, not local OK."""
        with self._lock:
            if (
                self._window is None
                or key.parameter_identity != id(parameter)
                or key.committed_weight_epoch != self.committed_weight_epoch
                or key.cache_generation != self.cache_generation
            ):
                raise Mxfp4StateError("WEIGHT_EPOCH_MISMATCH")
            ranks = tuple(component_ranks)
            if (
                not ranks
                or tuple(sorted(set(ranks))) != ranks
                or set(votes) != set(ranks)
                or any(type(v) is not int or v != 0 for v in votes.values())
            ):
                raise Mxfp4StateError("MATERIALIZATION_HEALTH_MISMATCH")
            if not completed_events(completion_events):
                raise Mxfp4StateError("MATERIALIZATION_NOT_READY")
            if (
                str(parameter.dtype) != "torch.bfloat16"
                or tuple(parameter.shape) != key.shape
                or str(parameter.device) != key.device
            ):
                raise Mxfp4StateError("WEIGHT_SCHEMA_MISMATCH")
            n, k = key.shape
            expected = {
                "w_fp4": (n, k // 2),
                "w_scales": (n // 32, k // 32),
                "wt_fp4": (k, n // 2),
                "wt_scales": (k // 32, n // 32),
            }
            if set(packed_refs) != set(expected):
                raise Mxfp4StateError(
                    "CACHE_LAYOUT_MISMATCH", "QW and same-QW transpose required"
                )
            for name, tensor in packed_refs.items():
                if (
                    str(tensor.dtype) != "torch.uint8"
                    or tuple(tensor.shape) != expected[name]
                    or str(tensor.device) != key.device
                ):
                    raise Mxfp4StateError("CACHE_LAYOUT_MISMATCH", name)
                tensor_range(tensor)
            scales = tensor_range(packed_refs["w_scales"])
            transpose_scales = tensor_range(packed_refs["wt_scales"])
            if scales != transpose_scales or tuple(
                packed_refs["wt_scales"].stride()
            ) != tuple(reversed(packed_refs["w_scales"].stride())):
                raise Mxfp4StateError(
                    "CACHE_LAYOUT_MISMATCH", "transpose must reuse original scale view"
                )
            q = tensor_range(packed_refs["w_fp4"])
            qt = tensor_range(packed_refs["wt_fp4"])
            source = tensor_range(parameter)
            if (
                any(overlaps(source, derived) for derived in (q, qt, scales))
                or overlaps(q, qt)
                or overlaps(q, scales)
                or overlaps(qt, scales)
            ):
                raise Mxfp4StateError("CACHE_LAYOUT_MISMATCH", "illegal cache alias")
            if any(entry.parameter_identity == id(parameter) for entry in self._cache):
                raise Mxfp4StateError(
                    "CACHE_GENERATION_MISMATCH", "one prepared layout per parameter"
                )
            self._cache[key] = (parameter, dict(packed_refs))

    def cache(self, key, parameter):
        with self._lock:
            if (
                key.committed_weight_epoch != self.committed_weight_epoch
                or key.cache_generation != self.cache_generation
            ):
                raise Mxfp4StateError("WEIGHT_EPOCH_MISMATCH")
            value = self._cache.get(key)
            if value is None:
                return None
            if value[0] is not parameter:
                raise Mxfp4StateError("WEIGHT_EPOCH_MISMATCH")
            return dict(value[1])

    def checkpoint(self):
        with self._lock:
            self._safe()
            return {
                "schema": "lumen.mxfp4_moe.weight_state.v1",
                "committed_weight_epoch": self.committed_weight_epoch,
                "attempted_step": self.attempted_step,
            }

    def load_checkpoint(self, record):
        with self._lock:
            self._safe()
            if (
                set(record) != {"schema", "committed_weight_epoch", "attempted_step"}
                or record["schema"] != "lumen.mxfp4_moe.weight_state.v1"
            ):
                raise Mxfp4StateError("REPLAY_FRONTIER_MISMATCH")
            epoch = uint(record["committed_weight_epoch"], "restored epoch")
            step = uint(record["attempted_step"], "restored attempted step")
            self._invalidate(advance_epoch=False)
            self.committed_weight_epoch, self.attempted_step = epoch, step


@dataclass(frozen=True)
class MainGradBinding:
    parameter: object
    main_grad: object
    owner: object
    region: StorageRange
    owner_region: StorageRange
    parameter_region: StorageRange
    shape: tuple
    device: str
    window: int
    weight_epoch: int

    @classmethod
    def capture(cls, parameter, owner, *, window, weight_epoch):
        grad = getattr(parameter, "main_grad", None)
        if (
            grad is None
            or not grad.is_contiguous()
            or str(grad.dtype) != "torch.float32"
            or str(owner.dtype) != "torch.float32"
            or str(parameter.dtype) != "torch.bfloat16"
            or tuple(grad.shape) != tuple(parameter.shape)
            or grad.device != parameter.device
        ):
            raise Mxfp4StateError("MAIN_GRAD_MISMATCH")
        region, owner_region = tensor_range(grad), tensor_range(owner)
        if (
            region.key != owner_region.key
            or region.start < owner_region.start
            or region.end > owner_region.end
            or region.start % 4
        ):
            raise Mxfp4StateError("MAIN_GRAD_MISMATCH", "not a valid DDP owner range")
        parameter_region = tensor_range(parameter)
        if overlaps(region, parameter_region):
            raise Mxfp4StateError("MAIN_GRAD_MISMATCH", "gradient aliases parameter")
        uint(window, "window")
        uint(weight_epoch, "epoch")
        return cls(
            parameter,
            grad,
            owner,
            region,
            owner_region,
            parameter_region,
            tuple(grad.shape),
            str(grad.device),
            window,
            weight_epoch,
        )

    def validate(self, *, window, weight_epoch):
        grad = getattr(self.parameter, "main_grad", None)
        if (
            grad is not self.main_grad
            or window != self.window
            or weight_epoch != self.weight_epoch
        ):
            raise Mxfp4StateError("MAIN_GRAD_MISMATCH", "late rebind/window/epoch")
        if (
            not grad.is_contiguous()
            or str(self.parameter.dtype) != "torch.bfloat16"
            or tuple(self.parameter.shape) != self.shape
            or str(self.parameter.device) != self.device
            or str(grad.dtype) != "torch.float32"
            or tuple(grad.shape) != self.shape
            or str(grad.device) != self.device
            or tensor_range(grad) != self.region
            or tensor_range(self.owner) != self.owner_region
            or tensor_range(self.parameter) != self.parameter_region
        ):
            raise Mxfp4StateError("MAIN_GRAD_MISMATCH", "late mutation")


def validate_disjoint_main_grads(bindings):
    for i, left in enumerate(bindings):
        for right in bindings[i + 1 :]:
            if left.parameter is right.parameter or overlaps(left.region, right.region):
                raise Mxfp4StateError(
                    "MAIN_GRAD_MISMATCH", "duplicate/overlapping parameter destinations"
                )


def checkpoint_sidecar(weight_state, rng_allocator, ledger, *, last_committed_step):
    """Capture the compact host sidecar at a caller-established all-rank safe point."""
    with weight_state._lock, rng_allocator._lock, ledger._lock:
        weights = weight_state.checkpoint()
        rng = rng_allocator.checkpoint()
        frontier = ledger.checkpoint(last_committed_step=last_committed_step)
        return {
            "schema": "lumen.mxfp4_moe.sidecar.v1",
            "weights": weights,
            "rng": rng,
            "ledger": frontier,
            "execution_manifest": ledger.manifest,
        }


def restore_sidecar(record, *, execution_manifest):
    from lumen.quantize.mxfp4_moe_rng import PhiloxDrawAllocator
    from lumen.utils.mxfp4_moe_ledger import ObligationLedger

    if (
        set(record) != {"schema", "weights", "rng", "ledger", "execution_manifest"}
        or record["schema"] != "lumen.mxfp4_moe.sidecar.v1"
        or record["execution_manifest"] != execution_manifest
    ):
        raise Mxfp4StateError("REPLAY_FRONTIER_MISMATCH", "sidecar schema/manifest")
    weights = WeightEpochState()
    weights.load_checkpoint(record["weights"])
    rng = PhiloxDrawAllocator.restore(record["rng"])
    ledger = ObligationLedger.restore(
        record["ledger"], manifest_digest=execution_manifest
    )
    return weights, rng, ledger


class RecomputeTape:
    """Closure-owned compact references; a second backward or third forward fails."""

    def __init__(self, context_key):
        self.context_key = tuple(context_key)
        self._records = []
        self._calls = 0
        self._cursor = 0
        self._executing = False
        self._backward = set()
        self._retired = False

    def begin(self):
        if self._executing or self._retired or self._calls >= 2 or self._backward:
            raise Mxfp4StateError("RECOMPUTE_CONTRACT_VIOLATION")
        self._calls += 1
        self._cursor = 0
        self._executing = True
        return "ORIGINAL" if self._calls == 1 else "REPLAY"

    def original(
        self,
        scope,
        *,
        route,
        bound,
        receipts,
        weight_epoch,
        cache_generation,
        compact_operands,
        leases,
        events,
        producer_frontier,
        post_f_status,
    ):
        if (
            not self._executing
            or self._calls != 1
            or any(r["scope"] == scope for r in self._records)
        ):
            raise Mxfp4StateError("RECOMPUTE_CONTRACT_VIOLATION")
        # Explicit slots prevent retaining a BF16 activation/weight bank accidentally.
        if set(compact_operands) != {
            "x_row",
            "x_row_scales",
            "x_h16t",
            "x_h16t_scales",
        }:
            raise Mxfp4StateError(
                "RECOMPUTE_CONTRACT_VIOLATION", "both packed views required"
            )
        if any(str(t.dtype) != "torch.uint8" for t in compact_operands.values()):
            raise Mxfp4StateError("RECOMPUTE_CONTRACT_VIOLATION", "compact uint8 only")
        if (
            type(producer_frontier) is not tuple
            or not producer_frontier
            or type(post_f_status) is not int
            or post_f_status != 0
        ):
            raise Mxfp4StateError(
                "RECOMPUTE_CONTRACT_VIOLATION", "post-F zero snapshot/frontier required"
            )
        for value in producer_frontier:
            uint(value, "producer frontier")
        self._records.append(
            {
                "scope": scope,
                "route": route,
                "bound": bound,
                "receipts": receipts,
                "producer_frontier": producer_frontier,
                "post_f_status": post_f_status,
                "weight_epoch": weight_epoch,
                "cache_generation": cache_generation,
                "compact_operands": dict(compact_operands),
                "leases": tuple(leases),
                "events": tuple(events),
            }
        )
        self._cursor += 1

    def replay(
        self, scope, *, counts, weight_epoch, cache_generation, producer_frontier
    ):
        if (
            not self._executing
            or self._calls != 2
            or self._cursor >= len(self._records)
        ):
            raise Mxfp4StateError("RECOMPUTE_CONTRACT_VIOLATION")
        record = self._records[self._cursor]
        if (
            type(producer_frontier) is not tuple
            or producer_frontier != record["producer_frontier"]
            or scope != record["scope"]
            or tuple(counts) != record["route"].counts
            or weight_epoch != record["weight_epoch"]
            or cache_generation != record["cache_generation"]
            or not completed_events(record["events"])
        ):
            raise Mxfp4StateError(
                "RECOMPUTE_CONTRACT_VIOLATION", "scope/route/epoch/producer"
            )
        self._cursor += 1
        # Identity-preserving payload; caller still validates framework token ordering.
        return dict(record)

    def end(self):
        if not self._executing or self._cursor != len(self._records):
            raise Mxfp4StateError("RECOMPUTE_CONTRACT_VIOLATION", "cursor mismatch")
        self._executing = False

    def claim_backward(self, scope, fc):
        if (
            self._executing
            or self._retired
            or (scope, fc) in self._backward
            or fc not in (1, 2)
            or not any(r["scope"] == scope for r in self._records)
        ):
            raise Mxfp4StateError("DUPLICATE_WGRAD_OR_INVALID_SCOPE")
        self._backward.add((scope, fc))

    def retire(self, *, consumer_events):
        if (
            self._executing
            or self._retired
            or len(self._backward) != 2 * len(self._records)
        ):
            raise Mxfp4StateError("CACHE_LIFETIME_MISMATCH")
        if not completed_events(consumer_events):
            return False
        self._records.clear()
        self._retired = True
        return True
