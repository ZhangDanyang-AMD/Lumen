# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0
"""Exact-count, route-ready Philox reservations; no random sampling or GPU calls."""

import hashlib
import struct
from dataclasses import dataclass
from threading import RLock

from lumen.utils.mxfp4_moe_ledger import INT64_MAX, Mxfp4StateError, uint

RNG_SCHEMA = "lumen.philox_logical_draw.v1"
ALLOCATION_MAPPING = "exact_count_route_ready_v1"
PUBLIC_MAPPING = "aiter.mxfp4.philox_element_q32.v1"
DY_ROW_DGRAD = 1
DY_H16T_WGRAD = 2


def limbs(value):
    uint(value, "u64")
    return value & 0xFFFFFFFF, value >> 32


def domain_seed_digest(run_seed, rank, domain):
    uint(run_seed, "run_seed")
    uint(rank, "rank", (1 << 32) - 1)
    if type(domain) is not int or domain not in (1, 2):
        raise Mxfp4StateError("SR_TOKEN_MISMATCH", "domain")
    data = (
        RNG_SCHEMA.encode("ascii")
        + b"\0"
        + struct.pack("<QIII", run_seed, rank, domain, 1)
    )
    return hashlib.sha256(data).hexdigest()


def domain_seed(run_seed, rank, domain):
    return int.from_bytes(
        bytes.fromhex(domain_seed_digest(run_seed, rank, domain))[:8], "little"
    )


@dataclass(frozen=True, order=True)
class ScopeKey:
    rng_epoch: int
    attempted_step: int
    microbatch_ordinal: int
    global_layer_ordinal: int
    logical_invocation_id: int

    def __post_init__(self):
        for name, value in self.__dict__.items():
            uint(value, name)

    def record(self):
        return list(self.__dict__.values())


@dataclass(frozen=True)
class RoutingMetadataV1:
    """Immutable snapshot of already-ready ordinary-AlltoAll CPU counts."""

    counts: tuple
    sorted_rows: int
    sender_rows: tuple
    topk: int = 6
    routing_epoch: int = 0

    def __post_init__(self):
        if type(self.counts) is not tuple or len(self.counts) not in (32, 64):
            raise Mxfp4StateError(
                "ROUTING_SCHEMA_MISMATCH", "E32/E64 CPU tuple required"
            )
        if type(self.sender_rows) is not tuple or len(self.sender_rows) != 256 // len(
            self.counts
        ):
            raise Mxfp4StateError(
                "ROUTING_SCHEMA_MISMATCH", "actual EP sender rows required"
            )
        for value in (
            *self.counts,
            *self.sender_rows,
            self.sorted_rows,
            self.routing_epoch,
        ):
            uint(value, "routing integer", INT64_MAX)
        if type(self.topk) is not int or self.topk != 6:
            raise Mxfp4StateError(
                "ROUTING_SCHEMA_MISMATCH", "signed distinct-topk fixture"
            )
        sender_total = uint(sum(self.sender_rows), "sender total", INT64_MAX)
        total = uint(sum(self.counts), "count total", INT64_MAX)
        cap = uint(
            sender_total * min(self.topk, len(self.counts)), "receiver cap", INT64_MAX
        )
        if (
            total != self.sorted_rows
            or total > cap
            or any(m > sender_total for m in self.counts)
        ):
            raise Mxfp4StateError(
                "CAPACITY_OR_INDEX_OVERFLOW", "post-dispatch counts/envelope"
            )
        uint(sum(self.padded_counts), "padded total", INT64_MAX)

    @property
    def padded_counts(self):
        return tuple(uint((m + 31) // 32 * 32, "pad32", INT64_MAX) for m in self.counts)

    def prefixes(self, domain, width=4096):
        if (
            type(width) is not int
            or width != 4096
            or type(domain) is not int
            or domain not in (1, 2)
        ):
            raise Mxfp4StateError("SR_TOKEN_MISMATCH", "width/domain")
        counts = self.counts if domain == 1 else self.padded_counts
        offsets = [0]
        for count in counts:
            offsets.append(
                uint(offsets[-1] + uint(width * count, "expert draw count"), "prefix")
            )
        return tuple(offsets)


@dataclass(frozen=True)
class SRTokenV1:
    scope: ScopeKey
    fc: int
    domain: int
    seed: int
    global_rank: int
    base_draw: int
    draws_reserved: int
    route: RoutingMetadataV1
    weight_epoch: int
    frontier_before: int
    frontier_after: int
    schema: str = RNG_SCHEMA
    allocation_mapping: str = ALLOCATION_MAPPING
    public_mapping: str = PUBLIC_MAPPING

    def __post_init__(self):
        if not isinstance(self.scope, ScopeKey) or not isinstance(
            self.route, RoutingMetadataV1
        ):
            raise Mxfp4StateError("SR_TOKEN_MISMATCH", "scope/route")
        if (
            type(self.fc) is not int
            or self.fc not in (1, 2)
            or type(self.domain) is not int
            or self.domain not in (1, 2)
        ):
            raise Mxfp4StateError("SR_TOKEN_MISMATCH", "FC/domain")
        for name in (
            "seed",
            "base_draw",
            "draws_reserved",
            "weight_epoch",
            "frontier_before",
            "frontier_after",
        ):
            uint(getattr(self, name), name)
        uint(self.global_rank, "rank", (1 << 32) - 1)
        uint(self.base_draw + self.draws_reserved, "exclusive frontier")
        if (
            self.draws_reserved != self.route.prefixes(self.domain)[-1]
            or self.frontier_after != self.frontier_before + 1
            or self.scope.logical_invocation_id != self.frontier_before
        ):
            raise Mxfp4StateError("SR_TOKEN_MISMATCH", "count/frontier")
        if (self.schema, self.allocation_mapping, self.public_mapping) != (
            RNG_SCHEMA,
            ALLOCATION_MAPPING,
            PUBLIC_MAPPING,
        ):
            raise Mxfp4StateError("SR_TOKEN_MISMATCH", "schema")

    @property
    def draws_used(self):
        return self.draws_reserved

    @property
    def draws_unused(self):
        return 0

    def expert_wire(self, expert):
        """Return exact R[7] for one public dense child, including empty receipts."""
        uint(expert, "expert", len(self.route.counts) - 1)
        offsets = self.route.prefixes(self.domain)
        base = uint(self.base_draw + offsets[expert], "expert base")
        count = offsets[expert + 1] - offsets[expert]
        uint(base + count, "exclusive frontier")
        return (1, *limbs(self.seed), *limbs(base), *limbs(count))

    def logical_index(self, expert, token, feature):
        uint(expert, "expert", len(self.route.counts) - 1)
        uint(feature, "feature", 4095)
        size = (self.route.counts if self.domain == 1 else self.route.padded_counts)[
            expert
        ]
        uint(token, "token", size - 1)
        offset = token * 4096 + feature if self.domain == 1 else feature * size + token
        return uint(self.route.prefixes(self.domain)[expert] + offset, "logical index")

    def record(self):
        return {
            "schema": self.schema,
            "allocation_mapping": self.allocation_mapping,
            "public_mapping": self.public_mapping,
            "scope": self.scope.record(),
            "fc": self.fc,
            "pass_id": self.domain,
            "operand": "dY",
            "domain": self.domain,
            "seed_limbs": list(limbs(self.seed)),
            "global_rank": self.global_rank,
            "base_draw": self.base_draw,
            "draws_reserved": self.draws_reserved,
            "draws_used": self.draws_used,
            "draws_unused": 0,
            "counts": self.route.counts,
            "padded_counts": self.route.padded_counts,
            "prefixes": self.route.prefixes(self.domain),
            "routing_epoch": self.route.routing_epoch,
            "weight_epoch": self.weight_epoch,
            "frontier_before": self.frontier_before,
            "frontier_after": self.frontier_after,
            "training_recipe_id": "lumen_a4w4_direct_training_v1",
            "operand_quant_recipe_id": "lumen_fp4_e2m1_e8m0_scale_block_1x32_v1",
            "physical_layout_id": "mxfp4_segmented_row_and_t_v1",
            "operand_transform_id": (
                "none" if self.domain == 1 else "hadamard16_all_plus_v1"
            ),
        }


class PhiloxDrawAllocator:
    """One rank's two-domain allocator with explicit topological declaration frontier.

    Preflight is caller-owned and must validate both FC candidates, main_grad and
    budgets before close_scope. Receipts remain live until the tape's last consumer.
    """

    def __init__(self, run_seed, global_rank, rng_epoch=0):
        self.run_seed = uint(run_seed, "run seed")
        self.global_rank = uint(global_rank, "rank", (1 << 32) - 1)
        self.rng_epoch = uint(rng_epoch, "rng epoch")
        self._seeds = {d: domain_seed(run_seed, global_rank, d) for d in (1, 2)}
        self._next = {1: 0, 2: 0}
        self.frontier = 0
        self._schedule = []
        self._ready = {}
        self._receipts = {}
        self._last_committed = None
        self._lock = RLock()

    @property
    def next_draw(self):
        return dict(self._next)

    def declare(self, scopes):
        """Register a closed graph block; input declaration order is immaterial."""
        scopes = tuple(sorted(scopes))
        with self._lock:
            if self._schedule or self._ready:
                raise Mxfp4StateError(
                    "DECLARATION_FRONTIER_MISMATCH", "previous block incomplete"
                )
            if len(set(scopes)) != len(scopes):
                raise Mxfp4StateError(
                    "DECLARATION_FRONTIER_MISMATCH", "duplicate scope"
                )
            for i, scope in enumerate(scopes):
                if not isinstance(scope, ScopeKey) or scope.rng_epoch != self.rng_epoch:
                    raise Mxfp4StateError("DECLARATION_FRONTIER_MISMATCH", "epoch")
                if scope.logical_invocation_id != self.frontier + i or (
                    self._last_committed and scope <= self._last_committed
                ):
                    raise Mxfp4StateError(
                        "DECLARATION_FRONTIER_MISMATCH", "logical order"
                    )
            self._schedule = list(scopes)

    def close_scope(self, scope, route, weight_epoch, *, preflight):
        with self._lock:
            if (
                scope not in self._schedule
                or scope in self._ready
                or scope in self._receipts
            ):
                raise Mxfp4StateError(
                    "DECLARATION_FRONTIER_MISMATCH", "unknown/duplicate close"
                )
            if not isinstance(route, RoutingMetadataV1):
                raise Mxfp4StateError("ROUTING_SCHEMA_MISMATCH")
            uint(weight_epoch, "weight epoch")
            if not callable(preflight):
                raise Mxfp4StateError("SR_TOKEN_MISMATCH", "both-FC preflight required")
            preflight(scope, route)
            counts = {d: route.prefixes(d)[-1] for d in (1, 2)}
            self._ready[scope] = (route, weight_epoch, counts)

    def reserve_next(self):
        """Commit only the next ready scope; both domains update all-or-nothing."""
        with self._lock:
            if not self._schedule or self._schedule[0] not in self._ready:
                return None
            scope = self._schedule[0]
            route, epoch, counts = self._ready[scope]
            after = {
                d: uint(self._next[d] + 2 * counts[d], "two-FC frontier")
                for d in (1, 2)
            }
            next_frontier = uint(self.frontier + 1, "declaration frontier")
            tokens = tuple(
                SRTokenV1(
                    scope,
                    fc,
                    d,
                    self._seeds[d],
                    self.global_rank,
                    self._next[d] + (fc - 1) * counts[d],
                    counts[d],
                    route,
                    epoch,
                    self.frontier,
                    next_frontier,
                )
                for fc in (1, 2)
                for d in (1, 2)
            )
            self._next = after
            self.frontier = next_frontier
            self._receipts[scope] = tokens
            self._last_committed = scope
            self._schedule.pop(0)
            del self._ready[scope]
            return tokens

    def receipts(self, scope, *, route, weight_epoch):
        with self._lock:
            if scope not in self._receipts:
                raise Mxfp4StateError("SR_TOKEN_MISMATCH", "no reservation")
            tokens = self._receipts[scope]
            if tokens[0].route is not route or tokens[0].weight_epoch != weight_epoch:
                raise Mxfp4StateError(
                    "SR_TOKEN_MISMATCH", "metadata identity or weight epoch"
                )
            return tokens

    def retire_scope(self, scope, *, events):
        from lumen.utils.mxfp4_moe_ledger import completed_events

        with self._lock:
            if scope not in self._receipts:
                raise Mxfp4StateError("SR_TOKEN_MISMATCH")
            if not completed_events(events):
                return False
            del self._receipts[scope]
            return True

    def checkpoint(self):
        with self._lock:
            if self._schedule or self._ready or self._receipts:
                raise Mxfp4StateError(
                    "REPLAY_FRONTIER_MISMATCH", "open declarations/consumers"
                )
            return {
                "schema": RNG_SCHEMA,
                "mapping": ALLOCATION_MAPPING,
                "run_seed": self.run_seed,
                "global_rank": self.global_rank,
                "rng_epoch": self.rng_epoch,
                "next_draw": {str(d): self._next[d] for d in (1, 2)},
                "seeds": {str(d): self._seeds[d] for d in (1, 2)},
                "seed_derivation_sha256": {
                    str(d): domain_seed_digest(self.run_seed, self.global_rank, d)
                    for d in (1, 2)
                },
                "next_logical_invocation_id": {str(d): self.frontier for d in (1, 2)},
                "declaration_frontier": self.frontier,
                "last_committed_scope": (
                    self._last_committed.record() if self._last_committed else None
                ),
            }

    @classmethod
    def restore(cls, record):
        expected = {
            "schema",
            "mapping",
            "run_seed",
            "global_rank",
            "rng_epoch",
            "next_draw",
            "seeds",
            "seed_derivation_sha256",
            "next_logical_invocation_id",
            "declaration_frontier",
            "last_committed_scope",
        }
        if (
            set(record) != expected
            or record["schema"] != RNG_SCHEMA
            or record["mapping"] != ALLOCATION_MAPPING
        ):
            raise Mxfp4StateError("REPLAY_FRONTIER_MISMATCH", "RNG schema")
        obj = cls(record["run_seed"], record["global_rank"], record["rng_epoch"])
        for field in (
            "seeds",
            "seed_derivation_sha256",
            "next_draw",
            "next_logical_invocation_id",
        ):
            if set(record[field]) != {"1", "2"}:
                raise Mxfp4StateError("REPLAY_FRONTIER_MISMATCH", field)
        obj.frontier = uint(record["declaration_frontier"], "frontier")
        for d in (1, 2):
            if (
                record["seed_derivation_sha256"][str(d)]
                != domain_seed_digest(obj.run_seed, obj.global_rank, d)
                or record["seeds"][str(d)] != obj._seeds[d]
                or record["next_logical_invocation_id"][str(d)] != obj.frontier
            ):
                raise Mxfp4StateError("REPLAY_FRONTIER_MISMATCH", "seed/invocation")
            obj._next[d] = uint(record["next_draw"][str(d)], "next draw")
        if record["last_committed_scope"] is not None:
            obj._last_committed = ScopeKey(*record["last_committed_scope"])
            if (
                obj._last_committed.rng_epoch != obj.rng_epoch
                or obj._last_committed.logical_invocation_id + 1 != obj.frontier
            ):
                raise Mxfp4StateError("REPLAY_FRONTIER_MISMATCH", "last scope")
        elif obj.frontier:
            raise Mxfp4StateError("REPLAY_FRONTIER_MISMATCH", "missing last scope")
        return obj
