"""Host allocator tests; these do not certify GPU Philox/quantizer output."""

import hashlib
import unittest

from lumen.quantize.mxfp4_moe_rng import (
    PhiloxDrawAllocator,
    RoutingMetadataV1,
    ScopeKey,
    domain_seed,
    limbs,
)
from lumen.utils.mxfp4_moe_ledger import UINT64_MAX, Mxfp4StateError


class _Done:
    def query(self):
        return True


def _route(prefix, experts=32):
    counts = tuple(prefix) + (0,) * (experts - len(prefix))
    return RoutingMetadataV1(counts, sum(counts), (100,) * (256 // experts))


def _preflight(scope, route):
    pass


class TestMxfp4MoeRng(unittest.TestCase):
    def test_all_signed_ragged_counts_and_noncoercing_wire_inputs(self):
        for experts in (32, 64):
            for count in (0, 1, 15, 16, 31, 32, 33, 63, 64, 127, 128, 129, 257):
                with self.subTest(experts=experts, count=count):
                    rng = PhiloxDrawAllocator(0, 0)
                    route = _route([count], experts)
                    scope = ScopeKey(0, 0, 0, 0, 0)
                    rng.declare((scope,))
                    rng.close_scope(scope, route, 0, preflight=_preflight)
                    receipts = rng.reserve_next()
                    self.assertEqual(receipts[0].draws_used, count * 4096)
                    self.assertEqual(
                        receipts[1].draws_used, ((count + 31) // 32) * 32 * 4096
                    )
                    with self.assertRaises(Mxfp4StateError):
                        receipts[0].expert_wire(True)
                    with self.assertRaises(Mxfp4StateError):
                        route.prefixes(True)

    def test_seed_bytes_and_limb_high_bits(self):
        # Independent construction of the signed byte preimage (no struct helper).
        preimage = (
            b"lumen.philox_logical_draw.v1\0"
            + (0xFEDCBA9876543210).to_bytes(8, "little")
            + (7).to_bytes(4, "little")
            + (2).to_bytes(4, "little")
            + b"\1\0\0\0"
        )
        expected = int.from_bytes(hashlib.sha256(preimage).digest()[:8], "little")
        self.assertEqual(domain_seed(0xFEDCBA9876543210, 7, 2), expected)
        self.assertNotEqual(expected, domain_seed(0xFEDCBA9876543210, 7, 1))
        self.assertEqual(limbs(UINT64_MAX), (0xFFFFFFFF, 0xFFFFFFFF))
        for bad in (True, -1, 1.0, UINT64_MAX + 1):
            with self.assertRaises(Mxfp4StateError):
                domain_seed(bad, 0, 1)

    def _two_layers(self, experts, reverse):
        allocator = PhiloxDrawAllocator(123, 0)
        scopes = (ScopeKey(0, 0, 0, 0, 0), ScopeKey(0, 0, 0, 1, 1))
        routes = (_route([0, 1, 33], experts), _route([17, 0, 2], experts))
        allocator.declare(reversed(scopes))
        order = (1, 0) if reverse else (0, 1)
        for i in order:
            allocator.close_scope(scopes[i], routes[i], 0, preflight=_preflight)
            if reverse and i == 1:
                self.assertIsNone(allocator.reserve_next())
                self.assertEqual(allocator.next_draw, {1: 0, 2: 0})
        tokens = allocator.reserve_next() + allocator.reserve_next()
        self.assertEqual(
            [(t.base_draw, t.draws_reserved) for t in tokens],
            [
                (0, 139264),
                (0, 393216),
                (139264, 139264),
                (393216, 393216),
                (278528, 77824),
                (786432, 262144),
                (356352, 77824),
                (1048576, 262144),
            ],
        )
        self.assertEqual(allocator.next_draw, {1: 434176, 2: 1310720})
        before = allocator.next_draw
        for i in (1, 0):
            original = allocator.receipts(scopes[i], route=routes[i], weight_epoch=0)
            self.assertIs(original[0].route, routes[i])
            self.assertIs(
                allocator.receipts(scopes[i], route=routes[i], weight_epoch=0), original
            )
        self.assertEqual(before, allocator.next_draw)
        with self.assertRaises(Mxfp4StateError):
            allocator.receipts(
                scopes[0], route=_route([0, 1, 33], experts), weight_epoch=0
            )
        for scope in scopes:
            allocator.retire_scope(scope, events=(_Done(),))
        return allocator.checkpoint(), [t.record() for t in tokens]

    def test_two_layer_order_invariance_and_checkpoint(self):
        for experts in (32, 64):
            with self.subTest(experts=experts):
                forward = self._two_layers(experts, False)
                reverse = self._two_layers(experts, True)
                self.assertEqual(forward, reverse)
                restored = PhiloxDrawAllocator.restore(forward[0])
                self.assertEqual(restored.checkpoint(), forward[0])
                next_scope = ScopeKey(0, 1, 0, 0, 2)
                restored.declare((next_scope,))
                restored.close_scope(
                    next_scope, _route([1], experts), 0, preflight=_preflight
                )
                self.assertEqual(restored.reserve_next()[0].base_draw, 434176)

    def test_element_indices_and_wire(self):
        allocator = PhiloxDrawAllocator(4, 1)
        scope = ScopeKey(0, 0, 0, 0, 0)
        route = _route([0, 1, 33])
        allocator.declare((scope,))
        allocator.close_scope(scope, route, 0, preflight=_preflight)
        d, w, _, _ = allocator.reserve_next()
        self.assertEqual(d.logical_index(2, 32, 4095), 139263)
        self.assertEqual(w.logical_index(2, 63, 4095), 393215)
        self.assertEqual(w.logical_index(1, 31, 0), 31)  # post-H pad consumes draws
        self.assertEqual(d.expert_wire(2)[3:], (4096, 0, 135168, 0))
        self.assertEqual(w.expert_wire(2)[3:], (131072, 0, 262144, 0))
        self.assertEqual(d.expert_wire(0)[5:], (0, 0))
        with self.assertRaises(Mxfp4StateError):
            d.logical_index(0, 0, 0)

    def test_preflight_failure_consumes_nothing(self):
        allocator = PhiloxDrawAllocator(0, 0)
        scope = ScopeKey(0, 0, 0, 0, 0)
        allocator.declare((scope,))

        def rejected(*args):
            raise ValueError("FC2 static miss")

        with self.assertRaises(ValueError):
            allocator.close_scope(scope, _route([1]), 0, preflight=rejected)
        self.assertEqual(allocator.next_draw, {1: 0, 2: 0})
        self.assertIsNone(allocator.reserve_next())
        allocator.close_scope(scope, _route([1]), 0, preflight=_preflight)
        allocator.reserve_next()
        with self.assertRaises(Mxfp4StateError):
            allocator.close_scope(scope, _route([1]), 0, preflight=_preflight)

    def test_overflow_is_atomic_and_empty_exhausted_frontier(self):
        allocator = PhiloxDrawAllocator(0, 0)
        record = allocator.checkpoint()
        record["next_draw"] = {"1": 0, "2": UINT64_MAX}
        allocator = PhiloxDrawAllocator.restore(record)
        scope = ScopeKey(0, 0, 0, 0, 0)
        allocator.declare((scope,))
        allocator.close_scope(scope, _route([1]), 0, preflight=_preflight)
        with self.assertRaises(Mxfp4StateError):
            allocator.reserve_next()
        self.assertEqual(allocator.next_draw, {1: 0, 2: UINT64_MAX})
        self.assertEqual(allocator.frontier, 0)
        empty = PhiloxDrawAllocator.restore(record)
        empty.declare((scope,))
        empty.close_scope(scope, _route([]), 0, preflight=_preflight)
        receipts = empty.reserve_next()
        self.assertEqual(empty.frontier, 1)
        self.assertTrue(
            all(
                t.draws_reserved == t.draws_used == t.draws_unused == 0
                for t in receipts
            )
        )
        self.assertEqual(receipts[1].expert_wire(0)[3:5], (0xFFFFFFFF, 0xFFFFFFFF))

    def test_bad_counts_and_restore_fail_closed(self):
        for bad in (-1, True, 1.5, 1 << 63):
            with self.assertRaises(Mxfp4StateError):
                _route([bad])
        with self.assertRaises(Mxfp4StateError):
            RoutingMetadataV1((1,) + (0,) * 31, 0, (1,) * 8)
        with self.assertRaises(Mxfp4StateError):
            RoutingMetadataV1((9,) + (0,) * 31, 9, (1,) * 8)
        record = PhiloxDrawAllocator(1, 0).checkpoint()
        record["seeds"]["1"] ^= 1
        with self.assertRaises(Mxfp4StateError):
            PhiloxDrawAllocator.restore(record)


if __name__ == "__main__":
    unittest.main()
