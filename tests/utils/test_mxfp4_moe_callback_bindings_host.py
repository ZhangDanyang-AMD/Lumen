"""Invoke the actual fixture callback factories after every creating loop ends."""

import unittest
from types import SimpleNamespace

from tests.mxfp4_callback_observers import (
    capture_probability,
    capture_router_map,
    gradient_emitter,
)


class ObservedValue:
    def __init__(self, name):
        self.name = name
        self.retained = 0

    def retain_grad(self):
        self.retained += 1

    def detach(self):
        return self

    def clone(self):
        return self


class CallbackBindingTests(unittest.TestCase):
    def test_probability_destinations_do_not_follow_the_last_loop_iteration(self):
        destinations, values, callbacks = [], [], []
        for index in range(3):
            target, value = {}, ObservedValue(index)
            destinations.append(target)
            values.append(value)
            callbacks.append(capture_probability(target, index))
        for callback, value in zip(callbacks, values):
            callback(None, (None, None, value))
        for index, (target, value) in enumerate(zip(destinations, values)):
            self.assertEqual(target, {index: value})
            self.assertEqual(value.retained, 1)

    def test_router_maps_retain_their_own_state_dictionary(self):
        targets, values, callbacks = [], [], []
        for index in range(3):
            target, value = {}, ObservedValue(index)
            targets.append(target)
            values.append(value)
            callbacks.append(capture_router_map(target))
        for callback, value in zip(callbacks, values):
            callback(None, (), (None, value))
        self.assertEqual([target["map"] for target in targets], values)

    def test_delayed_gradient_emitters_keep_each_lease(self):
        leases = [object(), object(), object()]
        emitted = []

        def arm(lease):
            emitted.append(lease)
            return lease

        handoff = SimpleNamespace(arm=arm)
        callbacks = [gradient_emitter(handoff, lease) for lease in leases]
        self.assertEqual([callback(None, None) for callback in callbacks], leases)
        self.assertEqual(emitted, leases)


if __name__ == "__main__":
    unittest.main()
