"""Tests for src/core/state_bus.AlimaStateBus (P-δ.1). Claude Generated."""

from __future__ import annotations

import unittest

from PyQt6.QtCore import QCoreApplication

# pyqtSignal needs a QApplication before the class body runs
_qapp = QCoreApplication.instance() or QCoreApplication([])

from src.core.state_bus import AlimaStateBus, reset


class TestAlimaStateBus(unittest.TestCase):
    def setUp(self):
        reset()
        self.bus = AlimaStateBus()

    def tearDown(self):
        reset()

    def test_singleton_returns_same_instance(self):
        self.assertIs(AlimaStateBus(), self.bus)

    def test_single_subscriber_receives_emit(self):
        received = []
        self.bus.subscribe("state.changed", lambda d: received.append(d))
        self.bus.emit_event("state.changed", {"op": "noop"})
        self.assertEqual(received, [{"op": "noop"}])

    def test_multi_subscriber_dispatch_order(self):
        order = []
        self.bus.subscribe("evt", lambda d: order.append(("a", d)))
        self.bus.subscribe("evt", lambda d: order.append(("b", d)))
        self.bus.emit_event("evt", {"x": 1})
        self.assertEqual(
            order, [("a", {"x": 1}), ("b", {"x": 1})]
        )

    def test_unsubscribe_stops_delivery(self):
        hits = []

        def handler(d):
            hits.append(d)

        self.bus.subscribe("evt", handler)
        self.bus.emit_event("evt", {"i": 1})
        self.bus.unsubscribe("evt", handler)
        self.bus.emit_event("evt", {"i": 2})
        self.assertEqual(hits, [{"i": 1}])

    def test_handler_exception_does_not_abort_dispatch(self):
        survived = []
        self.bus.subscribe("evt", lambda d: (_ for _ in ()).throw(RuntimeError("boom")))
        self.bus.subscribe("evt", lambda d: survived.append(d))
        self.bus.emit_event("evt", {"k": "v"})
        self.assertEqual(survived, [{"k": "v"}])

    def test_emit_validates_inputs(self):
        with self.assertRaises(ValueError):
            self.bus.emit_event("", {})
        with self.assertRaises(TypeError):
            self.bus.emit_event("evt", "not a dict")


if __name__ == "__main__":
    unittest.main()
