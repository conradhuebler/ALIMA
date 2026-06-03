"""Tests for src/core/state_bus.AlimaStateBus (P-δ.1 + thread safety). Claude Generated."""

from __future__ import annotations

import re
import threading
import unittest

from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import QApplication

# pyqtSignal needs a QApplication before the class body runs
_qapp = QApplication.instance() or QApplication([])

from src.core.state_bus import AlimaStateBus, reset
from src.core.agents.sub_agents.caching_tool_registry import make_tool_call_id


class TestAlimaStateBus(unittest.TestCase):
    def setUp(self):
        reset()
        self.bus = AlimaStateBus()

    def tearDown(self):
        reset()

    def test_singleton_returns_same_instance(self):
        self.assertIs(AlimaStateBus(), self.bus)

    def test_single_subscriber_receives_emit(self):
        """QueuedConnection: subscriber fires after ``processEvents``."""
        received = []
        self.bus.subscribe("state.changed", lambda d: received.append(d))
        self.bus.emit_event("state.changed", {"op": "noop"})
        QCoreApplication.processEvents()
        self.assertEqual(received, [{"op": "noop"}])

    def test_multi_subscriber_dispatch_order(self):
        order = []
        self.bus.subscribe("evt", lambda d: order.append(("a", d)))
        self.bus.subscribe("evt", lambda d: order.append(("b", d)))
        self.bus.emit_event("evt", {"x": 1})
        QCoreApplication.processEvents()
        # Order is preserved per subscription call (FIFO connect order).
        self.assertEqual(
            [k for k, _ in order], ["a", "b"]
        )
        self.assertTrue(all(payload == {"x": 1} for _, payload in order))

    def test_unsubscribe_stops_delivery(self):
        hits = []

        def handler(d):
            hits.append(d)

        self.bus.subscribe("evt", handler)
        self.bus.emit_event("evt", {"i": 1})
        QCoreApplication.processEvents()
        self.bus.unsubscribe("evt", handler)
        self.bus.emit_event("evt", {"i": 2})
        QCoreApplication.processEvents()
        self.assertEqual(hits, [{"i": 1}])

    def test_handler_exception_does_not_abort_dispatch(self):
        survived = []
        self.bus.subscribe("evt", lambda d: (_ for _ in ()).throw(RuntimeError("boom")))
        self.bus.subscribe("evt", lambda d: survived.append(d))
        self.bus.emit_event("evt", {"k": "v"})
        QCoreApplication.processEvents()
        self.assertEqual(survived, [{"k": "v"}])

    def test_emit_validates_inputs(self):
        with self.assertRaises(ValueError):
            self.bus.emit_event("", {})
        with self.assertRaises(TypeError):
            self.bus.emit_event("evt", "not a dict")

    def test_idempotent_subscribe(self):
        """Subscribing the same (event_type, handler) twice does not
        cause double-dispatch — the second call is a no-op."""
        hits: list[dict] = []
        handler = lambda d: hits.append(d)  # noqa: E731
        self.bus.subscribe("evt", handler)
        self.bus.subscribe("evt", handler)
        self.bus.emit_event("evt", {"x": 1})
        QCoreApplication.processEvents()
        self.assertEqual(len(hits), 1)

    def test_cross_thread_dispatch_runs_on_gui_thread(self):
        """Worker-thread ``emit_event`` delivers the slot on the GUI
        thread (where the bus QObject lives). This is the property
        that makes widget-manipulating subscribers safe.

        We assert it by capturing the current thread id from inside
        the handler: a queued delivery will show the GUI thread, not
        the worker thread.
        """
        gui_tid = threading.get_ident()
        captured: dict = {}

        def handler(d):
            captured["tid"] = threading.get_ident()
            captured["payload"] = d

        self.bus.subscribe("cross_thread_evt", handler)
        result: dict = {}

        def worker():
            self.bus.emit_event("cross_thread_evt", {"src": "worker"})
            result["emit_done"] = True

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        # Drain the queued event(s) on the GUI thread.
        QCoreApplication.processEvents()
        self.assertTrue(result.get("emit_done"))
        self.assertEqual(captured.get("payload"), {"src": "worker"})
        self.assertEqual(captured.get("tid"), gui_tid)


class TestMakeToolCallId(unittest.TestCase):
    """Phase A: ``make_tool_call_id`` is unique, short, and ``tc_`` prefixed."""

    def test_format(self):
        tc_id = make_tool_call_id()
        self.assertTrue(tc_id.startswith("tc_"))
        # 12 hex chars after the prefix.
        self.assertRegex(tc_id, r"^tc_[0-9a-f]{12}$")

    def test_uniqueness_1000_calls(self):
        ids = {make_tool_call_id() for _ in range(1000)}
        self.assertEqual(len(ids), 1000)


class TestNoDeadEvents(unittest.TestCase):
    """Phase F: every emitted event must have ≥ 1 subscriber.

    Catches future drift where a new event type is added but no panel
    wires up a handler. The list below enumerates the events this repo
    emits; any new addition should be reflected here.
    """

    def setUp(self):
        reset()
        self.bus = AlimaStateBus()

    def tearDown(self):
        reset()

    def test_pipeline_started_and_completed_have_consumer(self):
        """Phase F: the panel subscribes to both lifecycle events.

        We don't instantiate the panel (that would require the full
        QApplication + MainWindow); instead we read the panel's source
        for the subscription lines. Catches the common "the
        subscription got deleted in a refactor" failure mode.
        """
        import inspect
        from src.ui import pipeline_chat_panel

        source = inspect.getsource(pipeline_chat_panel)
        self.assertIn(
            "state.pipeline_started",
            source,
            "PipelineChatPanel must reference state.pipeline_started",
        )
        self.assertIn(
            "state.pipeline_completed",
            source,
            "PipelineChatPanel must reference state.pipeline_completed",
        )


if __name__ == "__main__":
    unittest.main()
