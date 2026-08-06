"""AlimaStateBus delivery in hosts without a running Qt event loop. Claude Generated.

The webapp imports Qt but never calls ``exec()``. ``DatabaseManager`` creates a
QCoreApplication for QtSql, so the main thread HAS an event dispatcher while no
loop spins. Under the old dispatcher-only probe, every event emitted from the
pipeline worker thread was queued for that dead loop and silently dropped — the
webapp log lost all bus-driven chrome (pipeline steps, agentic prompts, tool
calls) and only showed what direct callbacks rendered.
"""
from __future__ import annotations

import os
import sys
import threading
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QAbstractEventDispatcher, QCoreApplication

from src.core import state_bus as sb


class TestDirectDispatch(unittest.TestCase):

    def setUp(self):
        # A QCoreApplication without exec() — exactly the webapp's situation.
        self._app = QCoreApplication.instance() or QCoreApplication(sys.argv)
        sb.reset()
        sb.set_direct_dispatch(False)
        self.bus = sb.AlimaStateBus()  # created on the main thread
        self.received: list = []
        self.bus.subscribe("state.pipeline_step", self.received.append)

    def tearDown(self):
        sb.set_direct_dispatch(False)
        sb.reset()

    def _emit_from_worker(self):
        t = threading.Thread(
            target=lambda: self.bus.emit_event(
                "state.pipeline_step", {"step_id": "search", "status": "running"}
            )
        )
        t.start()
        t.join()

    def test_precondition_dispatcher_exists_without_loop(self):
        """Guards the premise: the probe alone can't tell a dead loop apart."""
        self.assertIsNotNone(QAbstractEventDispatcher.instance(self.bus.thread()))

    def test_queued_delivery_is_lost_without_a_running_loop(self):
        self._emit_from_worker()
        self.assertEqual(self.received, [])

    def test_direct_dispatch_delivers_cross_thread_events(self):
        sb.set_direct_dispatch(True)
        self._emit_from_worker()
        self.assertEqual(
            self.received, [{"step_id": "search", "status": "running"}]
        )

    def test_same_thread_emit_unaffected(self):
        self.bus.emit_event("state.pipeline_step", {"step_id": "x", "status": "completed"})
        self.assertEqual(len(self.received), 1)


class TestWebappEnablesDirectDispatch(unittest.TestCase):
    """The webapp lifespan must turn it on — otherwise the log stays empty."""

    def test_lifespan_sets_direct_dispatch(self):
        from fastapi.testclient import TestClient
        from src.webapp import app as appmod

        sb.set_direct_dispatch(False)
        try:
            with TestClient(appmod.app):
                self.assertTrue(sb._force_direct_dispatch)
        finally:
            sb.set_direct_dispatch(False)


if __name__ == "__main__":
    unittest.main()
