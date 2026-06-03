"""Tests for ImageAnalysisTab mini-log (Phase E). Claude Generated.

Mirrors :mod:`tests.test_analysis_review_tab` for the image-analysis
tab — same opt-in bus consumer pattern, same lifecycle (subscribe in
``setup_ui``, unsubscribe in ``closeEvent``).
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import QApplication

# Tear down any pre-existing QCoreApplication so QApplication can take
# its place (Qt forbids two Q*Application instances per process).
_existing = QCoreApplication.instance()
if _existing is not None and not isinstance(_existing, QApplication):
    sip.delete(_existing)

_qapp = QApplication.instance() or QApplication([])

from src.core import state_bus as state_bus_mod
from src.ui.image_analysis_tab import ImageAnalysisTab


class TestImageAnalysisTabMiniLog(unittest.TestCase):
    """Phase E: tab's mini-log subscribes to bus tool events."""

    def setUp(self):
        state_bus_mod.reset()
        self.bus = state_bus_mod.AlimaStateBus()
        # Tab needs an llm_service stub; main_window optional.
        llm_service = MagicMock()
        self.tab = ImageAnalysisTab(llm_service, main_window=None)

    def tearDown(self):
        try:
            self.tab.close()
        except Exception:
            pass
        state_bus_mod.reset()

    def test_mini_log_renderer_subscribed(self):
        self.assertTrue(hasattr(self.tab, "tool_log_renderer"))
        self.assertTrue(hasattr(self.tab, "tool_log_browser"))
        called = [s for s in self.bus._subscriptions if s[0] == "tool.called"]
        result = [s for s in self.bus._subscriptions if s[0] == "tool.result"]
        self.assertGreater(len(called), 0)
        self.assertGreater(len(result), 0)

    def test_synthetic_bus_event_renders_in_mini_log(self):
        self.bus.emit_event("tool.called", {
            "name": "image_caption",
            "arguments": {"path": "x.png"},
            "id": "tc_img1",
        })
        self.bus.emit_event("tool.result", {
            "name": "image_caption",
            "result": "a library",
            "id": "tc_img1",
            "status": "ok",
        })
        history = self.tab.tool_log_renderer.history
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].metadata["tool_name"], "image_caption")
        tc = self.tab.tool_log_renderer._tool_calls["tc_1"]
        self.assertEqual(tc["status"], "success")
        self.assertIn("library", tc["result"])

    def test_close_event_unsubscribes(self):
        from PyQt6.QtGui import QCloseEvent
        self.tab.closeEvent(QCloseEvent())
        found = False
        for event, handler, _slot in self.bus._subscriptions:
            if event != "tool.called":
                continue
            qualname = getattr(handler, "__qualname__", "")
            if "UnifiedMessageRenderer" in qualname:
                found = True
                break
        self.assertFalse(found, "renderer should have unsubscribed on close")


if __name__ == "__main__":
    unittest.main()
