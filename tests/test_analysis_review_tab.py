"""Tests for AnalysisReviewTab mini-log (Phase E). Claude Generated.

The tab owns a small read-only ``UnifiedMessageRenderer`` instance that
subscribes to ``AlimaStateBus.tool.called`` / ``tool.result``. This
file exercises the wiring without booting a real MainWindow.

Strategy: instantiate the tab with a ``QApplication`` (setup_ui builds
real QGroupBox/QTextBrowser/QCheckBox widgets that need a full GUI
application, not just a QCoreApplication). The tab's ``setup_ui``
builds the mini-log widget eagerly in ``__init__``, so by the time we
have a handle the renderer is already subscribed. We then emit
synthetic bus events and assert the renderer's history grew.

Test isolation: prior test files in the suite may have created a
``QCoreApplication`` singleton. Real Qt widgets (``QSplitter``,
``QGroupBox``) require a full ``QApplication``. If a QCoreApplication
exists we delete it via ``sip.delete`` so a fresh ``QApplication``
can be created; otherwise Qt aborts with a fatal "QApplication
required" message.
"""
from __future__ import annotations

import unittest

from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import QApplication

# Tear down any pre-existing QCoreApplication so QApplication can take
# its place (Qt forbids two Q*Application instances per process).
_existing = QCoreApplication.instance()
if _existing is not None and not isinstance(_existing, QApplication):
    sip.delete(_existing)

_qapp = QApplication.instance() or QApplication([])

import subprocess
import sys


def _qwebengine_works() -> bool:
    """Probe (in a child process) whether QWebEngineView can initialise.

    AnalysisReviewTab builds a QWebEngineView (Chromium) in init_detail_tabs,
    which hard-aborts (SIGABRT) where Chromium can't start (CI / sandbox / no GPU)
    — and a having a DISPLAY set is not sufficient. We probe in a subprocess so a
    crash is isolated; the parent then skips instead of aborting the whole suite.
    On a real developer machine the probe succeeds and the test runs. - Claude Generated
    """
    code = (
        "from PyQt6.QtWidgets import QApplication;"
        "from PyQt6.QtWebEngineWidgets import QWebEngineView;"
        "app=QApplication([]);v=QWebEngineView();v.setHtml('<html></html>')"
    )
    try:
        return subprocess.run(
            [sys.executable, "-c", code], capture_output=True, timeout=60
        ).returncode == 0
    except Exception:
        return False


_QWEBENGINE_OK = _qwebengine_works()

from src.core import state_bus as state_bus_mod
from src.ui.analysis_review_tab import AnalysisReviewTab


@unittest.skipUnless(_QWEBENGINE_OK, "QWebEngineView cannot initialise here (headless/CI)")
class TestAnalysisReviewTabMiniLog(unittest.TestCase):
    """Phase E: tab's mini-log subscribes to bus tool events."""

    def setUp(self):
        state_bus_mod.reset()
        self.bus = state_bus_mod.AlimaStateBus()
        self.tab = AnalysisReviewTab()

    def tearDown(self):
        try:
            self.tab.close()
        except Exception:
            pass
        state_bus_mod.reset()

    def test_mini_log_renderer_subscribed(self):
        """setup_ui() built the renderer and subscribed to the bus."""
        self.assertTrue(hasattr(self.tab, "tool_log_renderer"))
        self.assertTrue(hasattr(self.tab, "tool_log_browser"))
        # The bus should have at least one subscription for each event type.
        called = [s for s in self.bus._subscriptions if s[0] == "tool.called"]
        result = [s for s in self.bus._subscriptions if s[0] == "tool.result"]
        self.assertGreater(len(called), 0)
        self.assertGreater(len(result), 0)

    def test_synthetic_bus_event_renders_in_mini_log(self):
        """A bus tool.called/result pair ends up in the mini-log renderer."""
        self.bus.emit_event("tool.called", {
            "name": "search_gnd",
            "arguments": {"term": "Bibliothek"},
            "id": "tc_review1",
        })
        self.bus.emit_event("tool.result", {
            "name": "search_gnd",
            "result": '{"hits": 4}',
            "id": "tc_review1",
            "status": "ok",
        })
        # The mini-log renderer's history recorded the tool call.
        history = self.tab.tool_log_renderer.history
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].metadata["tool_name"], "search_gnd")
        # The tool-call block was updated to "success" — the cached tool
        # call entry should now have a status of "success".
        tc = self.tab.tool_log_renderer._tool_calls["tc_1"]
        self.assertEqual(tc["status"], "success")
        self.assertIn("hits", tc["result"])

    def test_close_event_unsubscribes(self):
        """Tab close → renderer unsubscribes from the bus."""
        # Widget is never shown, so close() doesn't fire closeEvent.
        # We exercise the handler directly: that's what we actually
        # want to lock in (the wiring, not Qt's show/close machinery).
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
