"""Fensterspion (ALIMA_TRACE_WINDOWS). Claude Generated."""

from __future__ import annotations

import logging
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QDialog, QLabel, QMenu, QPushButton, QWidget

from src.ui import window_trace

_app = QApplication.instance() or QApplication(["test"])


class TraceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tracer = window_trace.install(_app)

    @classmethod
    def tearDownClass(cls):
        _app.removeEventFilter(cls.tracer)

    def _shown(self, make):
        with self.assertLogs("alima.window_trace", level="WARNING") as cm:
            widget = make()
            widget.show()
            _app.processEvents()
            widget.hide()
            # assertLogs verlangt mindestens eine Meldung; für "keine" siehe _silent
        return cm.output

    def _silent(self, make):
        logger = logging.getLogger("alima.window_trace")
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        logger.addHandler(handler)
        try:
            widget = make()
            widget.show()
            _app.processEvents()
            widget.hide()
        finally:
            logger.removeHandler(handler)
        return records

    def test_a_parentless_label_that_gets_shown_is_reported_with_the_call_site(self):
        out = self._shown(lambda: QLabel("Max. Ergebnisse:"))
        self.assertEqual(len(out), 1)
        self.assertIn("FREMDES FENSTER", out[0])
        self.assertIn("QLabel", out[0])
        self.assertIn("Max. Ergebnisse:", out[0])
        self.assertIn("test_window_trace.py", out[0])          # Aufrufweg zeigt die show()-Stelle

    def test_a_parentless_button_is_reported(self):
        self.assertEqual(len(self._shown(lambda: QPushButton("Löschen"))), 1)

    def test_a_child_widget_is_not_a_window_and_stays_silent(self):
        parent = QWidget()
        records = self._silent(lambda: QLabel("im Layout", parent))
        self.assertEqual(records, [])

    def test_dialogs_and_menus_are_expected_windows(self):
        self.assertEqual(self._silent(QDialog), [])
        self.assertEqual(self._silent(QMenu), [])

    def test_the_gui_installs_it_only_with_the_environment_variable(self):
        import pathlib

        src = pathlib.Path("src/alima_gui.py").read_text(encoding="utf-8")
        self.assertRegex(src, r'if os\.environ\.get\("ALIMA_TRACE_WINDOWS"\):\s+from src\.ui\.window_trace import')


if __name__ == "__main__":
    unittest.main()
