"""Repetition warning bar: the idle state is really hidden.

Im Ruhezustand war die Leiste nur transparent gemalt: Emoji blieben sichtbar
(sie ignorieren "color: transparent") und der Abbrechen-Knopf war unsichtbar
klickbar. Eigene Datei, weil test_repetition_warning_bar.py QtWebEngine
importiert und danach keine QApplication mehr angelegt werden darf.
Claude Generated.
"""
from __future__ import annotations

import unittest


class HiddenStateIsReallyHiddenTest(unittest.TestCase):

    def _bar(self):
        from PyQt6.QtWidgets import QApplication

        from src.ui.repetition_warning_bar import RepetitionWarningBar

        app = QApplication.instance() or QApplication(["alima-tests"])
        bar = RepetitionWarningBar()
        bar.show()
        app.processEvents()
        self._keep = (app, bar)
        return bar

    def test_idle_bar_shows_no_text_and_no_clickable_button(self):
        bar = self._bar()
        self.assertEqual(bar.warning_icon_label.text(), "")
        for btn in (bar.abort_now_button, bar.continue_button, bar.dismiss_warning_button):
            self.assertFalse(btn.isVisible())

    def test_a_warning_makes_the_buttons_visible_again(self):
        bar = self._bar()
        bar.show_warning("char_repetition", "Details", [])
        self.assertTrue(bar.abort_now_button.isVisible())
        self.assertTrue(bar.continue_button.isVisible())

    def test_hide_after_a_warning_hides_the_buttons_again(self):
        bar = self._bar()
        bar._set_buttons_visible(True)
        bar.hide_warning()
        self.assertFalse(bar.abort_now_button.isVisible())


if __name__ == "__main__":
    unittest.main()
