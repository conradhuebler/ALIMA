"""Input area of the pipeline tab: compact, no clipped text, no lost '&'.

* A '&' followed by a space in a button/tab/group title is Qt's mnemonic
  marker and swallows the ampersand ("Drag & Drop" showed as "Drag_Drop").
* The two drop zones and the DOI field must keep their width in the narrow
  left column; the webcam switch must hide only the webcam zone.
Claude Generated.
"""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parent.parent
_kept: list = []

MNEMONIC_WIDGETS = {
    "QGroupBox", "QPushButton", "QCheckBox", "QRadioButton", "QToolButton",
    "QAction", "addTab", "setTabText", "setTitle", "addMenu", "addAction",
}
LONE_AMPERSAND_BEFORE_SPACE = re.compile(r"(?<!&)&(?!&)\s")


class NoSwallowedAmpersandTest(unittest.TestCase):
    def test_widget_titles_escape_a_literal_ampersand(self):
        offenders = []
        for path in sorted((REPO / "src").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name not in MNEMONIC_WIDGETS:
                    continue
                for arg in node.args:
                    if (isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                            and LONE_AMPERSAND_BEFORE_SPACE.search(arg.value)):
                        offenders.append(f"{path.relative_to(REPO)}:{node.lineno}: {arg.value!r}")
        self.assertEqual(offenders, [], "use '&&' for a literal ampersand")


def _qapp():
    from PyQt6.QtWidgets import QApplication

    if not _kept:
        _kept.append(QApplication.instance() or QApplication(["alima-tests"]))
    return _kept[0]


def _widget(webcam: bool):
    from src.ui.unified_input_widget import UnifiedInputWidget

    _qapp()
    ui = MagicMock()
    ui.enable_webcam_input = webcam
    with patch("src.utils.config_manager.ConfigManager.get_ui_config", return_value=ui):
        w = UnifiedInputWidget()
    w.resize(430, 700)
    w.show()
    _qapp().processEvents()
    _kept.append(w)
    return w


class InputAreaLayoutTest(unittest.TestCase):
    def test_no_group_boxes_repeat_what_the_zones_say(self):
        from PyQt6.QtWidgets import QGroupBox

        self.assertEqual(_widget(True).findChildren(QGroupBox), [])

    def test_the_doi_field_keeps_a_usable_width_in_a_narrow_column(self):
        w = _widget(True)
        self.assertGreaterEqual(w.doi_url_input.width(), 160)

    def test_both_zones_are_shown_with_the_webcam_enabled(self):
        w = _widget(True)
        self.assertFalse(w.drop_frame.isHidden())
        self.assertFalse(w.webcam_frame.isHidden())

    def test_switching_the_webcam_off_hides_only_its_zone(self):
        """The old code hid the *parent* of the webcam frame; with the group box
        gone that would have been the whole input area."""
        w = _widget(False)
        self.assertTrue(w.webcam_frame.isHidden())
        self.assertFalse(w.drop_frame.isHidden())
        self.assertFalse(w.doi_url_input.isHidden())
        self.assertFalse(w.text_display.isHidden())

    def test_zones_stay_compact_and_the_text_field_gets_the_height(self):
        w = _widget(True)
        self.assertLessEqual(w.drop_frame.height(), 110)
        self.assertGreater(w.text_display.height(), 3 * w.drop_frame.height())

    def test_zone_captions_are_not_clipped(self):
        """The zone padding used to be 20 px a side: 'PDF, Bilder, Text-Dateier'."""
        from PyQt6.QtWidgets import QLabel

        w = _widget(True)
        for frame in (w.drop_frame, w.webcam_frame):
            for label in frame.findChildren(QLabel):
                fm = label.fontMetrics()
                needed = fm.horizontalAdvance(label.text())
                self.assertTrue(label.wordWrap() or label.width() >= needed, label.text())

    def test_clear_button_sits_with_the_text_and_clears(self):
        from PyQt6.QtWidgets import QPushButton

        w = _widget(True)
        button = next(b for b in w.findChildren(QPushButton) if "Leeren" in b.text())
        # it lives in the text section, below the input methods
        self.assertGreater(button.mapTo(w, button.rect().topLeft()).y(),
                           w.doi_url_input.mapTo(w, w.doi_url_input.rect().topLeft()).y())
        got = []
        w.input_cleared.connect(lambda: got.append(True))
        w.text_display.setPlainText("etwas")
        button.click()
        self.assertEqual(w.text_display.toPlainText(), "")
        self.assertTrue(got)


if __name__ == "__main__":
    unittest.main()
