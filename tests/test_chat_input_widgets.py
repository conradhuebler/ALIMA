"""Tests for the extracted chat input leaf widgets (F-5 split). Claude Generated.

``SystemPromptDialog`` and ``ChatInputEdit`` were moved verbatim out of
``pipeline_chat_panel.py`` into ``chat_input_widgets.py``. These widgets are
light enough to construct under the shared offscreen QApplication (no
QWebEngineView), so we exercise the small pure-logic surface directly:

- the module imports and the classes are re-exported from the panel module
  (back-compat: ``pipeline_chat_panel`` still imports them);
- ``SystemPromptDialog.get_prompt`` round-trips and strips whitespace;
- ``ChatInputEdit`` exposes the ``submit`` signal and the overlay-button API.

The full Enter/Shift+Enter keyPress behavior is GUI-gated (operator click-test).
"""
from __future__ import annotations

import unittest

from src.ui.chat_input_widgets import ChatInputEdit, SystemPromptDialog


class TestModuleSurface(unittest.TestCase):

    def test_panel_reexports_the_widgets(self):
        # pipeline_chat_panel imports both names; the import must resolve to the
        # same classes so existing callers keep working after the split.
        from src.ui import pipeline_chat_panel as p

        self.assertIs(p.SystemPromptDialog, SystemPromptDialog)
        self.assertIs(p.ChatInputEdit, ChatInputEdit)


class TestSystemPromptDialog(unittest.TestCase):

    def test_get_prompt_round_trips_and_strips(self):
        dlg = SystemPromptDialog("  hello prompt  ")
        try:
            self.assertEqual(dlg.get_prompt(), "hello prompt")
            dlg.editor.setPlainText("changed\n")
            self.assertEqual(dlg.get_prompt(), "changed")
        finally:
            dlg.deleteLater()


class TestChatInputEdit(unittest.TestCase):

    def test_submit_signal_and_overlay_api(self):
        edit = ChatInputEdit()
        try:
            fired = []
            edit.submit.connect(lambda: fired.append(True))
            edit.submit.emit()
            self.assertEqual(fired, [True])
            # Overlay button API stores the button without raising.
            from PyQt6.QtWidgets import QPushButton

            btn = QPushButton("x")
            edit.set_overlay_button(btn, margin=4)
            self.assertIs(edit._overlay_btn, btn)
            self.assertEqual(edit._overlay_margin, 4)
        finally:
            edit.deleteLater()


if __name__ == "__main__":
    unittest.main()
