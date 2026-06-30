"""Leaf input widgets for the unified pipeline + chat panel. Claude Generated.

Extracted verbatim from ``pipeline_chat_panel.py`` (F-5 god-file split):

- ``SystemPromptDialog`` — small dialog for editing the assistant system prompt
  (was previously in the retired ``chat_widget.py``).
- ``ChatInputEdit`` — multi-line chat input (Enter = submit, Shift+Enter =
  newline) with an optional floating overlay send button.

Both are self-contained (no coupling back to ``PipelineChatPanel``); the panel
imports and embeds them.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTextEdit,
    QVBoxLayout,
)

from .styles import (
    LAYOUT,
    get_main_stylesheet,
    get_scaled_font,
)


# ----------------------------------------------------------------------
# System-Prompt edit dialog (was previously in chat_widget.py).
# ----------------------------------------------------------------------


class SystemPromptDialog(QDialog):
    """Small dialog for editing the assistant system prompt."""

    def __init__(self, current_prompt: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ System-Prompt bearbeiten")
        self.setMinimumSize(500, 300)
        self.setStyleSheet(get_main_stylesheet())

        layout = QVBoxLayout(self)
        m = LAYOUT.get("margin", 12)
        layout.setContentsMargins(m, m, m, m)
        layout.setSpacing(LAYOUT.get("spacing", 8))

        info = QLabel(
            "Dieser Prompt definiert die Rolle des Assistenten. "
            "Änderungen wirken sich sofort auf die nächste Nachricht aus."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(info)

        self.editor = QTextEdit()
        self.editor.setPlainText(current_prompt)
        self.editor.setFont(get_scaled_font(size_delta=-1, monospace=False))
        layout.addWidget(self.editor)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_prompt(self) -> str:
        return self.editor.toPlainText().strip()


# ----------------------------------------------------------------------
# ChatInputEdit — multi-line QTextEdit, Enter sends, Shift+Enter newline.
# ----------------------------------------------------------------------


class ChatInputEdit(QTextEdit):
    """Multi-line chat input. Enter = submit, Shift+Enter = newline.

    Optionally hosts a floating overlay button (the send icon) pinned to the
    bottom-right corner of the field and repositioned on every resize, for a
    modern chat-composer look. Claude Generated.
    """

    submit = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._overlay_btn = None
        self._overlay_margin = 8

    def set_overlay_button(self, button, margin: int = 8) -> None:
        """Pin ``button`` to the bottom-right corner of this field."""
        self._overlay_btn = button
        self._overlay_margin = margin
        button.setParent(self)
        button.raise_()
        self._reposition_overlay()

    def _reposition_overlay(self) -> None:
        btn = self._overlay_btn
        if btn is None:
            return
        m = self._overlay_margin
        btn.move(self.width() - btn.width() - m, self.height() - btn.height() - m)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_overlay()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
                return
            self.submit.emit()
            return
        super().keyPressEvent(event)
