"""Repetition-warning bar for the unified pipeline + chat panel. Claude Generated.

Extracted from ``pipeline_chat_panel.py`` (F-5 god-file split). A self-contained
``QFrame`` that shows the live repetition-detection warning (orange while active,
green once resolved, hidden otherwise), the grace-period countdown, and the
parameter-variation suggestion buttons.

The bar owns its own state machine and emits two signals; the host panel
(``PipelineChatPanel``) re-wires them to its public ``abort_generation_requested``
/ ``retry_with_variations`` signals and logs the retry line (which needs the
panel's ``current_step_id``):

- ``abort_requested()``      — operator pressed 🛑 Abbrechen.
- ``retry_requested(dict)``  — operator picked a suggestion (params payload).

Method bodies are moved verbatim; the only change is ``self.repetition_warning_frame``
→ ``self`` (the bar *is* the frame now).
"""
from __future__ import annotations

import time
from typing import Dict, List

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton


class RepetitionWarningBar(QFrame):
    """Collapsible repetition-warning bar (pipeline-only feature)."""

    abort_requested = pyqtSignal()
    retry_requested = pyqtSignal(dict)

    _STYLE_WARNING_GREEN = """
        QFrame {
            background-color: #1b3a1f;
            border: 1px solid #4caf50;
            border-radius: 3px;
            padding: 1px;
        }
        QLabel { color: #a5d6a7; }
        QPushButton {
            background-color: #2e7d32;
            color: #fff;
            border: none;
            border-radius: 3px;
            padding: 1px 5px;
            font-weight: bold;
        }
        QPushButton:hover { background-color: #43a047; }
    """

    _STYLE_WARNING_ORANGE = """
        QFrame {
            background-color: #3d2a00;
            border: 1px solid #ff9800;
            border-radius: 3px;
            padding: 1px;
        }
        QLabel { color: #ffcc80; }
        QPushButton {
            background-color: #ff9800;
            color: #1e1e1e;
            border: none;
            border-radius: 3px;
            padding: 1px 5px;
            font-weight: bold;
        }
        QPushButton:hover { background-color: #ffb74d; }
    """

    _STYLE_WARNING_HIDDEN = """
        QFrame { background: transparent; border: none; padding: 0; }
        QLabel { color: transparent; }
        QPushButton { background: transparent; border: none; color: transparent; }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self._warning_style_state = "hidden"
        self._last_shown_detection_type = ""
        self.current_suggestions: List[Dict] = []
        self.setStyleSheet(self._STYLE_WARNING_HIDDEN)

        bar = QHBoxLayout(self)
        bar.setContentsMargins(6, 1, 4, 1)
        bar.setSpacing(6)

        self.warning_icon_label = QLabel("⚠️")
        bar.addWidget(self.warning_icon_label)

        self.warning_title_label = QLabel("Wiederholung erkannt")
        self.warning_title_label.setStyleSheet("font-weight: bold; color: #ff9800;")
        bar.addWidget(self.warning_title_label)

        self.warning_details_label = QLabel("")
        self.warning_details_label.setWordWrap(False)
        self.warning_details_label.setStyleSheet("color: #ffe0b2;")
        bar.addWidget(self.warning_details_label, 1)

        self.countdown_label = QLabel("")
        self.countdown_label.setStyleSheet("color: #fff; font-weight: bold;")
        self.countdown_label.setVisible(False)
        bar.addWidget(self.countdown_label)

        self.suggestions_button_layout = QHBoxLayout()
        self.suggestions_button_layout.setSpacing(3)
        bar.addLayout(self.suggestions_button_layout)

        self.abort_now_button = QPushButton("🛑 Abbrechen")
        self.abort_now_button.setStyleSheet(
            "background-color: #d32f2f; color: white; font-weight: bold;"
            " border-radius: 3px; padding: 1px 5px;"
        )
        self.abort_now_button.clicked.connect(self._on_abort_requested)
        bar.addWidget(self.abort_now_button)

        self.continue_button = QPushButton("Fortfahren")
        self.continue_button.setStyleSheet(
            "background-color: #555; color: #ccc; padding: 1px 5px;"
        )
        self.continue_button.clicked.connect(self.hide_warning)
        bar.addWidget(self.continue_button)

        self.dismiss_warning_button = QPushButton("✕")
        self.dismiss_warning_button.setFixedSize(18, 18)
        self.dismiss_warning_button.setStyleSheet(
            "background-color: transparent; color: #ff9800; padding: 0;"
        )
        self.dismiss_warning_button.clicked.connect(self.hide_warning)
        bar.addWidget(self.dismiss_warning_button)

        self.grace_timer = QTimer(self)
        self.grace_timer.timeout.connect(self._update_countdown)
        self.grace_period_end = 0.0

    def show_warning(
        self,
        detection_type: str,
        details: str,
        suggestions: List[Dict],
        grace_period: bool = False,
        grace_seconds: float = 2.0,
    ):
        self.current_suggestions = suggestions
        already_showing = (
            self._warning_style_state == "orange"
            and self._last_shown_detection_type == detection_type
        )
        if self._warning_style_state != "orange":
            self.setStyleSheet(self._STYLE_WARNING_ORANGE)
            self._warning_style_state = "orange"
        self._last_shown_detection_type = detection_type

        if not already_showing:
            self.warning_icon_label.setText("⚠️")
            self.warning_title_label.setStyleSheet(
                "font-weight: bold; color: #ff9800;"
            )
            self.continue_button.setStyleSheet(
                "background-color: #555; color: #ccc; padding: 1px 5px;"
            )

            type_labels = {
                "char_pattern": "Zeichenwiederholung erkannt",
                "ngram": "Phrasenwiederholung erkannt",
                "window_similarity": "Textblock-Wiederholung erkannt",
            }
            self.warning_title_label.setText(
                type_labels.get(detection_type, "Wiederholung erkannt")
            )

            while self.suggestions_button_layout.count():
                item = self.suggestions_button_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            for i, suggestion in enumerate(suggestions[:3]):
                button = QPushButton(suggestion.get("label", f"Option {i+1}"))
                button.setToolTip(suggestion.get("description", ""))
                button.setStyleSheet("padding: 1px 4px;")
                params = suggestion.get("params", {})

                def make_handler(p):
                    return lambda: self._on_suggestion_clicked(p)

                button.clicked.connect(make_handler(params))
                self.suggestions_button_layout.addWidget(button)

        self.warning_details_label.setText(details)

        if grace_period and not already_showing:
            self.grace_period_end = time.time() + grace_seconds
            self.grace_timer.stop()
            self.grace_timer.start(200)
            self.countdown_label.setText(f"⏳ {grace_seconds:.1f}s")
        elif not grace_period:
            self.countdown_label.setText("")
            self.grace_timer.stop()

    def _update_countdown(self):
        remaining = self.grace_period_end - time.time()
        if remaining > 0:
            self.countdown_label.setText(f"⏳ {remaining:.1f}s")
        else:
            self.countdown_label.setText("⏳ …")
            self.grace_timer.stop()

    def hide_warning(self, resolved: bool = False):
        self.grace_timer.stop()
        self.countdown_label.setText("")
        self._last_shown_detection_type = ""

        if resolved:
            if self._warning_style_state != "green":
                self.setStyleSheet(self._STYLE_WARNING_GREEN)
                self._warning_style_state = "green"
            self.warning_icon_label.setText("✅")
            self.warning_title_label.setText(
                "Wiederholung behoben – Generation läuft weiter"
            )
            self.warning_title_label.setStyleSheet(
                "font-weight: bold; color: #4caf50;"
            )
            self.warning_details_label.setText("")
            while self.suggestions_button_layout.count():
                item = self.suggestions_button_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.continue_button.setStyleSheet(
                "background: transparent; border: none; color: transparent;"
            )
        else:
            if self._warning_style_state != "hidden":
                self.setStyleSheet(self._STYLE_WARNING_HIDDEN)
                self._warning_style_state = "hidden"
            self.warning_icon_label.setText("")
            self.warning_title_label.setText("")
            self.warning_details_label.setText("")
            self.countdown_label.setText("")
            self.continue_button.setStyleSheet(
                "background: transparent; border: none; color: transparent;"
            )
            while self.suggestions_button_layout.count():
                item = self.suggestions_button_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

    def _on_abort_requested(self):
        self.hide_warning()
        self.abort_requested.emit()

    def _on_suggestion_clicked(self, params: Dict):
        self.hide_warning()
        self.retry_requested.emit(params)
