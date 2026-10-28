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

    @staticmethod
    def _style_for(state: str) -> str:
        """Bar stylesheet for ``hidden`` / ``orange`` / ``green``, coloured
        from the current theme. ``hidden`` paints everything transparent (the
        buttons are hidden by ``_set_buttons_visible``).
        Children carry no colours of their own; the object names below address
        them. - Claude Generated"""
        from .styles import get_colors

        c = get_colors()
        if state == "hidden":
            return (
                "QFrame { background: transparent; border: none; padding: 0; }"
                "QLabel { color: transparent; }"
                "QPushButton { background: transparent; border: none; color: transparent; }"
            )
        tone = "warning" if state == "orange" else "success"
        fg, bg = c[tone], c[tone + "_bg"]
        on = c["on_" + tone]
        hover = c[tone + "_hover"]
        continue_btn = (
            f"QPushButton#repContinue {{ background-color: {c['surface_raised']}; "
            f"color: {c['text']}; border: 1px solid {c['border']}; padding: 0px 4px; font-weight: normal; }}"
            if state == "orange"
            else "QPushButton#repContinue { background: transparent; border: none; color: transparent; }"
        )
        return (
            f"QFrame {{ background-color: {bg}; border: 1px solid {fg}; "
            f"border-radius: 3px; padding: 1px; }}"
            f"QLabel {{ color: {c['text']}; border: none; background: transparent; padding: 0; }}"
            f"QLabel#repTitle {{ color: {fg}; font-weight: bold; }}"
            f"QLabel#repCountdown {{ font-weight: bold; }}"
            f"QPushButton {{ background-color: {fg}; color: {on}; border: none; "
            f"border-radius: 3px; padding: 1px 5px; font-weight: bold; }}"
            f"QPushButton:hover {{ background-color: {hover}; }}"
            f"QPushButton#repAbort {{ background-color: {c['error']}; color: {c['on_error']}; }}"
            f"QPushButton#repDismiss {{ background-color: transparent; color: {fg}; padding: 0; }}"
            + continue_btn
        )

    def _set_buttons_visible(self, visible: bool) -> None:
        """Knöpfe im Ruhezustand wirklich verstecken: nur transparent gemalt
        waren sie weiter klickbar und ihre Emoji sichtbar. - Claude Generated"""
        for btn in (self.abort_now_button, self.continue_button,
                    self.dismiss_warning_button):
            btn.setVisible(visible)

    def refresh_styles(self):
        """Re-colour the current state after a theme switch. - Claude Generated"""
        self.setStyleSheet(self._style_for(self._warning_style_state))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self._warning_style_state = "hidden"
        self._last_shown_detection_type = ""
        self.current_suggestions: List[Dict] = []
        self.setObjectName("repetitionBar")
        self.setStyleSheet(self._style_for("hidden"))

        bar = QHBoxLayout(self)
        bar.setContentsMargins(6, 1, 4, 1)
        bar.setSpacing(6)

        # Leer im Ruhezustand: ein Emoji ignoriert "color: transparent" und
        # bliebe sichtbar. - Claude Generated
        self.warning_icon_label = QLabel("")
        bar.addWidget(self.warning_icon_label)

        self.warning_title_label = QLabel("Wiederholung erkannt")
        self.warning_title_label.setObjectName("repTitle")
        bar.addWidget(self.warning_title_label)

        self.warning_details_label = QLabel("")
        self.warning_details_label.setWordWrap(False)
        bar.addWidget(self.warning_details_label, 1)

        self.countdown_label = QLabel("")
        self.countdown_label.setObjectName("repCountdown")
        self.countdown_label.setVisible(False)
        bar.addWidget(self.countdown_label)

        self.suggestions_button_layout = QHBoxLayout()
        self.suggestions_button_layout.setSpacing(3)
        bar.addLayout(self.suggestions_button_layout)

        self.abort_now_button = QPushButton("🛑 Abbrechen")
        self.abort_now_button.setObjectName("repAbort")
        self.abort_now_button.clicked.connect(self._on_abort_requested)
        bar.addWidget(self.abort_now_button)

        self.continue_button = QPushButton("Fortfahren")
        self.continue_button.setObjectName("repContinue")
        self.continue_button.clicked.connect(self.hide_warning)
        bar.addWidget(self.continue_button)

        self.dismiss_warning_button = QPushButton("✕")
        self.dismiss_warning_button.setFixedSize(18, 18)
        self.dismiss_warning_button.setObjectName("repDismiss")
        self.dismiss_warning_button.clicked.connect(self.hide_warning)
        bar.addWidget(self.dismiss_warning_button)

        # Die Knöpfe behalten ihren Platz, auch versteckt (die Breite der
        # übrigen Elemente springt beim Erscheinen nicht). Die Höhe der Leiste
        # wechselt zwischen Ruhe und Warnung um wenige Pixel, wie zuvor.
        for btn in (self.abort_now_button, self.continue_button,
                    self.dismiss_warning_button):
            policy = btn.sizePolicy()
            policy.setRetainSizeWhenHidden(True)
            btn.setSizePolicy(policy)
        self._set_buttons_visible(False)

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
        self._set_buttons_visible(True)
        if self._warning_style_state != "orange":
            self.setStyleSheet(self._style_for("orange"))
            self._warning_style_state = "orange"
        self._last_shown_detection_type = detection_type

        if not already_showing:
            self.warning_icon_label.setText("⚠️")

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
                self.setStyleSheet(self._style_for("green"))
                self._warning_style_state = "green"
            self._set_buttons_visible(True)
            self.warning_icon_label.setText("✅")
            self.warning_title_label.setText(
                "Wiederholung behoben – Generation läuft weiter"
            )
            self.warning_details_label.setText("")
            while self.suggestions_button_layout.count():
                item = self.suggestions_button_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        else:
            if self._warning_style_state != "hidden":
                self.setStyleSheet(self._style_for("hidden"))
                self._warning_style_state = "hidden"
            self._set_buttons_visible(False)
            self.warning_icon_label.setText("")
            self.warning_title_label.setText("")
            self.warning_details_label.setText("")
            self.countdown_label.setText("")
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
