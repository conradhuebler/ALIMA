"""Providers as a plain list, and model limits as one deviation list.

An accordion was tried first and was wrong: five entries that one wants to see at
once do not benefit from folding, and the fold hid mostly static text behind an
interaction. A list shows every provider at a glance; double-click edits, the
same as everywhere else.

The model limits moved out of the per-provider card for the same reason. They are
rare — typically none at all — so one section listing every deviation beats five
sections that are empty. The threshold belongs to the *model* (it follows the
context window), not to the task that triggers chunking. - Claude Generated
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .collapsible import CollapsibleSection


def _endpoint(provider: Any) -> str:
    url = getattr(provider, "base_url", "") or ""
    if url:
        return url
    host = getattr(provider, "host", "") or ""
    port = getattr(provider, "port", None)
    return f"{host}:{port}" if host else "(Vorgabe)"


def _needs_key(provider: Any) -> bool:
    return (
        getattr(provider, "provider_type", "") in ("gemini", "anthropic")
        and not getattr(provider, "api_key", "")
    )


class ProviderList(QWidget):
    """The providers, all visible. Double-click edits."""

    edit_requested = pyqtSignal(str)
    remove_requested = pyqtSignal(str)
    models_requested = pyqtSignal(str)
    add_requested = pyqtSignal()
    chunking_changed = pyqtSignal(str, str, int)
    chunking_removed = pyqtSignal(str, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)

        self.list = QListWidget()
        self.list.setAlternatingRowColors(True)
        self.list.itemDoubleClicked.connect(self._on_double_click)
        self.list.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        box.addWidget(self.list)

        hint = QLabel("Doppelklick auf einen Eintrag bearbeitet ihn.")
        hint.setStyleSheet("color: gray; font-size: 10px;")
        box.addWidget(hint)

        row = QHBoxLayout()
        self.add_button = QPushButton("➕ Hinzufügen…")
        self.add_button.clicked.connect(self.add_requested)
        self.edit_button = QPushButton("Bearbeiten…")
        self.edit_button.clicked.connect(lambda: self._emit_for_selection(self.edit_requested))
        self.remove_button = QPushButton("Entfernen")
        self.remove_button.clicked.connect(lambda: self._emit_for_selection(self.remove_requested))
        self.models_button = QPushButton("Modelle abfragen")
        self.models_button.setToolTip(
            "Fragt die Modellliste dieses einen Anbieters neu ab."
        )
        self.models_button.clicked.connect(lambda: self._emit_for_selection(self.models_requested))
        for button in (self.add_button, self.edit_button,
                       self.remove_button, self.models_button):
            row.addWidget(button)
        row.addStretch()
        box.addLayout(row)

        self.limits = CollapsibleSection("Modellgrenzen", subtitle="keine Abweichung")
        self._limits_box = QVBoxLayout(self.limits.body())
        self._limits_box.setContentsMargins(12, 2, 0, 4)
        self._limits_box.setSpacing(3)
        box.addWidget(self.limits)

        self.list.currentRowChanged.connect(self._update_buttons)
        self._update_buttons(-1)

    # -- content ---------------------------------------------------------

    def refresh(self, providers: List[Any], *, models: Dict[str, List[str]],
                thresholds: Dict[str, Dict[str, int]]) -> None:
        selected = self.selected_name()
        self.list.clear()
        self._models = dict(models or {})
        for provider in providers:
            name = getattr(provider, "name", "?")
            known = list(self._models.get(name) or getattr(provider, "available_models", []) or [])
            parts = [
                name,
                getattr(provider, "provider_type", ""),
                _endpoint(provider),
                f"{len(known)} Modelle",
            ]
            if not getattr(provider, "enabled", True):
                parts.append("aus")
            if _needs_key(provider):
                parts.append("kein Schlüssel")
            item = QListWidgetItem("   ·   ".join(parts))
            item.setData(Qt.ItemDataRole.UserRole, name)
            if not getattr(provider, "enabled", True) or _needs_key(provider):
                item.setForeground(self.palette().placeholderText())
            self.list.addItem(item)

        if not providers:
            empty = QListWidgetItem(
                "Noch kein Anbieter — ohne einen kann ALIMA kein Sprachmodell aufrufen."
            )
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self.list.addItem(empty)

        self._fit_height()
        self._rebuild_limits(providers, thresholds or {})
        if selected:
            self.select(selected)

    def _fit_height(self) -> None:
        """Show every provider without scrolling; they are few and all relevant."""
        rows = max(self.list.count(), 1)
        height = self.list.sizeHintForRow(0) if self.list.count() else 20
        total = rows * max(height, 18) + 2 * self.list.frameWidth() + 4
        self.list.setMinimumHeight(total)
        self.list.setMaximumHeight(total)

    # -- selection -------------------------------------------------------

    def selected_name(self) -> str:
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) or "" if item else ""

    def select(self, name: str) -> None:
        for row in range(self.list.count()):
            if self.list.item(row).data(Qt.ItemDataRole.UserRole) == name:
                self.list.setCurrentRow(row)
                return

    def names(self) -> List[str]:
        return [
            self.list.item(r).data(Qt.ItemDataRole.UserRole)
            for r in range(self.list.count())
            if self.list.item(r).data(Qt.ItemDataRole.UserRole)
        ]

    def row_texts(self) -> List[str]:
        return [self.list.item(r).text() for r in range(self.list.count())]

    def _emit_for_selection(self, signal) -> None:
        name = self.selected_name()
        if name:
            signal.emit(name)

    def _on_double_click(self, item: QListWidgetItem) -> None:
        name = item.data(Qt.ItemDataRole.UserRole)
        if name:
            self.edit_requested.emit(name)

    def _update_buttons(self, _row: int) -> None:
        has = bool(self.selected_name())
        for button in (self.edit_button, self.remove_button, self.models_button):
            button.setEnabled(has)

    # -- model limits ----------------------------------------------------

    def _rebuild_limits(self, providers: List[Any],
                        thresholds: Dict[str, Dict[str, int]]) -> None:
        while self._limits_box.count():
            item = self._limits_box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()

        note = QLabel(
            "Ab wie vielen Zeichen ein Text für ein bestimmtes Modell gestückelt "
            "wird. Ohne Eintrag entscheidet die automatische Erkennung. Hängt am "
            "Kontextfenster des Modells, nicht an der Aufgabe."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 10px;")
        self._limits_box.addWidget(note)

        flat = [
            (prov, model, value)
            for prov, per_model in sorted(thresholds.items())
            for model, value in sorted((per_model or {}).items())
        ]
        for prov, model, value in flat:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{prov} · {model}"), 1)
            spin = QSpinBox()
            spin.setRange(0, 1_000_000)
            spin.setSingleStep(1000)
            spin.setValue(int(value))
            spin.setSuffix(" Zeichen")
            spin.valueChanged.connect(
                lambda v, p=prov, m=model: self.chunking_changed.emit(p, m, v)
            )
            row.addWidget(spin)
            drop = QPushButton("✕")
            drop.setMaximumWidth(30)
            drop.setToolTip("Abweichung entfernen; es gilt wieder die Automatik")
            drop.clicked.connect(
                lambda _c, p=prov, m=model: self.chunking_removed.emit(p, m)
            )
            row.addWidget(drop)
            self._limits_box.addLayout(row)

        add_row = QHBoxLayout()
        self.limit_provider = QComboBox()
        self.limit_provider.addItems([getattr(p, "name", "?") for p in providers])
        self.limit_model = QComboBox()
        self.limit_provider.currentTextChanged.connect(self._refill_limit_models)
        add_row.addWidget(self.limit_provider)
        add_row.addWidget(self.limit_model, 1)
        add = QPushButton("Grenze setzen")
        add.clicked.connect(self._add_limit)
        add_row.addWidget(add)
        self._limits_box.addLayout(add_row)
        self._refill_limit_models(self.limit_provider.currentText())

        self.limits.set_subtitle(
            "keine Abweichung" if not flat
            else ("1 Abweichung" if len(flat) == 1 else f"{len(flat)} Abweichungen")
        )

    def _refill_limit_models(self, provider_name: str) -> None:
        self.limit_model.clear()
        self.limit_model.addItems(list(self._models.get(provider_name) or []))

    def _add_limit(self) -> None:
        provider = self.limit_provider.currentText()
        model = self.limit_model.currentText()
        if provider and model:
            self.chunking_changed.emit(provider, model, 24000)
