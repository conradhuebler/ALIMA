"""Providers as an accordion, with their model limits underneath.

Three to six providers with six fields each is a list, not a table: a table has
to squeeze the base URL and the key state into columns and then needs a dialog
for everything it could not fit. An accordion shows the identifying line always
and the rest on demand.

The chunking threshold lives here rather than with the task that triggers it: it
depends on a model's context window, not on what the model is asked to do. Per
task it would have to be maintained once per task for the same model, and the
copies would drift. Only deviations from the automatic value are listed, so the
usual case costs one line. - Claude Generated
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .collapsible import CollapsibleSection


def _key_state(provider: Any) -> str:
    if getattr(provider, "api_key", ""):
        return "Schlüssel hinterlegt"
    if getattr(provider, "provider_type", "") in ("gemini", "anthropic"):
        return "kein Schlüssel — Anbieter nicht nutzbar"
    return "kein Schlüssel nötig"


def _endpoint(provider: Any) -> str:
    url = getattr(provider, "base_url", "") or ""
    if url:
        return url
    host = getattr(provider, "host", "") or ""
    port = getattr(provider, "port", None)
    return f"{host}:{port}" if host else "(Vorgabe)"


class ProviderCard(CollapsibleSection):
    """One provider: identity in the header, the rest on demand."""

    edit_requested = pyqtSignal(str)
    remove_requested = pyqtSignal(str)
    models_requested = pyqtSignal(str)
    chunking_changed = pyqtSignal(str, str, int)
    chunking_removed = pyqtSignal(str, str)

    def __init__(self, provider: Any, *, models: List[str],
                 thresholds: Dict[str, int], parent: Optional[QWidget] = None) -> None:
        self._provider = provider
        name = getattr(provider, "name", "?")
        super().__init__(self._header(provider, models), parent=parent)

        box = QVBoxLayout(self.body())
        box.setContentsMargins(18, 0, 0, 8)

        box.addWidget(self._line("Typ", getattr(provider, "provider_type", "")))
        box.addWidget(self._line("Endpunkt", _endpoint(provider)))
        box.addWidget(self._line("API-Schlüssel", _key_state(provider)))

        preferred = getattr(provider, "preferred_model", "") or "(keine)"
        box.addWidget(self._line("Vorauswahl", preferred))
        hint = QLabel(
            "Bestimmt nur, was in den Auswahllisten vorausgewählt ist, nicht "
            "welches Modell läuft."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 10px;")
        box.addWidget(hint)

        fetch_row = QHBoxLayout()
        fetch = QPushButton("Modelle jetzt abfragen")
        fetch.clicked.connect(lambda: self.models_requested.emit(name))
        fetch_row.addWidget(fetch)
        fetch_row.addWidget(QLabel(f"{len(models)} bekannt"))
        fetch_row.addStretch()
        box.addLayout(fetch_row)

        self.limits = CollapsibleSection(
            "Modellgrenzen",
            subtitle=self._limits_subtitle(thresholds),
        )
        self._build_limits(self.limits.body(), name, models, thresholds)
        box.addWidget(self.limits)

        actions = QHBoxLayout()
        actions.addStretch()
        edit = QPushButton("Bearbeiten…")
        edit.clicked.connect(lambda: self.edit_requested.emit(name))
        remove = QPushButton("Entfernen")
        remove.clicked.connect(lambda: self.remove_requested.emit(name))
        actions.addWidget(edit)
        actions.addWidget(remove)
        box.addLayout(actions)

    # -- header ----------------------------------------------------------

    @staticmethod
    def _header(provider: Any, models: List[str]) -> str:
        mark = "" if getattr(provider, "enabled", True) else "  (aus)"
        return (
            f"{getattr(provider, 'name', '?')}  ·  "
            f"{getattr(provider, 'provider_type', '')}  ·  "
            f"{len(models)} Modelle{mark}"
        )

    @staticmethod
    def _limits_subtitle(thresholds: Dict[str, int]) -> str:
        n = len(thresholds or {})
        if not n:
            return "keine Abweichung"
        return "1 Abweichung" if n == 1 else f"{n} Abweichungen"

    # -- rows ------------------------------------------------------------

    @staticmethod
    def _line(label: str, value: str) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        caption = QLabel(f"{label}:")
        caption.setMinimumWidth(110)
        caption.setStyleSheet("color: gray;")
        layout.addWidget(caption)
        value_label = QLabel(str(value))
        value_label.setWordWrap(True)
        layout.addWidget(value_label, 1)
        return row

    def _build_limits(self, host: QWidget, provider_name: str,
                      models: List[str], thresholds: Dict[str, int]) -> None:
        box = QVBoxLayout(host)
        box.setContentsMargins(0, 0, 0, 0)

        note = QLabel(
            "Ab wie vielen Zeichen ein Text für dieses Modell gestückelt wird. "
            "Ohne Eintrag entscheidet die automatische Erkennung. Hängt am "
            "Kontextfenster des Modells, nicht an der Aufgabe."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 10px;")
        box.addWidget(note)

        for model, value in sorted((thresholds or {}).items()):
            row = QHBoxLayout()
            row.addWidget(QLabel(model), 1)
            spin = QSpinBox()
            spin.setRange(0, 1_000_000)
            spin.setSingleStep(1000)
            spin.setValue(int(value))
            spin.setSuffix(" Zeichen")
            spin.valueChanged.connect(
                lambda v, m=model: self.chunking_changed.emit(provider_name, m, v)
            )
            row.addWidget(spin)
            drop = QPushButton("✕")
            drop.setMaximumWidth(30)
            drop.setToolTip("Abweichung entfernen; es gilt wieder die Automatik")
            drop.clicked.connect(
                lambda _c, m=model: self.chunking_removed.emit(provider_name, m)
            )
            row.addWidget(drop)
            box.addLayout(row)

        add_row = QHBoxLayout()
        self.add_model = QComboBox()
        self.add_model.addItems([m for m in models if m not in (thresholds or {})])
        add_row.addWidget(self.add_model, 1)
        add = QPushButton("Grenze setzen")
        add.setEnabled(self.add_model.count() > 0)
        add.clicked.connect(
            lambda: self.chunking_changed.emit(
                provider_name, self.add_model.currentText(), 24000
            )
        )
        add_row.addWidget(add)
        box.addLayout(add_row)


class ProviderAccordion(QWidget):
    """The provider list. One card each, no table."""

    edit_requested = pyqtSignal(str)
    remove_requested = pyqtSignal(str)
    models_requested = pyqtSignal(str)
    chunking_changed = pyqtSignal(str, str, int)
    chunking_removed = pyqtSignal(str, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._box = QVBoxLayout(self)
        self._box.setContentsMargins(0, 0, 0, 0)
        self.cards: Dict[str, ProviderCard] = {}
        self._empty = QLabel(
            "Noch kein Anbieter eingerichtet. Ohne einen Anbieter kann ALIMA "
            "kein Sprachmodell aufrufen."
        )
        self._empty.setWordWrap(True)
        self._empty.setStyleSheet("color: gray;")
        self._box.addWidget(self._empty)

    def refresh(self, providers: List[Any], *, models: Dict[str, List[str]],
                thresholds: Dict[str, Dict[str, int]]) -> None:
        for card in self.cards.values():
            self._box.removeWidget(card)
            card.deleteLater()
        self.cards = {}

        self._empty.setVisible(not providers)
        for provider in providers:
            name = getattr(provider, "name", "?")
            card = ProviderCard(
                provider,
                models=list(models.get(name) or getattr(provider, "available_models", []) or []),
                thresholds=dict((thresholds or {}).get(name) or {}),
            )
            for signal in ("edit_requested", "remove_requested", "models_requested",
                           "chunking_changed", "chunking_removed"):
                getattr(card, signal).connect(getattr(self, signal))
            self._box.insertWidget(self._box.count() - 1, card)
            self.cards[name] = card

    def headers(self) -> List[str]:
        """Rendered header lines, for tests. - Claude Generated"""
        return [c._button.text() for c in self.cards.values()]
